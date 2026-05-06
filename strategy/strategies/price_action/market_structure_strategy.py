from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import MarketBias, StructureEventType, StructureLayer
from core.event_bus import EventBus
from core.logger import TradingLoggerAdapter
from strategy.config import StrategyConfig, StrategyDefinitionConfig
from strategy.enums import (
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError
from strategy.models import (
    FilterResult,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
)
from strategy.strategies.price_action.base import (
    PriceActionStrategyBase,
    apply_definition_metadata,
    clamp,
    enum_value,
    first_non_empty,
    parse_datetime,
    safe_bool,
    safe_float,
)


@dataclass(slots=True)
class MarketStructureStrategyParams:
    """
    Локальні параметри саме для market_structure_strategy.

    Значення можуть приходити з StrategyDefinitionConfig.metadata.
    Runtime-gating, enabled/symbol/timeframe/min_score/min_confidence
    залишаються в StrategyConfig / StrategyDefinitionConfig.runtime.
    """

    strategy_name: str = "market_structure_strategy"

    prefer_external_layer: bool = True
    require_alignment: bool = False
    require_break_event: bool = True
    allow_bos_entries: bool = True
    allow_choch_reversals: bool = True
    allow_mss_reversals: bool = True

    min_layer_confidence: float = 0.45
    min_alignment_score: float = 0.30
    min_break_confidence: float = 0.45
    min_trend_strength: float = 0.20

    external_bias_weight: float = 0.35
    internal_bias_weight: float = 0.20
    break_event_weight: float = 0.25
    alignment_weight: float = 0.10
    regime_alignment_weight: float = 0.10

    reverse_on_choch: bool = True
    reverse_on_mss: bool = True

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.market_structure.signal"

    freshness_feature_names: tuple[str, ...] = (
        "price_action.market_structure",
        "market_structure",
        "analytics.price_action.market_structure",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_layer_confidence",
                "min_alignment_score",
                "min_break_confidence",
                "min_trend_strength",
                "external_bias_weight",
                "internal_bias_weight",
                "break_event_weight",
                "alignment_weight",
                "regime_alignment_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "MarketStructureStrategyParams":
        return apply_definition_metadata(
            params=cls(),
            definition=definition,
        )


@dataclass(slots=True)
class BreakContext:
    event_type: StructureEventType | None = None
    direction: MarketBias = MarketBias.UNKNOWN
    confidence: float = 0.0
    timestamp: datetime | None = None
    layer: StructureLayer | None = None
    reference_price: float | None = None
    trigger_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_break(self) -> bool:
        return self.event_type in {
            StructureEventType.BOS,
            StructureEventType.CHOCH,
            StructureEventType.MSS,
        }


class MarketStructureStrategy(PriceActionStrategyBase):
    """
    Strategy layer wrapper around analytics.price_action.market_structure.

    Основні ідеї:
    - tolerant extraction із SignalContext.price_action та feature_map
    - підтримка internal/external layer
    - BOS = continuation / breakout
    - CHOCH / MSS = reversal
    - confidence/score gating
    - optional EventBus signal emission через PriceActionStrategyBase

    Інфраструктура:
    - logger, EventBus, runtime gating, freshness/regime filters, rejected/final evaluation
      беруться зі спільного PriceActionStrategyBase.
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
        strategy_name: str = "market_structure_strategy",
    ) -> None:
        super().__init__(
            config=config,
            strategy_name=strategy_name,
            params_cls=MarketStructureStrategyParams,
            event_bus=event_bus,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Typed params accessor — fixes "Cannot find reference in ParamsT"
    # ------------------------------------------------------------------

    @property
    def _p(self) -> MarketStructureStrategyParams:
        """Typed shortcut so IDE resolves all MarketStructureStrategyParams fields."""
        return cast(MarketStructureStrategyParams, self.params)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked

            structure = self._extract_market_structure_snapshot(context)
            if not structure:
                return self._rejected_evaluation(
                    context=context,
                    reason="market_structure_snapshot_missing",
                )

            freshness_filter = self._build_freshness_filter(
                context=context,
                filter_name="market_structure_freshness",
            )
            if freshness_filter is not None and freshness_filter.blocked:
                return self._rejected_evaluation(
                    context=context,
                    reason="stale_market_structure_feature",
                )

            side = self._resolve_side(context=context, structure=structure)
            if side == SignalSide.UNKNOWN:
                return self._rejected_evaluation(
                    context=context,
                    reason="no_directional_market_structure_signal",
                )

            score = self._compute_score(
                context=context,
                structure=structure,
                side=side,
            )
            confidence = self._compute_confidence(
                context=context,
                structure=structure,
                side=side,
            )
            reasons = self._build_reasons(
                structure=structure,
                side=side,
            )

            signal = self._build_signal(
                context=context,
                structure=structure,
                side=side,
                score=score,
                confidence=confidence,
                reasons=reasons,
                freshness_filter=freshness_filter,
            )

            return self._finalize_signal_evaluation(
                context=context,
                signal=signal,
                confidence=confidence,
                score=score,
                reasons=reasons,
            )

        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self._logger.exception(
                "Failed to evaluate market structure strategy | strategy=%s symbol=%s",
                self.name,
                getattr(context, "symbol", None),
            )
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate market structure context for {context.symbol}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_market_structure_snapshot(self, context: SignalContext) -> dict[str, Any]:
        candidates: list[Any] = [
            context.price_action.get("market_structure"),
            context.price_action.get("structure"),
            context.get_feature("price_action.market_structure"),
            context.get_feature("market_structure"),
            context.get_feature("analytics.price_action.market_structure"),
        ]

        for candidate in candidates:
            normalized = self._normalize_structure_snapshot(candidate)
            if normalized:
                return normalized

        return {}

    def _normalize_structure_snapshot(self, payload: Any) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        state = self._state_mapping_or_empty(payload_mapping)
        if not state:
            return {}

        internal = self._normalize_layer(state.get("internal"), StructureLayer.INTERNAL)
        external = self._normalize_layer(state.get("external"), StructureLayer.EXTERNAL)
        mtf_alignment = self._normalize_alignment(state.get("mtf_alignment"))

        last_event = self._extract_last_break_event(
            internal=internal,
            external=external,
        )
        symbol = first_non_empty(state.get("symbol"), payload_mapping.get("symbol"))
        timeframe = first_non_empty(state.get("timeframe"), payload_mapping.get("timeframe"))
        last_price = safe_float(
            first_non_empty(state.get("last_price"), payload_mapping.get("last_price")),
            0.0,
        )
        last_update = parse_datetime(
            first_non_empty(state.get("last_update"), payload_mapping.get("last_update"))
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_price": last_price if last_price > 0 else None,
            "last_update": last_update,
            "internal": internal,
            "external": external,
            "mtf_alignment": mtf_alignment,
            "last_break_event": last_event,
            "raw": dict(payload_mapping),
        }

    def _normalize_layer(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        return {
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
            "bias": self._parse_market_bias(payload_mapping.get("bias")),
            "confidence": clamp(safe_float(payload_mapping.get("confidence"), 0.0), 0.0, 1.0),
            "trend_strength": clamp(safe_float(payload_mapping.get("trend_strength"), 0.0), 0.0, 1.0),
            "in_breakout": safe_bool(payload_mapping.get("in_breakout"), False),
            "swing_count": int(safe_float(payload_mapping.get("swing_count"), 0.0)),
            "event_count": int(safe_float(payload_mapping.get("event_count"), 0.0)),
            "sequence": list(payload_mapping.get("sequence", []) or []),
            "last_bos": self._normalize_structure_event(
                payload_mapping.get("last_bos"),
                default_layer,
            ),
            "last_choch": self._normalize_structure_event(
                payload_mapping.get("last_choch"),
                default_layer,
            ),
            "last_mss": self._normalize_structure_event(
                payload_mapping.get("last_mss"),
                default_layer,
            ),
            "last_hh": self._normalize_structure_event(
                payload_mapping.get("last_hh"),
                default_layer,
            ),
            "last_hl": self._normalize_structure_event(
                payload_mapping.get("last_hl"),
                default_layer,
            ),
            "last_lh": self._normalize_structure_event(
                payload_mapping.get("last_lh"),
                default_layer,
            ),
            "last_ll": self._normalize_structure_event(
                payload_mapping.get("last_ll"),
                default_layer,
            ),
        }

    def _normalize_alignment(self, payload: Any) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        return {
            "higher_timeframe": payload_mapping.get("higher_timeframe"),
            "higher_timeframe_bias": self._parse_market_bias(payload_mapping.get("higher_timeframe_bias")),
            "higher_timeframe_confidence": clamp(
                safe_float(payload_mapping.get("higher_timeframe_confidence"), 0.0),
                0.0,
                1.0,
            ),
            "internal_bias_aligned": safe_bool(payload_mapping.get("internal_bias_aligned"), False),
            "external_bias_aligned": safe_bool(payload_mapping.get("external_bias_aligned"), False),
            "internal_with_external_aligned": safe_bool(
                payload_mapping.get("internal_with_external_aligned"),
                False,
            ),
            "alignment_score": clamp(
                safe_float(payload_mapping.get("alignment_score"), 0.0),
                0.0,
                1.0,
            ),
            "last_updated": parse_datetime(payload_mapping.get("last_updated")),
        }

    def _normalize_structure_event(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        event_type = self._parse_structure_event_type(payload_mapping.get("event_type"))
        layer = self._parse_structure_layer(payload_mapping.get("layer")) or default_layer
        direction = self._parse_market_bias(payload_mapping.get("direction"))
        confidence = clamp(safe_float(payload_mapping.get("confidence"), 0.0), 0.0, 1.0)

        return {
            "event_id": payload_mapping.get("event_id"),
            "event_type": event_type,
            "timestamp": parse_datetime(payload_mapping.get("timestamp")),
            "price": safe_float(payload_mapping.get("price"), 0.0),
            "layer": layer,
            "direction": direction,
            "reference_price": (
                safe_float(payload_mapping.get("reference_price"), 0.0)
                if payload_mapping.get("reference_price") is not None
                else None
            ),
            "reference_swing_id": payload_mapping.get("reference_swing_id"),
            "confidence": confidence,
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _extract_last_break_event(
        self,
        *,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> BreakContext:
        # FIX: timezone-aware _epoch — уникає TypeError при порівнянні
        # naive і aware datetime під час сортування кандидатів
        _epoch = datetime.min.replace(tzinfo=timezone.utc)

        candidates: list[dict[str, Any]] = []

        for layer_name, layer in (
            (StructureLayer.INTERNAL, internal),
            (StructureLayer.EXTERNAL, external),
        ):
            for key in ("last_bos", "last_choch", "last_mss"):
                event = layer.get(key)
                if event:
                    copied = dict(event)
                    copied["layer"] = layer_name
                    candidates.append(copied)

        if not candidates:
            return BreakContext()

        def _sort_key(item: dict[str, Any]) -> datetime:
            ts = item.get("timestamp")
            if ts is None:
                return _epoch
            # FIX: normalize naive datetime → aware щоб уникнути TypeError
            if isinstance(ts, datetime) and ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        candidates.sort(key=_sort_key, reverse=True)
        latest = candidates[0]

        return BreakContext(
            event_type=latest.get("event_type"),
            direction=latest.get("direction", MarketBias.UNKNOWN),
            confidence=clamp(safe_float(latest.get("confidence"), 0.0), 0.0, 1.0),
            timestamp=latest.get("timestamp"),
            layer=latest.get("layer"),
            reference_price=latest.get("reference_price"),
            trigger_price=latest.get("price"),
            metadata=dict(latest.get("metadata", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Direction / scoring
    # ------------------------------------------------------------------

    def _resolve_side(
        self,
        context: SignalContext,
        structure: Mapping[str, Any],
    ) -> SignalSide:
        internal = structure.get("internal", {})
        external = structure.get("external", {})
        mtf = structure.get("mtf_alignment", {})
        last_break = self._break_context_or_empty(structure.get("last_break_event"))

        # FIX: доступ через _p замість self.params
        primary_layer = external if self._p.prefer_external_layer else internal
        secondary_layer = internal if self._p.prefer_external_layer else external

        primary_bias = primary_layer.get("bias", MarketBias.UNKNOWN)

        if self._p.require_alignment and not mtf.get("internal_with_external_aligned", False):
            return SignalSide.UNKNOWN

        if self._p.require_break_event and not last_break.is_break:
            return SignalSide.UNKNOWN

        if last_break.is_break:
            if last_break.confidence < self._p.min_break_confidence:
                return SignalSide.UNKNOWN

            if last_break.event_type == StructureEventType.BOS and self._p.allow_bos_entries:
                return self._bias_to_side(last_break.direction)

            if last_break.event_type == StructureEventType.CHOCH and self._p.allow_choch_reversals:
                if self._p.reverse_on_choch:
                    return self._bias_to_side(last_break.direction)

            if last_break.event_type == StructureEventType.MSS and self._p.allow_mss_reversals:
                if self._p.reverse_on_mss:
                    return self._bias_to_side(last_break.direction)

        if not self._layer_eligible(primary_layer):
            return SignalSide.UNKNOWN

        secondary_bias = secondary_layer.get("bias", MarketBias.UNKNOWN)

        if primary_bias == secondary_bias and primary_bias in {MarketBias.BULLISH, MarketBias.BEARISH}:
            return self._bias_to_side(primary_bias)

        if primary_bias in {MarketBias.BULLISH, MarketBias.BEARISH}:
            return self._bias_to_side(primary_bias)

        return SignalSide.UNKNOWN

    def _layer_eligible(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False

        confidence = clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0)
        trend_strength = clamp(safe_float(layer.get("trend_strength"), 0.0), 0.0, 1.0)

        # FIX: доступ через _p
        if confidence < self._p.min_layer_confidence:
            return False

        if trend_strength < self._p.min_trend_strength:
            return False

        return True

    def _compute_score(
        self,
        context: SignalContext,
        structure: Mapping[str, Any],
        side: SignalSide,
    ) -> float:
        internal = structure.get("internal", {})
        external = structure.get("external", {})
        mtf = structure.get("mtf_alignment", {})
        last_break = self._break_context_or_empty(structure.get("last_break_event"))

        score = 0.0

        external_bias = self._bias_to_side(external.get("bias", MarketBias.UNKNOWN))
        internal_bias = self._bias_to_side(internal.get("bias", MarketBias.UNKNOWN))

        # FIX: доступ через _p
        if external_bias == side:
            score += self._p.external_bias_weight * (
                0.5 + 0.5 * clamp(safe_float(external.get("confidence"), 0.0), 0.0, 1.0)
            )

        if internal_bias == side:
            score += self._p.internal_bias_weight * (
                0.5 + 0.5 * clamp(safe_float(internal.get("confidence"), 0.0), 0.0, 1.0)
            )

        if last_break.is_break and self._bias_to_side(last_break.direction) == side:
            score += self._p.break_event_weight * (
                0.5 + 0.5 * clamp(last_break.confidence, 0.0, 1.0)
            )

        alignment_score = clamp(safe_float(mtf.get("alignment_score"), 0.0), 0.0, 1.0)
        score += self._p.alignment_weight * alignment_score

        score += self._p.regime_alignment_weight * self._regime_alignment_score(
            context=context,
            side=side,
        )

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        context: SignalContext,
        structure: Mapping[str, Any],
        side: SignalSide,
    ) -> float:
        internal = structure.get("internal", {})
        external = structure.get("external", {})
        mtf = structure.get("mtf_alignment", {})
        last_break = self._break_context_or_empty(structure.get("last_break_event"))

        components: list[float] = []

        external_bias = self._bias_to_side(external.get("bias", MarketBias.UNKNOWN))
        internal_bias = self._bias_to_side(internal.get("bias", MarketBias.UNKNOWN))

        if external_bias == side:
            components.append(clamp(safe_float(external.get("confidence"), 0.0), 0.0, 1.0))
            components.append(clamp(safe_float(external.get("trend_strength"), 0.0), 0.0, 1.0))

        if internal_bias == side:
            components.append(clamp(safe_float(internal.get("confidence"), 0.0), 0.0, 1.0))
            components.append(clamp(safe_float(internal.get("trend_strength"), 0.0), 0.0, 1.0))

        if last_break.is_break and self._bias_to_side(last_break.direction) == side:
            components.append(clamp(last_break.confidence, 0.0, 1.0))

        components.append(clamp(safe_float(mtf.get("alignment_score"), 0.0), 0.0, 1.0))
        components.append(self._regime_alignment_score(context=context, side=side))

        if not components:
            return 0.0

        return clamp(sum(components) / len(components), 0.0, 1.0)

    def _build_reasons(
        self,
        structure: Mapping[str, Any],
        side: SignalSide,
    ) -> list[str]:
        reasons: list[str] = []

        internal = structure.get("internal", {})
        external = structure.get("external", {})
        mtf = structure.get("mtf_alignment", {})
        last_break = self._break_context_or_empty(structure.get("last_break_event"))

        if side == SignalSide.LONG:
            reasons.append("market_structure_bullish")
        elif side == SignalSide.SHORT:
            reasons.append("market_structure_bearish")

        if self._bias_to_side(external.get("bias", MarketBias.UNKNOWN)) == side:
            reasons.append("external_bias_aligned")

        if self._bias_to_side(internal.get("bias", MarketBias.UNKNOWN)) == side:
            reasons.append("internal_bias_aligned")

        if last_break.is_break and self._bias_to_side(last_break.direction) == side:
            # FIX: isinstance-guard перед .value — event_type може бути None
            if isinstance(last_break.event_type, StructureEventType):
                reasons.append(f"latest_break_{last_break.event_type.value}")

        if mtf.get("internal_with_external_aligned"):
            reasons.append("internal_external_alignment")

        return reasons

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: SignalContext,
        structure: Mapping[str, Any],
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        freshness_filter: FilterResult | None,
    ) -> StrategySignal:
        last_break = self._break_context_or_empty(structure.get("last_break_event"))

        # FIX: витягуємо значення заздалегідь — уникаємо прямого ланцюжка
        # `.value` на потенційно None полях всередині dict-literal
        last_break_type = (
            last_break.event_type.value
            if isinstance(last_break.event_type, StructureEventType)
            else None
        )
        last_break_layer = (
            last_break.layer.value
            if isinstance(last_break.layer, StructureLayer)
            else None
        )
        # FIX: MarketBias.UNKNOWN — валідний enum, але direction завжди присутній
        # в BreakContext як дефолт, тому .value безпечний; проте додаємо guard
        # для консистентності з іншими класами
        last_break_direction = (
            last_break.direction.value
            if isinstance(last_break.direction, MarketBias)
            else None
        )

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=context.timeframe,
            setup_type=self._resolve_setup_type(last_break),
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=self._resolve_trigger_type(last_break),
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._resolve_priority(last_break=last_break, confidence=confidence),
            regime=self._resolve_market_regime(context),
            metadata={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.market_structure",
                "market_structure_timeframe": structure.get("timeframe"),
                "market_structure_last_update": (
                    structure.get("last_update").isoformat()
                    if isinstance(structure.get("last_update"), datetime)
                    else None
                ),
                "market_structure_last_price": structure.get("last_price"),
                "alignment_score": structure.get("mtf_alignment", {}).get("alignment_score"),
                "last_break_type": last_break_type,
                "last_break_direction": last_break_direction,
                "last_break_confidence": last_break.confidence,
                "last_break_layer": last_break_layer,
            },
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("price_action.market_structure")

        if freshness_filter is not None:
            signal.add_filter_result(freshness_filter)

        regime_filter = self._build_regime_filter(context=context, side=side)
        if regime_filter is not None:
            signal.add_filter_result(regime_filter)

        signal.validate()
        return signal

    def _resolve_setup_type(self, last_break: BreakContext) -> SetupType:
        if last_break.event_type == StructureEventType.BOS:
            return SetupType.BREAKOUT
        if last_break.event_type in {StructureEventType.CHOCH, StructureEventType.MSS}:
            return SetupType.REVERSAL
        return SetupType.CONTINUATION

    def _resolve_trigger_type(self, last_break: BreakContext) -> TriggerType:
        if last_break.is_break:
            return TriggerType.PRIMARY
        return TriggerType.DERIVED

    def _resolve_priority(
        self,
        *,
        last_break: BreakContext,
        confidence: float,
    ) -> SignalPriority:
        if last_break.event_type in {StructureEventType.CHOCH, StructureEventType.MSS} and confidence >= 0.70:
            return SignalPriority.HIGH
        if last_break.event_type == StructureEventType.BOS and confidence >= 0.75:
            return SignalPriority.HIGH
        if confidence >= 0.85:
            return SignalPriority.HIGH
        return SignalPriority.MEDIUM

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _break_context_or_empty(self, value: Any) -> BreakContext:
        return value if isinstance(value, BreakContext) else BreakContext()

    def _bias_to_side(self, bias: MarketBias) -> SignalSide:
        if bias == MarketBias.BULLISH:
            return SignalSide.LONG
        if bias == MarketBias.BEARISH:
            return SignalSide.SHORT
        return SignalSide.UNKNOWN

    def _parse_market_bias(self, value: Any) -> MarketBias:
        raw = enum_value(value)
        mapping = {
            "bullish": MarketBias.BULLISH,
            "bearish": MarketBias.BEARISH,
            "ranging": MarketBias.RANGING,
            "range": MarketBias.RANGING,
            "neutral": MarketBias.RANGING,
            "unknown": MarketBias.UNKNOWN,
        }
        return mapping.get(raw, MarketBias.UNKNOWN)

    def _parse_structure_layer(self, value: Any) -> StructureLayer | None:
        raw = enum_value(value)
        if raw == "internal":
            return StructureLayer.INTERNAL
        if raw == "external":
            return StructureLayer.EXTERNAL
        return None

    def _parse_structure_event_type(self, value: Any) -> StructureEventType | None:
        raw = enum_value(value)
        mapping = {
            "swing_high": StructureEventType.SWING_HIGH,
            "swing_low": StructureEventType.SWING_LOW,
            "hh": StructureEventType.HH,
            "hl": StructureEventType.HL,
            "lh": StructureEventType.LH,
            "ll": StructureEventType.LL,
            "bos": StructureEventType.BOS,
            "break_of_structure": StructureEventType.BOS,
            "choch": StructureEventType.CHOCH,
            "change_of_character": StructureEventType.CHOCH,
            "mss": StructureEventType.MSS,
            "market_structure_shift": StructureEventType.MSS,
        }
        return mapping.get(raw)