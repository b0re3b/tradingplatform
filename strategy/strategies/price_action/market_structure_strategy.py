from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from analytics.price_action.enums import MarketBias, StructureEventType, StructureLayer
from strategy.base import (
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
)
from strategy.config import StrategyConfig, StrategyDefinitionConfig
from strategy.enums import (
    FilterDecision,
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError, ValidationError
from strategy.models import (
    FilterResult,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        if value > 1_000_000_000_000:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    return None


def enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip().lower()


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


@dataclass(slots=True)
class MarketStructureStrategyParams:
    """
    Локальні параметри саме для market_structure_strategy.

    Очікується, що вони можуть приходити з:
    StrategyDefinitionConfig.metadata
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
        bounded = (
            "min_layer_confidence",
            "min_alignment_score",
            "min_break_confidence",
            "min_trend_strength",
            "external_bias_weight",
            "internal_bias_weight",
            "break_event_weight",
            "alignment_weight",
            "regime_alignment_weight",
        )
        for field_name in bounded:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{field_name} must be between 0.0 and 1.0")

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "MarketStructureStrategyParams":
        params = cls()
        if definition is None:
            params.validate()
            return params

        params.strategy_name = definition.name or params.strategy_name

        metadata = definition.metadata or {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name == "strategy_name":
                continue
            if field_name in metadata:
                setattr(params, field_name, metadata[field_name])

        params.validate()
        return params


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


class MarketStructureStrategy(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Production-ready strategy that converts analytics.price_action.market_structure
    snapshots/events into StrategySignal.

    Основні ідеї:
    - tolerant extraction із SignalContext.price_action та feature_map
    - підтримка internal/external layer
    - BOS = continuation / breakout
    - CHOCH / MSS = reversal
    - confidence/score gating
    - event emission hook через EventBus
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
        strategy_name: str = "market_structure_strategy",
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self._strategy_name = strategy_name
        self.validate_config()

        definition = None
        get_strategy = getattr(self.config, "get_strategy", None)
        if callable(get_strategy):
            definition = get_strategy(strategy_name)

        self.definition = definition
        self.params = MarketStructureStrategyParams.from_definition(definition)

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def priority(self) -> int:
        if self.definition is not None:
            return self.definition.priority
        return 100

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        self.validate_context(context)

        try:
            if not self._is_strategy_enabled():
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["strategy_disabled"],
                )

            if not self._symbol_allowed(context.symbol):
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["symbol_not_allowed"],
                )

            if not self._timeframe_allowed(context.timeframe):
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["timeframe_not_allowed"],
                )

            structure = self._extract_market_structure_snapshot(context)
            if not structure:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["market_structure_snapshot_missing"],
                )

            freshness_filter = self._build_freshness_filter(context)
            if freshness_filter is not None and freshness_filter.blocked:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["stale_market_structure_feature"],
                )

            side = self._resolve_side(context=context, structure=structure)
            if side == SignalSide.UNKNOWN:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["no_directional_market_structure_signal"],
                )

            score = self._compute_score(context=context, structure=structure, side=side)
            confidence = self._compute_confidence(context=context, structure=structure, side=side)

            reasons = self._build_reasons(structure=structure, side=side)
            signal = self._build_signal(
                context=context,
                structure=structure,
                side=side,
                score=score,
                confidence=confidence,
                reasons=reasons,
                freshness_filter=freshness_filter,
            )

            passed = self._passes_runtime_thresholds(signal)
            if not passed:
                signal.to_rejected()

            evaluation = self._evaluation(
                context=context,
                signal=signal,
                passed=passed,
                confidence=confidence,
                score=score,
                reasons=reasons,
            )

            return evaluation

        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate market structure context for {context.symbol}"
            ) from exc

    async def maybe_emit_signal(self, signal: StrategySignal) -> None:
        if not self.params.emit_signal_events:
            return

        await self.emit_event(
            self.params.signal_event_name,
            {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "side": signal.side.value,
                "timeframe": signal.timeframe.value,
                "setup_type": signal.setup_type.value,
                "score": signal.score,
                "confidence": signal.confidence,
                "status": signal.status.value,
                "priority": signal.priority.value,
                "reasons": list(signal.reasons),
                "confirmations": list(signal.confirmations),
                "source_features": list(signal.source_features),
                "metadata": dict(signal.metadata),
            },
            source=self.name,
        )

    # ------------------------------------------------------------------
    # Strategy gating
    # ------------------------------------------------------------------

    def _is_strategy_enabled(self) -> bool:
        is_strategy_enabled = getattr(self.config, "is_strategy_enabled", None)
        if callable(is_strategy_enabled):
            return bool(is_strategy_enabled(self.name))

        if self.definition is not None:
            return self.definition.runtime.enabled
        return self.config.runtime.enabled

    def _symbol_allowed(self, symbol: str) -> bool:
        runtime = self.definition.runtime if self.definition is not None else self.config.runtime
        return not runtime.symbols or symbol in runtime.symbols

    def _timeframe_allowed(self, timeframe: Timeframe) -> bool:
        runtime = self.definition.runtime if self.definition is not None else self.config.runtime
        return not runtime.timeframes or timeframe in runtime.timeframes

    def _passes_runtime_thresholds(self, signal: StrategySignal) -> bool:
        runtime = self.definition.runtime if self.definition is not None else self.config.runtime
        if signal.confidence < runtime.min_confidence:
            return False
        if signal.score < runtime.min_score:
            return False
        return True

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_market_structure_snapshot(self, context: SignalContext) -> dict[str, Any]:
        candidates: list[Any] = []

        candidates.append(context.price_action.get("market_structure"))
        candidates.append(context.price_action.get("structure"))
        candidates.append(context.get_feature("price_action.market_structure"))
        candidates.append(context.get_feature("market_structure"))
        candidates.append(context.get_feature("analytics.price_action.market_structure"))

        for candidate in candidates:
            normalized = self._normalize_structure_snapshot(candidate)
            if normalized:
                return normalized

        return {}

    def _normalize_structure_snapshot(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return {}

        state = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
        if not isinstance(state, Mapping):
            return {}

        internal = self._normalize_layer(state.get("internal"))
        external = self._normalize_layer(state.get("external"))
        mtf_alignment = self._normalize_alignment(state.get("mtf_alignment"))

        last_event = self._extract_last_break_event(state)
        symbol = first_non_empty(state.get("symbol"), payload.get("symbol"))
        timeframe = first_non_empty(state.get("timeframe"), payload.get("timeframe"))
        last_price = safe_float(first_non_empty(state.get("last_price"), payload.get("last_price")), 0.0)
        last_update = parse_datetime(first_non_empty(state.get("last_update"), payload.get("last_update")))

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_price": last_price if last_price > 0 else None,
            "last_update": last_update,
            "internal": internal,
            "external": external,
            "mtf_alignment": mtf_alignment,
            "last_break_event": last_event,
            "raw": dict(payload),
        }

    def _normalize_layer(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return {}

        return {
            "bias": self._parse_market_bias(payload.get("bias")),
            "confidence": clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            "trend_strength": clamp(safe_float(payload.get("trend_strength"), 0.0), 0.0, 1.0),
            "in_breakout": safe_bool(payload.get("in_breakout"), False),
            "swing_count": int(safe_float(payload.get("swing_count"), 0.0)),
            "event_count": int(safe_float(payload.get("event_count"), 0.0)),
            "sequence": list(payload.get("sequence", []) or []),
            "last_bos": self._normalize_structure_event(payload.get("last_bos"), StructureLayer.INTERNAL),
            "last_choch": self._normalize_structure_event(payload.get("last_choch"), StructureLayer.INTERNAL),
            "last_mss": self._normalize_structure_event(payload.get("last_mss"), StructureLayer.INTERNAL),
            "last_hh": self._normalize_structure_event(payload.get("last_hh"), StructureLayer.INTERNAL),
            "last_hl": self._normalize_structure_event(payload.get("last_hl"), StructureLayer.INTERNAL),
            "last_lh": self._normalize_structure_event(payload.get("last_lh"), StructureLayer.INTERNAL),
            "last_ll": self._normalize_structure_event(payload.get("last_ll"), StructureLayer.INTERNAL),
        }

    def _normalize_alignment(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return {}

        return {
            "higher_timeframe": payload.get("higher_timeframe"),
            "higher_timeframe_bias": self._parse_market_bias(payload.get("higher_timeframe_bias")),
            "higher_timeframe_confidence": clamp(
                safe_float(payload.get("higher_timeframe_confidence"), 0.0),
                0.0,
                1.0,
            ),
            "internal_bias_aligned": safe_bool(payload.get("internal_bias_aligned")),
            "external_bias_aligned": safe_bool(payload.get("external_bias_aligned")),
            "internal_with_external_aligned": safe_bool(payload.get("internal_with_external_aligned")),
            "alignment_score": clamp(safe_float(payload.get("alignment_score"), 0.0), 0.0, 1.0),
            "last_updated": parse_datetime(payload.get("last_updated")),
        }

    def _normalize_structure_event(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        if payload is None:
            return None

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return None

        event_type = self._parse_structure_event_type(payload.get("event_type"))
        layer = self._parse_structure_layer(payload.get("layer")) or default_layer
        direction = self._parse_market_bias(payload.get("direction"))
        confidence = clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0)

        return {
            "event_id": payload.get("event_id"),
            "event_type": event_type,
            "timestamp": parse_datetime(payload.get("timestamp")),
            "price": safe_float(payload.get("price"), 0.0),
            "layer": layer,
            "direction": direction,
            "reference_price": (
                safe_float(payload.get("reference_price"), 0.0)
                if payload.get("reference_price") is not None
                else None
            ),
            "reference_swing_id": payload.get("reference_swing_id"),
            "confidence": confidence,
            "metadata": dict(payload.get("metadata", {}) or {}),
        }

    def _extract_last_break_event(self, state: Mapping[str, Any]) -> BreakContext:
        candidates: list[dict[str, Any]] = []

        internal = self._normalize_layer(state.get("internal"))
        external = self._normalize_layer(state.get("external"))

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

        candidates.sort(
            key=lambda x: x.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
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
        last_break = structure.get("last_break_event")

        primary_layer = external if self.params.prefer_external_layer else internal
        secondary_layer = internal if self.params.prefer_external_layer else external

        primary_bias = primary_layer.get("bias", MarketBias.UNKNOWN)
        secondary_bias = secondary_layer.get("bias", MarketBias.UNKNOWN)

        if self.params.require_alignment and not mtf.get("internal_with_external_aligned", False):
            return SignalSide.UNKNOWN

        if self.params.require_break_event and not last_break.is_break:
            return SignalSide.UNKNOWN

        if last_break.is_break:
            if last_break.event_type == StructureEventType.BOS and self.params.allow_bos_entries:
                return self._bias_to_side(last_break.direction)

            if last_break.event_type == StructureEventType.CHOCH and self.params.allow_choch_reversals:
                if self.params.reverse_on_choch:
                    return self._bias_to_side(last_break.direction)

            if last_break.event_type == StructureEventType.MSS and self.params.allow_mss_reversals:
                if self.params.reverse_on_mss:
                    return self._bias_to_side(last_break.direction)

        if primary_bias == secondary_bias and primary_bias in {MarketBias.BULLISH, MarketBias.BEARISH}:
            return self._bias_to_side(primary_bias)

        if primary_bias in {MarketBias.BULLISH, MarketBias.BEARISH}:
            return self._bias_to_side(primary_bias)

        return SignalSide.UNKNOWN

    def _compute_score(
        self,
        context: SignalContext,
        structure: Mapping[str, Any],
        side: SignalSide,
    ) -> float:
        internal = structure.get("internal", {})
        external = structure.get("external", {})
        mtf = structure.get("mtf_alignment", {})
        last_break: BreakContext = structure.get("last_break_event", BreakContext())

        score = 0.0

        external_bias = self._bias_to_side(external.get("bias", MarketBias.UNKNOWN))
        internal_bias = self._bias_to_side(internal.get("bias", MarketBias.UNKNOWN))

        if external_bias == side:
            score += self.params.external_bias_weight * (
                0.5 + 0.5 * clamp(safe_float(external.get("confidence"), 0.0), 0.0, 1.0)
            )

        if internal_bias == side:
            score += self.params.internal_bias_weight * (
                0.5 + 0.5 * clamp(safe_float(internal.get("confidence"), 0.0), 0.0, 1.0)
            )

        if last_break.is_break and self._bias_to_side(last_break.direction) == side:
            score += self.params.break_event_weight * (
                0.5 + 0.5 * clamp(last_break.confidence, 0.0, 1.0)
            )

        alignment_score = clamp(safe_float(mtf.get("alignment_score"), 0.0), 0.0, 1.0)
        score += self.params.alignment_weight * alignment_score

        score += self.params.regime_alignment_weight * self._regime_alignment_score(
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
        last_break: BreakContext = structure.get("last_break_event", BreakContext())

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
        last_break: BreakContext = structure.get("last_break_event", BreakContext())

        if side == SignalSide.LONG:
            reasons.append("market_structure_bullish")
        elif side == SignalSide.SHORT:
            reasons.append("market_structure_bearish")

        if self._bias_to_side(external.get("bias", MarketBias.UNKNOWN)) == side:
            reasons.append("external_bias_aligned")

        if self._bias_to_side(internal.get("bias", MarketBias.UNKNOWN)) == side:
            reasons.append("internal_bias_aligned")

        if last_break.is_break and self._bias_to_side(last_break.direction) == side:
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
        last_break: BreakContext = structure.get("last_break_event", BreakContext())

        setup_type = self._resolve_setup_type(last_break)
        priority = self._resolve_priority(last_break=last_break, confidence=confidence)

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=context.timeframe,
            setup_type=setup_type,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=self._resolve_trigger_type(last_break),
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
            regime=self._resolve_regime(context),
            metadata={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.market_structure",
                "market_structure_timeframe": structure.get("timeframe"),
                "market_structure_last_update": (
                    structure.get("last_update").isoformat()
                    if structure.get("last_update") is not None
                    else None
                ),
                "market_structure_last_price": structure.get("last_price"),
                "alignment_score": structure.get("mtf_alignment", {}).get("alignment_score"),
                "last_break_type": (
                    last_break.event_type.value if last_break.event_type is not None else None
                ),
                "last_break_direction": last_break.direction.value,
                "last_break_confidence": last_break.confidence,
                "last_break_layer": (
                    last_break.layer.value if last_break.layer is not None else None
                ),
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
    # Filters
    # ------------------------------------------------------------------

    def _build_freshness_filter(self, context: SignalContext) -> FilterResult | None:
        for feature_name in self.params.freshness_feature_names:
            if context.has_feature(feature_name):
                is_stale = context.feature_is_stale(feature_name)
                return FilterResult(
                    name="market_structure_freshness",
                    decision=FilterDecision.BLOCK if is_stale else FilterDecision.PASS,
                    reason="feature_stale" if is_stale else "feature_fresh",
                    score_impact=-1.0 if is_stale else 0.0,
                    metadata={"feature_name": feature_name},
                )
        return None

    def _build_regime_filter(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> FilterResult | None:
        runtime = self.definition.runtime if self.definition is not None else self.config.runtime
        if not runtime.allowed_regimes:
            return None

        regime = self._resolve_regime(context)
        if MarketRegime.UNKNOWN in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
            )

        if regime in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
            )

        return FilterResult(
            name="market_regime",
            decision=FilterDecision.BLOCK,
            reason=f"regime_{regime.value}_not_allowed_for_{side.value}",
            score_impact=-1.0,
        )

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _evaluation(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal | None = None,
        passed: bool,
        confidence: float,
        score: float,
        reasons: list[str],
    ) -> StrategyEvaluation:
        evaluation = StrategyEvaluation(
            strategy_name=self.name,
            symbol=context.symbol,
            timestamp=context.timestamp,
            signal=signal,
            passed=passed,
            score=score,
            confidence=confidence,
            reasons=list(reasons),
            metadata={
                "category": StrategyCategory.PRICE_ACTION.value,
                "timeframe": context.timeframe.value,
            },
        )
        evaluation.validate()
        return evaluation

    def _regime_alignment_score(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> float:
        regime = self._resolve_regime(context)

        bullish_regimes = {
            MarketRegime.TRENDING_UP,
            MarketRegime.BREAKOUT,
            MarketRegime.SQUEEZE,
        }
        bearish_regimes = {
            MarketRegime.TRENDING_DOWN,
            MarketRegime.BREAKOUT,
            MarketRegime.SQUEEZE,
        }

        if regime == MarketRegime.UNKNOWN:
            return 0.5
        if regime == MarketRegime.RANGING:
            return 0.35
        if regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_DRIVEN}:
            return 0.40
        if side == SignalSide.LONG and regime in bullish_regimes:
            return 1.0
        if side == SignalSide.SHORT and regime in bearish_regimes:
            return 1.0
        return 0.20

    def _resolve_regime(self, context: SignalContext) -> MarketRegime:
        regime = context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN
        if isinstance(regime, MarketRegime):
            return regime

        raw = enum_value(regime)
        mapping = {
            "trending_up": MarketRegime.TRENDING_UP,
            "trending_down": MarketRegime.TRENDING_DOWN,
            "ranging": MarketRegime.RANGING,
            "breakout": MarketRegime.BREAKOUT,
            "squeeze": MarketRegime.SQUEEZE,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
            "low_volatility": MarketRegime.LOW_VOLATILITY,
            "news_driven": MarketRegime.NEWS_DRIVEN,
            "illiquid": MarketRegime.ILLIQUID,
            "risk_off": MarketRegime.RISK_OFF,
        }
        return mapping.get(raw, MarketRegime.UNKNOWN)

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
            "choch": StructureEventType.CHOCH,
            "mss": StructureEventType.MSS,
        }
        return mapping.get(raw)
