from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import StructureLayer, TrendDirection, TrendEventType, TrendRegime
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
class TrendContinuationStrategyParams:
    strategy_name: str = "trend_continuation_strategy"

    prefer_external_layer: bool = True
    require_internal_confirmation: bool = True
    allow_cross_layer_fallback: bool = True
    require_direction_alignment: bool = True

    allow_pullback_entries: bool = True
    block_exhausted_trend: bool = True
    block_high_reversal_risk: bool = True
    block_counter_regime: bool = False

    min_layer_confidence: float = 0.50
    min_trend_strength: float = 0.55
    min_continuation_probability: float = 0.55
    max_reversal_risk: float = 0.60
    max_exhaustion_score: float = 0.72
    max_consolidation_score: float = 0.70
    max_pullback_depth: float = 0.80

    external_confidence_weight: float = 0.26
    external_strength_weight: float = 0.20
    internal_confirmation_weight: float = 0.14
    continuation_probability_weight: float = 0.18
    structure_alignment_weight: float = 0.08
    regime_alignment_weight: float = 0.08
    pullback_bonus_weight: float = 0.03
    acceleration_bonus_weight: float = 0.03

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.trend_continuation.signal"

    freshness_feature_names: tuple[str, ...] = (
        "price_action.trend",
        "trend",
        "analytics.price_action.trend",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_layer_confidence",
                "min_trend_strength",
                "min_continuation_probability",
                "max_reversal_risk",
                "max_exhaustion_score",
                "max_consolidation_score",
                "max_pullback_depth",
                "external_confidence_weight",
                "external_strength_weight",
                "internal_confirmation_weight",
                "continuation_probability_weight",
                "structure_alignment_weight",
                "regime_alignment_weight",
                "pullback_bonus_weight",
                "acceleration_bonus_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "TrendContinuationStrategyParams":
        return apply_definition_metadata(
            params=cls(),
            definition=definition,
        )


@dataclass(slots=True)
class TrendContextView:
    symbol: str | None = None
    timeframe: str | None = None
    last_price: float | None = None
    last_update: datetime | None = None
    internal: dict[str, Any] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)
    global_state: dict[str, Any] = field(default_factory=dict)
    last_signal: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class TrendContinuationStrategy(PriceActionStrategyBase):
    """
    Strategy layer wrapper around analytics.price_action.trend.

    Основна логіка:
    - long continuation, коли тренд bullish і continuation probability достатня
    - short continuation, коли тренд bearish і continuation probability достатня
    - опційно вимагає internal confirmation
    - відсікає exhaustion / reversal risk / stale feature

    Інфраструктура:
    - runtime/config gating, logger, EventBus emission і базова оцінка йдуть через
      PriceActionStrategyBase, щоб не дублювати core-логіку в конкретній стратегії.
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
        strategy_name: str = "trend_continuation_strategy",
    ) -> None:
        super().__init__(
            config=config,
            strategy_name=strategy_name,
            params_cls=TrendContinuationStrategyParams,
            event_bus=event_bus,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Typed params accessor — fixes "Cannot find reference in ParamsT"
    # ------------------------------------------------------------------

    @property
    def _p(self) -> TrendContinuationStrategyParams:
        """Typed shortcut so IDE resolves all TrendContinuationStrategyParams fields."""
        return cast(TrendContinuationStrategyParams, self.params)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked

            trend = self._extract_trend_snapshot(context)
            if trend.symbol is None and not trend.external and not trend.internal:
                return self._rejected_evaluation(
                    context=context,
                    reason="trend_snapshot_missing",
                )

            freshness_filter = self._build_freshness_filter(
                context=context,
                filter_name="trend_freshness",
            )
            if freshness_filter is not None and freshness_filter.blocked:
                return self._rejected_evaluation(
                    context=context,
                    reason="stale_trend_feature",
                )

            side = self._resolve_side(context=context, trend=trend)
            if side == SignalSide.UNKNOWN:
                return self._rejected_evaluation(
                    context=context,
                    reason="no_valid_trend_continuation_setup",
                )

            primary_layer = self._select_primary_layer(trend)
            # FIX: доступ через _p замість self.params
            primary_layer_name = "external" if self._p.prefer_external_layer else "internal"
            secondary_layer = trend.internal if self._p.prefer_external_layer else trend.external
            secondary_layer_name = "internal" if self._p.prefer_external_layer else "external"

            score = self._compute_score(
                context=context,
                trend=trend,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )
            confidence = self._compute_confidence(
                context=context,
                trend=trend,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )

            reasons = self._build_reasons(
                trend=trend,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                primary_layer_name=primary_layer_name,
                secondary_layer_name=secondary_layer_name,
            )

            signal = self._build_signal(
                context=context,
                trend=trend,
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
                "Failed to evaluate trend continuation strategy | strategy=%s symbol=%s",
                self.name,
                getattr(context, "symbol", None),
            )
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate trend continuation for {context.symbol}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_trend_snapshot(self, context: SignalContext) -> TrendContextView:
        candidates: list[Any] = [
            context.price_action.get("trend"),
            context.get_feature("price_action.trend"),
            context.get_feature("trend"),
            context.get_feature("analytics.price_action.trend"),
        ]

        for candidate in candidates:
            normalized = self._normalize_trend_snapshot(candidate)
            if normalized.symbol is not None or normalized.external or normalized.internal:
                return normalized

        return TrendContextView()

    def _normalize_trend_snapshot(self, payload: Any) -> TrendContextView:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return TrendContextView()

        state = self._state_mapping_or_empty(payload_mapping)
        if not state:
            return TrendContextView()

        internal = self._normalize_trend_layer(state.get("internal"), StructureLayer.INTERNAL)
        external = self._normalize_trend_layer(state.get("external"), StructureLayer.EXTERNAL)
        global_state = self._normalize_global_state(state)

        return TrendContextView(
            symbol=first_non_empty(state.get("symbol"), payload_mapping.get("symbol")),
            timeframe=first_non_empty(state.get("timeframe"), payload_mapping.get("timeframe")),
            last_price=(
                safe_float(
                    first_non_empty(
                        state.get("last_price"),
                        payload_mapping.get("last_price"),
                    ),
                    0.0,
                )
                or None
            ),
            last_update=parse_datetime(
                first_non_empty(
                    state.get("last_update"),
                    payload_mapping.get("last_update"),
                )
            ),
            internal=internal,
            external=external,
            global_state=global_state,
            last_signal=self._extract_last_signal(internal, external),
            raw=dict(payload_mapping),
        )

    def _normalize_trend_layer(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        last_signal = self._mapping_or_empty(payload_mapping.get("last_signal")) or None

        return {
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
            "direction": self._parse_trend_direction(payload_mapping.get("direction")),
            "regime": self._parse_trend_regime(payload_mapping.get("regime")),
            "strength": clamp(safe_float(payload_mapping.get("strength"), 0.0), 0.0, 1.0),
            "confidence": clamp(safe_float(payload_mapping.get("confidence"), 0.0), 0.0, 1.0),
            "continuation_probability": clamp(
                safe_float(payload_mapping.get("continuation_probability"), 0.0),
                0.0,
                1.0,
            ),
            "reversal_risk": clamp(safe_float(payload_mapping.get("reversal_risk"), 0.0), 0.0, 1.0),
            "exhaustion_score": clamp(safe_float(payload_mapping.get("exhaustion_score"), 0.0), 0.0, 1.0),
            "pullback_depth": clamp(safe_float(payload_mapping.get("pullback_depth"), 0.0), 0.0, 1.0),
            "consolidation_score": clamp(safe_float(payload_mapping.get("consolidation_score"), 0.0), 0.0, 1.0),
            "structure_score": clamp(safe_float(payload_mapping.get("structure_score"), 0.0), -1.0, 1.0),
            "is_accelerating": safe_bool(payload_mapping.get("is_accelerating"), False),
            "is_exhausted": safe_bool(payload_mapping.get("is_exhausted"), False),
            "in_pullback": safe_bool(payload_mapping.get("in_pullback"), False),
            "last_signal": self._normalize_trend_signal(last_signal, default_layer),
        }

    def _normalize_trend_signal(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        return {
            "signal_id": payload_mapping.get("signal_id"),
            "timestamp": parse_datetime(payload_mapping.get("timestamp")),
            "symbol": payload_mapping.get("symbol"),
            "timeframe": payload_mapping.get("timeframe"),
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
            "event_type": self._parse_trend_event_type(payload_mapping.get("event_type")),
            "direction": self._parse_trend_direction(payload_mapping.get("direction")),
            "strength": clamp(safe_float(payload_mapping.get("strength"), 0.0), 0.0, 1.0),
            "confidence": clamp(safe_float(payload_mapping.get("confidence"), 0.0), 0.0, 1.0),
            "regime": self._parse_trend_regime(payload_mapping.get("regime")),
            "price": (
                safe_float(payload_mapping.get("price"), 0.0)
                if payload_mapping.get("price") is not None
                else None
            ),
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_global_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        global_candidates = (
            state.get("global_state"),
            state.get("summary"),
            state.get("metadata"),
        )

        for candidate in global_candidates:
            candidate_mapping = self._mapping_or_empty(candidate)
            if candidate_mapping:
                return dict(candidate_mapping)

        return {}

    def _extract_last_signal(
        self,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        # FIX: timezone-aware _epoch — аналогічно до FVGReactionStrategy
        _epoch = datetime.min.replace(tzinfo=timezone.utc)

        candidates = [
            signal
            for signal in (
                external.get("last_signal"),
                internal.get("last_signal"),
            )
            if signal is not None
        ]
        if not candidates:
            return None

        def _sort_key(x: dict[str, Any]) -> datetime:
            ts = x.get("timestamp")
            if ts is None:
                return _epoch
            if isinstance(ts, datetime) and ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        candidates.sort(key=_sort_key, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Direction resolution
    # ------------------------------------------------------------------

    def _select_primary_layer(self, trend: TrendContextView) -> dict[str, Any]:
        # FIX: доступ через _p
        return trend.external if self._p.prefer_external_layer else trend.internal

    def _resolve_side(self, context: SignalContext, trend: TrendContextView) -> SignalSide:
        primary = self._select_primary_layer(trend)
        # FIX: доступ через _p
        secondary = trend.internal if self._p.prefer_external_layer else trend.external

        primary_side = self._trend_direction_to_side(primary.get("direction", TrendDirection.UNKNOWN))
        secondary_side = self._trend_direction_to_side(secondary.get("direction", TrendDirection.UNKNOWN))

        if primary_side == SignalSide.UNKNOWN:
            if self._p.allow_cross_layer_fallback:
                return secondary_side
            return SignalSide.UNKNOWN

        if not self._layer_eligible(primary):
            return SignalSide.UNKNOWN

        if self._p.require_internal_confirmation and secondary_side != SignalSide.UNKNOWN:
            if secondary_side != primary_side:
                return SignalSide.UNKNOWN
            if not self._layer_confirmation_ok(secondary):
                return SignalSide.UNKNOWN

        if self._p.require_direction_alignment and secondary_side not in {SignalSide.UNKNOWN, primary_side}:
            return SignalSide.UNKNOWN

        if not self._side_regime_allowed(primary_side, primary):
            return SignalSide.UNKNOWN

        return primary_side

    def _layer_eligible(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False
        # FIX: доступ через _p
        if clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0) < self._p.min_layer_confidence:
            return False
        if clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0) < self._p.min_trend_strength:
            return False
        if clamp(safe_float(layer.get("continuation_probability"), 0.0), 0.0, 1.0) < self._p.min_continuation_probability:
            return False
        if self._p.block_high_reversal_risk and clamp(safe_float(layer.get("reversal_risk"), 0.0), 0.0, 1.0) > self._p.max_reversal_risk:
            return False
        if self._p.block_exhausted_trend:
            exhausted = safe_bool(layer.get("is_exhausted"), False)
            exhaustion_score = clamp(safe_float(layer.get("exhaustion_score"), 0.0), 0.0, 1.0)
            if exhausted or exhaustion_score > self._p.max_exhaustion_score:
                return False
        if clamp(safe_float(layer.get("consolidation_score"), 0.0), 0.0, 1.0) > self._p.max_consolidation_score:
            return False
        return True

    def _layer_confirmation_ok(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False
        # FIX: доступ через _p
        return (
            clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0) >= self._p.min_layer_confidence * 0.85
            and clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0) >= self._p.min_trend_strength * 0.85
        )

    def _side_regime_allowed(self, side: SignalSide, layer: Mapping[str, Any]) -> bool:
        # FIX: доступ через _p
        if not self._p.block_counter_regime:
            return True

        regime = layer.get("regime", TrendRegime.UNKNOWN)
        continuation_regimes = {
            TrendRegime.TRENDING,
            TrendRegime.PULLBACK,
        }
        return regime in continuation_regimes

    # ------------------------------------------------------------------
    # Score / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        context: SignalContext,
        trend: TrendContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> float:
        score = 0.0

        # FIX: доступ через _p
        score += self._p.external_confidence_weight * clamp(
            safe_float(primary_layer.get("confidence"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.external_strength_weight * clamp(
            safe_float(primary_layer.get("strength"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.continuation_probability_weight * clamp(
            safe_float(primary_layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            score += self._p.internal_confirmation_weight * clamp(
                safe_float(secondary_layer.get("confidence"), 0.0),
                0.0,
                1.0,
            )

        structure_alignment = self._structure_alignment_score(primary_layer, side)
        score += self._p.structure_alignment_weight * structure_alignment

        regime_alignment = self._regime_alignment_score(context=context, side=side)
        score += self._p.regime_alignment_weight * regime_alignment

        if self._p.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            pullback_depth = clamp(safe_float(primary_layer.get("pullback_depth"), 0.0), 0.0, 1.0)
            if pullback_depth <= self._p.max_pullback_depth:
                score += self._p.pullback_bonus_weight * (1.0 - pullback_depth)

        if safe_bool(primary_layer.get("is_accelerating"), False):
            score += self._p.acceleration_bonus_weight

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        context: SignalContext,
        trend: TrendContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> float:
        components: list[float] = [
            clamp(safe_float(primary_layer.get("confidence"), 0.0), 0.0, 1.0),
            clamp(safe_float(primary_layer.get("strength"), 0.0), 0.0, 1.0),
            clamp(safe_float(primary_layer.get("continuation_probability"), 0.0), 0.0, 1.0),
        ]

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            components.append(clamp(safe_float(secondary_layer.get("confidence"), 0.0), 0.0, 1.0))

        components.append(self._structure_alignment_score(primary_layer, side))
        components.append(self._regime_alignment_score(context=context, side=side))

        reversal_risk = clamp(safe_float(primary_layer.get("reversal_risk"), 0.0), 0.0, 1.0)
        exhaustion_score = clamp(safe_float(primary_layer.get("exhaustion_score"), 0.0), 0.0, 1.0)

        confidence = sum(components) / len(components)
        confidence *= (1.0 - reversal_risk * 0.35)
        confidence *= (1.0 - exhaustion_score * 0.25)

        return clamp(confidence, 0.0, 1.0)

    def _build_reasons(
        self,
        trend: TrendContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        primary_layer_name: str,
        secondary_layer_name: str,
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.append("trend_continuation_bullish")
        elif side == SignalSide.SHORT:
            reasons.append("trend_continuation_bearish")

        reasons.append(f"primary_layer_{primary_layer_name}")
        reasons.append(f"{primary_layer_name}_direction_{enum_value(primary_layer.get('direction'))}")
        reasons.append(f"{primary_layer_name}_regime_{enum_value(primary_layer.get('regime'))}")

        if self._trend_direction_to_side(secondary_layer.get("direction", TrendDirection.UNKNOWN)) == side:
            reasons.append(f"{secondary_layer_name}_confirmation")

        if safe_bool(primary_layer.get("is_accelerating"), False):
            reasons.append("trend_accelerating")

        # FIX: доступ через _p
        if self._p.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            reasons.append("continuation_pullback_entry")

        last_signal = trend.last_signal
        if last_signal is not None and self._trend_direction_to_side(
            last_signal.get("direction", TrendDirection.UNKNOWN)
        ) == side:
            event_type = last_signal.get("event_type")
            # FIX: isinstance-guard перед .value
            if isinstance(event_type, TrendEventType):
                reasons.append(f"last_trend_event_{event_type.value}")

        return reasons

    # ------------------------------------------------------------------
    # Signal build
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        context: SignalContext,
        trend: TrendContextView,
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        freshness_filter: FilterResult | None,
    ) -> StrategySignal:
        primary_layer = self._select_primary_layer(trend)
        last_signal = trend.last_signal

        priority = self._resolve_priority(
            confidence=confidence,
            primary_layer=primary_layer,
            last_signal=last_signal,
        )

        last_event_type = last_signal.get("event_type") if last_signal is not None else None
        last_layer = last_signal.get("layer") if last_signal is not None else None

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=context.timeframe,
            setup_type=SetupType.CONTINUATION,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=self._resolve_trigger_type(primary_layer, last_signal),
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
            regime=self._resolve_market_regime(context),
            metadata={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.trend",
                "trend_timeframe": trend.timeframe,
                "trend_last_update": trend.last_update.isoformat() if trend.last_update else None,
                "trend_last_price": trend.last_price,
                "primary_direction": enum_value(primary_layer.get("direction")),
                "primary_regime": enum_value(primary_layer.get("regime")),
                "primary_strength": safe_float(primary_layer.get("strength"), 0.0),
                "primary_confidence": safe_float(primary_layer.get("confidence"), 0.0),
                "continuation_probability": safe_float(primary_layer.get("continuation_probability"), 0.0),
                "reversal_risk": safe_float(primary_layer.get("reversal_risk"), 0.0),
                "exhaustion_score": safe_float(primary_layer.get("exhaustion_score"), 0.0),
                "in_pullback": safe_bool(primary_layer.get("in_pullback"), False),
                "is_accelerating": safe_bool(primary_layer.get("is_accelerating"), False),
                # FIX: isinstance-guard перед .value — уникає AttributeError якщо None
                "last_trend_event_type": (
                    last_event_type.value
                    if isinstance(last_event_type, TrendEventType)
                    else None
                ),
                "last_trend_event_layer": (
                    last_layer.value
                    if isinstance(last_layer, StructureLayer)
                    else None
                ),
            },
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("price_action.trend")

        if freshness_filter is not None:
            signal.add_filter_result(freshness_filter)

        regime_filter = self._build_regime_filter(context=context, side=side)
        if regime_filter is not None:
            signal.add_filter_result(regime_filter)

        signal.validate()
        return signal

    def _resolve_priority(
        self,
        *,
        confidence: float,
        primary_layer: Mapping[str, Any],
        last_signal: Mapping[str, Any] | None,
    ) -> SignalPriority:
        continuation_probability = clamp(
            safe_float(primary_layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )

        if confidence >= 0.85 and continuation_probability >= 0.80:
            return SignalPriority.HIGH

        if (
            last_signal is not None
            and last_signal.get("event_type") in {
                TrendEventType.TREND_CONTINUATION,
                TrendEventType.TREND_ACCELERATION,
                TrendEventType.TREND_ALIGNMENT,
            }
            and confidence >= 0.70
        ):
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    def _resolve_trigger_type(
        self,
        primary_layer: Mapping[str, Any],
        last_signal: Mapping[str, Any] | None,
    ) -> TriggerType:
        if last_signal is not None:
            event_type = last_signal.get("event_type")
            if event_type in {
                TrendEventType.TREND_CONTINUATION,
                TrendEventType.TREND_ACCELERATION,
                TrendEventType.TREND_ALIGNMENT,
            }:
                return TriggerType.PRIMARY
            if event_type in {
                TrendEventType.PULLBACK_STARTED,
                TrendEventType.PULLBACK_ENDED,
                TrendEventType.TREND_WEAKENING,
            }:
                return TriggerType.CONFIRMATION

        if safe_bool(primary_layer.get("in_pullback"), False):
            return TriggerType.CONFIRMATION

        return TriggerType.DERIVED

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _structure_alignment_score(
        self,
        layer: Mapping[str, Any],
        side: SignalSide,
    ) -> float:
        structure_score = clamp(safe_float(layer.get("structure_score"), 0.0), -1.0, 1.0)
        if side == SignalSide.LONG:
            return clamp(max(0.0, structure_score), 0.0, 1.0)
        if side == SignalSide.SHORT:
            return clamp(max(0.0, -structure_score), 0.0, 1.0)
        return 0.0

    def _trend_direction_to_side(self, direction: TrendDirection) -> SignalSide:
        if direction == TrendDirection.BULLISH:
            return SignalSide.LONG
        if direction == TrendDirection.BEARISH:
            return SignalSide.SHORT
        return SignalSide.UNKNOWN

    def _parse_structure_layer(self, value: Any) -> StructureLayer | None:
        raw = enum_value(value)
        if raw == "internal":
            return StructureLayer.INTERNAL
        if raw == "external":
            return StructureLayer.EXTERNAL
        return None

    def _parse_trend_direction(self, value: Any) -> TrendDirection:
        raw = enum_value(value)
        mapping = {
            "bullish": TrendDirection.BULLISH,
            "bearish": TrendDirection.BEARISH,
            "neutral": TrendDirection.NEUTRAL,
            "unknown": TrendDirection.UNKNOWN,
        }
        return mapping.get(raw, TrendDirection.UNKNOWN)

    def _parse_trend_regime(self, value: Any) -> TrendRegime:
        raw = enum_value(value)
        mapping = {
            "trending": TrendRegime.TRENDING,
            "trend": TrendRegime.TRENDING,
            "trending_up": TrendRegime.TRENDING,
            "trending_down": TrendRegime.TRENDING,
            "accelerating_up": TrendRegime.TRENDING,
            "accelerating_down": TrendRegime.TRENDING,
            "pullback": TrendRegime.PULLBACK,
            "pullback_uptrend": TrendRegime.PULLBACK,
            "pullback_downtrend": TrendRegime.PULLBACK,
            "consolidating": TrendRegime.CONSOLIDATING,
            "consolidation": TrendRegime.CONSOLIDATING,
            "ranging": TrendRegime.CONSOLIDATING,
            "reversing": TrendRegime.REVERSING,
            "reversal": TrendRegime.REVERSING,
            "exhausted": TrendRegime.EXHAUSTED,
            "exhaustion": TrendRegime.EXHAUSTED,
            "unknown": TrendRegime.UNKNOWN,
        }
        if raw in mapping:
            return mapping[raw]
        try:
            return TrendRegime(raw)
        except Exception:
            return TrendRegime.UNKNOWN

    def _parse_trend_event_type(self, value: Any) -> TrendEventType | None:
        raw = enum_value(value)
        if not raw:
            return None

        mapping = {
            "trend_started": TrendEventType.TREND_STARTED,
            "trend_continuation": TrendEventType.TREND_CONTINUATION,
            "continuation_confirmed": TrendEventType.TREND_CONTINUATION,
            "continuation": TrendEventType.TREND_CONTINUATION,
            "trend_acceleration": TrendEventType.TREND_ACCELERATION,
            "acceleration": TrendEventType.TREND_ACCELERATION,
            "trend_weakening": TrendEventType.TREND_WEAKENING,
            "weakening": TrendEventType.TREND_WEAKENING,
            "pullback_started": TrendEventType.PULLBACK_STARTED,
            "pullback": TrendEventType.PULLBACK_STARTED,
            "pullback_ended": TrendEventType.PULLBACK_ENDED,
            "resumption": TrendEventType.PULLBACK_ENDED,
            "trend_reversal": TrendEventType.TREND_REVERSAL,
            "reversal": TrendEventType.TREND_REVERSAL,
            "trend_exhaustion": TrendEventType.TREND_EXHAUSTION,
            "exhaustion": TrendEventType.TREND_EXHAUSTION,
            "trend_alignment": TrendEventType.TREND_ALIGNMENT,
            "alignment": TrendEventType.TREND_ALIGNMENT,
            "trend_disagreement": TrendEventType.TREND_DISAGREEMENT,
            "disagreement": TrendEventType.TREND_DISAGREEMENT,
        }
        if raw in mapping:
            return mapping[raw]
        try:
            return TrendEventType(raw)
        except Exception:
            return None