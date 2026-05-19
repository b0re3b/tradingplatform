from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import (
    StructureLayer,
    TrendDirection,
    TrendEventType,
    TrendRegime,
)
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
    """
    Local params for TrendContinuationStrategy.

    Runtime gates such as enabled/symbols/timeframes/min_score/min_confidence
    stay in StrategyConfig / StrategyDefinitionConfig.runtime. These params
    control only how this strategy consumes analytics.price_action.trend.
    """

    strategy_name: str = "trend_continuation_strategy"

    prefer_external_layer: bool = True
    require_internal_confirmation: bool = True
    allow_cross_layer_fallback: bool = True
    require_direction_alignment: bool = True

    # Trend quality gates.
    allow_pullback_entries: bool = True
    block_exhausted_trend: bool = True
    block_high_reversal_risk: bool = True
    block_counter_regime: bool = False
    require_structure_alignment: bool = False
    require_positive_momentum: bool = True
    require_positive_slope: bool = False
    require_higher_timeframe_alignment: bool = False

    min_layer_confidence: float = 0.50
    min_trend_strength: float = 0.55
    min_continuation_probability: float = 0.55
    min_directional_momentum: float = 0.10
    min_directional_slope: float = 0.00
    min_structure_score: float = 0.25
    min_internal_external_alignment: float = 0.35
    min_higher_timeframe_alignment: float = 0.35
    min_overall_trend_score: float = 0.35

    max_reversal_risk: float = 0.60
    max_exhaustion_score: float = 0.72
    max_consolidation_score: float = 0.70
    max_pullback_depth: float = 0.80

    # Score components. These are intentionally local weights. The final score
    # is clamped to [0, 1], so the weights do not have to sum exactly to 1.0.
    primary_confidence_weight: float = 0.16
    primary_strength_weight: float = 0.15
    continuation_probability_weight: float = 0.16
    momentum_direction_weight: float = 0.10
    slope_direction_weight: float = 0.07
    structure_alignment_weight: float = 0.10
    internal_confirmation_weight: float = 0.10
    global_alignment_weight: float = 0.08
    higher_timeframe_alignment_weight: float = 0.04
    regime_alignment_weight: float = 0.04
    pullback_bonus_weight: float = 0.04
    acceleration_bonus_weight: float = 0.04

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.trend_continuation.signal"

    freshness_feature_names: tuple[str, ...] = (
        "analytics.price_action",
        "analytics.price_action.trend",
        "price_action.trend",
        "trend",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_layer_confidence",
                "min_trend_strength",
                "min_continuation_probability",
                "min_directional_momentum",
                "min_directional_slope",
                "min_structure_score",
                "min_internal_external_alignment",
                "min_higher_timeframe_alignment",
                "min_overall_trend_score",
                "max_reversal_risk",
                "max_exhaustion_score",
                "max_consolidation_score",
                "max_pullback_depth",
                "primary_confidence_weight",
                "primary_strength_weight",
                "continuation_probability_weight",
                "momentum_direction_weight",
                "slope_direction_weight",
                "structure_alignment_weight",
                "internal_confirmation_weight",
                "global_alignment_weight",
                "higher_timeframe_alignment_weight",
                "regime_alignment_weight",
                "pullback_bonus_weight",
                "acceleration_bonus_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

        if (
            self.require_higher_timeframe_alignment
            and self.min_higher_timeframe_alignment <= 0
        ):
            raise ValueError(
                "min_higher_timeframe_alignment must be > 0 when "
                "require_higher_timeframe_alignment=True"
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
    """
    Normalized view of analytics.price_action.trend.TrendState.

    This view mirrors the current analytics contract and keeps the strategy
    independent from whether the input came as a dataclass, direct dict,
    {state: ...} EventBus payload, or PriceActionCompositeState child.
    """

    exchange: str | None = None
    market_type: str | None = None
    symbol: str | None = None
    exchange_symbol: str | None = None
    timeframe: str | None = None
    key: tuple[str, str, str, str] | None = None

    last_price: float | None = None
    last_update: datetime | None = None

    internal: dict[str, Any] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)

    internal_external_alignment: float = 0.0
    higher_timeframe_alignment: float = 0.0
    overall_trend_score: float = 0.0

    last_signal: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_feature: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class TrendContinuationStrategy(PriceActionStrategyBase):
    """
    Strategy wrapper around analytics.price_action.trend.

    It is aligned with the current TrendAnalyzer / TrendState contract:
    - consumes TrendState from PriceActionCompositeState or direct module feature;
    - validates futures scope through PriceActionStrategyBase;
    - uses internal/external layers, direction, regime, strength, confidence,
      momentum_direction_score, slope_direction_score, structure_score,
      continuation_probability, reversal_risk, exhaustion_score, pullback_depth,
      consolidation_score, is_accelerating, is_exhausted, in_pullback,
      is_aligned_with_structure and last_signal;
    - uses global state fields internal_external_alignment,
      higher_timeframe_alignment and overall_trend_score;
    - emits StrategySignal metadata that preserves the analytics source contract.
    """

    analytics_module_name = "trend"

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

    @property
    def _p(self) -> TrendContinuationStrategyParams:
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
                module_name=self.analytics_module_name,
                analytics_payload=trend.raw,
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
                    metadata={
                        "trend_source_feature": trend.source_feature,
                        "internal_external_alignment": trend.internal_external_alignment,
                        "higher_timeframe_alignment": trend.higher_timeframe_alignment,
                        "overall_trend_score": trend.overall_trend_score,
                    },
                )

            primary_layer = self._select_primary_layer(trend)
            primary_layer_name = (
                "external" if self._p.prefer_external_layer else "internal"
            )
            secondary_layer = (
                trend.internal if self._p.prefer_external_layer else trend.external
            )
            secondary_layer_name = (
                "internal" if self._p.prefer_external_layer else "external"
            )

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
                metadata={
                    "analytics_module": self.analytics_module_name,
                    "analytics_source_feature": trend.source_feature,
                },
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
    # Extraction / normalization
    # ------------------------------------------------------------------

    def _extract_trend_snapshot(self, context: SignalContext) -> TrendContextView:
        payload = self._extract_price_action_module(
            context,
            self.analytics_module_name,
            aliases=(
                "trend",
                "price_action.trend",
                "analytics.price_action.trend",
            ),
            require_scope_match=True,
        )
        if payload:
            return self._normalize_trend_snapshot(payload)

        # Defensive fallback for deployments that still pass legacy context
        # shapes while base.py is already updated.
        candidates: list[Any] = [
            self._mapping_or_empty(getattr(context, "price_action", None)).get("trend"),
            self._get_context_feature(context, "price_action.trend"),
            self._get_context_feature(context, "trend"),
            self._get_context_feature(context, "analytics.price_action.trend"),
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

        state = self._normalize_state_payload(payload_mapping)
        if not state:
            return TrendContextView()

        internal = self._normalize_trend_layer(
            state.get("internal"),
            StructureLayer.INTERNAL,
        )
        external = self._normalize_trend_layer(
            state.get("external"),
            StructureLayer.EXTERNAL,
        )

        metadata = dict(self._mapping_or_empty(state.get("metadata")))
        scope = self._extract_analytics_scope(state)
        key_values = scope.get("key") if isinstance(scope.get("key"), list) else []
        key_tuple: tuple[str, str, str, str] | None = None
        if len(key_values) == 4:
            key_tuple = (
                str(key_values[0]),
                str(key_values[1]),
                str(key_values[2]),
                str(key_values[3]),
            )

        last_signal = self._extract_last_signal(internal, external)

        return TrendContextView(
            exchange=scope.get("exchange"),
            market_type=scope.get("market_type"),
            symbol=first_non_empty(state.get("symbol"), scope.get("symbol")),
            exchange_symbol=first_non_empty(
                state.get("exchange_symbol"),
                scope.get("exchange_symbol"),
            ),
            timeframe=first_non_empty(state.get("timeframe"), scope.get("timeframe")),
            key=key_tuple,
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
                    state.get("updated_at"),
                    payload_mapping.get("last_update"),
                    metadata.get("last_update"),
                    metadata.get("updated_at"),
                )
            ),
            internal=internal,
            external=external,
            internal_external_alignment=clamp(
                safe_float(state.get("internal_external_alignment"), 0.0),
                0.0,
                1.0,
            ),
            higher_timeframe_alignment=clamp(
                safe_float(state.get("higher_timeframe_alignment"), 0.0),
                0.0,
                1.0,
            ),
            overall_trend_score=clamp(
                safe_float(state.get("overall_trend_score"), 0.0),
                0.0,
                1.0,
            ),
            last_signal=last_signal,
            metadata=metadata,
            source_feature=state.get("_source_feature"),
            raw=dict(state),
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
            "momentum_direction_score": clamp(
                safe_float(payload_mapping.get("momentum_direction_score"), 0.0),
                -1.0,
                1.0,
            ),
            "slope_direction_score": clamp(
                safe_float(payload_mapping.get("slope_direction_score"), 0.0),
                -1.0,
                1.0,
            ),
            "structure_score": clamp(
                safe_float(payload_mapping.get("structure_score"), 0.0),
                0.0,
                1.0,
            ),
            "continuation_probability": clamp(
                safe_float(payload_mapping.get("continuation_probability"), 0.0),
                0.0,
                1.0,
            ),
            "reversal_risk": clamp(
                safe_float(payload_mapping.get("reversal_risk"), 0.0),
                0.0,
                1.0,
            ),
            "exhaustion_score": clamp(
                safe_float(payload_mapping.get("exhaustion_score"), 0.0),
                0.0,
                1.0,
            ),
            "pullback_depth": clamp(
                safe_float(payload_mapping.get("pullback_depth"), 0.0),
                0.0,
                1.0,
            ),
            "consolidation_score": clamp(
                safe_float(payload_mapping.get("consolidation_score"), 0.0),
                0.0,
                1.0,
            ),
            "is_accelerating": safe_bool(payload_mapping.get("is_accelerating"), False),
            "is_exhausted": safe_bool(payload_mapping.get("is_exhausted"), False),
            "in_pullback": safe_bool(payload_mapping.get("in_pullback"), False),
            "is_aligned_with_structure": safe_bool(
                payload_mapping.get("is_aligned_with_structure"),
                False,
            ),
            "last_signal": self._normalize_trend_signal(last_signal, default_layer),
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
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

    def _extract_last_signal(
        self,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        epoch = datetime.min.replace(tzinfo=timezone.utc)
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

        def _sort_key(item: Mapping[str, Any]) -> datetime:
            ts = item.get("timestamp")
            if ts is None:
                return epoch
            if isinstance(ts, datetime) and ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            if isinstance(ts, datetime):
                return ts.astimezone(timezone.utc)
            return epoch

        candidates.sort(key=_sort_key, reverse=True)
        return dict(candidates[0])

    # ------------------------------------------------------------------
    # Direction / eligibility
    # ------------------------------------------------------------------

    def _select_primary_layer(self, trend: TrendContextView) -> dict[str, Any]:
        return trend.external if self._p.prefer_external_layer else trend.internal

    def _select_secondary_layer(self, trend: TrendContextView) -> dict[str, Any]:
        return trend.internal if self._p.prefer_external_layer else trend.external

    def _resolve_side(
        self,
        context: SignalContext,
        trend: TrendContextView,
    ) -> SignalSide:
        primary = self._select_primary_layer(trend)
        secondary = self._select_secondary_layer(trend)

        primary_side = self._trend_direction_to_side(
            primary.get("direction", TrendDirection.UNKNOWN)
        )
        secondary_side = self._trend_direction_to_side(
            secondary.get("direction", TrendDirection.UNKNOWN)
        )

        if primary_side != SignalSide.UNKNOWN:
            if not self._trend_state_eligible(trend):
                return SignalSide.UNKNOWN
            if not self._layer_eligible(primary, side=primary_side, primary=True):
                return SignalSide.UNKNOWN
            if not self._cross_layer_confirmation_ok(
                primary_side=primary_side,
                secondary_side=secondary_side,
                secondary_layer=secondary,
            ):
                return SignalSide.UNKNOWN
            if not self._side_regime_allowed(primary_side, primary):
                return SignalSide.UNKNOWN
            return primary_side

        if not self._p.allow_cross_layer_fallback or secondary_side == SignalSide.UNKNOWN:
            return SignalSide.UNKNOWN

        if not self._trend_state_eligible(trend, allow_weak_global=True):
            return SignalSide.UNKNOWN
        if not self._layer_eligible(secondary, side=secondary_side, primary=False):
            return SignalSide.UNKNOWN
        if not self._side_regime_allowed(secondary_side, secondary):
            return SignalSide.UNKNOWN
        return secondary_side

    def _trend_state_eligible(
        self,
        trend: TrendContextView,
        *,
        allow_weak_global: bool = False,
    ) -> bool:
        if trend.internal_external_alignment < self._p.min_internal_external_alignment:
            if self._p.require_direction_alignment and not allow_weak_global:
                return False

        if trend.overall_trend_score < self._p.min_overall_trend_score:
            if not allow_weak_global:
                return False

        if self._p.require_higher_timeframe_alignment:
            if trend.higher_timeframe_alignment < self._p.min_higher_timeframe_alignment:
                return False

        return True

    def _cross_layer_confirmation_ok(
        self,
        *,
        primary_side: SignalSide,
        secondary_side: SignalSide,
        secondary_layer: Mapping[str, Any],
    ) -> bool:
        if self._p.require_direction_alignment:
            if secondary_side not in {SignalSide.UNKNOWN, primary_side}:
                return False

        if not self._p.require_internal_confirmation:
            return True

        if secondary_side == SignalSide.UNKNOWN:
            return False

        if secondary_side != primary_side:
            return False

        return self._layer_confirmation_ok(secondary_layer, side=primary_side)

    def _layer_eligible(
        self,
        layer: Mapping[str, Any],
        *,
        side: SignalSide,
        primary: bool,
    ) -> bool:
        if not layer:
            return False

        confidence = clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0)
        strength = clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0)
        continuation_probability = clamp(
            safe_float(layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )
        reversal_risk = clamp(safe_float(layer.get("reversal_risk"), 0.0), 0.0, 1.0)
        exhaustion_score = clamp(safe_float(layer.get("exhaustion_score"), 0.0), 0.0, 1.0)
        consolidation_score = clamp(
            safe_float(layer.get("consolidation_score"), 0.0),
            0.0,
            1.0,
        )
        pullback_depth = clamp(safe_float(layer.get("pullback_depth"), 0.0), 0.0, 1.0)

        primary_multiplier = 1.0 if primary else 0.85
        if confidence < self._p.min_layer_confidence * primary_multiplier:
            return False
        if strength < self._p.min_trend_strength * primary_multiplier:
            return False
        if continuation_probability < self._p.min_continuation_probability * primary_multiplier:
            return False

        if self._p.require_positive_momentum:
            if (
                self._directional_score(layer.get("momentum_direction_score"), side)
                < self._p.min_directional_momentum
            ):
                return False

        if self._p.require_positive_slope:
            if (
                self._directional_score(layer.get("slope_direction_score"), side)
                < self._p.min_directional_slope
            ):
                return False

        if self._p.require_structure_alignment:
            aligned = safe_bool(layer.get("is_aligned_with_structure"), False)
            structure_score = clamp(safe_float(layer.get("structure_score"), 0.0), 0.0, 1.0)
            if not aligned and structure_score < self._p.min_structure_score:
                return False

        if self._p.block_high_reversal_risk and reversal_risk > self._p.max_reversal_risk:
            return False

        if self._p.block_exhausted_trend:
            exhausted = safe_bool(layer.get("is_exhausted"), False)
            if exhausted or exhaustion_score > self._p.max_exhaustion_score:
                return False

        if consolidation_score > self._p.max_consolidation_score:
            return False

        if safe_bool(layer.get("in_pullback"), False):
            if not self._p.allow_pullback_entries:
                return False
            if pullback_depth > self._p.max_pullback_depth:
                return False

        return True

    def _layer_confirmation_ok(
        self,
        layer: Mapping[str, Any],
        *,
        side: SignalSide,
    ) -> bool:
        if not layer:
            return False

        confidence_ok = (
            clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0)
            >= self._p.min_layer_confidence * 0.85
        )
        strength_ok = (
            clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0)
            >= self._p.min_trend_strength * 0.85
        )
        continuation_ok = (
            clamp(safe_float(layer.get("continuation_probability"), 0.0), 0.0, 1.0)
            >= self._p.min_continuation_probability * 0.85
        )
        momentum_ok = (
            self._directional_score(layer.get("momentum_direction_score"), side)
            >= max(0.0, self._p.min_directional_momentum * 0.75)
        )
        return confidence_ok and strength_ok and continuation_ok and momentum_ok

    def _side_regime_allowed(self, side: SignalSide, layer: Mapping[str, Any]) -> bool:
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

        score += self._p.primary_confidence_weight * clamp(
            safe_float(primary_layer.get("confidence"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.primary_strength_weight * clamp(
            safe_float(primary_layer.get("strength"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.continuation_probability_weight * clamp(
            safe_float(primary_layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )

        score += self._p.momentum_direction_weight * self._directional_score(
            primary_layer.get("momentum_direction_score"),
            side,
        )
        score += self._p.slope_direction_weight * self._directional_score(
            primary_layer.get("slope_direction_score"),
            side,
        )
        score += self._p.structure_alignment_weight * self._structure_alignment_score(
            primary_layer,
            side,
        )

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            score += self._p.internal_confirmation_weight * self._layer_confirmation_score(
                secondary_layer,
                side=side,
            )

        score += self._p.global_alignment_weight * self._global_alignment_score(trend)
        score += self._p.higher_timeframe_alignment_weight * trend.higher_timeframe_alignment
        score += self._p.regime_alignment_weight * self._regime_alignment_score(
            context=context,
            side=side,
        )

        if self._p.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            pullback_depth = clamp(safe_float(primary_layer.get("pullback_depth"), 0.0), 0.0, 1.0)
            if pullback_depth <= self._p.max_pullback_depth:
                score += self._p.pullback_bonus_weight * (1.0 - pullback_depth)

        if safe_bool(primary_layer.get("is_accelerating"), False):
            score += self._p.acceleration_bonus_weight

        reversal_risk = clamp(safe_float(primary_layer.get("reversal_risk"), 0.0), 0.0, 1.0)
        exhaustion_score = clamp(safe_float(primary_layer.get("exhaustion_score"), 0.0), 0.0, 1.0)
        consolidation_score = clamp(safe_float(primary_layer.get("consolidation_score"), 0.0), 0.0, 1.0)

        penalty = (reversal_risk * 0.10) + (exhaustion_score * 0.08) + (consolidation_score * 0.05)
        return clamp(score - penalty, 0.0, 1.0)

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
            self._directional_score(primary_layer.get("momentum_direction_score"), side),
            self._directional_score(primary_layer.get("slope_direction_score"), side),
            self._structure_alignment_score(primary_layer, side),
            self._global_alignment_score(trend),
        ]

        if trend.higher_timeframe_alignment > 0:
            components.append(trend.higher_timeframe_alignment)

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            components.append(self._layer_confirmation_score(secondary_layer, side=side))

        components.append(self._regime_alignment_score(context=context, side=side))

        confidence = sum(components) / len(components)

        reversal_risk = clamp(safe_float(primary_layer.get("reversal_risk"), 0.0), 0.0, 1.0)
        exhaustion_score = clamp(safe_float(primary_layer.get("exhaustion_score"), 0.0), 0.0, 1.0)
        consolidation_score = clamp(safe_float(primary_layer.get("consolidation_score"), 0.0), 0.0, 1.0)

        confidence *= 1.0 - reversal_risk * 0.30
        confidence *= 1.0 - exhaustion_score * 0.22
        confidence *= 1.0 - consolidation_score * 0.12

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

        if (
            self._directional_score(primary_layer.get("momentum_direction_score"), side)
            >= self._p.min_directional_momentum
        ):
            reasons.append("momentum_aligned")

        if (
            self._directional_score(primary_layer.get("slope_direction_score"), side)
            >= self._p.min_directional_slope
        ):
            reasons.append("slope_aligned")

        if safe_bool(primary_layer.get("is_aligned_with_structure"), False):
            reasons.append("structure_aligned")

        if trend.internal_external_alignment >= self._p.min_internal_external_alignment:
            reasons.append("internal_external_aligned")

        if trend.higher_timeframe_alignment >= self._p.min_higher_timeframe_alignment:
            reasons.append("higher_timeframe_aligned")

        if trend.overall_trend_score >= self._p.min_overall_trend_score:
            reasons.append("overall_trend_score_valid")

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            reasons.append(f"{secondary_layer_name}_confirmation")

        if safe_bool(primary_layer.get("is_accelerating"), False):
            reasons.append("trend_accelerating")

        if self._p.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            reasons.append("continuation_pullback_entry")

        last_signal = trend.last_signal
        if last_signal is not None and self._trend_direction_to_side(
            last_signal.get("direction", TrendDirection.UNKNOWN)
        ) == side:
            event_type = last_signal.get("event_type")
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
        secondary_layer = self._select_secondary_layer(trend)
        last_signal = trend.last_signal

        priority = self._resolve_priority(
            confidence=confidence,
            score=score,
            trend=trend,
            primary_layer=primary_layer,
            last_signal=last_signal,
        )

        last_event_type = last_signal.get("event_type") if last_signal is not None else None
        last_layer = last_signal.get("layer") if last_signal is not None else None

        analytics_metadata = self._build_analytics_source_metadata(
            module_name=self.analytics_module_name,
            payload=trend.raw,
            selected_entity=last_signal or primary_layer,
            extra={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.trend",
                "trend_timeframe": trend.timeframe,
                "trend_last_update": trend.last_update.isoformat() if trend.last_update else None,
                "trend_last_price": trend.last_price,
                "primary_layer": enum_value(primary_layer.get("layer")),
                "primary_direction": enum_value(primary_layer.get("direction")),
                "primary_regime": enum_value(primary_layer.get("regime")),
                "primary_strength": safe_float(primary_layer.get("strength"), 0.0),
                "primary_confidence": safe_float(primary_layer.get("confidence"), 0.0),
                "momentum_direction_score": safe_float(primary_layer.get("momentum_direction_score"), 0.0),
                "slope_direction_score": safe_float(primary_layer.get("slope_direction_score"), 0.0),
                "structure_score": safe_float(primary_layer.get("structure_score"), 0.0),
                "is_aligned_with_structure": safe_bool(primary_layer.get("is_aligned_with_structure"), False),
                "continuation_probability": safe_float(primary_layer.get("continuation_probability"), 0.0),
                "reversal_risk": safe_float(primary_layer.get("reversal_risk"), 0.0),
                "exhaustion_score": safe_float(primary_layer.get("exhaustion_score"), 0.0),
                "consolidation_score": safe_float(primary_layer.get("consolidation_score"), 0.0),
                "pullback_depth": safe_float(primary_layer.get("pullback_depth"), 0.0),
                "in_pullback": safe_bool(primary_layer.get("in_pullback"), False),
                "is_accelerating": safe_bool(primary_layer.get("is_accelerating"), False),
                "is_exhausted": safe_bool(primary_layer.get("is_exhausted"), False),
                "secondary_layer": enum_value(secondary_layer.get("layer")),
                "secondary_direction": enum_value(secondary_layer.get("direction")),
                "secondary_confidence": safe_float(secondary_layer.get("confidence"), 0.0),
                "internal_external_alignment": trend.internal_external_alignment,
                "higher_timeframe_alignment": trend.higher_timeframe_alignment,
                "overall_trend_score": trend.overall_trend_score,
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
            metadata=analytics_metadata,
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("analytics.price_action")
        signal.add_source_feature("analytics.price_action.trend")
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
        score: float,
        trend: TrendContextView,
        primary_layer: Mapping[str, Any],
        last_signal: Mapping[str, Any] | None,
    ) -> SignalPriority:
        continuation_probability = clamp(
            safe_float(primary_layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )

        if (
            confidence >= 0.85
            and score >= 0.80
            and continuation_probability >= 0.80
            and trend.overall_trend_score >= 0.70
        ):
            return SignalPriority.HIGH

        if (
            last_signal is not None
            and last_signal.get("event_type") in {
                TrendEventType.TREND_CONTINUATION,
                TrendEventType.TREND_ACCELERATION,
                TrendEventType.TREND_ALIGNMENT,
            }
            and confidence >= 0.70
            and score >= 0.65
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
    # Scoring helpers
    # ------------------------------------------------------------------

    def _directional_score(self, value: Any, side: SignalSide) -> float:
        signed = clamp(safe_float(value, 0.0), -1.0, 1.0)
        if side == SignalSide.LONG:
            return clamp(max(0.0, signed), 0.0, 1.0)
        if side == SignalSide.SHORT:
            return clamp(max(0.0, -signed), 0.0, 1.0)
        return 0.0

    def _structure_alignment_score(
        self,
        layer: Mapping[str, Any],
        side: SignalSide,
    ) -> float:
        structure_score = clamp(safe_float(layer.get("structure_score"), 0.0), 0.0, 1.0)
        aligned = safe_bool(layer.get("is_aligned_with_structure"), False)
        if aligned:
            return max(structure_score, 0.75)
        return structure_score

    def _layer_confirmation_score(
        self,
        layer: Mapping[str, Any],
        *,
        side: SignalSide,
    ) -> float:
        components = [
            clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0),
            clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0),
            clamp(safe_float(layer.get("continuation_probability"), 0.0), 0.0, 1.0),
            self._directional_score(layer.get("momentum_direction_score"), side),
        ]
        return clamp(sum(components) / len(components), 0.0, 1.0)

    def _global_alignment_score(self, trend: TrendContextView) -> float:
        components = [
            trend.internal_external_alignment,
            trend.overall_trend_score,
        ]
        if trend.higher_timeframe_alignment > 0:
            components.append(trend.higher_timeframe_alignment)
        return clamp(sum(components) / len(components), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Enum parsing helpers
    # ------------------------------------------------------------------

    def _trend_direction_to_side(self, direction: Any) -> SignalSide:
        parsed = self._parse_trend_direction(direction)
        if parsed == TrendDirection.BULLISH:
            return SignalSide.LONG
        if parsed == TrendDirection.BEARISH:
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
        if raw in mapping:
            return mapping[raw]
        try:
            return TrendDirection(raw)
        except Exception:
            return TrendDirection.UNKNOWN

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