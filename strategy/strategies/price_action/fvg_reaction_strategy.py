from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    StructureLayer,
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
    safe_float,
)


@dataclass(slots=True)
class FVGReactionStrategyParams:
    """
    Local params for FVGReactionStrategy.

    Runtime gates such as enabled/symbols/timeframes/min_score/min_confidence
    stay in StrategyConfig / StrategyDefinitionConfig.runtime. These params
    define how this strategy consumes analytics.price_action.fair_value_gap.
    """

    strategy_name: str = "fvg_reaction_strategy"

    prefer_external_layer: bool = True
    require_recent_event: bool = True
    require_directional_gap: bool = True
    require_respected_or_retested: bool = False

    allow_active_gap_proximity_entry: bool = True
    allow_fill_started_reaction: bool = True
    allow_partial_fill_reaction: bool = True
    allow_respected_reaction: bool = True
    allow_retested_reaction: bool = True
    allow_created_gap_continuation: bool = False
    allow_merged_gap_entry: bool = False

    block_invalidated_gaps: bool = True
    block_filled_gaps: bool = True
    block_counter_regime: bool = False
    block_excessive_layer_fill_activity: bool = True

    min_gap_strength: float = 0.45
    min_event_confidence: float = 0.45
    min_gap_fill_for_reaction: float = 0.05
    max_gap_fill_for_entry: float = 0.90
    max_distance_to_mid_pct: float = 0.0035
    max_recent_fill_activity: float = 0.95

    primary_gap_weight: float = 0.26
    event_confidence_weight: float = 0.20
    status_quality_weight: float = 0.14
    proximity_weight: float = 0.12
    fill_quality_weight: float = 0.10
    secondary_layer_alignment_weight: float = 0.08
    regime_alignment_weight: float = 0.05
    retest_bonus_weight: float = 0.03
    respect_bonus_weight: float = 0.02

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.fvg_reaction.signal"

    freshness_feature_names: tuple[str, ...] = (
        "analytics.price_action",
        "analytics.price_action.fair_value_gap",
        "price_action.fair_value_gap",
        "price_action.fvg",
        "fair_value_gap",
        "fvg",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_gap_strength",
                "min_event_confidence",
                "min_gap_fill_for_reaction",
                "max_gap_fill_for_entry",
                "max_distance_to_mid_pct",
                "max_recent_fill_activity",
                "primary_gap_weight",
                "event_confidence_weight",
                "status_quality_weight",
                "proximity_weight",
                "fill_quality_weight",
                "secondary_layer_alignment_weight",
                "regime_alignment_weight",
                "retest_bonus_weight",
                "respect_bonus_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "FVGReactionStrategyParams":
        return apply_definition_metadata(
            params=cls(),
            definition=definition,
        )


@dataclass(slots=True)
class FVGReactionContext:
    """
    Normalized view of analytics.price_action.fair_value_gap.FairValueGapState.
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

    last_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_feature: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class FVGReactionStrategy(PriceActionStrategyBase):
    """
    Strategy wrapper around analytics.price_action.fair_value_gap.

    Aligned with the current FairValueGapAnalyzer / FairValueGapState contract:
    - consumes FairValueGapState from PriceActionCompositeState or direct module feature;
    - validates futures scope through PriceActionStrategyBase;
    - supports internal/external LayerFVGState;
    - uses nearest/strongest bullish/bearish gaps;
    - uses FVG lifecycle events: fill started, partially filled, respected,
      retested, filled, invalidated, merged;
    - preserves analytics source metadata in StrategySignal.metadata.
    """

    analytics_module_name = "fair_value_gap"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
        strategy_name: str = "fvg_reaction_strategy",
    ) -> None:
        super().__init__(
            config=config,
            strategy_name=strategy_name,
            params_cls=FVGReactionStrategyParams,
            event_bus=event_bus,
            logger=logger,
        )

    @property
    def _p(self) -> FVGReactionStrategyParams:
        return cast(FVGReactionStrategyParams, self.params)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked

            fvg = self._extract_fvg_snapshot(context)
            if fvg.symbol is None and not fvg.external and not fvg.internal:
                return self._rejected_evaluation(
                    context=context,
                    reason="fvg_snapshot_missing",
                )

            freshness_filter = self._build_freshness_filter(
                context=context,
                filter_name="fvg_freshness",
                module_name=self.analytics_module_name,
                analytics_payload=fvg.raw,
            )
            if freshness_filter is not None and freshness_filter.blocked:
                return self._rejected_evaluation(
                    context=context,
                    reason="stale_fvg_feature",
                )

            primary_layer = self._select_primary_layer(fvg)
            secondary_layer = self._select_secondary_layer(fvg)

            selected_gap = self._select_reaction_gap(
                context=context,
                fvg=fvg,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )
            if selected_gap is None:
                return self._rejected_evaluation(
                    context=context,
                    reason="no_reactable_fvg_found",
                    metadata={
                        "analytics_module": self.analytics_module_name,
                        "analytics_source_feature": fvg.source_feature,
                        "last_fvg_event_type": (
                            enum_value(fvg.last_event.get("event_type"))
                            if fvg.last_event is not None
                            else None
                        ),
                    },
                )

            side = self._resolve_side(selected_gap)
            if side == SignalSide.UNKNOWN:
                return self._rejected_evaluation(
                    context=context,
                    reason="fvg_direction_not_tradeable",
                )

            if not self._gap_is_tradeable(selected_gap, primary_layer):
                return self._rejected_evaluation(
                    context=context,
                    reason="selected_fvg_not_tradeable",
                    metadata={
                        "gap_id": selected_gap.get("gap_id"),
                        "gap_status": enum_value(selected_gap.get("status")),
                        "gap_fill_percentage": selected_gap.get("fill_percentage"),
                        "gap_strength": selected_gap.get("strength"),
                    },
                )

            score = self._compute_score(
                context=context,
                fvg=fvg,
                selected_gap=selected_gap,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                last_event=fvg.last_event,
                side=side,
            )
            confidence = self._compute_confidence(
                context=context,
                fvg=fvg,
                selected_gap=selected_gap,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                last_event=fvg.last_event,
                side=side,
            )

            reasons = self._build_reasons(
                fvg=fvg,
                selected_gap=selected_gap,
                side=side,
            )

            signal = self._build_signal(
                context=context,
                fvg=fvg,
                selected_gap=selected_gap,
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
                    "analytics_source_feature": fvg.source_feature,
                },
            )

        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self._logger.exception(
                "Failed to evaluate FVG reaction strategy | strategy=%s symbol=%s",
                self.name,
                getattr(context, "symbol", None),
            )
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate FVG reaction for {context.symbol}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction / normalization
    # ------------------------------------------------------------------

    def _extract_fvg_snapshot(self, context: SignalContext) -> FVGReactionContext:
        payload = self._extract_price_action_module(
            context,
            self.analytics_module_name,
            aliases=(
                "fair_value_gap",
                "fvg",
                "price_action.fair_value_gap",
                "price_action.fvg",
                "analytics.price_action.fair_value_gap",
            ),
            require_scope_match=True,
        )
        if payload:
            return self._normalize_fvg_snapshot(payload)

        candidates: list[Any] = [
            self._mapping_or_empty(getattr(context, "price_action", None)).get("fair_value_gap"),
            self._mapping_or_empty(getattr(context, "price_action", None)).get("fvg"),
            self._get_context_feature(context, "price_action.fair_value_gap"),
            self._get_context_feature(context, "price_action.fvg"),
            self._get_context_feature(context, "fair_value_gap"),
            self._get_context_feature(context, "fvg"),
            self._get_context_feature(context, "analytics.price_action.fair_value_gap"),
        ]

        for candidate in candidates:
            normalized = self._normalize_fvg_snapshot(candidate)
            if normalized.symbol is not None or normalized.external or normalized.internal:
                return normalized

        return FVGReactionContext()

    def _normalize_fvg_snapshot(self, payload: Any) -> FVGReactionContext:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return FVGReactionContext()

        state = self._normalize_state_payload(payload_mapping)
        if not state:
            return FVGReactionContext()

        internal = self._normalize_fvg_layer(
            state.get("internal"),
            StructureLayer.INTERNAL,
        )
        external = self._normalize_fvg_layer(
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

        last_event = self._extract_last_event(internal, external)

        return FVGReactionContext(
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
            last_event=last_event,
            metadata=metadata,
            source_feature=state.get("_source_feature"),
            raw=dict(state),
        )

    def _normalize_fvg_layer(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        return {
            "layer": self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            "total_gaps": int(safe_float(payload_mapping.get("total_gaps"), 0.0)),
            "active_gaps": int(safe_float(payload_mapping.get("active_gaps"), 0.0)),
            "partially_filled_gaps": int(
                safe_float(payload_mapping.get("partially_filled_gaps"), 0.0)
            ),
            "filled_gaps": int(safe_float(payload_mapping.get("filled_gaps"), 0.0)),
            "respected_gaps": int(safe_float(payload_mapping.get("respected_gaps"), 0.0)),
            "invalidated_gaps": int(
                safe_float(payload_mapping.get("invalidated_gaps"), 0.0)
            ),
            "nearest_bullish_gap": self._normalize_gap(
                payload_mapping.get("nearest_bullish_gap"),
                default_layer,
            ),
            "nearest_bearish_gap": self._normalize_gap(
                payload_mapping.get("nearest_bearish_gap"),
                default_layer,
            ),
            "strongest_bullish_gap": self._normalize_gap(
                payload_mapping.get("strongest_bullish_gap"),
                default_layer,
            ),
            "strongest_bearish_gap": self._normalize_gap(
                payload_mapping.get("strongest_bearish_gap"),
                default_layer,
            ),
            "recent_fill_activity": clamp(
                safe_float(payload_mapping.get("recent_fill_activity"), 0.0),
                0.0,
                1.0,
            ),
            "last_event": self._normalize_fvg_event(
                payload_mapping.get("last_event"),
                default_layer,
            ),
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_gap(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        source_candle_indices = []
        for index in list(payload_mapping.get("source_candle_indices", []) or []):
            try:
                source_candle_indices.append(int(index))
            except (TypeError, ValueError):
                continue

        return {
            "gap_id": payload_mapping.get("gap_id"),
            "exchange": payload_mapping.get("exchange"),
            "market_type": payload_mapping.get("market_type"),
            "symbol": payload_mapping.get("symbol"),
            "exchange_symbol": payload_mapping.get("exchange_symbol"),
            "timeframe": payload_mapping.get("timeframe"),
            "key": list(payload_mapping.get("key", []) or []),
            "layer": self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            "direction": self._parse_fvg_direction(payload_mapping.get("direction")),
            "upper_bound": safe_float(payload_mapping.get("upper_bound"), 0.0),
            "lower_bound": safe_float(payload_mapping.get("lower_bound"), 0.0),
            "mid_price": safe_float(payload_mapping.get("mid_price"), 0.0),
            "size": safe_float(payload_mapping.get("size"), 0.0),
            "size_pct": clamp(
                safe_float(payload_mapping.get("size_pct"), 0.0),
                0.0,
                1.0,
            ),
            "strength": clamp(
                safe_float(payload_mapping.get("strength"), 0.0),
                0.0,
                1.0,
            ),
            "status": self._parse_fvg_status(payload_mapping.get("status")),
            "fill_percentage": clamp(
                safe_float(payload_mapping.get("fill_percentage"), 0.0),
                0.0,
                1.0,
            ),
            "touch_count": int(safe_float(payload_mapping.get("touch_count"), 0.0)),
            "retest_count": int(safe_float(payload_mapping.get("retest_count"), 0.0)),
            "created_at": parse_datetime(payload_mapping.get("created_at")),
            "updated_at": parse_datetime(payload_mapping.get("updated_at")),
            "first_touch_at": parse_datetime(payload_mapping.get("first_touch_at")),
            "filled_at": parse_datetime(payload_mapping.get("filled_at")),
            "respected_at": parse_datetime(payload_mapping.get("respected_at")),
            "invalidated_at": parse_datetime(payload_mapping.get("invalidated_at")),
            "created_index": self._optional_int(payload_mapping.get("created_index")),
            "last_touch_index": self._optional_int(payload_mapping.get("last_touch_index")),
            "last_fill_index": self._optional_int(payload_mapping.get("last_fill_index")),
            "source_candle_indices": source_candle_indices,
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_fvg_event(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        return {
            "event_id": payload_mapping.get("event_id"),
            "exchange": payload_mapping.get("exchange"),
            "market_type": payload_mapping.get("market_type"),
            "symbol": payload_mapping.get("symbol"),
            "exchange_symbol": payload_mapping.get("exchange_symbol"),
            "timeframe": payload_mapping.get("timeframe"),
            "key": list(payload_mapping.get("key", []) or []),
            "event_type": self._parse_fvg_event_type(payload_mapping.get("event_type")),
            "timestamp": parse_datetime(payload_mapping.get("timestamp")),
            "layer": self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            "gap_id": payload_mapping.get("gap_id"),
            "direction": self._parse_fvg_direction(payload_mapping.get("direction")),
            "upper_bound": safe_float(payload_mapping.get("upper_bound"), 0.0),
            "lower_bound": safe_float(payload_mapping.get("lower_bound"), 0.0),
            "fill_percentage": clamp(
                safe_float(payload_mapping.get("fill_percentage"), 0.0),
                0.0,
                1.0,
            ),
            "confidence": clamp(
                safe_float(payload_mapping.get("confidence"), 0.0),
                0.0,
                1.0,
            ),
            "reference_price": (
                safe_float(payload_mapping.get("reference_price"), 0.0)
                if payload_mapping.get("reference_price") is not None
                else None
            ),
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _extract_last_event(
        self,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        epoch = datetime.min.replace(tzinfo=timezone.utc)

        candidates = [
            event
            for event in (
                external.get("last_event"),
                internal.get("last_event"),
            )
            if event is not None
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
    # Selection / side
    # ------------------------------------------------------------------

    def _select_primary_layer(self, fvg: FVGReactionContext) -> dict[str, Any]:
        return fvg.external if self._p.prefer_external_layer else fvg.internal

    def _select_secondary_layer(self, fvg: FVGReactionContext) -> dict[str, Any]:
        return fvg.internal if self._p.prefer_external_layer else fvg.external

    def _select_reaction_gap(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        last_event = fvg.last_event
        current_price = self._resolve_current_price(context=context, fvg=fvg)

        if last_event is not None and self._event_is_usable_for_entry(last_event):
            event_gap = self._gap_from_event(
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                event=last_event,
            )
            if event_gap is not None and self._gap_is_tradeable(event_gap, primary_layer):
                return event_gap

        if self._p.require_recent_event:
            return None

        candidate_gaps = self._candidate_gaps(primary_layer)

        best_gap: dict[str, Any] | None = None
        best_score = -1.0

        for gap in candidate_gaps:
            if gap is None:
                continue

            if not self._gap_is_tradeable(gap, primary_layer):
                continue

            if (
                not self._p.allow_active_gap_proximity_entry
                and gap.get("status") == FVGStatus.ACTIVE
            ):
                continue

            local_score = self._gap_selection_score(
                gap=gap,
                current_price=current_price,
            )
            if local_score > best_score:
                best_score = local_score
                best_gap = gap

        return best_gap

    def _candidate_gaps(self, layer: Mapping[str, Any]) -> list[dict[str, Any]]:
        candidates = [
            layer.get("strongest_bullish_gap"),
            layer.get("strongest_bearish_gap"),
            layer.get("nearest_bullish_gap"),
            layer.get("nearest_bearish_gap"),
        ]
        return [dict(gap) for gap in candidates if isinstance(gap, Mapping)]

    def _gap_from_event(
        self,
        *,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        gap_id = event.get("gap_id")
        if not gap_id:
            return None

        for layer in (primary_layer, secondary_layer):
            for key in (
                "strongest_bullish_gap",
                "strongest_bearish_gap",
                "nearest_bullish_gap",
                "nearest_bearish_gap",
            ):
                gap = layer.get(key)
                if isinstance(gap, Mapping) and gap.get("gap_id") == gap_id:
                    return dict(gap)

        return self._gap_stub_from_event(event)

    def _gap_stub_from_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        gap_id = event.get("gap_id")
        direction = self._parse_fvg_direction(event.get("direction"))
        if not gap_id or direction is None:
            return None

        upper_bound = safe_float(event.get("upper_bound"), 0.0)
        lower_bound = safe_float(event.get("lower_bound"), 0.0)
        mid_price = (upper_bound + lower_bound) / 2.0 if upper_bound and lower_bound else 0.0

        return {
            "gap_id": gap_id,
            "layer": self._parse_structure_layer(event.get("layer")) or StructureLayer.INTERNAL,
            "direction": direction,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "mid_price": mid_price,
            "size": max(upper_bound - lower_bound, 0.0),
            "size_pct": 0.0,
            "strength": clamp(safe_float(event.get("confidence"), 0.0), 0.0, 1.0),
            "status": self._status_from_event_type(event.get("event_type")),
            "fill_percentage": clamp(safe_float(event.get("fill_percentage"), 0.0), 0.0, 1.0),
            "touch_count": 0,
            "retest_count": 1 if event.get("event_type") == FVGEventType.FVG_RETESTED else 0,
            "created_at": None,
            "updated_at": event.get("timestamp"),
            "first_touch_at": None,
            "filled_at": None,
            "respected_at": event.get("timestamp")
            if event.get("event_type") == FVGEventType.FVG_RESPECTED
            else None,
            "invalidated_at": None,
            "metadata": dict(event.get("metadata", {}) or {}),
        }

    def _resolve_side(self, gap: Mapping[str, Any]) -> SignalSide:
        direction = gap.get("direction")
        if direction == FVGDirection.BULLISH:
            return SignalSide.LONG
        if direction == FVGDirection.BEARISH:
            return SignalSide.SHORT
        return SignalSide.UNKNOWN

    def _event_is_usable_for_entry(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("event_type")
        confidence = clamp(safe_float(event.get("confidence"), 0.0), 0.0, 1.0)

        if confidence < self._p.min_event_confidence:
            return False

        if event_type == FVGEventType.FVG_FILL_STARTED:
            return self._p.allow_fill_started_reaction

        if event_type == FVGEventType.FVG_PARTIALLY_FILLED:
            return self._p.allow_partial_fill_reaction

        if event_type == FVGEventType.FVG_RESPECTED:
            return self._p.allow_respected_reaction

        if event_type == FVGEventType.FVG_RETESTED:
            return self._p.allow_retested_reaction

        if event_type == FVGEventType.FVG_CREATED:
            return self._p.allow_created_gap_continuation

        if event_type == FVGEventType.FVG_MERGED:
            return self._p.allow_merged_gap_entry

        if event_type in {
            FVGEventType.FVG_FILLED,
            FVGEventType.FVG_INVALIDATED,
        }:
            return False

        return False

    def _gap_is_tradeable(
        self,
        gap: Mapping[str, Any],
        layer: Mapping[str, Any],
    ) -> bool:
        if not gap:
            return False

        if self._p.require_directional_gap and gap.get("direction") not in {
            FVGDirection.BULLISH,
            FVGDirection.BEARISH,
        }:
            return False

        if clamp(safe_float(gap.get("strength"), 0.0), 0.0, 1.0) < self._p.min_gap_strength:
            return False

        status = gap.get("status", FVGStatus.ACTIVE)

        if self._p.block_invalidated_gaps and status == FVGStatus.INVALIDATED:
            return False

        if self._p.block_filled_gaps and status == FVGStatus.FILLED:
            return False

        fill_percentage = clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0)

        if fill_percentage > self._p.max_gap_fill_for_entry:
            return False

        if status in {FVGStatus.PARTIALLY_FILLED, FVGStatus.RESPECTED}:
            if fill_percentage < self._p.min_gap_fill_for_reaction:
                if int(safe_float(gap.get("retest_count"), 0.0)) <= 0:
                    return False

        if self._p.require_respected_or_retested:
            respected = status == FVGStatus.RESPECTED
            retested = int(safe_float(gap.get("retest_count"), 0.0)) > 0
            if not (respected or retested):
                return False

        if self._p.block_excessive_layer_fill_activity:
            recent_fill_activity = clamp(
                safe_float(layer.get("recent_fill_activity"), 0.0),
                0.0,
                1.0,
            )
            if recent_fill_activity > self._p.max_recent_fill_activity:
                return False

        return True

    # ------------------------------------------------------------------
    # Score / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
        selected_gap: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        current_price = self._resolve_current_price(context=context, fvg=fvg)

        score = 0.0
        score += self._p.primary_gap_weight * clamp(
            safe_float(selected_gap.get("strength"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.event_confidence_weight * self._event_score(
            selected_gap=selected_gap,
            last_event=last_event,
        )
        score += self._p.status_quality_weight * self._status_quality_score(
            selected_gap.get("status")
        )
        score += self._p.proximity_weight * self._proximity_score(
            current_price=current_price,
            gap=selected_gap,
        )
        score += self._p.fill_quality_weight * self._fill_quality_score(selected_gap)
        score += self._p.secondary_layer_alignment_weight * self._secondary_layer_alignment_score(
            secondary_layer=secondary_layer,
            selected_gap=selected_gap,
        )
        score += self._p.regime_alignment_weight * self._regime_alignment_score(
            context=context,
            side=side,
        )

        if int(safe_float(selected_gap.get("retest_count"), 0.0)) > 0:
            score += self._p.retest_bonus_weight

        if selected_gap.get("status") == FVGStatus.RESPECTED:
            score += self._p.respect_bonus_weight

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
        selected_gap: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        current_price = self._resolve_current_price(context=context, fvg=fvg)

        components = [
            clamp(safe_float(selected_gap.get("strength"), 0.0), 0.0, 1.0),
            self._event_score(selected_gap=selected_gap, last_event=last_event),
            self._status_quality_score(selected_gap.get("status")),
            self._proximity_score(current_price=current_price, gap=selected_gap),
            self._fill_quality_score(selected_gap),
            self._secondary_layer_alignment_score(
                secondary_layer=secondary_layer,
                selected_gap=selected_gap,
            ),
            self._regime_alignment_score(context=context, side=side),
        ]

        layer_fill_activity = clamp(
            safe_float(primary_layer.get("recent_fill_activity"), 0.0),
            0.0,
            1.0,
        )
        penalty = 0.20 * max(0.0, layer_fill_activity - 0.65)

        return clamp((sum(components) / len(components)) - penalty, 0.0, 1.0)

    def _event_score(
        self,
        *,
        selected_gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> float:
        if last_event is None:
            return 0.35

        if last_event.get("gap_id") != selected_gap.get("gap_id"):
            return 0.30

        event_type = last_event.get("event_type")
        confidence = clamp(safe_float(last_event.get("confidence"), 0.0), 0.0, 1.0)

        multiplier = {
            FVGEventType.FVG_RESPECTED: 1.00,
            FVGEventType.FVG_RETESTED: 0.95,
            FVGEventType.FVG_PARTIALLY_FILLED: 0.82,
            FVGEventType.FVG_FILL_STARTED: 0.72,
            FVGEventType.FVG_CREATED: 0.55,
            FVGEventType.FVG_MERGED: 0.50,
            FVGEventType.FVG_FILLED: 0.10,
            FVGEventType.FVG_INVALIDATED: 0.0,
        }.get(event_type, 0.35)

        return clamp(confidence * multiplier, 0.0, 1.0)

    def _status_quality_score(self, status: Any) -> float:
        parsed = self._parse_fvg_status(status)

        if parsed == FVGStatus.RESPECTED:
            return 1.0
        if parsed == FVGStatus.PARTIALLY_FILLED:
            return 0.78
        if parsed == FVGStatus.ACTIVE:
            return 0.62
        if parsed == FVGStatus.FILLED:
            return 0.20
        if parsed == FVGStatus.INVALIDATED:
            return 0.0

        return 0.35

    def _fill_quality_score(self, gap: Mapping[str, Any]) -> float:
        fill = clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0)
        status = gap.get("status")

        if status == FVGStatus.RESPECTED:
            return max(0.80, 1.0 - abs(fill - 0.50))

        if status == FVGStatus.PARTIALLY_FILLED:
            if fill < self._p.min_gap_fill_for_reaction:
                return 0.25
            if fill <= self._p.max_gap_fill_for_entry:
                return clamp(1.0 - abs(fill - 0.45), 0.0, 1.0)

        if status == FVGStatus.ACTIVE:
            return 1.0 - fill

        if status == FVGStatus.FILLED:
            return 0.15

        if status == FVGStatus.INVALIDATED:
            return 0.0

        return 0.35

    def _proximity_score(
        self,
        *,
        current_price: float | None,
        gap: Mapping[str, Any],
    ) -> float:
        if current_price is None or current_price <= 0:
            return 0.35

        mid_price = safe_float(gap.get("mid_price"), 0.0)
        if mid_price <= 0:
            lower = safe_float(gap.get("lower_bound"), 0.0)
            upper = safe_float(gap.get("upper_bound"), 0.0)
            if lower > 0 and upper > 0:
                mid_price = (lower + upper) / 2.0

        if mid_price <= 0:
            return 0.35

        distance_pct = abs(current_price - mid_price) / mid_price
        if distance_pct >= self._p.max_distance_to_mid_pct:
            return 0.0

        return clamp(
            1.0 - (distance_pct / max(self._p.max_distance_to_mid_pct, 1e-9)),
            0.0,
            1.0,
        )

    def _secondary_layer_alignment_score(
        self,
        *,
        secondary_layer: Mapping[str, Any],
        selected_gap: Mapping[str, Any],
    ) -> float:
        if not secondary_layer:
            return 0.35

        direction = selected_gap.get("direction")

        if direction == FVGDirection.BULLISH:
            ref_gap = (
                secondary_layer.get("nearest_bullish_gap")
                or secondary_layer.get("strongest_bullish_gap")
            )
        elif direction == FVGDirection.BEARISH:
            ref_gap = (
                secondary_layer.get("nearest_bearish_gap")
                or secondary_layer.get("strongest_bearish_gap")
            )
        else:
            ref_gap = None

        if not isinstance(ref_gap, Mapping):
            return 0.35

        ref_status = ref_gap.get("status")
        return clamp(
            0.55 * clamp(safe_float(ref_gap.get("strength"), 0.0), 0.0, 1.0)
            + 0.30 * self._status_quality_score(ref_status)
            + 0.15 * self._fill_quality_score(ref_gap),
            0.0,
            1.0,
        )

    def _gap_selection_score(
        self,
        *,
        gap: Mapping[str, Any],
        current_price: float | None,
    ) -> float:
        score = 0.0
        score += 0.35 * clamp(safe_float(gap.get("strength"), 0.0), 0.0, 1.0)
        score += 0.22 * self._status_quality_score(gap.get("status"))
        score += 0.18 * self._fill_quality_score(gap)
        score += 0.15 * self._proximity_score(current_price=current_price, gap=gap)
        score += 0.10 * min(1.0, safe_float(gap.get("retest_count"), 0.0) / 3.0)
        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reasons / signal build
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        *,
        fvg: FVGReactionContext,
        selected_gap: Mapping[str, Any],
        side: SignalSide,
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.append("bullish_fvg_reaction")
        elif side == SignalSide.SHORT:
            reasons.append("bearish_fvg_reaction")

        layer = selected_gap.get("layer")
        status = selected_gap.get("status")
        direction = selected_gap.get("direction")

        reasons.append(f"gap_layer_{enum_value(layer)}")
        reasons.append(f"gap_direction_{enum_value(direction)}")
        reasons.append(f"gap_status_{enum_value(status)}")

        if status == FVGStatus.RESPECTED:
            reasons.append("fvg_respected")

        if status == FVGStatus.PARTIALLY_FILLED:
            reasons.append("fvg_partially_filled")

        if int(safe_float(selected_gap.get("retest_count"), 0.0)) > 0:
            reasons.append("fvg_retested")

        fill_pct = clamp(safe_float(selected_gap.get("fill_percentage"), 0.0), 0.0, 1.0)
        if fill_pct >= self._p.min_gap_fill_for_reaction:
            reasons.append("fvg_has_reaction_fill")

        if fvg.last_event is not None:
            event_type = fvg.last_event.get("event_type")
            if isinstance(event_type, FVGEventType):
                reasons.append(f"last_fvg_event_{event_type.value}")

            if fvg.last_event.get("gap_id") == selected_gap.get("gap_id"):
                reasons.append("last_event_matches_selected_gap")

        return reasons

    def _build_signal(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
        selected_gap: Mapping[str, Any],
        score: float,
        confidence: float,
        reasons: list[str],
        freshness_filter: FilterResult | None,
    ) -> StrategySignal:
        side = self._resolve_side(selected_gap)
        last_event = fvg.last_event

        gap_layer = selected_gap.get("layer")
        gap_direction = selected_gap.get("direction")
        gap_status = selected_gap.get("status")
        last_event_type = last_event.get("event_type") if last_event is not None else None

        analytics_metadata = self._build_analytics_source_metadata(
            module_name=self.analytics_module_name,
            payload=fvg.raw,
            selected_entity=selected_gap,
            extra={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.fair_value_gap",
                "fvg_timeframe": fvg.timeframe,
                "fvg_last_update": fvg.last_update.isoformat() if fvg.last_update else None,
                "fvg_last_price": fvg.last_price,
                "gap_id": selected_gap.get("gap_id"),
                "gap_layer": enum_value(gap_layer),
                "gap_direction": enum_value(gap_direction),
                "gap_status": enum_value(gap_status),
                "gap_strength": safe_float(selected_gap.get("strength"), 0.0),
                "gap_fill_percentage": safe_float(selected_gap.get("fill_percentage"), 0.0),
                "gap_mid_price": safe_float(selected_gap.get("mid_price"), 0.0),
                "gap_upper_bound": safe_float(selected_gap.get("upper_bound"), 0.0),
                "gap_lower_bound": safe_float(selected_gap.get("lower_bound"), 0.0),
                "gap_size": safe_float(selected_gap.get("size"), 0.0),
                "gap_size_pct": safe_float(selected_gap.get("size_pct"), 0.0),
                "gap_touch_count": int(safe_float(selected_gap.get("touch_count"), 0.0)),
                "gap_retest_count": int(safe_float(selected_gap.get("retest_count"), 0.0)),
                "gap_created_at": (
                    selected_gap.get("created_at").isoformat()
                    if isinstance(selected_gap.get("created_at"), datetime)
                    else None
                ),
                "gap_updated_at": (
                    selected_gap.get("updated_at").isoformat()
                    if isinstance(selected_gap.get("updated_at"), datetime)
                    else None
                ),
                "gap_first_touch_at": (
                    selected_gap.get("first_touch_at").isoformat()
                    if isinstance(selected_gap.get("first_touch_at"), datetime)
                    else None
                ),
                "gap_filled_at": (
                    selected_gap.get("filled_at").isoformat()
                    if isinstance(selected_gap.get("filled_at"), datetime)
                    else None
                ),
                "gap_respected_at": (
                    selected_gap.get("respected_at").isoformat()
                    if isinstance(selected_gap.get("respected_at"), datetime)
                    else None
                ),
                "gap_invalidated_at": (
                    selected_gap.get("invalidated_at").isoformat()
                    if isinstance(selected_gap.get("invalidated_at"), datetime)
                    else None
                ),
                "gap_created_index": selected_gap.get("created_index"),
                "gap_last_touch_index": selected_gap.get("last_touch_index"),
                "gap_last_fill_index": selected_gap.get("last_fill_index"),
                "gap_source_candle_indices": list(
                    selected_gap.get("source_candle_indices", []) or []
                ),
                "last_fvg_event_id": (
                    last_event.get("event_id")
                    if last_event is not None
                    else None
                ),
                "last_fvg_event_type": enum_value(last_event_type),
                "last_fvg_event_confidence": (
                    safe_float(last_event.get("confidence"), 0.0)
                    if last_event is not None
                    else None
                ),
                "last_fvg_event_reference_price": (
                    last_event.get("reference_price")
                    if last_event is not None
                    else None
                ),
                "last_fvg_event_matches_gap": (
                    bool(last_event and last_event.get("gap_id") == selected_gap.get("gap_id"))
                ),
                "primary_layer": "external" if self._p.prefer_external_layer else "internal",
            },
        )

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.name,
            category=StrategyCategory.PRICE_ACTION,
            timeframe=context.timeframe,
            setup_type=self._resolve_setup_type(selected_gap, last_event),
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=self._resolve_trigger_type(selected_gap, last_event),
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._resolve_priority(
                selected_gap=selected_gap,
                last_event=last_event,
                confidence=confidence,
                score=score,
            ),
            regime=self._resolve_market_regime(context),
            metadata=analytics_metadata,
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("analytics.price_action")
        signal.add_source_feature("analytics.price_action.fair_value_gap")
        signal.add_source_feature("price_action.fair_value_gap")
        signal.add_source_feature("price_action.fvg")

        if freshness_filter is not None:
            signal.add_filter_result(freshness_filter)

        regime_filter = self._build_regime_filter(context=context, side=side)
        if regime_filter is not None:
            signal.add_filter_result(regime_filter)

        signal.validate()
        return signal

    def _resolve_setup_type(
        self,
        gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> SetupType:
        event_type = last_event.get("event_type") if last_event is not None else None

        if event_type in {
            FVGEventType.FVG_RESPECTED,
            FVGEventType.FVG_RETESTED,
        }:
            return SetupType.REVERSAL

        if event_type in {
            FVGEventType.FVG_FILL_STARTED,
            FVGEventType.FVG_PARTIALLY_FILLED,
        }:
            return SetupType.MEAN_REVERSION

        if gap.get("status") == FVGStatus.PARTIALLY_FILLED:
            return SetupType.MEAN_REVERSION

        if gap.get("status") == FVGStatus.RESPECTED:
            return SetupType.REVERSAL

        return SetupType.CONTINUATION

    def _resolve_trigger_type(
        self,
        gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> TriggerType:
        event_type = last_event.get("event_type") if last_event is not None else None

        if event_type in {
            FVGEventType.FVG_RESPECTED,
            FVGEventType.FVG_RETESTED,
        }:
            return TriggerType.PRIMARY

        if event_type in {
            FVGEventType.FVG_FILL_STARTED,
            FVGEventType.FVG_PARTIALLY_FILLED,
        }:
            return TriggerType.CONFIRMATION

        if gap.get("status") == FVGStatus.ACTIVE:
            return TriggerType.DERIVED

        return TriggerType.CONFIRMATION

    def _resolve_priority(
        self,
        *,
        selected_gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        confidence: float,
        score: float,
    ) -> SignalPriority:
        event_type = last_event.get("event_type") if last_event is not None else None
        gap_strength = clamp(safe_float(selected_gap.get("strength"), 0.0), 0.0, 1.0)

        if (
            event_type in {FVGEventType.FVG_RESPECTED, FVGEventType.FVG_RETESTED}
            and confidence >= 0.72
            and score >= 0.65
        ):
            return SignalPriority.HIGH

        if (
            selected_gap.get("status") == FVGStatus.RESPECTED
            and confidence >= 0.70
            and gap_strength >= 0.65
        ):
            return SignalPriority.HIGH

        if confidence >= 0.85 and score >= 0.78:
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _resolve_current_price(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
    ) -> float | None:
        candidates = (
            getattr(context, "last_price", None),
            getattr(context, "current_price", None),
            getattr(context, "price", None),
            self._get_context_feature(context, "last_price"),
            self._get_context_feature(context, "current_price"),
            self._get_context_feature(context, "price"),
            fvg.last_price,
        )

        for candidate in candidates:
            value = safe_float(candidate, 0.0)
            if value > 0:
                return value

        return None

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _status_from_event_type(self, event_type: Any) -> FVGStatus:
        parsed = self._parse_fvg_event_type(event_type)

        if parsed == FVGEventType.FVG_PARTIALLY_FILLED:
            return FVGStatus.PARTIALLY_FILLED
        if parsed == FVGEventType.FVG_FILLED:
            return FVGStatus.FILLED
        if parsed in {FVGEventType.FVG_RESPECTED, FVGEventType.FVG_RETESTED}:
            return FVGStatus.RESPECTED
        if parsed == FVGEventType.FVG_INVALIDATED:
            return FVGStatus.INVALIDATED

        return FVGStatus.ACTIVE

    # ------------------------------------------------------------------
    # Enum parsing
    # ------------------------------------------------------------------

    def _parse_structure_layer(self, value: Any) -> StructureLayer | None:
        raw = enum_value(value)

        if raw == "internal":
            return StructureLayer.INTERNAL

        if raw == "external":
            return StructureLayer.EXTERNAL

        try:
            return StructureLayer(raw)
        except Exception:
            return None

    def _parse_fvg_direction(self, value: Any) -> FVGDirection | None:
        raw = enum_value(value)

        if raw == "bullish":
            return FVGDirection.BULLISH

        if raw == "bearish":
            return FVGDirection.BEARISH

        try:
            return FVGDirection(raw)
        except Exception:
            return None

    def _parse_fvg_status(self, value: Any) -> FVGStatus | None:
        raw = enum_value(value)
        mapping = {
            "active": FVGStatus.ACTIVE,
            "partially_filled": FVGStatus.PARTIALLY_FILLED,
            "filled": FVGStatus.FILLED,
            "respected": FVGStatus.RESPECTED,
            "invalidated": FVGStatus.INVALIDATED,
        }

        if raw in mapping:
            return mapping[raw]

        try:
            return FVGStatus(raw)
        except Exception:
            return None

    def _parse_fvg_event_type(self, value: Any) -> FVGEventType | None:
        raw = enum_value(value)
        mapping = {
            "fvg_created": FVGEventType.FVG_CREATED,
            "created": FVGEventType.FVG_CREATED,
            "fvg_fill_started": FVGEventType.FVG_FILL_STARTED,
            "fill_started": FVGEventType.FVG_FILL_STARTED,
            "fvg_partially_filled": FVGEventType.FVG_PARTIALLY_FILLED,
            "partially_filled": FVGEventType.FVG_PARTIALLY_FILLED,
            "fvg_filled": FVGEventType.FVG_FILLED,
            "filled": FVGEventType.FVG_FILLED,
            "fvg_respected": FVGEventType.FVG_RESPECTED,
            "respected": FVGEventType.FVG_RESPECTED,
            "fvg_invalidated": FVGEventType.FVG_INVALIDATED,
            "invalidated": FVGEventType.FVG_INVALIDATED,
            "fvg_retested": FVGEventType.FVG_RETESTED,
            "retested": FVGEventType.FVG_RETESTED,
            "fvg_merged": FVGEventType.FVG_MERGED,
            "merged": FVGEventType.FVG_MERGED,
        }

        if raw in mapping:
            return mapping[raw]

        try:
            return FVGEventType(raw)
        except Exception:
            return None