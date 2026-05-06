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
    strategy_name: str = "fvg_reaction_strategy"

    prefer_external_layer: bool = True
    require_recent_event: bool = True
    require_directional_gap: bool = True
    require_respected_or_retested: bool = False
    allow_active_gap_proximity_entry: bool = True
    allow_partial_fill_reaction: bool = True
    allow_respected_reaction: bool = True
    allow_retested_reaction: bool = True

    block_invalidated_gaps: bool = True
    block_filled_gaps: bool = True
    block_counter_regime: bool = False

    min_gap_strength: float = 0.45
    min_event_confidence: float = 0.45
    min_gap_fill_for_reaction: float = 0.05
    max_gap_fill_for_entry: float = 0.90
    max_distance_to_mid_pct: float = 0.0035
    max_recent_fill_activity: float = 0.95

    primary_gap_weight: float = 0.28
    event_confidence_weight: float = 0.22
    status_quality_weight: float = 0.15
    proximity_weight: float = 0.12
    fill_quality_weight: float = 0.10
    regime_alignment_weight: float = 0.06
    retest_bonus_weight: float = 0.04
    respect_bonus_weight: float = 0.03

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.fvg_reaction.signal"

    freshness_feature_names: tuple[str, ...] = (
        "price_action.fvg",
        "price_action.fair_value_gap",
        "fair_value_gap",
        "fvg",
        "analytics.price_action.fair_value_gap",
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
    symbol: str | None = None
    timeframe: str | None = None
    last_price: float | None = None
    last_update: datetime | None = None
    internal: dict[str, Any] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)
    last_event: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class FVGReactionStrategy(PriceActionStrategyBase):
    """
    Strategy layer wrapper around analytics.price_action.fair_value_gap.

    Ідея:
    - bullish FVG reaction -> LONG
    - bearish FVG reaction -> SHORT
    - найсильніші сценарії: respected / retested / partial-fill reaction
    - fallback: active nearest gap near current price
    """

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

    # ------------------------------------------------------------------
    # Typed params accessor — fixes "Cannot find reference in ParamsT"
    # ------------------------------------------------------------------

    @property
    def _p(self) -> FVGReactionStrategyParams:
        """Typed shortcut so IDE resolves all FVGReactionStrategyParams fields."""
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
            )
            if freshness_filter is not None and freshness_filter.blocked:
                return self._rejected_evaluation(
                    context=context,
                    reason="stale_fvg_feature",
                )

            primary_layer = self._select_primary_layer(fvg)
            secondary_layer = (
                fvg.internal
                if self._p.prefer_external_layer
                else fvg.external
            )

            selected_gap = self._select_reaction_gap(
                context=context,
                fvg=fvg,
                primary_layer=primary_layer,
            )
            if selected_gap is None:
                return self._rejected_evaluation(
                    context=context,
                    reason="no_reactable_fvg_found",
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
                )

            score = self._compute_score(
                context=context,
                selected_gap=selected_gap,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                last_event=fvg.last_event,
                side=side,
            )
            confidence = self._compute_confidence(
                context=context,
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
    # Extraction
    # ------------------------------------------------------------------

    def _extract_fvg_snapshot(self, context: SignalContext) -> FVGReactionContext:
        candidates: list[Any] = [
            context.price_action.get("fvg"),
            context.price_action.get("fair_value_gap"),
            context.get_feature("price_action.fvg"),
            context.get_feature("price_action.fair_value_gap"),
            context.get_feature("fair_value_gap"),
            context.get_feature("fvg"),
            context.get_feature("analytics.price_action.fair_value_gap"),
        ]

        for candidate in candidates:
            normalized = self._normalize_fvg_snapshot(candidate)
            if normalized.symbol is not None or normalized.external or normalized.internal:
                return normalized

        return FVGReactionContext()

    def _normalize_fvg_snapshot(self, payload: Any) -> FVGReactionContext:
        if payload is None:
            return FVGReactionContext()

        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return FVGReactionContext()

        state = self._state_mapping_or_empty(payload_mapping)
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
        last_event = self._extract_last_event(internal, external)

        last_price = safe_float(
            first_non_empty(
                state.get("last_price"),
                payload_mapping.get("last_price"),
            ),
            0.0,
        )

        return FVGReactionContext(
            symbol=first_non_empty(
                state.get("symbol"),
                payload_mapping.get("symbol"),
            ),
            timeframe=first_non_empty(
                state.get("timeframe"),
                payload_mapping.get("timeframe"),
            ),
            last_price=last_price or None,
            last_update=parse_datetime(
                first_non_empty(
                    state.get("last_update"),
                    payload_mapping.get("last_update"),
                )
            ),
            internal=internal,
            external=external,
            last_event=last_event,
            raw=dict(payload_mapping),
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
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
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
            "recent_fill_activity": clamp(
                safe_float(payload_mapping.get("recent_fill_activity"), 0.0),
                0.0,
                1.0,
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

        return {
            "gap_id": payload_mapping.get("gap_id"),
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
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
            "created_index": (
                int(safe_float(payload_mapping.get("created_index"), 0.0))
                if payload_mapping.get("created_index") is not None
                else None
            ),
            "last_touch_index": (
                int(safe_float(payload_mapping.get("last_touch_index"), 0.0))
                if payload_mapping.get("last_touch_index") is not None
                else None
            ),
            "last_fill_index": (
                int(safe_float(payload_mapping.get("last_fill_index"), 0.0))
                if payload_mapping.get("last_fill_index") is not None
                else None
            ),
            "source_candle_indices": list(
                payload_mapping.get("source_candle_indices", []) or []
            ),
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
            "event_type": self._parse_fvg_event_type(payload_mapping.get("event_type")),
            "timestamp": parse_datetime(payload_mapping.get("timestamp")),
            "symbol": payload_mapping.get("symbol"),
            "timeframe": payload_mapping.get("timeframe"),
            "layer": self._parse_structure_layer(payload_mapping.get("layer")) or default_layer,
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
        # FIX: використовуємо timezone-aware datetime.min щоб уникнути
        # TypeError при порівнянні naive і aware datetime
        _epoch = datetime.min.replace(tzinfo=timezone.utc)

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

        def _sort_key(x: dict[str, Any]) -> datetime:
            ts = x.get("timestamp")
            if ts is None:
                return _epoch
            # якщо naive — прив'язуємо до UTC щоб уникнути помилки порівняння
            if isinstance(ts, datetime) and ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        candidates.sort(key=_sort_key, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Selection / side
    # ------------------------------------------------------------------

    def _select_primary_layer(self, fvg: FVGReactionContext) -> dict[str, Any]:
        return fvg.external if self._p.prefer_external_layer else fvg.internal

    def _select_reaction_gap(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext,
        primary_layer: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        last_event = fvg.last_event
        current_price = self._resolve_current_price(context=context, fvg=fvg)

        if (
            self._p.require_recent_event
            and last_event is not None
            and self._event_is_reaction_event(last_event)
            and clamp(
                safe_float(last_event.get("confidence"), 0.0),
                0.0,
                1.0,
            )
            >= self._p.min_event_confidence
        ):
            gap = self._gap_from_event(
                primary_layer=primary_layer,
                event=last_event,
            )
            if gap is not None:
                return gap

        candidate_gaps = [
            primary_layer.get("strongest_bullish_gap"),
            primary_layer.get("strongest_bearish_gap"),
            primary_layer.get("nearest_bullish_gap"),
            primary_layer.get("nearest_bearish_gap"),
        ]

        best_gap: dict[str, Any] | None = None
        best_score = -1.0

        for gap in candidate_gaps:
            if gap is None:
                continue

            if not self._gap_is_tradeable(gap, primary_layer):
                continue

            # FIX: доступ через _p замість self.params
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

    def _resolve_side(self, gap: Mapping[str, Any]) -> SignalSide:
        direction = gap.get("direction", FVGDirection.BULLISH)

        if direction == FVGDirection.BULLISH:
            return SignalSide.LONG

        if direction == FVGDirection.BEARISH:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _gap_is_tradeable(
        self,
        gap: Mapping[str, Any],
        layer: Mapping[str, Any],
    ) -> bool:
        if not gap:
            return False

        # FIX: доступ через _p замість self.params
        if self._p.require_directional_gap and gap.get("direction") not in {
            FVGDirection.BULLISH,
            FVGDirection.BEARISH,
        }:
            return False

        if (
            clamp(safe_float(gap.get("strength"), 0.0), 0.0, 1.0)
            < self._p.min_gap_strength
        ):
            return False

        status = gap.get("status", FVGStatus.ACTIVE)

        if self._p.block_invalidated_gaps and status == FVGStatus.INVALIDATED:
            return False

        if self._p.block_filled_gaps and status == FVGStatus.FILLED:
            return False

        if (
            clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0)
            > self._p.max_gap_fill_for_entry
        ):
            return False

        if self._p.require_respected_or_retested:
            respected = status == FVGStatus.RESPECTED
            retested = int(safe_float(gap.get("retest_count"), 0.0)) > 0

            if not (respected or retested):
                return False

        if (
            clamp(safe_float(layer.get("recent_fill_activity"), 0.0), 0.0, 1.0)
            > self._p.max_recent_fill_activity
        ):
            return False

        return True

    def _gap_selection_score(
        self,
        *,
        gap: Mapping[str, Any],
        current_price: float | None,
    ) -> float:
        score = 0.0
        score += 0.45 * clamp(safe_float(gap.get("strength"), 0.0), 0.0, 1.0)
        score += 0.20 * self._status_quality_score(gap.get("status"))
        score += 0.15 * (
            1.0 - clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0)
        )
        score += 0.10 * min(1.0, safe_float(gap.get("retest_count"), 0.0) / 3.0)
        score += 0.10 * self._proximity_score(
            current_price=current_price,
            gap=gap,
        )
        return clamp(score, 0.0, 1.0)

    def _gap_from_event(
        self,
        *,
        primary_layer: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        gap_id = event.get("gap_id")
        if not gap_id:
            return None

        for key in (
            "strongest_bullish_gap",
            "strongest_bearish_gap",
            "nearest_bullish_gap",
            "nearest_bearish_gap",
        ):
            gap = primary_layer.get(key)
            if gap is not None and gap.get("gap_id") == gap_id:
                return gap

        direction = event.get("direction")

        if direction == FVGDirection.BULLISH:
            return (
                primary_layer.get("nearest_bullish_gap")
                or primary_layer.get("strongest_bullish_gap")
            )

        if direction == FVGDirection.BEARISH:
            return (
                primary_layer.get("nearest_bearish_gap")
                or primary_layer.get("strongest_bearish_gap")
            )

        return None

    # ------------------------------------------------------------------
    # Score / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        selected_gap: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        # FIX: передаємо fvg=None явно — метод коректно обробляє None
        current_price = self._resolve_current_price(context=context, fvg=None)

        score = 0.0
        # FIX: скрізь self._p замість self.params
        score += self._p.primary_gap_weight * clamp(
            safe_float(selected_gap.get("strength"), 0.0),
            0.0,
            1.0,
        )

        event_confidence = clamp(
            safe_float(last_event.get("confidence"), 0.0)
            if last_event
            else 0.0,
            0.0,
            1.0,
        )
        score += self._p.event_confidence_weight * event_confidence

        score += self._p.status_quality_weight * self._status_quality_score(
            selected_gap.get("status")
        )
        score += self._p.proximity_weight * self._proximity_score(
            current_price=current_price,
            gap=selected_gap,
        )
        score += self._p.fill_quality_weight * self._fill_quality_score(selected_gap)

        secondary_alignment = self._secondary_layer_alignment_score(
            secondary_layer=secondary_layer,
            selected_gap=selected_gap,
        )
        score += self._p.regime_alignment_weight * max(
            self._regime_alignment_score(context=context, side=side),
            secondary_alignment,
        )

        if safe_float(selected_gap.get("retest_count"), 0.0) > 0:
            score += self._p.retest_bonus_weight

        if selected_gap.get("status") == FVGStatus.RESPECTED:
            score += self._p.respect_bonus_weight

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        context: SignalContext,
        selected_gap: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        # FIX: передаємо fvg=None явно — метод коректно обробляє None
        current_price = self._resolve_current_price(context=context, fvg=None)

        components: list[float] = [
            clamp(safe_float(selected_gap.get("strength"), 0.0), 0.0, 1.0),
            self._status_quality_score(selected_gap.get("status")),
            self._fill_quality_score(selected_gap),
            self._proximity_score(current_price=current_price, gap=selected_gap),
            self._regime_alignment_score(context=context, side=side),
        ]

        if last_event is not None:
            components.append(
                clamp(safe_float(last_event.get("confidence"), 0.0), 0.0, 1.0)
            )

        components.append(
            self._secondary_layer_alignment_score(
                secondary_layer=secondary_layer,
                selected_gap=selected_gap,
            )
        )

        return clamp(sum(components) / len(components), 0.0, 1.0)

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
        # FIX: перевірка isinstance перед .value — парсер може повернути None
        if isinstance(layer, StructureLayer):
            reasons.append(f"fvg_layer_{layer.value}")

        direction = selected_gap.get("direction")
        # FIX: перевірка isinstance перед .value
        if isinstance(direction, FVGDirection):
            reasons.append(f"fvg_direction_{direction.value}")

        status = selected_gap.get("status")
        # FIX: перевірка isinstance перед .value
        if isinstance(status, FVGStatus):
            reasons.append(f"fvg_status_{status.value}")

        if safe_float(selected_gap.get("retest_count"), 0.0) > 0:
            reasons.append("fvg_retested")

        if selected_gap.get("status") == FVGStatus.RESPECTED:
            reasons.append("fvg_respected")

        last_event = fvg.last_event
        if last_event is not None:
            event_type = last_event.get("event_type")
            # FIX: перевірка isinstance перед .value
            if isinstance(event_type, FVGEventType):
                reasons.append(f"last_fvg_event_{event_type.value}")

        return reasons

    # ------------------------------------------------------------------
    # Signal build
    # ------------------------------------------------------------------

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
        setup_type = self._resolve_setup_type(selected_gap, fvg.last_event)
        trigger_type = self._resolve_trigger_type(selected_gap, fvg.last_event)
        priority = self._resolve_priority(
            confidence=confidence,
            gap=selected_gap,
            last_event=fvg.last_event,
        )

        gap_layer = selected_gap.get("layer")
        gap_direction = selected_gap.get("direction")
        gap_status = selected_gap.get("status")

        last_event_type = (
            fvg.last_event.get("event_type")
            if fvg.last_event is not None
            else None
        )

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
            trigger_type=trigger_type,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
            regime=self._resolve_market_regime(context),
            metadata={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.fair_value_gap",
                "fvg_timeframe": fvg.timeframe,
                "fvg_last_update": (
                    fvg.last_update.isoformat()
                    if fvg.last_update
                    else None
                ),
                "fvg_last_price": fvg.last_price,
                "gap_id": selected_gap.get("gap_id"),
                # FIX: isinstance-guard перед .value щоб уникнути AttributeError
                "gap_layer": gap_layer.value if isinstance(gap_layer, StructureLayer) else None,
                "gap_direction": gap_direction.value if isinstance(gap_direction, FVGDirection) else None,
                "gap_status": gap_status.value if isinstance(gap_status, FVGStatus) else None,
                "gap_strength": safe_float(selected_gap.get("strength"), 0.0),
                "gap_fill_percentage": safe_float(
                    selected_gap.get("fill_percentage"),
                    0.0,
                ),
                "gap_mid_price": safe_float(selected_gap.get("mid_price"), 0.0),
                "gap_upper_bound": safe_float(selected_gap.get("upper_bound"), 0.0),
                "gap_lower_bound": safe_float(selected_gap.get("lower_bound"), 0.0),
                "gap_retest_count": int(
                    safe_float(selected_gap.get("retest_count"), 0.0)
                ),
                # FIX: isinstance-guard перед .value
                "last_fvg_event_type": (
                    last_event_type.value
                    if isinstance(last_event_type, FVGEventType)
                    else None
                ),
                "last_fvg_event_confidence": (
                    safe_float(fvg.last_event.get("confidence"), 0.0)
                    if fvg.last_event is not None
                    else None
                ),
            },
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("price_action.fair_value_gap")

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
        if last_event is not None and last_event.get("event_type") in {
            FVGEventType.FVG_RESPECTED,
            FVGEventType.FVG_RETESTED,
        }:
            return SetupType.REVERSAL

        if gap.get("status") == FVGStatus.PARTIALLY_FILLED:
            return SetupType.MEAN_REVERSION

        return SetupType.CONTINUATION

    def _resolve_trigger_type(
        self,
        gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> TriggerType:
        if last_event is not None:
            if last_event.get("event_type") in {
                FVGEventType.FVG_RESPECTED,
                FVGEventType.FVG_RETESTED,
            }:
                return TriggerType.PRIMARY

            if last_event.get("event_type") in {
                FVGEventType.FVG_PARTIALLY_FILLED,
                FVGEventType.FVG_FILL_STARTED,
            }:
                return TriggerType.CONFIRMATION

        if gap.get("status") == FVGStatus.ACTIVE:
            return TriggerType.DERIVED

        return TriggerType.CONFIRMATION

    def _resolve_priority(
        self,
        *,
        confidence: float,
        gap: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> SignalPriority:
        if (
            confidence >= 0.82
            and gap.get("status") in {
                FVGStatus.RESPECTED,
                FVGStatus.PARTIALLY_FILLED,
            }
        ):
            return SignalPriority.HIGH

        if (
            last_event is not None
            and last_event.get("event_type")
            in {
                FVGEventType.FVG_RESPECTED,
                FVGEventType.FVG_RETESTED,
            }
            and confidence >= 0.70
        ):
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    # ------------------------------------------------------------------
    # Helper logic
    # ------------------------------------------------------------------

    def _event_is_reaction_event(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("event_type")

        # FIX: доступ через _p
        if (
            event_type == FVGEventType.FVG_RESPECTED
            and self._p.allow_respected_reaction
        ):
            return True

        if (
            event_type == FVGEventType.FVG_RETESTED
            and self._p.allow_retested_reaction
        ):
            return True

        if (
            event_type
            in {
                FVGEventType.FVG_PARTIALLY_FILLED,
                FVGEventType.FVG_FILL_STARTED,
            }
            and self._p.allow_partial_fill_reaction
        ):
            return True

        return False

    def _resolve_current_price(
        self,
        *,
        context: SignalContext,
        fvg: FVGReactionContext | None,
    ) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return context.price.mid_price

            if context.price.last_price is not None:
                return context.price.last_price

        if fvg is not None and fvg.last_price is not None:
            return fvg.last_price

        return None

    def _status_quality_score(self, status: FVGStatus | None) -> float:
        if status == FVGStatus.RESPECTED:
            return 1.0

        if status == FVGStatus.PARTIALLY_FILLED:
            return 0.82

        if status == FVGStatus.ACTIVE:
            return 0.66

        if status == FVGStatus.FILLED:
            return 0.18

        if status == FVGStatus.INVALIDATED:
            return 0.0

        return 0.40

    def _fill_quality_score(self, gap: Mapping[str, Any]) -> float:
        fill = clamp(
            safe_float(gap.get("fill_percentage"), 0.0),
            0.0,
            1.0,
        )

        # FIX: доступ через _p
        if fill < self._p.min_gap_fill_for_reaction:
            return 0.35 if self._p.allow_active_gap_proximity_entry else 0.0

        if fill > self._p.max_gap_fill_for_entry:
            return 0.0

        center = 0.45
        distance = abs(fill - center)
        score = 1.0 - min(1.0, distance / max(center, 1e-9))

        return clamp(score, 0.0, 1.0)

    def _proximity_score(
        self,
        *,
        current_price: float | None,
        gap: Mapping[str, Any],
    ) -> float:
        if current_price is None:
            return 0.0

        mid_price = safe_float(gap.get("mid_price"), 0.0)
        if mid_price <= 0:
            return 0.0

        distance_pct = abs(current_price - mid_price) / mid_price
        # FIX: доступ через _p
        if distance_pct >= self._p.max_distance_to_mid_pct:
            return 0.0

        return clamp(
            1.0
            - (
                distance_pct
                / max(self._p.max_distance_to_mid_pct, 1e-9)
            ),
            0.0,
            1.0,
        )

    def _secondary_layer_alignment_score(
        self,
        *,
        secondary_layer: Mapping[str, Any],
        selected_gap: Mapping[str, Any],
    ) -> float:
        # FIX: якщо secondary_layer порожній — одразу повертаємо нейтральний score
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

        if ref_gap is None:
            return 0.35

        score = 0.0
        score += 0.6 * clamp(safe_float(ref_gap.get("strength"), 0.0), 0.0, 1.0)
        score += 0.4 * self._status_quality_score(ref_gap.get("status"))

        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Enum parsing
    # ------------------------------------------------------------------

    def _parse_structure_layer(self, value: Any) -> StructureLayer | None:
        raw = enum_value(value)

        if raw == "internal":
            return StructureLayer.INTERNAL

        if raw == "external":
            return StructureLayer.EXTERNAL

        return None

    def _parse_fvg_direction(self, value: Any) -> FVGDirection | None:
        raw = enum_value(value)

        if raw == "bullish":
            return FVGDirection.BULLISH

        if raw == "bearish":
            return FVGDirection.BEARISH

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
        return mapping.get(raw)

    def _parse_fvg_event_type(self, value: Any) -> FVGEventType | None:
        raw = enum_value(value)
        mapping = {
            "fvg_created": FVGEventType.FVG_CREATED,
            "fvg_fill_started": FVGEventType.FVG_FILL_STARTED,
            "fvg_partially_filled": FVGEventType.FVG_PARTIALLY_FILLED,
            "fvg_filled": FVGEventType.FVG_FILLED,
            "fvg_respected": FVGEventType.FVG_RESPECTED,
            "fvg_invalidated": FVGEventType.FVG_INVALIDATED,
            "fvg_retested": FVGEventType.FVG_RETESTED,
            "fvg_merged": FVGEventType.FVG_MERGED,
        }
        return mapping.get(raw)