from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import (
    LevelStatus,
    LevelType,
    SREventType,
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
class SupportResistanceReactionStrategyParams:
    """
    Local params for SupportResistanceReactionStrategy.

    Runtime gates such as enabled/symbols/timeframes/min_score/min_confidence
    stay in StrategyConfig / StrategyDefinitionConfig.runtime. These params
    define how this strategy consumes analytics.price_action.support_resistance.
    """

    strategy_name: str = "support_resistance_reaction_strategy"

    prefer_external_layer: bool = True
    require_recent_event: bool = True
    require_level_strength: bool = True

    allow_support_rejection_long: bool = True
    allow_resistance_rejection_short: bool = True
    allow_support_break_short: bool = True
    allow_resistance_break_long: bool = True
    allow_flip_support_long: bool = True
    allow_flip_resistance_short: bool = True
    allow_retest_entries: bool = True
    allow_touch_entries: bool = False
    allow_created_level_entries: bool = False
    allow_merged_level_entries: bool = False
    allow_nearest_level_fallback: bool = True

    block_inactive_levels: bool = True
    block_broken_non_flip_levels: bool = True
    block_counter_regime: bool = False

    min_level_strength: float = 0.45
    min_event_confidence: float = 0.45
    min_touch_count: int = 1
    min_rejection_count_for_reaction: int = 1
    min_retest_count_for_retest_entry: int = 1

    max_distance_to_level_pct: float = 0.0035
    max_zone_width_pct: float = 0.0060

    primary_level_weight: float = 0.24
    event_confidence_weight: float = 0.20
    status_quality_weight: float = 0.12
    proximity_weight: float = 0.14
    interaction_quality_weight: float = 0.12
    secondary_layer_alignment_weight: float = 0.08
    regime_alignment_weight: float = 0.05
    retest_bonus_weight: float = 0.03
    flip_bonus_weight: float = 0.02

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.support_resistance_reaction.signal"

    freshness_feature_names: tuple[str, ...] = (
        "analytics.price_action",
        "analytics.price_action.support_resistance",
        "price_action.support_resistance",
        "support_resistance",
        "sr",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_level_strength",
                "min_event_confidence",
                "max_distance_to_level_pct",
                "max_zone_width_pct",
                "primary_level_weight",
                "event_confidence_weight",
                "status_quality_weight",
                "proximity_weight",
                "interaction_quality_weight",
                "secondary_layer_alignment_weight",
                "regime_alignment_weight",
                "retest_bonus_weight",
                "flip_bonus_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

        for field_name in (
            "min_touch_count",
            "min_rejection_count_for_reaction",
            "min_retest_count_for_retest_entry",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
            setattr(self, field_name, value)

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "SupportResistanceReactionStrategyParams":
        return apply_definition_metadata(
            params=cls(),
            definition=definition,
        )


@dataclass(slots=True)
class SupportResistanceContext:
    """
    Normalized view of analytics.price_action.support_resistance.SupportResistanceState.
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


class SupportResistanceReactionStrategy(PriceActionStrategyBase):
    """
    Strategy wrapper around analytics.price_action.support_resistance.

    Main setups:
    - support rejection -> LONG;
    - resistance rejection -> SHORT;
    - resistance break / flip support / retest -> LONG;
    - support break / flip resistance / retest -> SHORT;
    - optional nearest-level fallback when no fresh event exists.

    This class is intentionally aligned only with analytics.price_action
    contracts. Broader alignment with the final strategy base hierarchy can be
    done later.
    """

    analytics_module_name = "support_resistance"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
        strategy_name: str = "support_resistance_reaction_strategy",
    ) -> None:
        super().__init__(
            config=config,
            strategy_name=strategy_name,
            params_cls=SupportResistanceReactionStrategyParams,
            event_bus=event_bus,
            logger=logger,
        )

    @property
    def _p(self) -> SupportResistanceReactionStrategyParams:
        return cast(SupportResistanceReactionStrategyParams, self.params)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked

            sr = self._extract_support_resistance_snapshot(context)
            if sr.symbol is None and not sr.external and not sr.internal:
                return self._rejected_evaluation(
                    context=context,
                    reason="support_resistance_snapshot_missing",
                )

            freshness_filter = self._build_freshness_filter(
                context=context,
                filter_name="support_resistance_freshness",
                module_name=self.analytics_module_name,
                analytics_payload=sr.raw,
            )
            if freshness_filter is not None and freshness_filter.blocked:
                return self._rejected_evaluation(
                    context=context,
                    reason="stale_support_resistance_feature",
                )

            primary_layer = self._select_primary_layer(sr)
            secondary_layer = self._select_secondary_layer(sr)

            selected_level = self._select_reaction_level(
                context=context,
                sr=sr,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )
            if selected_level is None:
                return self._rejected_evaluation(
                    context=context,
                    reason="no_reactable_support_resistance_level_found",
                    metadata={
                        "analytics_module": self.analytics_module_name,
                        "analytics_source_feature": sr.source_feature,
                        "last_sr_event_type": (
                            enum_value(sr.last_event.get("event_type"))
                            if sr.last_event is not None
                            else None
                        ),
                    },
                )

            side = self._resolve_side(
                level=selected_level,
                last_event=sr.last_event,
            )
            if side == SignalSide.UNKNOWN:
                return self._rejected_evaluation(
                    context=context,
                    reason="support_resistance_level_not_directional",
                    metadata={
                        "level_id": selected_level.get("level_id"),
                        "level_type": enum_value(selected_level.get("level_type")),
                        "level_status": enum_value(selected_level.get("status")),
                    },
                )

            if not self._level_is_tradeable(selected_level):
                return self._rejected_evaluation(
                    context=context,
                    reason="selected_support_resistance_level_not_tradeable",
                    metadata={
                        "level_id": selected_level.get("level_id"),
                        "level_type": enum_value(selected_level.get("level_type")),
                        "level_status": enum_value(selected_level.get("status")),
                        "level_strength": selected_level.get("strength"),
                    },
                )

            score = self._compute_score(
                context=context,
                sr=sr,
                selected_level=selected_level,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                last_event=sr.last_event,
                side=side,
            )
            confidence = self._compute_confidence(
                context=context,
                sr=sr,
                selected_level=selected_level,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                last_event=sr.last_event,
                side=side,
            )
            reasons = self._build_reasons(
                sr=sr,
                selected_level=selected_level,
                side=side,
            )

            signal = self._build_signal(
                context=context,
                sr=sr,
                selected_level=selected_level,
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
                    "analytics_source_feature": sr.source_feature,
                },
            )

        except StrategyEvaluationError:
            raise
        except Exception as exc:
            self._logger.exception(
                "Failed to evaluate support/resistance reaction strategy | strategy=%s symbol=%s",
                self.name,
                getattr(context, "symbol", None),
            )
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate support/resistance reaction for {context.symbol}"
            ) from exc

    # ------------------------------------------------------------------
    # Extraction / normalization
    # ------------------------------------------------------------------

    def _extract_support_resistance_snapshot(
        self,
        context: SignalContext,
    ) -> SupportResistanceContext:
        payload = self._extract_price_action_module(
            context,
            self.analytics_module_name,
            aliases=(
                "support_resistance",
                "sr",
                "price_action.support_resistance",
                "analytics.price_action.support_resistance",
            ),
            require_scope_match=True,
        )
        if payload:
            return self._normalize_support_resistance_snapshot(payload)

        candidates: list[Any] = [
            self._mapping_or_empty(getattr(context, "price_action", None)).get(
                "support_resistance"
            ),
            self._mapping_or_empty(getattr(context, "price_action", None)).get("sr"),
            self._get_context_feature(context, "price_action.support_resistance"),
            self._get_context_feature(context, "support_resistance"),
            self._get_context_feature(context, "sr"),
            self._get_context_feature(context, "analytics.price_action.support_resistance"),
        ]

        for candidate in candidates:
            normalized = self._normalize_support_resistance_snapshot(candidate)
            if normalized.symbol is not None or normalized.external or normalized.internal:
                return normalized

        return SupportResistanceContext()

    def _normalize_support_resistance_snapshot(
        self,
        payload: Any,
    ) -> SupportResistanceContext:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return SupportResistanceContext()

        state = self._normalize_state_payload(payload_mapping)
        if not state:
            return SupportResistanceContext()

        internal = self._normalize_sr_layer(
            state.get("internal"),
            StructureLayer.INTERNAL,
        )
        external = self._normalize_sr_layer(
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

        return SupportResistanceContext(
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

    def _normalize_sr_layer(
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
            "total_levels": int(safe_float(payload_mapping.get("total_levels"), 0.0)),
            "active_supports": int(
                safe_float(payload_mapping.get("active_supports"), 0.0)
            ),
            "active_resistances": int(
                safe_float(payload_mapping.get("active_resistances"), 0.0)
            ),
            "active_flip_supports": int(
                safe_float(payload_mapping.get("active_flip_supports"), 0.0)
            ),
            "active_flip_resistances": int(
                safe_float(payload_mapping.get("active_flip_resistances"), 0.0)
            ),
            "strongest_support": self._normalize_level(
                payload_mapping.get("strongest_support"),
                default_layer,
            ),
            "strongest_resistance": self._normalize_level(
                payload_mapping.get("strongest_resistance"),
                default_layer,
            ),
            "nearest_support": self._normalize_level(
                payload_mapping.get("nearest_support"),
                default_layer,
            ),
            "nearest_resistance": self._normalize_level(
                payload_mapping.get("nearest_resistance"),
                default_layer,
            ),
            "last_event": self._normalize_sr_event(
                payload_mapping.get("last_event"),
                default_layer,
            ),
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_level(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any] | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        source_prices: list[float] = []
        for price in list(payload_mapping.get("source_prices", []) or []):
            value = safe_float(price, 0.0)
            if value > 0:
                source_prices.append(value)

        return {
            "level_id": payload_mapping.get("level_id"),
            "exchange": payload_mapping.get("exchange"),
            "market_type": payload_mapping.get("market_type"),
            "symbol": payload_mapping.get("symbol"),
            "exchange_symbol": payload_mapping.get("exchange_symbol"),
            "timeframe": payload_mapping.get("timeframe"),
            "key": list(payload_mapping.get("key", []) or []),
            "layer": self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            "level_type": self._parse_level_type(payload_mapping.get("level_type")),
            "price": safe_float(payload_mapping.get("price"), 0.0),
            "upper_bound": safe_float(payload_mapping.get("upper_bound"), 0.0),
            "lower_bound": safe_float(payload_mapping.get("lower_bound"), 0.0),
            "strength": clamp(
                safe_float(payload_mapping.get("strength"), 0.0),
                0.0,
                1.0,
            ),
            "status": self._parse_level_status(payload_mapping.get("status")),
            "created_at": parse_datetime(payload_mapping.get("created_at")),
            "updated_at": parse_datetime(payload_mapping.get("updated_at")),
            "broken_at": parse_datetime(payload_mapping.get("broken_at")),
            "flipped_at": parse_datetime(payload_mapping.get("flipped_at")),
            "last_tested_at": parse_datetime(payload_mapping.get("last_tested_at")),
            "last_rejected_at": parse_datetime(payload_mapping.get("last_rejected_at")),
            "last_broken_at": parse_datetime(payload_mapping.get("last_broken_at")),
            "last_retested_at": parse_datetime(payload_mapping.get("last_retested_at")),
            "touch_count": int(safe_float(payload_mapping.get("touch_count"), 0.0)),
            "rejection_count": int(
                safe_float(payload_mapping.get("rejection_count"), 0.0)
            ),
            "break_count": int(safe_float(payload_mapping.get("break_count"), 0.0)),
            "retest_count": int(safe_float(payload_mapping.get("retest_count"), 0.0)),
            "source_count": int(safe_float(payload_mapping.get("source_count"), 0.0)),
            "source_swing_ids": [
                str(item)
                for item in list(payload_mapping.get("source_swing_ids", []) or [])
                if item is not None
            ],
            "source_prices": source_prices,
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_sr_event(
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
            "event_type": self._parse_sr_event_type(payload_mapping.get("event_type")),
            "timestamp": parse_datetime(payload_mapping.get("timestamp")),
            "layer": self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            "level_id": payload_mapping.get("level_id"),
            "level_type": self._parse_level_type(payload_mapping.get("level_type")),
            "price": safe_float(payload_mapping.get("price"), 0.0),
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

    def _select_primary_layer(self, sr: SupportResistanceContext) -> dict[str, Any]:
        return sr.external if self._p.prefer_external_layer else sr.internal

    def _select_secondary_layer(self, sr: SupportResistanceContext) -> dict[str, Any]:
        return sr.internal if self._p.prefer_external_layer else sr.external

    def _select_reaction_level(
        self,
        *,
        context: SignalContext,
        sr: SupportResistanceContext,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        last_event = sr.last_event
        current_price = self._resolve_current_price(context=context, sr=sr)

        if last_event is not None and self._event_is_usable_for_entry(last_event):
            event_level = self._level_from_event(
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
                event=last_event,
            )
            if event_level is not None and self._level_is_tradeable(event_level):
                return event_level

        if self._p.require_recent_event:
            return None

        if not self._p.allow_nearest_level_fallback:
            return None

        candidate_levels = self._candidate_levels(primary_layer)

        best_level: dict[str, Any] | None = None
        best_score = -1.0

        for level in candidate_levels:
            if not self._level_is_tradeable(level):
                continue

            local_score = self._level_selection_score(
                level=level,
                current_price=current_price,
            )
            if local_score > best_score:
                best_score = local_score
                best_level = level

        return best_level

    def _candidate_levels(self, layer: Mapping[str, Any]) -> list[dict[str, Any]]:
        candidates = [
            layer.get("strongest_support"),
            layer.get("strongest_resistance"),
            layer.get("nearest_support"),
            layer.get("nearest_resistance"),
        ]
        return [dict(level) for level in candidates if isinstance(level, Mapping)]

    def _level_from_event(
        self,
        *,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        level_id = event.get("level_id")
        if not level_id:
            return None

        for layer in (primary_layer, secondary_layer):
            for key in (
                "strongest_support",
                "strongest_resistance",
                "nearest_support",
                "nearest_resistance",
            ):
                level = layer.get(key)
                if isinstance(level, Mapping) and level.get("level_id") == level_id:
                    return dict(level)

        return self._level_stub_from_event(event)

    def _level_stub_from_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        level_id = event.get("level_id")
        level_type = self._parse_level_type(event.get("level_type"))
        if not level_id or level_type is None:
            return None

        price = safe_float(event.get("price"), 0.0)
        if price <= 0:
            return None

        return {
            "level_id": level_id,
            "layer": self._parse_structure_layer(event.get("layer"))
            or StructureLayer.INTERNAL,
            "level_type": level_type,
            "price": price,
            "upper_bound": price,
            "lower_bound": price,
            "strength": clamp(safe_float(event.get("confidence"), 0.0), 0.0, 1.0),
            "status": self._status_from_event_type(event.get("event_type")),
            "created_at": None,
            "updated_at": event.get("timestamp"),
            "broken_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_BROKEN
            else None,
            "flipped_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_FLIPPED
            else None,
            "last_tested_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_TOUCHED
            else None,
            "last_rejected_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_REJECTED
            else None,
            "last_broken_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_BROKEN
            else None,
            "last_retested_at": event.get("timestamp")
            if event.get("event_type") == SREventType.LEVEL_RETESTED
            else None,
            "touch_count": 1
            if event.get("event_type") == SREventType.LEVEL_TOUCHED
            else 0,
            "rejection_count": 1
            if event.get("event_type") == SREventType.LEVEL_REJECTED
            else 0,
            "break_count": 1
            if event.get("event_type") == SREventType.LEVEL_BROKEN
            else 0,
            "retest_count": 1
            if event.get("event_type") == SREventType.LEVEL_RETESTED
            else 0,
            "source_count": 0,
            "source_swing_ids": [],
            "source_prices": [],
            "metadata": dict(event.get("metadata", {}) or {}),
        }

    def _resolve_side(
        self,
        *,
        level: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> SignalSide:
        event_type = last_event.get("event_type") if last_event is not None else None
        level_type = self._parse_level_type(level.get("level_type"))

        if event_type == SREventType.LEVEL_REJECTED:
            if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
                return SignalSide.LONG
            if level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
                return SignalSide.SHORT

        if event_type == SREventType.LEVEL_BROKEN:
            if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
                return SignalSide.SHORT
            if level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
                return SignalSide.LONG

        if event_type in {SREventType.LEVEL_FLIPPED, SREventType.LEVEL_RETESTED}:
            return self._level_type_to_reaction_side(level_type)

        if event_type == SREventType.LEVEL_TOUCHED:
            return self._level_type_to_reaction_side(level_type)

        return self._level_type_to_reaction_side(level_type)

    def _event_is_usable_for_entry(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("event_type")
        confidence = clamp(safe_float(event.get("confidence"), 0.0), 0.0, 1.0)

        if confidence < self._p.min_event_confidence:
            return False

        level_type = self._parse_level_type(event.get("level_type"))

        if event_type == SREventType.LEVEL_REJECTED:
            if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
                return self._p.allow_support_rejection_long
            if level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
                return self._p.allow_resistance_rejection_short
            return False

        if event_type == SREventType.LEVEL_BROKEN:
            if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
                return self._p.allow_support_break_short
            if level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
                return self._p.allow_resistance_break_long
            return False

        if event_type == SREventType.LEVEL_FLIPPED:
            if level_type == LevelType.FLIP_SUPPORT:
                return self._p.allow_flip_support_long
            if level_type == LevelType.FLIP_RESISTANCE:
                return self._p.allow_flip_resistance_short
            return False

        if event_type == SREventType.LEVEL_RETESTED:
            return self._p.allow_retest_entries

        if event_type == SREventType.LEVEL_TOUCHED:
            return self._p.allow_touch_entries

        if event_type == SREventType.LEVEL_CREATED:
            return self._p.allow_created_level_entries

        if event_type == SREventType.LEVEL_MERGED:
            return self._p.allow_merged_level_entries

        return False

    def _level_is_tradeable(self, level: Mapping[str, Any]) -> bool:
        if not level:
            return False

        level_type = self._parse_level_type(level.get("level_type"))
        if level_type not in {
            LevelType.SUPPORT,
            LevelType.RESISTANCE,
            LevelType.FLIP_SUPPORT,
            LevelType.FLIP_RESISTANCE,
        }:
            return False

        status = self._parse_level_status(level.get("status"))

        if self._p.block_inactive_levels and status == LevelStatus.INACTIVE:
            return False

        if self._p.block_broken_non_flip_levels:
            if status == LevelStatus.BROKEN and level_type in {
                LevelType.SUPPORT,
                LevelType.RESISTANCE,
            }:
                return False

        if self._p.require_level_strength:
            strength = clamp(safe_float(level.get("strength"), 0.0), 0.0, 1.0)
            if strength < self._p.min_level_strength:
                return False

        if int(safe_float(level.get("touch_count"), 0.0)) < self._p.min_touch_count:
            if int(safe_float(level.get("source_count"), 0.0)) <= 0:
                return False

        return True

    # ------------------------------------------------------------------
    # Score / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        context: SignalContext,
        sr: SupportResistanceContext,
        selected_level: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        current_price = self._resolve_current_price(context=context, sr=sr)

        score = 0.0
        score += self._p.primary_level_weight * clamp(
            safe_float(selected_level.get("strength"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.event_confidence_weight * self._event_score(
            selected_level=selected_level,
            last_event=last_event,
        )
        score += self._p.status_quality_weight * self._status_quality_score(
            selected_level.get("status"),
            selected_level.get("level_type"),
        )
        score += self._p.proximity_weight * self._proximity_score(
            current_price=current_price,
            level=selected_level,
        )
        score += self._p.interaction_quality_weight * self._interaction_quality_score(
            selected_level,
            last_event=last_event,
        )
        score += self._p.secondary_layer_alignment_weight * self._secondary_layer_alignment_score(
            secondary_layer=secondary_layer,
            selected_level=selected_level,
        )
        score += self._p.regime_alignment_weight * self._regime_alignment_score(
            context=context,
            side=side,
        )

        if int(safe_float(selected_level.get("retest_count"), 0.0)) > 0:
            score += self._p.retest_bonus_weight

        if selected_level.get("level_type") in {
            LevelType.FLIP_SUPPORT,
            LevelType.FLIP_RESISTANCE,
        }:
            score += self._p.flip_bonus_weight

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        context: SignalContext,
        sr: SupportResistanceContext,
        selected_level: Mapping[str, Any],
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        side: SignalSide,
    ) -> float:
        current_price = self._resolve_current_price(context=context, sr=sr)

        components = [
            clamp(safe_float(selected_level.get("strength"), 0.0), 0.0, 1.0),
            self._event_score(selected_level=selected_level, last_event=last_event),
            self._status_quality_score(
                selected_level.get("status"),
                selected_level.get("level_type"),
            ),
            self._proximity_score(current_price=current_price, level=selected_level),
            self._interaction_quality_score(selected_level, last_event=last_event),
            self._secondary_layer_alignment_score(
                secondary_layer=secondary_layer,
                selected_level=selected_level,
            ),
            self._regime_alignment_score(context=context, side=side),
        ]

        return clamp(sum(components) / len(components), 0.0, 1.0)

    def _event_score(
        self,
        *,
        selected_level: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> float:
        if last_event is None:
            return 0.35

        if last_event.get("level_id") != selected_level.get("level_id"):
            return 0.30

        event_type = last_event.get("event_type")
        confidence = clamp(safe_float(last_event.get("confidence"), 0.0), 0.0, 1.0)

        multiplier = {
            SREventType.LEVEL_REJECTED: 1.00,
            SREventType.LEVEL_RETESTED: 0.94,
            SREventType.LEVEL_FLIPPED: 0.90,
            SREventType.LEVEL_BROKEN: 0.84,
            SREventType.LEVEL_TOUCHED: 0.62,
            SREventType.LEVEL_CREATED: 0.45,
            SREventType.LEVEL_MERGED: 0.42,
        }.get(event_type, 0.35)

        return clamp(confidence * multiplier, 0.0, 1.0)

    def _status_quality_score(
        self,
        status: Any,
        level_type: Any,
    ) -> float:
        parsed_status = self._parse_level_status(status)
        parsed_type = self._parse_level_type(level_type)

        if parsed_status == LevelStatus.INACTIVE:
            return 0.0

        if parsed_type in {LevelType.FLIP_SUPPORT, LevelType.FLIP_RESISTANCE}:
            if parsed_status == LevelStatus.ACTIVE:
                return 1.0
            if parsed_status == LevelStatus.BROKEN:
                return 0.55

        if parsed_status == LevelStatus.ACTIVE:
            return 0.78

        if parsed_status == LevelStatus.BROKEN:
            return 0.35

        return 0.25

    def _interaction_quality_score(
        self,
        level: Mapping[str, Any],
        *,
        last_event: Mapping[str, Any] | None,
    ) -> float:
        touch_count = int(safe_float(level.get("touch_count"), 0.0))
        rejection_count = int(safe_float(level.get("rejection_count"), 0.0))
        break_count = int(safe_float(level.get("break_count"), 0.0))
        retest_count = int(safe_float(level.get("retest_count"), 0.0))
        source_count = int(safe_float(level.get("source_count"), 0.0))

        score = 0.0
        score += min(0.25, touch_count / 5.0 * 0.25)
        score += min(0.25, rejection_count / 3.0 * 0.25)
        score += min(0.20, retest_count / 3.0 * 0.20)
        score += min(0.15, source_count / 4.0 * 0.15)

        if break_count > 0 and level.get("level_type") in {
            LevelType.FLIP_SUPPORT,
            LevelType.FLIP_RESISTANCE,
        }:
            score += 0.15

        if last_event is not None and last_event.get("level_id") == level.get("level_id"):
            if last_event.get("event_type") in {
                SREventType.LEVEL_REJECTED,
                SREventType.LEVEL_RETESTED,
                SREventType.LEVEL_FLIPPED,
            }:
                score += 0.15

        return clamp(score, 0.0, 1.0)

    def _secondary_layer_alignment_score(
        self,
        *,
        secondary_layer: Mapping[str, Any],
        selected_level: Mapping[str, Any],
    ) -> float:
        if not secondary_layer:
            return 0.35

        level_type = self._parse_level_type(selected_level.get("level_type"))

        if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
            ref_level = (
                secondary_layer.get("nearest_support")
                or secondary_layer.get("strongest_support")
            )
        elif level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            ref_level = (
                secondary_layer.get("nearest_resistance")
                or secondary_layer.get("strongest_resistance")
            )
        else:
            ref_level = None

        if not isinstance(ref_level, Mapping):
            return 0.35

        return clamp(
            0.55 * clamp(safe_float(ref_level.get("strength"), 0.0), 0.0, 1.0)
            + 0.25
            * self._status_quality_score(
                ref_level.get("status"),
                ref_level.get("level_type"),
            )
            + 0.20 * self._interaction_quality_score(ref_level, last_event=None),
            0.0,
            1.0,
        )

    def _proximity_score(
        self,
        *,
        current_price: float | None,
        level: Mapping[str, Any],
    ) -> float:
        if current_price is None or current_price <= 0:
            return 0.35

        level_price = safe_float(level.get("price"), 0.0)
        if level_price <= 0:
            return 0.35

        distance_pct = abs(current_price - level_price) / level_price
        if distance_pct >= self._p.max_distance_to_level_pct:
            return 0.0

        upper_bound = safe_float(level.get("upper_bound"), 0.0)
        lower_bound = safe_float(level.get("lower_bound"), 0.0)

        zone_penalty = 0.0
        if upper_bound > 0 and lower_bound > 0 and upper_bound >= lower_bound:
            zone_width_pct = (upper_bound - lower_bound) / level_price
            if zone_width_pct > self._p.max_zone_width_pct:
                zone_penalty = min(0.30, zone_width_pct)

        raw = 1.0 - (
            distance_pct / max(self._p.max_distance_to_level_pct, 1e-9)
        )
        return clamp(raw - zone_penalty, 0.0, 1.0)

    def _level_selection_score(
        self,
        *,
        level: Mapping[str, Any],
        current_price: float | None,
    ) -> float:
        score = 0.0
        score += 0.34 * clamp(safe_float(level.get("strength"), 0.0), 0.0, 1.0)
        score += 0.22 * self._status_quality_score(
            level.get("status"),
            level.get("level_type"),
        )
        score += 0.20 * self._interaction_quality_score(level, last_event=None)
        score += 0.18 * self._proximity_score(current_price=current_price, level=level)

        if level.get("level_type") in {
            LevelType.FLIP_SUPPORT,
            LevelType.FLIP_RESISTANCE,
        }:
            score += 0.06

        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reasons / signal build
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        *,
        sr: SupportResistanceContext,
        selected_level: Mapping[str, Any],
        side: SignalSide,
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.append("support_resistance_long_reaction")
        elif side == SignalSide.SHORT:
            reasons.append("support_resistance_short_reaction")

        level_type = selected_level.get("level_type")
        status = selected_level.get("status")
        layer = selected_level.get("layer")

        reasons.append(f"level_layer_{enum_value(layer)}")
        reasons.append(f"level_type_{enum_value(level_type)}")
        reasons.append(f"level_status_{enum_value(status)}")

        if level_type in {LevelType.FLIP_SUPPORT, LevelType.FLIP_RESISTANCE}:
            reasons.append("flip_level_context")

        if int(safe_float(selected_level.get("touch_count"), 0.0)) > 0:
            reasons.append("level_has_touches")

        if int(safe_float(selected_level.get("rejection_count"), 0.0)) >= (
            self._p.min_rejection_count_for_reaction
        ):
            reasons.append("level_has_rejections")

        if int(safe_float(selected_level.get("retest_count"), 0.0)) >= (
            self._p.min_retest_count_for_retest_entry
        ):
            reasons.append("level_has_retests")

        if int(safe_float(selected_level.get("source_count"), 0.0)) > 0:
            reasons.append("level_has_swing_sources")

        if sr.last_event is not None:
            event_type = sr.last_event.get("event_type")
            if isinstance(event_type, SREventType):
                reasons.append(f"last_sr_event_{event_type.value}")

            if sr.last_event.get("level_id") == selected_level.get("level_id"):
                reasons.append("last_event_matches_selected_level")

        return reasons

    def _build_signal(
        self,
        *,
        context: SignalContext,
        sr: SupportResistanceContext,
        selected_level: Mapping[str, Any],
        score: float,
        confidence: float,
        reasons: list[str],
        freshness_filter: FilterResult | None,
    ) -> StrategySignal:
        side = self._resolve_side(
            level=selected_level,
            last_event=sr.last_event,
        )
        last_event = sr.last_event

        level_type = selected_level.get("level_type")
        level_status = selected_level.get("status")
        last_event_type = last_event.get("event_type") if last_event is not None else None

        analytics_metadata = self._build_analytics_source_metadata(
            module_name=self.analytics_module_name,
            payload=sr.raw,
            selected_entity=selected_level,
            extra={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.support_resistance",
                "support_resistance_timeframe": sr.timeframe,
                "support_resistance_last_update": (
                    sr.last_update.isoformat() if sr.last_update else None
                ),
                "support_resistance_last_price": sr.last_price,
                "level_id": selected_level.get("level_id"),
                "level_layer": enum_value(selected_level.get("layer")),
                "level_type": enum_value(level_type),
                "level_status": enum_value(level_status),
                "level_price": safe_float(selected_level.get("price"), 0.0),
                "level_upper_bound": safe_float(selected_level.get("upper_bound"), 0.0),
                "level_lower_bound": safe_float(selected_level.get("lower_bound"), 0.0),
                "level_strength": safe_float(selected_level.get("strength"), 0.0),
                "level_touch_count": int(
                    safe_float(selected_level.get("touch_count"), 0.0)
                ),
                "level_rejection_count": int(
                    safe_float(selected_level.get("rejection_count"), 0.0)
                ),
                "level_break_count": int(
                    safe_float(selected_level.get("break_count"), 0.0)
                ),
                "level_retest_count": int(
                    safe_float(selected_level.get("retest_count"), 0.0)
                ),
                "level_source_count": int(
                    safe_float(selected_level.get("source_count"), 0.0)
                ),
                "level_source_swing_ids": list(
                    selected_level.get("source_swing_ids", []) or []
                ),
                "level_source_prices": list(selected_level.get("source_prices", []) or []),
                "level_created_at": self._datetime_to_iso(
                    selected_level.get("created_at")
                ),
                "level_updated_at": self._datetime_to_iso(
                    selected_level.get("updated_at")
                ),
                "level_broken_at": self._datetime_to_iso(
                    selected_level.get("broken_at")
                ),
                "level_flipped_at": self._datetime_to_iso(
                    selected_level.get("flipped_at")
                ),
                "level_last_tested_at": self._datetime_to_iso(
                    selected_level.get("last_tested_at")
                ),
                "level_last_rejected_at": self._datetime_to_iso(
                    selected_level.get("last_rejected_at")
                ),
                "level_last_broken_at": self._datetime_to_iso(
                    selected_level.get("last_broken_at")
                ),
                "level_last_retested_at": self._datetime_to_iso(
                    selected_level.get("last_retested_at")
                ),
                "last_sr_event_id": (
                    last_event.get("event_id") if last_event is not None else None
                ),
                "last_sr_event_type": enum_value(last_event_type),
                "last_sr_event_confidence": (
                    safe_float(last_event.get("confidence"), 0.0)
                    if last_event is not None
                    else None
                ),
                "last_sr_event_reference_price": (
                    last_event.get("reference_price") if last_event is not None else None
                ),
                "last_sr_event_matches_level": bool(
                    last_event
                    and last_event.get("level_id") == selected_level.get("level_id")
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
            setup_type=self._resolve_setup_type(selected_level, last_event),
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=confidence_to_strength(confidence),
            confidence_grade=confidence_to_grade(confidence),
            status=SignalStatus.NEW,
            trigger_type=self._resolve_trigger_type(selected_level, last_event),
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._resolve_priority(
                selected_level=selected_level,
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
        signal.add_source_feature("analytics.price_action.support_resistance")
        signal.add_source_feature("price_action.support_resistance")
        signal.add_source_feature("support_resistance")

        if freshness_filter is not None:
            signal.add_filter_result(freshness_filter)

        regime_filter = self._build_regime_filter(context=context, side=side)
        if regime_filter is not None:
            signal.add_filter_result(regime_filter)

        signal.validate()
        return signal

    def _resolve_setup_type(
        self,
        level: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> SetupType:
        event_type = last_event.get("event_type") if last_event is not None else None

        if event_type == SREventType.LEVEL_BROKEN:
            return SetupType.BREAKOUT

        if event_type in {
            SREventType.LEVEL_REJECTED,
            SREventType.LEVEL_RETESTED,
        }:
            return SetupType.REVERSAL

        if event_type == SREventType.LEVEL_FLIPPED:
            return SetupType.RETEST

        level_type = self._parse_level_type(level.get("level_type"))
        if level_type in {LevelType.FLIP_SUPPORT, LevelType.FLIP_RESISTANCE}:
            return SetupType.RETEST

        return SetupType.REVERSAL

    def _resolve_trigger_type(
        self,
        level: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
    ) -> TriggerType:
        event_type = last_event.get("event_type") if last_event is not None else None

        if event_type in {
            SREventType.LEVEL_REJECTED,
            SREventType.LEVEL_BROKEN,
            SREventType.LEVEL_FLIPPED,
            SREventType.LEVEL_RETESTED,
        }:
            return TriggerType.PRIMARY

        if event_type == SREventType.LEVEL_TOUCHED:
            return TriggerType.CONFIRMATION

        return TriggerType.DERIVED

    def _resolve_priority(
        self,
        *,
        selected_level: Mapping[str, Any],
        last_event: Mapping[str, Any] | None,
        confidence: float,
        score: float,
    ) -> SignalPriority:
        event_type = last_event.get("event_type") if last_event is not None else None
        level_strength = clamp(safe_float(selected_level.get("strength"), 0.0), 0.0, 1.0)

        if (
            event_type in {
                SREventType.LEVEL_REJECTED,
                SREventType.LEVEL_FLIPPED,
                SREventType.LEVEL_RETESTED,
            }
            and confidence >= 0.72
            and score >= 0.65
        ):
            return SignalPriority.HIGH

        if event_type == SREventType.LEVEL_BROKEN and confidence >= 0.75:
            return SignalPriority.HIGH

        if confidence >= 0.85 and score >= 0.78 and level_strength >= 0.70:
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _resolve_current_price(
        self,
        *,
        context: SignalContext,
        sr: SupportResistanceContext,
    ) -> float | None:
        candidates = (
            getattr(context, "last_price", None),
            getattr(context, "current_price", None),
            getattr(context, "price", None),
            self._get_context_feature(context, "last_price"),
            self._get_context_feature(context, "current_price"),
            self._get_context_feature(context, "price"),
            sr.last_price,
        )

        for candidate in candidates:
            value = safe_float(candidate, 0.0)
            if value > 0:
                return value

        return None

    def _level_type_to_reaction_side(self, level_type: LevelType | None) -> SignalSide:
        if level_type in {LevelType.SUPPORT, LevelType.FLIP_SUPPORT}:
            return SignalSide.LONG

        if level_type in {LevelType.RESISTANCE, LevelType.FLIP_RESISTANCE}:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _status_from_event_type(self, event_type: Any) -> LevelStatus:
        parsed = self._parse_sr_event_type(event_type)

        if parsed in {
            SREventType.LEVEL_CREATED,
            SREventType.LEVEL_MERGED,
            SREventType.LEVEL_TOUCHED,
            SREventType.LEVEL_REJECTED,
            SREventType.LEVEL_FLIPPED,
            SREventType.LEVEL_RETESTED,
        }:
            return LevelStatus.ACTIVE

        if parsed == SREventType.LEVEL_BROKEN:
            return LevelStatus.BROKEN

        return LevelStatus.ACTIVE

    def _datetime_to_iso(self, value: Any) -> str | None:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return None

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

    def _parse_level_type(self, value: Any) -> LevelType | None:
        raw = enum_value(value)

        mapping = {
            "support": LevelType.SUPPORT,
            "resistance": LevelType.RESISTANCE,
            "flip_support": LevelType.FLIP_SUPPORT,
            "flipped_support": LevelType.FLIP_SUPPORT,
            "support_flip": LevelType.FLIP_SUPPORT,
            "flip_resistance": LevelType.FLIP_RESISTANCE,
            "flipped_resistance": LevelType.FLIP_RESISTANCE,
            "resistance_flip": LevelType.FLIP_RESISTANCE,
        }

        if raw in mapping:
            return mapping[raw]

        try:
            return LevelType(raw)
        except Exception:
            return None

    def _parse_level_status(self, value: Any) -> LevelStatus | None:
        raw = enum_value(value)

        mapping = {
            "active": LevelStatus.ACTIVE,
            "broken": LevelStatus.BROKEN,
            "inactive": LevelStatus.INACTIVE,
        }

        if raw in mapping:
            return mapping[raw]

        try:
            return LevelStatus(raw)
        except Exception:
            return None

    def _parse_sr_event_type(self, value: Any) -> SREventType | None:
        raw = enum_value(value)

        mapping = {
            "level_created": SREventType.LEVEL_CREATED,
            "created": SREventType.LEVEL_CREATED,
            "level_merged": SREventType.LEVEL_MERGED,
            "merged": SREventType.LEVEL_MERGED,
            "level_touched": SREventType.LEVEL_TOUCHED,
            "touched": SREventType.LEVEL_TOUCHED,
            "touch": SREventType.LEVEL_TOUCHED,
            "level_rejected": SREventType.LEVEL_REJECTED,
            "rejected": SREventType.LEVEL_REJECTED,
            "rejection": SREventType.LEVEL_REJECTED,
            "level_broken": SREventType.LEVEL_BROKEN,
            "broken": SREventType.LEVEL_BROKEN,
            "break": SREventType.LEVEL_BROKEN,
            "breakout": SREventType.LEVEL_BROKEN,
            "level_flipped": SREventType.LEVEL_FLIPPED,
            "flipped": SREventType.LEVEL_FLIPPED,
            "flip": SREventType.LEVEL_FLIPPED,
            "level_retested": SREventType.LEVEL_RETESTED,
            "retested": SREventType.LEVEL_RETESTED,
            "retest": SREventType.LEVEL_RETESTED,
        }

        if raw in mapping:
            return mapping[raw]

        try:
            return SREventType(raw)
        except Exception:
            return None