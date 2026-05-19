from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, cast
from uuid import uuid4

from analytics.price_action.enums import (
    MarketBias,
    StructureEventType,
    StructureLayer,
    SwingType,
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
class MarketStructureStrategyParams:
    """
    Local params for MarketStructureStrategy.

    Runtime gates such as enabled/symbols/timeframes/min_score/min_confidence
    stay in StrategyConfig / StrategyDefinitionConfig.runtime. These params
    only define how this strategy consumes analytics.price_action.market_structure.
    """

    strategy_name: str = "market_structure_strategy"

    prefer_external_layer: bool = True
    require_alignment: bool = False
    require_break_event: bool = True
    require_primary_layer_eligible: bool = True

    allow_bos_entries: bool = True
    allow_choch_reversals: bool = True
    allow_mss_reversals: bool = True
    allow_bias_continuation_fallback: bool = True

    reverse_on_choch: bool = True
    reverse_on_mss: bool = True

    require_recent_swing_context: bool = False
    require_reference_swing_for_break: bool = False
    allow_breakout_state_without_break_event: bool = False

    min_layer_confidence: float = 0.45
    min_alignment_score: float = 0.30
    min_break_confidence: float = 0.45
    min_trend_strength: float = 0.20
    min_swing_strength: float = 0.25
    min_swing_progression_score: float = 0.20
    min_break_distance_pct: float = 0.0002

    primary_bias_weight: float = 0.22
    secondary_bias_weight: float = 0.12
    break_event_weight: float = 0.24
    alignment_weight: float = 0.14
    trend_strength_weight: float = 0.10
    swing_context_weight: float = 0.10
    breakout_state_weight: float = 0.04
    regime_alignment_weight: float = 0.04

    emit_signal_events: bool = False
    signal_event_name: str = "strategy.price_action.market_structure.signal"

    freshness_feature_names: tuple[str, ...] = (
        "analytics.price_action",
        "analytics.price_action.market_structure",
        "price_action.market_structure",
        "market_structure",
    )

    def validate(self) -> None:
        PriceActionStrategyBase.validate_bounded_fields(
            instance=self,
            field_names=(
                "min_layer_confidence",
                "min_alignment_score",
                "min_break_confidence",
                "min_trend_strength",
                "min_swing_strength",
                "min_swing_progression_score",
                "primary_bias_weight",
                "secondary_bias_weight",
                "break_event_weight",
                "alignment_weight",
                "trend_strength_weight",
                "swing_context_weight",
                "breakout_state_weight",
                "regime_alignment_weight",
            ),
            minimum=0.0,
            maximum=1.0,
        )

        if self.min_break_distance_pct < 0:
            raise ValueError("min_break_distance_pct must be >= 0")

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
class SwingContext:
    swing_id: str | None = None
    swing_type: SwingType | None = None
    timestamp: datetime | None = None
    price: float | None = None
    layer: StructureLayer | None = None
    index: int | None = None
    candle_open: float | None = None
    candle_high: float | None = None
    candle_low: float | None = None
    candle_close: float | None = None
    strength: float = 0.0
    is_confirmed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(
            self.swing_id
            and self.swing_type is not None
            and self.price is not None
        )


@dataclass(slots=True)
class StructureEventContext:
    event_id: str | None = None
    event_type: StructureEventType | None = None
    timestamp: datetime | None = None
    price: float | None = None
    layer: StructureLayer | None = None
    direction: MarketBias = MarketBias.UNKNOWN
    swing_id: str | None = None
    reference_price: float | None = None
    reference_swing_id: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_break(self) -> bool:
        return self.event_type in {
            StructureEventType.BOS,
            StructureEventType.CHOCH,
            StructureEventType.MSS,
        }

    @property
    def valid(self) -> bool:
        return self.event_type is not None and self.price is not None


@dataclass(slots=True)
class MarketStructureContextView:
    """
    Normalized view of analytics.price_action.market_structure.MarketStructureState.
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
    mtf_alignment: dict[str, Any] = field(default_factory=dict)

    last_break_event: StructureEventContext = field(default_factory=StructureEventContext)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_feature: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class MarketStructureStrategy(PriceActionStrategyBase):
    """
    Strategy wrapper around analytics.price_action.market_structure.

    Aligned with the current MarketStructureAnalyzer / MarketStructureState contract:
    - consumes MarketStructureState from PriceActionCompositeState or direct module feature;
    - validates futures scope through PriceActionStrategyBase;
    - uses internal/external StructureLayerState;
    - uses last_swing_high / previous_swing_high / last_swing_low / previous_swing_low;
    - uses HH/HL/LH/LL sequence context and BOS/CHOCH/MSS break context;
    - uses MultiTimeframeAlignment;
    - preserves analytics source metadata in StrategySignal.metadata.
    """

    analytics_module_name = "market_structure"

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

    @property
    def _p(self) -> MarketStructureStrategyParams:
        return cast(MarketStructureStrategyParams, self.params)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        try:
            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked

            structure = self._extract_market_structure_snapshot(context)
            if structure.symbol is None and not structure.external and not structure.internal:
                return self._rejected_evaluation(
                    context=context,
                    reason="market_structure_snapshot_missing",
                )

            freshness_filter = self._build_freshness_filter(
                context=context,
                filter_name="market_structure_freshness",
                module_name=self.analytics_module_name,
                analytics_payload=structure.raw,
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
                    metadata={
                        "analytics_module": self.analytics_module_name,
                        "analytics_source_feature": structure.source_feature,
                        "alignment_score": structure.mtf_alignment.get("alignment_score"),
                        "last_break_event_type": enum_value(
                            structure.last_break_event.event_type
                        ),
                        "last_break_direction": enum_value(
                            structure.last_break_event.direction
                        ),
                    },
                )

            primary_layer = self._select_primary_layer(structure)
            secondary_layer = self._select_secondary_layer(structure)

            score = self._compute_score(
                context=context,
                structure=structure,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )
            confidence = self._compute_confidence(
                context=context,
                structure=structure,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
            )
            reasons = self._build_reasons(
                structure=structure,
                side=side,
                primary_layer=primary_layer,
                secondary_layer=secondary_layer,
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
                metadata={
                    "analytics_module": self.analytics_module_name,
                    "analytics_source_feature": structure.source_feature,
                },
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
    # Extraction / normalization
    # ------------------------------------------------------------------

    def _extract_market_structure_snapshot(
        self,
        context: SignalContext,
    ) -> MarketStructureContextView:
        payload = self._extract_price_action_module(
            context,
            self.analytics_module_name,
            aliases=(
                "market_structure",
                "structure",
                "price_action.market_structure",
                "analytics.price_action.market_structure",
            ),
            require_scope_match=True,
        )
        if payload:
            return self._normalize_structure_snapshot(payload)

        candidates: list[Any] = [
            self._mapping_or_empty(getattr(context, "price_action", None)).get(
                "market_structure"
            ),
            self._mapping_or_empty(getattr(context, "price_action", None)).get(
                "structure"
            ),
            self._get_context_feature(context, "price_action.market_structure"),
            self._get_context_feature(context, "market_structure"),
            self._get_context_feature(context, "analytics.price_action.market_structure"),
        ]
        for candidate in candidates:
            normalized = self._normalize_structure_snapshot(candidate)
            if normalized.symbol is not None or normalized.external or normalized.internal:
                return normalized

        return MarketStructureContextView()

    def _normalize_structure_snapshot(self, payload: Any) -> MarketStructureContextView:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return MarketStructureContextView()

        state = self._normalize_state_payload(payload_mapping)
        if not state:
            return MarketStructureContextView()

        internal = self._normalize_layer(state.get("internal"), StructureLayer.INTERNAL)
        external = self._normalize_layer(state.get("external"), StructureLayer.EXTERNAL)
        mtf_alignment = self._normalize_alignment(state.get("mtf_alignment"))
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

        return MarketStructureContextView(
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
            mtf_alignment=mtf_alignment,
            last_break_event=self._extract_last_break_event(
                internal=internal,
                external=external,
            ),
            metadata=metadata,
            source_feature=state.get("_source_feature"),
            raw=dict(state),
        )

    def _normalize_layer(
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
            "bias": self._parse_market_bias(payload_mapping.get("bias")),
            "confidence": clamp(
                safe_float(payload_mapping.get("confidence"), 0.0),
                0.0,
                1.0,
            ),
            "trend_strength": clamp(
                safe_float(payload_mapping.get("trend_strength"), 0.0),
                0.0,
                1.0,
            ),
            "in_breakout": safe_bool(payload_mapping.get("in_breakout"), False),
            "last_swing_high": self._normalize_swing(
                payload_mapping.get("last_swing_high"),
                default_layer,
                SwingType.HIGH,
            ),
            "previous_swing_high": self._normalize_swing(
                payload_mapping.get("previous_swing_high"),
                default_layer,
                SwingType.HIGH,
            ),
            "last_swing_low": self._normalize_swing(
                payload_mapping.get("last_swing_low"),
                default_layer,
                SwingType.LOW,
            ),
            "previous_swing_low": self._normalize_swing(
                payload_mapping.get("previous_swing_low"),
                default_layer,
                SwingType.LOW,
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
            "swing_count": int(safe_float(payload_mapping.get("swing_count"), 0.0)),
            "event_count": int(safe_float(payload_mapping.get("event_count"), 0.0)),
            "sequence": [
                str(item)
                for item in list(payload_mapping.get("sequence", []) or [])
            ],
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _normalize_swing(
        self,
        payload: Any,
        default_layer: StructureLayer,
        default_swing_type: SwingType,
    ) -> SwingContext | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        price = (
            safe_float(payload_mapping.get("price"), 0.0)
            if payload_mapping.get("price") is not None
            else None
        )
        index = (
            int(safe_float(payload_mapping.get("index"), 0.0))
            if payload_mapping.get("index") is not None
            else None
        )

        return SwingContext(
            swing_id=payload_mapping.get("swing_id"),
            swing_type=self._parse_swing_type(payload_mapping.get("swing_type"))
            or default_swing_type,
            timestamp=parse_datetime(payload_mapping.get("timestamp")),
            price=price,
            layer=self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            index=index,
            candle_open=(
                safe_float(payload_mapping.get("candle_open"), 0.0)
                if payload_mapping.get("candle_open") is not None
                else None
            ),
            candle_high=(
                safe_float(payload_mapping.get("candle_high"), 0.0)
                if payload_mapping.get("candle_high") is not None
                else None
            ),
            candle_low=(
                safe_float(payload_mapping.get("candle_low"), 0.0)
                if payload_mapping.get("candle_low") is not None
                else None
            ),
            candle_close=(
                safe_float(payload_mapping.get("candle_close"), 0.0)
                if payload_mapping.get("candle_close") is not None
                else None
            ),
            strength=clamp(
                safe_float(payload_mapping.get("strength"), 0.0),
                0.0,
                1.0,
            ),
            is_confirmed=safe_bool(payload_mapping.get("is_confirmed"), False),
            metadata=dict(payload_mapping.get("metadata", {}) or {}),
        )

    def _normalize_structure_event(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> StructureEventContext | None:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return None

        return StructureEventContext(
            event_id=payload_mapping.get("event_id"),
            event_type=self._parse_structure_event_type(
                payload_mapping.get("event_type")
            ),
            timestamp=parse_datetime(payload_mapping.get("timestamp")),
            price=(
                safe_float(payload_mapping.get("price"), 0.0)
                if payload_mapping.get("price") is not None
                else None
            ),
            layer=self._parse_structure_layer(payload_mapping.get("layer"))
            or default_layer,
            direction=self._parse_market_bias(payload_mapping.get("direction")),
            swing_id=payload_mapping.get("swing_id"),
            reference_price=(
                safe_float(payload_mapping.get("reference_price"), 0.0)
                if payload_mapping.get("reference_price") is not None
                else None
            ),
            reference_swing_id=payload_mapping.get("reference_swing_id"),
            confidence=clamp(
                safe_float(payload_mapping.get("confidence"), 0.0),
                0.0,
                1.0,
            ),
            metadata=dict(payload_mapping.get("metadata", {}) or {}),
        )

    def _normalize_alignment(self, payload: Any) -> dict[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)
        if not payload_mapping:
            return {}

        return {
            "higher_timeframe": payload_mapping.get("higher_timeframe"),
            "higher_timeframe_bias": self._parse_market_bias(
                payload_mapping.get("higher_timeframe_bias")
            ),
            "higher_timeframe_confidence": clamp(
                safe_float(payload_mapping.get("higher_timeframe_confidence"), 0.0),
                0.0,
                1.0,
            ),
            "internal_bias_aligned": safe_bool(
                payload_mapping.get("internal_bias_aligned"),
                False,
            ),
            "external_bias_aligned": safe_bool(
                payload_mapping.get("external_bias_aligned"),
                False,
            ),
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
            "metadata": dict(payload_mapping.get("metadata", {}) or {}),
        }

    def _extract_last_break_event(
        self,
        *,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> StructureEventContext:
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        candidates: list[StructureEventContext] = []

        for layer in (external, internal):
            for key in ("last_bos", "last_choch", "last_mss"):
                event = layer.get(key)
                if isinstance(event, StructureEventContext) and event.is_break:
                    candidates.append(event)

        if not candidates:
            return StructureEventContext()

        def _sort_key(event: StructureEventContext) -> datetime:
            ts = event.timestamp
            if ts is None:
                return epoch
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)

        candidates.sort(key=_sort_key, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Direction / eligibility
    # ------------------------------------------------------------------

    def _select_primary_layer(
        self,
        structure: MarketStructureContextView,
    ) -> dict[str, Any]:
        return structure.external if self._p.prefer_external_layer else structure.internal

    def _select_secondary_layer(
        self,
        structure: MarketStructureContextView,
    ) -> dict[str, Any]:
        return structure.internal if self._p.prefer_external_layer else structure.external

    def _resolve_side(
        self,
        context: SignalContext,
        structure: MarketStructureContextView,
    ) -> SignalSide:
        primary_layer = self._select_primary_layer(structure)
        secondary_layer = self._select_secondary_layer(structure)
        last_break = structure.last_break_event

        if self._p.require_alignment and not self._alignment_eligible(structure):
            return SignalSide.UNKNOWN

        if last_break.is_break:
            if not self._break_event_eligible(last_break, structure=structure):
                return SignalSide.UNKNOWN

            side = self._side_from_break_event(last_break)
            if side == SignalSide.UNKNOWN:
                return SignalSide.UNKNOWN

            if self._p.require_primary_layer_eligible and not self._layer_eligible(
                primary_layer
            ):
                return SignalSide.UNKNOWN

            if not self._side_consistent_with_structure(
                side,
                primary_layer,
                secondary_layer,
                structure,
            ):
                return SignalSide.UNKNOWN

            return side

        if self._p.require_break_event and not self._p.allow_breakout_state_without_break_event:
            return SignalSide.UNKNOWN

        if self._p.allow_breakout_state_without_break_event:
            if safe_bool(primary_layer.get("in_breakout"), False) and self._layer_eligible(
                primary_layer
            ):
                return self._bias_to_side(primary_layer.get("bias", MarketBias.UNKNOWN))

        if not self._p.allow_bias_continuation_fallback:
            return SignalSide.UNKNOWN

        if not self._layer_eligible(primary_layer):
            return SignalSide.UNKNOWN

        primary_side = self._bias_to_side(primary_layer.get("bias", MarketBias.UNKNOWN))
        if primary_side == SignalSide.UNKNOWN:
            return SignalSide.UNKNOWN

        if not self._side_consistent_with_structure(
            primary_side,
            primary_layer,
            secondary_layer,
            structure,
        ):
            return SignalSide.UNKNOWN

        return primary_side

    def _break_event_eligible(
        self,
        event: StructureEventContext,
        *,
        structure: MarketStructureContextView,
    ) -> bool:
        if not event.is_break:
            return False

        if event.confidence < self._p.min_break_confidence:
            return False

        if event.event_type == StructureEventType.BOS and not self._p.allow_bos_entries:
            return False

        if (
            event.event_type == StructureEventType.CHOCH
            and not self._p.allow_choch_reversals
        ):
            return False

        if event.event_type == StructureEventType.MSS and not self._p.allow_mss_reversals:
            return False

        if self._p.require_reference_swing_for_break and not event.reference_swing_id:
            return False

        if self._p.min_break_distance_pct > 0 and event.reference_price and event.price:
            distance_pct = abs(event.price - event.reference_price) / event.reference_price
            if distance_pct < self._p.min_break_distance_pct:
                return False

        if self._p.require_recent_swing_context:
            event_layer = (
                structure.external
                if event.layer == StructureLayer.EXTERNAL
                else structure.internal
            )
            if not self._layer_has_recent_swings(event_layer):
                return False

        return True

    def _side_from_break_event(self, event: StructureEventContext) -> SignalSide:
        if event.event_type == StructureEventType.BOS:
            return self._bias_to_side(event.direction)

        if event.event_type == StructureEventType.CHOCH:
            if not self._p.reverse_on_choch:
                return SignalSide.UNKNOWN
            return self._bias_to_side(event.direction)

        if event.event_type == StructureEventType.MSS:
            if not self._p.reverse_on_mss:
                return SignalSide.UNKNOWN
            return self._bias_to_side(event.direction)

        return SignalSide.UNKNOWN

    def _alignment_eligible(self, structure: MarketStructureContextView) -> bool:
        mtf = structure.mtf_alignment
        if not mtf:
            return False

        if (
            clamp(safe_float(mtf.get("alignment_score"), 0.0), 0.0, 1.0)
            < self._p.min_alignment_score
        ):
            return False

        return safe_bool(mtf.get("internal_with_external_aligned"), False) or safe_bool(
            mtf.get("external_bias_aligned"),
            False,
        )

    def _layer_eligible(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False

        confidence = clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0)
        trend_strength = clamp(
            safe_float(layer.get("trend_strength"), 0.0),
            0.0,
            1.0,
        )

        if confidence < self._p.min_layer_confidence:
            return False

        if trend_strength < self._p.min_trend_strength:
            return False

        if self._p.require_recent_swing_context and not self._layer_has_recent_swings(
            layer
        ):
            return False

        return True

    def _layer_has_recent_swings(self, layer: Mapping[str, Any]) -> bool:
        last_high = layer.get("last_swing_high")
        last_low = layer.get("last_swing_low")
        return (
            isinstance(last_high, SwingContext)
            and last_high.valid
            and isinstance(last_low, SwingContext)
            and last_low.valid
        )

    def _side_consistent_with_structure(
        self,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
        structure: MarketStructureContextView,
    ) -> bool:
        primary_bias = primary_layer.get("bias", MarketBias.UNKNOWN)
        primary_side = self._bias_to_side(primary_bias)

        if primary_side not in {SignalSide.UNKNOWN, side}:
            return False

        if self._p.require_alignment:
            mtf = structure.mtf_alignment
            if side == SignalSide.LONG and mtf.get("higher_timeframe_bias") == MarketBias.BEARISH:
                return False
            if side == SignalSide.SHORT and mtf.get("higher_timeframe_bias") == MarketBias.BULLISH:
                return False

        secondary_bias = secondary_layer.get("bias", MarketBias.UNKNOWN)
        secondary_side = self._bias_to_side(secondary_bias)

        if secondary_side not in {SignalSide.UNKNOWN, side}:
            alignment_score = clamp(
                safe_float(structure.mtf_alignment.get("alignment_score"), 0.0),
                0.0,
                1.0,
            )
            if alignment_score < self._p.min_alignment_score:
                return False

        return True

    # ------------------------------------------------------------------
    # Score / confidence
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        context: SignalContext,
        structure: MarketStructureContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> float:
        score = 0.0

        score += self._p.primary_bias_weight * self._bias_alignment_score(
            primary_layer,
            side,
        )
        score += self._p.secondary_bias_weight * self._bias_alignment_score(
            secondary_layer,
            side,
        )
        score += self._p.break_event_weight * self._break_event_score(
            structure.last_break_event,
            side,
        )
        score += self._p.alignment_weight * self._mtf_alignment_score(structure, side)
        score += self._p.trend_strength_weight * clamp(
            safe_float(primary_layer.get("trend_strength"), 0.0),
            0.0,
            1.0,
        )
        score += self._p.swing_context_weight * self._swing_context_score(
            primary_layer,
            side,
        )

        if safe_bool(primary_layer.get("in_breakout"), False):
            score += self._p.breakout_state_weight

        score += self._p.regime_alignment_weight * self._regime_alignment_score(
            context=context,
            side=side,
        )

        return clamp(score, 0.0, 1.0)

    def _compute_confidence(
        self,
        context: SignalContext,
        structure: MarketStructureContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> float:
        components: list[float] = [
            clamp(safe_float(primary_layer.get("confidence"), 0.0), 0.0, 1.0),
            clamp(safe_float(primary_layer.get("trend_strength"), 0.0), 0.0, 1.0),
            self._bias_alignment_score(primary_layer, side),
            self._swing_context_score(primary_layer, side),
        ]

        if secondary_layer:
            components.append(self._bias_alignment_score(secondary_layer, side))

        if structure.last_break_event.is_break:
            components.append(self._break_event_score(structure.last_break_event, side))

        alignment_score = self._mtf_alignment_score(structure, side)
        if alignment_score > 0:
            components.append(alignment_score)

        components.append(self._regime_alignment_score(context=context, side=side))

        return clamp(sum(components) / len(components), 0.0, 1.0)

    def _bias_alignment_score(self, layer: Mapping[str, Any], side: SignalSide) -> float:
        if not layer:
            return 0.0

        layer_side = self._bias_to_side(layer.get("bias", MarketBias.UNKNOWN))
        confidence = clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0)
        trend_strength = clamp(
            safe_float(layer.get("trend_strength"), 0.0),
            0.0,
            1.0,
        )

        if layer_side == side:
            return clamp((confidence * 0.60) + (trend_strength * 0.40), 0.0, 1.0)

        if layer_side == SignalSide.UNKNOWN:
            return 0.25 * confidence

        return 0.0

    def _break_event_score(self, event: StructureEventContext, side: SignalSide) -> float:
        if not event.is_break:
            return 0.0

        if self._side_from_break_event(event) != side:
            return 0.0

        base = event.confidence

        if event.event_type == StructureEventType.BOS:
            base += 0.08
        elif event.event_type == StructureEventType.CHOCH:
            base += 0.10
        elif event.event_type == StructureEventType.MSS:
            base += 0.12

        if event.reference_swing_id:
            base += 0.04

        if event.swing_id:
            base += 0.03

        if event.reference_price and event.price:
            distance_pct = abs(event.price - event.reference_price) / event.reference_price
            base += min(0.08, distance_pct * 10.0)

        return clamp(base, 0.0, 1.0)

    def _mtf_alignment_score(
        self,
        structure: MarketStructureContextView,
        side: SignalSide,
    ) -> float:
        mtf = structure.mtf_alignment
        if not mtf:
            return 0.0

        score = clamp(safe_float(mtf.get("alignment_score"), 0.0), 0.0, 1.0)
        htf_bias = mtf.get("higher_timeframe_bias", MarketBias.UNKNOWN)
        htf_confidence = clamp(
            safe_float(mtf.get("higher_timeframe_confidence"), 0.0),
            0.0,
            1.0,
        )

        if self._bias_to_side(htf_bias) == side:
            score = max(score, htf_confidence)
        elif self._bias_to_side(htf_bias) not in {SignalSide.UNKNOWN, side}:
            score *= 0.50

        if safe_bool(mtf.get("internal_with_external_aligned"), False):
            score = max(score, self._p.min_alignment_score)

        return clamp(score, 0.0, 1.0)

    def _swing_context_score(self, layer: Mapping[str, Any], side: SignalSide) -> float:
        if not layer:
            return 0.0

        last_high = layer.get("last_swing_high")
        previous_high = layer.get("previous_swing_high")
        last_low = layer.get("last_swing_low")
        previous_low = layer.get("previous_swing_low")

        score = 0.0

        swing_strengths = [
            swing.strength
            for swing in (last_high, previous_high, last_low, previous_low)
            if isinstance(swing, SwingContext) and swing.valid
        ]
        if swing_strengths:
            score += min(0.45, sum(swing_strengths) / len(swing_strengths) * 0.45)

        progression = self._swing_progression_score(
            last_high=last_high,
            previous_high=previous_high,
            last_low=last_low,
            previous_low=previous_low,
            side=side,
        )
        score += 0.40 * progression

        sequence = [
            str(item).lower()
            for item in list(layer.get("sequence", []) or [])
        ]

        if side == SignalSide.LONG and any(
            item in {"hh", "hl", "bos"} for item in sequence[-4:]
        ):
            score += 0.15

        if side == SignalSide.SHORT and any(
            item in {"lh", "ll", "bos"} for item in sequence[-4:]
        ):
            score += 0.15

        return clamp(score, 0.0, 1.0)

    def _swing_progression_score(
        self,
        *,
        last_high: Any,
        previous_high: Any,
        last_low: Any,
        previous_low: Any,
        side: SignalSide,
    ) -> float:
        if side == SignalSide.LONG:
            higher_high = self._swing_price_gt(last_high, previous_high)
            higher_low = self._swing_price_gt(last_low, previous_low)

            if higher_high and higher_low:
                return 1.0
            if higher_high or higher_low:
                return 0.55
            return 0.0

        if side == SignalSide.SHORT:
            lower_high = self._swing_price_lt(last_high, previous_high)
            lower_low = self._swing_price_lt(last_low, previous_low)

            if lower_high and lower_low:
                return 1.0
            if lower_high or lower_low:
                return 0.55
            return 0.0

        return 0.0

    def _swing_price_gt(self, current: Any, previous: Any) -> bool:
        return (
            isinstance(current, SwingContext)
            and isinstance(previous, SwingContext)
            and current.price is not None
            and previous.price is not None
            and current.price > previous.price
        )

    def _swing_price_lt(self, current: Any, previous: Any) -> bool:
        return (
            isinstance(current, SwingContext)
            and isinstance(previous, SwingContext)
            and current.price is not None
            and previous.price is not None
            and current.price < previous.price
        )

    # ------------------------------------------------------------------
    # Reasons / signal build
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        structure: MarketStructureContextView,
        side: SignalSide,
        primary_layer: Mapping[str, Any],
        secondary_layer: Mapping[str, Any],
    ) -> list[str]:
        reasons: list[str] = []

        if side == SignalSide.LONG:
            reasons.append("market_structure_bullish")
        elif side == SignalSide.SHORT:
            reasons.append("market_structure_bearish")

        primary_name = "external" if self._p.prefer_external_layer else "internal"
        secondary_name = "internal" if self._p.prefer_external_layer else "external"

        reasons.append(f"primary_layer_{primary_name}")
        reasons.append(f"{primary_name}_bias_{enum_value(primary_layer.get('bias'))}")

        secondary_side = self._bias_to_side(
            secondary_layer.get("bias", MarketBias.UNKNOWN)
        )
        if secondary_side == side:
            reasons.append(f"{secondary_name}_bias_confirmation")

        last_break = structure.last_break_event
        if last_break.is_break:
            reasons.append(f"last_break_{enum_value(last_break.event_type)}")
            reasons.append(f"break_layer_{enum_value(last_break.layer)}")
            if last_break.reference_swing_id:
                reasons.append("break_has_reference_swing")

        if safe_bool(primary_layer.get("in_breakout"), False):
            reasons.append("primary_layer_in_breakout")

        swing_score = self._swing_context_score(primary_layer, side)
        if swing_score >= self._p.min_swing_progression_score:
            reasons.append("swing_progression_supports_side")

        mtf_score = self._mtf_alignment_score(structure, side)
        if mtf_score >= self._p.min_alignment_score:
            reasons.append("mtf_alignment_supports_side")

        return reasons

    def _build_signal(
        self,
        context: SignalContext,
        structure: MarketStructureContextView,
        side: SignalSide,
        score: float,
        confidence: float,
        reasons: list[str],
        freshness_filter: FilterResult | None,
    ) -> StrategySignal:
        primary_layer = self._select_primary_layer(structure)
        secondary_layer = self._select_secondary_layer(structure)
        last_break = structure.last_break_event

        selected_entity = (
            self._event_context_to_metadata(last_break)
            if last_break.is_break
            else None
        )
        if selected_entity is None:
            selected_entity = self._layer_selected_entity(primary_layer, side)

        analytics_metadata = self._build_analytics_source_metadata(
            module_name=self.analytics_module_name,
            payload=structure.raw,
            selected_entity=selected_entity,
            extra={
                "signal_id": uuid4().hex,
                "module": self.name,
                "source": "analytics.price_action.market_structure",
                "market_structure_timeframe": structure.timeframe,
                "market_structure_last_update": (
                    structure.last_update.isoformat()
                    if structure.last_update
                    else None
                ),
                "market_structure_last_price": structure.last_price,
                "primary_layer": enum_value(primary_layer.get("layer")),
                "primary_bias": enum_value(primary_layer.get("bias")),
                "primary_confidence": safe_float(primary_layer.get("confidence"), 0.0),
                "primary_trend_strength": safe_float(
                    primary_layer.get("trend_strength"),
                    0.0,
                ),
                "primary_in_breakout": safe_bool(
                    primary_layer.get("in_breakout"),
                    False,
                ),
                "secondary_layer": enum_value(secondary_layer.get("layer")),
                "secondary_bias": enum_value(secondary_layer.get("bias")),
                "secondary_confidence": safe_float(
                    secondary_layer.get("confidence"),
                    0.0,
                ),
                "alignment_score": safe_float(
                    structure.mtf_alignment.get("alignment_score"),
                    0.0,
                ),
                "higher_timeframe": structure.mtf_alignment.get("higher_timeframe"),
                "higher_timeframe_bias": enum_value(
                    structure.mtf_alignment.get("higher_timeframe_bias")
                ),
                "higher_timeframe_confidence": safe_float(
                    structure.mtf_alignment.get("higher_timeframe_confidence"),
                    0.0,
                ),
                "last_break_type": enum_value(last_break.event_type),
                "last_break_direction": enum_value(last_break.direction),
                "last_break_confidence": last_break.confidence,
                "last_break_layer": enum_value(last_break.layer),
                "last_break_event_id": last_break.event_id,
                "last_break_swing_id": last_break.swing_id,
                "last_break_reference_swing_id": last_break.reference_swing_id,
                "swing_context_score": self._swing_context_score(
                    primary_layer,
                    side,
                ),
            },
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
            priority=self._resolve_priority(
                last_break=last_break,
                confidence=confidence,
                score=score,
                structure=structure,
            ),
            regime=self._resolve_market_regime(context),
            metadata=analytics_metadata,
        )

        for reason in reasons:
            signal.add_reason(reason)

        signal.add_source_feature("analytics.price_action")
        signal.add_source_feature("analytics.price_action.market_structure")
        signal.add_source_feature("price_action.market_structure")

        if freshness_filter is not None:
            signal.add_filter_result(freshness_filter)

        regime_filter = self._build_regime_filter(context=context, side=side)
        if regime_filter is not None:
            signal.add_filter_result(regime_filter)

        signal.validate()
        return signal

    def _resolve_setup_type(self, last_break: StructureEventContext) -> SetupType:
        if last_break.event_type == StructureEventType.BOS:
            return SetupType.BREAKOUT

        if last_break.event_type in {StructureEventType.CHOCH, StructureEventType.MSS}:
            return SetupType.REVERSAL

        return SetupType.CONTINUATION

    def _resolve_trigger_type(self, last_break: StructureEventContext) -> TriggerType:
        if last_break.is_break:
            return TriggerType.PRIMARY
        return TriggerType.DERIVED

    def _resolve_priority(
        self,
        *,
        last_break: StructureEventContext,
        confidence: float,
        score: float,
        structure: MarketStructureContextView,
    ) -> SignalPriority:
        if (
            last_break.event_type in {StructureEventType.CHOCH, StructureEventType.MSS}
            and confidence >= 0.70
        ):
            return SignalPriority.HIGH

        if last_break.event_type == StructureEventType.BOS and confidence >= 0.75:
            return SignalPriority.HIGH

        if confidence >= 0.85 and score >= 0.75:
            return SignalPriority.HIGH

        if (
            self._mtf_alignment_score(structure, self._side_from_break_event(last_break))
            >= 0.75
            and confidence >= 0.72
        ):
            return SignalPriority.HIGH

        return SignalPriority.MEDIUM

    def _layer_selected_entity(
        self,
        layer: Mapping[str, Any],
        side: SignalSide,
    ) -> dict[str, Any] | None:
        if side == SignalSide.LONG:
            preferred = layer.get("last_swing_low") or layer.get("last_swing_high")
        elif side == SignalSide.SHORT:
            preferred = layer.get("last_swing_high") or layer.get("last_swing_low")
        else:
            preferred = None

        if isinstance(preferred, SwingContext):
            return self._swing_context_to_metadata(preferred)

        return None

    def _swing_context_to_metadata(self, swing: SwingContext) -> dict[str, Any]:
        return {
            "swing_id": swing.swing_id,
            "swing_type": enum_value(swing.swing_type),
            "timestamp": swing.timestamp.isoformat() if swing.timestamp else None,
            "price": swing.price,
            "layer": enum_value(swing.layer),
            "index": swing.index,
            "strength": swing.strength,
            "is_confirmed": swing.is_confirmed,
            "metadata": dict(swing.metadata),
        }

    def _event_context_to_metadata(
        self,
        event: StructureEventContext,
    ) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": enum_value(event.event_type),
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "price": event.price,
            "layer": enum_value(event.layer),
            "direction": enum_value(event.direction),
            "swing_id": event.swing_id,
            "reference_price": event.reference_price,
            "reference_swing_id": event.reference_swing_id,
            "confidence": event.confidence,
            "metadata": dict(event.metadata),
        }

    # ------------------------------------------------------------------
    # Enum parsing helpers
    # ------------------------------------------------------------------

    def _bias_to_side(self, bias: Any) -> SignalSide:
        parsed = self._parse_market_bias(bias)

        if parsed == MarketBias.BULLISH:
            return SignalSide.LONG

        if parsed == MarketBias.BEARISH:
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

        if raw in mapping:
            return mapping[raw]

        try:
            return MarketBias(raw)
        except Exception:
            return MarketBias.UNKNOWN

    def _parse_structure_layer(self, value: Any) -> StructureLayer | None:
        raw = enum_value(value)

        if raw == "internal":
            return StructureLayer.INTERNAL

        if raw == "external":
            return StructureLayer.EXTERNAL

        return None

    def _parse_swing_type(self, value: Any) -> SwingType | None:
        raw = enum_value(value)

        if raw == "high":
            return SwingType.HIGH

        if raw == "low":
            return SwingType.LOW

        try:
            return SwingType(raw)
        except Exception:
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

        if raw in mapping:
            return mapping[raw]

        try:
            return StructureEventType(raw)
        except Exception:
            return None