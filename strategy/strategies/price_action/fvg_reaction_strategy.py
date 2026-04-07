from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from analytics.price_action.enums import FVGDirection, FVGEventType, FVGStatus, StructureLayer
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
        bounded = (
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
        )
        for field_name in bounded:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{field_name} must be between 0.0 and 1.0")

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "FVGReactionStrategyParams":
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
class FVGReactionContext:
    symbol: str | None = None
    timeframe: str | None = None
    last_price: float | None = None
    last_update: datetime | None = None
    internal: dict[str, Any] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)
    last_event: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class FVGReactionStrategy(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
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
        event_bus: Any | None = None,
        logger: Any | None = None,
        strategy_name: str = "fvg_reaction_strategy",
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self._strategy_name = strategy_name
        self.validate_config()

        definition = None
        get_strategy = getattr(self.config, "get_strategy", None)
        if callable(get_strategy):
            definition = get_strategy(strategy_name)

        self.definition = definition
        self.params = FVGReactionStrategyParams.from_definition(definition)

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

            fvg = self._extract_fvg_snapshot(context)
            if fvg.symbol is None and not fvg.external and not fvg.internal:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["fvg_snapshot_missing"],
                )

            freshness_filter = self._build_freshness_filter(context)
            if freshness_filter is not None and freshness_filter.blocked:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["stale_fvg_feature"],
                )

            primary_layer = self._select_primary_layer(fvg)
            secondary_layer = fvg.internal if self.params.prefer_external_layer else fvg.external

            selected_gap = self._select_reaction_gap(context=context, fvg=fvg, primary_layer=primary_layer)
            if selected_gap is None:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["no_reactable_fvg_found"],
                )

            side = self._resolve_side(selected_gap)
            if side == SignalSide.UNKNOWN:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["fvg_direction_not_tradeable"],
                )

            if not self._gap_is_tradeable(selected_gap, primary_layer):
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["selected_fvg_not_tradeable"],
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

            passed = self._passes_runtime_thresholds(signal)
            if not passed:
                signal.to_rejected()

            return self._evaluation(
                context=context,
                signal=signal,
                passed=passed,
                confidence=confidence,
                score=score,
                reasons=reasons,
            )

        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.name}: failed to evaluate FVG reaction for {context.symbol}"
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
    # Gating
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

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return FVGReactionContext()

        state = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
        if not isinstance(state, Mapping):
            return FVGReactionContext()

        internal = self._normalize_fvg_layer(state.get("internal"), StructureLayer.INTERNAL)
        external = self._normalize_fvg_layer(state.get("external"), StructureLayer.EXTERNAL)
        last_event = self._extract_last_event(internal, external)

        return FVGReactionContext(
            symbol=first_non_empty(state.get("symbol"), payload.get("symbol")),
            timeframe=first_non_empty(state.get("timeframe"), payload.get("timeframe")),
            last_price=(
                safe_float(first_non_empty(state.get("last_price"), payload.get("last_price")), 0.0)
                or None
            ),
            last_update=parse_datetime(first_non_empty(state.get("last_update"), payload.get("last_update"))),
            internal=internal,
            external=external,
            last_event=last_event,
            raw=dict(payload),
        )

    def _normalize_fvg_layer(
        self,
        payload: Any,
        default_layer: StructureLayer,
    ) -> dict[str, Any]:
        if payload is None:
            return {}

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return {}

        return {
            "layer": self._parse_structure_layer(payload.get("layer")) or default_layer,
            "total_gaps": int(safe_float(payload.get("total_gaps"), 0.0)),
            "active_gaps": int(safe_float(payload.get("active_gaps"), 0.0)),
            "partially_filled_gaps": int(safe_float(payload.get("partially_filled_gaps"), 0.0)),
            "filled_gaps": int(safe_float(payload.get("filled_gaps"), 0.0)),
            "respected_gaps": int(safe_float(payload.get("respected_gaps"), 0.0)),
            "invalidated_gaps": int(safe_float(payload.get("invalidated_gaps"), 0.0)),
            "recent_fill_activity": clamp(safe_float(payload.get("recent_fill_activity"), 0.0), 0.0, 1.0),
            "nearest_bullish_gap": self._normalize_gap(payload.get("nearest_bullish_gap"), default_layer),
            "nearest_bearish_gap": self._normalize_gap(payload.get("nearest_bearish_gap"), default_layer),
            "strongest_bullish_gap": self._normalize_gap(payload.get("strongest_bullish_gap"), default_layer),
            "strongest_bearish_gap": self._normalize_gap(payload.get("strongest_bearish_gap"), default_layer),
            "last_event": self._normalize_fvg_event(payload.get("last_event"), default_layer),
            "metadata": dict(payload.get("metadata", {}) or {}),
        }

    def _normalize_gap(
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

        return {
            "gap_id": payload.get("gap_id"),
            "layer": self._parse_structure_layer(payload.get("layer")) or default_layer,
            "direction": self._parse_fvg_direction(payload.get("direction")),
            "upper_bound": safe_float(payload.get("upper_bound"), 0.0),
            "lower_bound": safe_float(payload.get("lower_bound"), 0.0),
            "mid_price": safe_float(payload.get("mid_price"), 0.0),
            "size": safe_float(payload.get("size"), 0.0),
            "size_pct": clamp(safe_float(payload.get("size_pct"), 0.0), 0.0, 1.0),
            "strength": clamp(safe_float(payload.get("strength"), 0.0), 0.0, 1.0),
            "status": self._parse_fvg_status(payload.get("status")),
            "fill_percentage": clamp(safe_float(payload.get("fill_percentage"), 0.0), 0.0, 1.0),
            "touch_count": int(safe_float(payload.get("touch_count"), 0.0)),
            "retest_count": int(safe_float(payload.get("retest_count"), 0.0)),
            "created_at": parse_datetime(payload.get("created_at")),
            "updated_at": parse_datetime(payload.get("updated_at")),
            "first_touch_at": parse_datetime(payload.get("first_touch_at")),
            "filled_at": parse_datetime(payload.get("filled_at")),
            "respected_at": parse_datetime(payload.get("respected_at")),
            "invalidated_at": parse_datetime(payload.get("invalidated_at")),
            "created_index": int(safe_float(payload.get("created_index"), 0.0)) if payload.get("created_index") is not None else None,
            "last_touch_index": int(safe_float(payload.get("last_touch_index"), 0.0)) if payload.get("last_touch_index") is not None else None,
            "last_fill_index": int(safe_float(payload.get("last_fill_index"), 0.0)) if payload.get("last_fill_index") is not None else None,
            "source_candle_indices": list(payload.get("source_candle_indices", []) or []),
            "metadata": dict(payload.get("metadata", {}) or {}),
        }

    def _normalize_fvg_event(
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

        return {
            "event_id": payload.get("event_id"),
            "event_type": self._parse_fvg_event_type(payload.get("event_type")),
            "timestamp": parse_datetime(payload.get("timestamp")),
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "layer": self._parse_structure_layer(payload.get("layer")) or default_layer,
            "gap_id": payload.get("gap_id"),
            "direction": self._parse_fvg_direction(payload.get("direction")),
            "upper_bound": safe_float(payload.get("upper_bound"), 0.0),
            "lower_bound": safe_float(payload.get("lower_bound"), 0.0),
            "fill_percentage": clamp(safe_float(payload.get("fill_percentage"), 0.0), 0.0, 1.0),
            "confidence": clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            "reference_price": (
                safe_float(payload.get("reference_price"), 0.0)
                if payload.get("reference_price") is not None
                else None
            ),
            "metadata": dict(payload.get("metadata", {}) or {}),
        }

    def _extract_last_event(
        self,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        candidates = [
            event for event in (
                external.get("last_event"),
                internal.get("last_event"),
            )
            if event is not None
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return candidates[0]

    # ------------------------------------------------------------------
    # Selection / side
    # ------------------------------------------------------------------

    def _select_primary_layer(self, fvg: FVGReactionContext) -> dict[str, Any]:
        return fvg.external if self.params.prefer_external_layer else fvg.internal

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
            self.params.require_recent_event
            and last_event is not None
            and self._event_is_reaction_event(last_event)
            and clamp(safe_float(last_event.get("confidence"), 0.0), 0.0, 1.0) >= self.params.min_event_confidence
        ):
            gap = self._gap_from_event(primary_layer=primary_layer, event=last_event)
            if gap is not None:
                return gap

        candidate_gaps = [
            primary_layer.get("strongest_bullish_gap"),
            primary_layer.get("strongest_bearish_gap"),
            primary_layer.get("nearest_bullish_gap"),
            primary_layer.get("nearest_bearish_gap"),
        ]

        filtered = [gap for gap in candidate_gaps if gap is not None]
        if not filtered:
            return None

        best_gap: dict[str, Any] | None = None
        best_score = -1.0

        for gap in filtered:
            if not self._gap_is_tradeable(gap, primary_layer):
                continue

            if not self.params.allow_active_gap_proximity_entry and gap.get("status") == FVGStatus.ACTIVE:
                continue

            local_score = self._gap_selection_score(gap=gap, current_price=current_price)
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

        if self.params.require_directional_gap and gap.get("direction") not in {
            FVGDirection.BULLISH,
            FVGDirection.BEARISH,
        }:
            return False

        if clamp(safe_float(gap.get("strength"), 0.0), 0.0, 1.0) < self.params.min_gap_strength:
            return False

        status = gap.get("status", FVGStatus.ACTIVE)

        if self.params.block_invalidated_gaps and status == FVGStatus.INVALIDATED:
            return False

        if self.params.block_filled_gaps and status == FVGStatus.FILLED:
            return False

        if clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0) > self.params.max_gap_fill_for_entry:
            return False

        if self.params.require_respected_or_retested:
            respected = status == FVGStatus.RESPECTED
            retested = int(safe_float(gap.get("retest_count"), 0.0)) > 0
            if not (respected or retested):
                return False

        if clamp(safe_float(layer.get("recent_fill_activity"), 0.0), 0.0, 1.0) > self.params.max_recent_fill_activity:
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
        score += 0.15 * (1.0 - clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0))
        score += 0.10 * min(1.0, safe_float(gap.get("retest_count"), 0.0) / 3.0)
        score += 0.10 * self._proximity_score(current_price=current_price, gap=gap)
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
            return primary_layer.get("nearest_bullish_gap") or primary_layer.get("strongest_bullish_gap")
        if direction == FVGDirection.BEARISH:
            return primary_layer.get("nearest_bearish_gap") or primary_layer.get("strongest_bearish_gap")
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
        current_price = self._resolve_current_price(context=context, fvg=None)

        score = 0.0
        score += self.params.primary_gap_weight * clamp(safe_float(selected_gap.get("strength"), 0.0), 0.0, 1.0)

        event_confidence = clamp(
            safe_float(last_event.get("confidence"), 0.0) if last_event else 0.0,
            0.0,
            1.0,
        )
        score += self.params.event_confidence_weight * event_confidence

        score += self.params.status_quality_weight * self._status_quality_score(selected_gap.get("status"))
        score += self.params.proximity_weight * self._proximity_score(
            current_price=current_price,
            gap=selected_gap,
        )
        score += self.params.fill_quality_weight * self._fill_quality_score(selected_gap)

        secondary_alignment = self._secondary_layer_alignment_score(
            secondary_layer=secondary_layer,
            selected_gap=selected_gap,
        )
        score += self.params.regime_alignment_weight * max(
            self._regime_alignment_score(context=context, side=side),
            secondary_alignment,
        )

        if safe_float(selected_gap.get("retest_count"), 0.0) > 0:
            score += self.params.retest_bonus_weight

        if selected_gap.get("status") == FVGStatus.RESPECTED:
            score += self.params.respect_bonus_weight

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
        current_price = self._resolve_current_price(context=context, fvg=None)
        components: list[float] = [
            clamp(safe_float(selected_gap.get("strength"), 0.0), 0.0, 1.0),
            self._status_quality_score(selected_gap.get("status")),
            self._fill_quality_score(selected_gap),
            self._proximity_score(current_price=current_price, gap=selected_gap),
            self._regime_alignment_score(context=context, side=side),
        ]

        if last_event is not None:
            components.append(clamp(safe_float(last_event.get("confidence"), 0.0), 0.0, 1.0))

        secondary_alignment = self._secondary_layer_alignment_score(
            secondary_layer=secondary_layer,
            selected_gap=selected_gap,
        )
        components.append(secondary_alignment)

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
        if layer is not None:
            reasons.append(f"fvg_layer_{layer.value}")

        direction = selected_gap.get("direction")
        if direction is not None:
            reasons.append(f"fvg_direction_{direction.value}")

        status = selected_gap.get("status")
        if status is not None:
            reasons.append(f"fvg_status_{status.value}")

        if safe_float(selected_gap.get("retest_count"), 0.0) > 0:
            reasons.append("fvg_retested")

        if selected_gap.get("status") == FVGStatus.RESPECTED:
            reasons.append("fvg_respected")

        last_event = fvg.last_event
        if last_event is not None and last_event.get("event_type") is not None:
            reasons.append(f"last_fvg_event_{last_event['event_type'].value}")

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
        priority = self._resolve_priority(confidence=confidence, gap=selected_gap, last_event=fvg.last_event)

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
                "fvg_last_update": fvg.last_update.isoformat() if fvg.last_update else None,
                "fvg_last_price": fvg.last_price,
                "gap_id": selected_gap.get("gap_id"),
                "gap_layer": selected_gap.get("layer").value if selected_gap.get("layer") is not None else None,
                "gap_direction": selected_gap.get("direction").value if selected_gap.get("direction") is not None else None,
                "gap_status": selected_gap.get("status").value if selected_gap.get("status") is not None else None,
                "gap_strength": safe_float(selected_gap.get("strength"), 0.0),
                "gap_fill_percentage": safe_float(selected_gap.get("fill_percentage"), 0.0),
                "gap_mid_price": safe_float(selected_gap.get("mid_price"), 0.0),
                "gap_upper_bound": safe_float(selected_gap.get("upper_bound"), 0.0),
                "gap_lower_bound": safe_float(selected_gap.get("lower_bound"), 0.0),
                "gap_retest_count": int(safe_float(selected_gap.get("retest_count"), 0.0)),
                "last_fvg_event_type": (
                    fvg.last_event.get("event_type").value
                    if fvg.last_event is not None and fvg.last_event.get("event_type") is not None
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
            and gap.get("status") in {FVGStatus.RESPECTED, FVGStatus.PARTIALLY_FILLED}
        ):
            return SignalPriority.HIGH

        if (
            last_event is not None
            and last_event.get("event_type") in {FVGEventType.FVG_RESPECTED, FVGEventType.FVG_RETESTED}
            and confidence >= 0.70
        ):
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
                    name="fvg_freshness",
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

        regime = self._resolve_market_regime(context)

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
    # Evaluation helper
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

    # ------------------------------------------------------------------
    # Helper logic
    # ------------------------------------------------------------------

    def _event_is_reaction_event(self, event: Mapping[str, Any]) -> bool:
        event_type = event.get("event_type")
        if event_type == FVGEventType.FVG_RESPECTED and self.params.allow_respected_reaction:
            return True
        if event_type == FVGEventType.FVG_RETESTED and self.params.allow_retested_reaction:
            return True
        if event_type in {FVGEventType.FVG_PARTIALLY_FILLED, FVGEventType.FVG_FILL_STARTED} and self.params.allow_partial_fill_reaction:
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
        fill = clamp(safe_float(gap.get("fill_percentage"), 0.0), 0.0, 1.0)
        if fill < self.params.min_gap_fill_for_reaction:
            return 0.35 if self.params.allow_active_gap_proximity_entry else 0.0
        if fill > self.params.max_gap_fill_for_entry:
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
        if distance_pct >= self.params.max_distance_to_mid_pct:
            return 0.0

        return clamp(1.0 - (distance_pct / max(self.params.max_distance_to_mid_pct, 1e-9)), 0.0, 1.0)

    def _secondary_layer_alignment_score(
        self,
        *,
        secondary_layer: Mapping[str, Any],
        selected_gap: Mapping[str, Any],
    ) -> float:
        direction = selected_gap.get("direction")
        if direction == FVGDirection.BULLISH:
            ref_gap = secondary_layer.get("nearest_bullish_gap") or secondary_layer.get("strongest_bullish_gap")
        elif direction == FVGDirection.BEARISH:
            ref_gap = secondary_layer.get("nearest_bearish_gap") or secondary_layer.get("strongest_bearish_gap")
        else:
            ref_gap = None

        if ref_gap is None:
            return 0.35

        score = 0.0
        score += 0.6 * clamp(safe_float(ref_gap.get("strength"), 0.0), 0.0, 1.0)
        score += 0.4 * self._status_quality_score(ref_gap.get("status"))
        return clamp(score, 0.0, 1.0)

    def _regime_alignment_score(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> float:
        regime = self._resolve_market_regime(context)

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
            return 0.55
        if regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_DRIVEN}:
            return 0.35
        if side == SignalSide.LONG and regime in bullish_regimes:
            return 1.0
        if side == SignalSide.SHORT and regime in bearish_regimes:
            return 1.0
        return 0.20 if self.params.block_counter_regime else 0.40

    def _resolve_market_regime(self, context: SignalContext) -> MarketRegime:
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