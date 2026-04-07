from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from analytics.price_action.enums import StructureLayer, TrendDirection, TrendEventType, TrendRegime
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
        bounded = (
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
        )
        for field_name in bounded:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{field_name} must be between 0.0 and 1.0")

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "TrendContinuationStrategyParams":
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


class TrendContinuationStrategy(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Strategy layer wrapper around analytics.price_action.trend.

    Основна логіка:
    - long continuation, коли тренд bullish і continuation probability достатня
    - short continuation, коли тренд bearish і continuation probability достатня
    - опційно вимагає internal confirmation
    - відсікає exhaustion / reversal risk / stale feature
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
        strategy_name: str = "trend_continuation_strategy",
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self._strategy_name = strategy_name
        self.validate_config()

        definition = None
        get_strategy = getattr(self.config, "get_strategy", None)
        if callable(get_strategy):
            definition = get_strategy(strategy_name)

        self.definition = definition
        self.params = TrendContinuationStrategyParams.from_definition(definition)

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

            trend = self._extract_trend_snapshot(context)
            if trend.symbol is None and not trend.external and not trend.internal:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["trend_snapshot_missing"],
                )

            freshness_filter = self._build_freshness_filter(context)
            if freshness_filter is not None and freshness_filter.blocked:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["stale_trend_feature"],
                )

            side = self._resolve_side(context=context, trend=trend)
            if side == SignalSide.UNKNOWN:
                return self._evaluation(
                    context=context,
                    passed=False,
                    confidence=0.0,
                    score=0.0,
                    reasons=["no_valid_trend_continuation_setup"],
                )

            primary_layer = self._select_primary_layer(trend)
            primary_layer_name = "external" if self.params.prefer_external_layer else "internal"
            secondary_layer = trend.internal if self.params.prefer_external_layer else trend.external
            secondary_layer_name = "internal" if self.params.prefer_external_layer else "external"

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
                f"{self.name}: failed to evaluate trend continuation for {context.symbol}"
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
        if payload is None:
            return TrendContextView()

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if not isinstance(payload, Mapping):
            return TrendContextView()

        state = payload.get("state") if isinstance(payload.get("state"), Mapping) else payload
        if not isinstance(state, Mapping):
            return TrendContextView()

        internal = self._normalize_trend_layer(state.get("internal"), StructureLayer.INTERNAL)
        external = self._normalize_trend_layer(state.get("external"), StructureLayer.EXTERNAL)
        global_state = self._normalize_global_state(state)

        return TrendContextView(
            symbol=first_non_empty(state.get("symbol"), payload.get("symbol")),
            timeframe=first_non_empty(state.get("timeframe"), payload.get("timeframe")),
            last_price=(
                safe_float(first_non_empty(state.get("last_price"), payload.get("last_price")), 0.0)
                or None
            ),
            last_update=parse_datetime(first_non_empty(state.get("last_update"), payload.get("last_update"))),
            internal=internal,
            external=external,
            global_state=global_state,
            last_signal=self._extract_last_signal(internal, external),
            raw=dict(payload),
        )

    def _normalize_trend_layer(
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

        last_signal = payload.get("last_signal")
        if hasattr(last_signal, "__dict__") and not isinstance(last_signal, Mapping):
            last_signal = vars(last_signal)

        return {
            "layer": self._parse_structure_layer(payload.get("layer")) or default_layer,
            "direction": self._parse_trend_direction(payload.get("direction")),
            "regime": self._parse_trend_regime(payload.get("regime")),
            "strength": clamp(safe_float(payload.get("strength"), 0.0), 0.0, 1.0),
            "confidence": clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            "continuation_probability": clamp(
                safe_float(payload.get("continuation_probability"), 0.0),
                0.0,
                1.0,
            ),
            "reversal_risk": clamp(safe_float(payload.get("reversal_risk"), 0.0), 0.0, 1.0),
            "exhaustion_score": clamp(safe_float(payload.get("exhaustion_score"), 0.0), 0.0, 1.0),
            "pullback_depth": clamp(safe_float(payload.get("pullback_depth"), 0.0), 0.0, 1.0),
            "consolidation_score": clamp(safe_float(payload.get("consolidation_score"), 0.0), 0.0, 1.0),
            "structure_score": clamp(safe_float(payload.get("structure_score"), 0.0), -1.0, 1.0),
            "is_accelerating": safe_bool(payload.get("is_accelerating"), False),
            "is_exhausted": safe_bool(payload.get("is_exhausted"), False),
            "in_pullback": safe_bool(payload.get("in_pullback"), False),
            "last_signal": self._normalize_trend_signal(last_signal, default_layer),
        }

    def _normalize_trend_signal(
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
            "signal_id": payload.get("signal_id"),
            "timestamp": parse_datetime(payload.get("timestamp")),
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "layer": self._parse_structure_layer(payload.get("layer")) or default_layer,
            "event_type": self._parse_trend_event_type(payload.get("event_type")),
            "direction": self._parse_trend_direction(payload.get("direction")),
            "strength": clamp(safe_float(payload.get("strength"), 0.0), 0.0, 1.0),
            "confidence": clamp(safe_float(payload.get("confidence"), 0.0), 0.0, 1.0),
            "regime": self._parse_trend_regime(payload.get("regime")),
            "price": safe_float(payload.get("price"), 0.0) if payload.get("price") is not None else None,
            "metadata": dict(payload.get("metadata", {}) or {}),
        }

    def _normalize_global_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        global_candidates = [
            state.get("global_state"),
            state.get("summary"),
            state.get("metadata"),
        ]

        for candidate in global_candidates:
            if candidate is None:
                continue
            if hasattr(candidate, "__dict__") and not isinstance(candidate, Mapping):
                candidate = vars(candidate)
            if isinstance(candidate, Mapping):
                return dict(candidate)

        return {}

    def _extract_last_signal(
        self,
        internal: Mapping[str, Any],
        external: Mapping[str, Any],
    ) -> dict[str, Any] | None:
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

        candidates.sort(
            key=lambda x: x.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return candidates[0]

    # ------------------------------------------------------------------
    # Direction resolution
    # ------------------------------------------------------------------

    def _select_primary_layer(self, trend: TrendContextView) -> dict[str, Any]:
        return trend.external if self.params.prefer_external_layer else trend.internal

    def _resolve_side(self, context: SignalContext, trend: TrendContextView) -> SignalSide:
        primary = self._select_primary_layer(trend)
        secondary = trend.internal if self.params.prefer_external_layer else trend.external

        primary_side = self._trend_direction_to_side(primary.get("direction", TrendDirection.UNKNOWN))
        secondary_side = self._trend_direction_to_side(secondary.get("direction", TrendDirection.UNKNOWN))

        if primary_side == SignalSide.UNKNOWN:
            if self.params.allow_cross_layer_fallback:
                return secondary_side
            return SignalSide.UNKNOWN

        if not self._layer_eligible(primary):
            return SignalSide.UNKNOWN

        if self.params.require_internal_confirmation and secondary_side != SignalSide.UNKNOWN:
            if secondary_side != primary_side:
                return SignalSide.UNKNOWN
            if not self._layer_confirmation_ok(secondary):
                return SignalSide.UNKNOWN

        if self.params.require_direction_alignment and secondary_side not in {SignalSide.UNKNOWN, primary_side}:
            return SignalSide.UNKNOWN

        if not self._side_regime_allowed(primary_side, primary):
            return SignalSide.UNKNOWN

        return primary_side

    def _layer_eligible(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False
        if clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0) < self.params.min_layer_confidence:
            return False
        if clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0) < self.params.min_trend_strength:
            return False
        if clamp(safe_float(layer.get("continuation_probability"), 0.0), 0.0, 1.0) < self.params.min_continuation_probability:
            return False
        if self.params.block_high_reversal_risk and clamp(safe_float(layer.get("reversal_risk"), 0.0), 0.0, 1.0) > self.params.max_reversal_risk:
            return False
        if self.params.block_exhausted_trend:
            exhausted = safe_bool(layer.get("is_exhausted"), False)
            exhaustion_score = clamp(safe_float(layer.get("exhaustion_score"), 0.0), 0.0, 1.0)
            if exhausted or exhaustion_score > self.params.max_exhaustion_score:
                return False
        if clamp(safe_float(layer.get("consolidation_score"), 0.0), 0.0, 1.0) > self.params.max_consolidation_score:
            return False
        return True

    def _layer_confirmation_ok(self, layer: Mapping[str, Any]) -> bool:
        if not layer:
            return False
        return (
            clamp(safe_float(layer.get("confidence"), 0.0), 0.0, 1.0) >= self.params.min_layer_confidence * 0.85
            and clamp(safe_float(layer.get("strength"), 0.0), 0.0, 1.0) >= self.params.min_trend_strength * 0.85
        )

    def _side_regime_allowed(self, side: SignalSide, layer: Mapping[str, Any]) -> bool:
        if not self.params.block_counter_regime:
            return True

        regime = layer.get("regime", TrendRegime.UNKNOWN)
        if side == SignalSide.LONG and regime in {
            TrendRegime.TRENDING_UP,
            TrendRegime.ACCELERATING_UP,
            TrendRegime.PULLBACK_UPTREND,
        }:
            return True
        if side == SignalSide.SHORT and regime in {
            TrendRegime.TRENDING_DOWN,
            TrendRegime.ACCELERATING_DOWN,
            TrendRegime.PULLBACK_DOWNTREND,
        }:
            return True
        return False

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

        score += self.params.external_confidence_weight * clamp(
            safe_float(primary_layer.get("confidence"), 0.0),
            0.0,
            1.0,
        )
        score += self.params.external_strength_weight * clamp(
            safe_float(primary_layer.get("strength"), 0.0),
            0.0,
            1.0,
        )
        score += self.params.continuation_probability_weight * clamp(
            safe_float(primary_layer.get("continuation_probability"), 0.0),
            0.0,
            1.0,
        )

        secondary_side = self._trend_direction_to_side(
            secondary_layer.get("direction", TrendDirection.UNKNOWN)
        )
        if secondary_side == side:
            score += self.params.internal_confirmation_weight * clamp(
                safe_float(secondary_layer.get("confidence"), 0.0),
                0.0,
                1.0,
            )

        structure_alignment = self._structure_alignment_score(primary_layer, side)
        score += self.params.structure_alignment_weight * structure_alignment

        regime_alignment = self._regime_alignment_score(context=context, side=side)
        score += self.params.regime_alignment_weight * regime_alignment

        if self.params.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            pullback_depth = clamp(safe_float(primary_layer.get("pullback_depth"), 0.0), 0.0, 1.0)
            if pullback_depth <= self.params.max_pullback_depth:
                score += self.params.pullback_bonus_weight * (1.0 - pullback_depth)

        if safe_bool(primary_layer.get("is_accelerating"), False):
            score += self.params.acceleration_bonus_weight

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

        if self.params.allow_pullback_entries and safe_bool(primary_layer.get("in_pullback"), False):
            reasons.append("continuation_pullback_entry")

        last_signal = trend.last_signal
        if last_signal is not None and self._trend_direction_to_side(last_signal.get("direction", TrendDirection.UNKNOWN)) == side:
            event_type = last_signal.get("event_type")
            if event_type is not None:
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
                "last_trend_event_type": (
                    last_signal.get("event_type").value
                    if last_signal is not None and last_signal.get("event_type") is not None
                    else None
                ),
                "last_trend_event_layer": (
                    last_signal.get("layer").value
                    if last_signal is not None and last_signal.get("layer") is not None
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
                TrendEventType.CONTINUATION_CONFIRMED,
                TrendEventType.ACCELERATION,
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
            if event_type in {TrendEventType.CONTINUATION_CONFIRMED, TrendEventType.ACCELERATION}:
                return TriggerType.PRIMARY
            if event_type in {TrendEventType.PULLBACK, TrendEventType.RESUMPTION}:
                return TriggerType.CONFIRMATION

        if safe_bool(primary_layer.get("in_pullback"), False):
            return TriggerType.CONFIRMATION

        return TriggerType.DERIVED

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_freshness_filter(self, context: SignalContext) -> FilterResult | None:
        for feature_name in self.params.freshness_feature_names:
            if context.has_feature(feature_name):
                is_stale = context.feature_is_stale(feature_name)
                return FilterResult(
                    name="trend_freshness",
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
            return 0.35
        if regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_DRIVEN}:
            return 0.40
        if side == SignalSide.LONG and regime in bullish_regimes:
            return 1.0
        if side == SignalSide.SHORT and regime in bearish_regimes:
            return 1.0
        return 0.20

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
        try:
            return TrendRegime(raw)
        except Exception:
            return TrendRegime.UNKNOWN

    def _parse_trend_event_type(self, value: Any) -> TrendEventType | None:
        raw = enum_value(value)
        if not raw:
            return None
        try:
            return TrendEventType(raw)
        except Exception:
            return None