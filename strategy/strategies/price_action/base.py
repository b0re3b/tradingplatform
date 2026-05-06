from __future__ import annotations

import inspect
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, Protocol, TypeVar, cast

from core.event_bus import EventBus
from core.logger import TradingLoggerAdapter, get_logger

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
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import ValidationError
from strategy.models import (
    FilterResult,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
)


class PriceActionStrategyParamsProtocol(Protocol):
    strategy_name: str
    emit_signal_events: bool
    signal_event_name: str
    freshness_feature_names: tuple[str, ...]

    def validate(self) -> None:
        ...

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "PriceActionStrategyParamsProtocol":
        ...


ParamsT = TypeVar("ParamsT", bound=PriceActionStrategyParamsProtocol)


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


def apply_definition_metadata(
    *,
    params: Any,
    definition: StrategyDefinitionConfig | None,
    skip_fields: set[str] | None = None,
) -> Any:
    """
    Універсальне застосування StrategyDefinitionConfig.metadata до params dataclass.

    Це дозволяє прибрати дублювання з:
    - MarketStructureStrategyParams.from_definition()
    - TrendContinuationStrategyParams.from_definition()
    - FVGReactionStrategyParams.from_definition()
    """
    skip_fields = skip_fields or {"strategy_name"}

    if definition is None:
        params.validate()
        return params

    if hasattr(params, "strategy_name"):
        params.strategy_name = definition.name or params.strategy_name

    metadata = definition.metadata or {}
    dataclass_fields = getattr(params, "__dataclass_fields__", {})

    for field_name in dataclass_fields.keys():
        if field_name in skip_fields:
            continue

        if field_name in metadata:
            setattr(params, field_name, metadata[field_name])

    params.validate()
    return params


class PriceActionStrategyBase(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Базовий клас для strategy/strategies/price_action.

    Дає спільну інфраструктуру для:
    - MarketStructureStrategy
    - TrendContinuationStrategy
    - FVGReactionStrategy
    - майбутніх SupportResistanceStrategy / LiquidityLevelsStrategy тощо

    Не містить конкретної торгової логіки.
    Конкретні стратегії мають реалізовувати:
    - evaluate()
    - extraction/normalization
    - score/confidence/reasons/build_signal
    """

    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_priority: int = 100

    def __init__(
        self,
        *,
        config: StrategyConfig,
        strategy_name: str,
        params_cls: type[ParamsT],
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
    ) -> None:
        resolved_logger = logger or get_logger(
            __name__,
            event_type="price_action_strategy",
            strategies=strategy_name,
        )

        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=resolved_logger,
        )

        self._strategy_name = strategy_name
        self._logger = resolved_logger

        self.validate_config()

        self.definition = self._resolve_definition(strategy_name)
        # FIX: використовуємо cast щоб зберегти точну типізацію ParamsT
        # від_definition є classmethod протоколу; виклик через cast є безпечним,
        # оскільки всі конкретні реалізації ParamsT гарантують його наявність.
        self.params: ParamsT = cast(
            ParamsT,
            params_cls.from_definition(self.definition),  # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def priority(self) -> int:
        if self.definition is not None:
            return self.definition.priority
        return self.default_priority

    # ------------------------------------------------------------------
    # Config / runtime helpers
    # ------------------------------------------------------------------

    def _resolve_definition(
        self,
        strategy_name: str,
    ) -> StrategyDefinitionConfig | None:
        get_strategy = getattr(self.config, "get_strategy", None)
        if callable(get_strategy):
            return get_strategy(strategy_name)
        return None

    @property
    def runtime(self) -> Any:
        if self.definition is not None:
            return self.definition.runtime
        return self.config.runtime

    def _is_strategy_enabled(self) -> bool:
        is_strategy_enabled = getattr(self.config, "is_strategy_enabled", None)
        if callable(is_strategy_enabled):
            return bool(is_strategy_enabled(self.name))

        if self.definition is not None:
            return bool(self.definition.runtime.enabled)

        return bool(self.config.runtime.enabled)

    def _symbol_allowed(self, symbol: str) -> bool:
        runtime = self.runtime
        return not runtime.symbols or symbol in runtime.symbols

    def _timeframe_allowed(self, timeframe: Timeframe) -> bool:
        runtime = self.runtime
        return not runtime.timeframes or timeframe in runtime.timeframes

    def _passes_runtime_thresholds(self, signal: StrategySignal) -> bool:
        runtime = self.runtime

        if signal.confidence < runtime.min_confidence:
            return False

        if signal.score < runtime.min_score:
            return False

        return True

    def _basic_runtime_gate(self, context: SignalContext) -> StrategyEvaluation | None:
        """
        Спільний pre-check для evaluate().

        Використання в дочірньому класі:

            blocked = self._basic_runtime_gate(context)
            if blocked is not None:
                return blocked
        """
        self.validate_context(context)

        if not self._is_strategy_enabled():
            return self._rejected_evaluation(
                context=context,
                reason="strategy_disabled",
            )

        if not self._symbol_allowed(context.symbol):
            return self._rejected_evaluation(
                context=context,
                reason="symbol_not_allowed",
            )

        if not self._timeframe_allowed(context.timeframe):
            return self._rejected_evaluation(
                context=context,
                reason="timeframe_not_allowed",
            )

        return None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_freshness_filter(
        self,
        context: SignalContext,
        *,
        filter_name: str | None = None,
    ) -> FilterResult | None:
        """
        FIX: Перевіряємо ВСІ freshness_feature_names.

        Попередня реалізація повертала результат першої знайденої фічі та
        ігнорувала решту, що могло пропустити stale-фічі.

        Поточна логіка:
        - якщо хоча б одна з фіч є stale → BLOCK (з ім'ям першої stale-фічі)
        - якщо всі знайдені фічі fresh → PASS (з ім'ям першої знайденої)
        - якщо жодна фіча не знайдена в контексті → None
        """
        first_found: str | None = None
        stale_feature: str | None = None

        for feature_name in self.params.freshness_feature_names:
            if not context.has_feature(feature_name):
                continue

            if first_found is None:
                first_found = feature_name

            if context.feature_is_stale(feature_name):
                stale_feature = feature_name
                break  # достатньо однієї stale-фічі щоб заблокувати

        if first_found is None:
            return None

        is_stale = stale_feature is not None
        feature_name_for_result = stale_feature if is_stale else first_found

        return FilterResult(
            name=filter_name or f"{self.name}_freshness",
            decision=FilterDecision.BLOCK if is_stale else FilterDecision.PASS,
            reason="feature_stale" if is_stale else "feature_fresh",
            score_impact=-1.0 if is_stale else 0.0,
            metadata={
                "feature_name": feature_name_for_result,
                "strategy_name": self.name,
            },
        )

    def _build_regime_filter(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> FilterResult | None:
        runtime = self.runtime

        if not runtime.allowed_regimes:
            return None

        regime = self._resolve_market_regime(context)

        if MarketRegime.UNKNOWN in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
                metadata={
                    "strategy_name": self.name,
                    "side": side.value,
                },
            )

        if regime in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
                metadata={
                    "strategy_name": self.name,
                    "side": side.value,
                },
            )

        return FilterResult(
            name="market_regime",
            decision=FilterDecision.BLOCK,
            reason=f"regime_{regime.value}_not_allowed_for_{side.value}",
            score_impact=-1.0,
            metadata={
                "strategy_name": self.name,
                "side": side.value,
            },
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
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
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        evaluation_metadata = {
            "category": self.category.value,
            "timeframe": context.timeframe.value,
        }

        if metadata:
            evaluation_metadata.update(metadata)

        evaluation = StrategyEvaluation(
            strategy_name=self.name,
            symbol=context.symbol,
            timestamp=context.timestamp,
            signal=signal,
            passed=passed,
            score=score,
            confidence=confidence,
            reasons=list(reasons),
            metadata=evaluation_metadata,
        )

        evaluation.validate()
        return evaluation

    def _rejected_evaluation(
        self,
        *,
        context: SignalContext,
        reason: str,
        confidence: float = 0.0,
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        return self._evaluation(
            context=context,
            signal=None,
            passed=False,
            confidence=confidence,
            score=score,
            reasons=[reason],
            metadata=metadata,
        )

    def _finalize_signal_evaluation(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal,
        confidence: float,
        score: float,
        reasons: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
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
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def maybe_emit_signal(self, signal: StrategySignal) -> None:
        """
        Безпечна публікація StrategySignal у EventBus.

        Якщо EventBus не запущений або emit падає, стратегія не повинна ламати
        evaluation-flow. Це telemetry/event side effect, а не основна логіка.
        """
        if not self.params.emit_signal_events:
            return

        payload = self._build_signal_event_payload(signal)

        try:
            await self.emit_event(
                self.params.signal_event_name,
                payload,
                source=self.name,
            )
        except RuntimeError:
            self._logger.warning(
                "Signal event skipped because EventBus is not running | signal_id=%s symbol=%s",
                payload.get("signal_id"),
                signal.symbol,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit strategy signal | signal_id=%s symbol=%s",
                payload.get("signal_id"),
                signal.symbol,
            )

    def _build_signal_event_payload(
        self,
        signal: StrategySignal,
    ) -> dict[str, Any]:
        metadata = dict(getattr(signal, "metadata", {}) or {})

        signal_id = (
            getattr(signal, "signal_id", None)
            or getattr(signal, "id", None)
            or metadata.get("signal_id")
        )

        payload = {
            "signal_id": signal_id,
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
            "metadata": metadata,
        }

        created_at = getattr(signal, "created_at", None) or getattr(signal, "timestamp", None)
        if created_at is not None:
            payload["created_at"] = (
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else created_at
            )

        origin = getattr(signal, "origin", None)
        if origin is not None:
            payload["origin"] = getattr(origin, "value", origin)

        category = getattr(signal, "category", None)
        if category is not None:
            payload["category"] = getattr(category, "value", category)

        trigger_type = getattr(signal, "trigger_type", None)
        if trigger_type is not None:
            payload["trigger_type"] = getattr(trigger_type, "value", trigger_type)

        return payload

    # ------------------------------------------------------------------
    # Market regime helpers
    # ------------------------------------------------------------------

    def _resolve_market_regime(self, context: SignalContext) -> MarketRegime:
        # Явна анотація Any дозволяє лінтеру коректно аналізувати гілки нижче.
        # context.regime.regime може бути MarketRegime, str або іншим raw-значенням
        # залежно від джерела даних, тому isinstance-перевірка є необхідною.
        regime: Any = (
            context.regime.regime
            if context.regime is not None
            else MarketRegime.UNKNOWN
        )

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

    def _regime_alignment_score(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> float:
        """
        FIX: BREAKOUT і SQUEEZE були одночасно в bullish_regimes і bearish_regimes,
        що робило їх нейтральними до сторони (завжди 1.0) — помилкова поведінка.

        Виправлена логіка:
        - TRENDING_UP  → сильне підтвердження для LONG
        - TRENDING_DOWN → сильне підтвердження для SHORT
        - BREAKOUT/SQUEEZE → помірне підтвердження для обох сторін (momentum-neutral)
        - RANGING → слабке підтвердження (контр-трендові умови)
        - HIGH_VOLATILITY/NEWS_DRIVEN → знижений score (ризик хибних сигналів)
        - UNKNOWN → нейтральний score
        - Все інше (ILLIQUID, RISK_OFF, LOW_VOLATILITY) → мінімальний score
        """
        regime = self._resolve_market_regime(context)

        if regime == MarketRegime.UNKNOWN:
            return 0.5

        if regime == MarketRegime.RANGING:
            return 0.35

        if regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_DRIVEN}:
            return 0.40

        # FIX: BREAKOUT/SQUEEZE є momentum-режимами без чіткої directional bias —
        # дають помірний score незалежно від side, а не максимальний.
        if regime in {MarketRegime.BREAKOUT, MarketRegime.SQUEEZE}:
            return 0.65

        if side == SignalSide.LONG and regime == MarketRegime.TRENDING_UP:
            return 1.0

        if side == SignalSide.SHORT and regime == MarketRegime.TRENDING_DOWN:
            return 1.0

        # Протилежний напрямок тренду або інші режими (ILLIQUID, RISK_OFF, тощо)
        return 0.20

    # Backward-compatible alias для класів, де вже використовується _resolve_regime()
    def _resolve_regime(self, context: SignalContext) -> MarketRegime:
        return self._resolve_market_regime(context)

    # ------------------------------------------------------------------
    # Generic parsing helpers
    # ------------------------------------------------------------------

    def _parse_timeframe(self, value: Any) -> Timeframe | None:
        if isinstance(value, Timeframe):
            return value

        raw = enum_value(value)
        if not raw:
            return None

        for timeframe in Timeframe:
            if timeframe.value == raw:
                return timeframe

        return None

    def _mapping_or_empty(self, payload: Any) -> Mapping[str, Any]:
        if payload is None:
            return {}

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if isinstance(payload, Mapping):
            return payload

        return {}

    def _state_mapping_or_empty(self, payload: Any) -> Mapping[str, Any]:
        payload_mapping = self._mapping_or_empty(payload)

        state = payload_mapping.get("state")
        if isinstance(state, Mapping):
            return state

        return payload_mapping

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_bounded_fields(
        *,
        instance: Any,
        field_names: tuple[str, ...],
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> None:
        for field_name in field_names:
            value = getattr(instance, field_name)

            if not minimum <= value <= maximum:
                raise ValidationError(
                    f"{field_name} must be between {minimum} and {maximum}"
                )

    @staticmethod
    def validate_non_negative_fields(
        *,
        instance: Any,
        field_names: tuple[str, ...],
    ) -> None:
        for field_name in field_names:
            value = getattr(instance, field_name)

            if value < 0:
                raise ValidationError(f"{field_name} must be >= 0")

    # ------------------------------------------------------------------
    # Optional hook for subclasses
    # ------------------------------------------------------------------

    async def maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value