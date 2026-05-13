from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger

from analytics.liquidity.models import LiquidityMapSnapshot
from strategy.base import (
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
    StrategyComponent,
)
from strategy.config import StrategyConfig, StrategyDefinitionConfig
from strategy.context import StrategyContext
from strategy.enums import FilterDecision, StrategyCategory
from strategy.models import FilterResult, StrategySignal


class BaseLiquidityStrategy(
    StrategyComponent,
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
    ABC,
):
    """
    Base class для всіх liquidity strategies.

    Відповідальність:
    - централізований logger через core.logger.get_logger;
    - EventBus signal emission через EventEmitterMixin;
    - StrategyConfig helpers;
    - безпечне діставання LiquidityMapSnapshot зі StrategyContext;
    - common pre-filters;
    - freshness validation;
    - symbol/timeframe/regime validation;
    - per-symbol/timeframe emit cooldown.

    Архітектурні правила:
    - не викликає analytics detectors напряму;
    - не будує LiquidityMapSnapshot самостійно;
    - не містить торгового execution;
    - не має Scheduler;
    - працює тільки з готовим analytics.liquidity.models.LiquidityMapSnapshot.
    """

    SIGNAL_TOPIC = "signal.generated"
    SNAPSHOT_FEATURE_NAME = "liquidity_map_snapshot"

    SNAPSHOT_DOMAIN_KEYS: tuple[str, ...] = (
        "snapshot",
        "liquidity_map_snapshot",
        "map_snapshot",
        "last_snapshot",
    )

    SNAPSHOT_FEATURE_KEYS: tuple[str, ...] = (
        "liquidity_map_snapshot",
        "liquidity.snapshot",
        "liquidity_snapshot",
        "liquidity.map.snapshot",
    )

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=logger
            or get_logger(
                __name__,
                service_name="strategy",
                event_type="liquidity_strategy",
            ),
        )

        self.validate_config()
        self._last_emitted_at: dict[str, datetime] = {}

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.LIQUIDITY

    @property
    def _strategy_cfg(self) -> StrategyDefinitionConfig | None:
        return self.config.get_strategy(self.strategy_name)

    @property
    def _runtime(self) -> Any:
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

    @property
    def priority(self) -> int:
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.priority if strategy_cfg is not None else 100

    def is_enabled(self) -> bool:
        strategy_cfg = self._strategy_cfg
        if strategy_cfg is None:
            return bool(getattr(self.config.runtime, "enabled", True))
        return bool(strategy_cfg.runtime.enabled)

    def required_features(self) -> set[str]:
        """
        Liquidity strategies завжди потребують LiquidityMapSnapshot.

        Якщо strategy config задає власні required_features, snapshot feature
        все одно додається примусово, бо дочірні liquidity strategy працюють
        саме від analytics/liquidity snapshot.
        """
        strategy_cfg = self._strategy_cfg

        if strategy_cfg is None:
            return {self.SNAPSHOT_FEATURE_NAME}

        configured = set(getattr(strategy_cfg, "required_features", set()) or set())
        configured.add(self.SNAPSHOT_FEATURE_NAME)
        return configured

    async def emit_signal(
        self,
        signal: StrategySignal,
        context: StrategyContext,
        **emit_kwargs: Any,
    ) -> StrategySignal | None:
        """
        Єдиний стандарт публікації StrategySignal в EventBus.

        Публікує lightweight summary + сам signal object для внутрішнього
        pipeline. Якщо StrategySignal має to_event_payload(), додає також
        payload-safe представлення для dashboard/storage/bots.
        """
        self._validate_signal_context_pair(signal=signal, context=context)

        if self._is_on_emit_cooldown(
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            timestamp=context.timestamp,
        ):
            self.log_debug(
                "Signal emit suppressed by cooldown",
                symbol=signal.symbol,
                timeframe=self._value(signal.timeframe),
                strategy_name=self.strategy_name,
            )
            return None

        signal_payload = self._to_payload(signal)

        payload: dict[str, Any] = {
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "category": self._value(signal.category),
            "timeframe": self._value(signal.timeframe),
            "side": self._value(signal.side),
            "score": signal.score,
            "confidence": signal.confidence,
            "status": self._value(signal.status),
            "priority": self._value(signal.priority),
            "source": self.strategy_name,
            "signal": signal,
            "signal_payload": signal_payload,
        }

        await self.emit_event(
            self.SIGNAL_TOPIC,
            payload,
            source=self.strategy_name,
            **emit_kwargs,
        )

        self._last_emitted_at[
            self._cooldown_key(signal.symbol, signal.timeframe)
        ] = context.timestamp

        self.log_info(
            "Strategy signal emitted",
            symbol=signal.symbol,
            timeframe=self._value(signal.timeframe),
            side=self._value(signal.side),
            confidence=signal.confidence,
            score=signal.score,
            strategy_name=self.strategy_name,
        )

        return signal

    def _extract_snapshot(self, context: StrategyContext) -> LiquidityMapSnapshot | None:
        """
        Дістає LiquidityMapSnapshot зі стандартних місць StrategyContext.

        Підтримує:
        - context.liquidity як dict-like object;
        - context.get_feature(...);
        - feature wrappers з .value / .data / .snapshot;
        - mapping wrappers з ключами value/data/snapshot.
        """
        liquidity_context = getattr(context, "liquidity", None)

        domain_candidates = [
            self._mapping_or_attr_get(liquidity_context, key)
            for key in self.SNAPSHOT_DOMAIN_KEYS
        ]

        feature_candidates: list[Any] = []
        for key in self.SNAPSHOT_FEATURE_KEYS:
            try:
                feature_candidates.append(context.get_feature(key))
            except Exception:
                self.log_debug(
                    "Failed to read liquidity feature from context",
                    symbol=getattr(context, "symbol", None),
                    feature_name=key,
                    strategy_name=self.strategy_name,
                )

        for candidate in [*domain_candidates, *feature_candidates]:
            snapshot = self._unwrap_snapshot_candidate(candidate)
            if snapshot is not None:
                return snapshot

        return None

    def _base_context_is_valid(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        """
        Загальна перевірка StrategyContext перед оцінкою liquidity setup.
        """
        if not self._snapshot_matches_context(context=context, snapshot=snapshot):
            return False

        if not self._is_symbol_allowed(context.symbol):
            self.log_debug(
                "Liquidity strategy skipped: symbol not allowed",
                symbol=context.symbol,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_timeframe_allowed(context.timeframe):
            self.log_debug(
                "Liquidity strategy skipped: timeframe not allowed",
                symbol=context.symbol,
                timeframe=self._value(context.timeframe),
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_regime_allowed(context):
            self.log_debug(
                "Liquidity strategy skipped: regime not allowed",
                symbol=context.symbol,
                timeframe=self._value(context.timeframe),
                regime=self._value(
                    context.regime.regime if context.regime is not None else None
                ),
                strategy_name=self.strategy_name,
            )
            return False

        if self._snapshot_is_stale(context, snapshot):
            self.log_debug(
                "Liquidity strategy skipped: stale liquidity snapshot",
                symbol=context.symbol,
                timeframe=self._value(snapshot.timeframe),
                snapshot_ts=snapshot.timestamp.isoformat(),
                context_ts=context.timestamp.isoformat(),
                strategy_name=self.strategy_name,
            )
            return False

        return True

    def _run_common_pre_filters(
            self,
            context: StrategyContext,
            snapshot: LiquidityMapSnapshot,
            current_price: float,
    ) -> list[FilterResult]:
        """
        Спільні pre-filters для liquidity strategies.

        Дочірні класи можуть додавати власні фільтри, але не повинні
        дублювати portfolio/spread/liquidity/price checks.
        """
        results: list[FilterResult] = []

        optional_filters: tuple[FilterResult | None, ...] = (
            self._portfolio_filter(context),
            self._spread_filter(context),
            self._liquidity_strength_filter(snapshot),
        )

        for filter_result in optional_filters:
            if filter_result is not None:
                results.append(filter_result)

        results.append(self._snapshot_presence_filter(snapshot))
        results.append(self._price_validation_filter(current_price))

        return results

    def _resolve_current_price(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> float | None:
        """
        Безпечно визначає current price.

        Пріоритет:
        1. context.price.mid_price;
        2. context.price.last_price;
        3. snapshot.current_price.
        """
        price_context = getattr(context, "price", None)

        for raw_price in (
            getattr(price_context, "mid_price", None),
            getattr(price_context, "last_price", None),
            getattr(snapshot, "current_price", None),
        ):
            price = self._safe_positive_float(raw_price)
            if price is not None:
                return price

        return None

    def _snapshot_is_stale(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        """
        Перевіряє freshness snapshot.

        Якщо StrategyContext має feature snapshot із власним is_stale(),
        використовує його. Інакше бере TTL із config.freshness.
        """
        try:
            feature = context.get_feature_snapshot(self.SNAPSHOT_FEATURE_NAME)
        except Exception:
            feature = None

        if feature is not None and hasattr(feature, "is_stale"):
            try:
                return bool(feature.is_stale(context.timestamp))
            except Exception:
                self.log_debug(
                    "Feature freshness check failed, falling back to TTL",
                    symbol=context.symbol,
                    timeframe=self._value(context.timeframe),
                    strategy_name=self.strategy_name,
                )

        ttl = self._resolve_snapshot_ttl_seconds()
        if ttl is None or ttl <= 0:
            return False

        if snapshot.timestamp is None or context.timestamp is None:
            return True

        age_seconds = abs((context.timestamp - snapshot.timestamp).total_seconds())
        return age_seconds > ttl

    def _is_symbol_allowed(self, symbol: str) -> bool:
        runtime = self._runtime
        allowed_symbols = set(getattr(runtime, "symbols", set()) or set())
        return not allowed_symbols or symbol in allowed_symbols

    def _is_timeframe_allowed(self, timeframe: Any) -> bool:
        runtime = self._runtime
        allowed_timeframes = getattr(runtime, "timeframes", None)

        if not allowed_timeframes:
            return True

        current = self._value(timeframe)
        allowed = {self._value(item) for item in allowed_timeframes}

        return current in allowed

    def _is_regime_allowed(self, context: StrategyContext) -> bool:
        runtime = self._runtime
        allowed_regimes = getattr(runtime, "allowed_regimes", None)

        if not allowed_regimes:
            return True

        regime = context.regime.regime if context.regime is not None else None
        if regime is None:
            return True

        current = self._value(regime)
        allowed = {self._value(item) for item in allowed_regimes}

        return current in allowed

    def _is_on_emit_cooldown(
        self,
        symbol: str,
        timeframe: Any,
        timestamp: datetime,
    ) -> bool:
        runtime = self._runtime
        cooldown_seconds = float(getattr(runtime, "emit_cooldown_seconds", 0) or 0)

        if cooldown_seconds <= 0:
            return False

        last_emitted_at = self._last_emitted_at.get(
            self._cooldown_key(symbol, timeframe)
        )
        if last_emitted_at is None:
            return False

        return timestamp - last_emitted_at < timedelta(seconds=cooldown_seconds)

    def _cooldown_key(self, symbol: str, timeframe: Any) -> str:
        return f"{self.strategy_name}:{symbol}:{self._value(timeframe)}"

    def _snapshot_matches_context(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        """
        Захищає strategy від snapshot іншого symbol/timeframe.
        """
        if snapshot.symbol != context.symbol:
            self.log_warning(
                "Liquidity strategy skipped: snapshot symbol mismatch",
                context_symbol=context.symbol,
                snapshot_symbol=snapshot.symbol,
                strategy_name=self.strategy_name,
            )
            return False

        if self._value(snapshot.timeframe) != self._value(context.timeframe):
            self.log_warning(
                "Liquidity strategy skipped: snapshot timeframe mismatch",
                symbol=context.symbol,
                context_timeframe=self._value(context.timeframe),
                snapshot_timeframe=self._value(snapshot.timeframe),
                strategy_name=self.strategy_name,
            )
            return False

        return True

    def _portfolio_filter(self, context: StrategyContext) -> FilterResult | None:
        portfolio = getattr(context, "portfolio", None)
        blocked_symbols = getattr(portfolio, "blocked_symbols", None)

        if not blocked_symbols:
            return None

        if context.symbol in blocked_symbols:
            return FilterResult(
                name="portfolio_blocked_symbol",
                decision=FilterDecision.BLOCK,
                reason=f"Symbol {context.symbol} is blocked by portfolio snapshot",
            )

        return FilterResult(
            name="portfolio_blocked_symbol",
            decision=FilterDecision.PASS,
            reason="Symbol is not blocked by portfolio snapshot",
        )

    def _spread_filter(self, context: StrategyContext) -> FilterResult | None:
        if not self.config.filters.enable_spread_filter:
            return None

        price_context = getattr(context, "price", None)
        if price_context is None:
            return FilterResult(
                name="spread_filter",
                decision=FilterDecision.PASS,
                reason="Price context unavailable; spread filter skipped",
            )

        spread_bps = getattr(price_context, "spread_bps", None)
        if spread_bps is None:
            return FilterResult(
                name="spread_filter",
                decision=FilterDecision.PASS,
                reason="Spread unavailable; spread filter skipped",
            )

        spread = self._safe_float(spread_bps)
        if spread is None:
            return FilterResult(
                name="spread_filter",
                decision=FilterDecision.BLOCK,
                reason="Spread value is invalid",
            )

        if spread > self.config.filters.max_spread_bps:
            return FilterResult(
                name="spread_filter",
                decision=FilterDecision.BLOCK,
                reason=f"Spread too high: {spread:.2f} bps",
            )

        return FilterResult(
            name="spread_filter",
            decision=FilterDecision.PASS,
            reason="Spread within threshold",
        )

    def _liquidity_strength_filter(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult | None:
        if not self.config.filters.enable_liquidity_filter:
            return None

        strongest_liquidity = max(
            float(snapshot.above_liquidity_score),
            float(snapshot.below_liquidity_score),
        )

        if strongest_liquidity < self.config.filters.min_liquidity_score:
            return FilterResult(
                name="liquidity_strength_filter",
                decision=FilterDecision.BLOCK,
                reason=f"Liquidity score too low: {strongest_liquidity:.4f}",
            )

        return FilterResult(
            name="liquidity_strength_filter",
            decision=FilterDecision.PASS,
            reason=f"Liquidity score OK: {strongest_liquidity:.4f}",
        )

    def _snapshot_presence_filter(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult:
        if not snapshot.has_levels():
            return FilterResult(
                name="liquidity_snapshot_presence",
                decision=FilterDecision.BLOCK,
                reason="Liquidity snapshot has no active levels or clusters",
            )

        return FilterResult(
            name="liquidity_snapshot_presence",
            decision=FilterDecision.PASS,
            reason="Liquidity snapshot contains active liquidity structures",
        )

    def _price_validation_filter(self, current_price: float) -> FilterResult:
        price = self._safe_positive_float(current_price)

        if price is None:
            return FilterResult(
                name="price_validation",
                decision=FilterDecision.BLOCK,
                reason="Current price must be positive and finite",
            )

        return FilterResult(
            name="price_validation",
            decision=FilterDecision.PASS,
            reason="Current price is valid",
        )

    def _resolve_snapshot_ttl_seconds(self) -> float | None:
        freshness = getattr(self.config, "freshness", None)
        if freshness is None:
            return None

        try:
            ttl = freshness.get_ttl(self.SNAPSHOT_FEATURE_NAME)
        except Exception:
            return None

        return self._safe_float(ttl)

    def _validate_signal_context_pair(
        self,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> None:
        if signal.symbol != context.symbol:
            raise ValueError(
                f"Signal/context symbol mismatch: "
                f"signal={signal.symbol}, context={context.symbol}"
            )

        if self._value(signal.timeframe) != self._value(context.timeframe):
            raise ValueError(
                f"Signal/context timeframe mismatch: "
                f"signal={self._value(signal.timeframe)}, "
                f"context={self._value(context.timeframe)}"
            )

        if signal.strategy_name != self.strategy_name:
            raise ValueError(
                f"Signal strategy mismatch: "
                f"signal={signal.strategy_name}, strategy={self.strategy_name}"
            )

    def _unwrap_snapshot_candidate(
        self,
        candidate: Any,
    ) -> LiquidityMapSnapshot | None:
        if isinstance(candidate, LiquidityMapSnapshot):
            return candidate

        if candidate is None:
            return None

        if isinstance(candidate, Mapping):
            for key in ("snapshot", "value", "data", "payload"):
                nested = candidate.get(key)
                if isinstance(nested, LiquidityMapSnapshot):
                    return nested

        for attr in ("snapshot", "value", "data", "payload"):
            nested = getattr(candidate, attr, None)
            if isinstance(nested, LiquidityMapSnapshot):
                return nested

        return None

    @staticmethod
    def _mapping_or_attr_get(source: Any, key: str, default: Any = None) -> Any:
        if source is None:
            return default

        if isinstance(source, Mapping):
            return source.get(key, default)

        return getattr(source, key, default)

    @staticmethod
    def _to_payload(value: Any) -> Any:
        to_event_payload = getattr(value, "to_event_payload", None)
        if callable(to_event_payload):
            return to_event_payload()

        to_payload = getattr(value, "to_payload", None)
        if callable(to_payload):
            return to_payload()

        return value

    @staticmethod
    def _value(value: Any) -> str:
        if value is None:
            return "unknown"

        raw = getattr(value, "value", value)
        return str(raw)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        if not isfinite(result):
            return None

        return result

    @classmethod
    def _safe_positive_float(cls, value: Any) -> float | None:
        result = cls._safe_float(value)
        if result is None or result <= 0:
            return None
        return result