from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
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

    Виносить спільну інфраструктурну логіку:
    - logger
    - EventBus emit
    - StrategyConfig helpers
    - liquidity snapshot extraction
    - common filters
    - freshness validation
    - symbol/timeframe/regime validation
    - emit cooldown
    """

    SIGNAL_TOPIC = "signal.generated"
    SNAPSHOT_FEATURE_NAME = "liquidity_map_snapshot"

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
    def priority(self) -> int:
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.priority if strategy_cfg is not None else 100

    def is_enabled(self) -> bool:
        strategy_cfg = self._strategy_cfg
        if strategy_cfg is None:
            return True
        return strategy_cfg.runtime.enabled

    def required_features(self) -> set[str]:
        strategy_cfg = self._strategy_cfg
        if strategy_cfg is None:
            return {self.SNAPSHOT_FEATURE_NAME}
        return set(strategy_cfg.required_features)

    async def emit_signal(
        self,
        signal: StrategySignal,
        context: StrategyContext,
        **emit_kwargs: Any,
    ) -> StrategySignal | None:
        """
        Єдиний стандарт публікації сигналу в EventBus.
        """

        if self._is_on_emit_cooldown(signal.symbol, context.timestamp):
            self.log_debug(
                "Signal emit suppressed by cooldown",
                symbol=signal.symbol,
                timeframe=signal.timeframe.value,
                strategy_name=self.strategy_name,
            )
            return None

        payload = {
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "category": signal.category.value,
            "timeframe": signal.timeframe.value,
            "side": signal.side.value,
            "score": signal.score,
            "confidence": signal.confidence,
            "status": signal.status.value,
            "signal": signal,
        }

        await self.emit_event(
            self.SIGNAL_TOPIC,
            payload,
            source=self.strategy_name,
            **emit_kwargs,
        )

        self._last_emitted_at[signal.symbol] = context.timestamp

        self.log_info(
            "Strategy signal emitted",
            symbol=signal.symbol,
            timeframe=signal.timeframe.value,
            side=signal.side.value,
            confidence=signal.confidence,
            score=signal.score,
            strategy_name=self.strategy_name,
        )

        return signal

    def _extract_snapshot(self, context: StrategyContext) -> LiquidityMapSnapshot | None:
        """
        Дістає LiquidityMapSnapshot зі стандартних місць StrategyContext.
        """

        domain_candidates = [
            context.liquidity.get("snapshot"),
            context.liquidity.get("liquidity_map_snapshot"),
            context.liquidity.get("map_snapshot"),
            context.liquidity.get("last_snapshot"),
        ]

        feature_candidates = [
            context.get_feature("liquidity_map_snapshot"),
            context.get_feature("liquidity.snapshot"),
            context.get_feature("liquidity_snapshot"),
            context.get_feature("liquidity.map.snapshot"),
        ]

        for candidate in [*domain_candidates, *feature_candidates]:
            if isinstance(candidate, LiquidityMapSnapshot):
                return candidate

        return None

    def _is_symbol_allowed(self, symbol: str) -> bool:
        runtime = self._runtime
        return not runtime.symbols or symbol in runtime.symbols

    def _is_timeframe_allowed(self, timeframe: Any) -> bool:
        runtime = self._runtime
        return not runtime.timeframes or timeframe in runtime.timeframes

    def _is_regime_allowed(self, context: StrategyContext) -> bool:
        runtime = self._runtime

        if not runtime.allowed_regimes:
            return True

        regime = context.regime.regime if context.regime is not None else None
        if regime is None:
            return True

        return regime in runtime.allowed_regimes

    def _snapshot_is_stale(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        feature = context.get_feature_snapshot(self.SNAPSHOT_FEATURE_NAME)
        if feature is not None:
            return feature.is_stale(context.timestamp)

        ttl = self.config.freshness.get_ttl(self.SNAPSHOT_FEATURE_NAME)
        age_seconds = abs((context.timestamp - snapshot.timestamp).total_seconds())
        return age_seconds > ttl

    def _resolve_current_price(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> float | None:
        if context.price is not None:
            if context.price.mid_price is not None:
                return float(context.price.mid_price)
            if context.price.last_price is not None:
                return float(context.price.last_price)

        return float(snapshot.current_price) if snapshot.current_price > 0 else None

    def _run_common_pre_filters(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        """
        Спільні фільтри для liquidity strategies.
        Дочірні класи можуть додавати власні фільтри окремо.
        """

        results: list[FilterResult] = []

        if context.portfolio is not None and context.symbol in context.portfolio.blocked_symbols:
            results.append(
                FilterResult(
                    name="portfolio_blocked_symbol",
                    decision=FilterDecision.BLOCK,
                    reason=f"Symbol {context.symbol} is blocked by portfolio snapshot",
                )
            )

        if self.config.filters.enable_spread_filter and context.price is not None:
            spread_bps = context.price.spread_bps

            if spread_bps is not None and spread_bps > self.config.filters.max_spread_bps:
                results.append(
                    FilterResult(
                        name="spread_filter",
                        decision=FilterDecision.BLOCK,
                        reason=f"Spread too high: {spread_bps:.2f} bps",
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="spread_filter",
                        decision=FilterDecision.PASS,
                        reason="Spread within threshold",
                    )
                )

        if self.config.filters.enable_liquidity_filter:
            strongest_liquidity = max(
                snapshot.above_liquidity_score,
                snapshot.below_liquidity_score,
            )

            if strongest_liquidity < self.config.filters.min_liquidity_score:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.BLOCK,
                        reason=f"Liquidity score too low: {strongest_liquidity:.4f}",
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="liquidity_strength_filter",
                        decision=FilterDecision.PASS,
                        reason=f"Liquidity score OK: {strongest_liquidity:.4f}",
                    )
                )

        if not snapshot.has_levels():
            results.append(
                FilterResult(
                    name="liquidity_snapshot_presence",
                    decision=FilterDecision.BLOCK,
                    reason="Liquidity snapshot has no active levels or clusters",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="liquidity_snapshot_presence",
                    decision=FilterDecision.PASS,
                    reason="Liquidity snapshot contains active liquidity structures",
                )
            )

        if current_price <= 0:
            results.append(
                FilterResult(
                    name="price_validation",
                    decision=FilterDecision.BLOCK,
                    reason="Current price must be positive",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="price_validation",
                    decision=FilterDecision.PASS,
                    reason="Current price is valid",
                )
            )

        return results

    def _base_context_is_valid(
        self,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        """
        Загальна перевірка StrategyContext перед оцінкою setup.
        """

        if not self._is_symbol_allowed(context.symbol):
            return False

        if not self._is_timeframe_allowed(context.timeframe):
            return False

        if not self._is_regime_allowed(context):
            return False

        if self._snapshot_is_stale(context, snapshot):
            self.log_debug(
                "Liquidity strategy skipped: stale liquidity snapshot",
                symbol=context.symbol,
                timeframe=str(snapshot.timeframe),
                snapshot_ts=snapshot.timestamp.isoformat(),
                context_ts=context.timestamp.isoformat(),
                strategy_name=self.strategy_name,
            )
            return False

        return True

    def _is_on_emit_cooldown(self, symbol: str, timestamp: datetime) -> bool:
        runtime = self._runtime
        cooldown_seconds = getattr(runtime, "emit_cooldown_seconds", 0)

        if cooldown_seconds <= 0:
            return False

        last_emitted_at = self._last_emitted_at.get(symbol)
        if last_emitted_at is None:
            return False

        return timestamp - last_emitted_at < timedelta(seconds=cooldown_seconds)

    @property
    def _runtime(self):
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime