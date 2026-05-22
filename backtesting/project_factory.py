from __future__ import annotations

from typing import Any

from core.config import Config
from core.event_bus import EventBus
from core.scheduler import Scheduler

from data.candles_cache import CandlesCache
from data.trades_cache import TradesCache
from data.orderbook_cache import OrderBookCache
from data.funding_cache import FundingCache
from data.open_interest_cache import OpenInterestCache

from risk.config import RiskConfig
from risk.risk_manager import RiskManager
from risk.state import RiskState

from strategy.engine import StrategyEngine
from strategy.presets import (
    build_default_strategy_config,
    build_default_strategy_registry,
)


class ProductionBacktestFactory:
    """
    Wires real project components for backtesting.

    This is not a fake factory:
    - uses real caches;
    - uses real StrategyEngine / SignalProcessor / StrategyRegistry;
    - uses real RiskManager with isolated paper RiskState;
    - execution remains handled by BacktestExecutionSimulator.
    """

    async def build_caches(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> list[Any]:
        return [
            CandlesCache(config=config, event_bus=event_bus, scheduler=scheduler),
            TradesCache(config=config, event_bus=event_bus, scheduler=scheduler),
            OrderBookCache(config=config, event_bus=event_bus, scheduler=scheduler),
            FundingCache(config=config, event_bus=event_bus, scheduler=scheduler),
            OpenInterestCache(config=config, event_bus=event_bus, scheduler=scheduler),
        ]

    async def build_analytics(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        caches: list[Any],
    ) -> list[Any]:
        services: list[Any] = []

        candles_cache = _find(caches, "CandlesCache")
        trades_cache = _find(caches, "TradesCache")
        orderbook_cache = _find(caches, "OrderBookCache")

        # Funding analytics
        try:
            from analytics.funding import FundingAnalyzer

            services.append(
                FundingAnalyzer(
                    event_bus=event_bus,
                    scheduler=scheduler,
                )
            )
        except Exception:
            pass

        # Open Interest analytics
        try:
            from analytics.open_interest import OIAnalyzer

            services.append(
                OIAnalyzer(
                    event_bus=event_bus,
                    scheduler=scheduler,
                )
            )
        except Exception:
            pass

        # Orderflow analytics
        if trades_cache is not None and orderbook_cache is not None:
            try:
                from analytics.orderflow.analyzer import OrderFlowAnalyzer

                services.append(
                    OrderFlowAnalyzer(
                        event_bus=event_bus,
                        scheduler=scheduler,
                        trades_cache=trades_cache,
                        orderbook_cache=orderbook_cache,
                        default_exchange="binance",
                        default_market_type="usdm_futures",
                        default_timeframe="1m",
                    )
                )
            except Exception:
                pass

        # Price action analytics per symbol/timeframe
        try:
            from analytics.price_action.price_action_analyzer import PriceActionAnalyzer

            for symbol in ("BTCUSDT", "ETHUSDT", "RIVERUSDT"):
                for timeframe in ("1m", "15m"):
                    services.append(
                        PriceActionAnalyzer(
                            symbol=symbol,
                            timeframe=timeframe,
                            event_bus=event_bus,
                            scheduler=scheduler,
                            exchange="binance",
                            market_type="usdm_futures",
                        )
                    )
        except Exception:
            pass

        return services

    async def build_strategy(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
    ) -> list[Any]:
        strategy_config = build_default_strategy_config(
            symbols=["BTCUSDT", "ETHUSDT", "RIVERUSDT"],
            preset_name="default",
            use_required_features=False,
        )

        registry = build_default_strategy_registry(
            config=strategy_config,
            event_bus=event_bus,
            scheduler=scheduler,
            strategy_factories=_strategy_factories(),
            strict=False,
        )

        engine = StrategyEngine(
            config=strategy_config,
            event_bus=event_bus,
            scheduler=scheduler,
            registry=registry,
        )

        return [engine]

    async def build_risk(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        initial_balance: float,
    ) -> Any:
        risk_state = RiskState()
        risk_state.update_account(
            balance=initial_balance,
            equity=initial_balance,
            free_balance=initial_balance,
            used_margin=0.0,
        )

        return RiskManager(
            RiskConfig(),
            event_bus=event_bus,
            scheduler=scheduler,
            state=risk_state,
            service_name="risk_manager.backtest",
        )


def _find(items: list[Any], class_name: str) -> Any | None:
    for item in items:
        if item.__class__.__name__ == class_name:
            return item
    return None


def _strategy_factories() -> dict[str, Any]:
    from strategy.strategies import (
        CvdDivergenceStrategy,
        OrderflowContinuationStrategy,
        OrderflowReversalStrategy,
        MarketStructureStrategy,
        FVGReactionStrategy,
        SupportResistanceReactionStrategy,
        TrendContinuationStrategy,
        OIAnomalyStrategy,
        OIBreakoutConfirmationStrategy,
        OICapitulationStrategy,
        OIDivergenceStrategy,
        FundingDivergenceStrategy,
        FundingExtremeReversalStrategy,
        ConfluenceStrategy,
        MeanReversionStackStrategy,
        TrendStackStrategy,
        OIFundingSqueezeStrategy,
        WhaleOrderflowBreakoutStrategy,
        LiquidityOrderflowReversalStrategy,
    )

    return {
        "cvd_divergence": CvdDivergenceStrategy,
        "orderflow_continuation": OrderflowContinuationStrategy,
        "orderflow_reversal": OrderflowReversalStrategy,
        "market_structure": MarketStructureStrategy,
        "fvg_reaction": FVGReactionStrategy,
        "support_resistance_reaction": SupportResistanceReactionStrategy,
        "trend_continuation": TrendContinuationStrategy,
        "oi_anomaly": OIAnomalyStrategy,
        "oi_breakout_confirmation": OIBreakoutConfirmationStrategy,
        "oi_capitulation": OICapitulationStrategy,
        "oi_divergence": OIDivergenceStrategy,
        "funding_divergence": FundingDivergenceStrategy,
        "funding_extreme_reversal": FundingExtremeReversalStrategy,
        "confluence": ConfluenceStrategy,
        "mean_reversion_stack": MeanReversionStackStrategy,
        "trend_stack": TrendStackStrategy,
        "oi_funding_squeeze": OIFundingSqueezeStrategy,
        "whale_orderflow_breakout": WhaleOrderflowBreakoutStrategy,
        "liquidity_orderflow_reversal": LiquidityOrderflowReversalStrategy,
    }