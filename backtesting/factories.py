from __future__ import annotations

import importlib
import inspect
from typing import Any, Protocol

from core.config import Config
from core.event_bus import EventBus
from core.logger import get_logger
from core.scheduler import Scheduler

from backtesting.config import BacktestConfig
from backtesting.exceptions import BacktestFactoryError


class BacktestFactory(Protocol):
    async def build_caches(self, *, config: Config, event_bus: EventBus, scheduler: Scheduler) -> list[Any]:
        ...

    async def build_analytics(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        caches: list[Any],
        market_state_store: Any | None = None,
        market_scheduler: Any | None = None,
    ) -> list[Any]:
        ...

    async def build_strategy(self, *, config: Config, event_bus: EventBus, scheduler: Scheduler) -> list[Any]:
        ...

    async def build_risk(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        initial_balance: float,
    ) -> Any:
        ...


class ProductionBacktestFactory:
    """Default production-component wiring for the backtest pipeline."""

    def __init__(self, backtest_config: BacktestConfig) -> None:
        self._bt = backtest_config
        self._logger = get_logger(
            __name__,
            service="backtesting.factory",
            event_type="backtest_factory",
        )

    async def build_caches(self, *, config: Config, event_bus: EventBus, scheduler: Scheduler) -> list[Any]:
        services: list[Any] = []

        for import_path, kwargs in (
            ("data.candles_cache:CandlesCache", {"config": config, "event_bus": event_bus, "scheduler": scheduler}),
            ("data.trades_cache:TradesCache", {"config": config, "event_bus": event_bus, "scheduler": scheduler}),
            ("data.orderbook_cache:OrderBookCache", {"config": config, "event_bus": event_bus, "scheduler": scheduler}),
            ("data.funding_cache:FundingCache", {"config": config, "event_bus": event_bus, "scheduler": scheduler}),
            ("data.open_interest_cache:OpenInterestCache", {"config": config, "event_bus": event_bus, "scheduler": scheduler}),
        ):
            service = self._safe_construct(import_path, kwargs)
            if service is not None:
                services.append(service)

        return services

    async def build_analytics(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        caches: list[Any],
        market_state_store: Any | None = None,
        market_scheduler: Any | None = None,
    ) -> list[Any]:
        services: list[Any] = []

        if self._bt.enable_funding:
            funding = self._safe_construct(
                "analytics.funding:FundingAnalyzer",
                {"event_bus": event_bus, "scheduler": scheduler},
            )
            if funding is not None:
                services.append(funding)

        if self._bt.enable_open_interest:
            oi = self._safe_construct(
                "analytics.open_interest:OIAnalyzer",
                {"event_bus": event_bus, "scheduler": scheduler},
            )
            if oi is not None:
                services.append(oi)

        if self._bt.enable_orderflow:
            trades_cache = self._find(caches, "TradesCache")
            orderbook_cache = self._find(caches, "OrderBookCache")
            if trades_cache is not None and orderbook_cache is not None:
                orderflow = self._safe_construct(
                    "analytics.orderflow.analyzer:OrderFlowAnalyzer",
                    {
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                        "trades_cache": trades_cache,
                        "orderbook_cache": orderbook_cache,
                        "default_exchange": self._bt.exchange,
                        "default_market_type": self._bt.market_type,
                        "default_timeframe": self._bt.timeframes[0],
                    },
                )
                if orderflow is not None:
                    services.append(orderflow)

        for symbol in self._bt.symbols:
            for timeframe in self._bt.timeframes:
                price_action = self._safe_construct(
                    "analytics.price_action.price_action_analyzer:PriceActionAnalyzer",
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "event_bus": event_bus,
                        "scheduler": scheduler,
                        "exchange": self._bt.exchange,
                        "market_type": self._bt.market_type,

                        # critical for state-driven price_action
                        "market_state_store": market_state_store,
                        "market_scheduler": market_scheduler,
                    },
                )
                if price_action is not None:
                    services.append(price_action)

        return services

    async def build_strategy(self, *, config: Config, event_bus: EventBus, scheduler: Scheduler) -> list[Any]:
        try:
            from strategy.engine import StrategyEngine
            from strategy.presets import build_default_strategy_config, build_default_strategy_registry

            strategy_config = build_default_strategy_config(
                symbols=list(self._bt.symbols),
                preset_name="default",
                use_required_features=False,
            )

            registry = build_default_strategy_registry(
                config=strategy_config,
                event_bus=event_bus,
                scheduler=scheduler,
                strategy_factories=self._strategy_factories(),
                strict=False,
            )

            return [
                StrategyEngine(
                    config=strategy_config,
                    event_bus=event_bus,
                    scheduler=scheduler,
                    registry=registry,
                )
            ]
        except Exception as exc:
            raise BacktestFactoryError(f"Failed to build strategy layer: {exc}") from exc

    async def build_risk(
        self,
        *,
        config: Config,
        event_bus: EventBus,
        scheduler: Scheduler,
        initial_balance: float,
    ) -> Any:
        try:
            from risk.config import RiskConfig
            from risk.risk_manager import RiskManager
            from risk.state import RiskState

            risk_state = RiskState()
            self._update_risk_account(risk_state, initial_balance)

            return RiskManager(
                RiskConfig(),
                event_bus=event_bus,
                scheduler=scheduler,
                state=risk_state,
                service_name="risk_manager.backtest",
            )
        except Exception as exc:
            raise BacktestFactoryError(f"Failed to build risk layer: {exc}") from exc

    def _safe_construct(self, import_path: str, kwargs: dict[str, Any]) -> Any | None:
        try:
            cls = self._import_attr(import_path)
            return cls(**kwargs)
        except Exception as exc:
            self._logger.warning(
                "Backtest component skipped | component=%s error=%s",
                import_path,
                exc,
            )
            return None

    @staticmethod
    def _import_attr(import_path: str) -> Any:
        module_name, _, attr_name = import_path.partition(":")
        if not module_name or not attr_name:
            raise ValueError(f"Invalid import path: {import_path}")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    @staticmethod
    def _find(items: list[Any], class_name: str) -> Any | None:
        for item in items:
            if item.__class__.__name__ == class_name:
                return item
        return None

    def _update_risk_account(self, risk_state: Any, balance: float) -> None:
        update = getattr(risk_state, "update_account", None)
        if not callable(update):
            return

        signature = inspect.signature(update)
        available = signature.parameters
        kwargs: dict[str, Any] = {
            "balance": balance,
            "equity": balance,
            "free_balance": balance,
            "used_margin": 0.0,
            "margin_used": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
        }
        filtered = {key: value for key, value in kwargs.items() if key in available}
        update(**filtered)

    def _strategy_factories(self) -> dict[str, Any]:
        candidates: dict[str, str] = {
            "cvd_divergence": "strategy.strategies.orderflow:CvdDivergenceStrategy",
            "orderflow_continuation": "strategy.strategies.orderflow:OrderflowContinuationStrategy",
            "orderflow_reversal": "strategy.strategies.orderflow:OrderflowReversalStrategy",
            "market_structure": "strategy.strategies.price_action:MarketStructureStrategy",
            "fvg_reaction": "strategy.strategies.price_action:FVGReactionStrategy",
            "support_resistance_reaction": "strategy.strategies.price_action:SupportResistanceReactionStrategy",
            "trend_continuation": "strategy.strategies.price_action:TrendContinuationStrategy",
            "oi_anomaly": "strategy.strategies.open_interest:OIAnomalyStrategy",
            "oi_breakout_confirmation": "strategy.strategies.open_interest:OIBreakoutConfirmationStrategy",
            "oi_capitulation": "strategy.strategies.open_interest:OICapitulationStrategy",
            "oi_divergence": "strategy.strategies.open_interest:OIDivergenceStrategy",
            "confluence": "strategy.strategies.hybrid:ConfluenceStrategy",
            "mean_reversion_stack": "strategy.strategies.hybrid:MeanReversionStackStrategy",
            "trend_stack": "strategy.strategies.hybrid:TrendStackStrategy",
            "liquidation_whale": "strategy.strategies.hybrid:LiquidationWhaleStrategy",
            "liquidity_orderflow_reversal": "strategy.strategies.hybrid:LiquidityOrderflowReversalStrategy",
            "oi_funding_squeeze": "strategy.strategies.hybrid:OIFundingSqueezeStrategy",
            "whale_orderflow_breakout": "strategy.strategies.hybrid:WhaleOrderflowBreakoutStrategy",
        }

        factories: dict[str, Any] = {}
        disabled = set(self._bt.disabled_strategy_domains)

        for strategy_name, import_path in candidates.items():
            domain = import_path.split(".strategies.", 1)[-1].split(":", 1)[0].split(".", 1)[0]
            if domain in disabled:
                continue
            try:
                factories[strategy_name] = self._import_attr(import_path)
            except Exception as exc:
                self._logger.warning(
                    "Strategy factory skipped | strategy=%s import=%s error=%s",
                    strategy_name,
                    import_path,
                    exc,
                )

        return factories
