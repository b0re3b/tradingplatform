from __future__ import annotations

from typing import Any, Callable

from core.config import Config
from core.event_bus import EventBus
from core.scheduler import Scheduler

from data.candles_cache import CandlesCache
from data.funding_cache import FundingCache
from data.market_stream import MarketStream
from data.open_interest_cache import OpenInterestCache
from data.orderbook_cache import OrderBookCache
from data.trades_cache import TradesCache

from exchanges.binance.binance_rest import BinanceRestClient
from exchanges.binance.binance_ws import BinanceWebSocketClient, BinanceWebSocketClientConfig
from exchanges.bybit.bybit_rest import BybitRestClient
from exchanges.bybit.bybit_ws import BybitWebSocketClient
from exchanges.okx.okx_rest import OkxRestClient
from exchanges.okx.okx_ws import OkxWebSocketClient
from exchanges.mexc.mexc_rest import MexcRestClient
from exchanges.mexc.mexc_ws import MexcWebSocketClient

from analytics.orderflow.analyzer import OrderFlowAnalyzer
from analytics.open_interest.oi_analyzer import OIAnalyzer
from analytics.funding.funding_analyzer import FundingAnalyzer
from analytics.liquidations.liquidation_stream import LiquidationStream
from analytics.liquidations.config import LiquidationStreamConfig
from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.liquidity_service import LiquidityService
from analytics.price_action.price_action_analyzer import PriceActionAnalyzer
from analytics.spoofing.analyzer import SpoofingAnalyzer
from analytics.spoofing.config import SpoofingConfig
from analytics.spreads.spread_analyzer import SpreadAnalyzer
from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import WhalesConfig

from strategy.engine import StrategyEngine
from strategy.presets import build_default_strategy_config, build_default_strategy_registry

from risk.config import RiskConfig
from risk.risk_manager import RiskManager

from execution.config import (
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
)
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from execution.sl_tp_manager import SLTPManager
from execution.smart_execution import SmartExecution
from execution.trade_executor import TradeExecutor

from bots.telegrambot.config import TelegramBotConfig
from bots.telegrambot.enums import TelegramTopic
from bots.telegrambot.service import TelegramBotService

from ai.config import build_default_news_ai_config
from ai.news_service import NewsAIService

from .runtime import RuntimeSettings, chunked
from .universe import ExchangeUniverse


# -----------------------------------------------------------------------------
# Exchanges / data
# -----------------------------------------------------------------------------


def build_rest_clients(config: Config, event_bus: EventBus, scheduler: Scheduler | None = None) -> dict[str, Any]:
    # Binance REST is both market-data capable and execution-capable.
    # Other REST clients are used for discovery/snapshots only.
    return {
        "binance": BinanceRestClient(config=config, event_bus=event_bus),
        "bybit": BybitRestClient(config=config, event_bus=event_bus),
        "okx": OkxRestClient(config=config, event_bus=event_bus),
        "mexc": MexcRestClient(config=config, event_bus=event_bus),
    }


def build_exchange_ws_clients(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    universe: ExchangeUniverse,
    settings: RuntimeSettings,
) -> dict[str, Any]:
    clients: dict[str, Any] = {}

    if "binance" in settings.market_data_exchanges:
        futures_ws_config = BinanceWebSocketClientConfig(
            public_ws_url="wss://fstream.binance.com/stream",
            private_ws_base_url="wss://fstream.binance.com/ws",
            rest_url="https://fapi.binance.com",
            symbols=[],
            streams=["trade", "depth", "kline", "forceorder"],
            depth_level="20",
            kline_interval="1m",
            orderbook_emit_min_interval_ms=500,
            orderbook_batch_max_size=2000,
            trade_emit_min_interval_ms=500,
            trade_batch_max_size=2000,
            enable_private_stream=False,
        )
        for index, symbols in enumerate(chunked(universe.binance, settings.ws_shard_size_binance), start=1):
            shard_config = BinanceWebSocketClientConfig(
                public_ws_url=futures_ws_config.public_ws_url,
                private_ws_base_url=futures_ws_config.private_ws_base_url,
                rest_url=futures_ws_config.rest_url,
                timeout_seconds=futures_ws_config.timeout_seconds,
                heartbeat_seconds=futures_ws_config.heartbeat_seconds,
                reconnect_delay_seconds=futures_ws_config.reconnect_delay_seconds,
                max_reconnect_attempts=futures_ws_config.max_reconnect_attempts,
                symbols=symbols,
                streams=futures_ws_config.streams,
                depth_level=futures_ws_config.depth_level,
                depth_speed=futures_ws_config.depth_speed,
                kline_interval=futures_ws_config.kline_interval,
                orderbook_emit_min_interval_ms=futures_ws_config.orderbook_emit_min_interval_ms,
                orderbook_batch_max_size=futures_ws_config.orderbook_batch_max_size,
                trade_emit_min_interval_ms=futures_ws_config.trade_emit_min_interval_ms,
                trade_batch_max_size=futures_ws_config.trade_batch_max_size,
                enable_private_stream=False,
            )
            clients[f"binance_{index}"] = BinanceWebSocketClient(
                config=config,
                event_bus=event_bus,
                scheduler=scheduler,
                ws_config=shard_config,
            )

    if "bybit" in settings.market_data_exchanges:
        for index, symbols in enumerate(chunked(universe.bybit, settings.ws_shard_size_bybit), start=1):
            clients[f"bybit_{index}"] = BybitWebSocketClient(
                config=config,
                event_bus=event_bus,
                symbols=symbols,
                streams=["trade", "orderbook", "kline", "liquidation"],
                category="linear",
                orderbook_depth=50,
                kline_interval="1",
                orderbook_emit_min_interval_ms=100,
                orderbook_batch_max_size=500,
                trade_emit_min_interval_ms=250,
                trade_batch_max_size=1000,
                enable_private_stream=False,
            )

    if "okx" in settings.market_data_exchanges:
        for index, inst_ids in enumerate(chunked(universe.okx, settings.ws_shard_size_okx), start=1):
            clients[f"okx_{index}"] = OkxWebSocketClient(
                config=config,
                event_bus=event_bus,
                inst_ids=inst_ids,
                streams=["trades", "books", "candle"],
                orderbook_channel="books5",
                candle_channel="candle1m",
                orderbook_emit_min_interval_ms=100,
                orderbook_batch_max_size=500,
                trade_emit_min_interval_ms=250,
                trade_batch_max_size=1000,
                enable_private_stream=False,
            )

    if "mexc" in settings.market_data_exchanges:
        for index, symbols in enumerate(chunked(universe.mexc, settings.ws_shard_size_mexc), start=1):
            clients[f"mexc_{index}"] = MexcWebSocketClient(
                config=config,
                event_bus=event_bus,
                symbols=symbols,
                streams=["deal", "depth", "kline"],
                kline_interval="Min1",
                orderbook_emit_min_interval_ms=100,
                orderbook_batch_max_size=500,
                trade_emit_min_interval_ms=250,
                trade_batch_max_size=1000,
                enable_private_stream=False,
            )

    return clients


def build_data_caches(config: Config, event_bus: EventBus, scheduler: Scheduler) -> dict[str, Any]:
    return {
        "orderbook": OrderBookCache(config=config, event_bus=event_bus, scheduler=scheduler),
        "trades": TradesCache(config=config, event_bus=event_bus, scheduler=scheduler),
        "candles": CandlesCache(config=config, event_bus=event_bus, scheduler=scheduler),
        "funding": FundingCache(config=config, event_bus=event_bus, scheduler=scheduler),
        "open_interest": OpenInterestCache(config=config, event_bus=event_bus, scheduler=scheduler),
    }


def build_market_stream(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    exchange_clients: dict[str, Any],
    caches: dict[str, Any],
) -> MarketStream:
    return MarketStream(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_clients=exchange_clients,
        caches=list(caches.values()),
    )


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------


def build_analytics_components(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    caches: dict[str, Any],
    universe: ExchangeUniverse,
) -> list[Any]:
    symbols = universe.all_canonical_symbols()
    binance_symbols = [symbol for symbol in universe.binance if str(symbol).strip()]
    price_action_symbols = binance_symbols or symbols or ["BTCUSDT"]
    price_action_timeframes = ("1m", "15m")

    # Dev-friendly defaults: liquidity still uses canonical cache-layer topics,
    # but it can build initial snapshots faster and also reacts to candles-cache
    # updates. For stricter production behavior, raise min_candles_for_snapshot
    # back to 30.
    liquidity_config = LiquidityConfig(
        candles_updated_input_topics=("market.candles.updated",),
        min_candles_for_snapshot=5,
    )
    liquidity_map = LiquidityMap(config=liquidity_config)

    components: list[Any] = [
        OrderFlowAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
            trades_cache=caches["trades"],
            orderbook_cache=caches["orderbook"],
            default_exchange="binance",
            default_market_type="usdm_futures",
            default_timeframe="1m",
        ),
        OIAnalyzer(event_bus=event_bus, scheduler=scheduler),
        FundingAnalyzer(event_bus=event_bus, scheduler=scheduler),
        LiquidationStream(
            event_bus=event_bus,
            scheduler=scheduler,
            config=LiquidationStreamConfig(),
        ),
        LiquidityService(
            event_bus=event_bus,
            scheduler=scheduler,
            config=liquidity_config,
            liquidity_map=liquidity_map,
        ),
        *[
            PriceActionAnalyzer(
                symbol,
                timeframe=timeframe,
                event_bus=event_bus,
                exchange="binance",
                market_type="usdm_futures",
                scheduler=scheduler,
            )
            for symbol in price_action_symbols
            for timeframe in price_action_timeframes
        ],
        SpoofingAnalyzer(
            event_bus=event_bus,
            scheduler=scheduler,
            config=SpoofingConfig(),
            orderbook_cache=caches["orderbook"],
        ),
        SpreadAnalyzer(event_bus=event_bus, scheduler=scheduler),
        WhaleAnalyzer(config=WhalesConfig(), event_bus=event_bus, scheduler=scheduler),
    ]
    return components


# -----------------------------------------------------------------------------
# Strategy factories
# -----------------------------------------------------------------------------


def build_strategy_factories() -> dict[str, Callable[..., Any]]:
    from strategy.strategies.orderflow import CvdDivergenceStrategy, OrderflowContinuationStrategy, OrderflowReversalStrategy
    from strategy.strategies.price_action import MarketStructureStrategy, FVGReactionStrategy, SupportResistanceReactionStrategy, TrendContinuationStrategy
    from strategy.strategies.open_interest import OIDivergenceStrategy, OIBreakoutConfirmationStrategy, OIAnomalyStrategy, OICapitulationStrategy
    from strategy.strategies.liquidations import LiquidationCascadeStrategy, SqueezeReversalStrategy
    from strategy.strategies.liquidity import EqualHighLowStrategy, LiquidityMapBiasStrategy, LiquiditySweepStrategy, StopHuntReversalStrategy
    from strategy.strategies.funding import FundingDivergenceStrategy, FundingExtremeReversalStrategy
    from strategy.strategies.spoofing import (
        SpoofingReversalStrategy,
        FakeLiquidityTrapStrategy,
        OrderPullReversalStrategy,
        PressureBluffReversalStrategy,
        LayeringTrapStrategy,
        SpoofingAbsorptionReversalStrategy,
        CompositeSpoofingStrategy,
    )
    from strategy.strategies.spreads import (
        SpotFuturesBasisStrategy,
        CrossExchangeArbStrategy,
        FundingAdjustedBasisStrategy,
        SpreadMeanReversionStrategy,
        SpreadMomentumStrategy,
    )
    from strategy.strategies.whales import (
        WhaleAbsorptionStrategy,
        WhaleBreakoutStrategy,
        WhaleAccumulationStrategy,
        WhaleDistributionStrategy,
        WhaleLiquidationReversalStrategy,
    )
    from strategy.strategies.hybrid.confluence_strategy import ConfluenceStrategy
    from strategy.strategies.hybrid.mean_reversion_stack_strategy import MeanReversionStackStrategy
    from strategy.strategies.hybrid.trend_stack_strategy import TrendStackStrategy
    from strategy.strategies.hybrid.liquidation_whale_strategy import LiquidationWhaleStrategy
    from strategy.strategies.hybrid.liquidity_orderflow_reversal_strategy import LiquidityOrderflowReversalStrategy
    from strategy.strategies.hybrid.oi_funding_squeeze_strategy import OIFundingSqueezeStrategy
    from strategy.strategies.hybrid.whale_orderflow_breakout_strategy import WhaleOrderflowBreakoutStrategy

    return {
        "cvd_divergence": CvdDivergenceStrategy,
        "orderflow_continuation": OrderflowContinuationStrategy,
        "orderflow_reversal": OrderflowReversalStrategy,
        "market_structure": MarketStructureStrategy,
        "fvg_reaction": FVGReactionStrategy,
        "support_resistance_reaction": SupportResistanceReactionStrategy,
        "trend_continuation": TrendContinuationStrategy,
        "oi_divergence": OIDivergenceStrategy,
        "oi_breakout_confirmation": OIBreakoutConfirmationStrategy,
        "oi_anomaly": OIAnomalyStrategy,
        "oi_capitulation": OICapitulationStrategy,
        "liquidation_cascade": LiquidationCascadeStrategy,
        "squeeze_reversal": SqueezeReversalStrategy,
        "equal_high_low": EqualHighLowStrategy,
        "liquidity_map_bias": LiquidityMapBiasStrategy,
        "liquidity_sweep": LiquiditySweepStrategy,
        "stop_hunt_reversal": StopHuntReversalStrategy,
        "funding_divergence": FundingDivergenceStrategy,
        "funding_extreme_reversal": FundingExtremeReversalStrategy,
        "spoofing_reversal": SpoofingReversalStrategy,
        "fake_liquidity_trap": FakeLiquidityTrapStrategy,
        "order_pull_reversal": OrderPullReversalStrategy,
        "pressure_bluff_reversal": PressureBluffReversalStrategy,
        "layering_trap": LayeringTrapStrategy,
        "spoofing_absorption_reversal": SpoofingAbsorptionReversalStrategy,
        "composite_spoofing": CompositeSpoofingStrategy,
        "spot_futures_basis": SpotFuturesBasisStrategy,
        "cross_exchange_arb": CrossExchangeArbStrategy,
        "funding_adjusted_basis": FundingAdjustedBasisStrategy,
        "spread_mean_reversion": SpreadMeanReversionStrategy,
        "spread_momentum": SpreadMomentumStrategy,
        "whale_absorption": WhaleAbsorptionStrategy,
        "whale_breakout": WhaleBreakoutStrategy,
        "whale_accumulation": WhaleAccumulationStrategy,
        "whale_distribution": WhaleDistributionStrategy,
        "whale_liquidation_reversal": WhaleLiquidationReversalStrategy,
        "confluence": ConfluenceStrategy,
        "mean_reversion_stack": MeanReversionStackStrategy,
        "trend_stack": TrendStackStrategy,
        "liquidation_whale": LiquidationWhaleStrategy,
        "liquidity_orderflow_reversal": LiquidityOrderflowReversalStrategy,
        "oi_funding_squeeze": OIFundingSqueezeStrategy,
        "whale_orderflow_breakout": WhaleOrderflowBreakoutStrategy,
    }


def build_strategy_engine(
    *,
    event_bus: EventBus,
    scheduler: Scheduler,
    universe: ExchangeUniverse,
) -> StrategyEngine:
    strategy_config = build_default_strategy_config(
        symbols=universe.all_canonical_symbols(),
        preset_name="default",
        use_required_features=False,
    )
    registry = build_default_strategy_registry(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        strategy_factories=build_strategy_factories(),
        strict=False,
    )
    return StrategyEngine(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        registry=registry,
    )


# -----------------------------------------------------------------------------
# Risk / execution / Telegram / News
# -----------------------------------------------------------------------------


def build_risk_manager(event_bus: EventBus, scheduler: Scheduler) -> RiskManager:
    return RiskManager(config=RiskConfig(), event_bus=event_bus, scheduler=scheduler)


def build_execution_components(
    *,
    event_bus: EventBus,
    scheduler: Scheduler,
    binance_rest: Any,
    settings: RuntimeSettings,
) -> list[Any]:
    # Safety: execution package must stay Binance USD-M Futures only.
    if settings.execution_exchange != "binance":
        raise ValueError("Execution exchange must be binance")
    if settings.execution_mode == "live" and not settings.live_trading_enabled:
        raise ValueError("Live execution requested but live trading flag is disabled")

    exchange_clients = {"binance": binance_rest}
    order_manager = OrderManager(
        OrderManagerConfig(),
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_clients=exchange_clients,
    )
    position_manager = PositionManager(
        PositionManagerConfig(),
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_clients=exchange_clients,
    )
    sltp_manager = SLTPManager(
        SLTPManagerConfig(),
        order_manager=order_manager,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    smart_execution = SmartExecution(SmartExecutionConfig())
    trade_executor = TradeExecutor(
        TradeExecutorConfig(),
        order_manager=order_manager,
        position_manager=position_manager,
        sltp_manager=sltp_manager,
        smart_execution=smart_execution,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    return [order_manager, position_manager, sltp_manager, trade_executor]


def build_telegram_service(event_bus: EventBus, scheduler: Scheduler) -> TelegramBotService | None:
    config = TelegramBotConfig.from_env()
    if not config.enabled:
        return None
    return TelegramBotService(config=config, event_bus=event_bus, scheduler=scheduler)


def build_news_service(event_bus: EventBus, scheduler: Scheduler) -> NewsAIService:
    # News is intentionally an independent Telegram/dashboard branch.
    # It should publish news.* / ai.news.* events only; StrategyEngine must not consume these.
    config = build_default_news_ai_config()
    return NewsAIService(config=config, event_bus=event_bus, scheduler=scheduler)


async def send_telegram_startup_message(telegram: TelegramBotService | None, text: str) -> None:
    if telegram is None:
        return
    await telegram.send_test_message(message=text, topic=TelegramTopic.SYSTEM)