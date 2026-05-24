from __future__ import annotations

import inspect
from typing import Any, Callable

from core.config import Config
from core.event_bus import EventBus
from core.scheduler import Scheduler

from data.candles_cache import CandlesCache
from data.funding_cache import FundingCache
from data.market_stream import MarketStream
from data.market_state import MarketStateConfig, MarketStateStore
from data.market_ingestion import MarketIngestionConfig, MarketIngestionService
from data.market_scheduler import MarketScheduler, MarketSchedulerConfig
from data.open_interest_cache import OpenInterestCache
from data.liquidations_cache import LiquidationsCache
from data.orderbook_cache import OrderBookCache
from data.trades_cache import TradesCache
from storage.parquet_storage import ParquetStorage, ParquetStorageConfig

from exchanges.binance.binance_rest import BinanceFuturesRestClientConfig, BinanceRestClient
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
# Compatibility helpers
# -----------------------------------------------------------------------------


def _supports_kw(callable_obj: Any, key: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    return key in signature.parameters


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if _supports_kw(callable_obj, key)}


def _construct(cls: type[Any], *args: Any, **kwargs: Any) -> Any:
    return cls(*args, **_filter_supported_kwargs(cls, kwargs))


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    return func(**_filter_supported_kwargs(func, kwargs))


# -----------------------------------------------------------------------------
# Exchanges / data
# -----------------------------------------------------------------------------


def build_market_state_store(
    config: Config,
    event_bus: EventBus | None = None,
    scheduler: Scheduler | None = None,
    settings: RuntimeSettings | None = None,
) -> MarketStateStore:
    """Build the central state-driven market data store.

    Price-action warmup can require far more than the default 2_000 candles
    when 1m history is loaded for multiple days. Keep the state store large
    enough for the configured startup warmup horizon.
    """
    max_candles = int(getattr(settings, "market_state_max_candles_per_scope", 12_000)) if settings is not None else 12_000
    return MarketStateStore(
        config=MarketStateConfig(
            max_candles_per_scope=max(1, max_candles),
            default_market_type=(
                getattr(settings, "analytics_market_type", "usdm_futures")
                if settings is not None
                else "usdm_futures"
            ),
        )
    )


def build_market_ingestion_service(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler | None = None,
    market_state_store: MarketStateStore,
    settings: RuntimeSettings | None = None,
) -> MarketIngestionService:
    """Build the single write boundary for WS/REST/warmup market data."""
    ingestion_config = MarketIngestionConfig(
        default_exchange=getattr(settings, "analytics_exchange", "binance") if settings is not None else "binance",
        default_market_type=getattr(settings, "analytics_market_type", "usdm_futures") if settings is not None else "usdm_futures",
        default_timeframe=(
            str(getattr(settings, "timeframes", ["1m"])[0])
            if settings is not None and getattr(settings, "timeframes", None)
            else "1m"
        ),
        emit_persistable_events=bool(getattr(settings, "market_ingestion_emit_persistable_events", True)) if settings is not None else True,
        emit_trade_persistable_events=bool(getattr(settings, "market_ingestion_persist_trades", False)) if settings is not None else False,
        emit_orderbook_snapshot_persistable_events=bool(getattr(settings, "market_ingestion_persist_orderbook_snapshots", True)) if settings is not None else True,
        suppress_batch_candle_events=bool(getattr(settings, "market_ingestion_suppress_batch_candle_events", True)) if settings is not None else True,
    )
    return MarketIngestionService(
        state_store=market_state_store,
        event_bus=event_bus,
        config=ingestion_config,
    )


def build_market_scheduler(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    market_state_store: MarketStateStore,
    settings: RuntimeSettings | None = None,
) -> MarketScheduler:
    """Build controlled dirty-snapshot evaluator for state-driven analytics."""
    market_scheduler_config = MarketSchedulerConfig(
        enabled=True,
        interval_seconds=float(getattr(settings, "market_scheduler_interval_seconds", 1.0)) if settings is not None else 1.0,
        batch_size=int(getattr(settings, "market_scheduler_batch_size", 100)) if settings is not None else 100,
        snapshot_depth=int(getattr(settings, "market_scheduler_snapshot_depth", 50)) if settings is not None else 50,
        run_immediately=False,
        emit_snapshot_ready_events=False,
    )
    return MarketScheduler(
        state_store=market_state_store,
        scheduler=scheduler,
        event_bus=event_bus,
        config=market_scheduler_config,
    )


def _build_binance_market_data_rest_config(
    config: Config,
    settings: RuntimeSettings | None,
) -> BinanceFuturesRestClientConfig:
    """
    Production Binance USD-M Futures REST config for public market-data flows.

    This client is used for symbol discovery, startup historical candles,
    funding/open-interest snapshots and Parquet backfill.  It must not inherit
    the execution/testnet REST URL, because execution may intentionally run on
    a Binance demo/testnet account while market data must remain production.
    """
    rest_config = BinanceFuturesRestClientConfig.from_core_config(config)
    rest_config.rest_url = (
        str(getattr(settings, "binance_rest_url", "") or "https://fapi.binance.com").rstrip("/")
        if settings is not None
        else "https://fapi.binance.com"
    )
    rest_config.emit_success_events = False
    rest_config.allow_private_read_without_credentials = True

    if settings is not None:
        rest_config.derivative_snapshot_poll_concurrency = max(
            1,
            int(getattr(settings, "derivative_snapshot_poll_concurrency", rest_config.derivative_snapshot_poll_concurrency)),
        )
        rest_config.derivative_snapshot_poll_batch_size = max(
            1,
            int(getattr(settings, "derivative_snapshot_poll_batch_size", rest_config.derivative_snapshot_poll_batch_size)),
        )
        rest_config.derivative_snapshot_poll_interval_seconds = max(
            1.0,
            float(getattr(settings, "derivative_snapshot_poll_interval_seconds", rest_config.derivative_snapshot_poll_interval_seconds)),
        )

    return rest_config


def _build_binance_execution_rest_config(config: Config) -> BinanceFuturesRestClientConfig:
    """
    Execution Binance USD-M Futures REST config.

    This intentionally follows core Config / EXCHANGE_TESTNET / credentials.
    It is the only Binance REST client passed to execution services.
    """
    return BinanceFuturesRestClientConfig.from_core_config(config)


def build_rest_clients(
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler | None = None,
    settings: RuntimeSettings | None = None,
    market_ingestion: MarketIngestionService | None = None,
) -> dict[str, Any]:
    """
    Build REST clients with separated responsibilities.

    Binance is intentionally split into two clients:
    - ``binance`` / ``binance_market_data``: production public market-data REST;
    - ``binance_execution``: execution/demo/testnet REST from core Config.

    This lets the system fetch real historical/live market data while keeping
    order/account/position execution isolated on paper/demo/testnet.
    """
    enabled_market_data = set(
        settings.market_data_exchanges
        if settings is not None
        else ["binance", "bybit", "okx", "mexc"]
    )

    clients: dict[str, Any] = {}

    if "binance" in enabled_market_data:
        binance_market_data = _construct(
            BinanceRestClient,
            config=config,
            event_bus=event_bus,
            rest_config=_build_binance_market_data_rest_config(config, settings),
            market_ingestion=market_ingestion,
        )
        # Backward-compatible canonical key used by universe discovery and warmup.
        clients["binance"] = binance_market_data
        clients["binance_market_data"] = binance_market_data

    if settings is not None and settings.execution_exchange == "binance":
        clients["binance_execution"] = _construct(
            BinanceRestClient,
            config=config,
            event_bus=event_bus,
            rest_config=_build_binance_execution_rest_config(config),
            # Execution client must not write market snapshots into state.
            market_ingestion=None,
        )

    if "bybit" in enabled_market_data:
        clients["bybit"] = _construct(BybitRestClient, config=config, event_bus=event_bus, market_ingestion=market_ingestion)
    if "okx" in enabled_market_data:
        clients["okx"] = _construct(OkxRestClient, config=config, event_bus=event_bus, market_ingestion=market_ingestion)
    if "mexc" in enabled_market_data:
        clients["mexc"] = _construct(MexcRestClient, config=config, event_bus=event_bus, market_ingestion=market_ingestion)

    return clients


def build_exchange_ws_clients(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    universe: ExchangeUniverse,
    settings: RuntimeSettings,
    market_ingestion: MarketIngestionService | None = None,
) -> dict[str, Any]:
    """
    Build WS clients from .env-driven RuntimeSettings.

    No stream/channel/timeframe/depth/batch values are hard-coded here; each one
    is read from settings populated by RuntimeSettings.from_env().
    """
    clients: dict[str, Any] = {}

    if "binance" in settings.market_data_exchanges:
        futures_ws_config = BinanceWebSocketClientConfig(
            public_ws_url=settings.binance_public_ws_url,
            private_ws_base_url=settings.binance_private_ws_base_url,
            rest_url=settings.binance_rest_url,
            symbols=[],
            streams=list(settings.binance_ws_streams),
            depth_level=str(settings.binance_ws_depth_level),
            kline_interval=str(settings.binance_ws_kline_interval),
            orderbook_emit_min_interval_ms=settings.binance_orderbook_emit_min_interval_ms,
            orderbook_batch_max_size=settings.binance_orderbook_batch_max_size,
            trade_emit_min_interval_ms=settings.binance_trade_emit_min_interval_ms,
            trade_batch_max_size=settings.binance_trade_batch_max_size,
            enable_private_stream=settings.binance_enable_private_stream,
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
                enable_private_stream=futures_ws_config.enable_private_stream,
            )
            clients[f"binance_{index}"] = BinanceWebSocketClient(
                config=config,
                event_bus=event_bus,
                scheduler=scheduler,
                ws_config=shard_config,
                market_ingestion=market_ingestion,
            )

    if "bybit" in settings.market_data_exchanges:
        for index, symbols in enumerate(chunked(universe.bybit, settings.ws_shard_size_bybit), start=1):
            clients[f"bybit_{index}"] = BybitWebSocketClient(
                config=config,
                event_bus=event_bus,
                symbols=symbols,
                streams=list(settings.bybit_ws_streams),
                category=settings.bybit_category,
                orderbook_depth=settings.bybit_orderbook_depth,
                kline_interval=settings.bybit_kline_interval,
                orderbook_emit_min_interval_ms=settings.bybit_orderbook_emit_min_interval_ms,
                orderbook_batch_max_size=settings.bybit_orderbook_batch_max_size,
                trade_emit_min_interval_ms=settings.bybit_trade_emit_min_interval_ms,
                trade_batch_max_size=settings.bybit_trade_batch_max_size,
                enable_private_stream=settings.bybit_enable_private_stream,
                market_ingestion=market_ingestion,
            )

    if "okx" in settings.market_data_exchanges:
        for index, inst_ids in enumerate(chunked(universe.okx, settings.ws_shard_size_okx), start=1):
            clients[f"okx_{index}"] = OkxWebSocketClient(
                config=config,
                event_bus=event_bus,
                inst_ids=inst_ids,
                streams=list(settings.okx_ws_streams),
                orderbook_channel=settings.okx_orderbook_channel,
                candle_channel=settings.okx_candle_channel,
                orderbook_emit_min_interval_ms=settings.okx_orderbook_emit_min_interval_ms,
                orderbook_batch_max_size=settings.okx_orderbook_batch_max_size,
                trade_emit_min_interval_ms=settings.okx_trade_emit_min_interval_ms,
                trade_batch_max_size=settings.okx_trade_batch_max_size,
                enable_private_stream=settings.okx_enable_private_stream,
                market_ingestion=market_ingestion,
            )

    if "mexc" in settings.market_data_exchanges:
        for index, symbols in enumerate(chunked(universe.mexc, settings.ws_shard_size_mexc), start=1):
            clients[f"mexc_{index}"] = MexcWebSocketClient(
                config=config,
                event_bus=event_bus,
                symbols=symbols,
                streams=list(settings.mexc_ws_streams),
                kline_interval=settings.mexc_kline_interval,
                orderbook_emit_min_interval_ms=settings.mexc_orderbook_emit_min_interval_ms,
                orderbook_batch_max_size=settings.mexc_orderbook_batch_max_size,
                trade_emit_min_interval_ms=settings.mexc_trade_emit_min_interval_ms,
                trade_batch_max_size=settings.mexc_trade_batch_max_size,
                enable_private_stream=settings.mexc_enable_private_stream,
                market_ingestion=market_ingestion,
            )

    return clients


def build_data_caches(
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    *,
    market_state_store: MarketStateStore | None = None,
    market_ingestion: MarketIngestionService | None = None,
) -> dict[str, Any]:
    """Build state-driven cache facades.

    These objects no longer subscribe to high-frequency raw market.* topics when
    used with the new state layer; they expose the familiar read/apply API for
    analytics and compatibility.
    """
    common = {
        "config": config,
        "event_bus": event_bus,
        "scheduler": scheduler,
        "state_store": market_state_store,
        "market_state": market_state_store,
        "market_state_store": market_state_store,
        "ingestion": market_ingestion,
        "market_ingestion": market_ingestion,
    }
    return {
        "orderbook": _construct(OrderBookCache, **common),
        "trades": _construct(TradesCache, **common),
        "candles": _construct(CandlesCache, **common),
        "funding": _construct(FundingCache, **common),
        "open_interest": _construct(OpenInterestCache, **common),
        "liquidations": _construct(LiquidationsCache, **common),
    }



def build_parquet_storage(config: Config, event_bus: EventBus, scheduler: Scheduler) -> ParquetStorage:
    """
    EventBus-driven market-data persistence.

    Must be started before startup warmup so REST candle/funding snapshots that
    pass through caches are persisted to Parquet before strategy/risk/execution
    are allowed to start.
    """
    return ParquetStorage(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        storage_config=ParquetStorageConfig.from_core_config(config),
    )

def build_market_stream(
    *,
    config: Config,
    event_bus: EventBus,
    scheduler: Scheduler,
    exchange_clients: dict[str, Any],
    caches: dict[str, Any],
    market_state_store: MarketStateStore | None = None,
    market_ingestion: MarketIngestionService | None = None,
    market_scheduler: MarketScheduler | None = None,
) -> MarketStream:
    return _construct(
        MarketStream,
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        exchange_clients=exchange_clients,
        caches=list(caches.values()),
        market_state=market_state_store,
        state_store=market_state_store,
        ingestion=market_ingestion,
        market_scheduler=market_scheduler,
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
    settings: RuntimeSettings,
    market_state_store: MarketStateStore | None = None,
    market_scheduler: MarketScheduler | None = None,
) -> list[Any]:
    """
    Build analytics from .env-driven RuntimeSettings.

    Symbols/timeframes/exchange/market_type are not defined in this factory.
    They are resolved from RuntimeSettings.from_env() and exchange discovery.
    """
    discovered_symbols = universe.all_canonical_symbols()
    configured_analytics_symbols = [
        str(symbol).upper()
        for symbol in settings.analytics_symbols
        if str(symbol).strip()
    ]

    price_action_symbols = [
        str(symbol).upper()
        for symbol in (settings.price_action_symbols or configured_analytics_symbols)
        if str(symbol).strip()
    ]
    if not price_action_symbols:
        source_exchange_symbols = getattr(universe, settings.price_action_exchange, [])
        price_action_symbols = [str(symbol).upper() for symbol in source_exchange_symbols if str(symbol).strip()]
    if not price_action_symbols:
        price_action_symbols = discovered_symbols

    price_action_timeframes = [str(tf).strip() for tf in settings.price_action_timeframes if str(tf).strip()]
    if not price_action_timeframes:
        price_action_timeframes = [str(tf).strip() for tf in settings.timeframes if str(tf).strip()]

    liquidity_config = LiquidityConfig(
        candles_updated_input_topics=tuple(settings.liquidity_candles_updated_topics),
        min_candles_for_snapshot=settings.liquidity_min_candles_for_snapshot,
    )
    liquidity_map = LiquidityMap(config=liquidity_config)

    components: list[Any] = []

    def add_component(
        component: Any,
        *,
        evaluator_name: str | None = None,
        evaluator_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        components.append(component)
        if market_scheduler is not None:
            callback = getattr(component, "process_market_snapshot", None)
            if callable(callback):
                name = evaluator_name or f"{component.__class__.__name__}:{len(components)}"
                market_scheduler.register_evaluator(
                    name=name,
                    callback=callback,
                    **(evaluator_kwargs or {}),
                )
        return component

    add_component(
        _construct(
            OrderFlowAnalyzer,
            event_bus=event_bus,
            scheduler=scheduler,
            trades_cache=caches.get("trades"),
            orderbook_cache=caches.get("orderbook"),
            market_state_store=market_state_store,
            default_exchange=settings.orderflow_default_exchange,
            default_market_type=settings.orderflow_default_market_type,
            default_timeframe=settings.orderflow_default_timeframe,
        ),
        evaluator_name="analytics.orderflow",
        evaluator_kwargs={
            "exchange": settings.orderflow_default_exchange,
            "market_type": settings.orderflow_default_market_type,
            # Trades/orderbook updates are symbol-level and usually have no
            # candle timeframe.  OrderFlowAnalyzer still uses its default
            # timeframe internally for rolling-window semantics.
            "dirty_reasons": {"trade", "trades_batch", "orderbook", "orderbook_resync_required", "rest_snapshot"},
        },
    )

    add_component(
        _construct(OIAnalyzer, event_bus=event_bus, scheduler=scheduler, market_state_store=market_state_store),
        evaluator_name="analytics.open_interest",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"open_interest"},
        },
    )
    add_component(
        _construct(FundingAnalyzer, event_bus=event_bus, scheduler=scheduler, market_state_store=market_state_store),
        evaluator_name="analytics.funding",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"funding"},
        },
    )
    add_component(
        _construct(
            LiquidationStream,
            event_bus=event_bus,
            scheduler=scheduler,
            config=LiquidationStreamConfig(),
            market_state_store=market_state_store,
        ),
        evaluator_name="analytics.liquidations.stream",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"liquidation"},
        },
    )
    add_component(
        _construct(
            LiquidityService,
            event_bus=event_bus,
            scheduler=scheduler,
            config=liquidity_config,
            liquidity_map=liquidity_map,
            market_state_store=market_state_store,
        ),
        evaluator_name="analytics.liquidity",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"candle", "candle_closed", "trade", "trades_batch", "orderbook", "orderbook_resync_required", "rest_snapshot", "warmup"},
        },
    )

    for symbol in price_action_symbols:
        for timeframe in price_action_timeframes:
            add_component(
                _construct(
                    PriceActionAnalyzer,
                    symbol,
                    timeframe=timeframe,
                    event_bus=event_bus,
                    exchange=settings.price_action_exchange,
                    market_type=settings.price_action_market_type,
                    scheduler=scheduler,
                    market_state_store=market_state_store,
                ),
                evaluator_name=f"analytics.price_action:{symbol}:{timeframe}",
                evaluator_kwargs={
                    "exchange": settings.price_action_exchange,
                    "market_type": settings.price_action_market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "dirty_reasons": {"candle", "candle_closed", "warmup"},
                    "max_snapshots_per_tick": 1,
                },
            )

    add_component(
        _construct(
            SpoofingAnalyzer,
            event_bus=event_bus,
            scheduler=scheduler,
            config=SpoofingConfig(),
            orderbook_cache=caches.get("orderbook"),
            market_state_store=market_state_store,
        ),
        evaluator_name="analytics.spoofing",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"orderbook", "orderbook_resync_required", "rest_snapshot"},
        },
    )
    add_component(
        _construct(SpreadAnalyzer, event_bus=event_bus, scheduler=scheduler, market_state_store=market_state_store),
        evaluator_name="analytics.spreads",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"price", "funding", "open_interest"},
        },
    )
    add_component(
        _construct(WhaleAnalyzer, config=WhalesConfig(), event_bus=event_bus, scheduler=scheduler, market_state_store=market_state_store),
        evaluator_name="analytics.whales",
        evaluator_kwargs={
            "exchange": settings.analytics_exchange,
            "market_type": settings.analytics_market_type,
            "dirty_reasons": {"trade", "trades_batch", "liquidation"},
        },
    )

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
    settings: RuntimeSettings,
) -> StrategyEngine:
    strategy_config = build_default_strategy_config(
        symbols=universe.all_canonical_symbols(),
        preset_name=settings.strategy_preset_name,
        use_required_features=settings.strategy_use_required_features,
    )
    registry = build_default_strategy_registry(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        strategy_factories=build_strategy_factories(),
        strict=settings.strategy_registry_strict,
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