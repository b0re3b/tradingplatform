# tests/analytics/orderflow/test_trade_flow_analyzers.py

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.orderflow.aggressive_trades import AggressiveTradesAnalyzer
from analytics.orderflow.config import (
    AggressiveTradesConfig,
    CvdConfig,
    VolumeDeltaConfig,
)
from analytics.orderflow.cvd import CvdAnalyzer
from analytics.orderflow.enums import OrderFlowEventTopic, OrderFlowSide
from analytics.orderflow.models import (
    NormalizedTrade,
    OrderFlowKey,
    make_orderflow_key,
    orderflow_key_to_dict,
    orderflow_key_to_string,
)
from analytics.orderflow.volume_delta import VolumeDeltaAnalyzer


# =============================================================================
# Constants
# =============================================================================

TRADES_TOPIC = "market.trades.updated"

DEFAULT_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="BTCUSDT",
    timeframe="1m",
)

BYBIT_KEY: OrderFlowKey = make_orderflow_key(
    exchange="bybit",
    market_type="linear",
    symbol="BTCUSDT",
    timeframe="1m",
)

ETH_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="ETHUSDT",
    timeframe="1m",
)

HIGHER_TF_KEY: OrderFlowKey = make_orderflow_key(
    exchange="binance",
    market_type="usdm_futures",
    symbol="BTCUSDT",
    timeframe="5m",
)


# =============================================================================
# Fakes
# =============================================================================


@dataclass(slots=True)
class FakeSubscription:
    pattern: str
    handler: Any
    name: str
    enabled: bool = True


@dataclass(slots=True)
class FakeScheduledJob:
    job_id: str
    name: str
    func: Any
    interval: float
    args: tuple
    kwargs: dict[str, Any]
    run_immediately: bool
    max_retries: int
    retry_delay: float
    timeout: float | None
    allow_overlap: bool
    enabled: bool = True


class FakeEventBus:
    """
    Strict EventBus fake.

    It records subscriptions, unsubscriptions and emitted analytics events.
    It can simulate emit failures to verify that analyzers do not crash when
    EventBus is temporarily unavailable.
    """

    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscribed: list[FakeSubscription] = []
        self.emitted: list[dict[str, Any]] = []
        self.fail_emit_topics: set[str] = set()

    def subscribe(
        self,
        pattern: str,
        handler: Any,
        *,
        name: str | None = None,
    ) -> FakeSubscription:
        subscription = FakeSubscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        if subscription not in self.subscriptions:
            raise RuntimeError(f"Unknown subscription: {subscription!r}")

        self.subscriptions.remove(subscription)
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        if topic in self.fail_emit_topics:
            raise RuntimeError(f"Simulated EventBus failure for topic={topic}")

        self.emitted.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": headers or {},
            }
        )
        return True


class FakeScheduler:
    """
    Strict Scheduler fake.

    BaseOrderFlowAnalyzer expects get_job_by_name() before add_interval_job().
    This fake also tracks removed jobs so lifecycle leaks are caught.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.disabled_job_ids: list[str] = []
        self.removed_job_ids: list[str] = []

    def add_interval_job(
        self,
        name: str,
        func: Any,
        *,
        interval: float,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
    ) -> str:
        if interval <= 0:
            raise ValueError("interval must be > 0")

        existing = self.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = FakeScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            args=args,
            kwargs=kwargs or {},
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            enabled=enabled,
        )
        return job_id

    def get_job_by_name(self, name: str) -> FakeScheduledJob | None:
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def disable_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.jobs[job_id].enabled = False
        self.disabled_job_ids.append(job_id)

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id)


class StrictTradesCache:
    """
    Scoped data-layer TradesCache fake.

    Production contract tested here:
        get_recent_trades(
            exchange=...,
            market_type=...,
            symbol=...,
            timeframe=...,
            limit=...,
        )

    It also supports get_trades_since(), get_trades() and get() with scoped
    kwargs because the analyzers currently support a migration lookup chain.
    """

    def __init__(
        self,
        trades_by_key: dict[OrderFlowKey, Any] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.trades_by_key = dict(trades_by_key or {})
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def set_trades(self, key: OrderFlowKey, trades: Any) -> None:
        self.trades_by_key[key] = trades

    def get_recent_trades(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "get_recent_trades",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
            }
        )
        return self._return(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or "1m",
        )

    def get_trades_since(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        since_ts: float,
    ) -> Any:
        self.calls.append(
            {
                "method": "get_trades_since",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "since_ts": since_ts,
            }
        )
        return self._return(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe="1m",
        )

    def get_trades(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        limit: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "method": "get_trades",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "limit": limit,
            }
        )
        return self._return(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe="1m",
        )

    def get(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
    ) -> Any:
        self.calls.append(
            {
                "method": "get",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
            }
        )
        return self._return(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe="1m",
        )

    def _return(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
    ) -> Any:
        if self.fail:
            raise RuntimeError("Simulated trades cache failure")

        scoped_key = make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.trades_by_key.get(scoped_key, [])


class AsyncTradesCache:
    """
    Async TradesCache fake.

    Protects analyzers from assuming sync-only cache methods.
    """

    def __init__(self, trades_by_key: dict[OrderFlowKey, Any]) -> None:
        self.trades_by_key = dict(trades_by_key)
        self.calls: list[dict[str, Any]] = []

    async def get_recent_trades(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
        limit: int | None = None,
    ) -> Any:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "method": "get_recent_trades",
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
            }
        )
        scoped_key = make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or "1m",
        )
        return self.trades_by_key.get(scoped_key, [])


class LegacySymbolOnlyTradesCache:
    """
    Compatibility fake exposing only symbol-only get().

    This is not the production contract. It exists to keep migration behavior
    explicit and separated from the scoped production cache tests.
    """

    def __init__(self, trades: Any) -> None:
        self.trades = trades
        self.calls: list[tuple[str, str]] = []

    def get(self, symbol: str) -> Any:
        self.calls.append(("get", symbol))
        return self.trades


# =============================================================================
# Factories
# =============================================================================


def now_ts(offset: float = 0.0) -> float:
    return time.time() + offset


def key(
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
) -> OrderFlowKey:
    return make_orderflow_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def raw_trade(
    *,
    trade_id: str | int | None,
    side: str,
    quantity: Any,
    price: Any = 100.0,
    ts: float | None = None,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    is_aggressive: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope = orderflow_key_to_dict(orderflow_key)

    payload = {
        **scope,
        "exchange_symbol": scope["symbol"],
        "side": side,
        "price": price,
        "quantity": quantity,
        "timestamp": now_ts() if ts is None else ts,
        "is_aggressive": is_aggressive,
    }

    if trade_id is not None:
        payload["trade_id"] = str(trade_id)

    if extra:
        payload.update(extra)

    return payload


def buy(
    trade_id: str | int | None,
    quantity: Any,
    *,
    price: Any = 100.0,
    ts: float | None = None,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    is_aggressive: bool = True,
) -> dict[str, Any]:
    return raw_trade(
        trade_id=trade_id,
        side="buy",
        quantity=quantity,
        price=price,
        ts=ts,
        orderflow_key=orderflow_key,
        is_aggressive=is_aggressive,
    )


def sell(
    trade_id: str | int | None,
    quantity: Any,
    *,
    price: Any = 100.0,
    ts: float | None = None,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    is_aggressive: bool = True,
) -> dict[str, Any]:
    return raw_trade(
        trade_id=trade_id,
        side="sell",
        quantity=quantity,
        price=price,
        ts=ts,
        orderflow_key=orderflow_key,
        is_aggressive=is_aggressive,
    )


def malformed_trades() -> list[Any]:
    return [
        None,
        "bad-trade",
        {"symbol": "BTCUSDT", "side": "buy"},
        {
            **orderflow_key_to_dict(DEFAULT_KEY),
            "side": "unknown",
            "price": 100,
            "quantity": 1,
            "timestamp": now_ts(),
        },
        {
            **orderflow_key_to_dict(DEFAULT_KEY),
            "side": "buy",
            "price": 0,
            "quantity": 1,
            "timestamp": now_ts(),
        },
        {
            **orderflow_key_to_dict(DEFAULT_KEY),
            "side": "sell",
            "price": 100,
            "quantity": -1,
            "timestamp": now_ts(),
        },
        {
            **orderflow_key_to_dict(DEFAULT_KEY),
            "side": "buy",
            "price": "nan",
            "quantity": 1,
            "timestamp": now_ts(),
        },
        {
            **orderflow_key_to_dict(DEFAULT_KEY),
            "side": "buy",
            "price": 100,
            "quantity": float("inf"),
            "timestamp": now_ts(),
        },
    ]


def cache_result(trades: Any) -> dict[str, Any]:
    return {"data": {"trades": trades}}


def make_cvd_config(**overrides: Any) -> CvdConfig:
    values = {
        "enabled": True,
        "emit_updates": True,
        "emit_signals": True,
        "min_signal_interval_sec": 3600.0,
        "health_log_interval_sec": 5.0,
        "cleanup_interval_sec": 5.0,
        "scheduler_job_timeout_sec": 0.5,
        "scheduler_job_retry_delay_sec": 0.1,
        "scheduler_job_max_retries": 1,
        "publish_priority": EventPriority.HIGH,
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
        "window_seconds": 30.0,
        "max_trades_per_symbol": 100,
        "max_cvd_points_per_symbol": 100,
        "min_trades_in_window": 2,
        "min_total_volume": 0.0,
        "bullish_delta_ratio_threshold": 0.2,
        "bearish_delta_ratio_threshold": -0.2,
        "bullish_cvd_change_threshold": 0.0,
        "bearish_cvd_change_threshold": 0.0,
        "bullish_cvd_slope_threshold": 0.0,
        "bearish_cvd_slope_threshold": 0.0,
        "bullish_impulse_threshold_pct": 0.0,
        "bearish_impulse_threshold_pct": 0.0,
        "require_delta_confirmation": False,
        "require_slope_confirmation": False,
    }
    values.update(overrides)

    config = CvdConfig(**values)
    config.validate()
    return config


def make_volume_delta_config(**overrides: Any) -> VolumeDeltaConfig:
    values = {
        "enabled": True,
        "emit_updates": True,
        "emit_signals": True,
        "min_signal_interval_sec": 3600.0,
        "health_log_interval_sec": 5.0,
        "cleanup_interval_sec": 5.0,
        "scheduler_job_timeout_sec": 0.5,
        "scheduler_job_retry_delay_sec": 0.1,
        "scheduler_job_max_retries": 1,
        "publish_priority": EventPriority.HIGH,
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
        "window_seconds": 30.0,
        "max_trades_per_symbol": 100,
        "min_trades_in_window": 2,
        "min_total_volume": 0.0,
        "bullish_delta_ratio_threshold": 0.2,
        "bearish_delta_ratio_threshold": -0.2,
        "bullish_volume_delta_threshold": 0.0,
        "bearish_volume_delta_threshold": 0.0,
        "bullish_cumulative_delta_threshold": 0.0,
        "bearish_cumulative_delta_threshold": 0.0,
        "require_ratio_and_absolute_confirmation": True,
    }
    values.update(overrides)

    config = VolumeDeltaConfig(**values)
    config.validate()
    return config


def make_aggressive_config(**overrides: Any) -> AggressiveTradesConfig:
    values = {
        "enabled": True,
        "emit_updates": True,
        "emit_signals": True,
        "min_signal_interval_sec": 3600.0,
        "health_log_interval_sec": 5.0,
        "cleanup_interval_sec": 5.0,
        "scheduler_job_timeout_sec": 0.5,
        "scheduler_job_retry_delay_sec": 0.1,
        "scheduler_job_max_retries": 1,
        "publish_priority": EventPriority.HIGH,
        "allowed_market_types": {"usdm_futures", "linear", "swap"},
        "window_seconds": 30.0,
        "max_trades_per_symbol": 100,
        "min_trades_in_window": 2,
        "bullish_buy_ratio_threshold": 0.65,
        "bearish_sell_ratio_threshold": 0.65,
        "bullish_delta_threshold": 0.0,
        "bearish_delta_threshold": 0.0,
        "large_trade_notional_threshold": 1_000.0,
        "min_large_trades_for_signal": 1,
        "burst_trades_threshold": 3,
        "burst_volume_threshold": 0.0,
        "burst_score_threshold": 1.0,
    }
    values.update(overrides)

    config = AggressiveTradesConfig(**values)
    config.validate()
    return config


def make_cvd_analyzer(
    trades: Any,
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: CvdConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> tuple[CvdAnalyzer, FakeEventBus, Any, FakeScheduler]:
    bus = event_bus or FakeEventBus()
    scheduler_obj = scheduler or FakeScheduler()
    trades_cache = cache or StrictTradesCache({orderflow_key: trades})

    analyzer = CvdAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=scheduler_obj,  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_cvd_config(),
        source_topic_patterns=(TRADES_TOPIC,),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )
    return analyzer, bus, trades_cache, scheduler_obj


def make_volume_delta_analyzer(
    trades: Any,
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: VolumeDeltaConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> tuple[VolumeDeltaAnalyzer, FakeEventBus, Any, FakeScheduler]:
    bus = event_bus or FakeEventBus()
    scheduler_obj = scheduler or FakeScheduler()
    trades_cache = cache or StrictTradesCache({orderflow_key: trades})

    analyzer = VolumeDeltaAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=scheduler_obj,  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_volume_delta_config(),
        source_topic_patterns=(TRADES_TOPIC,),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )
    return analyzer, bus, trades_cache, scheduler_obj


def make_aggressive_analyzer(
    trades: Any,
    *,
    orderflow_key: OrderFlowKey = DEFAULT_KEY,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: AggressiveTradesConfig | None = None,
    scheduler: FakeScheduler | None = None,
) -> tuple[AggressiveTradesAnalyzer, FakeEventBus, Any, FakeScheduler]:
    bus = event_bus or FakeEventBus()
    scheduler_obj = scheduler or FakeScheduler()
    trades_cache = cache or StrictTradesCache({orderflow_key: trades})

    analyzer = AggressiveTradesAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=scheduler_obj,  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_aggressive_config(),
        source_topic_patterns=(TRADES_TOPIC,),
        default_exchange="binance",
        default_market_type="usdm_futures",
        default_timeframe="1m",
    )
    return analyzer, bus, trades_cache, scheduler_obj


# =============================================================================
# Assertions
# =============================================================================


def emitted_events(event_bus: FakeEventBus, topic: str) -> list[dict[str, Any]]:
    return [item for item in event_bus.emitted if item["topic"] == topic]


def assert_scope_payload(payload: dict[str, Any], expected_key: OrderFlowKey) -> None:
    scope = orderflow_key_to_dict(expected_key)

    assert payload["exchange"] == scope["exchange"]
    assert payload["market_type"] == scope["market_type"]
    assert payload["symbol"] == scope["symbol"]
    assert payload["timeframe"] == scope["timeframe"]
    assert payload["scope"] == scope
    assert payload["scope_key"] == orderflow_key_to_string(expected_key)
    assert payload["key"] == list(expected_key)
    assert tuple(payload["orderflow_key"]) == expected_key


def assert_update_emitted(
    event_bus: FakeEventBus,
    *,
    topic: str,
    metric: str,
    expected_key: OrderFlowKey = DEFAULT_KEY,
) -> dict[str, Any]:
    events = emitted_events(event_bus, topic)
    assert len(events) >= 1

    emitted = events[-1]
    payload = emitted["payload"]

    assert emitted["priority"] == EventPriority.HIGH
    assert payload["metric"] == metric
    assert payload["source_type"] == "trades"
    assert_scope_payload(payload, expected_key)

    stats = payload["stats"]
    assert stats["metric"] == metric
    assert stats["source_type"] == "trades"
    assert_scope_payload(stats, expected_key)

    return payload


def assert_signal_emitted(
    event_bus: FakeEventBus,
    *,
    topic: str,
    metric: str,
    signal_type: str,
    side: str,
    expected_key: OrderFlowKey = DEFAULT_KEY,
) -> dict[str, Any]:
    events = emitted_events(event_bus, topic)
    assert len(events) >= 1

    payload = events[-1]["payload"]

    assert payload["metric"] == metric
    assert payload["source_type"] == "trades"
    assert payload["signal_type"] == signal_type
    assert payload["side"] == side
    assert payload["reason"]
    assert 0.0 <= payload["strength"] <= 1.0
    assert_scope_payload(payload, expected_key)

    context = payload["context"]
    assert context["scope"] == orderflow_key_to_dict(expected_key)
    assert context["scope_key"] == orderflow_key_to_string(expected_key)
    assert "trades_count" in context

    return payload


# =============================================================================
# Lifecycle / topic contract
# =============================================================================


@pytest.mark.parametrize(
    ("factory", "class_name"),
    [
        (make_cvd_analyzer, "CvdAnalyzer"),
        (make_volume_delta_analyzer, "VolumeDeltaAnalyzer"),
        (make_aggressive_analyzer, "AggressiveTradesAnalyzer"),
    ],
)
def test_trade_analyzer_registers_data_layer_topic_and_scheduler_jobs(
    factory: Any,
    class_name: str,
) -> None:
    analyzer, event_bus, _, scheduler = factory([])

    analyzer.register()

    assert analyzer.is_running is True
    assert [subscription.pattern for subscription in event_bus.subscriptions] == [
        TRADES_TOPIC
    ]
    assert class_name in event_bus.subscriptions[0].name

    assert len(scheduler.jobs) == 2
    job_names = {job.name for job in scheduler.jobs.values()}
    assert any("health" in name for name in job_names)
    assert any("cleanup" in name for name in job_names)

    for job in scheduler.jobs.values():
        assert job.enabled is True
        assert job.allow_overlap is False
        assert job.timeout == pytest.approx(0.5)
        assert job.retry_delay == pytest.approx(0.1)


@pytest.mark.parametrize(
    "factory",
    [make_cvd_analyzer, make_volume_delta_analyzer, make_aggressive_analyzer],
)
def test_trade_analyzer_rejects_raw_market_trade_topic_by_default(factory: Any) -> None:
    analyzer, event_bus, _, _ = factory([])
    analyzer._source_topic_patterns = ["market.trade"]  # noqa: SLF001

    with pytest.raises(ValueError, match="Raw market topic"):
        analyzer.register()

    assert event_bus.subscriptions == []


@pytest.mark.parametrize(
    "factory",
    [make_cvd_analyzer, make_volume_delta_analyzer, make_aggressive_analyzer],
)
def test_trade_analyzer_stop_unsubscribes_and_removes_jobs(factory: Any) -> None:
    analyzer, event_bus, _, scheduler = factory([])

    analyzer.register()
    created_subscriptions = list(event_bus.subscriptions)
    created_job_ids = set(scheduler.jobs)

    analyzer.stop()

    assert analyzer.is_running is False
    assert event_bus.subscriptions == []
    assert event_bus.unsubscribed == created_subscriptions
    assert set(scheduler.removed_job_ids) == created_job_ids
    assert scheduler.jobs == {}


# =============================================================================
# CVD analyzer tests
# =============================================================================


@pytest.mark.asyncio
async def test_cvd_process_key_calculates_positive_cvd_and_emits_bullish_signal() -> None:
    trades = [
        buy(1, 5, price=100, ts=now_ts(-3)),
        buy(2, 4, price=101, ts=now_ts(-2)),
        sell(3, 1, price=102, ts=now_ts(-1)),
    ]

    analyzer, event_bus, cache, _ = make_cvd_analyzer(
        trades,
        config=make_cvd_config(
            bullish_delta_ratio_threshold=0.2,
            require_delta_confirmation=False,
            require_slope_confirmation=False,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.key == DEFAULT_KEY
    assert stats.trades_count == 3
    assert stats.buy_volume == pytest.approx(9.0)
    assert stats.sell_volume == pytest.approx(1.0)
    assert stats.volume_delta == pytest.approx(8.0)
    assert stats.delta_ratio == pytest.approx(0.8)
    assert stats.cvd_close == pytest.approx(stats.cvd_value)
    assert stats.last_price == pytest.approx(102.0)

    assert cache.calls[-1]["method"] == "get_recent_trades"
    assert cache.calls[-1]["exchange"] == "binance"
    assert cache.calls[-1]["market_type"] == "usdm_futures"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"
    assert cache.calls[-1]["timeframe"] == "1m"

    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats
    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
        metric="cvd",
        expected_key=DEFAULT_KEY,
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_SIGNAL.value,
        metric="cvd",
        signal_type="bullish",
        side="buy",
        expected_key=DEFAULT_KEY,
    )


@pytest.mark.asyncio
async def test_cvd_process_key_calculates_negative_cvd_and_emits_bearish_signal() -> None:
    analyzer, event_bus, _, _ = make_cvd_analyzer(
        [
            sell(1, 6, price=100, ts=now_ts(-3)),
            sell(2, 3, price=99, ts=now_ts(-2)),
            buy(3, 1, price=98, ts=now_ts(-1)),
        ],
        config=make_cvd_config(
            bearish_delta_ratio_threshold=-0.2,
            require_delta_confirmation=False,
            require_slope_confirmation=False,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.buy_volume == pytest.approx(1.0)
    assert stats.sell_volume == pytest.approx(9.0)
    assert stats.volume_delta == pytest.approx(-8.0)
    assert stats.delta_ratio == pytest.approx(-0.8)
    assert stats.price_change == pytest.approx(-2.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
        metric="cvd",
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_SIGNAL.value,
        metric="cvd",
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_cvd_filters_duplicate_trades_between_runs_without_appending_twice() -> None:
    trades = [
        buy(1, 2, ts=now_ts(-2)),
        sell(2, 1, ts=now_ts(-1)),
    ]
    cache = StrictTradesCache({DEFAULT_KEY: trades})
    analyzer, event_bus, _, _ = make_cvd_analyzer(trades, cache=cache)

    first_stats = await analyzer.process_key(DEFAULT_KEY)
    first_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    first_store_size = len(analyzer._trades_by_key[DEFAULT_KEY])  # noqa: SLF001

    cache.set_trades(DEFAULT_KEY, list(trades))
    second_stats = await analyzer.process_key(DEFAULT_KEY)
    second_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    second_store_size = len(analyzer._trades_by_key[DEFAULT_KEY])  # noqa: SLF001

    assert first_stats is not None
    assert second_stats is not None
    assert first_store_size == 2
    assert second_store_size == 2
    assert second_stats.trades_count == first_stats.trades_count
    assert second_processed_trades == first_processed_trades

    update_events = emitted_events(event_bus, OrderFlowEventTopic.CVD_UPDATED.value)
    assert len(update_events) == 2


@pytest.mark.asyncio
async def test_cvd_same_trade_id_isolated_between_exchanges() -> None:
    cache = StrictTradesCache(
        {
            DEFAULT_KEY: [
                buy("same-id", 5, ts=now_ts(-2), orderflow_key=DEFAULT_KEY),
                sell("b2", 1, ts=now_ts(-1), orderflow_key=DEFAULT_KEY),
            ],
            BYBIT_KEY: [
                buy("same-id", 2, ts=now_ts(-2), orderflow_key=BYBIT_KEY),
                sell("y2", 1, ts=now_ts(-1), orderflow_key=BYBIT_KEY),
            ],
        }
    )
    analyzer, _, _, _ = make_cvd_analyzer([], cache=cache)

    binance_stats = await analyzer.process_key(DEFAULT_KEY)
    bybit_stats = await analyzer.process_key(BYBIT_KEY)

    assert binance_stats is not None
    assert bybit_stats is not None
    assert binance_stats.volume_delta == pytest.approx(4.0)
    assert bybit_stats.volume_delta == pytest.approx(1.0)

    assert len(analyzer._trades_by_key[DEFAULT_KEY]) == 2  # noqa: SLF001
    assert len(analyzer._trades_by_key[BYBIT_KEY]) == 2  # noqa: SLF001


@pytest.mark.asyncio
async def test_cvd_rejects_malformed_or_unknown_side_trades_without_crashing() -> None:
    analyzer, event_bus, _, _ = make_cvd_analyzer(
        malformed_trades(),
        config=make_cvd_config(min_trades_in_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


# =============================================================================
# VolumeDelta analyzer tests
# =============================================================================


@pytest.mark.asyncio
async def test_volume_delta_calculates_volume_notional_and_cumulative_delta() -> None:
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 5, price=100, ts=now_ts(-3)),
            buy(2, 2, price=110, ts=now_ts(-2)),
            sell(3, 1, price=120, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            bullish_delta_ratio_threshold=0.2,
            bullish_volume_delta_threshold=1.0,
            bullish_cumulative_delta_threshold=1.0,
            require_ratio_and_absolute_confirmation=True,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.key == DEFAULT_KEY
    assert stats.trades_count == 3
    assert stats.buy_volume == pytest.approx(7.0)
    assert stats.sell_volume == pytest.approx(1.0)
    assert stats.volume_delta == pytest.approx(6.0)
    assert stats.notional_delta == pytest.approx((5 * 100 + 2 * 110) - (1 * 120))
    assert stats.cumulative_volume_delta == pytest.approx(6.0)
    assert stats.delta_ratio == pytest.approx(0.75)
    assert stats.buy_ratio == pytest.approx(7 / 8)
    assert stats.last_price == pytest.approx(120.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
        metric="volume_delta",
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
        metric="volume_delta",
        signal_type="bullish",
        side="buy",
    )


@pytest.mark.asyncio
async def test_volume_delta_requires_min_total_volume_and_returns_none_when_too_small() -> None:
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 0.1, ts=now_ts(-2)),
            sell(2, 0.1, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            min_total_volume=10.0,
            min_trades_in_window=2,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_volume_delta_requires_ratio_and_absolute_confirmation_when_enabled() -> None:
    """
    Vulnerability test.

    A large absolute delta alone must not emit a signal when ratio confirmation
    is required and ratio is still neutral.
    """
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 51, ts=now_ts(-2)),
            sell(2, 49, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            bullish_delta_ratio_threshold=0.20,
            bullish_volume_delta_threshold=1.0,
            bullish_cumulative_delta_threshold=1.0,
            require_ratio_and_absolute_confirmation=True,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.volume_delta == pytest.approx(2.0)
    assert stats.delta_ratio == pytest.approx(0.02)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
        metric="volume_delta",
    )
    assert emitted_events(event_bus, OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value) == []


@pytest.mark.asyncio
async def test_volume_delta_emits_bearish_signal_for_sell_pressure() -> None:
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [
            sell(1, 8, price=100, ts=now_ts(-2)),
            buy(2, 1, price=99, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            bearish_delta_ratio_threshold=-0.2,
            bearish_volume_delta_threshold=-1.0,
            bearish_cumulative_delta_threshold=-1.0,
            require_ratio_and_absolute_confirmation=True,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.volume_delta == pytest.approx(-7.0)
    assert stats.delta_ratio == pytest.approx(-7 / 9)

    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
        metric="volume_delta",
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_volume_delta_signal_throttling_prevents_duplicate_signal_spam_per_key() -> None:
    trades = [
        buy(1, 5, ts=now_ts(-3)),
        buy(2, 5, ts=now_ts(-2)),
        sell(3, 1, ts=now_ts(-1)),
    ]
    cache = StrictTradesCache({DEFAULT_KEY: trades})
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        trades,
        cache=cache,
        config=make_volume_delta_config(
            min_signal_interval_sec=3600.0,
            bullish_delta_ratio_threshold=0.2,
            bullish_volume_delta_threshold=1.0,
            bullish_cumulative_delta_threshold=1.0,
            require_ratio_and_absolute_confirmation=True,
        ),
    )

    first_stats = await analyzer.process_key(DEFAULT_KEY)
    assert first_stats is not None

    cache.set_trades(DEFAULT_KEY, trades + [buy(4, 4, ts=now_ts())])
    second_stats = await analyzer.process_key(DEFAULT_KEY)
    assert second_stats is not None

    signals = emitted_events(event_bus, OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value)
    assert len(signals) == 1

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["signals_emitted"] == 1
    assert snapshot["metrics"]["skipped"] >= 1


# =============================================================================
# AggressiveTrades analyzer tests
# =============================================================================


@pytest.mark.asyncio
async def test_aggressive_trades_detects_bullish_large_buying_pressure() -> None:
    analyzer, event_bus, _, _ = make_aggressive_analyzer(
        [
            buy(1, 15, price=100, ts=now_ts(-4), is_aggressive=True),
            buy(2, 12, price=101, ts=now_ts(-3), is_aggressive=True),
            buy(3, 10, price=102, ts=now_ts(-2), is_aggressive=True),
            sell(4, 2, price=103, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            bullish_buy_ratio_threshold=0.65,
            bullish_delta_threshold=1.0,
            large_trade_notional_threshold=1_000.0,
            min_large_trades_for_signal=1,
            burst_trades_threshold=3,
            burst_volume_threshold=0.0,
            burst_score_threshold=0.1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.key == DEFAULT_KEY
    assert stats.trades_count == 4
    assert stats.aggressive_buy_count == 3
    assert stats.aggressive_sell_count == 1
    assert stats.aggressive_buy_volume == pytest.approx(37.0)
    assert stats.aggressive_sell_volume == pytest.approx(2.0)
    assert stats.net_volume_delta == pytest.approx(35.0)
    assert stats.buy_ratio == pytest.approx(37.0 / 39.0)
    assert stats.large_buy_trades >= 1
    assert stats.burst_score > 0

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
        metric="aggressive_trades",
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
        metric="aggressive_trades",
        signal_type="bullish",
        side="buy",
    )


@pytest.mark.asyncio
async def test_aggressive_trades_detects_bearish_large_selling_pressure() -> None:
    analyzer, event_bus, _, _ = make_aggressive_analyzer(
        [
            sell(1, 15, price=100, ts=now_ts(-4), is_aggressive=True),
            sell(2, 12, price=99, ts=now_ts(-3), is_aggressive=True),
            sell(3, 10, price=98, ts=now_ts(-2), is_aggressive=True),
            buy(4, 2, price=97, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            bearish_sell_ratio_threshold=0.65,
            bearish_delta_threshold=-1.0,
            large_trade_notional_threshold=1_000.0,
            min_large_trades_for_signal=1,
            burst_trades_threshold=3,
            burst_volume_threshold=0.0,
            burst_score_threshold=0.1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.aggressive_buy_volume == pytest.approx(2.0)
    assert stats.aggressive_sell_volume == pytest.approx(37.0)
    assert stats.net_volume_delta == pytest.approx(-35.0)
    assert stats.sell_ratio == pytest.approx(37.0 / 39.0)
    assert stats.large_sell_trades >= 1

    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
        metric="aggressive_trades",
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_aggressive_trades_ignores_non_aggressive_trades_for_pressure_stats() -> None:
    analyzer, event_bus, _, _ = make_aggressive_analyzer(
        [
            buy(1, 100, price=100, ts=now_ts(-4), is_aggressive=False),
            buy(2, 100, price=100, ts=now_ts(-3), is_aggressive=False),
            sell(3, 2, price=100, ts=now_ts(-2), is_aggressive=True),
            sell(4, 2, price=100, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            min_trades_in_window=2,
            large_trade_notional_threshold=1_000.0,
            burst_score_threshold=0.1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.trades_count == 4
    assert stats.aggressive_buy_count == 0
    assert stats.aggressive_sell_count == 2
    assert stats.aggressive_buy_volume == pytest.approx(0.0)
    assert stats.aggressive_sell_volume == pytest.approx(4.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
        metric="aggressive_trades",
    )

    signals = emitted_events(
        event_bus,
        OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
    )
    assert len(signals) <= 1


@pytest.mark.asyncio
async def test_aggressive_trades_requires_large_trade_confirmation_for_signal() -> None:
    analyzer, event_bus, _, _ = make_aggressive_analyzer(
        [
            buy(1, 1, price=100, ts=now_ts(-4), is_aggressive=True),
            buy(2, 1, price=101, ts=now_ts(-3), is_aggressive=True),
            buy(3, 1, price=102, ts=now_ts(-2), is_aggressive=True),
            sell(4, 0.1, price=103, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            bullish_buy_ratio_threshold=0.65,
            bullish_delta_threshold=0.1,
            large_trade_notional_threshold=10_000.0,
            min_large_trades_for_signal=1,
            burst_score_threshold=0.1,
        ),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.large_buy_trades == 0
    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
        metric="aggressive_trades",
    )
    assert emitted_events(event_bus, OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value) == []


# =============================================================================
# Cache compatibility / async / event handling
# =============================================================================


@pytest.mark.asyncio
async def test_async_trades_cache_method_is_awaited() -> None:
    cache = AsyncTradesCache(
        {
            DEFAULT_KEY: [
                buy(1, 3, ts=now_ts(-2)),
                sell(2, 1, ts=now_ts(-1)),
            ]
        }
    )
    analyzer, event_bus, _, _ = make_volume_delta_analyzer([], cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert cache.calls[-1]["method"] == "get_recent_trades"
    assert cache.calls[-1]["exchange"] == "binance"
    assert cache.calls[-1]["market_type"] == "usdm_futures"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"
    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
        metric="volume_delta",
    )


@pytest.mark.asyncio
async def test_legacy_symbol_only_get_fallback_still_works_but_is_not_primary_contract() -> None:
    cache = LegacySymbolOnlyTradesCache(
        cache_result(
            [
                buy(1, 3, ts=now_ts(-2)),
                sell(2, 1, ts=now_ts(-1)),
            ]
        )
    )
    analyzer, _, _, _ = make_cvd_analyzer([], cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert cache.calls == [("get", "BTCUSDT")]
    assert stats.volume_delta == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_handle_event_extracts_scoped_key_and_processes_cache_snapshot() -> None:
    cache = StrictTradesCache(
        {
            BYBIT_KEY: [
                buy(1, 3, ts=now_ts(-2), orderflow_key=BYBIT_KEY),
                sell(2, 1, ts=now_ts(-1), orderflow_key=BYBIT_KEY),
            ]
        }
    )
    analyzer, event_bus, _, _ = make_cvd_analyzer([], cache=cache)

    event = Event(
        topic=TRADES_TOPIC,
        payload={
            "data": {
                "scope": {
                    "exchange": "bybit",
                    "market_type": "linear",
                    "symbol": "btcusdt",
                    "timeframe": "1m",
                }
            }
        },
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert cache.calls[-1]["exchange"] == "bybit"
    assert cache.calls[-1]["market_type"] == "linear"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
        metric="cvd",
        expected_key=BYBIT_KEY,
    )


@pytest.mark.asyncio
async def test_handle_event_without_scoped_key_is_skipped_without_cache_call() -> None:
    cache = StrictTradesCache({DEFAULT_KEY: [buy(1, 1), sell(2, 1)]})
    analyzer, event_bus, _, _ = make_cvd_analyzer([], cache=cache)

    event = Event(
        topic=TRADES_TOPIC,
        payload={"data": {"price": 100.0}},
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert cache.calls == []
    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] == 1


@pytest.mark.asyncio
async def test_cache_exception_is_handled_without_crashing_or_emitting() -> None:
    cache = StrictTradesCache({DEFAULT_KEY: [buy(1, 1), sell(2, 1)]}, fail=True)
    analyzer, event_bus, _, _ = make_volume_delta_analyzer([], cache=cache)

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1 or snapshot["metrics"]["errors"] >= 1


# =============================================================================
# Scope filters / futures-only behavior
# =============================================================================


@pytest.mark.asyncio
async def test_spot_market_type_is_blocked_in_futures_only_mode() -> None:
    spot_key = key(exchange="binance", market_type="spot", symbol="BTCUSDT", timeframe="1m")
    cache = StrictTradesCache(
        {
            spot_key: [
                buy(1, 3, orderflow_key=spot_key),
                sell(2, 1, orderflow_key=spot_key),
            ]
        }
    )
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [],
        cache=cache,
        config=make_volume_delta_config(allowed_market_types={"usdm_futures", "linear", "swap"}),
    )

    stats = await analyzer.process_key(spot_key)

    assert stats is None
    assert cache.calls == []
    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] == 1


@pytest.mark.asyncio
async def test_allowed_exchange_symbol_and_timeframe_filters_are_enforced() -> None:
    cache = StrictTradesCache(
        {
            DEFAULT_KEY: [buy(1, 3), sell(2, 1)],
            ETH_KEY: [
                buy(1, 3, orderflow_key=ETH_KEY),
                sell(2, 1, orderflow_key=ETH_KEY),
            ],
            BYBIT_KEY: [
                buy(1, 3, orderflow_key=BYBIT_KEY),
                sell(2, 1, orderflow_key=BYBIT_KEY),
            ],
            HIGHER_TF_KEY: [
                buy(1, 3, orderflow_key=HIGHER_TF_KEY),
                sell(2, 1, orderflow_key=HIGHER_TF_KEY),
            ],
        }
    )
    analyzer, event_bus, _, _ = make_cvd_analyzer(
        [],
        cache=cache,
        config=make_cvd_config(
            allowed_exchanges={"binance"},
            allowed_market_types={"usdm_futures"},
            allowed_symbols={"BTCUSDT"},
            allowed_timeframes={"1m"},
        ),
    )

    assert await analyzer.process_key(DEFAULT_KEY) is not None
    assert await analyzer.process_key(ETH_KEY) is None
    assert await analyzer.process_key(BYBIT_KEY) is None
    assert await analyzer.process_key(HIGHER_TF_KEY) is None

    assert len(cache.calls) == 1
    assert cache.calls[0]["exchange"] == "binance"
    assert cache.calls[0]["market_type"] == "usdm_futures"
    assert cache.calls[0]["symbol"] == "BTCUSDT"
    assert cache.calls[0]["timeframe"] == "1m"

    updates = emitted_events(event_bus, OrderFlowEventTopic.CVD_UPDATED.value)
    assert len(updates) == 1
    assert_scope_payload(updates[0]["payload"], DEFAULT_KEY)


# =============================================================================
# Dirty payload / vulnerability tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty_trade", malformed_trades())
async def test_dirty_trade_payloads_are_rejected_without_emitting(dirty_trade: Any) -> None:
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [dirty_trade],
        config=make_volume_delta_config(min_trades_in_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []


@pytest.mark.asyncio
async def test_normalized_trade_model_from_cache_is_supported() -> None:
    trade_1 = NormalizedTrade(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
        side=OrderFlowSide.BUY,
        price=100.0,
        quantity=3.0,
        notional=300.0,
        timestamp=now_ts(-2),
        trade_id="model-1",
        is_aggressive=True,
    )
    trade_2 = NormalizedTrade(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
        side=OrderFlowSide.SELL,
        price=101.0,
        quantity=1.0,
        notional=101.0,
        timestamp=now_ts(-1),
        trade_id="model-2",
        is_aggressive=True,
    )

    analyzer, event_bus, _, _ = make_cvd_analyzer(cache_result([trade_1, trade_2]))

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert stats.volume_delta == pytest.approx(2.0)
    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
        metric="cvd",
    )


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=(
        "Current trade analyzers may still propagate NaN/inf if NormalizedTrade "
        "or numeric helpers allow them. Keep this as a vulnerability test until "
        "all trade analyzers use math.isfinite guards for price/qty/notional."
    ),
    strict=False,
)
async def test_non_finite_trade_values_are_rejected_instead_of_emitting_nan_stats() -> None:
    analyzer, event_bus, _, _ = make_volume_delta_analyzer(
        [
            buy(1, float("inf"), price=100, ts=now_ts(-2)),
            sell(2, 1, price=100, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(min_trades_in_window=1),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is None
    assert event_bus.emitted == []


# =============================================================================
# Emit failure / cleanup / concurrency
# =============================================================================


@pytest.mark.asyncio
async def test_update_emit_failure_is_captured_without_crashing_or_losing_latest_stats() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value)

    analyzer, _, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 3, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
        event_bus=event_bus,
        config=make_volume_delta_config(emit_signals=False),
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert event_bus.emitted == []
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats

    metrics = analyzer.stats()["metrics"]
    assert metrics["emit_errors"] == 1
    assert metrics["updates_emitted"] == 0


@pytest.mark.asyncio
async def test_signal_emit_failure_does_not_rollback_update_or_latest_stats() -> None:
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(OrderFlowEventTopic.CVD_SIGNAL.value)

    analyzer, _, _, _ = make_cvd_analyzer(
        [
            buy(1, 5, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
        event_bus=event_bus,
    )

    stats = await analyzer.process_key(DEFAULT_KEY)

    assert stats is not None
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) == stats
    assert len(emitted_events(event_bus, OrderFlowEventTopic.CVD_UPDATED.value)) == 1
    assert len(emitted_events(event_bus, OrderFlowEventTopic.CVD_SIGNAL.value)) == 0

    metrics = analyzer.stats()["metrics"]
    assert metrics["updates_emitted"] == 1
    assert metrics["signals_emitted"] == 0
    assert metrics["emit_errors"] == 1


@pytest.mark.asyncio
async def test_cleanup_removes_only_stale_scoped_state() -> None:
    cache = StrictTradesCache(
        {
            DEFAULT_KEY: [buy(1, 3, ts=now_ts(-2)), sell(2, 1, ts=now_ts(-1))],
            BYBIT_KEY: [
                buy(3, 3, ts=now_ts(-2), orderflow_key=BYBIT_KEY),
                sell(4, 1, ts=now_ts(-1), orderflow_key=BYBIT_KEY),
            ],
        }
    )
    analyzer, _, _, _ = make_volume_delta_analyzer(
        [],
        cache=cache,
        config=make_volume_delta_config(window_seconds=30.0),
    )

    assert await analyzer.process_key(DEFAULT_KEY) is not None
    assert await analyzer.process_key(BYBIT_KEY) is not None

    stale_ts = time.time() - 10_000
    for trade in analyzer._trades_by_key[DEFAULT_KEY]:  # noqa: SLF001
        trade.timestamp = stale_ts

    await analyzer.cleanup()

    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is None
    assert analyzer.get_latest_stats_by_key(BYBIT_KEY) is not None
    assert DEFAULT_KEY not in analyzer._trades_by_key  # noqa: SLF001
    assert BYBIT_KEY in analyzer._trades_by_key  # noqa: SLF001


@pytest.mark.asyncio
async def test_concurrent_process_key_calls_do_not_corrupt_state_or_duplicate_trades() -> None:
    trades = [
        buy(1, 3, ts=now_ts(-2)),
        sell(2, 1, ts=now_ts(-1)),
    ]
    cache = StrictTradesCache({DEFAULT_KEY: trades})
    analyzer, event_bus, _, _ = make_cvd_analyzer(trades, cache=cache)

    results = await asyncio.gather(
        analyzer.process_key(DEFAULT_KEY),
        analyzer.process_key(DEFAULT_KEY),
        analyzer.process_key(DEFAULT_KEY),
    )

    assert all(result is not None for result in results)
    assert analyzer.get_latest_stats_by_key(DEFAULT_KEY) is not None
    assert len(analyzer._trades_by_key[DEFAULT_KEY]) == 2  # noqa: SLF001

    updates = emitted_events(event_bus, OrderFlowEventTopic.CVD_UPDATED.value)
    assert len(updates) == 3

    metrics = analyzer.stats()["metrics"]
    assert metrics["processed"] == 3
    assert metrics["processed_trades"] == 2


# =============================================================================
# Backward compatibility
# =============================================================================


@pytest.mark.asyncio
async def test_process_symbol_wrapper_uses_explicit_default_futures_scope() -> None:
    analyzer, event_bus, cache, _ = make_volume_delta_analyzer(
        [
            buy(1, 3, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ]
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.key == DEFAULT_KEY

    assert cache.calls[-1]["exchange"] == "binance"
    assert cache.calls[-1]["market_type"] == "usdm_futures"
    assert cache.calls[-1]["symbol"] == "BTCUSDT"
    assert cache.calls[-1]["timeframe"] == "1m"

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
        metric="volume_delta",
        expected_key=DEFAULT_KEY,
    )