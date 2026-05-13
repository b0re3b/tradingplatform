# tests/analytics/orderflow/test_trade_flow_analyzers.py

from __future__ import annotations

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
from analytics.orderflow.models import NormalizedTrade
from analytics.orderflow.volume_delta import VolumeDeltaAnalyzer


# ---------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------


@dataclass(slots=True)
class FakeSubscription:
    pattern: str
    handler: Any
    name: str
    enabled: bool = True


class FakeEventBus:
    """
    Strict EventBus fake.

    It records subscriptions and emitted analytics events.
    It can simulate emit failures to verify that analyzers do not crash
    when EventBus is temporarily unavailable.
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
    Minimal Scheduler fake.

    Trade analyzer tests focus on calculations, but register() still must use
    Scheduler.add_interval_job() instead of uncontrolled asyncio loops.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
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
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "name": name,
            "func": func,
            "interval": interval,
            "args": args,
            "kwargs": kwargs or {},
            "run_immediately": run_immediately,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "timeout": timeout,
            "allow_overlap": allow_overlap,
            "enabled": enabled,
        }
        return job_id

    def disable_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.jobs[job_id]["enabled"] = False
        self.disabled_job_ids.append(job_id)

    def remove_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id)


class FakeTradesCache:
    """
    Flexible trades cache fake.

    Supported modes:
    - get_recent_trades
    - get_trades
    - get

    This allows testing fallback lookup order without depending on real
    data-cache implementation.
    """

    def __init__(
        self,
        trades: list[Any] | tuple[Any, ...] | dict[str, Any] | None = None,
        *,
        method: str = "get_recent_trades",
        fail: bool = False,
    ) -> None:
        self.trades = trades if trades is not None else []
        self.method = method
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def set_trades(self, trades: list[Any] | tuple[Any, ...] | dict[str, Any]) -> None:
        self.trades = trades

    def get_recent_trades(self, symbol: str) -> Any:
        if self.method != "get_recent_trades":
            raise AttributeError("get_recent_trades intentionally unavailable")
        return self._return("get_recent_trades", symbol)

    def get_trades(self, symbol: str) -> Any:
        if self.method != "get_trades":
            raise AttributeError("get_trades intentionally unavailable")
        return self._return("get_trades", symbol)

    def get(self, symbol: str) -> Any:
        if self.method != "get":
            raise AttributeError("get intentionally unavailable")
        return self._return("get", symbol)

    def _return(self, method_name: str, symbol: str) -> Any:
        self.calls.append((method_name, symbol))

        if self.fail:
            raise RuntimeError(f"Simulated cache failure from {method_name}")

        return self.trades


class FallbackOnlyTradesCache:
    """
    Cache fake that exposes only get().

    This catches analyzers that hardcode get_recent_trades() and do not honor
    the fallback contract.
    """

    def __init__(self, trades: list[Any] | dict[str, Any]) -> None:
        self.trades = trades
        self.calls: list[tuple[str, str]] = []

    def get(self, symbol: str) -> Any:
        self.calls.append(("get", symbol))
        return self.trades


# ---------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------


def now_ts(offset: float = 0.0) -> float:
    return time.time() + offset


def raw_trade(
    *,
    trade_id: str | int,
    side: str,
    quantity: float,
    price: float = 100.0,
    ts: float | None = None,
    symbol: str = "BTCUSDT",
    is_aggressive: bool = True,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": quantity,
        "timestamp": now_ts() if ts is None else ts,
        "trade_id": str(trade_id),
        "exchange": "binance",
        "is_aggressive": is_aggressive,
    }


def buy(
    trade_id: str | int,
    quantity: float,
    *,
    price: float = 100.0,
    ts: float | None = None,
    is_aggressive: bool = True,
) -> dict[str, Any]:
    return raw_trade(
        trade_id=trade_id,
        side="buy",
        quantity=quantity,
        price=price,
        ts=ts,
        is_aggressive=is_aggressive,
    )


def sell(
    trade_id: str | int,
    quantity: float,
    *,
    price: float = 100.0,
    ts: float | None = None,
    is_aggressive: bool = True,
) -> dict[str, Any]:
    return raw_trade(
        trade_id=trade_id,
        side="sell",
        quantity=quantity,
        price=price,
        ts=ts,
        is_aggressive=is_aggressive,
    )


def malformed_trades() -> list[Any]:
    return [
        None,
        "bad-trade",
        {"symbol": "BTCUSDT", "side": "buy"},
        {"symbol": "BTCUSDT", "side": "unknown", "price": 100, "quantity": 1},
        {"symbol": "BTCUSDT", "side": "buy", "price": 0, "quantity": 1, "timestamp": now_ts()},
        {"symbol": "BTCUSDT", "side": "sell", "price": 100, "quantity": -1, "timestamp": now_ts()},
    ]


def assert_update_emitted(
    event_bus: FakeEventBus,
    *,
    topic: str,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    events = [
        item for item in event_bus.emitted
        if item["topic"] == topic
    ]

    assert len(events) >= 1
    payload = events[-1]["payload"]

    assert payload["symbol"] == symbol
    assert payload["stats"]["symbol"] == symbol
    assert payload["stats"]["metric"] == payload["metric"]

    return payload


def assert_signal_emitted(
    event_bus: FakeEventBus,
    *,
    topic: str,
    signal_type: str,
    side: str,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    events = [
        item for item in event_bus.emitted
        if item["topic"] == topic
    ]

    assert len(events) >= 1
    payload = events[-1]["payload"]

    assert payload["symbol"] == symbol
    assert payload["signal_type"] == signal_type
    assert payload["side"] == side
    assert 0.0 <= payload["strength"] <= 1.0
    assert payload["reason"]

    return payload


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
    trades: list[Any] | tuple[Any, ...] | dict[str, Any],
    *,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: CvdConfig | None = None,
) -> tuple[CvdAnalyzer, FakeEventBus, Any]:
    bus = event_bus or FakeEventBus()
    trades_cache = cache or FakeTradesCache(trades)

    analyzer = CvdAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_cvd_config(),
        source_topic_patterns=("market.trade",),
    )
    return analyzer, bus, trades_cache


def make_volume_delta_analyzer(
    trades: list[Any] | tuple[Any, ...] | dict[str, Any],
    *,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: VolumeDeltaConfig | None = None,
) -> tuple[VolumeDeltaAnalyzer, FakeEventBus, Any]:
    bus = event_bus or FakeEventBus()
    trades_cache = cache or FakeTradesCache(trades)

    analyzer = VolumeDeltaAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_volume_delta_config(),
        source_topic_patterns=("market.trade",),
    )
    return analyzer, bus, trades_cache


def make_aggressive_analyzer(
    trades: list[Any] | tuple[Any, ...] | dict[str, Any],
    *,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: AggressiveTradesConfig | None = None,
) -> tuple[AggressiveTradesAnalyzer, FakeEventBus, Any]:
    bus = event_bus or FakeEventBus()
    trades_cache = cache or FakeTradesCache(trades)

    analyzer = AggressiveTradesAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        trades_cache=trades_cache,
        config=config or make_aggressive_config(),
        source_topic_patterns=("market.trade",),
    )
    return analyzer, bus, trades_cache


# ---------------------------------------------------------------------
# CVD analyzer tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cvd_process_symbol_calculates_positive_cvd_and_emits_bullish_signal() -> None:
    analyzer, event_bus, _ = make_cvd_analyzer(
        [
            buy(1, 5, price=100, ts=now_ts(-3)),
            buy(2, 4, price=101, ts=now_ts(-2)),
            sell(3, 1, price=102, ts=now_ts(-1)),
        ],
        config=make_cvd_config(
            bullish_delta_ratio_threshold=0.2,
            require_delta_confirmation=False,
            require_slope_confirmation=False,
        ),
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.symbol == "BTCUSDT"
    assert stats.trades_count == 3
    assert stats.buy_volume == pytest.approx(9.0)
    assert stats.sell_volume == pytest.approx(1.0)
    assert stats.volume_delta == pytest.approx(8.0)
    assert stats.delta_ratio == pytest.approx(0.8)
    assert stats.cvd_close == pytest.approx(stats.cvd_value)
    assert stats.last_price == pytest.approx(102.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_SIGNAL.value,
        signal_type="bullish",
        side="buy",
    )


@pytest.mark.asyncio
async def test_cvd_process_symbol_calculates_negative_cvd_and_emits_bearish_signal() -> None:
    analyzer, event_bus, _ = make_cvd_analyzer(
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

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.buy_volume == pytest.approx(1.0)
    assert stats.sell_volume == pytest.approx(9.0)
    assert stats.volume_delta == pytest.approx(-8.0)
    assert stats.delta_ratio == pytest.approx(-0.8)
    assert stats.price_change == pytest.approx(-2.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_UPDATED.value,
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.CVD_SIGNAL.value,
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_cvd_filters_duplicate_trades_between_runs_without_appending_twice() -> None:
    trades = [
        buy(1, 2, ts=now_ts(-2)),
        sell(2, 1, ts=now_ts(-1)),
    ]
    analyzer, event_bus, cache = make_cvd_analyzer(trades)

    first_stats = await analyzer.process_symbol("BTCUSDT")
    first_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    first_store_size = len(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001

    cache.set_trades(list(trades))
    second_stats = await analyzer.process_symbol("BTCUSDT")
    second_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    second_store_size = len(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001

    assert first_stats is not None
    assert second_stats is not None

    assert first_store_size == 2
    assert second_store_size == 2
    assert second_stats.trades_count == first_stats.trades_count
    assert second_processed_trades == first_processed_trades

    update_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.CVD_UPDATED.value
    ]
    assert len(update_events) == 2


@pytest.mark.asyncio
async def test_cvd_rejects_malformed_or_unknown_side_trades_without_crashing() -> None:
    analyzer, event_bus, _ = make_cvd_analyzer(
        malformed_trades(),
        config=make_cvd_config(min_trades_in_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_cvd_uses_cache_get_fallback_when_get_recent_trades_is_unavailable() -> None:
    cache = FallbackOnlyTradesCache(
        {
            "data": {
                "trades": [
                    buy(1, 3, ts=now_ts(-2)),
                    sell(2, 1, ts=now_ts(-1)),
                ]
            }
        }
    )

    analyzer, _, _ = make_cvd_analyzer(
        [],
        cache=cache,
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert cache.calls == [("get", "BTCUSDT")]
    assert stats.volume_delta == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_cvd_cache_exception_is_handled_without_crashing_or_emitting_signal() -> None:
    cache = FakeTradesCache(method="get_recent_trades", fail=True)
    analyzer, event_bus, _ = make_cvd_analyzer(
        [],
        cache=cache,
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_cvd_cleanup_removes_stale_symbol_state() -> None:
    analyzer, _, _ = make_cvd_analyzer(
        [
            buy(1, 2, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
        config=make_cvd_config(window_seconds=30.0),
    )

    stats = await analyzer.process_symbol("BTCUSDT")
    assert stats is not None
    assert "BTCUSDT" in analyzer._trades_by_symbol  # noqa: SLF001
    assert "BTCUSDT" in analyzer._last_stats_by_symbol  # noqa: SLF001

    stale_ts = time.time() - 10_000
    for trade in analyzer._trades_by_symbol["BTCUSDT"]:  # noqa: SLF001
        trade.timestamp = stale_ts

    for point in analyzer._cvd_points_by_symbol["BTCUSDT"]:  # noqa: SLF001
        point.timestamp = stale_ts

    await analyzer.cleanup()

    assert "BTCUSDT" not in analyzer._trades_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._cvd_points_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._last_stats_by_symbol  # noqa: SLF001


# ---------------------------------------------------------------------
# VolumeDelta analyzer tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_volume_delta_calculates_volume_notional_and_cumulative_delta() -> None:
    analyzer, event_bus, _ = make_volume_delta_analyzer(
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

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.symbol == "BTCUSDT"
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
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
        signal_type="bullish",
        side="buy",
    )


@pytest.mark.asyncio
async def test_volume_delta_requires_min_total_volume_and_returns_none_when_too_small() -> None:
    analyzer, event_bus, _ = make_volume_delta_analyzer(
        [
            buy(1, 0.1, ts=now_ts(-2)),
            sell(2, 0.1, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            min_total_volume=10.0,
            min_trades_in_window=2,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

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
    analyzer, event_bus, _ = make_volume_delta_analyzer(
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

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.volume_delta == pytest.approx(2.0)
    assert stats.delta_ratio == pytest.approx(0.02)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
    )

    signal_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value
    ]
    assert signal_events == []


@pytest.mark.asyncio
async def test_volume_delta_emits_bearish_signal_for_sell_pressure() -> None:
    analyzer, event_bus, _ = make_volume_delta_analyzer(
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

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.volume_delta == pytest.approx(-7.0)
    assert stats.delta_ratio == pytest.approx(-7 / 9)

    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value,
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_volume_delta_signal_throttling_prevents_duplicate_signal_spam() -> None:
    trades = [
        buy(1, 5, ts=now_ts(-3)),
        buy(2, 5, ts=now_ts(-2)),
        sell(3, 1, ts=now_ts(-1)),
    ]
    analyzer, event_bus, cache = make_volume_delta_analyzer(
        trades,
        config=make_volume_delta_config(
            min_signal_interval_sec=3600.0,
            bullish_delta_ratio_threshold=0.2,
            bullish_volume_delta_threshold=1.0,
            bullish_cumulative_delta_threshold=1.0,
            require_ratio_and_absolute_confirmation=True,
        ),
    )

    first_stats = await analyzer.process_symbol("BTCUSDT")
    assert first_stats is not None

    cache.set_trades(
        trades
        + [
            buy(4, 4, ts=now_ts()),
        ]
    )
    second_stats = await analyzer.process_symbol("BTCUSDT")
    assert second_stats is not None

    signals = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.VOLUME_DELTA_SIGNAL.value
    ]
    assert len(signals) == 1

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["signals_emitted"] == 1
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_volume_delta_cleanup_removes_stale_state() -> None:
    analyzer, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 3, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(window_seconds=30.0),
    )

    stats = await analyzer.process_symbol("BTCUSDT")
    assert stats is not None
    assert "BTCUSDT" in analyzer._trades_by_symbol  # noqa: SLF001

    stale_ts = time.time() - 10_000
    for trade in analyzer._trades_by_symbol["BTCUSDT"]:  # noqa: SLF001
        trade.timestamp = stale_ts

    await analyzer.cleanup()

    assert "BTCUSDT" not in analyzer._trades_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._last_stats_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._last_seen_trade_key_by_symbol  # noqa: SLF001


# ---------------------------------------------------------------------
# AggressiveTrades analyzer tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggressive_trades_detects_buy_pressure_large_trade_and_burst() -> None:
    analyzer, event_bus, _ = make_aggressive_analyzer(
        [
            buy(1, 20, price=100, ts=now_ts(-4), is_aggressive=True),
            buy(2, 15, price=100, ts=now_ts(-3), is_aggressive=True),
            buy(3, 10, price=100, ts=now_ts(-2), is_aggressive=True),
            sell(4, 2, price=100, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            min_trades_in_window=2,
            bullish_buy_ratio_threshold=0.65,
            bullish_delta_threshold=1.0,
            large_trade_notional_threshold=1_000.0,
            min_large_trades_for_signal=1,
            burst_trades_threshold=3,
            burst_volume_threshold=1.0,
            burst_score_threshold=1.0,
        ),
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.symbol == "BTCUSDT"
    assert stats.trades_count == 4
    assert stats.aggressive_buy_count == 3
    assert stats.aggressive_sell_count == 1
    assert stats.aggressive_buy_volume == pytest.approx(45.0)
    assert stats.aggressive_sell_volume == pytest.approx(2.0)
    assert stats.net_volume_delta == pytest.approx(43.0)
    assert stats.buy_ratio == pytest.approx(45 / 47)
    assert stats.large_buy_trades >= 1
    assert stats.burst_score >= 1.0

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
    )
    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
        signal_type="bullish",
        side="buy",
    )


@pytest.mark.asyncio
async def test_aggressive_trades_detects_sell_pressure_and_bearish_signal() -> None:
    analyzer, event_bus, _ = make_aggressive_analyzer(
        [
            sell(1, 20, price=100, ts=now_ts(-4), is_aggressive=True),
            sell(2, 15, price=100, ts=now_ts(-3), is_aggressive=True),
            sell(3, 10, price=100, ts=now_ts(-2), is_aggressive=True),
            buy(4, 2, price=100, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            min_trades_in_window=2,
            bearish_sell_ratio_threshold=0.65,
            bearish_delta_threshold=1.0,
            large_trade_notional_threshold=1_000.0,
            min_large_trades_for_signal=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.aggressive_sell_count == 3
    assert stats.aggressive_buy_count == 1
    assert stats.net_volume_delta == pytest.approx(-43.0)
    assert stats.sell_ratio == pytest.approx(45 / 47)
    assert stats.large_sell_trades >= 1

    assert_signal_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value,
        signal_type="bearish",
        side="sell",
    )


@pytest.mark.asyncio
async def test_aggressive_trades_ignores_non_aggressive_trades_for_pressure_stats() -> None:
    analyzer, event_bus, _ = make_aggressive_analyzer(
        [
            buy(1, 100, ts=now_ts(-4), is_aggressive=False),
            buy(2, 100, ts=now_ts(-3), is_aggressive=False),
            sell(3, 5, ts=now_ts(-2), is_aggressive=True),
            sell(4, 5, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            min_trades_in_window=2,
            bearish_sell_ratio_threshold=0.65,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.trades_count == 2
    assert stats.aggressive_buy_count == 0
    assert stats.aggressive_sell_count == 2
    assert stats.aggressive_buy_volume == 0
    assert stats.aggressive_sell_volume == pytest.approx(10.0)

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
    )


@pytest.mark.asyncio
async def test_aggressive_trades_does_not_signal_when_large_trade_requirement_not_met() -> None:
    analyzer, event_bus, _ = make_aggressive_analyzer(
        [
            buy(1, 4, price=100, ts=now_ts(-3), is_aggressive=True),
            buy(2, 4, price=100, ts=now_ts(-2), is_aggressive=True),
            sell(3, 1, price=100, ts=now_ts(-1), is_aggressive=True),
        ],
        config=make_aggressive_config(
            min_trades_in_window=2,
            bullish_buy_ratio_threshold=0.65,
            bullish_delta_threshold=1.0,
            large_trade_notional_threshold=10_000.0,
            min_large_trades_for_signal=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.buy_ratio > 0.65
    assert stats.large_buy_trades == 0

    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.AGGRESSIVE_TRADES_UPDATED.value,
    )

    signal_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.AGGRESSIVE_TRADES_SIGNAL.value
    ]
    assert signal_events == []


@pytest.mark.asyncio
async def test_aggressive_trades_filters_duplicate_trades_between_runs() -> None:
    trades = [
        buy(1, 10, price=100, ts=now_ts(-3), is_aggressive=True),
        sell(2, 1, price=100, ts=now_ts(-2), is_aggressive=True),
    ]
    analyzer, _, cache = make_aggressive_analyzer(
        trades,
        config=make_aggressive_config(min_trades_in_window=2),
    )

    first_stats = await analyzer.process_symbol("BTCUSDT")
    first_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    first_store_size = len(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001

    cache.set_trades(list(trades))
    second_stats = await analyzer.process_symbol("BTCUSDT")
    second_processed_trades = analyzer.stats()["metrics"]["processed_trades"]
    second_store_size = len(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001

    assert first_stats is not None
    assert second_stats is not None
    assert first_store_size == 2
    assert second_store_size == 2
    assert second_stats.trades_count == first_stats.trades_count
    assert second_processed_trades == first_processed_trades


@pytest.mark.asyncio
async def test_aggressive_trades_malformed_payloads_are_skipped_without_crashing() -> None:
    analyzer, event_bus, _ = make_aggressive_analyzer(
        malformed_trades(),
        config=make_aggressive_config(min_trades_in_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_aggressive_trades_cache_exception_is_handled_without_crashing() -> None:
    cache = FakeTradesCache(method="get_recent_trades", fail=True)
    analyzer, event_bus, _ = make_aggressive_analyzer(
        [],
        cache=cache,
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


# ---------------------------------------------------------------------
# Event handler / sorting / maxlen behavior
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trade_analyzers_process_symbol_from_eventbus_event_payload_symbol() -> None:
    trades = [
        buy(1, 3, ts=now_ts(-2)),
        sell(2, 1, ts=now_ts(-1)),
    ]
    analyzer, event_bus, _ = make_volume_delta_analyzer(
        trades,
        config=make_volume_delta_config(min_trades_in_window=2),
    )

    event = Event(
        topic="market.trade",
        payload={"data": {"symbol": "btcusdt"}},
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert analyzer.get_latest_stats("BTCUSDT") is not None
    assert_update_emitted(
        event_bus,
        topic=OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value,
    )


@pytest.mark.asyncio
async def test_trade_analyzers_skip_event_without_symbol_without_crashing() -> None:
    analyzer, event_bus, _ = make_volume_delta_analyzer(
        [
            buy(1, 3, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
    )

    event = Event(
        topic="market.trade",
        payload={"data": {"price": 100}},
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_trade_normalization_sorts_by_timestamp_before_calculation() -> None:
    """
    Vulnerability test.

    Exchange/cache payloads often arrive out of order. The analyzer should sort
    normalized trades before calculating last_price/window stats.
    """
    analyzer, _, _ = make_volume_delta_analyzer(
        [
            buy(3, 1, price=103, ts=1003.0),
            buy(1, 1, price=101, ts=1001.0),
            sell(2, 1, price=102, ts=1002.0),
        ],
        config=make_volume_delta_config(
            min_trades_in_window=2,
            bullish_delta_ratio_threshold=0.0,
            require_ratio_and_absolute_confirmation=False,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.last_price == pytest.approx(103.0)

    stored = list(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001
    assert [trade.trade_id for trade in stored] == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_max_trades_per_symbol_is_enforced_by_bounded_deque() -> None:
    analyzer, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 1, ts=now_ts(-5)),
            sell(2, 1, ts=now_ts(-4)),
            buy(3, 1, ts=now_ts(-3)),
            sell(4, 1, ts=now_ts(-2)),
            buy(5, 1, ts=now_ts(-1)),
        ],
        config=make_volume_delta_config(
            max_trades_per_symbol=3,
            min_trades_in_window=2,
            require_ratio_and_absolute_confirmation=False,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None

    stored = list(analyzer._trades_by_symbol["BTCUSDT"])  # noqa: SLF001
    assert len(stored) == 3
    assert [trade.trade_id for trade in stored] == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_emit_failure_does_not_rollback_calculated_trade_state() -> None:
    """
    Vulnerability test.

    EventBus can fail temporarily. Calculation state should still be stored,
    emit_errors should increase, and process_symbol() should not crash.
    """
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(OrderFlowEventTopic.VOLUME_DELTA_UPDATED.value)

    analyzer, _, _ = make_volume_delta_analyzer(
        [
            buy(1, 5, ts=now_ts(-2)),
            sell(2, 1, ts=now_ts(-1)),
        ],
        event_bus=event_bus,
        config=make_volume_delta_config(
            min_trades_in_window=2,
            emit_signals=False,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert analyzer.get_latest_stats("BTCUSDT") is not None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 1
    assert snapshot["metrics"]["emit_errors"] == 1


def test_make_trade_key_is_stable_for_trade_id_and_fallback_fields() -> None:
    analyzer, _, _ = make_volume_delta_analyzer([])

    with_trade_id = NormalizedTrade.create(
        symbol="btcusdt",
        side=OrderFlowSide.BUY,
        price=100,
        quantity=1,
        timestamp=123,
        trade_id="abc",
    )
    same_trade_id_different_price = NormalizedTrade.create(
        symbol="btcusdt",
        side=OrderFlowSide.SELL,
        price=999,
        quantity=9,
        timestamp=999,
        trade_id="abc",
    )
    no_trade_id = NormalizedTrade.create(
        symbol="btcusdt",
        side=OrderFlowSide.BUY,
        price=100,
        quantity=1,
        timestamp=123,
        trade_id=None,
    )

    assert analyzer.make_trade_key(with_trade_id) == analyzer.make_trade_key(
        same_trade_id_different_price
    )

    fallback_key = analyzer.make_trade_key(no_trade_id)
    assert "BTCUSDT" in fallback_key
    assert "100" in fallback_key
    assert "123" in fallback_key