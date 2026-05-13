# tests/analytics/orderflow/test_orderbook_imbalance_analyzer.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.orderflow.config import OrderbookImbalanceConfig
from analytics.orderflow.enums import OrderFlowEventTopic
from analytics.orderflow.models import OrderbookLevel, OrderbookSnapshot
from analytics.orderflow.orderbook_imbalance import OrderbookImbalanceAnalyzer


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

    It records emitted analytics events and can simulate EventBus failures.
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

    It verifies that analyzer lifecycle can register Scheduler interval jobs.
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


class FakeOrderbookCache:
    """
    Flexible orderbook cache fake.

    Supported modes:
    - get_snapshot
    - get_orderbook
    - get

    This lets tests verify fallback behavior without depending on real cache.
    """

    def __init__(
        self,
        snapshot: Any = None,
        *,
        method: str = "get_snapshot",
        fail: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.method = method
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def set_snapshot(self, snapshot: Any) -> None:
        self.snapshot = snapshot

    def get_snapshot(self, symbol: str) -> Any:
        if self.method != "get_snapshot":
            raise AttributeError("get_snapshot intentionally unavailable")
        return self._return("get_snapshot", symbol)

    def get_orderbook(self, symbol: str) -> Any:
        if self.method != "get_orderbook":
            raise AttributeError("get_orderbook intentionally unavailable")
        return self._return("get_orderbook", symbol)

    def get(self, symbol: str) -> Any:
        if self.method != "get":
            raise AttributeError("get intentionally unavailable")
        return self._return("get", symbol)

    def _return(self, method_name: str, symbol: str) -> Any:
        self.calls.append((method_name, symbol))

        if self.fail:
            raise RuntimeError(f"Simulated orderbook cache failure from {method_name}")

        return self.snapshot


class FallbackOnlyOrderbookCache:
    """
    Cache exposing only get().

    This catches analyzers that hardcode get_snapshot() and do not honor the
    fallback contract.
    """

    def __init__(self, snapshot: Any) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[str, str]] = []

    def get(self, symbol: str) -> Any:
        self.calls.append(("get", symbol))
        return self.snapshot


# ---------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------


def now_ts(offset: float = 0.0) -> float:
    return time.time() + offset


def level(price: float, size: float) -> list[float]:
    return [price, size]


def snapshot_dict(
    *,
    symbol: str = "BTCUSDT",
    bids: list[Any] | None = None,
    asks: list[Any] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "bids": bids if bids is not None else [],
        "asks": asks if asks is not None else [],
        "timestamp": now_ts() if ts is None else ts,
        "exchange": "binance",
        "sequence_id": "seq-1",
    }


def bullish_snapshot(
    *,
    symbol: str = "BTCUSDT",
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        symbol=symbol,
        ts=ts,
        bids=[
            level(100.0, 50.0),
            level(99.5, 30.0),
            level(99.0, 20.0),
        ],
        asks=[
            level(100.5, 10.0),
            level(101.0, 10.0),
            level(101.5, 5.0),
        ],
    )


def bearish_snapshot(
    *,
    symbol: str = "BTCUSDT",
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        symbol=symbol,
        ts=ts,
        bids=[
            level(100.0, 5.0),
            level(99.5, 10.0),
            level(99.0, 10.0),
        ],
        asks=[
            level(100.5, 40.0),
            level(101.0, 30.0),
            level(101.5, 30.0),
        ],
    )


def neutral_snapshot(
    *,
    symbol: str = "BTCUSDT",
    ts: float | None = None,
) -> dict[str, Any]:
    return snapshot_dict(
        symbol=symbol,
        ts=ts,
        bids=[
            level(100.0, 25.0),
            level(99.5, 25.0),
        ],
        asks=[
            level(100.5, 25.0),
            level(101.0, 25.0),
        ],
    )


def malformed_snapshots() -> list[Any]:
    return [
        None,
        "bad-snapshot",
        {},
        {"symbol": "BTCUSDT"},
        {"symbol": "BTCUSDT", "bids": [], "asks": []},
        {"symbol": "BTCUSDT", "bids": [[100, 1]], "asks": []},
        {"symbol": "BTCUSDT", "bids": [], "asks": [[101, 1]]},
        {"symbol": "BTCUSDT", "bids": [[0, 1]], "asks": [[101, 1]]},
        {"symbol": "BTCUSDT", "bids": [[100, -1]], "asks": [[101, 1]]},
    ]


def make_config(**overrides: Any) -> OrderbookImbalanceConfig:
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
        "depth_levels": 3,
        "min_total_volume": 0.0,
        "bullish_ratio_threshold": 0.60,
        "bearish_ratio_threshold": 0.40,
        "normalize_ratio_to_minus_one_one": False,
        "smooth_window": 1,
    }
    values.update(overrides)

    config = OrderbookImbalanceConfig(**values)
    config.validate()
    return config


def make_analyzer(
    snapshot: Any,
    *,
    event_bus: FakeEventBus | None = None,
    cache: Any | None = None,
    config: OrderbookImbalanceConfig | None = None,
) -> tuple[OrderbookImbalanceAnalyzer, FakeEventBus, Any]:
    bus = event_bus or FakeEventBus()
    orderbook_cache = cache or FakeOrderbookCache(snapshot)

    analyzer = OrderbookImbalanceAnalyzer(
        event_bus=bus,  # type: ignore[arg-type]
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        orderbook_cache=orderbook_cache,
        config=config or make_config(),
        source_topic_patterns=("market.orderbook",),
    )
    return analyzer, bus, orderbook_cache


def assert_update_emitted(
    event_bus: FakeEventBus,
    *,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value
    ]

    assert len(events) >= 1

    emitted = events[-1]
    payload = emitted["payload"]

    assert emitted["source"] == "orderbook_imbalance"
    assert emitted["priority"] == EventPriority.HIGH

    assert payload["symbol"] == symbol
    assert payload["metric"] == "orderbook_imbalance"
    assert payload["source_type"] == "orderbook"

    stats = payload["stats"]
    assert stats["symbol"] == symbol
    assert stats["metric"] == "orderbook_imbalance"
    assert stats["source_type"] == "orderbook"

    return payload


def assert_signal_emitted(
    event_bus: FakeEventBus,
    *,
    signal_type: str,
    side: str,
    reason: str,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value
    ]

    assert len(events) >= 1

    payload = events[-1]["payload"]

    assert payload["symbol"] == symbol
    assert payload["metric"] == "orderbook_imbalance"
    assert payload["source_type"] == "orderbook"
    assert payload["signal_type"] == signal_type
    assert payload["side"] == side
    assert payload["reason"] == reason
    assert 0.0 <= payload["strength"] <= 1.0

    context = payload["context"]
    assert "imbalance_ratio" in context
    assert "imbalance_diff" in context
    assert "bid_volume" in context
    assert "ask_volume" in context
    assert "depth_levels_used" in context

    return payload


# ---------------------------------------------------------------------
# Core calculation tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_symbol_calculates_bid_dominant_imbalance_and_bullish_signal() -> None:
    analyzer, event_bus, _ = make_analyzer(
        bullish_snapshot(),
        config=make_config(
            depth_levels=3,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.symbol == "BTCUSDT"
    assert stats.bid_volume == pytest.approx(100.0)
    assert stats.ask_volume == pytest.approx(25.0)
    assert stats.imbalance_ratio == pytest.approx(100.0 / 125.0)
    assert stats.imbalance_diff == pytest.approx((100.0 - 25.0) / 125.0)
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.spread == pytest.approx(0.5)
    assert stats.mid_price == pytest.approx(100.25)
    assert stats.depth_levels_used == 3

    assert_update_emitted(event_bus)
    assert_signal_emitted(
        event_bus,
        signal_type="bullish",
        side="buy",
        reason="orderbook_bid_imbalance",
    )


@pytest.mark.asyncio
async def test_process_symbol_calculates_ask_dominant_imbalance_and_bearish_signal() -> None:
    analyzer, event_bus, _ = make_analyzer(
        bearish_snapshot(),
        config=make_config(
            depth_levels=3,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.bid_volume == pytest.approx(25.0)
    assert stats.ask_volume == pytest.approx(100.0)
    assert stats.imbalance_ratio == pytest.approx(25.0 / 125.0)
    assert stats.imbalance_diff == pytest.approx((25.0 - 100.0) / 125.0)
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.spread == pytest.approx(0.5)

    assert_update_emitted(event_bus)
    assert_signal_emitted(
        event_bus,
        signal_type="bearish",
        side="sell",
        reason="orderbook_ask_imbalance",
    )


@pytest.mark.asyncio
async def test_neutral_imbalance_emits_update_but_no_signal() -> None:
    analyzer, event_bus, _ = make_analyzer(
        neutral_snapshot(),
        config=make_config(
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.imbalance_ratio == pytest.approx(0.5)
    assert stats.imbalance_diff == pytest.approx(0.0)

    assert_update_emitted(event_bus)

    signal_events = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value
    ]
    assert signal_events == []


@pytest.mark.asyncio
async def test_depth_levels_limit_is_respected_and_ignores_deeper_liquidity() -> None:
    """
    Vulnerability test.

    Deep liquidity outside configured depth must not distort top-of-book signal.
    """
    analyzer, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(100.0, 10.0),
                level(99.5, 10.0),
                level(90.0, 10_000.0),
            ],
            asks=[
                level(100.5, 5.0),
                level(101.0, 5.0),
                level(110.0, 10_000.0),
            ],
        ),
        config=make_config(depth_levels=2, smooth_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.depth_levels_used == 2
    assert stats.bid_volume == pytest.approx(20.0)
    assert stats.ask_volume == pytest.approx(10.0)
    assert stats.imbalance_ratio == pytest.approx(20.0 / 30.0)


@pytest.mark.asyncio
async def test_orderbook_levels_are_sorted_before_calculation() -> None:
    """
    Vulnerability test.

    Cache/exchange snapshots can arrive unsorted. Analyzer must sort bids
    descending and asks ascending before selecting best prices and depth.
    """
    analyzer, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(99.0, 100.0),
                level(100.0, 10.0),
                level(99.5, 20.0),
            ],
            asks=[
                level(102.0, 100.0),
                level(100.5, 10.0),
                level(101.0, 20.0),
            ],
        ),
        config=make_config(depth_levels=2, smooth_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.best_bid == pytest.approx(100.0)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.bid_volume == pytest.approx(30.0)
    assert stats.ask_volume == pytest.approx(30.0)
    assert stats.depth_levels_used == 2


@pytest.mark.asyncio
async def test_invalid_orderbook_levels_are_filtered_before_calculation() -> None:
    analyzer, _, _ = make_analyzer(
        snapshot_dict(
            bids=[
                level(0.0, 999.0),
                level(100.0, -1.0),
                level(99.5, 10.0),
            ],
            asks=[
                level(0.0, 999.0),
                level(101.0, -1.0),
                level(100.5, 5.0),
            ],
        ),
        config=make_config(depth_levels=3, smooth_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert stats.bid_volume == pytest.approx(10.0)
    assert stats.ask_volume == pytest.approx(5.0)
    assert stats.best_bid == pytest.approx(99.5)
    assert stats.best_ask == pytest.approx(100.5)
    assert stats.depth_levels_used == 1


@pytest.mark.asyncio
async def test_min_total_volume_blocks_noise_and_emits_nothing() -> None:
    analyzer, event_bus, _ = make_analyzer(
        snapshot_dict(
            bids=[level(100.0, 1.0)],
            asks=[level(100.5, 1.0)],
        ),
        config=make_config(min_total_volume=10.0, smooth_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


# ---------------------------------------------------------------------
# Ratio normalization / smoothing tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_minus_one_to_one_normalization_mode_changes_ratio_but_signal_uses_denormalized_ratio() -> None:
    analyzer, event_bus, _ = make_analyzer(
        bullish_snapshot(),
        config=make_config(
            normalize_ratio_to_minus_one_one=True,
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None

    raw_ratio = 100.0 / 125.0
    expected_normalized = (raw_ratio * 2.0) - 1.0

    assert stats.imbalance_ratio == pytest.approx(expected_normalized)
    assert stats.imbalance_diff == pytest.approx(expected_normalized)

    assert_update_emitted(event_bus)
    assert_signal_emitted(
        event_bus,
        signal_type="bullish",
        side="buy",
        reason="orderbook_bid_imbalance",
    )


@pytest.mark.asyncio
async def test_smoothing_window_averages_recent_ratios_and_can_prevent_signal_spike() -> None:
    """
    Vulnerability test.

    A single spoof-like bid spike should be smoothed if smooth_window > 1.
    """
    cache = FakeOrderbookCache(
        snapshot_dict(
            bids=[level(100.0, 90.0)],
            asks=[level(100.5, 10.0)],
        )
    )

    analyzer, event_bus, _ = make_analyzer(
        [],
        cache=cache,
        config=make_config(
            bullish_ratio_threshold=0.80,
            bearish_ratio_threshold=0.20,
            smooth_window=3,
        ),
    )

    first_stats = await analyzer.process_symbol("BTCUSDT")
    assert first_stats is not None
    assert first_stats.imbalance_ratio == pytest.approx(0.90)

    cache.set_snapshot(
        snapshot_dict(
            bids=[level(100.0, 50.0)],
            asks=[level(100.5, 50.0)],
        )
    )
    second_stats = await analyzer.process_symbol("BTCUSDT")
    assert second_stats is not None
    assert second_stats.imbalance_ratio == pytest.approx((0.90 + 0.50) / 2.0)

    cache.set_snapshot(
        snapshot_dict(
            bids=[level(100.0, 50.0)],
            asks=[level(100.5, 50.0)],
        )
    )
    third_stats = await analyzer.process_symbol("BTCUSDT")
    assert third_stats is not None
    assert third_stats.imbalance_ratio == pytest.approx((0.90 + 0.50 + 0.50) / 3.0)

    signals = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value
    ]

    assert len(signals) == 1
    assert signals[0]["payload"]["signal_type"] == "bullish"


@pytest.mark.asyncio
async def test_smoothing_history_is_bounded_to_configured_window() -> None:
    cache = FakeOrderbookCache(
        snapshot_dict(bids=[level(100, 80)], asks=[level(101, 20)])
    )
    analyzer, _, _ = make_analyzer(
        [],
        cache=cache,
        config=make_config(smooth_window=2),
    )

    await analyzer.process_symbol("BTCUSDT")

    cache.set_snapshot(snapshot_dict(bids=[level(100, 50)], asks=[level(101, 50)]))
    await analyzer.process_symbol("BTCUSDT")

    cache.set_snapshot(snapshot_dict(bids=[level(100, 20)], asks=[level(101, 80)]))
    await analyzer.process_symbol("BTCUSDT")

    history = analyzer._ratio_history_by_symbol["BTCUSDT"]  # noqa: SLF001

    assert len(history) == 2
    assert history == pytest.approx([0.50, 0.20])


# ---------------------------------------------------------------------
# Cache extraction / malformed snapshots
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_get_snapshot_dict_payload_is_supported() -> None:
    cache = FakeOrderbookCache(
        bullish_snapshot(),
        method="get_snapshot",
    )
    analyzer, _, _ = make_analyzer([], cache=cache)

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert cache.calls == [("get_snapshot", "BTCUSDT")]


@pytest.mark.asyncio
async def test_cache_get_orderbook_fallback_is_supported() -> None:
    cache = FakeOrderbookCache(
        bullish_snapshot(),
        method="get_orderbook",
    )
    analyzer, _, _ = make_analyzer([], cache=cache)

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert cache.calls == [("get_orderbook", "BTCUSDT")]


@pytest.mark.asyncio
async def test_cache_get_fallback_is_supported() -> None:
    cache = FallbackOnlyOrderbookCache(
        {
            "data": bullish_snapshot(),
        }
    )
    analyzer, _, _ = make_analyzer([], cache=cache)

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert cache.calls == [("get", "BTCUSDT")]


@pytest.mark.asyncio
async def test_cache_can_return_orderbook_snapshot_model_directly() -> None:
    snapshot_model = OrderbookSnapshot.create(
        symbol="btcusdt",
        bids=[
            OrderbookLevel(price=100.0, size=10.0),
            OrderbookLevel(price=99.5, size=10.0),
        ],
        asks=[
            OrderbookLevel(price=100.5, size=5.0),
            OrderbookLevel(price=101.0, size=5.0),
        ],
        timestamp=now_ts(),
        exchange="binance",
    )

    analyzer, _, _ = make_analyzer(snapshot_model)

    stats = await analyzer.process_symbol("btcusdt")

    assert stats is not None
    assert stats.symbol == "BTCUSDT"
    assert stats.bid_volume == pytest.approx(20.0)
    assert stats.ask_volume == pytest.approx(10.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_snapshot", malformed_snapshots())
async def test_malformed_snapshots_are_skipped_without_crashing(bad_snapshot: Any) -> None:
    analyzer, event_bus, _ = make_analyzer(
        bad_snapshot,
        config=make_config(smooth_window=1),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_cache_exception_is_handled_without_crashing_or_emitting() -> None:
    cache = FakeOrderbookCache(
        bullish_snapshot(),
        method="get_snapshot",
        fail=True,
    )
    analyzer, event_bus, _ = make_analyzer([], cache=cache)

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 0
    assert snapshot["metrics"]["skipped"] >= 1


# ---------------------------------------------------------------------
# EventBus handling / throttling / cleanup / state
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_event_processes_symbol_from_nested_payload() -> None:
    analyzer, event_bus, _ = make_analyzer(
        bullish_snapshot(),
    )

    event = Event(
        topic="market.orderbook",
        payload={"data": {"symbol": "btcusdt"}},
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert analyzer.get_latest_stats("BTCUSDT") is not None
    assert_update_emitted(event_bus)


@pytest.mark.asyncio
async def test_handle_event_without_symbol_is_skipped_without_cache_call() -> None:
    cache = FakeOrderbookCache(bullish_snapshot())
    analyzer, event_bus, _ = make_analyzer([], cache=cache)

    event = Event(
        topic="market.orderbook",
        payload={"data": {"price": 100}},
    )

    await analyzer._handle_event(event)  # noqa: SLF001

    assert cache.calls == []
    assert event_bus.emitted == []
    assert analyzer.stats()["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_signal_throttling_prevents_duplicate_orderbook_signal_spam() -> None:
    cache = FakeOrderbookCache(bullish_snapshot())
    analyzer, event_bus, _ = make_analyzer(
        [],
        cache=cache,
        config=make_config(
            bullish_ratio_threshold=0.60,
            bearish_ratio_threshold=0.40,
            min_signal_interval_sec=3600.0,
            smooth_window=1,
        ),
    )

    first_stats = await analyzer.process_symbol("BTCUSDT")
    assert first_stats is not None

    cache.set_snapshot(bullish_snapshot())
    second_stats = await analyzer.process_symbol("BTCUSDT")
    assert second_stats is not None

    signals = [
        item for item in event_bus.emitted
        if item["topic"] == OrderFlowEventTopic.ORDERBOOK_IMBALANCE_SIGNAL.value
    ]

    assert len(signals) == 1

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["signals_emitted"] == 1
    assert snapshot["metrics"]["skipped"] >= 1


@pytest.mark.asyncio
async def test_emit_failure_does_not_rollback_calculated_orderbook_state() -> None:
    """
    Vulnerability test.

    EventBus can fail temporarily. Analyzer should keep calculated state and
    increase emit_errors instead of crashing or rolling back stats.
    """
    event_bus = FakeEventBus()
    event_bus.fail_emit_topics.add(OrderFlowEventTopic.ORDERBOOK_IMBALANCE_UPDATED.value)

    analyzer, _, _ = make_analyzer(
        bullish_snapshot(),
        event_bus=event_bus,
        config=make_config(
            emit_signals=False,
            smooth_window=1,
        ),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert analyzer.get_latest_stats("BTCUSDT") is not None
    assert event_bus.emitted == []

    snapshot = analyzer.stats()
    assert snapshot["metrics"]["processed"] == 1
    assert snapshot["metrics"]["emit_errors"] == 1


@pytest.mark.asyncio
async def test_cleanup_removes_stale_orderbook_state() -> None:
    analyzer, _, _ = make_analyzer(
        bullish_snapshot(ts=now_ts(-1)),
        config=make_config(cleanup_interval_sec=5.0, smooth_window=2),
    )

    stats = await analyzer.process_symbol("BTCUSDT")

    assert stats is not None
    assert "BTCUSDT" in analyzer._last_stats_by_symbol  # noqa: SLF001
    assert "BTCUSDT" in analyzer._last_snapshot_ts_by_symbol  # noqa: SLF001
    assert "BTCUSDT" in analyzer._ratio_history_by_symbol  # noqa: SLF001

    analyzer._last_snapshot_ts_by_symbol["BTCUSDT"] = time.time() - 10_000  # noqa: SLF001

    await analyzer.cleanup()

    assert "BTCUSDT" not in analyzer._last_stats_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._last_snapshot_ts_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._ratio_history_by_symbol  # noqa: SLF001
    assert "BTCUSDT" not in analyzer._last_signal_ts_by_symbol  # noqa: SLF001


def test_stats_exposes_orderbook_specific_config_and_metrics() -> None:
    analyzer, _, _ = make_analyzer(
        bullish_snapshot(),
        config=make_config(
            depth_levels=7,
            min_total_volume=100.0,
            bullish_ratio_threshold=0.70,
            bearish_ratio_threshold=0.30,
            normalize_ratio_to_minus_one_one=True,
            smooth_window=4,
        ),
    )

    snapshot = analyzer.stats()

    assert snapshot["running"] is False
    assert snapshot["metric"] == "orderbook_imbalance"
    assert snapshot["source_type"] == "orderbook"
    assert snapshot["config"]["depth_levels"] == 7
    assert snapshot["config"]["min_total_volume"] == 100.0
    assert snapshot["config"]["bullish_ratio_threshold"] == 0.70
    assert snapshot["config"]["bearish_ratio_threshold"] == 0.30
    assert snapshot["config"]["normalize_ratio_to_minus_one_one"] is True
    assert snapshot["config"]["smooth_window"] == 4
    assert snapshot["tracked_symbols"] == 0