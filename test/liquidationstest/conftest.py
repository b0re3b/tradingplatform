# test/liquidationstest/conftest.py

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from core.event_bus import EventBus, QueueFullPolicy
from core.scheduler import Scheduler

from analytics.liquidations.config import (
    CascadeDetectorConfig,
    LiquidationStreamConfig,
)
from analytics.liquidations.enums import LiquidationSide
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import LiquidationEvent
from analytics.liquidations.state import LiquidationState


# ============================================================
# Fake exchange adapter
# ============================================================

class FakeLiquidationExchangeAdapter:
    """
    Fake adapter for LiquidationStream tests.

    Імітує exchange adapter contract:
    - connect_liquidations()
    - disconnect_liquidations()
    - recv_liquidation()
    """

    def __init__(
        self,
        *,
        name: str = "binance",
        payloads: list[dict[str, Any] | None] | None = None,
    ) -> None:
        self._name = name

        self.connected: bool = False
        self.disconnected: bool = False
        self.connected_symbols: tuple[str, ...] | None = None

        self._payloads: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        for payload in payloads or []:
            self._payloads.put_nowait(payload)

    @property
    def name(self) -> str:
        return self._name

    async def connect_liquidations(self, symbols: tuple[str, ...]) -> None:
        self.connected = True
        self.disconnected = False
        self.connected_symbols = symbols

    async def disconnect_liquidations(self) -> None:
        self.connected = False
        self.disconnected = True

    async def recv_liquidation(self) -> dict[str, Any] | None:
        try:
            return self._payloads.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def push_payload(self, payload: dict[str, Any] | None) -> None:
        self._payloads.put_nowait(payload)


# ============================================================
# Core fixtures
# ============================================================

@pytest_asyncio.fixture
async def event_bus() -> AsyncGenerator[EventBus, None]:
    """
    Real core.EventBus fixture.

    Важливо:
    - strict-compatible fixture через pytest_asyncio.fixture;
    - EventBus реально стартує worker-и;
    - після тесту завжди виконується graceful stop.
    """
    bus = EventBus(
        max_queue_size=1_000,
        worker_count=1,
        queue_full_policy=QueueFullPolicy.DROP_NEW,
        max_retries=0,
        retry_delay=0.01,
        enable_metrics=True,
        service_name="test_event_bus",
    )

    await bus.start()

    try:
        yield bus
    finally:
        await bus.stop(drain=True, timeout=1.0)


@pytest_asyncio.fixture
async def scheduler(event_bus: EventBus) -> AsyncGenerator[Scheduler, None]:
    """
    Real core.Scheduler fixture.

    Важливо:
    - strict-compatible fixture через pytest_asyncio.fixture;
    - використовує real EventBus;
    - після тесту зупиняє running jobs.
    """
    test_scheduler = Scheduler(
        event_bus=event_bus,
        tick_interval=0.01,
        service_name="test_scheduler",
    )

    await test_scheduler.start()

    try:
        yield test_scheduler
    finally:
        await test_scheduler.stop(
            wait_running_jobs=True,
            timeout=1.0,
        )


# ============================================================
# Liquidations shared fixtures
# ============================================================

@pytest.fixture
def liquidation_state() -> LiquidationState:
    return LiquidationState(
        max_events_per_symbol=100,
    )


@pytest.fixture
def liquidation_metrics() -> LiquidationMetrics:
    return LiquidationMetrics()


@pytest.fixture
def stream_config() -> LiquidationStreamConfig:
    return LiquidationStreamConfig(
        enabled=True,
        exchanges=("binance",),
        symbols=("BTCUSDT", "ETHUSDT"),

        max_buffer_size_per_symbol=100,

        emit_raw_events=True,
        emit_large_events=True,

        large_liquidation_threshold_usd=Decimal("100000"),
        stale_event_threshold_seconds=60,

        publish_topic_raw="market.liquidation.raw",
        publish_topic_normalized="market.liquidation.normalized",
        publish_topic_large="market.liquidation.large",
        publish_topic_health="system.analytics.liquidations.stream.health",
        publish_topic_snapshot="analytics.liquidation.stream.snapshot",

        healthcheck_interval_seconds=1.0,
        snapshot_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        healthcheck_job_name="liquidation_stream_healthcheck",
        snapshot_job_name="liquidation_stream_snapshot",
        cleanup_job_name="liquidation_stream_cleanup",

        scheduler_job_timeout_seconds=1.0,
        scheduler_job_max_retries=0,
        scheduler_job_retry_delay_seconds=0.01,

        reconnect_on_health_degraded=False,
        reconnect_cooldown_seconds=1.0,

        consumer_idle_sleep_seconds=0.001,
        consumer_error_sleep_seconds=0.01,

        deduplication_enabled=True,
        recent_payload_fingerprints_size=1_000,
        recent_large_events_size=100,
    )


@pytest.fixture
def cascade_config() -> CascadeDetectorConfig:
    return CascadeDetectorConfig(
        enabled=True,

        input_topic="market.liquidation.normalized",

        window_seconds=10,
        min_events=3,
        min_total_notional_usd=Decimal("100000"),
        min_side_imbalance_ratio=0.70,

        cooldown_seconds=5,

        acceleration_enabled=True,
        min_acceleration_ratio=1.10,

        price_compaction_enabled=True,
        max_price_range_pct=2.0,

        continuation_score_weight=0.40,
        imbalance_score_weight=0.25,
        notional_score_weight=0.20,
        acceleration_score_weight=0.15,

        low_severity_threshold=0.30,
        medium_severity_threshold=0.55,
        high_severity_threshold=0.75,
        extreme_severity_threshold=0.90,

        publish_topic_detected="analytics.liquidation.cascade_detected",
        publish_topic_exhaustion="analytics.liquidation.exhaustion_detected",
        publish_topic_snapshot="analytics.liquidation.detector.snapshot",
        publish_topic_health="system.analytics.liquidations.detector.health",

        snapshot_interval_seconds=1.0,
        healthcheck_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        snapshot_job_name="liquidation_cascade_detector_snapshot",
        healthcheck_job_name="liquidation_cascade_detector_healthcheck",
        cleanup_job_name="liquidation_cascade_detector_cleanup",

        scheduler_job_timeout_seconds=1.0,
        scheduler_job_max_retries=0,
        scheduler_job_retry_delay_seconds=0.01,

        recent_signals_limit=50,
    )


@pytest.fixture
def fake_exchange_adapter() -> FakeLiquidationExchangeAdapter:
    return FakeLiquidationExchangeAdapter(
        name="binance",
    )


# ============================================================
# Payload fixtures
# ============================================================

@pytest.fixture
def raw_liquidation_payload() -> dict[str, Any]:
    """
    Flat payload, який має нормально пройти LiquidationStream.normalize_event().
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "price": "65000",
        "quantity": "2",
        "timestamp": now_ms,
    }


@pytest.fixture
def raw_large_liquidation_payload() -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "price": "65000",
        "quantity": "3",
        "timestamp": now_ms,
    }


@pytest.fixture
def raw_small_liquidation_payload() -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "price": "65000",
        "quantity": "0.01",
        "timestamp": now_ms,
    }


@pytest.fixture
def raw_invalid_liquidation_payload() -> dict[str, Any]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "side": "UNKNOWN",
        "price": "0",
        "quantity": "0",
        "timestamp": now_ms,
    }


@pytest.fixture
def raw_stale_liquidation_payload() -> dict[str, Any]:
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    stale_ms = int(stale_ts.timestamp() * 1000)

    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "price": "65000",
        "quantity": "2",
        "timestamp": stale_ms,
    }


@pytest.fixture
def binance_force_order_payload() -> dict[str, Any]:
    """
    Binance-like forceOrder payload.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    return {
        "e": "forceOrder",
        "E": now_ms,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "p": "65000",
            "q": "2",
            "T": now_ms,
        },
    }


# ============================================================
# Domain event fixtures
# ============================================================

@pytest.fixture
def liquidation_event() -> LiquidationEvent:
    return LiquidationEvent(
        exchange="binance",
        symbol="BTCUSDT",
        side=LiquidationSide.LONG,
        price=Decimal("65000"),
        quantity=Decimal("2"),
        notional_usd=Decimal("130000"),
        timestamp=datetime.now(timezone.utc),
        source="test",
        correlation_id="test-correlation-id",
    )


@pytest.fixture
def liquidation_events_for_cascade() -> list[LiquidationEvent]:
    now = datetime.now(timezone.utc)

    return [
        LiquidationEvent(
            exchange="binance",
            symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            notional_usd=Decimal("65000"),
            timestamp=now - timedelta(seconds=3),
            source="test",
        ),
        LiquidationEvent(
            exchange="binance",
            symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            price=Decimal("64950"),
            quantity=Decimal("1"),
            notional_usd=Decimal("64950"),
            timestamp=now - timedelta(seconds=2),
            source="test",
        ),
        LiquidationEvent(
            exchange="binance",
            symbol="BTCUSDT",
            side=LiquidationSide.LONG,
            price=Decimal("64900"),
            quantity=Decimal("1"),
            notional_usd=Decimal("64900"),
            timestamp=now - timedelta(seconds=1),
            source="test",
        ),
    ]