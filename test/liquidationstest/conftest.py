# tests/analytics/liquidations/conftest.py

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from core.event_bus import Event, EventBus, QueueFullPolicy
from core.scheduler import Scheduler

from analytics.liquidations.config import (
    CascadeDetectorConfig,
    LiquidationStreamConfig,
)
from analytics.liquidations.enums import LiquidationSide
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import LiquidationEvent
from analytics.liquidations.state import LiquidationState


# =============================================================================
# Time helpers
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


# =============================================================================
# Async test helpers
# =============================================================================

@pytest.fixture
def wait_for_events() -> Callable[..., Any]:
    """
    Helper для очікування EventBus events у тестах.

    Використання:
        await wait_for_events(received, expected_count=1)
    """

    async def _wait_for_events(
        received: list[Event],
        *,
        expected_count: int = 1,
        timeout: float = 0.75,
        poll_interval: float = 0.01,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout

        while len(received) < expected_count:
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(poll_interval)

    return _wait_for_events


@pytest.fixture
def event_collector() -> Callable[[EventBus, str], list[Event]]:
    """
    Створює collector для конкретного topic.

    Використання:
        received = event_collector(event_bus, "market.liquidation.normalized")
    """

    def _collector(event_bus: EventBus, topic: str) -> list[Event]:
        received: list[Event] = []

        event_bus.subscribe(
            topic,
            lambda event: received.append(event),
            name=f"test.collector.{topic}",
        )

        return received

    return _collector


# =============================================================================
# Core fixtures
# =============================================================================

@pytest_asyncio.fixture
async def event_bus() -> AsyncGenerator[EventBus, None]:
    """
    Реальний core.EventBus.

    Навмисно використовуємо справжній EventBus, а не mock:
    - перевіряємо реальні subscriptions;
    - перевіряємо async queue processing;
    - перевіряємо correlation_id/source/headers propagation;
    - ближче до production-поведінки.
    """
    bus = EventBus(
        max_queue_size=2_000,
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
    Реальний core.Scheduler.

    Scheduler потрібен для перевірки:
    - healthcheck jobs;
    - snapshot jobs;
    - cleanup jobs;
    - коректного unregister/close lifecycle.
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


# =============================================================================
# Shared domain fixtures
# =============================================================================

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
    """
    Config для LiquidationStream у новій логіці.

    Stream:
        слухає market.liquidation;
        публікує market.liquidation.normalized;
        публікує market.liquidation.large;
        публікує market.liquidations.updated.

    Важливо:
        market_types/timeframes не обмежуємо занадто вузько,
        щоб у тестах можна було перевіряти full-scope isolation.
    """
    return LiquidationStreamConfig(
        enabled=True,

        exchanges=("binance", "bybit", "okx"),
        symbols=("BTCUSDT", "ETHUSDT", "BTCUSD"),
        market_types=("usdm_futures", "coinm_futures", "linear", "inverse", "swap"),
        timeframes=("realtime", "1m"),

        max_buffer_size_per_symbol=100,

        emit_raw_events=True,
        emit_large_events=True,
        emit_updated_events=True,

        large_liquidation_threshold_usd=Decimal("100000"),
        stale_event_threshold_seconds=60,

        input_topic_raw="market.liquidation",
        input_topics_raw=("market.liquidation",),

        publish_topic_raw="market.liquidation.raw",
        publish_topic_normalized="market.liquidation.normalized",
        publish_topic_large="market.liquidation.large",
        publish_topic_updated="market.liquidations.updated",
        publish_topic_health="system.analytics.liquidations.stream.health",
        publish_topic_snapshot="analytics.liquidations.stream.snapshot",

        healthcheck_interval_seconds=1.0,
        snapshot_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        healthcheck_job_name="test.liquidations.stream.healthcheck",
        snapshot_job_name="test.liquidations.stream.snapshot",
        cleanup_job_name="test.liquidations.stream.cleanup",

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
def strict_btc_usdm_stream_config() -> LiquidationStreamConfig:
    """
    Вузький config для тестів scoped filtering.

    Має пропускати тільки:
        binance + BTCUSDT + usdm_futures + realtime
    """
    return LiquidationStreamConfig(
        enabled=True,

        exchanges=("binance",),
        symbols=("BTCUSDT",),
        market_types=("usdm_futures",),
        timeframes=("realtime",),

        max_buffer_size_per_symbol=100,

        emit_raw_events=True,
        emit_large_events=True,
        emit_updated_events=True,

        large_liquidation_threshold_usd=Decimal("100000"),
        stale_event_threshold_seconds=60,

        input_topic_raw="market.liquidation",
        input_topics_raw=("market.liquidation",),

        publish_topic_raw="market.liquidation.raw",
        publish_topic_normalized="market.liquidation.normalized",
        publish_topic_large="market.liquidation.large",
        publish_topic_updated="market.liquidations.updated",
        publish_topic_health="system.analytics.liquidations.stream.health",
        publish_topic_snapshot="analytics.liquidations.stream.snapshot",

        healthcheck_interval_seconds=1.0,
        snapshot_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        healthcheck_job_name="test.strict.liquidations.stream.healthcheck",
        snapshot_job_name="test.strict.liquidations.stream.snapshot",
        cleanup_job_name="test.strict.liquidations.stream.cleanup",

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
    """
    Config для CascadeDetector.

    Налаштування підібрані так, щоб:
    - 3 liquidation events могли сформувати cascade;
    - acceleration/price compaction реально перевірялися;
    - cooldown був достатньо короткий для unit-тестів.
    """
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

        publish_topic_detected="analytics.liquidations.cascade_detected",
        publish_topic_exhaustion="analytics.liquidations.exhaustion_detected",
        publish_topic_snapshot="analytics.liquidations.detector.snapshot",
        publish_topic_health="system.analytics.liquidations.detector.health",

        snapshot_interval_seconds=1.0,
        healthcheck_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        snapshot_job_name="test.liquidations.detector.snapshot",
        healthcheck_job_name="test.liquidations.detector.healthcheck",
        cleanup_job_name="test.liquidations.detector.cleanup",

        scheduler_job_timeout_seconds=1.0,
        scheduler_job_max_retries=0,
        scheduler_job_retry_delay_seconds=0.01,

        recent_signals_limit=50,
    )


@pytest.fixture
def strict_cascade_config() -> CascadeDetectorConfig:
    """
    Трохи жорсткіший detector config для threshold-negative тестів.
    """
    return CascadeDetectorConfig(
        enabled=True,

        input_topic="market.liquidation.normalized",

        window_seconds=10,
        min_events=4,
        min_total_notional_usd=Decimal("500000"),
        min_side_imbalance_ratio=0.85,

        cooldown_seconds=5,

        acceleration_enabled=True,
        min_acceleration_ratio=1.50,

        price_compaction_enabled=True,
        max_price_range_pct=0.50,

        continuation_score_weight=0.40,
        imbalance_score_weight=0.25,
        notional_score_weight=0.20,
        acceleration_score_weight=0.15,

        low_severity_threshold=0.30,
        medium_severity_threshold=0.55,
        high_severity_threshold=0.75,
        extreme_severity_threshold=0.90,

        publish_topic_detected="analytics.liquidations.cascade_detected",
        publish_topic_exhaustion="analytics.liquidations.exhaustion_detected",
        publish_topic_snapshot="analytics.liquidations.detector.snapshot",
        publish_topic_health="system.analytics.liquidations.detector.health",

        snapshot_interval_seconds=1.0,
        healthcheck_interval_seconds=1.0,
        cleanup_interval_seconds=1.0,

        snapshot_job_name="test.strict.liquidations.detector.snapshot",
        healthcheck_job_name="test.strict.liquidations.detector.healthcheck",
        cleanup_job_name="test.strict.liquidations.detector.cleanup",

        scheduler_job_timeout_seconds=1.0,
        scheduler_job_max_retries=0,
        scheduler_job_retry_delay_seconds=0.01,

        recent_signals_limit=50,
    )


# =============================================================================
# Raw payload factories
# =============================================================================

@pytest.fixture
def make_raw_liquidation_payload() -> Callable[..., dict[str, Any]]:
    """
    Factory для flat normalized exchange payload.

    Це payload, який exchange adapter мав би публікувати в:
        EventBus.emit("market.liquidation", payload)

    За замовчуванням side="SELL":
        SELL liquidation order => liquidation of LONG position.
    """

    def _make_raw_liquidation_payload(
        *,
        exchange: str = "binance",
        symbol: str = "BTCUSDT",
        exchange_symbol: str | None = None,
        market_type: str = "usdm_futures",
        timeframe: str = "realtime",
        side: str = "SELL",
        price: str | Decimal = "65000",
        quantity: str | Decimal = "2",
        timestamp: datetime | int | None = None,
        trade_id: str | None = "test-trade-1",
        order_id: str | None = "test-order-1",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if timestamp is None:
            timestamp_ms = to_ms(utc_now())
        elif isinstance(timestamp, datetime):
            timestamp_ms = to_ms(timestamp)
        else:
            timestamp_ms = int(timestamp)

        payload: dict[str, Any] = {
            "exchange": exchange,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "market_type": market_type,
            "timeframe": timeframe,
            "side": side,
            "price": str(price),
            "quantity": str(quantity),
            "timestamp": timestamp_ms,
            "trade_id": trade_id,
            "order_id": order_id,
        }

        if extra:
            payload.update(extra)

        return payload

    return _make_raw_liquidation_payload


@pytest.fixture
def raw_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="SELL",
        price="65000",
        quantity="2",
        trade_id="binance-btc-liquidation-1",
        order_id="binance-btc-order-1",
    )


@pytest.fixture
def raw_large_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="SELL",
        price="65000",
        quantity="3",
        trade_id="binance-btc-large-liquidation-1",
        order_id="binance-btc-large-order-1",
    )


@pytest.fixture
def raw_small_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="SELL",
        price="65000",
        quantity="0.01",
        trade_id="binance-btc-small-liquidation-1",
        order_id="binance-btc-small-order-1",
    )


@pytest.fixture
def raw_invalid_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="UNKNOWN",
        price="0",
        quantity="0",
        trade_id="invalid-liquidation-1",
        order_id="invalid-order-1",
    )


@pytest.fixture
def raw_stale_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="SELL",
        price="65000",
        quantity="2",
        timestamp=utc_now() - timedelta(minutes=10),
        trade_id="stale-liquidation-1",
        order_id="stale-order-1",
    )


@pytest.fixture
def raw_buy_side_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """
    BUY liquidation order має нормалізуватися як SHORT liquidation.
    """
    return make_raw_liquidation_payload(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side="BUY",
        price="65000",
        quantity="2",
        trade_id="buy-side-liquidation-1",
        order_id="buy-side-order-1",
    )


# =============================================================================
# Exchange-specific raw payloads
# =============================================================================

@pytest.fixture
def binance_force_order_payload() -> dict[str, Any]:
    """
    Binance-like forceOrder payload.

    Очікування:
        S="SELL" => LONG liquidation.
    """
    now_ms = to_ms(utc_now())

    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": "realtime",
        "e": "forceOrder",
        "E": now_ms,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "p": "65000",
            "q": "2",
            "T": now_ms,
            "i": "binance-force-order-1",
        },
    }


@pytest.fixture
def bybit_linear_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="bybit",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="linear",
        timeframe="realtime",
        side="Sell",
        price="65000",
        quantity="2",
        trade_id="bybit-linear-liquidation-1",
        order_id="bybit-linear-order-1",
    )


@pytest.fixture
def okx_swap_liquidation_payload(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return make_raw_liquidation_payload(
        exchange="okx",
        symbol="BTCUSDT",
        exchange_symbol="BTC-USDT-SWAP",
        market_type="swap",
        timeframe="realtime",
        side="sell",
        price="65000",
        quantity="2",
        trade_id="okx-swap-liquidation-1",
        order_id="okx-swap-order-1",
    )


# =============================================================================
# Full-scope isolation payloads
# =============================================================================

@pytest.fixture
def same_symbol_different_market_type_payloads(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Однаковий exchange+symbol, але різний market_type.

    Ці payload-и не мають потрапити в один state/window.
    """
    now = utc_now()

    return [
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            price="65000",
            quantity="1",
            timestamp=now,
            trade_id="same-symbol-usdm-1",
            order_id="same-symbol-usdm-order-1",
        ),
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSD_PERP",
            market_type="coinm_futures",
            timeframe="realtime",
            side="SELL",
            price="65000",
            quantity="1",
            timestamp=now + timedelta(milliseconds=1),
            trade_id="same-symbol-coinm-1",
            order_id="same-symbol-coinm-order-1",
        ),
    ]


@pytest.fixture
def same_symbol_different_timeframe_payloads(
    make_raw_liquidation_payload: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Однаковий exchange+market_type+symbol, але різний timeframe.
    """
    now = utc_now()

    return [
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side="SELL",
            price="65000",
            quantity="1",
            timestamp=now,
            trade_id="same-symbol-realtime-1",
            order_id="same-symbol-realtime-order-1",
        ),
        make_raw_liquidation_payload(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="1m",
            side="SELL",
            price="65000",
            quantity="1",
            timestamp=now + timedelta(milliseconds=1),
            trade_id="same-symbol-1m-1",
            order_id="same-symbol-1m-order-1",
        ),
    ]


# =============================================================================
# Domain event factories
# =============================================================================

@pytest.fixture
def make_liquidation_event() -> Callable[..., LiquidationEvent]:
    def _make_liquidation_event(
        *,
        exchange: str = "binance",
        symbol: str = "BTCUSDT",
        exchange_symbol: str | None = None,
        market_type: str = "usdm_futures",
        timeframe: str = "realtime",
        side: LiquidationSide = LiquidationSide.LONG,
        price: Decimal = Decimal("65000"),
        quantity: Decimal = Decimal("2"),
        notional_usd: Decimal | None = None,
        timestamp: datetime | None = None,
        trade_id: str | None = None,
        order_id: str | None = None,
        event_id: str | None = None,
        correlation_id: str | None = "test-correlation-id",
        source: str = "test",
        metadata: dict[str, Any] | None = None,
    ) -> LiquidationEvent:
        resolved_notional = notional_usd if notional_usd is not None else price * quantity

        return LiquidationEvent(
            exchange=exchange,
            symbol=symbol,
            exchange_symbol=exchange_symbol or symbol,
            market_type=market_type,
            timeframe=timeframe,
            side=side,
            price=price,
            quantity=quantity,
            notional_usd=resolved_notional,
            timestamp=timestamp or utc_now(),
            trade_id=trade_id,
            order_id=order_id,
            event_id=event_id,
            correlation_id=correlation_id,
            source=source,
            metadata=metadata or {},
        )

    return _make_liquidation_event


@pytest.fixture
def liquidation_event(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> LiquidationEvent:
    return make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side=LiquidationSide.LONG,
        price=Decimal("65000"),
        quantity=Decimal("2"),
        trade_id="domain-liquidation-1",
        order_id="domain-order-1",
    )


@pytest.fixture
def short_liquidation_event(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> LiquidationEvent:
    return make_liquidation_event(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        market_type="usdm_futures",
        timeframe="realtime",
        side=LiquidationSide.SHORT,
        price=Decimal("65000"),
        quantity=Decimal("2"),
        trade_id="domain-short-liquidation-1",
        order_id="domain-short-order-1",
    )


@pytest.fixture
def liquidation_events_for_cascade(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Події, які мають проходити cascade thresholds.

    Всі events мають один full scope:
        binance:usdm_futures:BTCUSDT:realtime

    Timestamps розставлені так, щоб друга половина window була інтенсивнішою,
    а acceleration_ratio міг пройти min_acceleration_ratio.
    """
    now = utc_now()

    return [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            timestamp=now - timedelta(seconds=3),
            trade_id="cascade-1",
            order_id="cascade-order-1",
            correlation_id="cascade-correlation-id",
        ),
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("64950"),
            quantity=Decimal("1.2"),
            timestamp=now - timedelta(seconds=1),
            trade_id="cascade-2",
            order_id="cascade-order-2",
            correlation_id="cascade-correlation-id",
        ),
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("64900"),
            quantity=Decimal("1.5"),
            timestamp=now - timedelta(milliseconds=200),
            trade_id="cascade-3",
            order_id="cascade-order-3",
            correlation_id="cascade-correlation-id",
        ),
    ]


@pytest.fixture
def mixed_side_liquidation_events(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Події з недостатнім side dominance.

    Корисно для negative threshold tests.
    """
    now = utc_now()

    return [
        make_liquidation_event(
            side=LiquidationSide.LONG,
            timestamp=now - timedelta(seconds=3),
            quantity=Decimal("1"),
            trade_id="mixed-side-1",
        ),
        make_liquidation_event(
            side=LiquidationSide.SHORT,
            timestamp=now - timedelta(seconds=2),
            quantity=Decimal("1"),
            trade_id="mixed-side-2",
        ),
        make_liquidation_event(
            side=LiquidationSide.LONG,
            timestamp=now - timedelta(seconds=1),
            quantity=Decimal("1"),
            trade_id="mixed-side-3",
        ),
    ]


@pytest.fixture
def same_symbol_different_market_type_events(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Однаковий exchange+symbol, але різний market_type.

    Має використовуватись для тестів, які доводять, що detector/state
    не змішують usdm_futures і coinm_futures.
    """
    now = utc_now()

    return [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            timestamp=now - timedelta(seconds=3),
            trade_id="usdm-event-1",
            correlation_id="usdm-correlation-id",
        ),
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSD_PERP",
            market_type="coinm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            timestamp=now - timedelta(seconds=2),
            trade_id="coinm-event-1",
            correlation_id="coinm-correlation-id",
        ),
    ]


@pytest.fixture
def same_symbol_different_timeframe_events(
    make_liquidation_event: Callable[..., LiquidationEvent],
) -> list[LiquidationEvent]:
    """
    Однаковий exchange+market_type+symbol, але різний timeframe.
    """
    now = utc_now()

    return [
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="realtime",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            timestamp=now - timedelta(seconds=3),
            trade_id="realtime-event-1",
            correlation_id="realtime-correlation-id",
        ),
        make_liquidation_event(
            exchange="binance",
            symbol="BTCUSDT",
            exchange_symbol="BTCUSDT",
            market_type="usdm_futures",
            timeframe="1m",
            side=LiquidationSide.LONG,
            price=Decimal("65000"),
            quantity=Decimal("1"),
            timestamp=now - timedelta(seconds=2),
            trade_id="one-minute-event-1",
            correlation_id="one-minute-correlation-id",
        ),
    ]


# =============================================================================
# State seeding helpers
# =============================================================================

@pytest.fixture
def seed_liquidation_state() -> Callable[[LiquidationState, list[LiquidationEvent]], None]:
    def _seed_liquidation_state(
        state: LiquidationState,
        events: list[LiquidationEvent],
    ) -> None:
        for event in events:
            state.add_event(event)

    return _seed_liquidation_state