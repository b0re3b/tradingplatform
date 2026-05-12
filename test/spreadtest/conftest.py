from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Awaitable, Callable

import pytest

from core.event_bus import Event, EventPriority
from analytics.spreads.config import (
    CrossExchangeSpreadConfig,
    SpotFuturesSpreadConfig,
)
from analytics.spreads.enums import InstrumentType
from analytics.spreads.models import FundingSnapshot, QuoteSnapshot


# =============================================================================
# Fake infrastructure
# =============================================================================

@dataclass(slots=True)
class FakeSubscription:
    """
    Мінімальний fake Subscription для тестів analyzer.register()/unregister().

    Не дублює core.Subscription повністю, а тримає лише те, що потрібно
    analyzer-ам у unit/runtime тестах.
    """

    topic_pattern: str
    handler: Callable[..., Any]
    name: str | None = None


@dataclass(slots=True)
class FakeScheduledJob:
    """
    Мінімальна модель scheduled job для тестування інтеграції зі Scheduler.
    """

    job_id: str
    name: str
    func: Callable[..., Any]
    interval: float
    run_immediately: bool = False
    max_retries: int = 0
    retry_delay: float = 0.0
    timeout: float | None = None
    allow_overlap: bool = False
    enabled: bool = True


class FakeEventBus:
    """
    Fake EventBus для unit/runtime тестів.

    Підтримує:
    - subscribe()
    - unsubscribe()
    - emit()
    - запис усіх emitted events
    - симуляцію emit failure / rejection
    """

    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.emitted: list[dict[str, Any]] = []

        self.should_raise_on_emit: bool = False
        self.should_reject_emit: bool = False

        self.unsubscribe_calls: int = 0
        self.emit_calls: int = 0
        self.subscribe_calls: int = 0

    def subscribe(
        self,
        topic_pattern: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> FakeSubscription:
        self.subscribe_calls += 1

        subscription = FakeSubscription(
            topic_pattern=topic_pattern,
            handler=handler,
            name=name,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        self.unsubscribe_calls += 1

        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    async def emit(
        self,
        topic: str,
        payload: Any = None,
        *,
        priority: EventPriority | None = None,
        **kwargs: Any,
    ) -> bool:
        self.emit_calls += 1

        if self.should_raise_on_emit:
            raise RuntimeError("FakeEventBus emit failed")

        if self.should_reject_emit:
            return False

        self.emitted.append(
            {
                "topic": topic,
                "payload": payload,
                "priority": priority,
                "kwargs": dict(kwargs),
            }
        )
        return True

    def clear(self) -> None:
        self.subscriptions.clear()
        self.emitted.clear()
        self.unsubscribe_calls = 0
        self.emit_calls = 0
        self.subscribe_calls = 0
        self.should_raise_on_emit = False
        self.should_reject_emit = False

    def emitted_topics(self) -> list[str]:
        return [item["topic"] for item in self.emitted]

    def emitted_for_topic(self, topic: str) -> list[dict[str, Any]]:
        return [item for item in self.emitted if item["topic"] == topic]


class FakeScheduler:
    """
    Fake Scheduler для runtime/lifecycle тестів.

    Підтримує:
    - add_interval_job()
    - get_job_by_name()
    - get_job()
    - remove_job()
    - enable_job()
    - disable_job()
    - run_job_now()
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.add_interval_job_calls: int = 0
        self.remove_job_calls: int = 0
        self.enable_job_calls: int = 0
        self.disable_job_calls: int = 0
        self.run_job_now_calls: int = 0

        self._counter: int = 0

    def add_interval_job(
        self,
        *,
        name: str,
        func: Callable[..., Any],
        interval: float,
        run_immediately: bool = False,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        timeout: float | None = None,
        allow_overlap: bool = False,
        enabled: bool = True,
        **_: Any,
    ) -> str:
        self.add_interval_job_calls += 1

        existing = self.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        self._counter += 1
        job_id = f"fake-job-{self._counter}"

        self.jobs[job_id] = FakeScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            run_immediately=run_immediately,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            allow_overlap=allow_overlap,
            enabled=enabled,
        )

        return job_id

    def get_job(self, job_id: str) -> FakeScheduledJob | None:
        return self.jobs.get(job_id)

    def get_job_by_name(self, name: str) -> FakeScheduledJob | None:
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def remove_job(self, job_id: str) -> None:
        self.remove_job_calls += 1
        self.jobs.pop(job_id, None)

    def enable_job(self, job_id: str) -> None:
        self.enable_job_calls += 1
        job = self.jobs.get(job_id)
        if job is not None:
            job.enabled = True

    def disable_job(self, job_id: str) -> None:
        self.disable_job_calls += 1
        job = self.jobs.get(job_id)
        if job is not None:
            job.enabled = False

    async def run_job_now(self, job_id: str) -> Any:
        self.run_job_now_calls += 1

        job = self.jobs[job_id]
        result = job.func()

        if isinstance(result, Awaitable):
            return await result

        return result

    def list_jobs(self) -> list[FakeScheduledJob]:
        return list(self.jobs.values())

    def clear(self) -> None:
        self.jobs.clear()
        self.add_interval_job_calls = 0
        self.remove_job_calls = 0
        self.enable_job_calls = 0
        self.disable_job_calls = 0
        self.run_job_now_calls = 0
        self._counter = 0


# =============================================================================
# Event helpers
# =============================================================================

def make_event(
    topic: str,
    payload: Any,
    *,
    source: str = "test",
    priority: EventPriority = EventPriority.NORMAL,
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """
    Створює core.event_bus.Event для handler-тестів.

    Якщо твій Event у core має інший constructor contract, достатньо буде
    змінити тільки цей helper, а не всі тести.
    """

    kwargs: dict[str, Any] = {
        "topic": topic,
        "payload": payload,
        "source": source,
        "priority": priority,
    }

    if timestamp is not None:
        kwargs["timestamp"] = timestamp

    if metadata is not None:
        kwargs["metadata"] = metadata

    return Event(**kwargs)


# =============================================================================
# Pytest fixtures: infrastructure
# =============================================================================

@pytest.fixture
def event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def now() -> datetime:
    return datetime.utcnow()


@pytest.fixture
def old_time(now: datetime) -> datetime:
    return now - timedelta(minutes=10)


@pytest.fixture
def future_time(now: datetime) -> datetime:
    return now + timedelta(seconds=10)


# =============================================================================
# Pytest fixtures: configs
# =============================================================================

@pytest.fixture
def spot_futures_config() -> SpotFuturesSpreadConfig:
    """
    Config для більшості SpotFuturesSpreadAnalyzer тестів.

    Значення підібрані так, щоб:
    - quote-и не ставали stale випадково;
    - throttling/cooldown не блокували assertions;
    - scheduler jobs були швидкими;
    - signal thresholds були достатньо низькими для контрольованих тестів.
    """

    return SpotFuturesSpreadConfig(
        enabled=True,
        service_name="test_spot_futures_spread_analyzer",
        max_quote_age_ms=60_000,
        max_quote_skew_ms=5_000,
        rolling_window_size=20,
        ema_alpha=Decimal("0.2"),
        min_emit_interval_ms=0,
        cooldown_seconds=0,
        cleanup_interval_seconds=1.0,
        heartbeat_interval_seconds=1.0,
        stale_state_ttl_seconds=60.0,
        max_cached_quotes=1_000,
        max_cached_snapshots=1_000,
        max_cached_windows=1_000,
        anomaly_zscore_threshold=Decimal("2.5"),
        widening_bps_threshold=Decimal("5"),
        min_basis_bps=Decimal("1"),
        mean_reversion_bps_threshold=Decimal("5"),
    )


@pytest.fixture
def cross_exchange_config() -> CrossExchangeSpreadConfig:
    """
    Config для більшості CrossExchangeSpreadAnalyzer тестів.

    Fee/slippage/buffer за замовчуванням нульові, щоб arbitrage assertions
    були простими й детермінованими.
    """

    return CrossExchangeSpreadConfig(
        enabled=True,
        service_name="test_cross_exchange_spread_analyzer",
        max_quote_age_ms=60_000,
        max_quote_skew_ms=5_000,
        rolling_window_size=20,
        ema_alpha=Decimal("0.2"),
        min_emit_interval_ms=0,
        cooldown_seconds=0,
        cleanup_interval_seconds=1.0,
        heartbeat_interval_seconds=1.0,
        stale_state_ttl_seconds=60.0,
        max_cached_quotes=1_000,
        max_cached_snapshots=1_000,
        max_cached_windows=1_000,
        anomaly_zscore_threshold=Decimal("2.5"),
        widening_bps_threshold=Decimal("5"),
        arbitrage_min_bps=Decimal("1"),
        default_trade_size=Decimal("1"),
        min_trade_size=Decimal("0.001"),
        max_trade_size=Decimal("100"),
        default_taker_fee_rate=Decimal("0"),
        slippage_max_bps=Decimal("0"),
        safety_buffer_bps=Decimal("0"),
    )


@pytest.fixture
def disabled_spot_futures_config(
    spot_futures_config: SpotFuturesSpreadConfig,
) -> SpotFuturesSpreadConfig:
    return SpotFuturesSpreadConfig(
        **{
            **spot_futures_config.__dict__,
            "enabled": False,
        }
    )


@pytest.fixture
def disabled_cross_exchange_config(
    cross_exchange_config: CrossExchangeSpreadConfig,
) -> CrossExchangeSpreadConfig:
    return CrossExchangeSpreadConfig(
        **{
            **cross_exchange_config.__dict__,
            "enabled": False,
        }
    )


# =============================================================================
# Quote / funding factories
# =============================================================================

def make_quote(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    instrument_type: InstrumentType = InstrumentType.SPOT,
    bid: Decimal | str | None = Decimal("100"),
    ask: Decimal | str | None = Decimal("101"),
    bid_size: Decimal | str | None = Decimal("10"),
    ask_size: Decimal | str | None = Decimal("10"),
    last_price: Decimal | str | None = None,
    mark_price: Decimal | str | None = None,
    index_price: Decimal | str | None = None,
    timestamp: datetime | None = None,
    received_at: datetime | None = None,
    sequence_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> QuoteSnapshot:
    resolved_timestamp = timestamp or datetime.utcnow()
    resolved_received_at = received_at or resolved_timestamp

    return QuoteSnapshot(
        exchange=exchange,
        symbol=symbol,
        instrument_type=instrument_type,
        bid=Decimal(str(bid)) if bid is not None else None,
        ask=Decimal(str(ask)) if ask is not None else None,
        bid_size=Decimal(str(bid_size)) if bid_size is not None else None,
        ask_size=Decimal(str(ask_size)) if ask_size is not None else None,
        last_price=Decimal(str(last_price)) if last_price is not None else None,
        mark_price=Decimal(str(mark_price)) if mark_price is not None else None,
        index_price=Decimal(str(index_price)) if index_price is not None else None,
        timestamp=resolved_timestamp,
        received_at=resolved_received_at,
        sequence_id=sequence_id,
        metadata=metadata or {},
    )


def make_funding(
    *,
    exchange: str = "binance",
    symbol: str = "BTCUSDT",
    funding_rate: Decimal | str = Decimal("0.0001"),
    timestamp: datetime | None = None,
    next_funding_time: datetime | None = None,
    predicted_rate: Decimal | str | None = None,
    interval_hours: int | None = 8,
    metadata: dict[str, Any] | None = None,
) -> FundingSnapshot:
    return FundingSnapshot(
        exchange=exchange,
        symbol=symbol,
        funding_rate=Decimal(str(funding_rate)),
        timestamp=timestamp or datetime.utcnow(),
        next_funding_time=next_funding_time,
        predicted_rate=(
            Decimal(str(predicted_rate))
            if predicted_rate is not None
            else None
        ),
        interval_hours=interval_hours,
        metadata=metadata or {},
    )


@pytest.fixture
def quote_factory() -> Callable[..., QuoteSnapshot]:
    return make_quote


@pytest.fixture
def funding_factory() -> Callable[..., FundingSnapshot]:
    return make_funding


# =============================================================================
# Common market data fixtures
# =============================================================================

@pytest.fixture
def spot_quote(now: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("100"),
        ask=Decimal("101"),
        timestamp=now,
        received_at=now,
    )


@pytest.fixture
def futures_quote(now: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.PERPETUAL,
        bid=Decimal("103"),
        ask=Decimal("104"),
        timestamp=now,
        received_at=now,
    )


@pytest.fixture
def funding_snapshot(now: datetime) -> FundingSnapshot:
    return make_funding(
        exchange="binance",
        symbol="BTCUSDT",
        funding_rate=Decimal("0.0001"),
        timestamp=now,
    )


@pytest.fixture
def binance_quote(now: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("100"),
        ask=Decimal("101"),
        timestamp=now,
        received_at=now,
    )


@pytest.fixture
def bybit_quote(now: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="bybit",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("103"),
        ask=Decimal("104"),
        timestamp=now,
        received_at=now,
    )


@pytest.fixture
def profitable_cross_exchange_quotes(
    now: datetime,
) -> tuple[QuoteSnapshot, QuoteSnapshot]:
    """
    Binance дешевший, Bybit дорожчий:
    buy binance ask=100, sell bybit bid=103.
    """

    cheap_quote = make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("99"),
        ask=Decimal("100"),
        timestamp=now,
        received_at=now,
    )

    expensive_quote = make_quote(
        exchange="bybit",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("103"),
        ask=Decimal("104"),
        timestamp=now,
        received_at=now,
    )

    return cheap_quote, expensive_quote


@pytest.fixture
def stale_quote(old_time: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("100"),
        ask=Decimal("101"),
        timestamp=old_time,
        received_at=old_time,
    )


@pytest.fixture
def incomplete_quote(now: datetime) -> QuoteSnapshot:
    return make_quote(
        exchange="binance",
        symbol="BTCUSDT",
        instrument_type=InstrumentType.SPOT,
        bid=Decimal("100"),
        ask=None,
        timestamp=now,
        received_at=now,
    )


# =============================================================================
# Event fixtures
# =============================================================================

@pytest.fixture
def quote_event_factory() -> Callable[[QuoteSnapshot], Event]:
    def _factory(quote: QuoteSnapshot) -> Event:
        return make_event("market.quote.updated", quote)

    return _factory


@pytest.fixture
def funding_event_factory() -> Callable[[FundingSnapshot], Event]:
    def _factory(funding: FundingSnapshot) -> Event:
        return make_event("market.funding.updated", funding)

    return _factory


@pytest.fixture
def invalid_payload_event() -> Event:
    return make_event(
        "market.quote.updated",
        {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "bid": "100",
            "ask": "101",
        },
    )


# =============================================================================
# Assertion helpers
# =============================================================================

def assert_emitted_topic(event_bus: FakeEventBus, topic: str) -> None:
    assert topic in event_bus.emitted_topics()


def assert_not_emitted_topic(event_bus: FakeEventBus, topic: str) -> None:
    assert topic not in event_bus.emitted_topics()


@pytest.fixture
def emitted_topic_assertion() -> Callable[[FakeEventBus, str], None]:
    return assert_emitted_topic


@pytest.fixture
def not_emitted_topic_assertion() -> Callable[[FakeEventBus, str], None]:
    return assert_not_emitted_topic