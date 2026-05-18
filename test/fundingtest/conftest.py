# tests/analytics/funding/conftest.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from analytics.funding.config import FundingAnalyzerConfig
from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingTimeframe,
)
from analytics.funding.funding_analyzer import FundingAnalyzer, FundingMarketContext
from analytics.funding.funding_divergence import FundingDivergenceDetector
from analytics.funding.funding_extremes import FundingExtremesDetector
from analytics.funding.funding_flip_detector import FundingFlipDetector
from analytics.funding.funding_pressure import FundingPressureAnalyzer
from analytics.funding.funding_regime_detector import FundingRegimeDetector
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
    make_funding_key,
)


# =============================================================================
# Canonical hard-test scope
# =============================================================================

TEST_EXCHANGE = "binance"
TEST_EXCHANGE_ENUM = FundingDataSource.BINANCE
TEST_MARKET_TYPE = "usdm_futures"
TEST_SYMBOL = "BTCUSDT"
TEST_EXCHANGE_SYMBOL = "BTCUSDT"
TEST_TIMEFRAME = FundingTimeframe.H1
TEST_KEY = (
    TEST_EXCHANGE,
    TEST_MARKET_TYPE,
    TEST_SYMBOL,
    TEST_TIMEFRAME.value,
)


# =============================================================================
# Time fixtures
# =============================================================================

@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def earlier_utc(now_utc: datetime) -> datetime:
    return now_utc - timedelta(hours=1)


@pytest.fixture
def later_utc(now_utc: datetime) -> datetime:
    return now_utc + timedelta(hours=1)


# =============================================================================
# Fake Event / EventBus / Scheduler
# =============================================================================

@dataclass(slots=True)
class FakePublishedEvent:
    topic: str
    payload: dict[str, Any]
    kwargs: dict[str, Any]


class FakeEvent:
    """
    Мінімальний Event-compatible object.

    FundingAnalyzer зараз читає:
    - event.payload
    - event.correlation_id

    Але topic/source/metadata залишаємо для жорстких assertions
    у event-flow тестах.
    """

    def __init__(
        self,
        payload: Any,
        *,
        topic: str = "test.event",
        correlation_id: str | None = "test-correlation-id",
        source: str = "tests.analytics.funding",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.topic = topic
        self.payload = payload
        self.correlation_id = correlation_id
        self.source = source
        self.metadata = metadata or {}


class FakeEventBus:
    """
    Fake під реальний мінімальний контракт core.EventBus, який використовує FundingAnalyzer:

    - subscribe(topic, handler, name=...)
    - unsubscribe(subscription)
    - async emit(topic, payload, **kwargs)

    Важливо:
    - не викликає handlers автоматично;
    - зберігає всі publish attempts;
    - дозволяє жорстко перевіряти topic, payload, priority, source, correlation_id, headers.
    """

    def __init__(self) -> None:
        self.subscriptions: list[Any] = []
        self.unsubscribed: list[Any] = []
        self.published: list[FakePublishedEvent] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        subscription = SimpleNamespace(
            topic=topic,
            pattern=topic,
            handler=handler,
            name=name,
            kwargs=kwargs,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Any) -> None:
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.published.append(
            FakePublishedEvent(
                topic=topic,
                payload=payload or {},
                kwargs=kwargs,
            )
        )

    def topics(self) -> list[str]:
        return [event.topic for event in self.published]

    def payloads_for(self, topic: str) -> list[dict[str, Any]]:
        return [event.payload for event in self.published if event.topic == topic]

    def last_payload_for(self, topic: str) -> dict[str, Any] | None:
        payloads = self.payloads_for(topic)
        return payloads[-1] if payloads else None

    def clear_published(self) -> None:
        self.published.clear()


class FailingEmitEventBus(FakeEventBus):
    """
    EventBus, який падає на emit.

    Потрібен для resilience-тестів:
    - analyzer не має залишати lock у locked-state;
    - state має бути або консистентним, або явно перевіреним після failure.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__()
        self.exc = exc or RuntimeError("fake event bus emit failed")
        self.emit_attempts: list[FakePublishedEvent] = []

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        attempt = FakePublishedEvent(topic=topic, payload=payload or {}, kwargs=kwargs)
        self.emit_attempts.append(attempt)
        raise self.exc


@dataclass(slots=True)
class FakeScheduledJob:
    job_id: str
    name: str
    func: Callable[..., Any]
    interval: float
    timeout: float | None
    max_retries: int
    retry_delay: float
    allow_overlap: bool
    run_immediately: bool
    enabled: bool


class FakeScheduler:
    """
    Fake під мінімальний контракт core.Scheduler, який використовує FundingAnalyzer:

    - get_job_by_name(name)
    - add_interval_job(...)
    - disable_job(job_id)
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.disabled_job_ids: list[str] = []
        self.added_jobs: list[FakeScheduledJob] = []
        self._counter = 0

    def get_job_by_name(self, name: str) -> FakeScheduledJob | None:
        for job in self.jobs.values():
            if job.name == name:
                return job
        return None

    def add_interval_job(
        self,
        *,
        name: str,
        func: Callable[..., Any],
        interval: float,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        allow_overlap: bool = False,
        run_immediately: bool = False,
        enabled: bool = True,
        **_: Any,
    ) -> str:
        self._counter += 1
        job_id = f"fake-job-{self._counter}"

        job = FakeScheduledJob(
            job_id=job_id,
            name=name,
            func=func,
            interval=interval,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            allow_overlap=allow_overlap,
            run_immediately=run_immediately,
            enabled=enabled,
        )

        self.jobs[job_id] = job
        self.added_jobs.append(job)
        return job_id

    def disable_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise KeyError(job_id)

        self.jobs[job_id].enabled = False
        self.disabled_job_ids.append(job_id)


@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def failing_event_bus() -> FailingEmitEventBus:
    return FailingEmitEventBus()


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def make_event() -> Callable[..., FakeEvent]:
    def _make_event(
        payload: Any,
        *,
        topic: str = "test.event",
        correlation_id: str | None = "test-correlation-id",
        source: str = "tests.analytics.funding",
        metadata: dict[str, Any] | None = None,
    ) -> FakeEvent:
        return FakeEvent(
            payload=payload,
            topic=topic,
            correlation_id=correlation_id,
            source=source,
            metadata=metadata,
        )

    return _make_event


# =============================================================================
# Config fixtures
# =============================================================================

@pytest.fixture
def funding_analyzer_config() -> FundingAnalyzerConfig:
    """
    Default hard-test config.

    Parquet вимикаємо, бо unit/integration тести analyzer-а не мають
    торкатися файлового storage, якщо це не окремий parquet test.
    Cooldown/emit interval ставимо 0, щоб assertions не були flaky.
    """

    return FundingAnalyzerConfig(
        enabled=True,
        default_market_type=TEST_MARKET_TYPE,
        default_timeframe=TEST_TIMEFRAME,
        allowed_market_types={TEST_MARKET_TYPE, "linear", "swap"},
        max_history_per_key=50,
        history_window_size=50,
        statistics_window_size=20,
        min_samples_for_statistics=1,
        max_tracked_keys=100,
        max_cached_contexts=100,
        max_cached_statistics=100,
        max_cached_signals=100,
        max_context_age_ms=60_000,
        max_snapshot_age_ms=60_000,
        use_open_interest_context=True,
        use_price_context=True,
        use_trades_context=True,
        use_cvd_context=True,
        use_liquidation_context=True,
        emit_snapshots=True,
        emit_regime_events=True,
        emit_extreme_events=True,
        emit_flip_events=True,
        emit_divergence_events=True,
        emit_pressure_events=True,
        emit_signals=True,
        emit_analytics_events=True,
        signal_on_regime_change=True,
        signal_on_high_pressure=True,
        signal_on_flip=True,
        signal_on_extreme=True,
        signal_on_divergence=True,
        signal_min_confidence=0.0,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        cleanup_interval_sec=60.0,
        heartbeat_interval_sec=60.0,
        stale_state_ttl_sec=3600.0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
        parquet_flush_interval_sec=60.0,
        metadata={"test_suite": "analytics.funding.hard"},
    )


@pytest.fixture
def funding_analyzer_config_no_emit() -> FundingAnalyzerConfig:
    return FundingAnalyzerConfig(
        default_market_type=TEST_MARKET_TYPE,
        default_timeframe=TEST_TIMEFRAME,
        min_samples_for_statistics=1,
        emit_snapshots=False,
        emit_regime_events=False,
        emit_extreme_events=False,
        emit_flip_events=False,
        emit_divergence_events=False,
        emit_pressure_events=False,
        emit_signals=False,
        emit_analytics_events=False,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )


# =============================================================================
# Payload factories
# =============================================================================

@pytest.fixture
def make_funding_payload(
    now_utc: datetime,
    later_utc: datetime,
) -> Callable[..., dict[str, Any]]:
    def _make_funding_payload(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: str = TEST_EXCHANGE,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: str | FundingTimeframe = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        funding_rate: float | str = 0.0001,
        predicted_funding_rate: float | str | None = 0.00012,
        mark_price: float | str | None = 50_000.0,
        index_price: float | str | None = 49_950.0,
        open_interest: float | str | None = 1_000_000.0,
        volume_24h: float | str | None = 250_000_000.0,
        event_time: datetime | str | int | float | None = None,
        received_at: datetime | str | int | float | None = None,
        next_funding_time: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe.value if isinstance(timeframe, FundingTimeframe) else timeframe,
            "exchange_symbol": exchange_symbol,
            "funding_rate": funding_rate,
            "predicted_funding_rate": predicted_funding_rate,
            "mark_price": mark_price,
            "index_price": index_price,
            "open_interest": open_interest,
            "volume_24h": volume_24h,
            "event_time": event_time or now_utc,
            "received_at": received_at or now_utc,
            "next_funding_time": next_funding_time or later_utc,
            "metadata": metadata or {"origin": "unit-test", "stream": "funding"},
        }
        payload.update(extra)
        return payload

    return _make_funding_payload


@pytest.fixture
def make_context_payload(now_utc: datetime) -> Callable[..., dict[str, Any]]:
    def _make_context_payload(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: str = TEST_EXCHANGE,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: str | FundingTimeframe = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        open_interest: float | str | None = 1_000_000.0,
        open_interest_value: float | str | None = None,
        close: float | str | None = 50_000.0,
        price: float | str | None = 50_000.0,
        cvd: float | str | None = 10_000.0,
        cumulative_volume_delta: float | str | None = None,
        side: str = "long",
        qty: float | str | None = None,
        quantity: float | str | None = None,
        notional: float | str | None = 100_000.0,
        long_liquidations: float | str | None = None,
        short_liquidations: float | str | None = None,
        event_time: datetime | str | int | float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe.value if isinstance(timeframe, FundingTimeframe) else timeframe,
            "exchange_symbol": exchange_symbol,
            "open_interest": open_interest,
            "open_interest_value": open_interest_value,
            "close": close,
            "price": price,
            "cvd": cvd,
            "cumulative_volume_delta": cumulative_volume_delta,
            "side": side,
            "qty": qty,
            "quantity": quantity,
            "notional": notional,
            "long_liquidations": long_liquidations,
            "short_liquidations": short_liquidations,
            "event_time": event_time or now_utc,
        }
        payload.update(extra)
        return payload

    return _make_context_payload


@pytest.fixture
def make_nested_trade_payload(
    make_context_payload: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def _make_nested_trade_payload(**kwargs: Any) -> dict[str, Any]:
        base = make_context_payload(**kwargs)
        return {
            "exchange": base["exchange"],
            "market_type": base["market_type"],
            "symbol": base["symbol"],
            "timeframe": base["timeframe"],
            "exchange_symbol": base["exchange_symbol"],
            "trade": {
                "exchange": base["exchange"],
                "market_type": base["market_type"],
                "symbol": base["symbol"],
                "timeframe": base["timeframe"],
                "exchange_symbol": base["exchange_symbol"],
                "price": base["price"],
            },
        }

    return _make_nested_trade_payload


@pytest.fixture
def make_nested_liquidation_payload(
    make_context_payload: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def _make_nested_liquidation_payload(**kwargs: Any) -> dict[str, Any]:
        base = make_context_payload(**kwargs)
        return {
            "exchange": base["exchange"],
            "market_type": base["market_type"],
            "symbol": base["symbol"],
            "timeframe": base["timeframe"],
            "exchange_symbol": base["exchange_symbol"],
            "liquidation": {
                "exchange": base["exchange"],
                "market_type": base["market_type"],
                "symbol": base["symbol"],
                "timeframe": base["timeframe"],
                "exchange_symbol": base["exchange_symbol"],
                "side": base["side"],
                "qty": base["qty"],
                "quantity": base["quantity"],
                "price": base["price"],
                "notional": base["notional"],
            },
        }

    return _make_nested_liquidation_payload


# =============================================================================
# Model factories
# =============================================================================

@pytest.fixture
def make_snapshot(now_utc: datetime, later_utc: datetime) -> Callable[..., FundingSnapshot]:
    def _make_snapshot(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        funding_rate: float = 0.0001,
        predicted_funding_rate: float | None = 0.00012,
        mark_price: float | None = 50_000.0,
        index_price: float | None = 49_950.0,
        open_interest: float | None = 1_000_000.0,
        volume_24h: float | None = 250_000_000.0,
        next_funding_time: datetime | None = None,
        event_time: datetime | None = None,
        received_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingSnapshot:
        return FundingSnapshot(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            funding_rate=funding_rate,
            predicted_funding_rate=predicted_funding_rate,
            mark_price=mark_price,
            index_price=index_price,
            open_interest=open_interest,
            volume_24h=volume_24h,
            next_funding_time=next_funding_time or later_utc,
            event_time=event_time or now_utc,
            received_at=received_at or now_utc,
            metadata=metadata or {},
        )

    return _make_snapshot


@pytest.fixture
def make_statistics(now_utc: datetime) -> Callable[..., FundingStatistics]:
    def _make_statistics(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        current_rate: float = 0.0001,
        mean_rate: float = 0.00002,
        median_rate: float = 0.00002,
        std_rate: float = 0.00005,
        min_rate: float = -0.0001,
        max_rate: float = 0.0003,
        zscore: float | None = 1.6,
        percentile: float | None = 85.0,
        sample_size: int = 100,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        updated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingStatistics:
        return FundingStatistics(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            current_rate=current_rate,
            mean_rate=mean_rate,
            median_rate=median_rate,
            std_rate=std_rate,
            min_rate=min_rate,
            max_rate=max_rate,
            zscore=zscore,
            percentile=percentile,
            sample_size=sample_size,
            window_start=window_start or (now_utc - timedelta(hours=24)),
            window_end=window_end or now_utc,
            updated_at=updated_at or now_utc,
            metadata=metadata or {},
        )

    return _make_statistics


@pytest.fixture
def make_regime_state(now_utc: datetime) -> Callable[..., FundingRegimeState]:
    def _make_regime_state(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        regime: FundingRegime = FundingRegime.EXTREME_POSITIVE,
        bias: FundingBias = FundingBias.SQUEEZE_RISK_LONGS,
        current_rate: float = 0.00035,
        mean_rate: float | None = 0.00002,
        zscore: float | None = 3.0,
        percentile: float | None = 98.0,
        confidence: float = 0.95,
        changed: bool = True,
        previous_regime: FundingRegime | None = FundingRegime.NEUTRAL,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingRegimeState:
        return FundingRegimeState(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            regime=regime,
            bias=bias,
            current_rate=current_rate,
            mean_rate=mean_rate,
            zscore=zscore,
            percentile=percentile,
            confidence=confidence,
            changed=changed,
            previous_regime=previous_regime,
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_regime_state


@pytest.fixture
def make_pressure_state(now_utc: datetime) -> Callable[..., FundingPressureState]:
    def _make_pressure_state(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        direction: FundingPressureDirection = FundingPressureDirection.LONG,
        level: FundingPressureLevel = FundingPressureLevel.EXTREME,
        bias: FundingBias = FundingBias.SQUEEZE_RISK_LONGS,
        funding_rate: float = 0.00035,
        pressure_score: float = 0.92,
        oi_confirmation: bool = True,
        price_stall_confirmation: bool = True,
        squeeze_probability: float | None = 0.85,
        mean_reversion_probability: float | None = 0.75,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingPressureState:
        return FundingPressureState(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            direction=direction,
            level=level,
            bias=bias,
            funding_rate=funding_rate,
            pressure_score=pressure_score,
            oi_confirmation=oi_confirmation,
            price_stall_confirmation=price_stall_confirmation,
            squeeze_probability=squeeze_probability,
            mean_reversion_probability=mean_reversion_probability,
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_pressure_state


@pytest.fixture
def make_flip_event(now_utc: datetime) -> Callable[..., FundingFlipEvent]:
    def _make_flip_event(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        flip_type: FundingFlipType = FundingFlipType.NEGATIVE_TO_POSITIVE,
        previous_rate: float = -0.00015,
        current_rate: float = 0.00035,
        confidence: float = 0.8,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingFlipEvent:
        return FundingFlipEvent(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            flip_type=flip_type,
            previous_rate=previous_rate,
            current_rate=current_rate,
            confidence=confidence,
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_flip_event


@pytest.fixture
def make_extreme_event(now_utc: datetime) -> Callable[..., FundingExtremeEvent]:
    def _make_extreme_event(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        extreme_type: FundingExtremeType = FundingExtremeType.PERCENTILE_HIGH,
        regime: FundingRegime = FundingRegime.EXTREME_POSITIVE,
        funding_rate: float = 0.00035,
        zscore: float | None = 3.0,
        percentile: float | None = 98.0,
        severity: float = 0.9,
        is_reversal_risk: bool = True,
        is_squeeze_risk: bool = True,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingExtremeEvent:
        return FundingExtremeEvent(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            extreme_type=extreme_type,
            regime=regime,
            funding_rate=funding_rate,
            zscore=zscore,
            percentile=percentile,
            severity=severity,
            is_reversal_risk=is_reversal_risk,
            is_squeeze_risk=is_squeeze_risk,
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_extreme_event


@pytest.fixture
def make_divergence_event(now_utc: datetime) -> Callable[..., FundingDivergenceEvent]:
    def _make_divergence_event(
        *,
        symbol: str = TEST_SYMBOL,
        exchange: FundingDataSource | str = TEST_EXCHANGE_ENUM,
        market_type: str = TEST_MARKET_TYPE,
        timeframe: FundingTimeframe | str = TEST_TIMEFRAME,
        exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
        divergence_type: FundingDivergenceType = FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
        funding_rate: float = 0.00035,
        price_change_pct: float | None = -0.006,
        oi_change_pct: float | None = 0.10,
        cvd_change: float | None = -20_000.0,
        long_liquidations: float | None = 150_000.0,
        short_liquidations: float | None = None,
        confidence: float = 0.82,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingDivergenceEvent:
        return FundingDivergenceEvent(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            divergence_type=divergence_type,
            funding_rate=funding_rate,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            cvd_change=cvd_change,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            confidence=confidence,
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_divergence_event


@pytest.fixture
def make_market_context(now_utc: datetime) -> Callable[..., FundingMarketContext]:
    def _make_market_context(
        *,
        latest_open_interest: float | None = 1_100_000.0,
        previous_open_interest: float | None = 1_000_000.0,
        latest_price: float | None = 50_010.0,
        previous_price: float | None = 50_000.0,
        latest_cvd: float | None = 25_000.0,
        previous_cvd: float | None = 10_000.0,
        long_liquidations: float | None = 150_000.0,
        short_liquidations: float | None = 75_000.0,
        updated_at: datetime | None = None,
        liquidation_updated_at: datetime | None = None,
    ) -> FundingMarketContext:
        return FundingMarketContext(
            latest_open_interest=latest_open_interest,
            previous_open_interest=previous_open_interest,
            latest_price=latest_price,
            previous_price=previous_price,
            latest_cvd=latest_cvd,
            previous_cvd=previous_cvd,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            updated_at=updated_at or now_utc,
            liquidation_updated_at=liquidation_updated_at or now_utc,
        )

    return _make_market_context


# =============================================================================
# Real detector fixtures
# =============================================================================

@pytest.fixture
def regime_detector() -> FundingRegimeDetector:
    return FundingRegimeDetector()


@pytest.fixture
def pressure_analyzer() -> FundingPressureAnalyzer:
    return FundingPressureAnalyzer()


@pytest.fixture
def flip_detector() -> FundingFlipDetector:
    return FundingFlipDetector()


@pytest.fixture
def extremes_detector() -> FundingExtremesDetector:
    return FundingExtremesDetector()


@pytest.fixture
def divergence_detector() -> FundingDivergenceDetector:
    return FundingDivergenceDetector()


# =============================================================================
# Detector doubles for FundingAnalyzer orchestration tests
# =============================================================================

class StubRegimeDetector:
    def __init__(
        self,
        make_regime_state: Callable[..., FundingRegimeState],
        *,
        regime: FundingRegime = FundingRegime.EXTREME_POSITIVE,
        bias: FundingBias = FundingBias.SQUEEZE_RISK_LONGS,
        changed: bool = True,
        confidence: float = 0.95,
    ) -> None:
        self.make_regime_state = make_regime_state
        self.regime = regime
        self.bias = bias
        self.changed = changed
        self.confidence = confidence
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingRegimeState:
        self.calls.append(kwargs)

        snapshot: FundingSnapshot = kwargs["snapshot"]
        statistics: FundingStatistics = kwargs["statistics"]

        return self.make_regime_state(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            regime=self.regime,
            bias=self.bias,
            current_rate=snapshot.funding_rate,
            mean_rate=statistics.mean_rate,
            zscore=statistics.zscore,
            percentile=statistics.percentile,
            confidence=self.confidence,
            changed=self.changed,
            previous_regime=FundingRegime.NEUTRAL if self.changed else self.regime,
            event_time=snapshot.event_time,
            metadata={"stub": "regime"},
        )


class FaultyRegimeDetector:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("regime detector failed")
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.exc


class StubPressureAnalyzer:
    def __init__(
        self,
        make_pressure_state: Callable[..., FundingPressureState],
        *,
        level: FundingPressureLevel = FundingPressureLevel.EXTREME,
        direction: FundingPressureDirection = FundingPressureDirection.LONG,
        bias: FundingBias = FundingBias.SQUEEZE_RISK_LONGS,
        pressure_score: float = 0.92,
        squeeze_probability: float = 0.85,
        mean_reversion_probability: float = 0.75,
    ) -> None:
        self.make_pressure_state = make_pressure_state
        self.level = level
        self.direction = direction
        self.bias = bias
        self.pressure_score = pressure_score
        self.squeeze_probability = squeeze_probability
        self.mean_reversion_probability = mean_reversion_probability
        self.calls: list[dict[str, Any]] = []

    def analyze(self, **kwargs: Any) -> FundingPressureState:
        self.calls.append(kwargs)

        snapshot: FundingSnapshot = kwargs["snapshot"]

        return self.make_pressure_state(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            direction=self.direction,
            level=self.level,
            bias=self.bias,
            funding_rate=snapshot.funding_rate,
            pressure_score=self.pressure_score,
            oi_confirmation=True,
            price_stall_confirmation=True,
            squeeze_probability=self.squeeze_probability,
            mean_reversion_probability=self.mean_reversion_probability,
            event_time=snapshot.event_time,
            metadata={"stub": "pressure"},
        )

    def is_high_pressure(self, state: FundingPressureState) -> bool:
        return state.level in {
            FundingPressureLevel.HIGH,
            FundingPressureLevel.EXTREME,
        }

    def is_squeeze_risk(
        self,
        state: FundingPressureState,
        threshold: float = 0.65,
    ) -> bool:
        return (state.squeeze_probability or 0.0) >= threshold

    def is_long_crowded(self, state: FundingPressureState) -> bool:
        return state.direction == FundingPressureDirection.LONG and self.is_high_pressure(state)

    def is_short_crowded(self, state: FundingPressureState) -> bool:
        return state.direction == FundingPressureDirection.SHORT and self.is_high_pressure(state)

    def build_summary(self, state: FundingPressureState) -> str:
        return (
            f"Stub pressure: {state.exchange.value}:{state.market_type}:"
            f"{state.symbol}:{state.timeframe.value} "
            f"level={state.level.value} score={state.pressure_score:.4f}"
        )


class StubFlipDetector:
    def __init__(
        self,
        make_flip_event: Callable[..., FundingFlipEvent],
        *,
        return_none: bool = False,
        flip_type: FundingFlipType = FundingFlipType.NEGATIVE_TO_POSITIVE,
        confidence: float = 0.8,
    ) -> None:
        self.make_flip_event = make_flip_event
        self.return_none = return_none
        self.flip_type = flip_type
        self.confidence = confidence
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingFlipEvent | None:
        self.calls.append(kwargs)

        if self.return_none:
            return None

        current_snapshot: FundingSnapshot = kwargs["current_snapshot"]
        previous_snapshot: FundingSnapshot | None = kwargs.get("previous_snapshot")

        return self.make_flip_event(
            symbol=current_snapshot.symbol,
            exchange=current_snapshot.exchange,
            market_type=current_snapshot.market_type,
            timeframe=current_snapshot.timeframe,
            exchange_symbol=current_snapshot.exchange_symbol,
            flip_type=self.flip_type,
            previous_rate=(
                previous_snapshot.funding_rate
                if previous_snapshot is not None
                else -abs(current_snapshot.funding_rate)
            ),
            current_rate=current_snapshot.funding_rate,
            confidence=self.confidence,
            event_time=current_snapshot.event_time,
            metadata={"stub": "flip"},
        )

    def build_summary(self, event: FundingFlipEvent) -> str:
        return (
            f"Stub flip: {event.exchange.value}:{event.market_type}:"
            f"{event.symbol}:{event.timeframe.value} type={event.flip_type.value}"
        )


class StubExtremesDetector:
    def __init__(
        self,
        make_extreme_event: Callable[..., FundingExtremeEvent],
        *,
        return_none: bool = False,
        extreme_type: FundingExtremeType = FundingExtremeType.PERCENTILE_HIGH,
        severity: float = 0.9,
    ) -> None:
        self.make_extreme_event = make_extreme_event
        self.return_none = return_none
        self.extreme_type = extreme_type
        self.severity = severity
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingExtremeEvent | None:
        self.calls.append(kwargs)

        if self.return_none:
            return None

        snapshot: FundingSnapshot = kwargs["snapshot"]
        statistics: FundingStatistics = kwargs["statistics"]
        regime_state: FundingRegimeState | None = kwargs.get("regime_state")

        return self.make_extreme_event(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            extreme_type=self.extreme_type,
            regime=regime_state.regime if regime_state is not None else FundingRegime.EXTREME_POSITIVE,
            funding_rate=snapshot.funding_rate,
            zscore=statistics.zscore,
            percentile=statistics.percentile,
            severity=self.severity,
            is_reversal_risk=True,
            is_squeeze_risk=True,
            event_time=snapshot.event_time,
            metadata={"stub": "extreme"},
        )

    def build_summary(self, event: FundingExtremeEvent) -> str:
        return (
            f"Stub extreme: {event.exchange.value}:{event.market_type}:"
            f"{event.symbol}:{event.timeframe.value} type={event.extreme_type.value}"
        )


class StubDivergenceDetector:
    def __init__(
        self,
        make_divergence_event: Callable[..., FundingDivergenceEvent],
        *,
        return_none: bool = False,
        divergence_type: FundingDivergenceType = FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
        confidence: float = 0.82,
    ) -> None:
        self.make_divergence_event = make_divergence_event
        self.return_none = return_none
        self.divergence_type = divergence_type
        self.confidence = confidence
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingDivergenceEvent | None:
        self.calls.append(kwargs)

        if self.return_none:
            return None

        snapshot: FundingSnapshot = kwargs["snapshot"]

        return self.make_divergence_event(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            divergence_type=self.divergence_type,
            funding_rate=snapshot.funding_rate,
            price_change_pct=kwargs.get("price_change_pct"),
            oi_change_pct=kwargs.get("oi_change_pct"),
            cvd_change=kwargs.get("cvd_change"),
            long_liquidations=kwargs.get("long_liquidations"),
            short_liquidations=kwargs.get("short_liquidations"),
            confidence=self.confidence,
            event_time=snapshot.event_time,
            metadata={"stub": "divergence"},
        )

    def is_bullish_divergence(self, event: FundingDivergenceEvent) -> bool:
        return event.divergence_type in {
            FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
            FundingDivergenceType.OI_UP_FUNDING_DOWN,
            FundingDivergenceType.CVD_UP_FUNDING_DOWN,
            FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
        }

    def is_bearish_divergence(self, event: FundingDivergenceEvent) -> bool:
        return event.divergence_type in {
            FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
            FundingDivergenceType.OI_UP_FUNDING_UP_PRICE_STALLED,
            FundingDivergenceType.CVD_DOWN_FUNDING_UP,
            FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
        }

    def build_summary(self, event: FundingDivergenceEvent) -> str:
        return (
            f"Stub divergence: {event.exchange.value}:{event.market_type}:"
            f"{event.symbol}:{event.timeframe.value} type={event.divergence_type.value}"
        )


@pytest.fixture
def stub_regime_detector(
    make_regime_state: Callable[..., FundingRegimeState],
) -> StubRegimeDetector:
    return StubRegimeDetector(make_regime_state)


@pytest.fixture
def faulty_regime_detector() -> FaultyRegimeDetector:
    return FaultyRegimeDetector()


@pytest.fixture
def stub_pressure_analyzer(
    make_pressure_state: Callable[..., FundingPressureState],
) -> StubPressureAnalyzer:
    return StubPressureAnalyzer(make_pressure_state)


@pytest.fixture
def stub_flip_detector(
    make_flip_event: Callable[..., FundingFlipEvent],
) -> StubFlipDetector:
    return StubFlipDetector(make_flip_event)


@pytest.fixture
def stub_extremes_detector(
    make_extreme_event: Callable[..., FundingExtremeEvent],
) -> StubExtremesDetector:
    return StubExtremesDetector(make_extreme_event)


@pytest.fixture
def stub_divergence_detector(
    make_divergence_event: Callable[..., FundingDivergenceEvent],
) -> StubDivergenceDetector:
    return StubDivergenceDetector(make_divergence_event)


# =============================================================================
# Analyzer factories
# =============================================================================

@pytest.fixture
def make_funding_analyzer(
    fake_event_bus: FakeEventBus,
    fake_scheduler: FakeScheduler,
    funding_analyzer_config: FundingAnalyzerConfig,
    stub_regime_detector: StubRegimeDetector,
    stub_pressure_analyzer: StubPressureAnalyzer,
    stub_flip_detector: StubFlipDetector,
    stub_extremes_detector: StubExtremesDetector,
    stub_divergence_detector: StubDivergenceDetector,
) -> Callable[..., FundingAnalyzer]:
    def _make_funding_analyzer(
        *,
        event_bus: Any | None = None,
        scheduler: Any | None = None,
        config: FundingAnalyzerConfig | None = None,
        regime_detector: Any | None = None,
        pressure_analyzer: Any | None = None,
        flip_detector: Any | None = None,
        extremes_detector: Any | None = None,
        divergence_detector: Any | None = None,
        parquet_storage: Any | None = None,
    ) -> FundingAnalyzer:
        return FundingAnalyzer(
            event_bus=event_bus if event_bus is not None else fake_event_bus,  # type: ignore[arg-type]
            scheduler=scheduler if scheduler is not None else fake_scheduler,  # type: ignore[arg-type]
            config=config if config is not None else funding_analyzer_config,
            regime_detector=regime_detector if regime_detector is not None else stub_regime_detector,  # type: ignore[arg-type]
            pressure_analyzer=pressure_analyzer if pressure_analyzer is not None else stub_pressure_analyzer,  # type: ignore[arg-type]
            flip_detector=flip_detector if flip_detector is not None else stub_flip_detector,  # type: ignore[arg-type]
            extremes_detector=extremes_detector if extremes_detector is not None else stub_extremes_detector,  # type: ignore[arg-type]
            divergence_detector=divergence_detector if divergence_detector is not None else stub_divergence_detector,  # type: ignore[arg-type]
            parquet_storage=parquet_storage,
        )

    return _make_funding_analyzer


@pytest.fixture
def funding_analyzer(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
) -> FundingAnalyzer:
    return make_funding_analyzer()


# =============================================================================
# Convenience helpers for tests
# =============================================================================

@pytest.fixture
def funding_key() -> tuple[str, str, str, str]:
    return TEST_KEY


@pytest.fixture
def make_key() -> Callable[..., tuple[str, str, str, str]]:
    def _make_key(
        *,
        exchange: str | FundingDataSource = TEST_EXCHANGE,
        market_type: str = TEST_MARKET_TYPE,
        symbol: str = TEST_SYMBOL,
        timeframe: str | FundingTimeframe = TEST_TIMEFRAME,
    ) -> tuple[str, str, str, str]:
        return make_funding_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    return _make_key


@pytest.fixture
def run_funding_update(
    make_event: Callable[..., FakeEvent],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> Callable[..., Any]:
    async def _run_funding_update(
        analyzer: FundingAnalyzer,
        *,
        topic: str | None = None,
        correlation_id: str | None = "corr-hard-funding",
        **payload_overrides: Any,
    ) -> None:
        event = make_event(
            make_funding_payload(**payload_overrides),
            topic=topic or analyzer.config.funding_event_name,
            correlation_id=correlation_id,
        )
        await analyzer.on_funding(event)

    return _run_funding_update