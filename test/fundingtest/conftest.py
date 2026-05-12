# tests/analytics/funding/conftest.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from analytics.funding.funding_analyzer import (
    FundingAnalyzer,
    FundingAnalyzerConfig,
    FundingMarketContext,
)
from analytics.funding.funding_divergence import FundingDivergenceDetector
from analytics.funding.funding_extremes import FundingExtremesDetector
from analytics.funding.funding_flip_detector import FundingFlipDetector
from analytics.funding.funding_pressure import FundingPressureAnalyzer
from analytics.funding.funding_regime_detector import FundingRegimeDetector
from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingEventType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingAnalyticsEvent,
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
    FundingSignal,
    FundingSnapshot,
    FundingStatistics,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def earlier_utc(now_utc: datetime) -> datetime:
    return now_utc - timedelta(hours=1)


# ---------------------------------------------------------------------------
# Lightweight Event / EventBus / Scheduler fakes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FakePublishedEvent:
    topic: str
    payload: dict[str, Any]
    kwargs: dict[str, Any]


class FakeEvent:
    """
    Мінімальний fake Event для handler-тестів.

    FundingAnalyzer використовує з event щонайменше:
    - payload
    - correlation_id

    Для більшості unit/integration-style тестів цього достатньо.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        topic: str = "test.event",
        correlation_id: str | None = "test-correlation-id",
        source: str = "test",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.topic = topic
        self.payload = payload
        self.correlation_id = correlation_id
        self.source = source
        self.metadata = metadata or {}


class FakeEventBus:
    """
    Fake EventBus для тестування FundingAnalyzer без реального core.EventBus.

    Підтримує API, який використовує FundingAnalyzer:
    - subscribe(topic, handler, name=...)
    - unsubscribe(subscription)
    - async emit(topic, payload, **kwargs)
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
    Fake Scheduler для перевірки register/unregister cleanup job.

    FundingAnalyzer використовує:
    - get_job_by_name(name)
    - add_interval_job(...)
    - disable_job(job_id)
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.disabled_job_ids: list[str] = []
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

        self.jobs[job_id] = FakeScheduledJob(
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
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def make_event() -> Callable[..., FakeEvent]:
    def _make_event(
        payload: dict[str, Any],
        *,
        topic: str = "test.event",
        correlation_id: str | None = "test-correlation-id",
        source: str = "test",
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


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


@pytest.fixture
def make_snapshot(now_utc: datetime) -> Callable[..., FundingSnapshot]:
    def _make_snapshot(
        *,
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        funding_rate: float = 0.0001,
        predicted_funding_rate: float | None = None,
        mark_price: float | None = 50_000.0,
        index_price: float | None = 49_950.0,
        open_interest: float | None = 1_000_000.0,
        volume_24h: float | None = 100_000_000.0,
        next_funding_time: datetime | None = None,
        event_time: datetime | None = None,
        received_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingSnapshot:
        return FundingSnapshot(
            symbol=symbol,
            exchange=exchange,
            funding_rate=funding_rate,
            predicted_funding_rate=predicted_funding_rate,
            mark_price=mark_price,
            index_price=index_price,
            open_interest=open_interest,
            volume_24h=volume_24h,
            next_funding_time=next_funding_time,
            event_time=event_time or now_utc,
            received_at=received_at or now_utc,
            metadata=metadata or {},
        )

    return _make_snapshot


@pytest.fixture
def make_statistics(now_utc: datetime) -> Callable[..., FundingStatistics]:
    def _make_statistics(
        *,
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
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
    ) -> FundingStatistics:
        return FundingStatistics(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
        )

    return _make_statistics


@pytest.fixture
def make_regime_state(now_utc: datetime) -> Callable[..., FundingRegimeState]:
    def _make_regime_state(
        *,
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        regime: FundingRegime = FundingRegime.POSITIVE,
        bias: FundingBias = FundingBias.LONG_BIAS,
        current_rate: float = 0.0001,
        mean_rate: float | None = 0.00002,
        zscore: float | None = 1.6,
        percentile: float | None = 85.0,
        confidence: float = 0.75,
        changed: bool = False,
        previous_regime: FundingRegime | None = None,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingRegimeState:
        return FundingRegimeState(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        direction: FundingPressureDirection = FundingPressureDirection.LONG,
        level: FundingPressureLevel = FundingPressureLevel.HIGH,
        bias: FundingBias = FundingBias.OVERCROWDED_LONGS,
        funding_rate: float = 0.0001,
        pressure_score: float = 0.75,
        oi_confirmation: bool = True,
        price_stall_confirmation: bool = True,
        squeeze_probability: float | None = 0.65,
        mean_reversion_probability: float | None = 0.55,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingPressureState:
        return FundingPressureState(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        flip_type: FundingFlipType = FundingFlipType.NEGATIVE_TO_POSITIVE,
        previous_rate: float = -0.00008,
        current_rate: float = 0.0001,
        confidence: float = 0.7,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingFlipEvent:
        return FundingFlipEvent(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        extreme_type: FundingExtremeType = FundingExtremeType.PERCENTILE_HIGH,
        regime: FundingRegime = FundingRegime.EXTREME_POSITIVE,
        funding_rate: float = 0.00035,
        zscore: float | None = 2.7,
        percentile: float | None = 98.0,
        severity: float = 0.85,
        is_reversal_risk: bool = True,
        is_squeeze_risk: bool = True,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingExtremeEvent:
        return FundingExtremeEvent(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        divergence_type: FundingDivergenceType = FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
        funding_rate: float = 0.00012,
        price_change_pct: float | None = -0.004,
        oi_change_pct: float | None = 0.01,
        cvd_change: float | None = -10_000.0,
        long_liquidations: float | None = None,
        short_liquidations: float | None = None,
        confidence: float = 0.65,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingDivergenceEvent:
        return FundingDivergenceEvent(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
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
def make_signal(now_utc: datetime) -> Callable[..., FundingSignal]:
    def _make_signal(
        *,
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        signal_type: FundingSignalType = FundingSignalType.PRESSURE_BUILDUP,
        bias: FundingBias = FundingBias.OVERCROWDED_LONGS,
        regime: FundingRegime = FundingRegime.POSITIVE,
        score: float = -0.75,
        confidence: float = 0.75,
        description: str = "Funding pressure buildup",
        supporting_factors: list[str] | None = None,
        tags: list[str] | None = None,
        event_time: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FundingSignal:
        return FundingSignal(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            signal_type=signal_type,
            bias=bias,
            regime=regime,
            score=score,
            confidence=confidence,
            description=description,
            supporting_factors=supporting_factors or ["pressure_score=0.75"],
            tags=tags or ["funding", "pressure"],
            event_time=event_time or now_utc,
            metadata=metadata or {},
        )

    return _make_signal


@pytest.fixture
def make_analytics_event(now_utc: datetime) -> Callable[..., FundingAnalyticsEvent]:
    def _make_analytics_event(
        *,
        event_type: FundingEventType = FundingEventType.SNAPSHOT,
        symbol: str = "BTCUSDT",
        exchange: FundingDataSource = FundingDataSource.BINANCE,
        timeframe: FundingTimeframe = FundingTimeframe.H1,
        payload: dict[str, Any] | None = None,
        event_time: datetime | None = None,
        source: str = "analytics.funding.test",
        metadata: dict[str, Any] | None = None,
    ) -> FundingAnalyticsEvent:
        return FundingAnalyticsEvent(
            event_type=event_type,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            payload=payload or {},
            event_time=event_time or now_utc,
            source=source,
            metadata=metadata or {},
        )

    return _make_analytics_event


@pytest.fixture
def make_market_context(now_utc: datetime) -> Callable[..., FundingMarketContext]:
    def _make_market_context(
        *,
        latest_open_interest: float | None = 1_050_000.0,
        previous_open_interest: float | None = 1_000_000.0,
        latest_price: float | None = 50_000.0,
        previous_price: float | None = 49_980.0,
        latest_cvd: float | None = 25_000.0,
        previous_cvd: float | None = 10_000.0,
        long_liquidations: float | None = 100_000.0,
        short_liquidations: float | None = 50_000.0,
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


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


@pytest.fixture
def make_funding_payload(now_utc: datetime) -> Callable[..., dict[str, Any]]:
    def _make_funding_payload(
        *,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        funding_rate: float | str = 0.0001,
        predicted_funding_rate: float | str | None = None,
        mark_price: float | str | None = 50_000.0,
        index_price: float | str | None = 49_950.0,
        open_interest: float | str | None = 1_000_000.0,
        volume_24h: float | str | None = 100_000_000.0,
        next_funding_time: datetime | str | None = None,
        event_time: datetime | str | None = None,
        received_at: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "exchange": exchange,
            "funding_rate": funding_rate,
            "predicted_funding_rate": predicted_funding_rate,
            "mark_price": mark_price,
            "index_price": index_price,
            "open_interest": open_interest,
            "volume_24h": volume_24h,
            "next_funding_time": next_funding_time,
            "event_time": event_time or now_utc,
            "received_at": received_at or now_utc,
            "metadata": metadata or {},
        }

    return _make_funding_payload


@pytest.fixture
def make_context_payload() -> Callable[..., dict[str, Any]]:
    def _make_context_payload(
        *,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        **fields: Any,
    ) -> dict[str, Any]:
        payload = {
            "symbol": symbol,
            "exchange": exchange,
        }
        payload.update(fields)
        return payload

    return _make_context_payload


# ---------------------------------------------------------------------------
# Real detector fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stub components for FundingAnalyzer orchestration tests
# ---------------------------------------------------------------------------


class StubRegimeDetector:
    def __init__(self, state: FundingRegimeState) -> None:
        self.state = state
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingRegimeState:
        self.calls.append(kwargs)
        snapshot: FundingSnapshot = kwargs["snapshot"]

        # Повертаємо state, синхронізований із поточним snapshot,
        # щоб downstream signal/event assertions були стабільними.
        self.state.symbol = snapshot.symbol
        self.state.exchange = snapshot.exchange
        self.state.current_rate = snapshot.funding_rate
        self.state.event_time = snapshot.event_time
        return self.state


class StubPressureAnalyzer:
    def __init__(self, state: FundingPressureState) -> None:
        self.state = state
        self.calls: list[dict[str, Any]] = []
        self.high_pressure = True

    def analyze(self, **kwargs: Any) -> FundingPressureState:
        self.calls.append(kwargs)
        snapshot: FundingSnapshot = kwargs["snapshot"]

        self.state.symbol = snapshot.symbol
        self.state.exchange = snapshot.exchange
        self.state.funding_rate = snapshot.funding_rate
        self.state.event_time = snapshot.event_time
        return self.state

    def is_high_pressure(self, pressure_state: FundingPressureState) -> bool:
        return self.high_pressure

    def build_summary(self, pressure_state: FundingPressureState) -> str:
        return (
            f"Funding pressure {pressure_state.level.value} "
            f"{pressure_state.direction.value}"
        )


class StubFlipDetector:
    def __init__(self, event: FundingFlipEvent | None) -> None:
        self.event = event
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingFlipEvent | None:
        self.calls.append(kwargs)
        current_snapshot: FundingSnapshot = kwargs["current_snapshot"]

        if self.event is not None:
            self.event.symbol = current_snapshot.symbol
            self.event.exchange = current_snapshot.exchange
            self.event.current_rate = current_snapshot.funding_rate
            self.event.event_time = current_snapshot.event_time

        return self.event

    def build_summary(self, flip_event: FundingFlipEvent) -> str:
        return f"Funding flip {flip_event.flip_type.value}"


class StubExtremesDetector:
    def __init__(self, event: FundingExtremeEvent | None) -> None:
        self.event = event
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> FundingExtremeEvent | None:
        self.calls.append(kwargs)
        snapshot: FundingSnapshot = kwargs["snapshot"]

        if self.event is not None:
            self.event.symbol = snapshot.symbol
            self.event.exchange = snapshot.exchange
            self.event.funding_rate = snapshot.funding_rate
            self.event.event_time = snapshot.event_time

        return self.event

    def build_summary(self, extreme_event: FundingExtremeEvent) -> str:
        return f"Funding extreme {extreme_event.extreme_type.value}"


class StubDivergenceDetector:
    def __init__(self, event: FundingDivergenceEvent | None) -> None:
        self.event = event
        self.calls: list[dict[str, Any]] = []
        self.bullish = False
        self.bearish = True

    def detect(self, **kwargs: Any) -> FundingDivergenceEvent | None:
        self.calls.append(kwargs)
        snapshot: FundingSnapshot = kwargs["snapshot"]

        if self.event is not None:
            self.event.symbol = snapshot.symbol
            self.event.exchange = snapshot.exchange
            self.event.funding_rate = snapshot.funding_rate
            self.event.event_time = snapshot.event_time

        return self.event

    def build_summary(self, divergence_event: FundingDivergenceEvent) -> str:
        return f"Funding divergence {divergence_event.divergence_type.value}"

    def is_bullish_divergence(self, divergence_event: FundingDivergenceEvent) -> bool:
        return self.bullish

    def is_bearish_divergence(self, divergence_event: FundingDivergenceEvent) -> bool:
        return self.bearish


@pytest.fixture
def stub_regime_detector(
    make_regime_state: Callable[..., FundingRegimeState],
) -> StubRegimeDetector:
    return StubRegimeDetector(make_regime_state(changed=True))


@pytest.fixture
def stub_pressure_analyzer(
    make_pressure_state: Callable[..., FundingPressureState],
) -> StubPressureAnalyzer:
    return StubPressureAnalyzer(make_pressure_state())


@pytest.fixture
def stub_flip_detector(
    make_flip_event: Callable[..., FundingFlipEvent],
) -> StubFlipDetector:
    return StubFlipDetector(make_flip_event())


@pytest.fixture
def stub_extremes_detector(
    make_extreme_event: Callable[..., FundingExtremeEvent],
) -> StubExtremesDetector:
    return StubExtremesDetector(make_extreme_event())


@pytest.fixture
def stub_divergence_detector(
    make_divergence_event: Callable[..., FundingDivergenceEvent],
) -> StubDivergenceDetector:
    return StubDivergenceDetector(make_divergence_event())


# ---------------------------------------------------------------------------
# FundingAnalyzer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer_config() -> FundingAnalyzerConfig:
    return FundingAnalyzerConfig(
        history_size=10,
        publish_updated_event=True,
        publish_regime_event_on_every_update=False,
        publish_pressure_event_on_every_update=False,
        publish_signal_event=True,
        state_lock_timeout_sec=0.05,
        enable_cleanup_job=True,
        cleanup_interval_sec=60.0,
        cleanup_timeout_sec=5.0,
        stale_context_ttl_sec=60.0,
        stale_liquidation_ttl_sec=30.0,
    )


@pytest.fixture
def analyzer_config_no_cleanup() -> FundingAnalyzerConfig:
    return FundingAnalyzerConfig(
        history_size=10,
        publish_updated_event=True,
        publish_regime_event_on_every_update=False,
        publish_pressure_event_on_every_update=False,
        publish_signal_event=True,
        state_lock_timeout_sec=0.05,
        enable_cleanup_job=False,
        stale_context_ttl_sec=60.0,
        stale_liquidation_ttl_sec=30.0,
    )


@pytest.fixture
def funding_analyzer(
    fake_event_bus: FakeEventBus,
    fake_scheduler: FakeScheduler,
    analyzer_config: FundingAnalyzerConfig,
    stub_regime_detector: StubRegimeDetector,
    stub_pressure_analyzer: StubPressureAnalyzer,
    stub_flip_detector: StubFlipDetector,
    stub_extremes_detector: StubExtremesDetector,
    stub_divergence_detector: StubDivergenceDetector,
) -> FundingAnalyzer:
    return FundingAnalyzer(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
        config=analyzer_config,
        regime_detector=stub_regime_detector,  # type: ignore[arg-type]
        pressure_analyzer=stub_pressure_analyzer,  # type: ignore[arg-type]
        flip_detector=stub_flip_detector,  # type: ignore[arg-type]
        extremes_detector=stub_extremes_detector,  # type: ignore[arg-type]
        divergence_detector=stub_divergence_detector,  # type: ignore[arg-type]
    )


@pytest.fixture
def funding_analyzer_no_cleanup(
    fake_event_bus: FakeEventBus,
    analyzer_config_no_cleanup: FundingAnalyzerConfig,
    stub_regime_detector: StubRegimeDetector,
    stub_pressure_analyzer: StubPressureAnalyzer,
    stub_flip_detector: StubFlipDetector,
    stub_extremes_detector: StubExtremesDetector,
    stub_divergence_detector: StubDivergenceDetector,
) -> FundingAnalyzer:
    return FundingAnalyzer(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=None,
        config=analyzer_config_no_cleanup,
        regime_detector=stub_regime_detector,  # type: ignore[arg-type]
        pressure_analyzer=stub_pressure_analyzer,  # type: ignore[arg-type]
        flip_detector=stub_flip_detector,  # type: ignore[arg-type]
        extremes_detector=stub_extremes_detector,  # type: ignore[arg-type]
        divergence_detector=stub_divergence_detector,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def assert_published_topic() -> Callable[[FakeEventBus, str, int], None]:
    def _assert_published_topic(
        event_bus: FakeEventBus,
        topic: str,
        expected_count: int = 1,
    ) -> None:
        actual_count = event_bus.topics().count(topic)
        assert actual_count == expected_count, (
            f"Expected topic={topic!r} to be published {expected_count} times, "
            f"got {actual_count}. Published topics: {event_bus.topics()}"
        )

    return _assert_published_topic


@pytest.fixture
def assert_not_published_topic() -> Callable[[FakeEventBus, str], None]:
    def _assert_not_published_topic(
        event_bus: FakeEventBus,
        topic: str,
    ) -> None:
        assert topic not in event_bus.topics(), (
            f"Expected topic={topic!r} not to be published. "
            f"Published topics: {event_bus.topics()}"
        )

    return _assert_not_published_topic