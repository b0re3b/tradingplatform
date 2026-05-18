# tests/analytics/open_interest/test_oi_analyzer_event_flow.py

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from core.event_bus import Event, EventPriority

from analytics.open_interest.config import (
    OIAnalyzerConfig,
    OICooldowns,
    OIMaintenanceConfig,
    OIThresholds,
    OIWindows,
)
from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
    OIEventType,
    OIMarketEventType,
    OIRegime,
    OISignalStrength,
)
from analytics.open_interest.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OIAnomalyResult,
    OIDivergenceResult,
    OIKey,
    OIRegimeResult,
    make_oi_key,
)
from analytics.open_interest.oi_analyzer import OIAnalyzer


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXCHANGE_SYMBOL = "BTCUSDT"
DEFAULT_CONTEXT_MARKET_TYPE = "usdm_futures"
DEFAULT_CONTEXT_TIMEFRAME = "1m"


def now_ts() -> float:
    return time.time()


def key(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
) -> OIKey:
    return make_oi_key(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
    )


def assert_default_scope_payload(payload: dict[str, Any]) -> None:
    assert payload["exchange"] == DEFAULT_EXCHANGE
    assert payload["market_type"] == DEFAULT_CONTEXT_MARKET_TYPE
    assert payload["symbol"] == DEFAULT_SYMBOL
    assert payload["timeframe"] == DEFAULT_CONTEXT_TIMEFRAME
    assert payload["scope"] == {
        "exchange": DEFAULT_EXCHANGE,
        "market_type": DEFAULT_CONTEXT_MARKET_TYPE,
        "symbol": DEFAULT_SYMBOL,
        "timeframe": DEFAULT_CONTEXT_TIMEFRAME,
    }
    assert payload["scope_key"] == (
        f"{DEFAULT_EXCHANGE}:{DEFAULT_CONTEXT_MARKET_TYPE}:"
        f"{DEFAULT_SYMBOL}:{DEFAULT_CONTEXT_TIMEFRAME}"
    )


# ---------------------------------------------------------------------------
# Fake core infrastructure
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EmittedEvent:
    topic: str
    payload: dict[str, Any]
    priority: EventPriority
    source: str | None
    correlation_id: str | None
    headers: dict[str, Any] | None = None


class FakeEventBus:
    """
    Minimal EventBus test double.

    We are not testing the real EventBus worker loop here.
    We are testing whether OIAnalyzer correctly uses the core contract:
    - subscribe(topic, handler, name=...)
    - unsubscribe(subscription)
    - emit(topic, payload, priority=..., source=..., correlation_id=..., headers=...)
    """

    def __init__(self, *, fail_on_topics: set[str] | None = None) -> None:
        self.subscriptions: list[Any] = []
        self.unsubscribed: list[Any] = []
        self.emitted: list[EmittedEvent] = []
        self.fail_on_topics = set(fail_on_topics or set())

    def subscribe(self, pattern, handler, *, name=None):
        subscription = SimpleNamespace(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous"),
            enabled=True,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription) -> None:
        self.unsubscribed.append(subscription)
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        if topic in self.fail_on_topics:
            raise RuntimeError(f"forced emit failure: {topic}")

        self.emitted.append(
            EmittedEvent(
                topic=topic,
                payload=payload,
                priority=priority,
                source=source,
                correlation_id=correlation_id,
                headers=headers,
            )
        )
        return True

    def topics(self) -> list[str]:
        return [item.topic for item in self.emitted]

    def by_topic(self, topic: str) -> list[EmittedEvent]:
        return [item for item in self.emitted if item.topic == topic]

    def clear(self) -> None:
        self.emitted.clear()


class FakeSchedulerJob:
    def __init__(self, job_id: str, name: str) -> None:
        self.job_id = job_id
        self.name = name


class FakeScheduler:
    """
    Minimal Scheduler test double.

    Current OIAnalyzer registers jobs via add_interval_job() and removes them
    via remove_job(). It should not create uncontrolled async loops.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.add_calls: list[dict[str, Any]] = []
        self.removed: list[str] = []

    def add_interval_job(
        self,
        name: str,
        func,
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
        job_id = f"job-{len(self.jobs) + 1}-{name}"

        record = {
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
        self.jobs[job_id] = record
        self.add_calls.append(record)
        return job_id

    def get_job_by_name(self, name: str):
        for job in self.jobs.values():
            if job["name"] == name:
                return FakeSchedulerJob(
                    job_id=job["job_id"],
                    name=job["name"],
                )
        return None

    def remove_job(self, job_id: str) -> bool:
        self.removed.append(job_id)
        self.jobs.pop(job_id, None)
        return True


# ---------------------------------------------------------------------------
# Detector stubs
# ---------------------------------------------------------------------------

class StubRegimeDetector:
    def __init__(self, regime: OIRegime = OIRegime.NEUTRAL) -> None:
        self.regime = regime
        self.calls: list[Any] = []

    def detect(self, features) -> OIRegimeResult:
        self.calls.append(features)
        return OIRegimeResult(
            regime=self.regime,
            confidence=0.91 if self.regime is not OIRegime.NEUTRAL else 0.25,
            reasons=[f"stub_{self.regime.value.lower()}"],
            score=0.88 if self.regime is not OIRegime.NEUTRAL else 0.0,
        )


class StubAnomalyDetector:
    def __init__(
        self,
        *,
        detected: bool = False,
        anomaly_type: OIAnomalyType = OIAnomalyType.NONE,
    ) -> None:
        self.detected = detected
        self.anomaly_type = anomaly_type
        self.calls: list[Any] = []

    def detect(self, features) -> OIAnomalyResult:
        self.calls.append(features)
        return OIAnomalyResult(
            detected=self.detected,
            anomaly_type=self.anomaly_type,
            strength=OISignalStrength.HIGH if self.detected else OISignalStrength.LOW,
            confidence=0.83 if self.detected else 0.0,
            reasons=[f"stub_{self.anomaly_type.value.lower()}"] if self.detected else [],
            score=0.81 if self.detected else 0.0,
        )


def detected_divergence(
    divergence_type: OIDivergenceType = OIDivergenceType.PRICE_UP_OI_DOWN,
) -> OIDivergenceResult:
    return OIDivergenceResult(
        detected=True,
        divergence_type=divergence_type,
        confidence=0.82,
        reasons=[f"stub_{divergence_type.value.lower()}"],
        window_size=5,
        score=0.79,
    )


def not_detected_divergence() -> OIDivergenceResult:
    return OIDivergenceResult(
        detected=False,
        divergence_type=OIDivergenceType.NONE,
        confidence=0.0,
        reasons=["stub_no_divergence"],
        window_size=5,
        score=0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> OIAnalyzerConfig:
    return OIAnalyzerConfig(
        enabled=True,
        source_name="test_oi_analyzer",
        default_exchange=DEFAULT_EXCHANGE,
        default_market_type=DEFAULT_CONTEXT_MARKET_TYPE,
        default_timeframe=DEFAULT_CONTEXT_TIMEFRAME,
        allowed_market_types={"usdm_futures", "coinm_futures", "linear", "swap"},
        emit_updates=True,
        emit_regime_changes=True,
        emit_divergences=True,
        emit_anomalies=True,
        emit_squeeze_events=True,
        emit_capitulation_events=True,
        emit_metrics=True,
        require_price_context=False,
        require_volume_confirmation=True,
        normalize_symbol=True,
        store_full_analysis=True,
        stale_context_after_sec=3_600.0,
        stale_state_cleanup_after_sec=7_200.0,
        thresholds=OIThresholds(
            min_oi_change_pct=0.25,
            min_price_change_pct=0.20,
            volume_confirmation_ratio=1.15,
            aggressive_flow_confirmation=0.10,
            funding_extreme_positive=0.010,
            funding_extreme_negative=-0.010,
            divergence_min_price_move_pct=0.35,
            divergence_max_oi_response_pct=0.10,
            divergence_min_confidence=0.55,
            anomaly_zscore_threshold=2.5,
            extreme_anomaly_zscore_threshold=3.5,
            overheated_zscore_threshold=2.8,
            capitulation_price_move_pct=1.25,
            capitulation_oi_drop_pct=1.00,
            deleveraging_oi_drop_pct=1.50,
            squeeze_funding_abs_threshold=0.015,
            squeeze_oi_build_pct=0.75,
            pressure_score_trend_threshold=0.35,
            pressure_score_exhaustion_threshold=0.75,
        ),
        windows=OIWindows(
            history_size=20,
            fast_window=3,
            slow_window=6,
            zscore_window=6,
            divergence_window=5,
            pressure_window=3,
            volume_window=4,
        ),
        cooldowns=OICooldowns(
            regime_change_cooldown_sec=30.0,
            divergence_event_cooldown_sec=30.0,
            anomaly_event_cooldown_sec=30.0,
            squeeze_event_cooldown_sec=30.0,
            capitulation_event_cooldown_sec=30.0,
        ),
        maintenance=OIMaintenanceConfig(
            enable_periodic_cleanup=True,
            cleanup_interval_sec=17.0,
            enable_metrics_emit=True,
            metrics_interval_sec=11.0,
            cleanup_job_name="test.analytics.open_interest.cleanup",
            metrics_job_name="test.analytics.open_interest.metrics",
            cleanup_job_timeout_sec=3.0,
            metrics_job_timeout_sec=2.0,
            scheduler_job_max_retries=2,
            scheduler_job_retry_delay_sec=0.25,
        ),
    )


@pytest.fixture()
def event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture()
def scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture()
def analyzer(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> OIAnalyzer:
    item = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )
    item.regime_detector = StubRegimeDetector(OIRegime.NEUTRAL)  # type: ignore[assignment]
    item.anomaly_detector = StubAnomalyDetector(detected=False)  # type: ignore[assignment]
    return item


def event(
    topic: str,
    payload: Any,
    *,
    correlation_id: str = "corr-test-1",
    source: str = "test_feed",
) -> Event:
    return Event(
        topic=topic,
        payload=payload,
        source=source,
        correlation_id=correlation_id,
    )


def candle_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
    timestamp: float | None = None,
    close: float = 30_000.0,
    volume: float = 1_000.0,
    quote_volume: float | None = None,
    is_closed: bool = True,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "is_closed": is_closed,
    }


def candles_updated_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    candles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles or [],
    }


def oi_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
    timestamp: float | None = None,
    oi: float = 1_000.0,
    open_interest_value: float | None = None,
    mark_price: float | None = None,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "open_interest": oi,
        "open_interest_value": open_interest_value,
        "mark_price": mark_price,
    }


def trade_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    timestamp: float | None = None,
    price: float = 30_010.0,
    qty: float = 2.5,
    side: str = "buy",
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "price": price,
        "qty": qty,
        "side": side,
    }


def trades_updated_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": trades or [],
    }


def funding_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    timestamp: float | None = None,
    funding_rate: float = 0.012,
    predicted_rate: float | None = None,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "funding_rate": funding_rate,
        "predicted_rate": predicted_rate,
    }


def liquidation_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    timestamp: float | None = None,
    long_liquidations: float = 100.0,
    short_liquidations: float = 300.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "long_liquidations": long_liquidations,
        "short_liquidations": short_liquidations,
    }


def orderflow_payload(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
    timestamp: float | None = None,
    cvd_delta: float = 123.0,
    aggressive_buy_volume: float = 700.0,
    aggressive_sell_volume: float = 300.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "cvd_delta": cvd_delta,
        "aggressive_buy_volume": aggressive_buy_volume,
        "aggressive_sell_volume": aggressive_sell_volume,
    }


async def seed_price_context(
    analyzer: OIAnalyzer,
    *,
    ts: float | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_CONTEXT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL.lower(),
    timeframe: str = DEFAULT_CONTEXT_TIMEFRAME,
) -> None:
    base_ts = ts if ts is not None else now_ts()

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=base_ts,
                close=30_000.0,
                volume=900.0,
            ),
        )
    )
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=base_ts + 1.0,
                close=30_120.0,
                volume=1_100.0,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Registration / Scheduler integration
# ---------------------------------------------------------------------------

def test_register_subscribes_to_production_topics_and_scheduler_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    analyzer.register()

    patterns = {subscription.pattern for subscription in event_bus.subscriptions}
    names = {subscription.name for subscription in event_bus.subscriptions}

    assert patterns == set(config.production_input_topics)
    assert patterns == {
        OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
        OIMarketEventType.CANDLE_CLOSED.topic,
        OIMarketEventType.CANDLES_UPDATED.topic,
        OIMarketEventType.TRADES_UPDATED.topic,
        OIMarketEventType.FUNDING_UPDATED.topic,
        OIMarketEventType.ORDERFLOW_UPDATED.topic,
        OIMarketEventType.LIQUIDATIONS_UPDATED.topic,
    }

    assert names == {
        f"oi_analyzer.on_open_interest_updated:{OIMarketEventType.OPEN_INTEREST_UPDATED.topic}",
        f"oi_analyzer.on_candle_closed:{OIMarketEventType.CANDLE_CLOSED.topic}",
        f"oi_analyzer.on_candles_updated:{OIMarketEventType.CANDLES_UPDATED.topic}",
        f"oi_analyzer.on_trades_updated:{OIMarketEventType.TRADES_UPDATED.topic}",
        f"oi_analyzer.on_funding_updated:{OIMarketEventType.FUNDING_UPDATED.topic}",
        f"oi_analyzer.on_orderflow_updated:{OIMarketEventType.ORDERFLOW_UPDATED.topic}",
        f"oi_analyzer.on_liquidations_updated:{OIMarketEventType.LIQUIDATIONS_UPDATED.topic}",
    }

    assert len(event_bus.subscriptions) == len(config.production_input_topics)
    assert len(scheduler.jobs) == 2

    cleanup_job = next(
        job for job in scheduler.jobs.values()
        if job["name"] == config.maintenance.cleanup_job_name
    )
    metrics_job = next(
        job for job in scheduler.jobs.values()
        if job["name"] == config.maintenance.metrics_job_name
    )

    assert cleanup_job["func"] == analyzer.cleanup_stale_state
    assert cleanup_job["interval"] == config.maintenance.cleanup_interval_sec
    assert cleanup_job["timeout"] == config.maintenance.cleanup_job_timeout_sec
    assert cleanup_job["allow_overlap"] is False
    assert cleanup_job["run_immediately"] is False
    assert cleanup_job["max_retries"] == config.maintenance.scheduler_job_max_retries
    assert cleanup_job["retry_delay"] == config.maintenance.scheduler_job_retry_delay_sec

    assert metrics_job["func"] == analyzer.emit_metrics
    assert metrics_job["interval"] == config.maintenance.metrics_interval_sec
    assert metrics_job["timeout"] == config.maintenance.metrics_job_timeout_sec
    assert metrics_job["allow_overlap"] is False
    assert metrics_job["run_immediately"] is False

    stats = analyzer.stats()
    assert stats["registered"] is True
    assert stats["subscriptions"] == len(config.production_input_topics)
    assert stats["cleanup_job_registered"] is True
    assert stats["metrics_job_registered"] is True


def test_register_is_idempotent_and_does_not_duplicate_subscriptions_or_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    analyzer.register()
    analyzer.register()
    analyzer.register()

    assert len(event_bus.subscriptions) == len(config.production_input_topics)
    assert len(scheduler.jobs) == 2
    assert len(scheduler.add_calls) == 2


def test_register_without_scheduler_does_not_create_jobs_but_keeps_subscriptions(
    event_bus: FakeEventBus,
    config: OIAnalyzerConfig,
) -> None:
    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=None,
        config=config,
    )

    analyzer.register()

    assert len(event_bus.subscriptions) == len(config.production_input_topics)
    assert analyzer.stats()["registered"] is True
    assert analyzer.stats()["cleanup_job_registered"] is False
    assert analyzer.stats()["metrics_job_registered"] is False


def test_unregister_removes_subscriptions_and_scheduler_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    analyzer.register()
    job_ids = set(scheduler.jobs)

    analyzer.unregister()

    assert event_bus.subscriptions == []
    assert len(event_bus.unsubscribed) == len(config.production_input_topics)
    assert set(scheduler.removed) == job_ids
    assert scheduler.jobs == {}
    assert analyzer.stats()["registered"] is False
    assert analyzer.stats()["subscriptions"] == 0
    assert analyzer.stats()["cleanup_job_registered"] is False
    assert analyzer.stats()["metrics_job_registered"] is False


# ---------------------------------------------------------------------------
# Context handlers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_candle_closed_updates_price_volume_context_and_buffers(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(timestamp=ts, close=30_000.0, volume=1_000.0),
        )
    )
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(timestamp=ts + 1.0, close=30_300.0, volume=2_000.0),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.exchange == DEFAULT_EXCHANGE
    assert context.market_type == DEFAULT_CONTEXT_MARKET_TYPE
    assert context.symbol == DEFAULT_SYMBOL
    assert context.timeframe == DEFAULT_CONTEXT_TIMEFRAME
    assert context.price == pytest.approx(30_300.0)
    assert context.price_delta == pytest.approx(300.0)
    assert context.price_delta_pct == pytest.approx(1.0)
    assert context.volume == pytest.approx(2_000.0)
    assert context.volume_ma is not None
    assert context.volume_ratio is not None

    buffers = analyzer._buffers[key()]
    assert list(buffers.price_values) == [30_000.0, 30_300.0]
    assert list(buffers.volume_values) == [1_000.0, 2_000.0]

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["candle_events_processed"] == 2
    assert runtime["processed_by_topic"][OIMarketEventType.CANDLE_CLOSED.topic] == 2


@pytest.mark.asyncio()
async def test_candles_updated_applies_batch_and_preserves_scope(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_candles_updated(
        event(
            OIMarketEventType.CANDLES_UPDATED.topic,
            candles_updated_payload(
                candles=[
                    candle_payload(timestamp=ts, close=30_000.0, volume=1_000.0),
                    candle_payload(timestamp=ts + 1.0, close=30_120.0, volume=1_500.0),
                    candle_payload(timestamp=ts + 2.0, close=30_240.0, volume=1_800.0),
                ]
            ),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_context is not None
    assert state.last_context.price == pytest.approx(30_240.0)

    buffers = analyzer._buffers[key()]
    assert list(buffers.price_values) == [30_000.0, 30_120.0, 30_240.0]
    assert list(buffers.volume_values) == [1_000.0, 1_500.0, 1_800.0]

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["candles_updated_events_processed"] == 3
    assert runtime["processed_by_topic"][OIMarketEventType.CANDLES_UPDATED.topic] == 1


@pytest.mark.asyncio()
async def test_trades_updated_updates_price_volume_and_aggressive_flow_without_candle(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_trades_updated(
        event(
            OIMarketEventType.TRADES_UPDATED.topic,
            trades_updated_payload(
                trades=[
                    trade_payload(timestamp=ts, price=30_010.0, qty=2.5, side="buy"),
                    trade_payload(timestamp=ts + 1.0, price=30_000.0, qty=1.5, side="sell"),
                ]
            ),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.price == pytest.approx(30_000.0)
    assert context.price_delta == pytest.approx(-10.0)
    assert context.aggressive_buy_volume == pytest.approx(2.5)
    assert context.aggressive_sell_volume == pytest.approx(1.5)
    assert context.aggressive_flow_imbalance == pytest.approx((2.5 - 1.5) / (2.5 + 1.5))

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["trades_events_processed"] == 2


@pytest.mark.asyncio()
async def test_trade_with_negative_qty_does_not_poison_volume_or_aggressive_flow(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_trades_updated(
        event(
            OIMarketEventType.TRADES_UPDATED.topic,
            trade_payload(timestamp=ts, price=30_010.0, qty=-999.0, side="buy"),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.price == pytest.approx(30_010.0)
    assert context.volume is None
    assert context.aggressive_buy_volume is None
    assert context.aggressive_sell_volume is None

    buffers = analyzer._buffers[key()]
    assert list(buffers.price_values) == [30_010.0]
    assert list(buffers.volume_values) == []


@pytest.mark.asyncio()
async def test_funding_liquidation_and_orderflow_enrich_existing_context(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await seed_price_context(analyzer, ts=ts)
    await analyzer.on_funding(
        event(
            OIMarketEventType.FUNDING_UPDATED.topic,
            funding_payload(timestamp=ts + 2.0, funding_rate=0.018, predicted_rate=0.019),
        )
    )
    await analyzer.on_liquidation(
        event(
            OIMarketEventType.LIQUIDATIONS_UPDATED.topic,
            liquidation_payload(
                timestamp=ts + 3.0,
                long_liquidations=700.0,
                short_liquidations=100.0,
            ),
        )
    )
    await analyzer.on_orderflow_update(
        event(
            OIMarketEventType.ORDERFLOW_UPDATED.topic,
            orderflow_payload(
                timestamp=ts + 4.0,
                cvd_delta=-321.0,
                aggressive_buy_volume=250.0,
                aggressive_sell_volume=900.0,
            ),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.funding_rate == pytest.approx(0.018)
    assert context.predicted_funding_rate == pytest.approx(0.019)
    assert context.long_liquidations == pytest.approx(700.0)
    assert context.short_liquidations == pytest.approx(100.0)
    assert context.liquidation_imbalance == pytest.approx((100.0 - 700.0) / 800.0)
    assert context.cvd_delta == pytest.approx(-321.0)
    assert context.aggressive_buy_volume == pytest.approx(250.0)
    assert context.aggressive_sell_volume == pytest.approx(900.0)

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["funding_events_processed"] == 1
    assert runtime["liquidations_events_processed"] == 1
    assert runtime["orderflow_events_processed"] == 1


@pytest.mark.asyncio()
async def test_payload_aliases_and_millisecond_timestamps_are_normalized(
    analyzer: OIAnalyzer,
) -> None:
    timestamp_ms = int(now_ts() * 1000)

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            {
                "venue": "ByBit",
                "market_type": "linear",
                "instrument": "ethusdt",
                "timeframe": "5m",
                "T": timestamp_ms,
                "c": "2100.5",
                "v": "123.45",
            },
        )
    )

    state = analyzer.get_state("bybit", "ETHUSDT", "linear", "5m")
    assert state is not None
    assert state.last_context is not None
    assert state.last_context.timestamp == pytest.approx(timestamp_ms / 1000.0)
    assert state.last_context.price == pytest.approx(2100.5)
    assert state.last_context.volume == pytest.approx(123.45)


# ---------------------------------------------------------------------------
# Open Interest event flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_on_open_interest_requires_price_context_when_config_enabled(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    config.require_price_context = True
    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )
    analyzer.regime_detector = StubRegimeDetector(OIRegime.LONG_BUILDUP)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(  # type: ignore[assignment]
        detected=True,
        anomaly_type=OIAnomalyType.OI_SPIKE,
    )

    ts = now_ts()

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts, oi=1_000.0),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is None
    assert state.last_analysis is None
    assert state.last_regime is OIRegime.NEUTRAL
    assert event_bus.emitted == []

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["open_interest_events_processed"] == 1
    assert runtime["skipped_missing_context"] == 1


@pytest.mark.asyncio()
async def test_on_open_interest_builds_analysis_updates_state_and_emits_updated_event(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
            correlation_id="corr-oi-updated",
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is not None
    assert state.last_analysis is not None
    assert state.last_regime is OIRegime.NEUTRAL

    features = state.last_features
    assert features.key == key()
    assert features.oi == pytest.approx(1_000.0)
    assert features.price == pytest.approx(30_120.0)

    assert event_bus.topics() == [OIEventType.UPDATED.topic]

    emitted = event_bus.by_topic(OIEventType.UPDATED.topic)[0]
    assert emitted.priority is EventPriority.NORMAL
    assert emitted.source == "test_oi_analyzer"
    assert emitted.correlation_id == "corr-oi-updated"

    payload = emitted.payload
    assert_default_scope_payload(payload)
    assert payload["snapshot"]["oi"] == pytest.approx(1_000.0)
    assert payload["features"]["oi"] == pytest.approx(1_000.0)
    assert payload["metadata"]["source_topic"] == OIMarketEventType.OPEN_INTEREST_UPDATED.topic
    assert payload["metadata"]["correlation_id"] == "corr-oi-updated"

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["analyses_built"] == 1
    assert runtime["emitted_updates"] == 1


@pytest.mark.asyncio()
async def test_open_interest_missing_oi_or_key_is_ignored_without_mutating_state(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            {
                "exchange": DEFAULT_EXCHANGE,
                "market_type": DEFAULT_CONTEXT_MARKET_TYPE,
                "symbol": DEFAULT_SYMBOL,
                "timeframe": DEFAULT_CONTEXT_TIMEFRAME,
                "timestamp": now_ts(),
            },
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            {
                "exchange": DEFAULT_EXCHANGE,
                "market_type": DEFAULT_CONTEXT_MARKET_TYPE,
                "timestamp": now_ts(),
                "open_interest": 123.0,
            },
        )
    )

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["skipped_missing_oi"] >= 1
    assert runtime["skipped_invalid_payload"] >= 1


@pytest.mark.asyncio()
async def test_disabled_config_ignores_all_handlers_without_side_effects(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    config.enabled = False
    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )

    ts = now_ts()

    await analyzer.on_candle(event(OIMarketEventType.CANDLE_CLOSED.topic, candle_payload(timestamp=ts)))
    await analyzer.on_candles_updated(
        event(OIMarketEventType.CANDLES_UPDATED.topic, candles_updated_payload())
    )
    await analyzer.on_trades_updated(event(OIMarketEventType.TRADES_UPDATED.topic, trade_payload(timestamp=ts)))
    await analyzer.on_funding(event(OIMarketEventType.FUNDING_UPDATED.topic, funding_payload(timestamp=ts)))
    await analyzer.on_liquidation(
        event(OIMarketEventType.LIQUIDATIONS_UPDATED.topic, liquidation_payload(timestamp=ts))
    )
    await analyzer.on_orderflow_update(
        event(OIMarketEventType.ORDERFLOW_UPDATED.topic, orderflow_payload(timestamp=ts))
    )
    await analyzer.on_open_interest(
        event(OIMarketEventType.OPEN_INTEREST_UPDATED.topic, oi_payload(timestamp=ts))
    )

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []


@pytest.mark.asyncio()
async def test_invalid_non_mapping_payloads_do_not_escape_handlers(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    bad_event = event(OIMarketEventType.CANDLE_CLOSED.topic, ["not", "a", "mapping"])

    await analyzer.on_candle(bad_event)
    await analyzer.on_candles_updated(bad_event)
    await analyzer.on_trades_updated(bad_event)
    await analyzer.on_funding(bad_event)
    await analyzer.on_liquidation(bad_event)
    await analyzer.on_orderflow_update(bad_event)
    await analyzer.on_open_interest(bad_event)

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []
    assert analyzer.stats()["runtime_stats"]["errors_count"] >= 0


# ---------------------------------------------------------------------------
# Emission matrix / cooldown behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_regime_divergence_anomaly_squeeze_and_capitulation_events_are_emitted_with_priorities(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    analyzer.regime_detector = StubRegimeDetector(OIRegime.SQUEEZE_SETUP)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(  # type: ignore[assignment]
        detected=True,
        anomaly_type=OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
    )
    monkeypatch.setattr(
        analyzer,
        "_detect_divergence_if_possible",
        lambda _key: detected_divergence(OIDivergenceType.PRICE_UP_OI_DOWN),
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_200.0),
            correlation_id="corr-special-events",
        )
    )

    topics = event_bus.topics()

    assert topics == [
        OIEventType.UPDATED.topic,
        OIEventType.REGIME_CHANGED.topic,
        OIEventType.DIVERGENCE_DETECTED.topic,
        OIEventType.ANOMALY_DETECTED.topic,
        OIEventType.SQUEEZE_SETUP.topic,
        OIEventType.CAPITULATION_DETECTED.topic,
    ]

    by_topic = {item.topic: item for item in event_bus.emitted}

    assert by_topic[OIEventType.UPDATED.topic].priority is EventPriority.NORMAL
    assert by_topic[OIEventType.REGIME_CHANGED.topic].priority is EventPriority.HIGH
    assert by_topic[OIEventType.DIVERGENCE_DETECTED.topic].priority is EventPriority.HIGH
    assert by_topic[OIEventType.ANOMALY_DETECTED.topic].priority is EventPriority.HIGH
    assert by_topic[OIEventType.SQUEEZE_SETUP.topic].priority is EventPriority.HIGH
    assert by_topic[OIEventType.CAPITULATION_DETECTED.topic].priority is EventPriority.CRITICAL

    assert all(item.source == "test_oi_analyzer" for item in event_bus.emitted)
    assert all(item.correlation_id == "corr-special-events" for item in event_bus.emitted)

    assert by_topic[OIEventType.REGIME_CHANGED.topic].payload["previous_regime"] == OIRegime.NEUTRAL.value
    assert by_topic[OIEventType.REGIME_CHANGED.topic].payload["new_regime"] == OIRegime.SQUEEZE_SETUP.value
    assert by_topic[OIEventType.DIVERGENCE_DETECTED.topic].payload["divergence_type"] == OIDivergenceType.PRICE_UP_OI_DOWN.value
    assert by_topic[OIEventType.ANOMALY_DETECTED.topic].payload["anomaly_type"] == OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP.value
    assert by_topic[OIEventType.CAPITULATION_DETECTED.topic].payload["anomaly_type"] == OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP.value

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["emitted_updates"] == 1
    assert runtime["emitted_regime_changes"] == 1
    assert runtime["emitted_divergences"] == 1
    assert runtime["emitted_anomalies"] == 1
    assert runtime["emitted_squeeze_setups"] == 1
    assert runtime["emitted_capitulations"] == 1


@pytest.mark.asyncio()
async def test_cooldown_suppresses_duplicate_high_level_events_but_not_updates(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    analyzer.regime_detector = StubRegimeDetector(OIRegime.SQUEEZE_SETUP)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(  # type: ignore[assignment]
        detected=True,
        anomaly_type=OIAnomalyType.OI_SPIKE,
    )
    monkeypatch.setattr(
        analyzer,
        "_detect_divergence_if_possible",
        lambda _key: detected_divergence(OIDivergenceType.PRICE_UP_OI_DOWN),
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 3.0, oi=1_010.0),
        )
    )

    topics = event_bus.topics()

    assert topics.count(OIEventType.UPDATED.topic) == 2
    assert topics.count(OIEventType.REGIME_CHANGED.topic) == 1
    assert topics.count(OIEventType.DIVERGENCE_DETECTED.topic) == 1
    assert topics.count(OIEventType.ANOMALY_DETECTED.topic) == 1
    assert topics.count(OIEventType.SQUEEZE_SETUP.topic) == 1


@pytest.mark.asyncio()
async def test_high_level_events_emit_again_after_cooldown_window(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    analyzer.regime_detector = StubRegimeDetector(OIRegime.SQUEEZE_SETUP)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(  # type: ignore[assignment]
        detected=True,
        anomaly_type=OIAnomalyType.OI_SPIKE,
    )
    monkeypatch.setattr(
        analyzer,
        "_detect_divergence_if_possible",
        lambda _key: detected_divergence(OIDivergenceType.PRICE_UP_OI_DOWN),
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 40.0, oi=1_100.0),
        )
    )

    topics = event_bus.topics()

    assert topics.count(OIEventType.UPDATED.topic) == 2
    assert topics.count(OIEventType.DIVERGENCE_DETECTED.topic) == 2
    assert topics.count(OIEventType.ANOMALY_DETECTED.topic) == 2
    assert topics.count(OIEventType.SQUEEZE_SETUP.topic) == 2


@pytest.mark.asyncio()
async def test_not_detected_divergence_and_anomaly_do_not_emit_high_level_events(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    analyzer.regime_detector = StubRegimeDetector(OIRegime.NEUTRAL)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(detected=False)  # type: ignore[assignment]
    monkeypatch.setattr(
        analyzer,
        "_detect_divergence_if_possible",
        lambda _key: not_detected_divergence(),
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )

    assert event_bus.topics() == [OIEventType.UPDATED.topic]


# ---------------------------------------------------------------------------
# Scope isolation / filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_scope_isolation_for_same_symbol_different_market_type_and_timeframe(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await seed_price_context(
        analyzer,
        ts=ts,
        market_type="usdm_futures",
        timeframe="1m",
    )
    await seed_price_context(
        analyzer,
        ts=ts,
        market_type="coinm_futures",
        timeframe="5m",
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(
                market_type="usdm_futures",
                timeframe="1m",
                timestamp=ts + 3.0,
                oi=1_000.0,
            ),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(
                market_type="coinm_futures",
                timeframe="5m",
                timestamp=ts + 3.0,
                oi=2_000.0,
            ),
        )
    )

    state_1m = analyzer.get_state(DEFAULT_EXCHANGE, DEFAULT_SYMBOL, "usdm_futures", "1m")
    state_5m = analyzer.get_state(DEFAULT_EXCHANGE, DEFAULT_SYMBOL, "coinm_futures", "5m")

    assert state_1m is not None
    assert state_5m is not None
    assert state_1m.key != state_5m.key

    assert state_1m.last_features is not None
    assert state_5m.last_features is not None
    assert state_1m.last_features.oi == pytest.approx(1_000.0)
    assert state_5m.last_features.oi == pytest.approx(2_000.0)

    assert key(market_type="usdm_futures", timeframe="1m") in analyzer._buffers
    assert key(market_type="coinm_futures", timeframe="5m") in analyzer._buffers


@pytest.mark.asyncio()
async def test_scope_filters_skip_disallowed_symbol_without_state_mutation(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    config.allowed_symbols = {"ETHUSDT"}

    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(symbol="BTCUSDT", oi=1_000.0),
        )
    )

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert analyzer.stats()["runtime_stats"]["skipped_by_scope_filter"] == 1
    assert event_bus.emitted == []


# ---------------------------------------------------------------------------
# Cleanup / metrics / health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_cleanup_removes_stale_state_buffers_context_and_cooldowns(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_now = 20_000.0
    stale_ts = fake_now - analyzer.config.stale_state_cleanup_after_sec - 1.0
    current_key = key()

    state = analyzer._get_or_create_state(current_key)
    state.touch(stale_ts)
    analyzer._get_or_create_buffers(current_key)
    analyzer._last_context_ts[current_key] = stale_ts
    analyzer._cooldowns[(*current_key, "anomaly")] = stale_ts

    monkeypatch.setattr(analyzer, "_now", lambda: fake_now)

    await analyzer.cleanup_stale_state()

    assert current_key not in analyzer._states
    assert current_key not in analyzer._buffers
    assert current_key not in analyzer._last_context_ts
    assert analyzer._cooldowns == {}

    assert event_bus.topics() == [OIEventType.STATE_CLEANED.topic]
    emitted = event_bus.emitted[0]
    assert emitted.priority is EventPriority.LOW
    assert emitted.payload["removed_count"] == 1
    assert emitted.payload["removed_keys"][0]["market_type"] == DEFAULT_CONTEXT_MARKET_TYPE

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["cleanup_runs"] == 1
    assert runtime["cleanup_removed_states"] == 1
    assert runtime["emitted_state_cleaned"] == 1


@pytest.mark.asyncio()
async def test_cleanup_noops_without_emitting_when_nothing_is_stale(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_now = 20_000.0
    current_key = key()

    state = analyzer._get_or_create_state(current_key)
    state.touch(fake_now)
    analyzer._get_or_create_buffers(current_key)

    monkeypatch.setattr(analyzer, "_now", lambda: fake_now)

    await analyzer.cleanup_stale_state()

    assert current_key in analyzer._states
    assert current_key in analyzer._buffers
    assert event_bus.emitted == []
    assert analyzer.stats()["runtime_stats"]["cleanup_runs"] == 1


@pytest.mark.asyncio()
async def test_emit_metrics_and_health_publish_low_priority_diagnostics(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    analyzer.register()

    await analyzer.emit_metrics()
    await analyzer.emit_health()

    assert event_bus.topics() == [
        OIEventType.METRICS.topic,
        OIEventType.HEALTH.topic,
    ]

    metrics = event_bus.by_topic(OIEventType.METRICS.topic)[0]
    health = event_bus.by_topic(OIEventType.HEALTH.topic)[0]

    assert metrics.priority is EventPriority.LOW
    assert health.priority is EventPriority.LOW
    assert metrics.payload["registered"] is True
    assert health.payload["registered"] is True
    assert health.payload["scope"] == "exchange:market_type:symbol:timeframe"

    assert analyzer.stats()["runtime_stats"]["emitted_metrics"] == 1


# ---------------------------------------------------------------------------
# History bounds / resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_history_buffers_are_bounded_by_config_history_size(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    config.require_price_context = False
    config.emit_regime_changes = False
    config.emit_divergences = False
    config.emit_anomalies = False
    config.emit_squeeze_events = False
    config.emit_capitulation_events = False

    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )
    analyzer.regime_detector = StubRegimeDetector(OIRegime.NEUTRAL)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(detected=False)  # type: ignore[assignment]

    ts = now_ts()

    for idx in range(config.windows.history_size + 7):
        await analyzer.on_open_interest(
            event(
                OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
                oi_payload(timestamp=ts + idx, oi=1_000.0 + idx),
            )
        )

    buffers = analyzer._buffers[key()]

    assert len(buffers.oi_values) == config.windows.history_size
    assert len(buffers.oi_timestamps) == config.windows.history_size
    assert len(buffers.feature_history) == config.windows.history_size

    assert list(buffers.oi_values)[0] == pytest.approx(1_000.0 + 7)
    assert list(buffers.oi_values)[-1] == pytest.approx(1_000.0 + config.windows.history_size + 6)

    assert event_bus.topics().count(OIEventType.UPDATED.topic) == config.windows.history_size + 7
    assert analyzer.stats()["runtime_stats"]["analyses_built"] == config.windows.history_size + 7


@pytest.mark.asyncio()
async def test_emit_failure_inside_open_interest_handler_is_caught_after_state_update(
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    event_bus = FakeEventBus(fail_on_topics={OIEventType.UPDATED.topic})
    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )
    analyzer.regime_detector = StubRegimeDetector(OIRegime.NEUTRAL)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(detected=False)  # type: ignore[assignment]

    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_features is not None
    assert state.last_analysis is not None

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["analyses_built"] == 1
    assert runtime["errors_count"] == 1
    assert "forced emit failure" in (runtime["last_error"] or "")


@pytest.mark.asyncio()
async def test_error_in_detector_is_caught_and_does_not_escape_handler(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    class BrokenRegimeDetector:
        def detect(self, _features):
            raise RuntimeError("boom-regime")

    analyzer.regime_detector = BrokenRegimeDetector()  # type: ignore[assignment]

    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )

    runtime = analyzer.stats()["runtime_stats"]
    assert runtime["errors_count"] == 1
    assert "boom-regime" in (runtime["last_error"] or "")
    assert event_bus.emitted == []


@pytest.mark.asyncio()
async def test_out_of_order_context_events_do_not_break_analysis(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    ts = now_ts()

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(timestamp=ts + 10.0, close=30_300.0, volume=2_000.0),
        )
    )
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE_CLOSED.topic,
            candle_payload(timestamp=ts + 5.0, close=30_000.0, volume=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST_UPDATED.topic,
            oi_payload(timestamp=ts + 11.0, oi=1_050.0),
        )
    )

    state = analyzer.get_state(
        DEFAULT_EXCHANGE,
        DEFAULT_SYMBOL,
        DEFAULT_CONTEXT_MARKET_TYPE,
        DEFAULT_CONTEXT_TIMEFRAME,
    )
    assert state is not None
    assert state.last_features is not None
    assert state.last_analysis is not None

    assert OIEventType.UPDATED.topic in event_bus.topics()
    assert analyzer.stats()["runtime_stats"]["errors_count"] == 0