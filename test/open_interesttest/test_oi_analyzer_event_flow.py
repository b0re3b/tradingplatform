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
    OIAnomalyResult,
    OIDivergenceResult,
    OIRegimeResult,
)
from analytics.open_interest.oi_analyzer import OIAnalyzer


# Якщо твій package root саме trading_system.*, заміни imports вище на:
# from trading_system.analytics.open_interest...


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


class FakeEventBus:
    """
    Minimal EventBus test double.

    Ми не тестуємо core.EventBus worker loop.
    Ми тестуємо, чи OIAnalyzer правильно користується core-контрактом:
    - subscribe(pattern, handler, name=...)
    - unsubscribe(subscription)
    - emit(topic, payload, priority=..., source=..., correlation_id=...)
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
            )
        )
        return True

    def topics(self) -> list[str]:
        return [item.topic for item in self.emitted]

    def by_topic(self, topic: str) -> list[EmittedEvent]:
        return [item for item in self.emitted if item.topic == topic]

    def clear(self) -> None:
        self.emitted.clear()


class FakeScheduler:
    """
    Minimal Scheduler test double.

    Ми перевіряємо, що OIAnalyzer реєструє periodic jobs через
    add_interval_job(), а не створює власні asyncio loops.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.disabled: list[str] = []
        self.add_calls: list[dict[str, Any]] = []

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

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def disable_job(self, job_id: str) -> None:
        self.disabled.append(job_id)
        if job_id in self.jobs:
            self.jobs[job_id]["enabled"] = False


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
        emit_updates=True,
        emit_regime_changes=True,
        emit_divergences=True,
        emit_anomalies=True,
        emit_squeeze_events=True,
        emit_capitulation_events=True,
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


def now_ts() -> float:
    return time.time()


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
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    close: float = 30_000.0,
    volume: float = 1_000.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "close": close,
        "volume": volume,
    }


def oi_payload(
    *,
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    oi: float = 1_000.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "open_interest": oi,
    }


def trade_payload(
    *,
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    price: float = 30_010.0,
    qty: float = 2.5,
    side: str = "buy",
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "price": price,
        "qty": qty,
        "side": side,
    }


def funding_payload(
    *,
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    funding_rate: float = 0.012,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "funding_rate": funding_rate,
    }


def liquidation_payload(
    *,
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    long_liquidations: float = 100.0,
    short_liquidations: float = 300.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "long_liquidations": long_liquidations,
        "short_liquidations": short_liquidations,
    }


def orderflow_payload(
    *,
    exchange: str = "binance",
    symbol: str = "btcusdt",
    timestamp: float | None = None,
    cvd_delta: float = 123.0,
    aggressive_buy_volume: float = 700.0,
    aggressive_sell_volume: float = 300.0,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": timestamp if timestamp is not None else now_ts(),
        "cvd_delta": cvd_delta,
        "aggressive_buy_volume": aggressive_buy_volume,
        "aggressive_sell_volume": aggressive_sell_volume,
    }


async def seed_price_context(
    analyzer: OIAnalyzer,
    *,
    ts: float | None = None,
    exchange: str = "binance",
    symbol: str = "btcusdt",
) -> None:
    base_ts = ts if ts is not None else now_ts()
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            candle_payload(
                exchange=exchange,
                symbol=symbol,
                timestamp=base_ts,
                close=30_000.0,
                volume=900.0,
            ),
        )
    )
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            candle_payload(
                exchange=exchange,
                symbol=symbol,
                timestamp=base_ts + 1.0,
                close=30_120.0,
                volume=1_100.0,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Registration / Scheduler integration
# ---------------------------------------------------------------------------

def test_register_subscribes_to_required_market_topics_and_scheduler_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    analyzer.register()

    patterns = {subscription.pattern for subscription in event_bus.subscriptions}
    names = {subscription.name for subscription in event_bus.subscriptions}

    assert patterns == {
        OIMarketEventType.OPEN_INTEREST.topic,
        OIMarketEventType.CANDLE.topic,
        OIMarketEventType.TRADE.topic,
        OIMarketEventType.FUNDING.topic,
        OIMarketEventType.LIQUIDATION.topic,
        OIMarketEventType.ORDERFLOW_UPDATED.topic,
    }

    assert names == {
        "oi_analyzer.on_open_interest",
        "oi_analyzer.on_candle",
        "oi_analyzer.on_trade",
        "oi_analyzer.on_funding",
        "oi_analyzer.on_liquidation",
        "oi_analyzer.on_orderflow_update",
    }

    assert len(scheduler.jobs) == 2
    assert {job["name"] for job in scheduler.jobs.values()} == {
        config.maintenance.cleanup_job_name,
        config.maintenance.metrics_job_name,
    }

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

    assert metrics_job["func"] == analyzer.emit_metrics
    assert metrics_job["interval"] == config.maintenance.metrics_interval_sec
    assert metrics_job["timeout"] == config.maintenance.metrics_job_timeout_sec
    assert metrics_job["allow_overlap"] is False
    assert metrics_job["run_immediately"] is False

    assert analyzer.stats()["registered"] is True
    assert analyzer.stats()["cleanup_job_registered"] is True
    assert analyzer.stats()["metrics_job_registered"] is True


def test_register_is_idempotent_and_does_not_duplicate_subscriptions_or_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
) -> None:
    analyzer.register()
    analyzer.register()
    analyzer.register()

    assert len(event_bus.subscriptions) == 6
    assert len(scheduler.jobs) == 2
    assert len(scheduler.add_calls) == 2


def test_register_without_scheduler_does_not_create_background_loop_but_keeps_subscriptions(
    event_bus: FakeEventBus,
    config: OIAnalyzerConfig,
) -> None:
    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=None,
        config=config,
    )

    analyzer.register()

    assert len(event_bus.subscriptions) == 6
    assert analyzer.stats()["registered"] is True
    assert analyzer.stats()["cleanup_job_registered"] is False
    assert analyzer.stats()["metrics_job_registered"] is False


def test_unregister_removes_subscriptions_and_disables_scheduler_jobs(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
) -> None:
    analyzer.register()
    job_ids = set(scheduler.jobs)

    analyzer.unregister()

    assert event_bus.subscriptions == []
    assert len(event_bus.unsubscribed) == 6
    assert set(scheduler.disabled) == job_ids
    assert all(job["enabled"] is False for job in scheduler.jobs.values())
    assert analyzer.stats()["registered"] is False
    assert analyzer.stats()["subscriptions"] == 0


# ---------------------------------------------------------------------------
# Context handlers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_candle_updates_price_volume_context_and_buffers(analyzer: OIAnalyzer) -> None:
    ts = now_ts()

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            candle_payload(timestamp=ts, close=30_000.0, volume=1_000.0),
        )
    )
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            candle_payload(timestamp=ts + 1.0, close=30_300.0, volume=2_000.0),
        )
    )

    state = analyzer.get_state("BINANCE", "BTCUSDT")
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.exchange == "binance"
    assert context.symbol == "BTCUSDT"
    assert context.price == pytest.approx(30_300.0)
    assert context.price_delta == pytest.approx(300.0)
    assert context.price_delta_pct == pytest.approx(1.0)
    assert context.volume == pytest.approx(2_000.0)
    assert context.volume_ma is not None
    assert context.volume_ratio is not None

    buffers = analyzer._buffers[("binance", "BTCUSDT")]
    assert list(buffers.price_values) == [30_000.0, 30_300.0]
    assert list(buffers.volume_values) == [1_000.0, 2_000.0]


@pytest.mark.asyncio()
async def test_trade_updates_price_volume_and_aggressive_flow_without_requiring_candle(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_trade(
        event(
            OIMarketEventType.TRADE.topic,
            trade_payload(timestamp=ts, price=30_010.0, qty=2.5, side="buy"),
        )
    )
    await analyzer.on_trade(
        event(
            OIMarketEventType.TRADE.topic,
            trade_payload(timestamp=ts + 1.0, price=30_000.0, qty=1.5, side="sell"),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.price == pytest.approx(30_000.0)
    assert context.price_delta == pytest.approx(-10.0)
    assert context.aggressive_buy_volume == pytest.approx(2.5)
    assert context.aggressive_sell_volume == pytest.approx(1.5)
    assert context.aggressive_flow_imbalance == pytest.approx((2.5 - 1.5) / (2.5 + 1.5))


@pytest.mark.asyncio()
async def test_trade_with_negative_qty_does_not_poison_volume_or_aggressive_flow(
    analyzer: OIAnalyzer,
) -> None:
    ts = now_ts()

    await analyzer.on_trade(
        event(
            OIMarketEventType.TRADE.topic,
            trade_payload(timestamp=ts, price=30_010.0, qty=-999.0, side="buy"),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.price == pytest.approx(30_010.0)
    assert context.volume is None
    assert context.aggressive_buy_volume is None
    assert context.aggressive_sell_volume is None

    buffers = analyzer._buffers[("binance", "BTCUSDT")]
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
            OIMarketEventType.FUNDING.topic,
            funding_payload(timestamp=ts + 2.0, funding_rate=0.018),
        )
    )
    await analyzer.on_liquidation(
        event(
            OIMarketEventType.LIQUIDATION.topic,
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

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_context is not None

    context = state.last_context
    assert context.funding_rate == pytest.approx(0.018)
    assert context.long_liquidations == pytest.approx(700.0)
    assert context.short_liquidations == pytest.approx(100.0)
    assert context.liquidation_imbalance == pytest.approx((100.0 - 700.0) / 800.0)
    assert context.cvd_delta == pytest.approx(-321.0)
    assert context.aggressive_buy_volume == pytest.approx(250.0)
    assert context.aggressive_sell_volume == pytest.approx(900.0)


@pytest.mark.asyncio()
async def test_payload_aliases_and_millisecond_timestamps_are_normalized(
    analyzer: OIAnalyzer,
) -> None:
    timestamp_ms = int(now_ts() * 1000)

    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            {
                "venue": "ByBit",
                "instrument": "ethusdt",
                "T": timestamp_ms,
                "c": "2100.5",
                "v": "123.45",
            },
        )
    )

    state = analyzer.get_state("bybit", "ETHUSDT")
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
    analyzer.anomaly_detector = StubAnomalyDetector(detected=True, anomaly_type=OIAnomalyType.OI_SPIKE)  # type: ignore[assignment]

    ts = now_ts()

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts, oi=1_000.0),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is None
    assert state.last_analysis is None
    assert state.last_regime is OIRegime.NEUTRAL
    assert event_bus.emitted == []


@pytest.mark.asyncio()
async def test_on_open_interest_builds_analysis_updates_state_and_emits_updated_event(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    ts = now_ts()
    await seed_price_context(analyzer, ts=ts)

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
            correlation_id="corr-oi-updated",
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is not None
    assert state.last_analysis is not None
    assert state.last_regime is OIRegime.NEUTRAL

    assert event_bus.topics() == [OIEventType.UPDATED.topic]

    emitted = event_bus.by_topic(OIEventType.UPDATED.topic)[0]
    assert emitted.priority is EventPriority.NORMAL
    assert emitted.source == "test_oi_analyzer"
    assert emitted.correlation_id == "corr-oi-updated"

    payload = emitted.payload
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange"] == "binance"
    assert payload["snapshot"]["oi"] == pytest.approx(1_000.0)
    assert payload["metadata"]["source_topic"] == OIMarketEventType.OPEN_INTEREST.topic
    assert payload["metadata"]["correlation_id"] == "corr-oi-updated"


@pytest.mark.asyncio()
async def test_open_interest_missing_oi_or_key_is_ignored_without_mutating_state(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            {"exchange": "binance", "symbol": "btcusdt", "timestamp": now_ts()},
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            {"exchange": "binance", "timestamp": now_ts(), "oi": 123.0},
        )
    )

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []


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

    await analyzer.on_candle(event(OIMarketEventType.CANDLE.topic, candle_payload(timestamp=ts)))
    await analyzer.on_trade(event(OIMarketEventType.TRADE.topic, trade_payload(timestamp=ts)))
    await analyzer.on_funding(event(OIMarketEventType.FUNDING.topic, funding_payload(timestamp=ts)))
    await analyzer.on_liquidation(event(OIMarketEventType.LIQUIDATION.topic, liquidation_payload(timestamp=ts)))
    await analyzer.on_orderflow_update(event(OIMarketEventType.ORDERFLOW_UPDATED.topic, orderflow_payload(timestamp=ts)))
    await analyzer.on_open_interest(event(OIMarketEventType.OPEN_INTEREST.topic, oi_payload(timestamp=ts)))

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []


@pytest.mark.asyncio()
async def test_invalid_non_mapping_payloads_do_not_escape_handlers(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    bad_event = event(OIMarketEventType.CANDLE.topic, ["not", "a", "mapping"])

    await analyzer.on_candle(bad_event)
    await analyzer.on_trade(bad_event)
    await analyzer.on_funding(bad_event)
    await analyzer.on_liquidation(bad_event)
    await analyzer.on_orderflow_update(bad_event)
    await analyzer.on_open_interest(bad_event)

    assert analyzer.stats()["states"] == 0
    assert analyzer.stats()["buffers"] == 0
    assert event_bus.emitted == []


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
            OIMarketEventType.OPEN_INTEREST.topic,
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

    regime_payload = by_topic[OIEventType.REGIME_CHANGED.topic].payload
    assert regime_payload["previous_regime"] == OIRegime.NEUTRAL.value
    assert regime_payload["new_regime"] == OIRegime.SQUEEZE_SETUP.value

    divergence_payload = by_topic[OIEventType.DIVERGENCE_DETECTED.topic].payload
    assert divergence_payload["divergence_type"] == OIDivergenceType.PRICE_UP_OI_DOWN.value
    assert divergence_payload["window_size"] == 5

    anomaly_payload = by_topic[OIEventType.ANOMALY_DETECTED.topic].payload
    assert anomaly_payload["anomaly_type"] == OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP.value
    assert anomaly_payload["strength"] == OISignalStrength.HIGH.value

    capitulation_payload = by_topic[OIEventType.CAPITULATION_DETECTED.topic].payload
    assert capitulation_payload["anomaly_type"] == OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP.value
    assert capitulation_payload["reasons"]


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
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 3.0, oi=1_010.0),
        )
    )

    topics = event_bus.topics()

    assert topics.count(OIEventType.UPDATED.topic) == 2

    # regime_changed emits once because previous regime changes only first time.
    assert topics.count(OIEventType.REGIME_CHANGED.topic) == 1

    # cooldown suppresses repeated high-level spam events.
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
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 40.0, oi=1_100.0),
        )
    )

    topics = event_bus.topics()

    assert topics.count(OIEventType.UPDATED.topic) == 2
    assert topics.count(OIEventType.DIVERGENCE_DETECTED.topic) == 2
    assert topics.count(OIEventType.ANOMALY_DETECTED.topic) == 2
    assert topics.count(OIEventType.SQUEEZE_SETUP.topic) == 2


@pytest.mark.asyncio()
async def test_no_full_analysis_storage_still_emits_but_does_not_keep_heavy_result(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
) -> None:
    config.store_full_analysis = False
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
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is not None
    assert state.last_analysis is None

    assert event_bus.topics() == [OIEventType.UPDATED.topic]
    assert event_bus.emitted[0].payload["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# Maintenance: metrics / health / cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio()
async def test_emit_metrics_publishes_stats_payload(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    analyzer.register()

    await analyzer.emit_metrics()

    assert event_bus.topics() == [OIEventType.METRICS.topic]
    emitted = event_bus.emitted[0]

    assert emitted.priority is EventPriority.LOW
    assert emitted.source == "test_oi_analyzer"
    assert emitted.payload["registered"] is True
    assert emitted.payload["enabled"] is True
    assert emitted.payload["subscriptions"] == 6
    assert emitted.payload["cleanup_job_registered"] is True
    assert emitted.payload["metrics_job_registered"] is True
    assert isinstance(emitted.payload["instruments"], list)


@pytest.mark.asyncio()
async def test_emit_health_publishes_lifecycle_and_scheduler_metadata(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    analyzer.register()

    await analyzer.emit_health()

    assert event_bus.topics() == [OIEventType.HEALTH.topic]
    payload = event_bus.emitted[0].payload

    assert payload["registered"] is True
    assert payload["enabled"] is True
    assert payload["states"] == 0
    assert payload["buffers"] == 0
    assert payload["subscriptions"] == 6
    assert payload["scheduler_available"] is True
    assert payload["cleanup_job_id"] is not None
    assert payload["metrics_job_id"] is not None


@pytest.mark.asyncio()
async def test_cleanup_stale_state_removes_state_buffers_context_and_cooldowns(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_now = 10_000.0
    new_now = old_now + analyzer.config.stale_state_cleanup_after_sec + 1.0

    key_stale = ("binance", "BTCUSDT")
    key_fresh = ("bybit", "ETHUSDT")

    stale_state = analyzer._get_or_create_state(key_stale)
    stale_state.touch(old_now)
    analyzer._get_or_create_buffers(key_stale)
    analyzer._last_context_ts[key_stale] = old_now
    analyzer._cooldowns[(key_stale[0], key_stale[1], "anomaly")] = old_now
    analyzer._cooldowns[(key_stale[0], key_stale[1], "divergence")] = old_now

    fresh_state = analyzer._get_or_create_state(key_fresh)
    fresh_state.touch(new_now)
    analyzer._get_or_create_buffers(key_fresh)
    analyzer._last_context_ts[key_fresh] = new_now
    analyzer._cooldowns[(key_fresh[0], key_fresh[1], "anomaly")] = new_now

    monkeypatch.setattr(analyzer, "_now", lambda: new_now)

    await analyzer.cleanup_stale_state()

    assert key_stale not in analyzer._states
    assert key_stale not in analyzer._buffers
    assert key_stale not in analyzer._last_context_ts
    assert not any(cd_key[:2] == key_stale for cd_key in analyzer._cooldowns)

    assert key_fresh in analyzer._states
    assert key_fresh in analyzer._buffers
    assert key_fresh in analyzer._last_context_ts
    assert any(cd_key[:2] == key_fresh for cd_key in analyzer._cooldowns)

    assert event_bus.topics() == [OIEventType.STATE_CLEANED.topic]
    payload = event_bus.emitted[0].payload
    assert payload["removed_count"] == 1
    assert payload["removed_keys"] == [{"exchange": "binance", "symbol": "BTCUSDT"}]
    assert event_bus.emitted[0].priority is EventPriority.LOW


@pytest.mark.asyncio()
async def test_cleanup_noops_without_emitting_when_nothing_is_stale(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_now = 20_000.0
    key = ("binance", "BTCUSDT")

    state = analyzer._get_or_create_state(key)
    state.touch(fake_now)
    analyzer._get_or_create_buffers(key)

    monkeypatch.setattr(analyzer, "_now", lambda: fake_now)

    await analyzer.cleanup_stale_state()

    assert key in analyzer._states
    assert key in analyzer._buffers
    assert event_bus.emitted == []


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
                OIMarketEventType.OPEN_INTEREST.topic,
                oi_payload(timestamp=ts + idx, oi=1_000.0 + idx),
            )
        )

    buffers = analyzer._buffers[("binance", "BTCUSDT")]

    assert len(buffers.oi_values) == config.windows.history_size
    assert len(buffers.oi_timestamps) == config.windows.history_size
    assert len(buffers.feature_history) == config.windows.history_size

    assert list(buffers.oi_values)[0] == pytest.approx(1_000.0 + 7)
    assert list(buffers.oi_values)[-1] == pytest.approx(1_000.0 + config.windows.history_size + 6)

    assert event_bus.topics().count(OIEventType.UPDATED.topic) == config.windows.history_size + 7


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
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=ts + 2.0, oi=1_000.0),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_snapshot is not None
    assert state.last_features is not None
    assert state.last_regime is OIRegime.NEUTRAL

    # FakeEventBus raises before appending.
    assert event_bus.emitted == []


@pytest.mark.asyncio()
async def test_analyzer_does_not_cross_contaminate_multiple_instruments(
    analyzer: OIAnalyzer,
    event_bus: FakeEventBus,
) -> None:
    ts = now_ts()

    await seed_price_context(analyzer, ts=ts, exchange="binance", symbol="btcusdt")
    await seed_price_context(analyzer, ts=ts, exchange="bybit", symbol="ethusdt")

    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(exchange="binance", symbol="btcusdt", timestamp=ts + 3.0, oi=1_000.0),
        )
    )
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(exchange="bybit", symbol="ethusdt", timestamp=ts + 3.0, oi=2_000.0),
        )
    )

    btc_state = analyzer.get_state("binance", "BTCUSDT")
    eth_state = analyzer.get_state("bybit", "ETHUSDT")

    assert btc_state is not None
    assert eth_state is not None
    assert btc_state.last_snapshot is not None
    assert eth_state.last_snapshot is not None
    assert btc_state.last_snapshot.oi == pytest.approx(1_000.0)
    assert eth_state.last_snapshot.oi == pytest.approx(2_000.0)

    stats = analyzer.stats()
    instruments = {
        (item["exchange"], item["symbol"]): item
        for item in stats["instruments"]
    }

    assert ("binance", "BTCUSDT") in instruments
    assert ("bybit", "ETHUSDT") in instruments
    assert instruments[("binance", "BTCUSDT")]["has_state"] is True
    assert instruments[("bybit", "ETHUSDT")]["has_state"] is True

    updated_events = event_bus.by_topic(OIEventType.UPDATED.topic)
    assert len(updated_events) == 2
    assert {item.payload["symbol"] for item in updated_events} == {"BTCUSDT", "ETHUSDT"}


@pytest.mark.asyncio()
async def test_stale_context_is_not_used_for_required_price_context(
    event_bus: FakeEventBus,
    scheduler: FakeScheduler,
    config: OIAnalyzerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.require_price_context = True
    config.stale_context_after_sec = 10.0
    config.stale_state_cleanup_after_sec = 100.0

    analyzer = OIAnalyzer(
        event_bus=event_bus,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        config=config,
    )
    analyzer.regime_detector = StubRegimeDetector(OIRegime.LONG_BUILDUP)  # type: ignore[assignment]
    analyzer.anomaly_detector = StubAnomalyDetector(detected=True, anomaly_type=OIAnomalyType.OI_SPIKE)  # type: ignore[assignment]

    base_ts = 1_000.0

    monkeypatch.setattr(analyzer, "_now", lambda: base_ts)
    await analyzer.on_candle(
        event(
            OIMarketEventType.CANDLE.topic,
            candle_payload(timestamp=base_ts, close=30_000.0, volume=1_000.0),
        )
    )

    monkeypatch.setattr(analyzer, "_now", lambda: base_ts + 11.0)
    await analyzer.on_open_interest(
        event(
            OIMarketEventType.OPEN_INTEREST.topic,
            oi_payload(timestamp=base_ts + 11.0, oi=1_000.0),
        )
    )

    state = analyzer.get_state("binance", "btcusdt")
    assert state is not None
    assert state.last_context is not None
    assert state.last_snapshot is not None

    # Context exists in state, but _get_context_for_key treated it as stale.
    assert state.last_features is None
    assert state.last_analysis is None
    assert event_bus.emitted == []