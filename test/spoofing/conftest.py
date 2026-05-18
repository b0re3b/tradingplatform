# tests/analytics/spoofing/conftest.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from analytics.spoofing import (
    DetectorDecision,
    DetectorResult,
    OrderbookLevelSnapshot,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingConfig,
    SpoofingFeatures,
    SpoofingPattern,
    SpoofingSide,
    TrackedWall,
)


DEFAULT_TEST_EXCHANGE = "binance"
DEFAULT_TEST_SYMBOL = "BTCUSDT"
DEFAULT_TEST_MARKET_TYPE = "perpetual"
DEFAULT_TEST_TIMEFRAME = "realtime"


# =============================================================================
# Fake core infrastructure
# =============================================================================


@dataclass(slots=True)
class EmittedEventRecord:
    """
    Запис події, яку тестовий EventBus отримав через emit().

    Дає можливість перевіряти:
    - topic;
    - payload;
    - priority;
    - source;
    - correlation_id;
    - headers/metadata;
    - додаткові kwargs.
    """

    topic: str
    payload: dict[str, Any]
    priority: Any = None
    source: str | None = None
    correlation_id: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubscriptionRecord:
    """
    Запис підписки, яку тестовий EventBus отримав через subscribe().

    Fake також має unsubscribe(), бо production analyzer має вміти
    коректно знімати підписки під час lifecycle/stop tests.
    """

    topic: str
    handler: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def unsubscribe(self) -> None:
        self.active = False


class FakeEventBus:
    """
    Мінімальний fake EventBus під контракт core.event_bus.EventBus.

    Його ціль — не імітувати всю шину подій, а дати тестам можливість
    перевірити, що analytics.spoofing:
    - підписується на правильні production topics;
    - не підписується на raw legacy topics без явного дозволу;
    - публікує правильні analytics.spoofing.* events;
    - передає correlation_id/source/headers;
    - коректно працює з unsubscribe lifecycle.
    """

    def __init__(self) -> None:
        self.subscriptions: list[SubscriptionRecord] = []
        self.emitted: list[EmittedEventRecord] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        **kwargs: Any,
    ) -> SubscriptionRecord:
        subscription = SubscriptionRecord(
            topic=topic,
            handler=handler,
            kwargs=kwargs,
        )
        self.subscriptions.append(subscription)
        return subscription

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: Any = None,
        source: str | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        self.emitted.append(
            EmittedEventRecord(
                topic=topic,
                payload=payload,
                priority=priority,
                source=source,
                correlation_id=correlation_id,
                headers=headers or {},
                kwargs=kwargs,
            )
        )
        return True

    def topics(self) -> list[str]:
        return [item.topic for item in self.emitted]

    def subscribed_topics(self, *, active_only: bool = False) -> list[str]:
        if active_only:
            return [item.topic for item in self.subscriptions if item.active]
        return [item.topic for item in self.subscriptions]

    def last_event(self) -> EmittedEventRecord:
        if not self.emitted:
            raise AssertionError("FakeEventBus has no emitted events")
        return self.emitted[-1]

    def events_for_topic(self, topic: str) -> list[EmittedEventRecord]:
        return [item for item in self.emitted if item.topic == topic]


@dataclass(slots=True)
class IntervalJobRecord:
    """
    Запис interval job, яку SpoofingAnalyzer реєструє через Scheduler.
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
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeScheduler:
    """
    Мінімальний fake Scheduler під contract core.scheduler.Scheduler.

    Потрібен для перевірки, що SpoofingAnalyzer реєструє cleanup job
    через add_interval_job(), а не запускає власний uncontrolled asyncio loop.
    """

    def __init__(self) -> None:
        self.interval_jobs: list[IntervalJobRecord] = []

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
        **kwargs: Any,
    ) -> str:
        job_id = f"test-job-{len(self.interval_jobs) + 1}"

        self.interval_jobs.append(
            IntervalJobRecord(
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
                kwargs=kwargs,
            )
        )

        return job_id

    def get_job_by_name(self, name: str) -> IntervalJobRecord | None:
        for job in self.interval_jobs:
            if job.name == name:
                return job
        return None

    def get_job(self, job_id: str) -> IntervalJobRecord | None:
        for job in self.interval_jobs:
            if job.job_id == job_id:
                return job
        return None

    def enable_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        job.enabled = True
        return True

    def disable_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            return False
        job.enabled = False
        return True

    def remove_job(self, job_id: str) -> bool:
        before = len(self.interval_jobs)
        self.interval_jobs = [
            job for job in self.interval_jobs if job.job_id != job_id
        ]
        return len(self.interval_jobs) != before


class FakeEvent:
    """
    Простий fake Event під analyzer callback.

    Дає payload і кілька можливих місць для correlation_id, бо analyzer може
    діставати correlation_id з event_id, correlation_id, metadata або headers.
    """

    def __init__(
        self,
        payload: Any,
        *,
        event_id: str | None = "test-event-id",
        correlation_id: str | None = "test-correlation-id",
        topic: str = "market.orderbook.updated",
        source: str = "tests",
        metadata: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> None:
        self.payload = payload
        self.topic = topic
        self.source = source
        self.event_id = event_id
        self.correlation_id = correlation_id
        self.metadata = metadata or {}
        self.headers = headers or {}

        if correlation_id is not None:
            self.metadata.setdefault("correlation_id", correlation_id)
            self.headers.setdefault("correlation_id", correlation_id)


# =============================================================================
# Unsafe/poison snapshot для adversarial tests
# =============================================================================


@dataclass(slots=True)
class UnsafeOrderbookLevelSnapshot:
    """
    Snapshot-like object для poisoned tests.

    Реальний OrderbookLevelSnapshot валідуює price/size у __post_init__ і
    кидає ValueError для price <= 0. Частина adversarial tests спеціально
    передає такі рівні в analyzer/detector, щоб перевірити фільтрацію.
    Тому для невалідних значень factory повертає цей lightweight object.
    """

    symbol: str
    exchange: str
    side: SpoofingSide
    price: float
    size: float
    market_type: str = DEFAULT_TEST_MARKET_TYPE
    timeframe: str = DEFAULT_TEST_TIMEFRAME
    exchange_symbol: str | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    sequence_id: int | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            str(self.exchange).strip().lower(),
            str(self.market_type or DEFAULT_TEST_MARKET_TYPE).strip().lower(),
            str(self.symbol).strip().upper(),
            str(self.timeframe or DEFAULT_TEST_TIMEFRAME).strip(),
        )

    @property
    def scope(self) -> dict[str, str]:
        exchange, market_type, symbol, timeframe = self.key
        return {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange_symbol": self.exchange_symbol or symbol,
        }

    @property
    def level_key(self) -> str:
        exchange, market_type, symbol, timeframe = self.key
        side = self.side.value if isinstance(self.side, SpoofingSide) else str(self.side)
        return f"{exchange}:{market_type}:{symbol}:{timeframe}:{side}:{self.price:.12f}"


# =============================================================================
# Core fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def mock_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def spoofing_config() -> SpoofingConfig:
    """
    Базовий тестовий SpoofingConfig.

    Налаштування достатньо permissive, щоб tests могли легко створювати
    positive detector cases без нереалістично великих synthetic orderbook values.

    Водночас config уже key-first:
        exchange + market_type + symbol + timeframe
    """

    config = SpoofingConfig()

    config.enabled = True

    # Scoped defaults for futures-first tests.
    config.default_exchange = DEFAULT_TEST_EXCHANGE
    config.default_market_type = DEFAULT_TEST_MARKET_TYPE
    config.default_timeframe = DEFAULT_TEST_TIMEFRAME

    # Legacy-compatible aliases.
    config.exchange = DEFAULT_TEST_EXCHANGE
    config.symbols = []

    # Scoped allowlists disabled by default.
    config.exchange_allowlist = None
    config.market_type_allowlist = None
    config.symbol_allowlist = None
    config.timeframe_allowlist = None

    # Wall detection.
    config.wall_detection.enabled = True
    config.wall_detection.min_wall_size_abs = 10_000.0
    config.wall_detection.min_wall_size_ratio = 3.0
    config.wall_detection.max_distance_from_mid_bps = 100.0
    config.wall_detection.near_best_quote_bps = 10.0
    config.wall_detection.min_levels_to_scan = 3
    config.wall_detection.max_levels_to_scan = 50

    # Persistence.
    config.persistence.enabled = True
    config.persistence.wall_ttl_ms = 15_000
    config.persistence.min_tracking_lifetime_ms = 0
    config.persistence.cleanup_interval_ms = 2_000
    config.persistence.max_walls_per_key = 500
    config.persistence.max_walls_per_symbol = 500
    config.persistence.max_history_events_per_level = 200
    config.persistence.size_update_epsilon = 1e-9
    config.persistence.price_rounding_decimals = 8

    # Pull detection.
    config.pull_detection.enabled = True
    config.pull_detection.max_pull_lifetime_ms = 2_500
    config.pull_detection.min_pull_ratio = 0.60
    config.pull_detection.max_fill_ratio_for_pull = 0.25
    config.pull_detection.min_removed_notional = 10_000.0
    config.pull_detection.fast_pull_lifetime_ms = 750
    config.pull_detection.strong_pull_ratio = 0.85

    # Fake liquidity.
    config.fake_liquidity.enabled = True
    config.fake_liquidity.max_fill_ratio = 0.20
    config.fake_liquidity.min_pull_ratio = 0.70
    config.fake_liquidity.max_lifetime_ms = 4_000
    config.fake_liquidity.min_price_reaction_bps = 2.0

    # Layering.
    config.layering.enabled = True
    config.layering.min_layers = 3
    config.layering.max_price_gap_bps_between_layers = 20.0
    config.layering.min_total_layer_notional = 10_000.0
    config.layering.synchronized_pull_window_ms = 1_000

    # Flip pressure.
    config.flip_pressure.enabled = True
    config.flip_pressure.min_price_reaction_bps = 3.0
    config.flip_pressure.reaction_window_ms = 3_000
    config.flip_pressure.min_pressure_flip_strength = 0.60

    # Scoring.
    config.scoring.enabled = True
    config.scoring.detection_threshold = 0.50
    config.scoring.high_severity_threshold = 0.75
    config.scoring.critical_severity_threshold = 0.90
    config.scoring.min_confidence = 0.30
    config.scoring.confidence_boost_on_detector_agreement = 0.10
    config.scoring.max_confidence = 0.99

    # Analyzer.
    config.analyzer.enabled = True
    config.analyzer.publish_updates = True
    config.analyzer.publish_detected_only = False
    config.analyzer.publish_lifecycle_events = True
    config.analyzer.publish_score_updates = True
    config.analyzer.publish_errors = True

    config.analyzer.max_tracked_walls_per_key = 500
    config.analyzer.max_tracked_walls_per_symbol = 500
    config.analyzer.max_detector_results_per_cycle = 50

    # Production topics only by default.
    config.analyzer.source_topic_patterns_orderbook = ("market.orderbook.updated",)
    config.analyzer.source_topic_patterns_trade = ("market.trades.updated",)
    config.analyzer.event_topic_orderbook = "market.orderbook.updated"
    config.analyzer.event_topic_trade = "market.trades.updated"
    config.analyzer.legacy_raw_orderbook_topic = "market.orderbook"
    config.analyzer.legacy_raw_trade_topic = "market.trade"
    config.analyzer.allow_legacy_raw_topics = False

    config.analyzer.event_topic_lifecycle = "analytics.spoofing.lifecycle"
    config.analyzer.event_topic_updated = "analytics.spoofing.updated"
    config.analyzer.event_topic_detected = "analytics.spoofing.detected"
    config.analyzer.event_topic_score_updated = "analytics.spoofing.score_updated"
    config.analyzer.event_topic_error = "analytics.spoofing.error"

    config.analyzer.scheduler_cleanup_enabled = True
    config.analyzer.scheduler_cleanup_job_name = "analytics.spoofing.persistence_cleanup"

    config.validate()
    return config


# =============================================================================
# Time fixtures / helpers
# =============================================================================


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def utc_dt_factory(fixed_now: datetime):
    def factory(*, milliseconds_ago: float = 0.0) -> datetime:
        return fixed_now - timedelta(milliseconds=milliseconds_ago)

    return factory


# =============================================================================
# Domain factories
# =============================================================================


@pytest.fixture
def orderbook_snapshot_factory(fixed_now: datetime):
    """
    Factory для normalized OrderbookLevelSnapshot.

    Для валідних values повертає реальний OrderbookLevelSnapshot.
    Для poisoned values повертає UnsafeOrderbookLevelSnapshot, щоб тести могли
    перевіряти фільтрацію analyzer/detector без падіння в dataclass constructor.
    """

    def factory(
        *,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        side: SpoofingSide = SpoofingSide.BID,
        price: float = 100.0,
        size: float = 100.0,
        best_bid: float | None = 99.9,
        best_ask: float | None = 100.1,
        mid_price: float | None = 100.0,
        spread: float | None = 0.2,
        sequence_id: int | None = 1,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderbookLevelSnapshot | UnsafeOrderbookLevelSnapshot:
        kwargs = {
            "symbol": symbol,
            "exchange": exchange,
            "market_type": market_type,
            "timeframe": timeframe,
            "exchange_symbol": exchange_symbol,
            "side": side,
            "price": price,
            "size": size,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "sequence_id": sequence_id,
            "timestamp": timestamp or fixed_now,
            "metadata": metadata or {},
        }

        try:
            return OrderbookLevelSnapshot(**kwargs)
        except (TypeError, ValueError):
            return UnsafeOrderbookLevelSnapshot(**kwargs)

    return factory


@pytest.fixture
def tracked_wall_factory(fixed_now: datetime):
    """
    Factory для TrackedWall.

    Дає змогу точно контролювати:
    - scope: exchange + market_type + symbol + timeframe;
    - lifetime_ms;
    - pull_ratio/fill_ratio через estimated sizes;
    - side/price/state;
    - repetition/update counters.
    """

    def factory(
        *,
        wall_id: str | None = None,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        side: SpoofingSide = SpoofingSide.BID,
        price: float = 100.0,
        lifetime_ms: float = 500.0,
        initial_size: float = 1000.0,
        current_size: float = 100.0,
        max_size: float = 1000.0,
        min_size: float = 100.0,
        best_bid_at_creation: float | None = 99.9,
        best_ask_at_creation: float | None = 100.1,
        mid_price_at_creation: float | None = 100.0,
        total_added_size: float = 0.0,
        total_removed_size: float = 900.0,
        estimated_filled_size: float = 50.0,
        estimated_pulled_size: float = 900.0,
        updates_count: int = 3,
        touch_count: int = 0,
        near_touch_count: int = 1,
        state: OrderbookWallState = OrderbookWallState.PULLED,
        metadata: dict[str, Any] | None = None,
    ) -> TrackedWall:
        first_seen_at = fixed_now - timedelta(milliseconds=lifetime_ms)
        normalized_exchange = str(exchange).strip().lower()
        normalized_market_type = str(market_type or DEFAULT_TEST_MARKET_TYPE).strip().lower()
        normalized_symbol = str(symbol).strip().upper()
        normalized_timeframe = str(timeframe or DEFAULT_TEST_TIMEFRAME).strip()
        normalized_side = side.value if isinstance(side, SpoofingSide) else str(side)

        resolved_wall_id = wall_id or (
            f"{normalized_exchange}:{normalized_market_type}:"
            f"{normalized_symbol}:{normalized_timeframe}:"
            f"{normalized_side}:{price:.8f}"
        )

        return TrackedWall(
            wall_id=resolved_wall_id,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            side=side,
            price=price,
            first_seen_at=first_seen_at,
            last_seen_at=fixed_now,
            initial_size=initial_size,
            current_size=current_size,
            max_size=max_size,
            min_size=min_size,
            best_bid_at_creation=best_bid_at_creation,
            best_ask_at_creation=best_ask_at_creation,
            mid_price_at_creation=mid_price_at_creation,
            total_added_size=total_added_size,
            total_removed_size=total_removed_size,
            estimated_filled_size=estimated_filled_size,
            estimated_pulled_size=estimated_pulled_size,
            updates_count=updates_count,
            touch_count=touch_count,
            near_touch_count=near_touch_count,
            state=state,
            metadata=metadata or {},
        )

    return factory


@pytest.fixture
def spoofing_features_factory(fixed_now: datetime):
    """
    Factory для SpoofingFeatures.

    Використовується scoring tests і detector_result_factory.
    """

    def factory(
        *,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        side: SpoofingSide = SpoofingSide.BID,
        price: float = 100.0,
        wall_size: float = 1000.0,
        wall_size_ratio: float = 8.0,
        distance_from_mid_bps: float = 5.0,
        lifetime_ms: float = 500.0,
        updates_count: int = 3,
        repetition_count: int = 2,
        fill_ratio: float = 0.05,
        pull_ratio: float = 0.90,
        cancel_to_fill_ratio: float | None = None,
        price_reaction_bps: float = 5.0,
        pressure_flip_strength: float = 0.0,
        layering_score: float = 0.0,
        is_near_best_quote: bool = True,
        is_fast_pull: bool = True,
        is_fake_liquidity: bool = True,
        is_layering: bool = False,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpoofingFeatures:
        resolved_cancel_to_fill_ratio = (
            cancel_to_fill_ratio
            if cancel_to_fill_ratio is not None
            else pull_ratio / max(fill_ratio, 1e-9)
        )

        return SpoofingFeatures(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            side=side,
            price=price,
            wall_size=wall_size,
            wall_size_ratio=wall_size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=lifetime_ms,
            updates_count=updates_count,
            repetition_count=repetition_count,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            cancel_to_fill_ratio=resolved_cancel_to_fill_ratio,
            price_reaction_bps=price_reaction_bps,
            pressure_flip_strength=pressure_flip_strength,
            layering_score=layering_score,
            is_near_best_quote=is_near_best_quote,
            is_fast_pull=is_fast_pull,
            is_fake_liquidity=is_fake_liquidity,
            is_layering=is_layering,
            timestamp=timestamp or fixed_now,
            metadata=metadata or {},
        )

    return factory


@pytest.fixture
def detector_result_factory(spoofing_features_factory, fixed_now: datetime):
    """
    Factory для DetectorResult.

    detector за замовчуванням НЕ None, бо SpoofingScoreEngine при merge
    читає result.detector.value.
    """

    def factory(
        *,
        detector: SpoofingComponent = SpoofingComponent.ORDER_PULL_DETECTOR,
        decision: DetectorDecision = DetectorDecision.POSITIVE,
        score: float = 0.85,
        confidence: float = 0.80,
        reason: str = "test detector result",
        features: SpoofingFeatures | None = None,
        wall_id: str | None = "wall-1",
        pattern: SpoofingPattern = SpoofingPattern.PULL_AND_REVERSAL,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        **feature_overrides: Any,
    ) -> DetectorResult:
        resolved_features = (
            features
            if features is not None
            else spoofing_features_factory(**feature_overrides)
        )

        return DetectorResult(
            detector=detector,
            decision=decision,
            score=score,
            confidence=confidence,
            reason=reason,
            features=resolved_features,
            wall_id=wall_id,
            pattern=pattern,
            timestamp=timestamp or fixed_now,
            metadata=metadata or {},
        )

    return factory


# =============================================================================
# Scenario factories
# =============================================================================


@pytest.fixture
def raw_orderbook_payload_factory():
    """
    Factory для legacy/raw market.orderbook-like payload.

    Залишено для наявних tests, але payload уже містить market_type/timeframe,
    щоб не втрачати futures scope.
    """

    def factory(
        *,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        bids: list[tuple[float, float]] | None = None,
        asks: list[tuple[float, float]] | None = None,
        best_bid: float = 99.9,
        best_ask: float = 100.1,
        sequence_id: int = 1,
        timestamp_ms: int = 1_767_268_800_000,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "exchange": exchange,
            "market_type": market_type,
            "timeframe": timeframe,
            "exchange_symbol": exchange_symbol or symbol,
            "bids": bids
            if bids is not None
            else [
                (99.9, 1.0),
                (99.8, 1.2),
                (99.7, 1000.0),
            ],
            "asks": asks
            if asks is not None
            else [
                (100.1, 1.0),
                (100.2, 1.1),
                (100.3, 1.3),
            ],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "sequence_id": sequence_id,
            "timestamp_ms": timestamp_ms,
            "metadata": metadata or {},
        }

    return factory


@pytest.fixture
def orderbook_updated_payload_factory(raw_orderbook_payload_factory):
    """
    Factory для production data-layer payload:
        OrderBookCache -> market.orderbook.updated -> SpoofingAnalyzer

    Це не змінює структуру tests, але дає готовий helper для нових
    adversarial cases у наявних файлах.
    """

    def factory(**kwargs: Any) -> dict[str, Any]:
        payload = raw_orderbook_payload_factory(**kwargs)
        return {
            **payload,
            "topic_contract": "market.orderbook.updated",
            "source": "data.orderbook_cache",
            "book": {
                "bids": payload["bids"],
                "asks": payload["asks"],
                "best_bid": payload["best_bid"],
                "best_ask": payload["best_ask"],
                "sequence_id": payload["sequence_id"],
            },
        }

    return factory


@pytest.fixture
def wall_snapshot_set_factory(orderbook_snapshot_factory):
    """
    Factory для набору snapshots із одним явним великим wall-рівнем.

    Корисно для analyzer pipeline та OrderbookWallDetector tests.
    """

    def factory(
        *,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        side: SpoofingSide = SpoofingSide.BID,
        wall_price: float = 99.7,
        wall_size: float = 1000.0,
        normal_size: float = 1.0,
        mid_price: float = 100.0,
    ) -> list[OrderbookLevelSnapshot | UnsafeOrderbookLevelSnapshot]:
        return [
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=side,
                price=99.9,
                size=normal_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=side,
                price=99.8,
                size=normal_size * 1.1,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=side,
                price=wall_price,
                size=wall_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=SpoofingSide.ASK,
                price=100.1,
                size=normal_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=SpoofingSide.ASK,
                price=100.2,
                size=normal_size * 1.2,
                mid_price=mid_price,
            ),
        ]

    return factory


@pytest.fixture
def layering_walls_factory(tracked_wall_factory):
    """
    Factory для multi-level layering scenario.

    Створює кілька близьких BID/ASK walls одного scoped futures market.
    """

    def factory(
        *,
        symbol: str = DEFAULT_TEST_SYMBOL,
        exchange: str = DEFAULT_TEST_EXCHANGE,
        market_type: str = DEFAULT_TEST_MARKET_TYPE,
        timeframe: str = DEFAULT_TEST_TIMEFRAME,
        exchange_symbol: str | None = None,
        side: SpoofingSide = SpoofingSide.BID,
        base_price: float = 99.90,
        count: int = 3,
        step: float = 0.05,
        size: float = 1000.0,
        lifetime_ms: float = 500.0,
    ) -> list[TrackedWall]:
        walls: list[TrackedWall] = []

        for index in range(count):
            price = (
                base_price - index * step
                if side == SpoofingSide.BID
                else base_price + index * step
            )

            walls.append(
                tracked_wall_factory(
                    wall_id=(
                        f"{exchange}:{market_type}:{symbol}:"
                        f"{timeframe}:{side.value}:{price:.8f}"
                    ),
                    symbol=symbol,
                    exchange=exchange,
                    market_type=market_type,
                    timeframe=timeframe,
                    exchange_symbol=exchange_symbol,
                    side=side,
                    price=price,
                    initial_size=size,
                    current_size=size * 0.10,
                    max_size=size,
                    min_size=size * 0.10,
                    estimated_pulled_size=size * 0.90,
                    estimated_filled_size=size * 0.02,
                    lifetime_ms=lifetime_ms,
                    updates_count=3,
                    state=OrderbookWallState.PULLED,
                    metadata={"layer_index": index},
                )
            )

        return walls

    return factory


# =============================================================================
# Assert helpers
# =============================================================================


@pytest.fixture
def assert_event_emitted():
    def helper(
        event_bus: FakeEventBus,
        topic: str,
        *,
        min_count: int = 1,
    ) -> list[EmittedEventRecord]:
        events = event_bus.events_for_topic(topic)

        assert len(events) >= min_count, (
            f"Expected at least {min_count} event(s) for topic={topic!r}, "
            f"got {len(events)}. Emitted topics: {event_bus.topics()}"
        )

        return events

    return helper


@pytest.fixture
def assert_no_event_emitted():
    def helper(event_bus: FakeEventBus, topic: str) -> None:
        events = event_bus.events_for_topic(topic)

        assert not events, (
            f"Expected no events for topic={topic!r}, "
            f"got {len(events)}. Emitted topics: {event_bus.topics()}"
        )

    return helper


@pytest.fixture
def assert_positive_detector_result():
    def helper(result: DetectorResult | None) -> DetectorResult:
        assert result is not None
        assert result.decision == DetectorDecision.POSITIVE
        assert result.score > 0.0
        assert result.confidence > 0.0
        assert result.features is not None
        return result

    return helper