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
    - headers/metadata.
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
    """

    topic: str
    handler: Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeEventBus:
    """
    Мінімальний fake EventBus під контракт core.event_bus.EventBus.

    Його ціль — не імітувати всю шину подій, а дати тестам можливість
    перевірити, що analytics.spoofing:
    - підписується на правильні topics;
    - публікує правильні analytics.spoofing.* events;
    - передає correlation_id/source/headers.
    """

    def __init__(self) -> None:
        self.subscriptions: list[SubscriptionRecord] = []
        self.emitted: list[EmittedEventRecord] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        **kwargs: Any,
    ) -> None:
        self.subscriptions.append(
            SubscriptionRecord(
                topic=topic,
                handler=handler,
                kwargs=kwargs,
            )
        )

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
    через add_interval_job(), а не запускає власний loop.
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


class FakeEvent:
    """
    Простий fake Event під callback SpoofingAnalyzer.on_orderbook_event().

    Дає payload і кілька можливих місць для correlation_id, бо analyzer може
    діставати correlation_id з event attribute або metadata/headers залежно
    від реалізації helper-а.
    """

    def __init__(
        self,
        payload: Any,
        *,
        correlation_id: str | None = "test-correlation-id",
        topic: str = "market.orderbook",
        source: str = "tests",
        metadata: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
    ) -> None:
        self.payload = payload
        self.topic = topic
        self.source = source
        self.correlation_id = correlation_id
        self.metadata = metadata or {}
        self.headers = headers or {}

        if correlation_id is not None:
            self.metadata.setdefault("correlation_id", correlation_id)
            self.headers.setdefault("correlation_id", correlation_id)


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

    Налаштування зроблені досить permissive, щоб tests могли легко створювати
    positive detector cases без надмірно великих synthetic orderbook values.
    """

    config = SpoofingConfig()

    config.enabled = True
    config.exchange = None
    config.symbols = []

    # Wall detection
    config.wall_detection.enabled = True
    config.wall_detection.min_wall_size_abs = 10_000.0
    config.wall_detection.min_wall_size_ratio = 3.0
    config.wall_detection.max_distance_from_mid_bps = 100.0
    config.wall_detection.near_best_quote_bps = 10.0
    config.wall_detection.min_levels_to_scan = 3
    config.wall_detection.max_levels_to_scan = 50

    # Persistence
    config.persistence.enabled = True
    config.persistence.wall_ttl_ms = 15_000
    config.persistence.min_tracking_lifetime_ms = 0
    config.persistence.cleanup_interval_ms = 2_000
    config.persistence.max_walls_per_symbol = 500
    config.persistence.max_history_events_per_level = 200
    config.persistence.size_update_epsilon = 1e-9
    config.persistence.price_rounding_decimals = 8

    # Pull detection
    config.pull_detection.enabled = True
    config.pull_detection.max_pull_lifetime_ms = 2_500
    config.pull_detection.min_pull_ratio = 0.60
    config.pull_detection.max_fill_ratio_for_pull = 0.25
    config.pull_detection.min_removed_notional = 10_000.0
    config.pull_detection.fast_pull_lifetime_ms = 750
    config.pull_detection.strong_pull_ratio = 0.85

    # Fake liquidity
    config.fake_liquidity.enabled = True
    config.fake_liquidity.max_fill_ratio = 0.20
    config.fake_liquidity.min_pull_ratio = 0.70
    config.fake_liquidity.max_lifetime_ms = 4_000
    config.fake_liquidity.min_price_reaction_bps = 2.0

    # Layering
    config.layering.enabled = True
    config.layering.min_layers = 3
    config.layering.max_price_gap_bps_between_layers = 20.0
    config.layering.min_total_layer_notional = 10_000.0
    config.layering.synchronized_pull_window_ms = 1_000

    # Flip pressure
    config.flip_pressure.enabled = True
    config.flip_pressure.min_price_reaction_bps = 3.0
    config.flip_pressure.reaction_window_ms = 3_000
    config.flip_pressure.min_pressure_flip_strength = 0.60

    # Scoring
    config.scoring.enabled = True
    config.scoring.detection_threshold = 0.50
    config.scoring.high_severity_threshold = 0.75
    config.scoring.critical_severity_threshold = 0.90
    config.scoring.min_confidence = 0.30
    config.scoring.confidence_boost_on_detector_agreement = 0.10
    config.scoring.max_confidence = 0.99

    # Analyzer
    config.analyzer.enabled = True
    config.analyzer.publish_updates = True
    config.analyzer.publish_detected_only = False
    config.analyzer.publish_lifecycle_events = True
    config.analyzer.publish_score_updates = True
    config.analyzer.publish_errors = True
    config.analyzer.max_tracked_walls_per_symbol = 500
    config.analyzer.max_detector_results_per_cycle = 50
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

    Використовується analyzer/wall detector/persistence tests.
    """

    def factory(
        *,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
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
    ) -> OrderbookLevelSnapshot:
        return OrderbookLevelSnapshot(
            symbol=symbol,
            exchange=exchange,
            side=side,
            price=price,
            size=size,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            sequence_id=sequence_id,
            timestamp=timestamp or fixed_now,
            metadata=metadata or {},
        )

    return factory


@pytest.fixture
def tracked_wall_factory(fixed_now: datetime):
    """
    Factory для TrackedWall.

    Дає змогу точно контролювати:
    - lifetime_ms;
    - pull_ratio;
    - fill_ratio;
    - side/price/state;
    - repetition/update count.
    """

    def factory(
        *,
        wall_id: str | None = None,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
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
        resolved_wall_id = wall_id or f"{exchange}:{symbol}:{side.value}:{price:.8f}"

        return TrackedWall(
            wall_id=resolved_wall_id,
            symbol=symbol,
            exchange=exchange,
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
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
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
    Factory для raw market.orderbook payload.

    Використовується тестами SpoofingAnalyzer.process_event_payload().
    """

    def factory(
        *,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        bids: list[tuple[float, float]] | None = None,
        asks: list[tuple[float, float]] | None = None,
        best_bid: float = 99.9,
        best_ask: float = 100.1,
        sequence_id: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "exchange": exchange,
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
            "metadata": metadata or {},
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
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        side: SpoofingSide = SpoofingSide.BID,
        wall_price: float = 99.7,
        wall_size: float = 1000.0,
        normal_size: float = 1.0,
        mid_price: float = 100.0,
    ) -> list[OrderbookLevelSnapshot]:
        return [
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                side=side,
                price=99.9,
                size=normal_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                side=side,
                price=99.8,
                size=normal_size * 1.1,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                side=side,
                price=wall_price,
                size=wall_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
                side=SpoofingSide.ASK,
                price=100.1,
                size=normal_size,
                mid_price=mid_price,
            ),
            orderbook_snapshot_factory(
                symbol=symbol,
                exchange=exchange,
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

    Створює кілька близьких BID/ASK walls одного symbol/exchange.
    """

    def factory(
        *,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
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
                    wall_id=f"{exchange}:{symbol}:{side.value}:{price:.8f}",
                    symbol=symbol,
                    exchange=exchange,
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