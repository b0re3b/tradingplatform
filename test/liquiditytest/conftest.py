# tests/analytics/liquidity/conftest.py

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from analytics.liquidity.config import LiquidityConfig
from analytics.liquidity.enums import (
    ClusterStrength,
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    LiquidityStatus,
    SweepStatus,
)
from analytics.liquidity.liquidity_map import LiquidityMap
from analytics.liquidity.liquidity_service import LiquidityService
from analytics.liquidity.models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquiditySignal,
    LiquidityZone,
    StopCluster,
)
from analytics.liquidity.scoring import LiquidityScorer
from analytics.liquidity.stop_clusters import StopClustersDetector
from analytics.liquidity.equal_highs_lows import EqualHighsLowsDetector


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


TEST_SYMBOL = "BTCUSDT"
TEST_TIMEFRAME = "1m"
BASE_TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------
# Fake core infrastructure
# ---------------------------------------------------------------------


@dataclass(slots=True)
class FakeSubscription:
    topic: str
    handler: Callable[..., Any]
    name: str | None = None
    active: bool = True


@dataclass(slots=True)
class PublishedEvent:
    topic: str
    payload: dict[str, Any]
    priority: Any = None
    source: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FakeEventBus:
    """
    Minimal fake EventBus для тестування LiquidityService.

    Підтримує:
    - subscribe()
    - unsubscribe()
    - emit()

    Зберігає всі subscriptions і emitted events, щоб тести могли
    перевіряти інтеграційну поведінку без запуску реального EventBus.
    """

    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.unsubscribed: list[FakeSubscription] = []
        self.published_events: list[PublishedEvent] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> FakeSubscription:
        subscription = FakeSubscription(
            topic=topic,
            handler=handler,
            name=name,
        )
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        subscription.active = False
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: Any = None,
        source: str | None = None,
        **_: Any,
    ) -> None:
        self.published_events.append(
            PublishedEvent(
                topic=topic,
                payload=dict(payload),
                priority=priority,
                source=source,
            )
        )

    def topics(self) -> list[str]:
        return [subscription.topic for subscription in self.subscriptions]

    def emitted_topics(self) -> list[str]:
        return [event.topic for event in self.published_events]

    def events_for(self, topic: str) -> list[PublishedEvent]:
        return [event for event in self.published_events if event.topic == topic]


@dataclass(slots=True)
class FakeScheduledJob:
    id: str
    name: str
    callback: Callable[..., Any]
    interval_seconds: float
    timeout: float | None = None
    max_retries: int = 0
    retry_delay: float = 0.0
    enabled: bool = True


class FakeScheduler:
    """
    Minimal fake Scheduler для перевірки registration/remove jobs.

    Підтримує обидві сигнатури:
    - add_interval_job(name=..., callback=..., interval_seconds=...)
    - add_interval_job(name=..., func=..., interval=...)

    Другий варіант використовує поточний LiquidityService.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, FakeScheduledJob] = {}
        self.removed_job_ids: list[str] = []

    def add_interval_job(
        self,
        *,
        name: str,
        func: Callable[..., Awaitable[Any]] | Callable[..., Any] | None = None,
        callback: Callable[..., Awaitable[Any]] | Callable[..., Any] | None = None,
        interval: float | None = None,
        interval_seconds: float | None = None,
        timeout: float | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
        run_immediately: bool = False,
        allow_overlap: bool = False,
        enabled: bool = True,
        **_: Any,
    ) -> str:
        resolved_callback = func or callback
        resolved_interval = interval if interval is not None else interval_seconds

        if resolved_callback is None:
            raise TypeError("FakeScheduler.add_interval_job requires func or callback")

        if resolved_interval is None:
            raise TypeError(
                "FakeScheduler.add_interval_job requires interval or interval_seconds"
            )

        job_id = f"job-{len(self.jobs) + 1}"

        self.jobs[job_id] = FakeScheduledJob(
            id=job_id,
            name=name,
            callback=resolved_callback,
            interval_seconds=float(resolved_interval),
            timeout=timeout,
            max_retries=int(max_retries),
            retry_delay=float(retry_delay),
            enabled=bool(enabled),
        )

        return job_id

    def remove_job(self, job_id: str) -> None:
        self.removed_job_ids.append(job_id)
        self.jobs.pop(job_id, None)

    def job_names(self) -> list[str]:
        return [job.name for job in self.jobs.values()]


@dataclass(slots=True)
class FakeMarketEvent:
    """
    Lightweight event object для direct handler tests.

    Має ті поля, які зазвичай потрібні service handlers:
    - topic
    - payload
    - timestamp
    - source
    """

    topic: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "tests"


# ---------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def symbol() -> str:
    return TEST_SYMBOL


@pytest.fixture
def timeframe() -> str:
    return TEST_TIMEFRAME


@pytest.fixture
def base_ts() -> datetime:
    return BASE_TS


@pytest.fixture
def liquidity_config() -> LiquidityConfig:
    """
    Deterministic test config.

    Важливо:
    - use_atr_tolerance=False, щоб tests не залежали від volatility;
    - snapshot_rebuild_min_interval_seconds=0.0, щоб service-тести
      не блокувались throttle/debounce;
    - min_candles_for_snapshot низький для компактних fixtures.
    """

    return LiquidityConfig(
        enabled=True,
        pivot_lookback=2,
        pivot_lookforward=2,
        min_swing_distance_pct=0.0010,
        equal_level_tolerance_pct=0.0012,
        min_equal_touches=2,
        max_equal_cluster_width_pct=0.0020,
        stop_cluster_padding_pct=0.0010,
        cluster_merge_distance_pct=0.0010,
        max_active_levels=50,
        max_active_clusters=25,
        level_expiry_bars=300,
        min_confidence=0.0,
        use_atr_tolerance=False,
        atr_period=14,
        atr_tolerance_multiplier=0.15,
        min_atr_tolerance_pct=0.0003,
        max_atr_tolerance_pct=0.0030,
        use_volume_in_scoring=True,
        use_reaction_strength_in_scoring=True,
        use_orderbook_in_stop_clusters=True,
        use_time_decay=False,
        use_partial_sweep_penalty=True,
        max_candles_per_context=100,
        min_candles_for_snapshot=10,
        max_contexts=100,
        snapshot_rebuild_min_interval_seconds=0.0,
        rebuild_on_orderbook_updates=True,
        rebuild_on_price_updates=True,
        publish_events=True,
        emit_map_updates=True,
        emit_level_events=True,
        emit_cluster_events=True,
        emit_sweep_events=True,
        emit_signal_events=True,
        emit_state_metrics=True,
        cleanup_enabled=True,
        cleanup_interval_seconds=60.0,
        state_metrics_interval_seconds=30.0,
        healthcheck_interval_seconds=30.0,
        scheduler_job_timeout_seconds=5.0,
        scheduler_job_max_retries=1,
        scheduler_job_retry_delay_seconds=1.0,
        incremental_mode=True,
    )


@pytest.fixture
def disabled_liquidity_config(liquidity_config: LiquidityConfig) -> LiquidityConfig:
    liquidity_config.enabled = False
    return liquidity_config


@pytest.fixture
def scorer(liquidity_config: LiquidityConfig) -> LiquidityScorer:
    return LiquidityScorer(config=liquidity_config)


@pytest.fixture
def equal_detector(
    liquidity_config: LiquidityConfig,
    scorer: LiquidityScorer,
) -> EqualHighsLowsDetector:
    return EqualHighsLowsDetector(
        config=liquidity_config,
        scorer=scorer,
    )


@pytest.fixture
def stop_detector(
    liquidity_config: LiquidityConfig,
    scorer: LiquidityScorer,
) -> StopClustersDetector:
    return StopClustersDetector(
        config=liquidity_config,
        scorer=scorer,
    )


@pytest.fixture
def liquidity_map(
    liquidity_config: LiquidityConfig,
    scorer: LiquidityScorer,
    equal_detector: EqualHighsLowsDetector,
    stop_detector: StopClustersDetector,
) -> LiquidityMap:
    return LiquidityMap(
        config=liquidity_config,
        equal_detector=equal_detector,
        stop_detector=stop_detector,
        scorer=scorer,
    )


@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def liquidity_service(
    fake_event_bus: FakeEventBus,
    fake_scheduler: FakeScheduler,
    liquidity_config: LiquidityConfig,
    liquidity_map: LiquidityMap,
) -> LiquidityService:
    return LiquidityService(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
        config=liquidity_config,
        liquidity_map=liquidity_map,
    )


@pytest.fixture
def liquidity_service_without_scheduler(
    fake_event_bus: FakeEventBus,
    liquidity_config: LiquidityConfig,
    liquidity_map: LiquidityMap,
) -> LiquidityService:
    return LiquidityService(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=None,
        config=liquidity_config,
        liquidity_map=liquidity_map,
    )


# ---------------------------------------------------------------------
# Candle factories
# ---------------------------------------------------------------------


def make_candle(
    *,
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    base_ts: datetime = BASE_TS,
) -> dict[str, Any]:
    open_time = base_ts + timedelta(minutes=index)
    close_time = open_time + timedelta(minutes=1)

    return {
        "symbol": TEST_SYMBOL,
        "timeframe": TEST_TIMEFRAME,
        "open_time": open_time,
        "close_time": close_time,
        "timestamp": close_time,
        "open": float(open_),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


def make_candles_from_ohlc(
    rows: Sequence[tuple[float, float, float, float]],
    *,
    volume: float = 100.0,
    base_ts: datetime = BASE_TS,
) -> list[dict[str, Any]]:
    return [
        make_candle(
            index=index,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            base_ts=base_ts,
        )
        for index, (open_, high, low, close) in enumerate(rows)
    ]


@pytest.fixture
def candles_with_equal_highs() -> list[dict[str, Any]]:
    """
    Дані з трьома очевидними pivot highs біля 105.

    Підходить для:
    - EqualHighsLowsDetector;
    - LiquidityMap.build_snapshot();
    - service candle event tests.
    """

    return make_candles_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 102.0, 100.0, 101.5),
            (101.5, 103.0, 100.8, 102.4),
            (102.4, 105.00, 101.8, 103.0),  # pivot high
            (103.0, 103.6, 101.5, 102.0),
            (102.0, 102.5, 100.5, 101.2),
            (101.2, 103.5, 100.8, 102.8),
            (102.8, 105.05, 101.9, 103.1),  # pivot high
            (103.1, 103.8, 101.2, 102.1),
            (102.1, 102.7, 100.9, 101.6),
            (101.6, 103.4, 100.7, 102.6),
            (102.6, 104.96, 101.8, 103.0),  # pivot high
            (103.0, 103.5, 101.4, 102.2),
            (102.2, 102.6, 100.6, 101.1),
            (101.1, 101.9, 99.8, 100.4),
        ],
        volume=120.0,
    )


@pytest.fixture
def candles_with_equal_lows() -> list[dict[str, Any]]:
    """
    Дані з трьома очевидними pivot lows біля 95.
    """

    return make_candles_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.3),
            (100.3, 100.8, 98.0, 99.1),
            (99.1, 100.0, 96.4, 97.2),
            (97.2, 98.1, 95.00, 96.3),  # pivot low
            (96.3, 99.0, 96.1, 98.0),
            (98.0, 100.2, 97.4, 99.5),
            (99.5, 100.0, 96.6, 97.0),
            (97.0, 98.0, 95.04, 96.2),  # pivot low
            (96.2, 99.2, 96.0, 98.4),
            (98.4, 100.5, 97.6, 99.8),
            (99.8, 100.1, 96.5, 97.2),
            (97.2, 98.2, 94.96, 96.1),  # pivot low
            (96.1, 98.9, 95.8, 98.2),
            (98.2, 100.6, 97.8, 99.7),
            (99.7, 101.0, 98.9, 100.4),
        ],
        volume=110.0,
    )


@pytest.fixture
def candles_with_both_sides() -> list[dict[str, Any]]:
    """
    Дані з equal highs біля 105 і equal lows біля 95.
    Корисно для LiquidityMap pressure/bias tests.
    """

    return make_candles_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 102.0, 98.5, 101.2),
            (101.2, 103.0, 97.0, 102.0),
            (102.0, 105.00, 95.20, 100.0),  # high + low pivot region
            (100.0, 102.0, 96.5, 98.5),
            (98.5, 101.0, 95.00, 99.3),     # pivot low
            (99.3, 103.0, 97.0, 102.5),
            (102.5, 105.04, 96.8, 101.0),   # pivot high
            (101.0, 102.0, 95.06, 98.7),    # pivot low
            (98.7, 101.0, 96.7, 100.5),
            (100.5, 103.0, 97.5, 102.0),
            (102.0, 104.97, 95.02, 100.2),  # both-side region
            (100.2, 102.5, 96.4, 101.0),
            (101.0, 103.0, 98.0, 102.2),
            (102.2, 103.2, 99.0, 101.5),
        ],
        volume=130.0,
    )


@pytest.fixture
def candles_without_clear_equal_levels() -> list[dict[str, Any]]:
    """
    Trending/uneven candles без очевидних equal highs/lows.
    """

    rows: list[tuple[float, float, float, float]] = []

    price = 100.0
    for index in range(20):
        open_ = price
        high = price + 1.0 + index * 0.15
        low = price - 0.8
        close = price + 0.45
        rows.append((open_, high, low, close))
        price = close

    return make_candles_from_ohlc(rows, volume=90.0)


@pytest.fixture
def too_few_candles() -> list[dict[str, Any]]:
    return make_candles_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.5),
            (100.5, 102.0, 100.0, 101.5),
            (101.5, 103.0, 100.8, 102.4),
        ],
        volume=100.0,
    )


# ---------------------------------------------------------------------
# Orderbook fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def orderbook_near_buy_side_cluster() -> dict[str, list[list[float]]]:
    """
    Ask liquidity біля 105 підсилює buy-side stop cluster.
    """

    return {
        "bids": [
            [99.80, 8.0],
            [99.50, 10.0],
            [99.00, 6.5],
        ],
        "asks": [
            [104.90, 45.0],
            [105.00, 60.0],
            [105.10, 42.0],
        ],
    }


@pytest.fixture
def orderbook_near_sell_side_cluster() -> dict[str, list[list[float]]]:
    """
    Bid liquidity біля 95 підсилює sell-side stop cluster.
    """

    return {
        "bids": [
            [95.10, 42.0],
            [95.00, 58.0],
            [94.90, 40.0],
        ],
        "asks": [
            [100.50, 8.0],
            [101.00, 10.0],
            [101.50, 6.5],
        ],
    }


@pytest.fixture
def balanced_orderbook() -> dict[str, list[list[float]]]:
    return {
        "bids": [
            [99.90, 10.0],
            [99.70, 12.0],
            [99.50, 11.0],
        ],
        "asks": [
            [100.10, 10.0],
            [100.30, 12.0],
            [100.50, 11.0],
        ],
    }


# ---------------------------------------------------------------------
# Liquidity level factories
# ---------------------------------------------------------------------


def make_liquidity_level(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
    price: float,
    side: LiquiditySide,
    level_type: LiquidityLevelType | None = None,
    status: LiquidityStatus = LiquidityStatus.ACTIVE,
    sweep_status: SweepStatus = SweepStatus.NOT_SWEPT,
    confidence: float = 0.75,
    touches_count: int = 3,
    reaction_count: int = 2,
    source: str = "test",
    seen_at: datetime = BASE_TS,
    metadata: dict[str, Any] | None = None,
) -> LiquidityLevel:
    if level_type is None:
        level_type = (
            LiquidityLevelType.EQUAL_HIGHS
            if side == LiquiditySide.BUY_SIDE
            else LiquidityLevelType.EQUAL_LOWS
        )

    swept_at = seen_at if sweep_status in {SweepStatus.SWEPT, SweepStatus.PARTIALLY_SWEPT} else None
    invalidated_at = seen_at if status == LiquidityStatus.INVALIDATED else None
    expired_at = seen_at if status == LiquidityStatus.EXPIRED else None

    return LiquidityLevel(
        symbol=symbol,
        timeframe=timeframe,
        level_type=level_type,
        side=side,
        price=price,
        status=status,
        sweep_status=sweep_status,
        confidence=confidence,
        touches_count=touches_count,
        reaction_count=reaction_count,
        first_seen_at=seen_at - timedelta(minutes=20),
        last_seen_at=seen_at,
        swept_at=swept_at,
        invalidated_at=invalidated_at,
        expired_at=expired_at,
        source=source,
        metadata=dict(metadata or {}),
    )


def make_equal_level(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
    price: float,
    side: LiquiditySide,
    level_type: LiquidityLevelType | None = None,
    status: LiquidityStatus = LiquidityStatus.ACTIVE,
    sweep_status: SweepStatus = SweepStatus.NOT_SWEPT,
    confidence: float = 0.80,
    touches_count: int = 3,
    reaction_count: int = 2,
    tolerance_pct: float = 0.001,
    cluster_low: float | None = None,
    cluster_high: float | None = None,
    level_prices: list[float] | None = None,
    pivot_indexes: list[int] | None = None,
    seen_at: datetime = BASE_TS,
) -> EqualLevel:
    if level_type is None:
        level_type = (
            LiquidityLevelType.EQUAL_HIGHS
            if side == LiquiditySide.BUY_SIDE
            else LiquidityLevelType.EQUAL_LOWS
        )

    if cluster_low is None:
        cluster_low = price * (1.0 - tolerance_pct / 2.0)

    if cluster_high is None:
        cluster_high = price * (1.0 + tolerance_pct / 2.0)

    if level_prices is None:
        level_prices = [price, price * 1.0002, price * 0.9998]

    if pivot_indexes is None:
        pivot_indexes = [3, 7, 11]

    swept_at = seen_at if sweep_status in {SweepStatus.SWEPT, SweepStatus.PARTIALLY_SWEPT} else None

    return EqualLevel(
        symbol=symbol,
        timeframe=timeframe,
        level_type=level_type,
        side=side,
        price=price,
        status=status,
        sweep_status=sweep_status,
        confidence=confidence,
        touches_count=touches_count,
        reaction_count=reaction_count,
        first_seen_at=seen_at - timedelta(minutes=20),
        last_seen_at=seen_at,
        swept_at=swept_at,
        source="test_equal_level",
        tolerance_pct=tolerance_pct,
        cluster_low=cluster_low,
        cluster_high=cluster_high,
        level_prices=level_prices,
        pivot_indexes=pivot_indexes,
        metadata={"fixture": "equal_level"},
    )


@pytest.fixture
def buy_side_levels() -> list[LiquidityLevel]:
    return [
        make_liquidity_level(price=104.95, side=LiquiditySide.BUY_SIDE, confidence=0.72),
        make_liquidity_level(price=105.00, side=LiquiditySide.BUY_SIDE, confidence=0.80),
        make_liquidity_level(price=105.06, side=LiquiditySide.BUY_SIDE, confidence=0.76),
    ]


@pytest.fixture
def sell_side_levels() -> list[LiquidityLevel]:
    return [
        make_liquidity_level(price=95.06, side=LiquiditySide.SELL_SIDE, confidence=0.72),
        make_liquidity_level(price=95.00, side=LiquiditySide.SELL_SIDE, confidence=0.80),
        make_liquidity_level(price=94.95, side=LiquiditySide.SELL_SIDE, confidence=0.76),
    ]


@pytest.fixture
def mixed_side_levels(
    buy_side_levels: list[LiquidityLevel],
    sell_side_levels: list[LiquidityLevel],
) -> list[LiquidityLevel]:
    return [*buy_side_levels, *sell_side_levels]


@pytest.fixture
def swept_buy_side_level() -> LiquidityLevel:
    return make_liquidity_level(
        price=105.00,
        side=LiquiditySide.BUY_SIDE,
        status=LiquidityStatus.SWEPT,
        sweep_status=SweepStatus.SWEPT,
        confidence=0.65,
    )


@pytest.fixture
def partially_swept_buy_side_level() -> LiquidityLevel:
    return make_liquidity_level(
        price=105.00,
        side=LiquiditySide.BUY_SIDE,
        status=LiquidityStatus.ACTIVE,
        sweep_status=SweepStatus.PARTIALLY_SWEPT,
        confidence=0.70,
    )


@pytest.fixture
def invalidated_buy_side_level() -> LiquidityLevel:
    return make_liquidity_level(
        price=105.00,
        side=LiquiditySide.BUY_SIDE,
        status=LiquidityStatus.INVALIDATED,
        sweep_status=SweepStatus.NOT_SWEPT,
        confidence=0.70,
    )


@pytest.fixture
def expired_sell_side_level() -> LiquidityLevel:
    return make_liquidity_level(
        price=95.00,
        side=LiquiditySide.SELL_SIDE,
        status=LiquidityStatus.EXPIRED,
        sweep_status=SweepStatus.NOT_SWEPT,
        confidence=0.70,
    )


@pytest.fixture
def equal_high_level() -> EqualLevel:
    return make_equal_level(
        price=105.00,
        side=LiquiditySide.BUY_SIDE,
        level_type=LiquidityLevelType.EQUAL_HIGHS,
    )


@pytest.fixture
def equal_low_level() -> EqualLevel:
    return make_equal_level(
        price=95.00,
        side=LiquiditySide.SELL_SIDE,
        level_type=LiquidityLevelType.EQUAL_LOWS,
    )


# ---------------------------------------------------------------------
# Stop cluster / zone / signal / snapshot factories
# ---------------------------------------------------------------------


def make_stop_cluster(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
    side: LiquiditySide,
    low_price: float,
    high_price: float,
    confidence: float = 0.80,
    estimated_stop_density: float = 0.75,
    touches_count: int = 3,
    strength: ClusterStrength = ClusterStrength.MEDIUM,
    swept: bool = False,
    invalidated: bool = False,
    source_levels: list[LiquidityLevel] | None = None,
    metadata: dict[str, Any] | None = None,
) -> StopCluster:
    center_price = (float(low_price) + float(high_price)) / 2.0
    now = BASE_TS

    return StopCluster(
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        low_price=low_price,
        high_price=high_price,
        center_price=center_price,
        confidence=confidence,
        estimated_stop_density=estimated_stop_density,
        touches_count=touches_count,
        source_level_type=LiquidityLevelType.STOP_CLUSTER,
        strength=strength,
        created_at=now - timedelta(minutes=10),
        updated_at=now,
        invalidated_at=now if invalidated else None,
        swept_at=now if swept else None,
        source_levels=list(source_levels or []),
        metadata=dict(metadata or {}),
    )


@pytest.fixture
def buy_side_stop_cluster(buy_side_levels: list[LiquidityLevel]) -> StopCluster:
    return make_stop_cluster(
        side=LiquiditySide.BUY_SIDE,
        low_price=104.90,
        high_price=105.10,
        confidence=0.82,
        estimated_stop_density=0.78,
        touches_count=5,
        strength=ClusterStrength.HIGH,
        source_levels=buy_side_levels,
        metadata={"fixture": "buy_side_stop_cluster"},
    )


@pytest.fixture
def sell_side_stop_cluster(sell_side_levels: list[LiquidityLevel]) -> StopCluster:
    return make_stop_cluster(
        side=LiquiditySide.SELL_SIDE,
        low_price=94.90,
        high_price=95.10,
        confidence=0.78,
        estimated_stop_density=0.74,
        touches_count=4,
        strength=ClusterStrength.MEDIUM,
        source_levels=sell_side_levels,
        metadata={"fixture": "sell_side_stop_cluster"},
    )


@pytest.fixture
def swept_buy_side_stop_cluster(buy_side_levels: list[LiquidityLevel]) -> StopCluster:
    return make_stop_cluster(
        side=LiquiditySide.BUY_SIDE,
        low_price=104.90,
        high_price=105.10,
        confidence=0.65,
        estimated_stop_density=0.55,
        touches_count=5,
        swept=True,
        source_levels=buy_side_levels,
        metadata={"fixture": "swept_buy_side_stop_cluster"},
    )


def make_liquidity_zone(
    *,
    side: LiquiditySide,
    low_price: float,
    high_price: float,
    score: float = 0.75,
    label: str | None = None,
    source_types: list[LiquidityLevelType] | None = None,
) -> LiquidityZone:
    return LiquidityZone(
        symbol=TEST_SYMBOL,
        timeframe=TEST_TIMEFRAME,
        side=side,
        low_price=low_price,
        high_price=high_price,
        score=score,
        label=label,
        source_types=source_types or [LiquidityLevelType.STOP_CLUSTER],
        metadata={"fixture": "liquidity_zone"},
    )


@pytest.fixture
def buy_side_zone() -> LiquidityZone:
    return make_liquidity_zone(
        side=LiquiditySide.BUY_SIDE,
        low_price=104.90,
        high_price=105.10,
        score=0.80,
        label="buy_side_liquidity",
        source_types=[LiquidityLevelType.EQUAL_HIGHS, LiquidityLevelType.STOP_CLUSTER],
    )


@pytest.fixture
def sell_side_zone() -> LiquidityZone:
    return make_liquidity_zone(
        side=LiquiditySide.SELL_SIDE,
        low_price=94.90,
        high_price=95.10,
        score=0.75,
        label="sell_side_liquidity",
        source_types=[LiquidityLevelType.EQUAL_LOWS, LiquidityLevelType.STOP_CLUSTER],
    )


def make_liquidity_signal(
    *,
    bias: LiquidityBias = LiquidityBias.NEUTRAL,
    nearest_buy_side_liquidity: LiquidityLevel | StopCluster | None = None,
    nearest_sell_side_liquidity: LiquidityLevel | StopCluster | None = None,
    confidence: float = 0.70,
) -> LiquiditySignal:
    return LiquiditySignal(
        symbol=TEST_SYMBOL,
        timeframe=TEST_TIMEFRAME,
        timestamp=BASE_TS,
        bias=bias,
        nearest_buy_side_liquidity=nearest_buy_side_liquidity,
        nearest_sell_side_liquidity=nearest_sell_side_liquidity,
        sweep_risk_up=0.65,
        sweep_risk_down=0.45,
        magnet_score_up=0.70,
        magnet_score_down=0.40,
        confidence=confidence,
        explanation="test liquidity signal",
        metadata={"fixture": "liquidity_signal"},
    )


def make_snapshot(
    *,
    current_price: float = 100.0,
    active_levels: list[LiquidityLevel] | None = None,
    equal_levels: list[EqualLevel] | None = None,
    stop_clusters: list[StopCluster] | None = None,
    zones: list[LiquidityZone] | None = None,
    nearest_above_level: LiquidityLevel | StopCluster | None = None,
    nearest_below_level: LiquidityLevel | StopCluster | None = None,
    strongest_cluster_above: StopCluster | None = None,
    strongest_cluster_below: StopCluster | None = None,
    above_liquidity_score: float = 0.70,
    below_liquidity_score: float = 0.50,
    liquidity_pressure_score: float = 0.20,
    bias: LiquidityBias = LiquidityBias.UP,
    signal: LiquiditySignal | None = None,
    metadata: dict[str, Any] | None = None,
) -> LiquidityMapSnapshot:
    return LiquidityMapSnapshot(
        symbol=TEST_SYMBOL,
        timeframe=TEST_TIMEFRAME,
        timestamp=BASE_TS,
        current_price=current_price,
        active_levels=list(active_levels or []),
        equal_levels=list(equal_levels or []),
        stop_clusters=list(stop_clusters or []),
        zones=list(zones or []),
        nearest_above_level=nearest_above_level,
        nearest_below_level=nearest_below_level,
        strongest_cluster_above=strongest_cluster_above,
        strongest_cluster_below=strongest_cluster_below,
        above_liquidity_score=above_liquidity_score,
        below_liquidity_score=below_liquidity_score,
        liquidity_pressure_score=liquidity_pressure_score,
        bias=bias,
        signal=signal,
        metadata=dict(metadata or {"fixture": "snapshot"}),
    )


@pytest.fixture
def complete_snapshot(
    equal_high_level: EqualLevel,
    equal_low_level: EqualLevel,
    buy_side_stop_cluster: StopCluster,
    sell_side_stop_cluster: StopCluster,
    buy_side_zone: LiquidityZone,
    sell_side_zone: LiquidityZone,
) -> LiquidityMapSnapshot:
    signal = make_liquidity_signal(
        bias=LiquidityBias.UP,
        nearest_buy_side_liquidity=buy_side_stop_cluster,
        nearest_sell_side_liquidity=sell_side_stop_cluster,
        confidence=0.75,
    )

    return make_snapshot(
        current_price=100.0,
        active_levels=[equal_high_level, equal_low_level],
        equal_levels=[equal_high_level, equal_low_level],
        stop_clusters=[buy_side_stop_cluster, sell_side_stop_cluster],
        zones=[buy_side_zone, sell_side_zone],
        nearest_above_level=buy_side_stop_cluster,
        nearest_below_level=sell_side_stop_cluster,
        strongest_cluster_above=buy_side_stop_cluster,
        strongest_cluster_below=sell_side_stop_cluster,
        above_liquidity_score=0.75,
        below_liquidity_score=0.55,
        liquidity_pressure_score=0.20,
        bias=LiquidityBias.UP,
        signal=signal,
        metadata={"fixture": "complete_snapshot"},
    )


# ---------------------------------------------------------------------
# Event payload factories
# ---------------------------------------------------------------------


def make_market_candle_payload(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
    candle: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle": dict(candle),
    }


def make_market_orderbook_payload(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
    orderbook: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "orderbook": {
            "bids": list(orderbook.get("bids", [])),
            "asks": list(orderbook.get("asks", [])),
        },
    }


def make_market_price_payload(
    *,
    symbol: str = TEST_SYMBOL,
    timeframe: str | None = TEST_TIMEFRAME,
    price: float = 100.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "price": float(price),
    }

    if timeframe is not None:
        payload["timeframe"] = timeframe

    return payload


@pytest.fixture
def candle_event_factory() -> Callable[[dict[str, Any]], FakeMarketEvent]:
    def _factory(candle: dict[str, Any]) -> FakeMarketEvent:
        return FakeMarketEvent(
            topic="market.candle.closed",
            payload=make_market_candle_payload(candle=candle),
        )

    return _factory


@pytest.fixture
def orderbook_event_factory() -> Callable[[dict[str, Any]], FakeMarketEvent]:
    def _factory(orderbook: dict[str, Any]) -> FakeMarketEvent:
        return FakeMarketEvent(
            topic="market.orderbook.updated",
            payload=make_market_orderbook_payload(orderbook=orderbook),
        )

    return _factory


@pytest.fixture
def price_event_factory() -> Callable[[float], FakeMarketEvent]:
    def _factory(price: float) -> FakeMarketEvent:
        return FakeMarketEvent(
            topic="market.price.updated",
            payload=make_market_price_payload(price=price),
        )

    return _factory


@pytest.fixture
def invalid_market_event() -> FakeMarketEvent:
    return FakeMarketEvent(
        topic="market.candle.closed",
        payload={"bad": "payload"},
    )


# ---------------------------------------------------------------------
# Optional helper assertions
# ---------------------------------------------------------------------


def assert_score01(value: float) -> None:
    assert 0.0 <= value <= 1.0


def assert_signed_score(value: float) -> None:
    assert -1.0 <= value <= 1.0


@pytest.fixture
def score01_assertion() -> Callable[[float], None]:
    return assert_score01


@pytest.fixture
def signed_score_assertion() -> Callable[[float], None]:
    return assert_signed_score