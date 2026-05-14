from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest
import pytest_asyncio

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.whales.analyzer import WhaleAnalyzer
from analytics.whales.config import (
    LargeTradeDetectorConfig,
    WhaleClusterAnalyzerConfig,
    WhaleTrackerConfig,
    WhalesConfig,
)
from analytics.whales.large_trade_detector import LargeTradeDetector
from analytics.whales.whale_cluster_analyzer import WhaleClusterAnalyzer
from analytics.whales.whale_tracker import WhaleTracker


# =============================================================================
# Constants
# =============================================================================


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXCHANGE = "binance"

MARKET_TRADE_TOPIC = "market.trade"
MARKET_LIQUIDATION_TOPIC = "market.liquidation"

LARGE_TRADE_TOPIC = "analytics.whales.large_trade"
WHALE_ACTIVITY_TOPIC = "analytics.whales.whale_activity"
WHALE_PRESSURE_TOPIC = "analytics.whales.whale_pressure"
WHALE_LIQUIDATION_CONTEXT_TOPIC = "analytics.whales.whale_liquidation_context"
WHALE_CLUSTER_TOPIC = "analytics.whales.whale_cluster"
WHALE_CLUSTER_UPDATE_TOPIC = "analytics.whales.whale_cluster_update"
WHALE_CLUSTER_EXHAUSTION_TOPIC = "analytics.whales.whale_cluster_exhaustion"

ANALYTICS_WHALES_WILDCARD = "analytics.whales.*"


# =============================================================================
# Core object helpers
# =============================================================================


def _make_event_bus() -> EventBus:
    """
    Створює core EventBus для тестів.

    Якщо у твоєму core.EventBus пізніше з'явиться обов'язковий config,
    краще змінити тільки цей helper, а не всі тести.
    """
    return EventBus()


def _make_scheduler(event_bus: EventBus) -> Scheduler:
    """
    Створює core Scheduler для тестів.

    Підтримує обидва можливі варіанти constructor-а:
    - Scheduler(event_bus=event_bus)
    - Scheduler()
    """
    try:
        return Scheduler(event_bus=event_bus)
    except TypeError:
        return Scheduler()


async def _maybe_start(obj: Any) -> None:
    start = getattr(obj, "start", None)
    if start is None:
        return

    result = start()
    if isinstance(result, Awaitable):
        await result


async def _maybe_stop(obj: Any) -> None:
    stop = getattr(obj, "stop", None)
    if stop is None:
        return

    result = stop()
    if isinstance(result, Awaitable):
        await result


# =============================================================================
# Event collector
# =============================================================================


class EventCollector:
    """
    Збирає EventBus події для integration-тестів.

    Використання:
        collector.subscribe("analytics.whales.*")
        await event_bus.emit("market.trade", payload)
        event = await collector.wait_for_topic("analytics.whales.large_trade")
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.events: list[Event] = []
        self._subscriptions: list[Any] = []
        self._condition = asyncio.Condition()

    def subscribe(self, topic_pattern: str = ANALYTICS_WHALES_WILDCARD) -> None:
        subscription = self.event_bus.subscribe(
            topic_pattern,
            self.handle_event,
            name=f"tests.event_collector.{topic_pattern}",
        )
        self._subscriptions.append(subscription)

    async def handle_event(self, event: Event) -> None:
        async with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def unsubscribe_all(self) -> None:
        while self._subscriptions:
            subscription = self._subscriptions.pop()
            self.event_bus.unsubscribe(subscription)

    def clear(self) -> None:
        self.events.clear()

    def by_topic(self, topic: str) -> list[Event]:
        return [event for event in self.events if event.topic == topic]

    def payloads_by_topic(self, topic: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []

        for event in self.by_topic(topic):
            if isinstance(event.payload, dict):
                payloads.append(event.payload)
            elif isinstance(event.payload, Mapping):
                payloads.append(dict(event.payload))

        return payloads

    async def wait_until(
        self,
        predicate: Callable[[list[Event]], bool],
        *,
        timeout: float = 1.0,
    ) -> list[Event]:
        async def _wait() -> list[Event]:
            async with self._condition:
                while not predicate(self.events):
                    await self._condition.wait()
                return list(self.events)

        return await asyncio.wait_for(_wait(), timeout=timeout)

    async def wait_for_topic(
        self,
        topic: str,
        *,
        count: int = 1,
        timeout: float = 1.0,
    ) -> list[Event]:
        return await self.wait_until(
            lambda events: sum(event.topic == topic for event in events) >= count,
            timeout=timeout,
        )

    async def wait_for_any_topic(
        self,
        topics: set[str],
        *,
        timeout: float = 1.0,
    ) -> Event:
        events = await self.wait_until(
            lambda items: any(event.topic in topics for event in items),
            timeout=timeout,
        )

        for event in events:
            if event.topic in topics:
                return event

        raise AssertionError(f"No event collected for topics: {sorted(topics)}")

    async def wait_for_payload(
        self,
        topic: str,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 1.0,
    ) -> dict[str, Any]:
        async def _matches(events: list[Event]) -> bool:
            for event in events:
                if event.topic != topic:
                    continue

                payload = event.payload
                if not isinstance(payload, Mapping):
                    continue

                if predicate(dict(payload)):
                    return True

            return False

        await self.wait_until(
            lambda events: asyncio.run_coroutine_threadsafe(
                _matches(events),
                asyncio.get_running_loop(),
            ).result(),
            timeout=timeout,
        )

        for payload in self.payloads_by_topic(topic):
            if predicate(payload):
                return payload

        raise AssertionError(f"No matching payload collected for topic: {topic}")


# =============================================================================
# pytest / pytest-asyncio fixtures
# =============================================================================


@pytest_asyncio.fixture
async def event_bus() -> EventBus:
    bus = _make_event_bus()
    await _maybe_start(bus)

    try:
        yield bus
    finally:
        await _maybe_stop(bus)


@pytest_asyncio.fixture
async def scheduler(event_bus: EventBus) -> Scheduler:
    scheduler_obj = _make_scheduler(event_bus)
    await _maybe_start(scheduler_obj)

    try:
        yield scheduler_obj
    finally:
        await _maybe_stop(scheduler_obj)


@pytest_asyncio.fixture
async def event_collector(event_bus: EventBus) -> EventCollector:
    collector = EventCollector(event_bus)
    collector.subscribe(ANALYTICS_WHALES_WILDCARD)

    try:
        yield collector
    finally:
        collector.unsubscribe_all()


# =============================================================================
# Fast configs
# =============================================================================


@pytest.fixture
def large_trade_detector_config_fast() -> LargeTradeDetectorConfig:
    """
    Швидкий config для LargeTradeDetector.

    Налаштування підібрані так, щоб:
    - single large trade легко тригерив absolute detection;
    - relative detection можна було тестувати на малих вибірках;
    - cooldown не заважав тестам;
    - cleanup можна було тестувати без довгого очікування.
    """
    return LargeTradeDetectorConfig(
        enabled=True,
        default_abs_notional_threshold=50_000.0,
        symbol_abs_thresholds={
            "ETHUSDT": 25_000.0,
            "SOLUSDT": 10_000.0,
        },
        use_relative_detection=True,
        rolling_window_size=20,
        min_samples_for_relative_detection=3,
        zscore_threshold=2.0,
        min_notional_filter=1.0,
        side_filter=None,
        signal_cooldown_sec=0.0,
        symbol_cooldown_sec={},
        cleanup_interval_sec=0.25,
        stats_ttl_sec=0.25,
        recalibration_interval=100,
        input_event_name=MARKET_TRADE_TOPIC,
        output_event_name=LARGE_TRADE_TOPIC,
        emit_on_bus=True,
        log_signals=False,
    )


@pytest.fixture
def whale_tracker_config_fast() -> WhaleTrackerConfig:
    """
    Швидкий config для WhaleTracker.

    Мінімальні thresholds дозволяють тестувати:
    - whale_activity на 2 large trades;
    - whale_pressure на 2 large trades з дисбалансом;
    - liquidation_context без великих synthetic payload-ів.
    """
    return WhaleTrackerConfig(
        enabled=True,
        large_trade_event_name=LARGE_TRADE_TOPIC,
        liquidation_event_name=MARKET_LIQUIDATION_TOPIC,
        whale_activity_event_name=WHALE_ACTIVITY_TOPIC,
        whale_pressure_event_name=WHALE_PRESSURE_TOPIC,
        whale_liquidation_context_event_name=WHALE_LIQUIDATION_CONTEXT_TOPIC,
        cluster_window_sec=30,
        pressure_window_sec=30,
        liquidation_window_sec=30,
        cluster_min_trades=2,
        cluster_min_total_notional=80_000.0,
        pressure_min_trades=2,
        pressure_min_total_notional=80_000.0,
        pressure_imbalance_ratio_threshold=0.60,
        liquidation_context_min_notional=10_000.0,
        whale_activity_cooldown_sec=0.0,
        whale_pressure_cooldown_sec=0.0,
        whale_liquidation_context_cooldown_sec=0.0,
        cleanup_interval_sec=0.25,
        stats_ttl_sec=0.25,
        emit_on_bus=True,
        log_signals=False,
        subscribe_liquidations=True,
    )


@pytest.fixture
def whale_cluster_analyzer_config_fast() -> WhaleClusterAnalyzerConfig:
    """
    Швидкий config для WhaleClusterAnalyzer.

    Thresholds занижені, щоб integration-тести могли доходити до cluster-сигналів
    без десятків synthetic подій.
    """
    return WhaleClusterAnalyzerConfig(
        enabled=True,
        whale_activity_event_name=WHALE_ACTIVITY_TOPIC,
        whale_pressure_event_name=WHALE_PRESSURE_TOPIC,
        whale_liquidation_context_event_name=WHALE_LIQUIDATION_CONTEXT_TOPIC,
        whale_cluster_event_name=WHALE_CLUSTER_TOPIC,
        whale_cluster_update_event_name=WHALE_CLUSTER_UPDATE_TOPIC,
        whale_cluster_exhaustion_event_name=WHALE_CLUSTER_EXHAUSTION_TOPIC,
        analysis_window_sec=60,
        cluster_ttl_sec=120,
        min_activity_signals=1,
        min_total_activity_notional=50_000.0,
        activity_weight=0.45,
        pressure_weight=0.35,
        liquidation_context_weight=0.10,
        persistence_weight=0.10,
        min_cluster_score_to_emit=0.20,
        min_continuation_probability_to_emit=0.20,
        min_exhaustion_probability_to_emit=0.50,
        cluster_emit_cooldown_sec=0.0,
        cluster_update_cooldown_sec=0.0,
        cluster_exhaustion_cooldown_sec=0.0,
        cleanup_interval_sec=0.25,
        stats_ttl_sec=0.25,
        emit_on_bus=True,
        log_signals=False,
    )


@pytest.fixture
def whales_config_fast(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    whale_tracker_config_fast: WhaleTrackerConfig,
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
) -> WhalesConfig:
    return WhalesConfig(
        enabled=True,
        auto_start_components=True,
        large_trade_detector=large_trade_detector_config_fast,
        whale_tracker=whale_tracker_config_fast,
        whale_cluster_analyzer=whale_cluster_analyzer_config_fast,
    )


@pytest.fixture
def whales_config_factory(
    whales_config_fast: WhalesConfig,
) -> Callable[..., WhalesConfig]:
    """
    Factory для тестів, де треба швидко змінити enabled/auto_start або підконфіги.

    Приклад:
        config = whales_config_factory(enabled=False)
        config = whales_config_factory(
            large_trade_detector=replace(
                whales_config_fast.large_trade_detector,
                emit_on_bus=False,
            )
        )
    """

    def _factory(
        *,
        enabled: bool | None = None,
        auto_start_components: bool | None = None,
        large_trade_detector: LargeTradeDetectorConfig | None = None,
        whale_tracker: WhaleTrackerConfig | None = None,
        whale_cluster_analyzer: WhaleClusterAnalyzerConfig | None = None,
    ) -> WhalesConfig:
        config = WhalesConfig(
            enabled=whales_config_fast.enabled if enabled is None else enabled,
            auto_start_components=(
                whales_config_fast.auto_start_components
                if auto_start_components is None
                else auto_start_components
            ),
            large_trade_detector=(
                whales_config_fast.large_trade_detector
                if large_trade_detector is None
                else large_trade_detector
            ),
            whale_tracker=(
                whales_config_fast.whale_tracker
                if whale_tracker is None
                else whale_tracker
            ),
            whale_cluster_analyzer=(
                whales_config_fast.whale_cluster_analyzer
                if whale_cluster_analyzer is None
                else whale_cluster_analyzer
            ),
        )
        config.validate()
        return config

    return _factory


# =============================================================================
# Runtime component fixtures
# =============================================================================


@pytest.fixture
def large_trade_detector(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> LargeTradeDetector:
    return LargeTradeDetector(
        config=large_trade_detector_config_fast,
        event_bus=event_bus,
        scheduler=scheduler,
    )


@pytest.fixture
def whale_tracker(
    whale_tracker_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleTracker:
    return WhaleTracker(
        config=whale_tracker_config_fast,
        event_bus=event_bus,
        scheduler=scheduler,
    )


@pytest.fixture
def whale_cluster_analyzer(
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleClusterAnalyzer:
    return WhaleClusterAnalyzer(
        config=whale_cluster_analyzer_config_fast,
        event_bus=event_bus,
        scheduler=scheduler,
    )


@pytest.fixture
def whale_analyzer(
    whales_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleAnalyzer:
    return WhaleAnalyzer(
        config=whales_config_fast,
        event_bus=event_bus,
        scheduler=scheduler,
    )


# =============================================================================
# Payload factories
# =============================================================================


@pytest.fixture
def now_ms() -> Callable[[], int]:
    return lambda: int(time.time() * 1000)


@pytest.fixture
def raw_trade_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для raw market.trade payload.

    Підтримує both plain та nested data payload:
        raw_trade_payload_factory()
        raw_trade_payload_factory(nested=True)
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        price: float = 50_000.0,
        quantity: float = 2.0,
        side: str = "buy",
        timestamp_ms: int | None = None,
        trade_id: str = "trade-1",
        exchange: str = DEFAULT_EXCHANGE,
        nested: bool = False,
        maker_flag: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": symbol,
            "price": price,
            "quantity": quantity,
            "side": side,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
            "trade_id": trade_id,
            "exchange": exchange,
        }

        if maker_flag is not None:
            payload["m"] = maker_flag

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "data": payload,
            }

        return payload

    return _factory


@pytest.fixture
def large_trade_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для analytics.whales.large_trade payload.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        side: str = "buy",
        price: float = 50_000.0,
        quantity: float = 2.0,
        notional: float | None = None,
        timestamp_ms: int | None = None,
        trade_id: str = "large-trade-1",
        exchange: str = DEFAULT_EXCHANGE,
        zscore: float = 0.0,
        trigger_type: str = "absolute",
        abs_threshold: float = 50_000.0,
        mean_notional: float = 10_000.0,
        std_notional: float = 5_000.0,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        computed_notional = notional if notional is not None else price * quantity

        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_type": "large_trade",
            "detector": "LargeTradeDetector",
            "created_at_ms": now_ms(),
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "notional": computed_notional,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
            "abs_threshold": abs_threshold,
            "mean_notional": mean_notional,
            "std_notional": std_notional,
            "zscore": zscore,
            "trigger_type": trigger_type,
            "trade_id": trade_id,
            "exchange": exchange,
        }

        if extra:
            payload.update(extra)

        if nested:
            return {"data": payload}

        return payload

    return _factory


@pytest.fixture
def liquidation_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для raw market.liquidation payload.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        side: str = "sell",
        price: float = 49_500.0,
        quantity: float = 1.0,
        notional: float | None = None,
        timestamp_ms: int | None = None,
        liquidation_id: str = "liq-1",
        exchange: str = DEFAULT_EXCHANGE,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        computed_notional = notional if notional is not None else price * quantity

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "notional": computed_notional,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
            "liquidation_id": liquidation_id,
            "exchange": exchange,
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "data": payload,
            }

        return payload

    return _factory


@pytest.fixture
def whale_activity_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для analytics.whales.whale_activity payload.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        side: str = "buy",
        trade_count: int = 2,
        total_notional: float = 120_000.0,
        avg_notional: float = 60_000.0,
        max_notional: float = 70_000.0,
        window_sec: int = 30,
        timestamp_ms: int | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_type": "whale_activity",
            "detector": "WhaleTracker",
            "created_at_ms": now_ms(),
            "symbol": symbol,
            "side": side,
            "trade_count": trade_count,
            "total_notional": total_notional,
            "avg_notional": avg_notional,
            "max_notional": max_notional,
            "window_sec": window_sec,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
        }

        if extra:
            payload.update(extra)

        if nested:
            return {"data": payload}

        return payload

    return _factory


@pytest.fixture
def whale_pressure_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для analytics.whales.whale_pressure payload.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        dominant_side: str = "buy",
        buy_trade_count: int = 3,
        sell_trade_count: int = 1,
        buy_notional: float = 180_000.0,
        sell_notional: float = 40_000.0,
        total_notional: float | None = None,
        imbalance_ratio: float = 0.82,
        net_flow_notional: float | None = None,
        window_sec: int = 30,
        timestamp_ms: int | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        computed_total = (
            total_notional
            if total_notional is not None
            else buy_notional + sell_notional
        )
        computed_net_flow = (
            net_flow_notional
            if net_flow_notional is not None
            else buy_notional - sell_notional
        )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_type": "whale_pressure",
            "detector": "WhaleTracker",
            "created_at_ms": now_ms(),
            "symbol": symbol,
            "dominant_side": dominant_side,
            "buy_trade_count": buy_trade_count,
            "sell_trade_count": sell_trade_count,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "total_notional": computed_total,
            "imbalance_ratio": imbalance_ratio,
            "net_flow_notional": computed_net_flow,
            "pressure_type": "buy_pressure"
            if computed_net_flow > 0
            else "sell_pressure",
            "window_sec": window_sec,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
        }

        if extra:
            payload.update(extra)

        if nested:
            return {"data": payload}

        return payload

    return _factory


@pytest.fixture
def whale_liquidation_context_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для analytics.whales.whale_liquidation_context payload.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        whale_side: str = "buy",
        whale_total_notional: float = 160_000.0,
        whale_trade_count: int = 2,
        liquidation_side: str = "sell",
        liquidation_total_notional: float = 80_000.0,
        liquidation_count: int = 2,
        context_strength: float = 0.75,
        timestamp_ms: int | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_type": "whale_liquidation_context",
            "detector": "WhaleTracker",
            "created_at_ms": now_ms(),
            "symbol": symbol,
            "whale_side": whale_side,
            "whale_total_notional": whale_total_notional,
            "whale_trade_count": whale_trade_count,
            "liquidation_side": liquidation_side,
            "liquidation_total_notional": liquidation_total_notional,
            "liquidation_count": liquidation_count,
            "context_strength": context_strength,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
        }

        if extra:
            payload.update(extra)

        if nested:
            return {"data": payload}

        return payload

    return _factory


# =============================================================================
# Common assertion helpers
# =============================================================================


@pytest.fixture
def assert_payload_has_common_signal_fields() -> Callable[[Mapping[str, Any]], None]:
    def _assert(payload: Mapping[str, Any]) -> None:
        assert payload["schema_version"] == 1
        assert isinstance(payload["event_type"], str)
        assert isinstance(payload["detector"], str)
        assert isinstance(payload["created_at_ms"], int)

    return _assert


@pytest.fixture
def assert_symbol_payload() -> Callable[[Mapping[str, Any], str], None]:
    def _assert(payload: Mapping[str, Any], symbol: str = DEFAULT_SYMBOL) -> None:
        assert payload["symbol"] == symbol.upper()
        assert isinstance(payload["timestamp_ms"], int)
        assert payload["timestamp_ms"] > 0

    return _assert


# =============================================================================
# Small async utilities
# =============================================================================


@pytest.fixture
def eventually() -> Callable[..., Awaitable[None]]:
    """
    Helper для тестів, де EventBus/Scheduler обробляє події асинхронно.

    Приклад:
        await eventually(lambda: len(collector.by_topic(LARGE_TRADE_TOPIC)) == 1)
    """

    async def _eventually(
        predicate: Callable[[], bool],
        *,
        timeout: float = 1.0,
        interval: float = 0.01,
    ) -> None:
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(interval)

        assert predicate()

    return _eventually


@pytest.fixture
def no_bus_emit_config_factory(
    whales_config_fast: WhalesConfig,
) -> Callable[[], WhalesConfig]:
    """
    Config для direct API тестів, коли не хочемо, щоб компоненти паралельно
    публікували події в EventBus.
    """

    def _factory() -> WhalesConfig:
        large_trade_detector = replace(
            whales_config_fast.large_trade_detector,
            emit_on_bus=False,
            log_signals=False,
        )
        whale_tracker = replace(
            whales_config_fast.whale_tracker,
            emit_on_bus=False,
            log_signals=False,
        )
        whale_cluster_analyzer = replace(
            whales_config_fast.whale_cluster_analyzer,
            emit_on_bus=False,
            log_signals=False,
        )

        config = WhalesConfig(
            enabled=True,
            auto_start_components=True,
            large_trade_detector=large_trade_detector,
            whale_tracker=whale_tracker,
            whale_cluster_analyzer=whale_cluster_analyzer,
        )
        config.validate()
        return config

    return _factory