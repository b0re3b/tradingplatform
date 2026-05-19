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
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_TIMEFRAME = "realtime"

# Production data-layer topics.
TRADES_UPDATED_TOPIC = "market.trades.updated"
LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"

# Legacy/raw exchange-adapter topics.
RAW_TRADE_TOPIC = "market.trade"
RAW_LIQUIDATION_TOPIC = "market.liquidation"

# Backward-compatible aliases for old tests.
# New tests should prefer RAW_* / *_UPDATED names explicitly.
MARKET_TRADE_TOPIC = RAW_TRADE_TOPIC
MARKET_LIQUIDATION_TOPIC = RAW_LIQUIDATION_TOPIC
MARKET_TRADES_UPDATED_TOPIC = TRADES_UPDATED_TOPIC
MARKET_LIQUIDATIONS_UPDATED_TOPIC = LIQUIDATIONS_UPDATED_TOPIC

# Analytics whale topics.
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

    Якщо core.EventBus пізніше отримає обов'язковий config,
    змінити треба буде тільки цей helper.
    """
    return EventBus()


def _make_scheduler(event_bus: EventBus) -> Scheduler:
    """
    Створює core Scheduler для тестів.

    Підтримує два варіанти constructor-а:
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
        await event_bus.emit("market.trades.updated", payload)
        events = await collector.wait_for_topic("analytics.whales.large_trade")
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
        """
        Чекає payload без run_coroutine_threadsafe всередині активного event loop.

        Це важливо для async pytest: predicate синхронний, швидкий і не блокує loop.
        """

        def _matches(events: list[Event]) -> bool:
            for event in events:
                if event.topic != topic:
                    continue

                payload = event.payload
                if not isinstance(payload, Mapping):
                    continue

                if predicate(dict(payload)):
                    return True

            return False

        await self.wait_until(_matches, timeout=timeout)

        for payload in self.payloads_by_topic(topic):
            if predicate(payload):
                return payload

        raise AssertionError(f"No matching payload collected for topic: {topic}")

    async def wait_for_no_topic(
        self,
        topic: str,
        *,
        timeout: float = 0.05,
    ) -> None:
        """
        Невеликий helper для negative assertions.

        Використовувати обережно: це intentional sleep, але локалізований
        в одному helper-і.
        """
        await asyncio.sleep(timeout)
        assert self.by_topic(topic) == []


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
    Швидкий production config для LargeTradeDetector.

    Production input:
        market.trades.updated

    Legacy raw market.trade навмисно вимкнений за замовчуванням.
    Для legacy/raw тестів використовуй large_trade_detector_legacy_config_fast.
    """
    return LargeTradeDetectorConfig(
        enabled=True,
        default_exchange=DEFAULT_EXCHANGE,
        default_market_type=DEFAULT_MARKET_TYPE,
        default_timeframe=DEFAULT_TIMEFRAME,
        default_abs_notional_threshold=50_000.0,
        symbol_abs_thresholds={
            "ETHUSDT": 25_000.0,
            "SOLUSDT": 10_000.0,
        },
        scoped_abs_thresholds={},
        use_relative_detection=True,
        rolling_window_size=20,
        min_samples_for_relative_detection=3,
        zscore_threshold=2.0,
        min_notional_filter=1.0,
        side_filter=None,
        signal_cooldown_sec=0.0,
        symbol_cooldown_sec={},
        scoped_cooldown_sec={},
        cleanup_interval_sec=0.25,
        stats_ttl_sec=0.25,
        recalibration_interval=100,
        input_event_name=TRADES_UPDATED_TOPIC,
        input_event_patterns=(TRADES_UPDATED_TOPIC,),
        output_event_name=LARGE_TRADE_TOPIC,
        raw_input_event_name=RAW_TRADE_TOPIC,
        allow_legacy_raw_topics=False,
        emit_on_bus=True,
        log_signals=False,
    )


@pytest.fixture
def large_trade_detector_legacy_config_fast(
    large_trade_detector_config_fast: LargeTradeDetectorConfig,
) -> LargeTradeDetectorConfig:
    """
    Config тільки для legacy/raw tests.

    Дозволяє detector-у слухати market.trade.
    Production-тести не повинні використовувати цей fixture.
    """
    return replace(
        large_trade_detector_config_fast,
        input_event_name=TRADES_UPDATED_TOPIC,
        input_event_patterns=(TRADES_UPDATED_TOPIC,),
        raw_input_event_name=RAW_TRADE_TOPIC,
        allow_legacy_raw_topics=True,
    )


@pytest.fixture
def whale_tracker_config_fast() -> WhaleTrackerConfig:
    """
    Швидкий production config для WhaleTracker.

    Production input:
        analytics.whales.large_trade
        market.liquidations.updated

    Raw market.liquidation вимкнений за замовчуванням.
    """
    return WhaleTrackerConfig(
        enabled=True,
        default_exchange=DEFAULT_EXCHANGE,
        default_market_type=DEFAULT_MARKET_TYPE,
        default_timeframe=DEFAULT_TIMEFRAME,
        large_trade_event_name=LARGE_TRADE_TOPIC,
        liquidation_event_name=LIQUIDATIONS_UPDATED_TOPIC,
        liquidation_event_patterns=(LIQUIDATIONS_UPDATED_TOPIC,),
        raw_liquidation_event_name=RAW_LIQUIDATION_TOPIC,
        allow_legacy_raw_topics=False,
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
def whale_tracker_legacy_config_fast(
    whale_tracker_config_fast: WhaleTrackerConfig,
) -> WhaleTrackerConfig:
    """
    Config тільки для legacy/raw liquidation tests.
    """
    return replace(
        whale_tracker_config_fast,
        liquidation_event_name=LIQUIDATIONS_UPDATED_TOPIC,
        liquidation_event_patterns=(LIQUIDATIONS_UPDATED_TOPIC,),
        raw_liquidation_event_name=RAW_LIQUIDATION_TOPIC,
        allow_legacy_raw_topics=True,
    )


@pytest.fixture
def whale_cluster_analyzer_config_fast() -> WhaleClusterAnalyzerConfig:
    """
    Швидкий config для WhaleClusterAnalyzer.

    Thresholds занижені, щоб integration-тести доходили до cluster-сигналів
    без десятків synthetic подій.
    """
    return WhaleClusterAnalyzerConfig(
        enabled=True,
        default_exchange=DEFAULT_EXCHANGE,
        default_market_type=DEFAULT_MARKET_TYPE,
        default_timeframe=DEFAULT_TIMEFRAME,
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
def whales_legacy_config_fast(
    large_trade_detector_legacy_config_fast: LargeTradeDetectorConfig,
    whale_tracker_legacy_config_fast: WhaleTrackerConfig,
    whale_cluster_analyzer_config_fast: WhaleClusterAnalyzerConfig,
) -> WhalesConfig:
    """
    Повний package config для migration/legacy tests.

    Production integration-тести мають використовувати whales_config_fast.
    """
    return WhalesConfig(
        enabled=True,
        auto_start_components=True,
        large_trade_detector=large_trade_detector_legacy_config_fast,
        whale_tracker=whale_tracker_legacy_config_fast,
        whale_cluster_analyzer=whale_cluster_analyzer_config_fast,
    )


@pytest.fixture
def whales_config_factory(
    whales_config_fast: WhalesConfig,
) -> Callable[..., WhalesConfig]:
    """
    Factory для тестів, де треба швидко змінити enabled/auto_start або підконфіги.
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
def large_trade_detector_legacy(
    large_trade_detector_legacy_config_fast: LargeTradeDetectorConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> LargeTradeDetector:
    return LargeTradeDetector(
        config=large_trade_detector_legacy_config_fast,
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
def whale_tracker_legacy(
    whale_tracker_legacy_config_fast: WhaleTrackerConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleTracker:
    return WhaleTracker(
        config=whale_tracker_legacy_config_fast,
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


@pytest.fixture
def whale_analyzer_legacy(
    whales_legacy_config_fast: WhalesConfig,
    event_bus: EventBus,
    scheduler: Scheduler,
) -> WhaleAnalyzer:
    return WhaleAnalyzer(
        config=whales_legacy_config_fast,
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

    Це payload рівня exchange adapter.
    У production він має йти в TradesCache, а не напряму в LargeTradeDetector.
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        nested: bool = False,
        maker_flag: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "price": price,
            "quantity": quantity,
            "side": side,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
            "trade_id": trade_id,
        }

        if maker_flag is not None:
            payload["m"] = maker_flag

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

        return payload

    return _factory


@pytest.fixture
def trades_updated_payload_factory(
    raw_trade_payload_factory: Callable[..., dict[str, Any]],
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для production data-layer event payload:
        market.trades.updated

    Підтримує batch trades, змішаний valid/invalid batch, extra metadata.
    """

    def _factory(
        *,
        trades: list[dict[str, Any]] | None = None,
        symbol: str = DEFAULT_SYMBOL,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        price: float = 50_000.0,
        quantity: float = 2.0,
        side: str = "buy",
        count: int = 1,
        notional: float | None = None,
        timestamp_ms: int | None = None,
        batch_id: str = "trades-updated-1",
        nested_data: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_ts = timestamp_ms if timestamp_ms is not None else now_ms()

        if trades is None:
            trades = []
            for index in range(count):
                trade_price = price
                trade_quantity = (
                    quantity
                    if notional is None
                    else notional / trade_price
                )
                trades.append(
                    raw_trade_payload_factory(
                        symbol=symbol,
                        price=trade_price,
                        quantity=trade_quantity,
                        side=side,
                        timestamp_ms=base_ts + index,
                        trade_id=f"{batch_id}-{index}",
                        exchange=exchange,
                        market_type=market_type,
                        timeframe=timeframe,
                    )
                )

        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp_ms": base_ts,
            "received_at_ms": now_ms(),
            "batch_id": batch_id,
            "trades": trades,
            "count": len(trades),
            "source": "tests.trades_cache",
        }

        if extra:
            payload.update(extra)

        if nested_data:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

        return payload

    return _factory


@pytest.fixture
def raw_liquidation_payload_factory(
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для raw market.liquidation payload.

    У production має йти в LiquidationCache / liquidation analytics layer,
    а не напряму в WhaleTracker.
    """

    def _factory(
        *,
        symbol: str = DEFAULT_SYMBOL,
        side: str = "sell",
        price: float = 50_000.0,
        quantity: float = 2.0,
        notional: float | None = None,
        timestamp_ms: int | None = None,
        liquidation_id: str = "liq-1",
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        computed_notional = notional if notional is not None else price * quantity

        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "side": side,
            "price": price,
            "quantity": quantity,
            "notional": computed_notional,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else now_ms(),
            "liquidation_id": liquidation_id,
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

        return payload

    return _factory


@pytest.fixture
def liquidation_payload_factory(
    raw_liquidation_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """
    Backward-compatible alias для старих тестів.

    Нові тести мають явно використовувати:
    - raw_liquidation_payload_factory
    - liquidations_updated_payload_factory
    """
    return raw_liquidation_payload_factory


@pytest.fixture
def liquidations_updated_payload_factory(
    raw_liquidation_payload_factory: Callable[..., dict[str, Any]],
    now_ms: Callable[[], int],
) -> Callable[..., dict[str, Any]]:
    """
    Factory для production data-layer event payload:
        market.liquidations.updated
    """

    def _factory(
        *,
        liquidations: list[dict[str, Any]] | None = None,
        symbol: str = DEFAULT_SYMBOL,
        side: str = "sell",
        price: float = 50_000.0,
        quantity: float = 2.0,
        notional: float | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        timestamp_ms: int | None = None,
        batch_id: str = "liquidations-updated-1",
        nested_data: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_ts = timestamp_ms if timestamp_ms is not None else now_ms()

        if liquidations is None:
            liquidations = [
                raw_liquidation_payload_factory(
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=quantity,
                    notional=notional,
                    timestamp_ms=base_ts,
                    liquidation_id=f"{batch_id}-0",
                    exchange=exchange,
                    market_type=market_type,
                    timeframe=timeframe,
                )
            ]

        payload: dict[str, Any] = {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp_ms": base_ts,
            "received_at_ms": now_ms(),
            "batch_id": batch_id,
            "liquidations": liquidations,
            "count": len(liquidations),
            "source": "tests.liquidation_cache",
        }

        if len(liquidations) == 1:
            payload["liquidation"] = liquidations[0]

        if extra:
            payload.update(extra)

        if nested_data:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
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
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        zscore: float = 0.0,
        trigger_type: str = "absolute",
        abs_threshold: float = 50_000.0,
        mean_notional: float = 10_000.0,
        std_notional: float = 5_000.0,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        computed_notional = notional if notional is not None else price * quantity
        created_at_ms = now_ms()

        payload: dict[str, Any] = {
            "schema_version": 2,
            "event_type": "large_trade",
            "detector": "LargeTradeDetector",
            "created_at_ms": created_at_ms,
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "side": side,
            "price": price,
            "quantity": quantity,
            "notional": computed_notional,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else created_at_ms,
            "trade_id": trade_id,
            "zscore": zscore,
            "trigger_type": trigger_type,
            "abs_threshold": abs_threshold,
            "mean_notional": mean_notional,
            "std_notional": std_notional,
            "metadata": {
                "source": "tests.large_trade_detector",
            },
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
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
        avg_notional: float | None = None,
        max_notional: float = 80_000.0,
        window_sec: int = 30,
        timestamp_ms: int | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at_ms = now_ms()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "event_type": "whale_activity",
            "detector": "WhaleTracker",
            "created_at_ms": created_at_ms,
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "side": side,
            "trade_count": trade_count,
            "total_notional": total_notional,
            "avg_notional": (
                avg_notional
                if avg_notional is not None
                else total_notional / max(1, trade_count)
            ),
            "max_notional": max_notional,
            "window_sec": window_sec,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else created_at_ms,
            "metadata": {
                "source": "tests.whale_tracker",
            },
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

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
        buy_trade_count: int = 2,
        sell_trade_count: int = 0,
        buy_notional: float = 120_000.0,
        sell_notional: float = 0.0,
        total_notional: float | None = None,
        imbalance_ratio: float | None = None,
        net_flow_notional: float | None = None,
        window_sec: int = 30,
        timestamp_ms: int | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at_ms = now_ms()
        computed_total = (
            total_notional
            if total_notional is not None
            else buy_notional + sell_notional
        )
        dominant_notional = max(buy_notional, sell_notional)
        computed_imbalance = (
            imbalance_ratio
            if imbalance_ratio is not None
            else (
                dominant_notional / computed_total
                if computed_total > 0
                else 0.0
            )
        )
        computed_net_flow = (
            net_flow_notional
            if net_flow_notional is not None
            else buy_notional - sell_notional
        )

        payload: dict[str, Any] = {
            "schema_version": 2,
            "event_type": "whale_pressure",
            "detector": "WhaleTracker",
            "created_at_ms": created_at_ms,
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "dominant_side": dominant_side,
            "buy_trade_count": buy_trade_count,
            "sell_trade_count": sell_trade_count,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "total_notional": computed_total,
            "imbalance_ratio": computed_imbalance,
            "net_flow_notional": computed_net_flow,
            "window_sec": window_sec,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else created_at_ms,
            "metadata": {
                "source": "tests.whale_tracker",
            },
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

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
        liquidation_total_notional: float = 100_000.0,
        liquidation_count: int = 1,
        context_strength: float = 0.75,
        timestamp_ms: int | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        nested: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at_ms = now_ms()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "event_type": "whale_liquidation_context",
            "detector": "WhaleTracker",
            "created_at_ms": created_at_ms,
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol or symbol,
            "timeframe": timeframe,
            "whale_side": whale_side,
            "whale_total_notional": whale_total_notional,
            "whale_trade_count": whale_trade_count,
            "liquidation_side": liquidation_side,
            "liquidation_total_notional": liquidation_total_notional,
            "liquidation_count": liquidation_count,
            "context_strength": context_strength,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else created_at_ms,
            "metadata": {
                "source": "tests.whale_tracker",
            },
        }

        if extra:
            payload.update(extra)

        if nested:
            return {
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": payload,
            }

        return payload

    return _factory


# =============================================================================
# Common emit helpers
# =============================================================================

@pytest.fixture
def emit_trades_updated(
    event_bus: EventBus,
    trades_updated_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., Awaitable[bool]]:
    """
    Emit helper для production trade batches.
    """

    async def _emit(
        *,
        payload: dict[str, Any] | None = None,
        correlation_id: str = "corr-trades-updated",
        source: str = "tests.trades_cache",
        **factory_kwargs: Any,
    ) -> bool:
        event_payload = payload or trades_updated_payload_factory(**factory_kwargs)
        return await event_bus.emit(
            TRADES_UPDATED_TOPIC,
            event_payload,
            source=source,
            correlation_id=correlation_id,
        )

    return _emit


@pytest.fixture
def emit_raw_trade(
    event_bus: EventBus,
    raw_trade_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., Awaitable[bool]]:
    """
    Emit helper для legacy raw market.trade.
    """

    async def _emit(
        *,
        payload: dict[str, Any] | None = None,
        correlation_id: str = "corr-raw-trade",
        source: str = "tests.market_stream",
        **factory_kwargs: Any,
    ) -> bool:
        event_payload = payload or raw_trade_payload_factory(**factory_kwargs)
        return await event_bus.emit(
            RAW_TRADE_TOPIC,
            event_payload,
            source=source,
            correlation_id=correlation_id,
        )

    return _emit


@pytest.fixture
def emit_liquidations_updated(
    event_bus: EventBus,
    liquidations_updated_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., Awaitable[bool]]:
    """
    Emit helper для production liquidation batches.
    """

    async def _emit(
        *,
        payload: dict[str, Any] | None = None,
        correlation_id: str = "corr-liquidations-updated",
        source: str = "tests.liquidation_cache",
        **factory_kwargs: Any,
    ) -> bool:
        event_payload = payload or liquidations_updated_payload_factory(**factory_kwargs)
        return await event_bus.emit(
            LIQUIDATIONS_UPDATED_TOPIC,
            event_payload,
            source=source,
            correlation_id=correlation_id,
        )

    return _emit


@pytest.fixture
def emit_raw_liquidation(
    event_bus: EventBus,
    raw_liquidation_payload_factory: Callable[..., dict[str, Any]],
) -> Callable[..., Awaitable[bool]]:
    """
    Emit helper для legacy raw market.liquidation.
    """

    async def _emit(
        *,
        payload: dict[str, Any] | None = None,
        correlation_id: str = "corr-raw-liquidation",
        source: str = "tests.liquidation_stream",
        **factory_kwargs: Any,
    ) -> bool:
        event_payload = payload or raw_liquidation_payload_factory(**factory_kwargs)
        return await event_bus.emit(
            RAW_LIQUIDATION_TOPIC,
            event_payload,
            source=source,
            correlation_id=correlation_id,
        )

    return _emit


# =============================================================================
# Assertion helpers
# =============================================================================

@pytest.fixture
def assert_payload_has_common_signal_fields() -> Callable[[Mapping[str, Any]], None]:
    def _assert(payload: Mapping[str, Any]) -> None:
        assert "schema_version" in payload
        assert "event_type" in payload
        assert "detector" in payload
        assert "created_at_ms" in payload
        assert isinstance(payload["created_at_ms"], int)
        assert payload["created_at_ms"] > 0

    return _assert


@pytest.fixture
def assert_symbol_payload() -> Callable[[Mapping[str, Any], str], None]:
    def _assert(
        payload: Mapping[str, Any],
        symbol: str,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> None:
        assert payload["symbol"] == symbol.upper()
        assert payload["exchange"] == exchange
        assert payload["market_type"] == market_type
        assert payload["timeframe"] == timeframe
        assert "scope" in payload

    return _assert


@pytest.fixture
def assert_no_whale_events(
    event_collector: EventCollector,
) -> Callable[[], None]:
    def _assert() -> None:
        assert event_collector.by_topic(LARGE_TRADE_TOPIC) == []
        assert event_collector.by_topic(WHALE_ACTIVITY_TOPIC) == []
        assert event_collector.by_topic(WHALE_PRESSURE_TOPIC) == []
        assert event_collector.by_topic(WHALE_LIQUIDATION_CONTEXT_TOPIC) == []
        assert event_collector.by_topic(WHALE_CLUSTER_TOPIC) == []
        assert event_collector.by_topic(WHALE_CLUSTER_UPDATE_TOPIC) == []
        assert event_collector.by_topic(WHALE_CLUSTER_EXHAUSTION_TOPIC) == []

    return _assert