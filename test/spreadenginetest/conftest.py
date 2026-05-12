# tests/strategy/spreads/conftest.py

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pytest
import pytest_asyncio

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from strategy.strategies.spreads import (
    STRATEGY_SIGNAL_CANCELLED_EVENT,
    STRATEGY_SIGNAL_CLOSED_EVENT,
    STRATEGY_SIGNAL_GENERATED_EVENT,
    STRATEGY_SIGNAL_REJECTED_EVENT,
    STRATEGY_SIGNAL_UPDATED_EVENT,
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    CrossExchangeArbStrategy,
    CrossExchangeArbStrategyConfig,
    SpotFuturesBasisStrategy,
    SpotFuturesBasisStrategyConfig,
)


# ============================================================
# Async helpers
# ============================================================

async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _safe_start(component: Any) -> None:
    start = getattr(component, "start", None)
    if start is not None:
        await _maybe_await(start())


async def _safe_stop(component: Any) -> None:
    stop = getattr(component, "stop", None)
    if stop is not None:
        await _maybe_await(stop())


def _make_scheduler(event_bus: EventBus) -> Scheduler:
    """
    Робить fixture трохи стійкішою до різних сигнатур Scheduler.

    У твоїй core-архітектурі Scheduler може бути з EventBus або без нього,
    залежно від конкретної реалізації.
    """
    try:
        return Scheduler(event_bus=event_bus)
    except TypeError:
        return Scheduler()


# ============================================================
# Event collector
# ============================================================

@dataclass(slots=True)
class SignalCollector:
    """
    Збирає strategy-level signal.* події з EventBus.

    У тестах краще перевіряти саме ці emitted payload-и, бо це реальний
    контракт пакету strategy/spreads з risk/execution layer.
    """

    events: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    async def handler(self, event: Event) -> None:
        payload = event.payload
        if isinstance(payload, dict):
            self.events[event.topic].append(payload)
        else:
            self.events[event.topic].append({"payload": payload})

    def by_topic(self, topic: str) -> list[dict[str, Any]]:
        return list(self.events.get(topic, []))

    def latest(self, topic: str) -> dict[str, Any] | None:
        items = self.events.get(topic, [])
        return items[-1] if items else None

    def count(self, topic: str) -> int:
        return len(self.events.get(topic, []))

    async def wait_for(
        self,
        topic: str,
        *,
        count: int = 1,
        timeout: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Корисно для integration-style тестів через EventBus.emit(),
        де handler може виконатись асинхронно.
        """
        deadline = asyncio.get_running_loop().time() + timeout

        while asyncio.get_running_loop().time() < deadline:
            items = self.by_topic(topic)
            if len(items) >= count:
                return items
            await asyncio.sleep(0.01)

        return self.by_topic(topic)

    def clear(self) -> None:
        self.events.clear()


# ============================================================
# Dummy strategy for BaseSpreadStrategy tests
# ============================================================

class DummySpreadStrategy(BaseSpreadStrategy):
    """
    Мінімальна concrete strategy для тестування BaseSpreadStrategy.

    Не тестує domain-логіку basis/arb, лише базовий lifecycle,
    EventBus subscribe/emit, dedup, cooldown і cleanup.
    """

    STRATEGY_NAME = "dummy_spread_strategy"
    TEST_EVENT = "analytics.spreads.dummy"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseSpreadStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            config=config or BaseSpreadStrategyConfig(
                cleanup_closed_states_interval_seconds=0,
            ),
            scheduler=scheduler,
            service_name=self.STRATEGY_NAME,
        )
        self.received_payloads: list[Any] = []

    async def _subscribe_events(self) -> None:
        await self._subscribe_payload(
            self.TEST_EVENT,
            self.on_dummy_payload,
            name="on_dummy_payload",
        )

    async def on_dummy_payload(self, payload: Any) -> None:
        if not self.is_running:
            return
        self.received_payloads.append(payload)

    def get_stats(self) -> dict[str, Any]:
        return self.get_base_stats()


# ============================================================
# Core fixtures
# ============================================================

@pytest_asyncio.fixture
async def event_bus() -> EventBus:
    bus = EventBus()
    await _safe_start(bus)

    try:
        yield bus
    finally:
        await _safe_stop(bus)


@pytest_asyncio.fixture
async def scheduler(event_bus: EventBus) -> Scheduler:
    scheduler_instance = _make_scheduler(event_bus)
    await _safe_start(scheduler_instance)

    try:
        yield scheduler_instance
    finally:
        await _safe_stop(scheduler_instance)


@pytest_asyncio.fixture
async def signal_collector(event_bus: EventBus) -> SignalCollector:
    collector = SignalCollector()

    topics = [
        STRATEGY_SIGNAL_GENERATED_EVENT,
        STRATEGY_SIGNAL_UPDATED_EVENT,
        STRATEGY_SIGNAL_REJECTED_EVENT,
        STRATEGY_SIGNAL_CANCELLED_EVENT,
        STRATEGY_SIGNAL_CLOSED_EVENT,
    ]

    subscriptions = [
        event_bus.subscribe(
            topic,
            collector.handler,
            name=f"test.signal_collector.{topic}",
        )
        for topic in topics
    ]

    try:
        yield collector
    finally:
        for subscription in subscriptions:
            try:
                event_bus.unsubscribe(subscription)
            except Exception:
                pass


# ============================================================
# Strategy fixtures
# ============================================================

@pytest.fixture
def base_strategy_config() -> BaseSpreadStrategyConfig:
    return BaseSpreadStrategyConfig(
        enabled=True,
        cooldown_seconds=10,
        min_confidence=0,
        max_snapshot_age_ms=3_000,
        max_signal_age_ms=5_000,
        cleanup_closed_states_interval_seconds=0,
        cleanup_closed_states_older_than_seconds=3_600,
        emit_lifecycle_events=True,
    )


@pytest.fixture
def spot_basis_config() -> SpotFuturesBasisStrategyConfig:
    return SpotFuturesBasisStrategyConfig(
        enabled=True,
        cooldown_seconds=10,
        min_confidence=0,
        max_snapshot_age_ms=3_000,
        max_signal_age_ms=5_000,
        cleanup_closed_states_interval_seconds=0,
        emit_lifecycle_events=False,

        entry_zscore=2,
        exit_zscore=0.75,
        reduce_zscore=1.25,
        stop_zscore=4.5,

        min_funding_adjusted_edge=0,
        min_basis_abs=0,

        require_mean_reversion_signal=False,
        require_regime_shift_confirmation=False,
        allow_regime_shift_entry=True,

        close_on_data_quality_signal=True,
        block_entry_on_data_quality_signal=True,
        max_signals_per_key=20,
    )


@pytest.fixture
def cross_arb_config() -> CrossExchangeArbStrategyConfig:
    return CrossExchangeArbStrategyConfig(
        enabled=True,
        cooldown_seconds=10,
        min_confidence=0,
        max_snapshot_age_ms=3_000,
        max_signal_age_ms=5_000,
        cleanup_closed_states_interval_seconds=0,
        emit_lifecycle_events=False,

        entry_min_net_edge=0,
        entry_min_bps=5,
        exit_min_net_edge=0,
        exit_min_bps=0,

        min_update_net_edge_delta=0.5,
        min_update_net_edge_bps_delta=1,
        min_update_confidence_delta=0.05,

        require_persistence=False,
        min_persistence_ms=500,

        require_arbitrage_signal_confirmation=False,
        max_signals_per_key=20,

        close_on_snapshot_edge_loss=True,
        close_on_stale_snapshot=True,
        close_on_snapshot_status_not_active=True,
        update_from_snapshot_metadata=True,

        close_on_data_quality_signal=True,
        block_entry_on_data_quality_signal=True,
    )


@pytest.fixture
def dummy_spread_strategy(
    event_bus: EventBus,
    scheduler: Scheduler,
    base_strategy_config: BaseSpreadStrategyConfig,
) -> DummySpreadStrategy:
    return DummySpreadStrategy(
        event_bus=event_bus,
        scheduler=scheduler,
        config=base_strategy_config,
    )


@pytest.fixture
def spot_basis_strategy(
    event_bus: EventBus,
    scheduler: Scheduler,
    spot_basis_config: SpotFuturesBasisStrategyConfig,
) -> SpotFuturesBasisStrategy:
    return SpotFuturesBasisStrategy(
        event_bus=event_bus,
        scheduler=scheduler,
        config=spot_basis_config,
    )


@pytest.fixture
def cross_arb_strategy(
    event_bus: EventBus,
    scheduler: Scheduler,
    cross_arb_config: CrossExchangeArbStrategyConfig,
) -> CrossExchangeArbStrategy:
    return CrossExchangeArbStrategy(
        event_bus=event_bus,
        scheduler=scheduler,
        config=cross_arb_config,
    )


# ============================================================
# Pytest marks
# ============================================================

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as asyncio-based",
    )