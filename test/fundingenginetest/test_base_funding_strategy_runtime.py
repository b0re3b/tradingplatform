# tests/strategy/funding/test_base_funding_strategy_runtime.py
from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
)

from strategy.strategies.funding.base import (
    BaseFundingStrategy,
    BaseFundingStrategyConfig,
    FundingSetupStatus,
    FundingStrategyDirection,
    FundingStrategyState,
    utc_now,
)


# =============================================================================
# Concrete test implementation
# =============================================================================


class TestFundingRuntimeStrategy(BaseFundingStrategy):
    """
    Minimal concrete strategy for testing BaseFundingStrategy runtime behavior.

    It intentionally does not contain domain logic. The goal is to verify that the
    base class correctly handles lifecycle, subscriptions, state transitions,
    locks, scheduler cleanup, event emission and normalized analytics handlers.
    """

    def __init__(
        self,
        *,
        event_bus: Any,
        config: BaseFundingStrategyConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            config=config,
            scheduler=scheduler,
            service_name=(config.service_name if config else "pytest_funding_runtime_strategy"),
        )
        self.test_events_seen: list[dict[str, Any]] = []
        self.updated_hooks_seen: list[tuple[FundingStrategyState, dict[str, Any], Any]] = []
        self.signal_hooks_seen: list[tuple[FundingStrategyState, Any, Any]] = []

    @property
    def strategy_name(self) -> str:
        return "pytest_funding_runtime"

    def register_subscriptions(self) -> None:
        self.subscribe(
            "analytics.funding.test",
            self.on_test_event,
            name=f"{self.strategy_name}.on_test_event",
        )

    async def on_test_event(self, event: Any) -> None:
        self.test_events_seen.append(self.extract_payload(event))

    async def on_after_funding_updated(
        self,
        state: FundingStrategyState,
        payload: dict[str, Any],
        event: Any,
    ) -> None:
        self.updated_hooks_seen.append((state, payload, event))

    async def on_after_funding_signal(
        self,
        state: FundingStrategyState,
        signal: Any,
        event: Any,
    ) -> None:
        self.signal_hooks_seen.append((state, signal, event))


# =============================================================================
# Local helpers
# =============================================================================


def _make_strategy(
    *,
    event_bus_spy: Any,
    scheduler_spy: Any | None = None,
    config: BaseFundingStrategyConfig,
) -> TestFundingRuntimeStrategy:
    return TestFundingRuntimeStrategy(
        event_bus=event_bus_spy,
        scheduler=scheduler_spy,
        config=config,
    )


def _make_setup_state(strategy: TestFundingRuntimeStrategy) -> FundingStrategyState:
    state = strategy.get_state("BTCUSDT", "binance")
    strategy.set_setup_detected(
        state,
        direction=FundingStrategyDirection.LONG,
        setup_type="pytest_setup",
        score=0.72,
        confidence=0.81,
        reason="pytest_setup_detected",
        tags=["pytest", "runtime"],
        metadata={"source": "unit_test"},
    )
    return state


def _last_record(event_bus_spy: Any) -> Any:
    assert event_bus_spy.emitted, "Expected at least one emitted event"
    return event_bus_spy.emitted[-1]


def _assert_base_payload(
    payload: dict[str, Any],
    *,
    event_kind: str,
    status: FundingSetupStatus,
    direction: FundingStrategyDirection,
) -> None:
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange"] == "binance"
    assert payload["strategy"] == "pytest_funding_runtime"
    assert payload["strategy_name"] == "pytest_funding_runtime"
    assert payload["strategy_namespace"] == "strategy.funding.test"
    assert payload["event_kind"] == event_kind
    assert payload["status"] == status.value
    assert payload["direction"] == direction.value
    assert payload["setup_type"] == "pytest_setup"
    assert isinstance(payload["score"], float)
    assert isinstance(payload["confidence"], float)
    assert "state" in payload


# =============================================================================
# Construction / lifecycle / subscriptions
# =============================================================================


def test_init_requires_event_bus(base_strategy_config: BaseFundingStrategyConfig) -> None:
    with pytest.raises(ValueError, match="event_bus is required"):
        TestFundingRuntimeStrategy(
            event_bus=None,
            config=base_strategy_config,
        )


def test_init_validates_config(event_bus_spy: Any, base_strategy_config: BaseFundingStrategyConfig) -> None:
    base_strategy_config.setup_ttl_sec = 0.0

    with pytest.raises(ValueError, match="setup_ttl_sec must be > 0"):
        TestFundingRuntimeStrategy(
            event_bus=event_bus_spy,
            config=base_strategy_config,
        )


@pytest.mark.asyncio
async def test_start_registers_base_and_child_subscriptions_and_scheduler_cleanup_job(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    await strategy.start()

    assert strategy.stats()["running"] is True
    assert strategy.stats()["registered"] is True
    assert strategy.stats()["subscriptions"] == 3

    subscribed_patterns = {subscription.pattern for subscription in event_bus_spy.subscribed}
    assert subscribed_patterns == {
        "analytics.funding.updated",
        "analytics.funding.signal",
        "analytics.funding.test",
    }

    assert len(scheduler_spy.added_jobs) == 1
    cleanup_job = scheduler_spy.added_jobs[0]
    assert cleanup_job.name == "pytest_funding_runtime.cleanup_expired_states"
    assert cleanup_job.interval == base_strategy_config.cleanup_interval_sec
    assert cleanup_job.timeout == base_strategy_config.cleanup_job_timeout_sec
    assert strategy.stats()["cleanup_job_id"] == cleanup_job.job_id


@pytest.mark.asyncio
async def test_start_is_idempotent_when_already_running(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    await strategy.start()
    first_subscription_count = len(event_bus_spy.subscribed)
    first_job_count = len(scheduler_spy.added_jobs)

    await strategy.start()

    assert len(event_bus_spy.subscribed) == first_subscription_count
    assert len(scheduler_spy.added_jobs) == first_job_count
    assert strategy.stats()["subscriptions"] == 3


@pytest.mark.asyncio
async def test_stop_unregisters_subscriptions_and_scheduler_cleanup_job(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    await strategy.start()
    cleanup_job_id = strategy.stats()["cleanup_job_id"]

    await strategy.stop()

    assert strategy.stats()["running"] is False
    assert strategy.stats()["registered"] is False
    assert strategy.stats()["subscriptions"] == 0
    assert len(event_bus_spy.unsubscribed) == 3
    assert cleanup_job_id in scheduler_spy.removed_job_ids
    assert strategy.stats()["cleanup_job_id"] is None


@pytest.mark.asyncio
async def test_restart_stops_and_starts_strategy_again(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    await strategy.start()
    first_cleanup_job_id = strategy.stats()["cleanup_job_id"]

    await strategy.restart()

    assert strategy.stats()["running"] is True
    assert strategy.stats()["registered"] is True
    assert strategy.stats()["subscriptions"] == 3
    assert first_cleanup_job_id in scheduler_spy.removed_job_ids
    assert strategy.stats()["cleanup_job_id"] is not None
    assert strategy.stats()["cleanup_job_id"] != first_cleanup_job_id


def test_register_is_idempotent_without_start(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    strategy.register()
    strategy.register()

    assert strategy.stats()["registered"] is True
    assert strategy.stats()["subscriptions"] == 3
    assert len(event_bus_spy.subscribed) == 3


def test_unregister_is_safe_when_not_registered(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    strategy.unregister()

    assert strategy.stats()["registered"] is False
    assert strategy.stats()["subscriptions"] == 0
    assert event_bus_spy.unsubscribed == []


# =============================================================================
# State access / state lifecycle
# =============================================================================


def test_get_state_normalizes_symbol_exchange_and_reuses_state(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    state_1 = strategy.get_state("btcusdt", "BINANCE")
    state_2 = strategy.get_state("BTCUSDT", "binance")

    assert state_1 is state_2
    assert state_1.symbol == "BTCUSDT"
    assert state_1.exchange == "binance"
    assert state_1.key == "BTCUSDT:binance"
    assert state_1.strategy_name == "pytest_funding_runtime"


def test_set_setup_detected_sets_active_state_ttl_tags_and_metadata(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = strategy.get_state("BTCUSDT", "binance")

    before = utc_now()
    strategy.set_setup_detected(
        state,
        direction=FundingStrategyDirection.SHORT,
        setup_type="pytest_short_setup",
        score=1.5,
        confidence=-0.25,
        reason="detected",
        tags=["a", "b"],
        metadata={"nested": {"ok": True}},
    )
    after = utc_now()

    assert state.status == FundingSetupStatus.SETUP_DETECTED
    assert state.direction == FundingStrategyDirection.SHORT
    assert state.setup_type == "pytest_short_setup"
    assert state.score == 1.0
    assert state.confidence == 0.0
    assert state.reason == "detected"
    assert state.reasons == ["detected"]
    assert state.tags == ["a", "b"]
    assert state.metadata == {"nested": {"ok": True}}
    assert state.expires_at is not None
    assert before + timedelta(seconds=base_strategy_config.setup_ttl_sec) <= state.expires_at <= after + timedelta(
        seconds=base_strategy_config.setup_ttl_sec
    )
    assert state.is_active() is True


def test_set_confirmed_updates_state_once_when_reconfirm_disabled(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    assert base_strategy_config.allow_reconfirm is False

    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_confirmed(
        state,
        score=0.88,
        confidence=0.91,
        reason="first_confirmation",
        tags=["confirmed_once"],
        metadata={"confirmation_source": "first"},
    )
    first_confirmed_at = state.confirmed_at

    strategy.set_confirmed(
        state,
        score=0.11,
        confidence=0.22,
        reason="second_confirmation_should_be_ignored",
        tags=["confirmed_twice"],
        metadata={"confirmation_source": "second"},
    )

    assert state.status == FundingSetupStatus.CONFIRMED
    assert state.score == 0.88
    assert state.confidence == 0.91
    assert state.reason == "first_confirmation"
    assert "confirmed_once" in state.tags
    assert "confirmed_twice" not in state.tags
    assert state.metadata["confirmation_source"] == "first"
    assert state.confirmed_at == first_confirmed_at


def test_set_confirmed_allows_reconfirm_when_enabled(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    base_strategy_config.allow_reconfirm = True

    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_confirmed(
        state,
        score=0.70,
        confidence=0.70,
        reason="first_confirmation",
        tags=["first"],
    )
    strategy.set_confirmed(
        state,
        score=0.95,
        confidence=0.96,
        reason="second_confirmation",
        tags=["second"],
    )

    assert state.status == FundingSetupStatus.CONFIRMED
    assert state.score == 0.95
    assert state.confidence == 0.96
    assert state.reason == "second_confirmation"
    assert "first" in state.tags
    assert "second" in state.tags


def test_set_invalidated_without_cooldown_preserves_invalidated_status(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_invalidated(
        state,
        reason="manual_invalidation",
        cooldown=False,
        metadata={"invalidation_source": "test"},
    )

    assert state.status == FundingSetupStatus.INVALIDATED
    assert state.reason == "manual_invalidation"
    assert "manual_invalidation" in state.reasons
    assert state.invalidated_at is not None
    assert state.cooldown_until is None
    assert state.metadata["invalidation_source"] == "test"


def test_set_invalidated_with_cooldown_enters_cooldown(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_invalidated(
        state,
        reason="cooldown_invalidation",
        cooldown=True,
        metadata={"invalidation_source": "test"},
    )

    assert state.status == FundingSetupStatus.COOLDOWN
    assert state.reason == "cooldown_invalidation"
    assert state.invalidated_at is not None
    assert state.cooldown_until is not None
    assert state.cooldown_until > utc_now()
    assert strategy.is_in_cooldown(state) is True


def test_set_expired_with_cooldown_enters_cooldown(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_expired(state, reason="ttl_expired", cooldown=True)

    assert state.status == FundingSetupStatus.COOLDOWN
    assert state.reason == "ttl_expired"
    assert state.invalidated_at is not None
    assert state.cooldown_until is not None
    assert "ttl_expired" in state.reasons


def test_is_in_cooldown_resets_expired_cooldown_to_idle(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.set_cooldown(state, cooldown_sec=1.0, reason="short_cooldown")
    state.cooldown_until = utc_now() - timedelta(seconds=1.0)

    assert strategy.is_in_cooldown(state) is False
    assert state.status == FundingSetupStatus.IDLE
    assert state.cooldown_until is None


def test_get_active_states_excludes_expired_and_cooldown_states(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    active_state = strategy.get_state("BTCUSDT", "binance")
    strategy.set_setup_detected(
        active_state,
        direction=FundingStrategyDirection.LONG,
        setup_type="active_setup",
        score=0.7,
        confidence=0.7,
    )

    expired_state = strategy.get_state("ETHUSDT", "binance")
    strategy.set_setup_detected(
        expired_state,
        direction=FundingStrategyDirection.SHORT,
        setup_type="expired_setup",
        score=0.7,
        confidence=0.7,
    )
    expired_state.expires_at = utc_now() - timedelta(seconds=1.0)

    cooldown_state = strategy.get_state("SOLUSDT", "binance")
    strategy.set_setup_detected(
        cooldown_state,
        direction=FundingStrategyDirection.LONG,
        setup_type="cooldown_setup",
        score=0.7,
        confidence=0.7,
    )
    strategy.set_cooldown(cooldown_state, cooldown_sec=30.0, reason="cooldown")

    active_states = strategy.get_active_states()

    assert "BTCUSDT:binance" in active_states
    assert "ETHUSDT:binance" not in active_states
    assert "SOLUSDT:binance" not in active_states


@pytest.mark.asyncio
async def test_cleanup_expired_states_expires_active_states_and_emits_expired_event(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    state.expires_at = utc_now() - timedelta(seconds=1.0)

    expired_count = await strategy.cleanup_expired_states(emit_events=True)

    assert expired_count == 1
    assert state.status == FundingSetupStatus.COOLDOWN

    record = _last_record(event_bus_spy)
    assert record.topic == "strategy.funding.test.expired"
    assert record.priority == base_strategy_config.expiration_priority
    assert record.source == base_strategy_config.source_name
    assert record.payload["event_kind"] == "expired"
    assert record.payload["status"] == FundingSetupStatus.COOLDOWN.value
    assert record.payload["trigger"] == "scheduler_cleanup"


@pytest.mark.asyncio
async def test_cleanup_expired_states_can_skip_emitting_events(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    state.expires_at = utc_now() - timedelta(seconds=1.0)

    expired_count = await strategy.cleanup_expired_states(emit_events=False)

    assert expired_count == 1
    assert state.status == FundingSetupStatus.COOLDOWN
    assert event_bus_spy.emitted == []


# =============================================================================
# Locking
# =============================================================================


@pytest.mark.asyncio
async def test_acquire_symbol_lock_returns_lock_and_release_unlocks_it(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    lock = await strategy.acquire_symbol_lock("btcusdt", "BINANCE")

    assert lock is not None
    assert lock.locked() is True
    strategy.release_symbol_lock(lock)
    assert lock.locked() is False


@pytest.mark.asyncio
async def test_acquire_symbol_lock_returns_none_on_timeout(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    base_strategy_config.state_lock_timeout_sec = 0.01
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    first_lock = await strategy.acquire_symbol_lock("BTCUSDT", "binance")
    assert first_lock is not None

    second_lock = await strategy.acquire_symbol_lock("BTCUSDT", "binance")

    assert second_lock is None

    strategy.release_symbol_lock(first_lock)


@pytest.mark.asyncio
async def test_on_funding_signal_returns_without_hook_when_lock_timeout(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_funding_signal_event: Any,
) -> None:
    base_strategy_config.state_lock_timeout_sec = 0.01
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    lock = await strategy.acquire_symbol_lock("BTCUSDT", "binance")
    assert lock is not None

    await strategy.on_funding_signal(make_funding_signal_event())

    assert strategy.signal_hooks_seen == []

    strategy.release_symbol_lock(lock)


# =============================================================================
# Freshness / payload extraction
# =============================================================================


def test_is_stale_event_rejects_old_events_and_accepts_recent_events(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    base_strategy_config.event_stale_after_sec = 60.0
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    assert strategy.is_stale_event(utc_now() - timedelta(seconds=120.0)) is True
    assert strategy.is_stale_event(utc_now() - timedelta(seconds=10.0)) is False
    assert strategy.is_stale_event(None) is False


def test_extract_symbol_exchange_accepts_nested_analytics_envelope(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    payload = {
        "payload": {
            "regime_state": {
                "symbol": "ethusdt",
                "exchange": "BYBIT",
                "regime": FundingRegime.POSITIVE.value,
                "bias": FundingBias.LONG_BIAS.value,
            }
        }
    }

    symbol, exchange = strategy.extract_symbol_exchange(payload)

    assert symbol == "ETHUSDT"
    assert exchange == "bybit"


def test_extract_payload_accepts_mapping_payload_and_object_payload(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_test_event: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    event = make_test_event(
        "analytics.funding.test",
        {"symbol": "BTCUSDT", "exchange": "binance", "value": 1},
    )

    assert strategy.extract_payload(event)["value"] == 1
    assert strategy.extract_payload({"payload": {"value": 2}})["value"] == 2
    assert strategy.extract_payload({"value": 3})["value"] == 3
    assert strategy.extract_payload(None) == {}


# =============================================================================
# Emit behavior
# =============================================================================


@pytest.mark.asyncio
async def test_emit_setup_builds_topic_payload_headers_priority_and_updates_emit_stats(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    await strategy.emit_setup(
        state,
        extra_payload={
            "trigger": "unit_test",
            "correlation_id": "corr-setup",
            "source_event_id": "source-1",
        },
    )

    record = _last_record(event_bus_spy)
    assert record.topic == "strategy.funding.test.setup"
    assert record.priority == base_strategy_config.setup_priority
    assert record.source == base_strategy_config.source_name
    assert record.correlation_id == "corr-setup"
    assert record.headers == {
        "strategy": "pytest_funding_runtime",
        "strategy_namespace": "strategy.funding.test",
        "event_kind": "setup",
        "symbol": "BTCUSDT",
        "exchange": "binance",
    }

    _assert_base_payload(
        record.payload,
        event_kind="setup",
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert record.payload["trigger"] == "unit_test"
    assert record.payload["source_event_id"] == "source-1"
    assert state.emit_count == 1
    assert state.last_emit_time is not None


@pytest.mark.asyncio
async def test_emit_confirmed_uses_confirmation_topic_and_priority(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    strategy.set_confirmed(
        state,
        score=0.90,
        confidence=0.92,
        reason="confirmed",
        tags=["confirmed"],
    )

    await strategy.emit_confirmed(
        state,
        extra_payload={"trigger": "unit_test", "correlation_id": "corr-confirmed"},
    )

    record = _last_record(event_bus_spy)
    assert record.topic == "strategy.funding.test.confirmed"
    assert record.priority == base_strategy_config.confirmation_priority
    assert record.correlation_id == "corr-confirmed"

    _assert_base_payload(
        record.payload,
        event_kind="confirmed",
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
    )
    assert record.payload["trigger"] == "unit_test"


@pytest.mark.asyncio
async def test_emit_invalidated_keeps_event_kind_even_when_state_is_cooldown(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    strategy.set_invalidated(
        state,
        reason="invalidated",
        cooldown=True,
        metadata={"invalidation_source": "unit_test"},
    )

    await strategy.emit_invalidated(
        state,
        extra_payload={"trigger": "unit_test", "correlation_id": "corr-invalidated"},
    )

    record = _last_record(event_bus_spy)
    assert record.topic == "strategy.funding.test.invalidated"
    assert record.priority == base_strategy_config.invalidation_priority
    assert record.correlation_id == "corr-invalidated"

    _assert_base_payload(
        record.payload,
        event_kind="invalidated",
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
    )
    assert record.payload["reason"] == "invalidated"
    assert record.payload["metadata"]["invalidation_source"] == "unit_test"


@pytest.mark.asyncio
async def test_emit_respects_disabled_setup_events(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    base_strategy_config.emit_setup_events = False
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    await strategy.emit_setup(state, extra_payload={"trigger": "unit_test"})

    assert event_bus_spy.emitted == []
    assert state.emit_count == 0
    assert state.last_emit_time is None


@pytest.mark.asyncio
async def test_emit_returns_false_and_does_not_increment_emit_count_when_event_bus_rejects(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    event_bus_spy.raise_on_emit = RuntimeError("queue rejected")

    accepted = await strategy._emit(
        event_name="strategy.funding.test.setup",
        payload=strategy.build_base_signal_payload(
            state,
            event_kind="setup",
            extra_payload={"trigger": "unit_test"},
        ),
    )

    assert accepted is False
    assert event_bus_spy.emitted == []
    assert state.emit_count == 0
    assert state.last_emit_time is None


@pytest.mark.asyncio
async def test_emit_returns_false_and_does_not_increment_emit_count_on_unexpected_error(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    event_bus_spy.raise_on_emit = ValueError("unexpected emit failure")

    accepted = await strategy._emit(
        event_name="strategy.funding.test.setup",
        payload=strategy.build_base_signal_payload(
            state,
            event_kind="setup",
            extra_payload={"trigger": "unit_test"},
        ),
    )

    assert accepted is False
    assert event_bus_spy.emitted == []
    assert state.emit_count == 0
    assert state.last_emit_time is None


def test_build_base_signal_payload_includes_funding_and_analytics_context(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    regime_payload: Any = None,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)

    strategy.attach_regime(
        state,
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "regime": FundingRegime.POSITIVE.value,
            "bias": FundingBias.LONG_BIAS.value,
            "confidence": 0.80,
        },
    )
    strategy.attach_pressure(
        state,
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "direction": FundingPressureDirection.LONG.value,
            "level": FundingPressureLevel.HIGH.value,
            "pressure_score": 0.77,
            "squeeze_probability": 0.65,
            "mean_reversion_probability": 0.60,
        },
    )

    payload = strategy.build_base_signal_payload(
        state,
        event_kind="setup",
        extra_payload={"trigger": "unit_test"},
    )

    assert payload["funding_context"]["regime"] == FundingRegime.POSITIVE.value
    assert payload["funding_context"]["bias"] == FundingBias.LONG_BIAS.value
    assert payload["funding_context"]["pressure_direction"] == FundingPressureDirection.LONG.value
    assert payload["funding_context"]["pressure_level"] == FundingPressureLevel.HIGH.value
    assert payload["analytics_context"]["regime"] is not None
    assert payload["analytics_context"]["pressure"] is not None
    assert payload["trigger"] == "unit_test"


# =============================================================================
# Base normalized analytics handlers
# =============================================================================


@pytest.mark.asyncio
async def test_on_funding_updated_attaches_context_and_calls_hook(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_funding_updated_event: Any,
    regime_payload: Any,
    pressure_payload: Any,
    positive_extreme_payload: Any,
    bullish_divergence_payload: Any,
    negative_to_positive_flip_payload: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    event = make_funding_updated_event(
        regime_state=regime_payload(
            regime=FundingRegime.POSITIVE,
            bias=FundingBias.LONG_BIAS,
            confidence=0.80,
        ),
        pressure_state=pressure_payload(
            direction=FundingPressureDirection.LONG,
            level=FundingPressureLevel.HIGH,
            pressure_score=0.78,
        ),
        extreme_event=positive_extreme_payload(severity=0.88),
        divergence_event=bullish_divergence_payload(confidence=0.79),
        flip_event=negative_to_positive_flip_payload(confidence=0.82),
    )

    await strategy.on_funding_updated(event)

    state = strategy.get_state("BTCUSDT", "binance")
    assert state.last_funding_updated_payload is not None
    assert state.last_analytics_update_time is not None
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_extreme is not None
    assert state.last_divergence is not None
    assert state.last_flip is not None

    assert len(strategy.updated_hooks_seen) == 1
    hook_state, hook_payload, hook_event = strategy.updated_hooks_seen[0]
    assert hook_state is state
    assert hook_event is event
    assert hook_payload["symbol"] == "BTCUSDT"
    assert hook_payload["exchange"] == "binance"


@pytest.mark.asyncio
async def test_on_funding_updated_ignores_payload_without_symbol(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_test_event: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    event = make_test_event(
        "analytics.funding.updated",
        {
            "payload": {
                "regime_state": {
                    "exchange": "binance",
                    "regime": FundingRegime.POSITIVE.value,
                }
            }
        },
    )

    await strategy.on_funding_updated(event)

    assert strategy.updated_hooks_seen == []
    assert strategy.get_all_states() == {}


@pytest.mark.asyncio
async def test_on_funding_signal_builds_signal_attaches_state_and_calls_hook(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_funding_signal_event: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    event = make_funding_signal_event(
        signal_type=FundingSignalType.REVERSION_SETUP,
        bias=FundingBias.LONG_BIAS,
        score=0.75,
        confidence=0.84,
    )

    await strategy.on_funding_signal(event)

    state = strategy.get_state("BTCUSDT", "binance")
    assert state.last_signal is not None
    assert state.last_signal_time is not None

    assert len(strategy.signal_hooks_seen) == 1
    hook_state, hook_signal, hook_event = strategy.signal_hooks_seen[0]
    assert hook_state is state
    assert hook_event is event

    signal_type = getattr(hook_signal, "signal_type", None)
    score = getattr(hook_signal, "score", None)
    confidence = getattr(hook_signal, "confidence", None)

    if signal_type is None and isinstance(hook_signal, dict):
        signal_type = hook_signal.get("signal_type")
        score = hook_signal.get("score")
        confidence = hook_signal.get("confidence")

    assert signal_type is not None
    assert float(score) == pytest.approx(0.75)
    assert float(confidence) == pytest.approx(0.84)


@pytest.mark.asyncio
async def test_on_funding_signal_ignores_payload_without_symbol(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_test_event: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )

    event = make_test_event(
        "analytics.funding.signal",
        {
            "exchange": "binance",
            "signal_type": FundingSignalType.REVERSION_SETUP.value,
            "score": 0.75,
            "confidence": 0.84,
        },
    )

    await strategy.on_funding_signal(event)

    assert strategy.signal_hooks_seen == []
    assert strategy.get_all_states() == {}


@pytest.mark.asyncio
async def test_base_handlers_expire_active_state_before_calling_hook(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
    make_funding_signal_event: Any,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    state.expires_at = utc_now() - timedelta(seconds=1.0)

    await strategy.on_funding_signal(make_funding_signal_event())

    assert state.status == FundingSetupStatus.COOLDOWN
    assert state.reason == "setup_expired"
    assert len(strategy.signal_hooks_seen) == 1


# =============================================================================
# Reset / stats
# =============================================================================


def test_reset_state_can_preserve_cooldown_and_context(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    strategy.attach_regime(
        state,
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "regime": FundingRegime.POSITIVE.value,
            "bias": FundingBias.LONG_BIAS.value,
            "confidence": 0.80,
        },
    )
    strategy.set_cooldown(state, cooldown_sec=60.0, reason="cooldown_before_reset")
    cooldown_until = state.cooldown_until

    reset = strategy.reset_state(
        "BTCUSDT",
        "binance",
        preserve_cooldown=True,
        preserve_context=True,
    )

    assert reset is not state
    assert reset.status == FundingSetupStatus.COOLDOWN
    assert reset.cooldown_until == cooldown_until
    assert reset.last_regime is not None
    assert reset.reason is None
    assert reset.setup_type is None


def test_reset_state_can_drop_cooldown_and_context(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    state = _make_setup_state(strategy)
    strategy.attach_regime(
        state,
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "regime": FundingRegime.POSITIVE.value,
            "bias": FundingBias.LONG_BIAS.value,
            "confidence": 0.80,
        },
    )
    strategy.set_cooldown(state, cooldown_sec=60.0, reason="cooldown_before_reset")

    reset = strategy.reset_state(
        "BTCUSDT",
        "binance",
        preserve_cooldown=False,
        preserve_context=False,
    )

    assert reset.status == FundingSetupStatus.IDLE
    assert reset.cooldown_until is None
    assert reset.last_regime is None
    assert reset.reason is None


def test_stats_reports_runtime_counters(
    event_bus_spy: Any,
    scheduler_spy: Any,
    base_strategy_config: BaseFundingStrategyConfig,
) -> None:
    strategy = _make_strategy(
        event_bus_spy=event_bus_spy,
        scheduler_spy=scheduler_spy,
        config=base_strategy_config,
    )
    _make_setup_state(strategy)
    strategy.get_state("ETHUSDT", "binance")

    stats = strategy.stats()

    assert stats["strategy"] == "pytest_funding_runtime"
    assert stats["namespace"] == "strategy.funding.test"
    assert stats["registered"] is False
    assert stats["running"] is False
    assert stats["stopping"] is False
    assert stats["subscriptions"] == 0
    assert stats["states_total"] == 2
    assert stats["states_active"] == 1
    assert stats["locks"] == 0
    assert stats["cleanup_job_id"] is None