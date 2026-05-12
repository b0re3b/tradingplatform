# test/spreadenginetest/test_base_spread_strategy.py

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from strategy.strategies.spreads import (
    STATE_CLOSED,
    STATE_OPEN,
    STRATEGY_SIGNAL_GENERATED_EVENT,
)


pytestmark = pytest.mark.asyncio


DUMMY_TEST_EVENT = "analytics.spreads.dummy"
DUMMY_STRATEGY_NAME = "dummy_spread_strategy"


# ============================================================
# Lifecycle
# ============================================================

async def test_start_registers_strategy_once(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy

    assert strategy.is_running is False
    assert strategy.is_registered is False

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.is_registered is True
    assert len(strategy._subscriptions) == 1

    await strategy.start()

    assert strategy.is_running is True
    assert strategy.is_registered is True
    assert len(strategy._subscriptions) == 1


async def test_stop_prevents_processing_but_keeps_registration(
    dummy_spread_strategy: Any,
    event_bus: Any,
) -> None:
    strategy = dummy_spread_strategy

    await strategy.start()
    await strategy.stop()

    assert strategy.is_running is False
    assert strategy.is_registered is True
    assert len(strategy._subscriptions) == 1

    await event_bus.emit(
        DUMMY_TEST_EVENT,
        {"value": "must_not_be_processed"},
    )

    await _flush_event_bus()

    assert strategy.received_payloads == []


async def test_unregister_removes_subscriptions(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy

    await strategy.start()

    assert strategy.is_registered is True
    assert len(strategy._subscriptions) == 1

    await strategy.unregister()

    assert strategy.is_registered is False
    assert strategy._subscriptions == []


async def test_eventbus_payload_subscription_processes_only_when_running(
    dummy_spread_strategy: Any,
    event_bus: Any,
) -> None:
    strategy = dummy_spread_strategy

    await strategy.register()

    await event_bus.emit(
        DUMMY_TEST_EVENT,
        {"value": "ignored_before_start"},
    )
    await _flush_event_bus()

    assert strategy.received_payloads == []

    await strategy.start()

    await event_bus.emit(
        DUMMY_TEST_EVENT,
        {"value": "processed_after_start"},
    )
    await _flush_event_bus()

    assert strategy.received_payloads == [{"value": "processed_after_start"}]


# ============================================================
# Strategy signal contract
# ============================================================

async def test_emit_generated_signal_payload_has_required_contract(
    dummy_spread_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = dummy_spread_strategy
    now = datetime.utcnow()

    await strategy.start()

    payload = await strategy._emit_generated(
        action="OPEN_TEST",
        symbol="btc/usdt",
        state_key="BTCUSDT|binance|bybit",
        exchange_a="Binance",
        exchange_b="Bybit",
        reason="test_open",
        confidence=Decimal("0.91"),
        spread_type="test_spread",
        timestamp=now,
        metadata={"source": "unit_test"},
    )

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 1
    assert emitted[0] == payload

    assert payload["strategy"] == DUMMY_STRATEGY_NAME
    assert payload["action"] == "OPEN_TEST"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange_a"] == "binance"
    assert payload["exchange_b"] == "bybit"
    assert payload["state_key"] == "BTCUSDT|binance|bybit"
    assert payload["spread_type"] == "test_spread"
    assert payload["reason"] == "test_open"
    assert payload["confidence"] == "0.91"
    assert payload["timestamp"] == now.isoformat()
    assert payload["metadata"] == {"source": "unit_test"}

    stats = strategy.get_stats()
    assert stats["signals_generated"] == 1


# ============================================================
# Dedup / cooldown
# ============================================================

async def test_mark_event_seen_ignores_same_timestamp_for_same_key(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    timestamp = datetime.utcnow()
    key = "snapshot|BTCUSDT|binance|bybit"

    first_seen = strategy._mark_event_seen(key=key, timestamp=timestamp)
    second_seen = strategy._mark_event_seen(key=key, timestamp=timestamp)

    assert first_seen is False
    assert second_seen is True

    stats = strategy.get_stats()
    assert stats["duplicate_skips"] == 1


async def test_mark_event_seen_allows_newer_timestamp_for_same_key(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    key = "snapshot|BTCUSDT|binance|bybit"

    first_timestamp = datetime.utcnow()
    second_timestamp = first_timestamp + timedelta(milliseconds=1)

    assert strategy._mark_event_seen(key=key, timestamp=first_timestamp) is False
    assert strategy._mark_event_seen(key=key, timestamp=second_timestamp) is False

    stats = strategy.get_stats()
    assert stats["duplicate_skips"] == 0


async def test_cooldown_blocks_repeated_signal_before_window_expires(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    key = "BTCUSDT|binance|bybit"

    first_time = datetime.utcnow()
    second_time = first_time + timedelta(seconds=5)

    assert strategy._should_skip_by_cooldown(key=key, now=first_time) is False
    assert strategy._should_skip_by_cooldown(key=key, now=second_time) is True

    stats = strategy.get_stats()
    assert stats["cooldown_skips"] == 1


async def test_cooldown_allows_signal_after_window_expires(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    key = "BTCUSDT|binance|bybit"

    first_time = datetime.utcnow()
    second_time = first_time + timedelta(seconds=11)

    assert strategy._should_skip_by_cooldown(key=key, now=first_time) is False
    assert strategy._should_skip_by_cooldown(key=key, now=second_time) is False

    stats = strategy.get_stats()
    assert stats["cooldown_skips"] == 0


# ============================================================
# State / cleanup
# ============================================================

async def test_cleanup_removes_only_old_closed_states(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    now = datetime.utcnow()

    old_closed = strategy._get_or_create_state(
        key="old_closed",
        symbol="BTCUSDT",
        exchange_a="binance",
        exchange_b="bybit",
    )
    recent_closed = strategy._get_or_create_state(
        key="recent_closed",
        symbol="ETHUSDT",
        exchange_a="binance",
        exchange_b="okx",
    )
    active_state = strategy._get_or_create_state(
        key="active_state",
        symbol="SOLUSDT",
        exchange_a="binance",
        exchange_b="mexc",
    )

    strategy._set_state_closed(
        old_closed,
        status=STATE_CLOSED,
        reason="old_closed",
        now=now - timedelta(hours=2),
    )
    strategy._set_state_closed(
        recent_closed,
        status=STATE_CLOSED,
        reason="recent_closed",
        now=now - timedelta(minutes=10),
    )
    strategy._set_state_open(
        active_state,
        bias="test",
        reason="active",
        entry_value=Decimal("10"),
        confidence=Decimal("0.8"),
        now=now - timedelta(hours=3),
    )

    strategy._last_signal_times["old_closed"] = now - timedelta(hours=2)
    strategy._last_event_times["old_closed"] = now - timedelta(hours=2)

    removed = strategy.cleanup_closed_states(
        older_than_seconds=3_600,
        now=now,
    )

    assert removed == 1
    assert strategy._get_state("old_closed") is None

    recent = strategy._get_state("recent_closed")
    assert recent is not None
    assert recent.status == STATE_CLOSED

    active = strategy._get_state("active_state")
    assert active is not None
    assert active.status == STATE_OPEN

    assert "old_closed" not in strategy._last_signal_times
    assert "old_closed" not in strategy._last_event_times


async def test_cleanup_keeps_closed_state_without_closed_at(
    dummy_spread_strategy: Any,
) -> None:
    strategy = dummy_spread_strategy
    now = datetime.utcnow()

    state = strategy._get_or_create_state(
        key="closed_without_timestamp",
        symbol="BTCUSDT",
        exchange_a="binance",
        exchange_b="bybit",
    )
    state.status = STATE_CLOSED
    state.closed_at = None

    removed = strategy.cleanup_closed_states(
        older_than_seconds=0,
        now=now,
    )

    assert removed == 0
    assert strategy._get_state("closed_without_timestamp") is state


# ============================================================
# Local helpers
# ============================================================

async def _flush_event_bus() -> None:
    """
    Дає EventBus handlers шанс виконатись після emit().
    Якщо EventBus обробляє події синхронно всередині emit(),
    цей helper не зашкодить.
    """
    await asyncio.sleep(0.05)