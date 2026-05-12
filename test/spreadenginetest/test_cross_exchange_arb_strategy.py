# test/spreadenginetest/test_cross_exchange_arb_strategy.py

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from analytics.spreads.enums import (
    InstrumentType,
    SpreadSignalType,
    SpreadType,
)

from strategy.strategies.spreads import (
    ARBITRAGE_OPPORTUNITY_EVENT,
    CROSS_EXCHANGE_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT,
    STATE_CANCELLED,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_PENDING,
    STATE_REJECTED,
    STRATEGY_SIGNAL_CANCELLED_EVENT,
    STRATEGY_SIGNAL_CLOSED_EVENT,
    STRATEGY_SIGNAL_GENERATED_EVENT,
    STRATEGY_SIGNAL_REJECTED_EVENT,
    STRATEGY_SIGNAL_UPDATED_EVENT,
)

try:
    from factories import (
        d,
        active_status,
        expired_status,
        futures_type,
        make_arb_data_quality_signal,
        make_arb_signal,
        make_arb_snapshot_edge_lost,
        make_arb_snapshot_inactive,
        make_arbitrage_opportunity,
        make_cross_exchange_snapshot,
        make_expired_arb_opportunity,
        make_inactive_arb_opportunity,
        make_spread_signal,
        make_stale_arb_snapshot,
        make_valid_arb_opportunity,
        stale_time,
        utcnow,
    )
except ImportError:
    from .factories import (
        d,
        active_status,
        expired_status,
        futures_type,
        make_arb_data_quality_signal,
        make_arb_signal,
        make_arb_snapshot_edge_lost,
        make_arb_snapshot_inactive,
        make_arbitrage_opportunity,
        make_cross_exchange_snapshot,
        make_expired_arb_opportunity,
        make_inactive_arb_opportunity,
        make_spread_signal,
        make_stale_arb_snapshot,
        make_valid_arb_opportunity,
        stale_time,
        utcnow,
    )


pytestmark = pytest.mark.asyncio


# ============================================================
# Open / update / close lifecycle
# ============================================================

async def test_valid_opportunity_opens_arb_signal(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opportunities_received"] == 1
    assert stats["opened_setups"] == 1
    assert len(strategy.active_states) == 1

    state = strategy.active_states[0]
    assert state.status == STATE_OPEN
    assert state.symbol == "BTCUSDT"
    assert state.exchange_a == "binance"
    assert state.exchange_b == "bybit"
    assert state.bias == "arb"
    assert state.entry_net_edge == d("65")
    assert state.entry_value == d("16")
    assert state.confidence == d("0.90")
    assert state.last_reason == "open_arbitrage_setup"

    assert emitted[0]["strategy"] == "cross_exchange_arb"
    assert emitted[0]["action"] == "OPEN_ARB"
    assert emitted[0]["symbol"] == "BTCUSDT"
    assert emitted[0]["exchange_a"] == "binance"
    assert emitted[0]["exchange_b"] == "bybit"
    assert emitted[0]["spread_type"] == SpreadType.CROSS_EXCHANGE.value
    assert emitted[0]["reason"] == "profitable_active_arbitrage_opportunity"


async def test_profitable_opportunity_below_entry_bps_is_rejected(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity(
        spread_bps="2",
        net_edge="65",
        is_profitable=True,
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["rejected_setups"] == 1
    assert len(strategy.closed_states) == 1
    assert strategy.closed_states[0].status == STATE_REJECTED

    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "opportunity_not_tradeable"


async def test_unprofitable_opportunity_is_rejected(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_arbitrage_opportunity(
        net_edge="0",
        spread_bps="0",
        is_profitable=False,
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["rejected_setups"] == 1

    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "opportunity_not_tradeable"


async def test_active_arb_updates_when_edge_changes_enough(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    first = make_valid_arb_opportunity()
    second = make_valid_arb_opportunity(
        timestamp=first.timestamp + timedelta(seconds=1),
        net_edge="90",
        spread_bps="22",
        confidence="0.95",
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(second)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_UPDATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["updated_setups"] == 1
    assert len(strategy.active_states) == 1

    state = strategy.active_states[0]
    assert state.status == STATE_OPEN
    assert state.entry_net_edge == d("90")
    assert state.entry_value == d("22")
    assert state.confidence == d("0.95")
    assert state.last_reason == "update_arbitrage_setup"

    assert emitted[0]["action"] == "UPDATE_ARB"
    assert emitted[0]["reason"] == "arbitrage_setup_updated"


async def test_active_arb_ignores_or_rejects_deteriorated_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    first = make_valid_arb_opportunity()
    deteriorated = make_valid_arb_opportunity(
        timestamp=first.timestamp + timedelta(seconds=1),
        net_edge="0",
        spread_bps="0",
        confidence="0.90",
        is_profitable=True,
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(deteriorated)

    await _flush_event_bus()

    stats = strategy.get_stats()

    assert stats["opened_setups"] == 1
    assert stats["opportunities_received"] == 2
    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_OPEN

    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 1


# ============================================================
# Cancel / inactive / expired
# ============================================================

async def test_expired_opportunity_cancels_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    first = make_valid_arb_opportunity()
    expired = make_expired_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(expired)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CANCELLED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["expired_opportunities"] == 1
    assert stats["cancelled_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1
    assert strategy.closed_states[0].status == STATE_CANCELLED

    assert emitted[0]["action"] == "CANCEL_ARB"
    assert emitted[0]["reason"] == "opportunity_expired"


async def test_inactive_opportunity_cancels_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    first = make_valid_arb_opportunity()
    inactive = make_inactive_arb_opportunity(
        timestamp=first.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(inactive)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CANCELLED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["inactive_opportunities"] == 1
    assert stats["cancelled_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1
    assert strategy.closed_states[0].status == STATE_CANCELLED

    assert emitted[0]["action"] == "CANCEL_ARB"
    assert emitted[0]["reason"] == "opportunity_not_active"


async def test_expired_opportunity_without_active_state_only_counts_expired(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_expired_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["expired_opportunities"] == 1
    assert stats["cancelled_setups"] == 0
    assert stats["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_CANCELLED_EVENT) == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


# ============================================================
# Persistence / arbitrage confirmation
# ============================================================

async def test_persistence_required_keeps_pending_before_open(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_persistence = True
    strategy.config.min_persistence_ms = 500

    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["persistence_waits"] == 1
    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_PENDING
    assert strategy.active_states[0].last_reason == "waiting_persistence_confirmation"
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_persistence_required_keeps_pending_until_confirmed(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_persistence = True
    strategy.config.min_persistence_ms = 500

    first = make_valid_arb_opportunity()
    second = make_valid_arb_opportunity(
        timestamp=first.timestamp + timedelta(milliseconds=600),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(second)

    await _flush_event_bus()

    stats = strategy.get_stats()

    assert stats["persistence_waits"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) in {0, 1}

    if stats["opened_setups"] == 1:
        assert len(strategy.active_states) == 1
        assert strategy.active_states[0].status == STATE_OPEN
    else:
        assert stats["opened_setups"] == 0
        assert len(strategy.active_states) == 1
        assert strategy.active_states[0].status == STATE_PENDING


async def test_arbitrage_confirmation_required_blocks_without_signal(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_arbitrage_signal_confirmation = True

    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["persistence_waits"] == 1
    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_PENDING
    assert strategy.active_states[0].last_reason == "waiting_arbitrage_signal_confirmation"
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_arbitrage_confirmation_signal_allows_open(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_arbitrage_signal_confirmation = True

    base_time = utcnow()
    signal = make_arb_signal(timestamp=base_time)
    opportunity = make_valid_arb_opportunity(
        timestamp=base_time + timedelta(milliseconds=100),
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["arbitrage_signal_confirmations"] == 1
    assert stats["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_ARB"


async def test_stale_arbitrage_confirmation_does_not_allow_open(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_arbitrage_signal_confirmation = True

    signal = make_arb_signal(timestamp=stale_time(seconds=30))
    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.on_spread_signal(signal)
    await strategy.on_arbitrage_opportunity(opportunity)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["ignored_signals"] == 1
    assert stats["arbitrage_signal_confirmations"] == 0
    assert stats["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


# ============================================================
# SpreadSignal behavior / signal bucket / invalid correlation
# ============================================================

async def test_arb_signal_is_stored_and_counted(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    signal = make_arb_signal()

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["arbitrage_signal_confirmations"] == 1
    assert stats["tracked_signal_keys"] == 1
    assert stats["tracked_signals"] == 1


async def test_wrong_spread_type_signal_is_ignored(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    signal = make_spread_signal(
        spread_type=SpreadType.SPOT_FUTURES,
        signal_type=SpreadSignalType.MEAN_REVERSION,
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["ignored_signals"] == 1
    assert stats["tracked_signals"] == 0


async def test_uncorrelatable_signal_is_ignored(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    signal = make_arb_signal(
        exchange_a=None,
        exchange_b=None,
        metadata={},
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["ignored_signals"] == 1
    assert stats["invalid_contracts"] == 1
    assert stats["tracked_signals"] == 0


async def test_signal_bucket_is_limited_to_max_signals_per_key(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.max_signals_per_key = 3

    base_time = utcnow()

    await strategy.start()

    for index in range(5):
        signal = make_arb_signal(
            timestamp=base_time + timedelta(milliseconds=index),
            message=f"arb-signal-{index}",
        )
        await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 5
    assert stats["tracked_signal_keys"] == 1
    assert stats["tracked_signals"] == 3
    assert stats["arbitrage_signal_confirmations"] == 5

    key = strategy._build_state_key(
        "BTCUSDT",
        "binance",
        "bybit",
        futures_type().value,
    )
    stored = strategy._latest_signals[key]
    assert [signal.message for signal in stored] == [
        "arb-signal-2",
        "arb-signal-3",
        "arb-signal-4",
    ]


# ============================================================
# Data quality guards
# ============================================================

async def test_data_quality_signal_blocks_new_entry(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    base_time = utcnow()

    signal = make_arb_data_quality_signal(
        timestamp=base_time,
        signal_type=SpreadSignalType.STALE_DATA,
    )
    opportunity = make_valid_arb_opportunity(
        timestamp=base_time + timedelta(milliseconds=100),
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)
    await strategy.on_arbitrage_opportunity(opportunity)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["data_quality_blocks"] >= 1
    assert stats["ignored_opportunities"] >= 1
    assert stats["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_data_quality_signal_cancels_active_setup_from_signal_handler(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    signal = make_arb_data_quality_signal(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
        signal_type=SpreadSignalType.INVALID_DATA,
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    assert strategy.get_stats()["opened_setups"] == 1
    assert len(strategy.active_states) == 1

    await strategy.on_spread_signal(signal)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CANCELLED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["cancelled_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1
    assert strategy.closed_states[0].status == STATE_CANCELLED
    assert strategy.closed_states[0].last_reason == "data_quality_signal_invalid_data"

    assert emitted[0]["action"] == "CANCEL_ARB"
    assert emitted[0]["reason"] == "data_quality_signal_invalid_data"


async def test_data_quality_signal_cancels_active_setup_from_opportunity_guard(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    first = make_valid_arb_opportunity()
    signal = make_arb_data_quality_signal(
        timestamp=first.timestamp + timedelta(seconds=1),
        signal_type=SpreadSignalType.STALE_DATA,
    )
    second = make_valid_arb_opportunity(
        timestamp=first.timestamp + timedelta(seconds=2),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_spread_signal(signal)
    await strategy.on_arbitrage_opportunity(second)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CANCELLED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["cancelled_setups"] >= 1
    assert strategy.active_states == []
    assert emitted[0]["action"] == "CANCEL_ARB"


# ============================================================
# Snapshot-driven lifecycle
# ============================================================

async def test_cross_exchange_snapshot_edge_loss_closes_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    snapshot = make_arb_snapshot_edge_lost(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)
    await strategy.on_cross_exchange_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["snapshot_edge_closes"] == 1
    assert stats["closed_setups"] == 1
    assert strategy.active_states == []
    assert strategy.closed_states[0].status == STATE_CLOSED

    assert emitted[0]["action"] == "CLOSE_ARB"
    assert emitted[0]["reason"] == "snapshot_edge_below_exit_threshold"


async def test_cross_exchange_snapshot_status_inactive_closes_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    snapshot = make_arb_snapshot_inactive(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)
    await strategy.on_cross_exchange_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["snapshot_status_closes"] == 1
    assert stats["closed_setups"] == 1
    assert strategy.active_states == []

    assert emitted[0]["action"] == "CLOSE_ARB"
    assert emitted[0]["reason"] == "snapshot_opportunity_status_not_active"


async def test_stale_snapshot_closes_active_setup_when_enabled(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    snapshot = make_stale_arb_snapshot()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)
    await strategy.on_cross_exchange_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["freshness_skips"] >= 1
    assert stats["closed_setups"] == 1
    assert strategy.active_states == []

    assert emitted[0]["action"] == "CLOSE_ARB"
    assert emitted[0]["reason"] == "snapshot_stale_for_active_setup"


async def test_cross_exchange_snapshot_updates_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    snapshot = make_cross_exchange_snapshot(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
        opportunity_net_edge="90",
        opportunity_net_edge_bps="22",
        opportunity_confidence="0.97",
        spread_bps="22",
        net_spread="90",
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)
    await strategy.on_cross_exchange_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_UPDATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["snapshot_updates"] == 1
    assert stats["updated_setups"] == 1
    assert len(strategy.active_states) == 1

    state = strategy.active_states[0]
    assert state.status == STATE_OPEN
    assert state.entry_net_edge == d("90")
    assert state.entry_value == d("22")

    assert emitted[0]["action"] == "UPDATE_ARB"
    assert emitted[0]["reason"] == "arbitrage_setup_updated_from_snapshot"


async def test_snapshot_missing_arbitrage_metadata_is_ignored_as_invalid_contract(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    snapshot = make_cross_exchange_snapshot(metadata={})
    snapshot.metadata.clear()

    await strategy.start()
    await strategy.on_cross_exchange_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["snapshots_received"] == 1
    assert stats["invalid_contracts"] == 1
    assert stats["ignored_snapshots"] == 1
    assert signal_collector.count(STRATEGY_SIGNAL_CLOSED_EVENT) == 0


# ============================================================
# Guards / rejects
# ============================================================

async def test_invalid_opportunity_payload_increments_invalid_payloads(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy

    await strategy.start()
    await strategy.on_arbitrage_opportunity({"not": "ArbitrageOpportunity"})

    stats = strategy.get_stats()
    assert stats["invalid_payloads"] == 1
    assert stats["opportunities_received"] == 0


async def test_invalid_snapshot_payload_increments_invalid_payloads(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy

    await strategy.start()
    await strategy.on_cross_exchange_snapshot({"not": "SpreadSnapshot"})

    stats = strategy.get_stats()
    assert stats["invalid_payloads"] == 1
    assert stats["snapshots_received"] == 0


async def test_invalid_signal_payload_increments_invalid_payloads(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy

    await strategy.start()
    await strategy.on_spread_signal({"not": "SpreadSignal"})

    stats = strategy.get_stats()
    assert stats["invalid_payloads"] == 1
    assert stats["spread_signals_received"] == 0


async def test_symbol_allowlist_rejects_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.allowed_symbols = {"ETHUSDT"}

    opportunity = make_valid_arb_opportunity(symbol="BTCUSDT")

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["symbol_skips"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "symbol_not_allowed"


async def test_exchange_allowlist_rejects_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.allowed_exchanges = {"okx"}

    opportunity = make_valid_arb_opportunity(
        buy_exchange="binance",
        sell_exchange="bybit",
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["exchange_skips"] >= 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "exchange_not_allowed"


async def test_instrument_type_allowlist_rejects_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.allowed_instrument_types = {"spot"}

    opportunity = make_valid_arb_opportunity(
        buy_instrument_type=futures_type(),
        sell_instrument_type=futures_type(),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "instrument_type_not_allowed"


async def test_low_confidence_rejects_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.min_confidence = Decimal("0.80")

    opportunity = make_valid_arb_opportunity(confidence="0.50")

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["confidence_skips"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_ARB"
    assert emitted[0]["reason"] == "confidence_below_threshold"


async def test_duplicate_opportunity_is_ignored(
    cross_arb_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = cross_arb_strategy
    timestamp = utcnow()

    first = make_valid_arb_opportunity(timestamp=timestamp)
    duplicate = make_valid_arb_opportunity(timestamp=timestamp)

    await strategy.start()
    await strategy.on_arbitrage_opportunity(first)
    await strategy.on_arbitrage_opportunity(duplicate)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opportunities_received"] == 2
    assert stats["duplicate_skips"] == 1
    assert stats["opened_setups"] == 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 1


# ============================================================
# Public read API
# ============================================================

async def test_get_latest_opportunity_returns_last_seen_opportunity(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)

    latest = strategy.get_latest_opportunity(
        symbol="btc/usdt",
        buy_exchange="BINANCE",
        sell_exchange="BYBIT",
        instrument_type=futures_type(),
    )

    assert latest is opportunity


async def test_get_latest_snapshot_returns_last_seen_snapshot(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity()
    snapshot = make_cross_exchange_snapshot(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_arbitrage_opportunity(opportunity)
    await strategy.on_cross_exchange_snapshot(snapshot)

    latest = strategy.get_latest_snapshot(
        symbol="btc/usdt",
        buy_exchange="BINANCE",
        sell_exchange="BYBIT",
        instrument_type=futures_type(),
    )

    assert latest is snapshot


async def test_prunes_stale_signals_from_bucket(
    cross_arb_strategy: Any,
) -> None:
    strategy = cross_arb_strategy

    stale_signal = make_arb_signal(timestamp=stale_time(seconds=30))
    fresh_signal = make_arb_signal(timestamp=utcnow())

    key = strategy._build_state_key(
        "BTCUSDT",
        "binance",
        "bybit",
        futures_type().value,
    )
    strategy._latest_signals[key] = [stale_signal, fresh_signal]

    removed = strategy._prune_stale_signals(key)

    assert removed == 1
    assert strategy._latest_signals[key] == [fresh_signal]
    assert strategy.get_stats()["stale_signals_removed"] == 1


# ============================================================
# EventBus integration
# ============================================================

async def test_eventbus_opportunity_flow_generates_open_arb_signal(
    cross_arb_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity()

    await strategy.start()

    await event_bus.emit(
        ARBITRAGE_OPPORTUNITY_EVENT,
        opportunity,
    )

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_ARB"


async def test_eventbus_signal_then_opportunity_flow_uses_confirmation(
    cross_arb_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = cross_arb_strategy
    strategy.config.require_arbitrage_signal_confirmation = True

    base_time = utcnow()
    signal = make_arb_signal(timestamp=base_time)
    opportunity = make_valid_arb_opportunity(
        timestamp=base_time + timedelta(milliseconds=100),
    )

    await strategy.start()

    await event_bus.emit(SPREAD_SIGNAL_EVENT, signal)
    await event_bus.emit(ARBITRAGE_OPPORTUNITY_EVENT, opportunity)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["arbitrage_signal_confirmations"] == 1
    assert stats["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_ARB"


async def test_eventbus_snapshot_edge_loss_flow_closes_active_setup(
    cross_arb_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = cross_arb_strategy

    opportunity = make_valid_arb_opportunity()
    snapshot = make_arb_snapshot_edge_lost(
        timestamp=opportunity.timestamp + timedelta(seconds=1),
    )

    await strategy.start()

    await event_bus.emit(ARBITRAGE_OPPORTUNITY_EVENT, opportunity)
    await event_bus.emit(CROSS_EXCHANGE_SNAPSHOT_EVENT, snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["closed_setups"] == 1
    assert emitted[0]["action"] == "CLOSE_ARB"


async def test_stopped_strategy_does_not_process_eventbus_opportunity(
    cross_arb_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = cross_arb_strategy
    opportunity = make_valid_arb_opportunity()

    await strategy.start()
    await strategy.stop()

    await event_bus.emit(
        ARBITRAGE_OPPORTUNITY_EVENT,
        opportunity,
    )

    await _flush_event_bus()

    assert strategy.get_stats()["opportunities_received"] == 0
    assert strategy.get_stats()["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


# ============================================================
# Local helpers
# ============================================================

async def _flush_event_bus() -> None:
    await asyncio.sleep(0.05)