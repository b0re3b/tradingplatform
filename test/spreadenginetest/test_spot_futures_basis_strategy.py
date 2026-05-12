# test/spreadenginetest/test_spot_futures_basis_strategy.py

from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from analytics.spreads.enums import (
    InstrumentType,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)

from strategy.strategies.spreads import (
    SPOT_FUTURES_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT,
    STATE_CLOSED,
    STATE_OPEN,
    STRATEGY_SIGNAL_CLOSED_EVENT,
    STRATEGY_SIGNAL_GENERATED_EVENT,
    STRATEGY_SIGNAL_REJECTED_EVENT,
    STRATEGY_SIGNAL_UPDATED_EVENT,
)

try:
    from factories import (
        d,
        futures_type,
        make_basis_close_snapshot,
        make_basis_data_quality_signal,
        make_basis_stop_snapshot,
        make_mean_reversion_signal,
        make_regime_shift_signal,
        make_spot_futures_snapshot,
        make_stale_basis_snapshot,
        make_valid_basis_snapshot,
        spot_type,
        stale_time,
        utcnow,
    )
except ImportError:
    from .factories import (
        d,
        futures_type,
        make_basis_close_snapshot,
        make_basis_data_quality_signal,
        make_basis_stop_snapshot,
        make_mean_reversion_signal,
        make_regime_shift_signal,
        make_spot_futures_snapshot,
        make_stale_basis_snapshot,
        make_valid_basis_snapshot,
        spot_type,
        stale_time,
        utcnow,
    )


pytestmark = pytest.mark.asyncio


# ============================================================
# Open behavior
# ============================================================

async def test_valid_snapshot_opens_basis_signal(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["opened_setups"] == 1
    assert len(strategy.active_states) == 1

    state = strategy.active_states[0]
    assert state.status == STATE_OPEN
    assert state.symbol == "BTCUSDT"
    assert state.exchange_a == "binance"
    assert state.exchange_b == "bybit"
    assert state.entry_zscore == d("2.5")
    assert state.entry_value == d("25")
    assert state.entry_net_edge == d("80")

    assert emitted[0]["strategy"] == "spot_futures_basis"
    assert emitted[0]["action"] == "OPEN_BASIS"
    assert emitted[0]["symbol"] == "BTCUSDT"
    assert emitted[0]["exchange_a"] == "binance"
    assert emitted[0]["exchange_b"] == "bybit"
    assert emitted[0]["spread_type"] == SpreadType.SPOT_FUTURES.value
    assert emitted[0]["reason"] == "mean_reversion_basis_setup"


async def test_negative_zscore_snapshot_opens_basis_signal(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot(
        zscore="-2.8",
        raw_spread="-100",
        net_spread="-95",
        funding_adjusted_spread="-80",
        spread_bps="-25",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["opened_setups"] == 1
    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_OPEN
    assert strategy.active_states[0].entry_zscore == d("-2.8")
    assert emitted[0]["action"] == "OPEN_BASIS"


async def test_snapshot_below_entry_threshold_is_ignored(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_spot_futures_snapshot(
        zscore="1.25",
        spread_bps="10",
        funding_adjusted_spread="20",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["snapshots_received"] == 1
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0
    assert strategy.active_states == []


async def test_snapshot_with_not_allowed_regime_is_ignored(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_spot_futures_snapshot(
        zscore="2.5",
        regime=SpreadRegime.NORMAL,
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_snapshot_with_low_funding_adjusted_edge_is_ignored(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.min_funding_adjusted_edge = Decimal("50")

    snapshot = make_valid_basis_snapshot(
        funding_adjusted_spread="10",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_snapshot_with_low_basis_abs_is_ignored(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.min_basis_abs = Decimal("200")

    snapshot = make_valid_basis_snapshot(
        raw_spread="50",
        net_spread="45",
        funding_adjusted_spread="40",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


# ============================================================
# Active lifecycle: reduce / close / stop / update
# ============================================================

async def test_active_basis_reduces_when_zscore_enters_reduce_zone(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot()

    reduce_snapshot = make_spot_futures_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
        zscore="1.0",
        regime=SpreadRegime.ELEVATED,
        basis="60",
        raw_spread="60",
        net_spread="55",
        spread_bps="12",
        funding_adjusted_spread="35",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spot_futures_snapshot(reduce_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_UPDATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()

    assert stats["opened_setups"] == 1
    assert stats["reduced_setups"] + stats["updated_setups"] == 1

    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_OPEN

    assert emitted
    assert emitted[0]["action"] in {"REDUCE_BASIS", "UPDATE_BASIS"}
    assert emitted[0]["reason"] in {
        "basis_reversion_progressed",
        "basis_state_updated",
    }

    if emitted[0]["action"] == "REDUCE_BASIS":
        assert stats["reduced_setups"] == 1
        assert strategy.active_states[0].last_reason == "reduce_basis_setup"
    else:
        assert stats["updated_setups"] == 1
        assert strategy.active_states[0].last_reason == "update_basis_setup"


async def test_active_basis_closes_on_mean_reversion(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot()
    close_snapshot = make_basis_close_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spot_futures_snapshot(close_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["closed_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1

    state = strategy.closed_states[0]
    assert state.status == STATE_CLOSED
    assert state.last_reason == "basis_mean_reverted"

    assert emitted[0]["action"] == "CLOSE_BASIS"
    assert emitted[0]["reason"] == "basis_mean_reverted"


async def test_active_basis_stops_on_worsening_zscore(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot()
    stop_snapshot = make_basis_stop_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spot_futures_snapshot(stop_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["stopped_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1

    state = strategy.closed_states[0]
    assert state.status == STATE_CLOSED
    assert state.last_reason == "basis_dislocation_worsened"

    assert emitted[0]["action"] == "STOP_BASIS"
    assert emitted[0]["reason"] == "basis_dislocation_worsened"


async def test_bias_flip_stops_active_setup(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot(
        zscore="2.5",
        raw_spread="100",
        net_spread="95",
        funding_adjusted_spread="80",
        spread_bps="25",
    )
    flip_snapshot = make_valid_basis_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
        zscore="-2.6",
        raw_spread="-100",
        net_spread="-95",
        funding_adjusted_spread="-80",
        spread_bps="-25",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spot_futures_snapshot(flip_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["bias_flips"] == 1
    assert stats["stopped_setups"] == 1
    assert strategy.active_states == []

    assert emitted[0]["action"] == "STOP_BASIS"
    assert emitted[0]["reason"] == "basis_bias_flipped"


async def test_active_basis_updates_when_confidence_changes(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot(
        metadata={"confidence": "0.70"},
    )
    update_snapshot = make_valid_basis_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
        zscore="2.6",
        spread_bps="27",
        funding_adjusted_spread="85",
        metadata={"confidence": "0.90"},
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spot_futures_snapshot(update_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_UPDATED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["updated_setups"] >= 1
    assert len(strategy.active_states) == 1
    assert strategy.active_states[0].status == STATE_OPEN

    assert emitted[-1]["action"] in {"UPDATE_BASIS", "REDUCE_BASIS"}


# ============================================================
# SpreadSignal / confirmations
# ============================================================

async def test_mean_reversion_confirmation_is_stored_and_counted(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    signal = make_mean_reversion_signal()

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["mean_reversion_confirmations"] == 1
    assert stats["tracked_signal_keys"] == 1
    assert stats["tracked_signals"] == 1

    stored = strategy.get_latest_signals(
        symbol="BTCUSDT",
        spot_exchange="binance",
        futures_exchange="bybit",
    )
    assert stored == [signal]


async def test_requires_mean_reversion_signal_before_opening(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.require_mean_reversion_signal = True

    first_snapshot = make_valid_basis_snapshot()
    signal = make_mean_reversion_signal(
        timestamp=first_snapshot.timestamp + timedelta(milliseconds=100),
    )
    second_snapshot = make_valid_basis_snapshot(
        timestamp=first_snapshot.timestamp + timedelta(seconds=1),
    )

    await strategy.start()

    await strategy.on_spot_futures_snapshot(first_snapshot)
    await _flush_event_bus()

    assert strategy.get_stats()["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0

    await strategy.on_spread_signal(signal)
    await strategy.on_spot_futures_snapshot(second_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_BASIS"


async def test_regime_shift_signal_blocks_entry_when_not_allowed(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.allow_regime_shift_entry = False

    snapshot_time = utcnow()
    signal = make_regime_shift_signal(timestamp=snapshot_time)
    snapshot = make_valid_basis_snapshot(
        timestamp=snapshot_time + timedelta(milliseconds=100),
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["regime_shift_confirmations"] == 1
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_stale_signal_is_ignored_and_not_stored(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    signal = make_mean_reversion_signal(
        timestamp=stale_time(seconds=30),
    )

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["ignored_signals"] == 1
    assert stats["freshness_skips"] == 1
    assert stats["tracked_signals"] == 0


async def test_uncorrelatable_signal_is_ignored(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    signal = make_mean_reversion_signal(
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


async def test_wrong_spread_type_signal_is_ignored(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    signal = make_mean_reversion_signal()
    _set_attr(signal, "spread_type", SpreadType.CROSS_EXCHANGE)

    await strategy.start()
    await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 1
    assert stats["ignored_signals"] == 1
    assert stats["tracked_signals"] == 0


async def test_signal_bucket_is_limited_to_max_signals_per_key(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.max_signals_per_key = 3

    base_time = utcnow()

    await strategy.start()

    for index in range(5):
        signal = make_mean_reversion_signal(
            timestamp=base_time + timedelta(milliseconds=index),
            message=f"signal-{index}",
        )
        await strategy.on_spread_signal(signal)

    stats = strategy.get_stats()
    assert stats["spread_signals_received"] == 5
    assert stats["tracked_signal_keys"] == 1
    assert stats["tracked_signals"] == 3

    stored = strategy.get_latest_signals(
        symbol="BTCUSDT",
        spot_exchange="binance",
        futures_exchange="bybit",
    )
    assert len(stored) == 3
    assert [signal.message for signal in stored] == [
        "signal-2",
        "signal-3",
        "signal-4",
    ]


# ============================================================
# Data quality guards
# ============================================================

async def test_data_quality_signal_blocks_new_entry(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    base_time = utcnow()

    data_quality_signal = make_basis_data_quality_signal(
        timestamp=base_time,
        signal_type=SpreadSignalType.STALE_DATA,
    )
    snapshot = make_valid_basis_snapshot(
        timestamp=base_time + timedelta(milliseconds=100),
    )

    await strategy.start()
    await strategy.on_spread_signal(data_quality_signal)
    await strategy.on_spot_futures_snapshot(snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["data_quality_blocks"] >= 1
    assert stats["opened_setups"] == 0
    assert stats["ignored_snapshots"] >= 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


async def test_data_quality_signal_stops_active_setup_from_signal_handler(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    open_snapshot = make_valid_basis_snapshot()
    data_quality_signal = make_basis_data_quality_signal(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
        signal_type=SpreadSignalType.INVALID_DATA,
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)

    assert strategy.get_stats()["opened_setups"] == 1
    assert len(strategy.active_states) == 1

    await strategy.on_spread_signal(data_quality_signal)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["data_quality_blocks"] >= 1
    assert stats["stopped_setups"] == 1
    assert strategy.active_states == []
    assert len(strategy.closed_states) == 1

    assert emitted[0]["action"] == "STOP_BASIS"
    assert emitted[0]["reason"] == "data_quality_signal_invalid_data"


async def test_data_quality_signal_stops_active_setup_from_snapshot_guard(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy

    open_snapshot = make_valid_basis_snapshot()
    data_quality_signal = make_basis_data_quality_signal(
        timestamp=open_snapshot.timestamp + timedelta(seconds=1),
        signal_type=SpreadSignalType.STALE_DATA,
    )
    next_snapshot = make_valid_basis_snapshot(
        timestamp=open_snapshot.timestamp + timedelta(seconds=2),
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(open_snapshot)
    await strategy.on_spread_signal(data_quality_signal)
    await strategy.on_spot_futures_snapshot(next_snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_CLOSED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["opened_setups"] == 1
    assert stats["stopped_setups"] >= 1
    assert strategy.active_states == []
    assert emitted[0]["action"] == "STOP_BASIS"


# ============================================================
# Snapshot guards / invalid contracts / allowlists
# ============================================================

async def test_invalid_payload_increments_invalid_payloads(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy

    await strategy.start()
    await strategy.on_spot_futures_snapshot({"not": "a SpreadSnapshot"})

    stats = strategy.get_stats()
    assert stats["invalid_payloads"] == 1
    assert stats["snapshots_received"] == 0


async def test_snapshot_with_wrong_spread_type_is_rejected(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()
    _set_attr(snapshot, "spread_type", SpreadType.CROSS_EXCHANGE)

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["snapshots_received"] == 1
    assert stats["invalid_contracts"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_BASIS"
    assert emitted[0]["reason"] == "unsupported_spread_type"


async def test_snapshot_with_leg_a_not_spot_is_rejected(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()
    _set_attr(snapshot, "leg_a_type", futures_type())

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["invalid_contracts"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["reason"] == "leg_a_not_spot"


async def test_snapshot_with_leg_b_not_derivative_is_rejected(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()
    _set_attr(snapshot, "leg_b_type", spot_type())

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["invalid_contracts"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["reason"] == "leg_b_not_derivative"


async def test_stale_snapshot_is_rejected(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_stale_basis_snapshot()

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["snapshots_received"] == 1
    assert stats["freshness_skips"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["action"] == "REJECT_BASIS"
    assert emitted[0]["reason"] == "snapshot_stale"


async def test_symbol_allowlist_rejects_snapshot(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.allowed_symbols = {"ETHUSDT"}

    snapshot = make_valid_basis_snapshot(symbol="BTCUSDT")

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["symbol_skips"] == 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["reason"] == "symbol_not_allowed"


async def test_spot_exchange_allowlist_rejects_snapshot(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.allowed_spot_exchanges = {"okx"}

    snapshot = make_valid_basis_snapshot(
        spot_exchange="binance",
        futures_exchange="bybit",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["exchange_skips"] >= 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["reason"] == "exchange_not_allowed"


async def test_futures_exchange_allowlist_rejects_snapshot(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.allowed_futures_exchanges = {"okx"}

    snapshot = make_valid_basis_snapshot(
        spot_exchange="binance",
        futures_exchange="bybit",
    )

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_REJECTED_EVENT,
        count=1,
    )

    stats = strategy.get_stats()
    assert stats["exchange_skips"] >= 1
    assert stats["rejected_setups"] == 1
    assert emitted[0]["reason"] == "exchange_not_allowed"


async def test_duplicate_snapshot_is_ignored(
    spot_basis_strategy: Any,
    signal_collector: Any,
) -> None:
    strategy = spot_basis_strategy
    timestamp = utcnow()

    first_snapshot = make_valid_basis_snapshot(timestamp=timestamp)
    duplicate_snapshot = make_valid_basis_snapshot(timestamp=timestamp)

    await strategy.start()
    await strategy.on_spot_futures_snapshot(first_snapshot)
    await strategy.on_spot_futures_snapshot(duplicate_snapshot)

    await _flush_event_bus()

    stats = strategy.get_stats()
    assert stats["snapshots_received"] == 2
    assert stats["duplicate_skips"] == 1
    assert stats["opened_setups"] == 1
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 1


# ============================================================
# Public read API
# ============================================================

async def test_get_latest_snapshot_returns_last_valid_snapshot(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()

    await strategy.start()
    await strategy.on_spot_futures_snapshot(snapshot)

    latest = strategy.get_latest_snapshot(
        symbol="btc/usdt",
        spot_exchange="BINANCE",
        futures_exchange="BYBIT",
    )

    assert latest is snapshot


async def test_get_latest_signals_prunes_stale_signals(
    spot_basis_strategy: Any,
) -> None:
    strategy = spot_basis_strategy

    stale_signal = make_mean_reversion_signal(
        timestamp=stale_time(seconds=30),
    )
    fresh_signal = make_mean_reversion_signal(
        timestamp=utcnow(),
    )

    key = strategy._build_state_key("BTCUSDT", "binance", "bybit")
    strategy._latest_signals[key] = [stale_signal, fresh_signal]

    latest = strategy.get_latest_signals(
        symbol="BTCUSDT",
        spot_exchange="binance",
        futures_exchange="bybit",
    )

    assert latest == [fresh_signal]
    assert strategy.get_stats()["stale_signals_removed"] == 1


# ============================================================
# EventBus integration
# ============================================================

async def test_eventbus_snapshot_flow_generates_open_basis_signal(
    spot_basis_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()

    await strategy.start()

    await event_bus.emit(
        SPOT_FUTURES_SNAPSHOT_EVENT,
        snapshot,
    )

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_BASIS"


async def test_eventbus_signal_then_snapshot_flow_uses_confirmation(
    spot_basis_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = spot_basis_strategy
    strategy.config.require_mean_reversion_signal = True

    base_time = utcnow()
    signal = make_mean_reversion_signal(timestamp=base_time)
    snapshot = make_valid_basis_snapshot(
        timestamp=base_time + timedelta(milliseconds=100),
    )

    await strategy.start()

    await event_bus.emit(SPREAD_SIGNAL_EVENT, signal)
    await event_bus.emit(SPOT_FUTURES_SNAPSHOT_EVENT, snapshot)

    emitted = await signal_collector.wait_for(
        STRATEGY_SIGNAL_GENERATED_EVENT,
        count=1,
    )

    assert strategy.get_stats()["mean_reversion_confirmations"] == 1
    assert strategy.get_stats()["opened_setups"] == 1
    assert emitted[0]["action"] == "OPEN_BASIS"


async def test_stopped_strategy_does_not_process_eventbus_snapshot(
    spot_basis_strategy: Any,
    signal_collector: Any,
    event_bus: Any,
) -> None:
    strategy = spot_basis_strategy
    snapshot = make_valid_basis_snapshot()

    await strategy.start()
    await strategy.stop()

    await event_bus.emit(
        SPOT_FUTURES_SNAPSHOT_EVENT,
        snapshot,
    )

    await _flush_event_bus()

    assert strategy.get_stats()["snapshots_received"] == 0
    assert strategy.get_stats()["opened_setups"] == 0
    assert signal_collector.count(STRATEGY_SIGNAL_GENERATED_EVENT) == 0


# ============================================================
# Local helpers
# ============================================================

async def _flush_event_bus() -> None:
    await asyncio.sleep(0.05)


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """
    Тести не тестують models напряму, але іноді треба зіпсувати
    валідний factory object, щоб пройти negative path strategy.
    """
    try:
        setattr(obj, name, value)
    except Exception as exc:
        raise AssertionError(
            f"Cannot mutate test object attribute {name!r}; "
            "create a dedicated invalid factory for this model."
        ) from exc