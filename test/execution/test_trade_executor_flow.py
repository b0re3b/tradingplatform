from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from execution.config import (
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
)
from execution.enums import ExecutionMode, OrderSide, OrderStatus, OrderType, ExecutionStatus
from execution.order_manager import OrderManager
from execution.position_manager import PositionManager
from execution.sl_tp_manager import SLTPManager
from execution.smart_execution import SmartExecution
from execution.trade_executor import TradeExecutor

from risk.enums import OrderIntent, PositionSide, RiskMode


pytestmark = pytest.mark.asyncio


# =============================================================================
# Helpers
# =============================================================================


async def build_execution_stack(
    *,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    order_manager_config: OrderManagerConfig,
    position_manager_config: PositionManagerConfig,
    sltp_manager_config: SLTPManagerConfig,
    smart_execution_config: SmartExecutionConfig,
    trade_executor_config: TradeExecutorConfig,
    market_context_provider=None,
) -> tuple[
    OrderManager,
    PositionManager,
    SLTPManager,
    SmartExecution,
    TradeExecutor,
]:
    order_manager = OrderManager(
        order_manager_config,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    position_manager = PositionManager(
        position_manager_config,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    sltp_manager = SLTPManager(
        sltp_manager_config,
        order_manager=order_manager,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
    )

    smart_execution = SmartExecution(smart_execution_config)

    trade_executor = TradeExecutor(
        trade_executor_config,
        order_manager=order_manager,
        position_manager=position_manager,
        sltp_manager=sltp_manager,
        smart_execution=smart_execution,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        market_context_provider=market_context_provider,
    )

    # Start lower-level services first, orchestrator last.
    await order_manager.start()
    await position_manager.start()
    await sltp_manager.start()
    await trade_executor.start()

    return order_manager, position_manager, sltp_manager, smart_execution, trade_executor


def make_execution_stack_configs(
    *,
    smart_mode: ExecutionMode = ExecutionMode.MARKET,
    prefer_market_for_exits: bool = True,
    kill_switch_allows_reduce_only: bool = True,
) -> tuple[
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
]:
    order_config = OrderManagerConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        submit_retries=0,
        cancel_retries=0,
        reconcile_enabled=True,
        generate_client_order_id=True,
        emit_acknowledged_events=True,
        emit_partially_filled_events=True,
    )
    order_config.validate()

    position_config = PositionManagerConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        reconcile_enabled=True,
        emit_unchanged_snapshots=False,
        emit_pnl_updates=True,
    )
    position_config.validate()

    sltp_config = SLTPManagerConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        auto_place_on_position_opened=True,
        auto_cancel_on_position_closed=True,
        auto_resize_on_position_updated=True,
        use_close_position_for_full_stop=True,
        use_close_position_for_full_take_profit=False,
        require_reduce_only=True,
        trailing_stop_enabled=True,
        min_trailing_callback_rate=0.1,
        max_trailing_callback_rate=5.0,
    )
    sltp_config.validate()

    smart_config = SmartExecutionConfig(
        enabled=True,
        default_mode=smart_mode,
        fallback_mode=ExecutionMode.MARKET,
        prefer_limit_for_entries=False,
        prefer_market_for_exits=prefer_market_for_exits,
        allow_order_splitting=True,
        max_split_count=5,
        min_leg_notional=5.0,
        twap_enabled=True,
    )
    smart_config.validate()

    trade_config = TradeExecutorConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        auto_subscribe=True,
        register_scheduler_jobs=True,
        allow_new_entries=True,
        allow_position_reductions=True,
        allow_position_closes=True,
        reject_expired_risk_reservations=True,
        kill_switch_blocks_new_entries=True,
        kill_switch_cancels_open_orders=True,
        kill_switch_allows_reduce_only=kill_switch_allows_reduce_only,
        execution_timeout_seconds=30.0,
        max_concurrent_executions=10,
        per_symbol_execution_lock=True,
    )
    trade_config.validate()

    return order_config, position_config, sltp_config, smart_config, trade_config


async def emit_signal_confirmed(fake_event_bus, payload: Mapping[str, Any]) -> None:
    await fake_event_bus.emit("signal.confirmed", dict(payload))


# =============================================================================
# Full open flow
# =============================================================================


async def test_full_open_long_flow_creates_order_position_and_protective_orders(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    (
        order_manager,
        position_manager,
        sltp_manager,
        smart_execution,
        trade_executor,
    ) = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            final_size=0.01,
            final_leverage=5.0,
            final_notional=500.0,
            final_margin=100.0,
            final_risk_amount=50.0,
            stop_loss=49_000.0,
            take_profit=53_000.0,
            signal_id="sig-open-long",
            strategy_name="flow_strategy",
            reservation_id="res-open-long",
        ),
    )

    topics = fake_event_bus.topics()

    assert "execution.trade_requested" in topics
    assert "execution.trade_accepted" in topics
    assert "execution.execution_started" in topics
    assert "execution.execution_plan_created" in topics
    assert "execution.order_submit_started" in topics
    assert "execution.order_submitted" in topics
    assert "execution.order_filled" in topics
    assert "position.opened" in topics
    assert "execution.stop_loss_placed" in topics
    assert "execution.take_profit_placed" in topics

    # Binance calls:
    # 1 market entry + 1 stop-loss + 1 take-profit.
    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) == 3

    entry_call = create_calls[0].kwargs
    stop_call = create_calls[1].kwargs
    tp_call = create_calls[2].kwargs

    assert entry_call["symbol"] == "BTCUSDT"
    assert entry_call["side"] == "BUY"
    assert entry_call["order_type"] == "MARKET"
    assert entry_call["quantity"] == pytest.approx(0.01)
    assert entry_call["position_side"] == "LONG"
    assert entry_call["reduce_only"] is None
    assert entry_call["close_position"] is None

    assert stop_call["side"] == "SELL"
    assert stop_call["order_type"] == "STOP_MARKET"
    assert stop_call["position_side"] == "LONG"
    assert stop_call["quantity"] is None
    assert stop_call["close_position"] is True
    assert stop_call["stop_price"] == pytest.approx(49_000.0)

    assert tp_call["side"] == "SELL"
    assert tp_call["order_type"] == "TAKE_PROFIT_MARKET"
    assert tp_call["position_side"] == "LONG"
    assert tp_call["quantity"] == pytest.approx(0.01)
    assert tp_call["reduce_only"] is True
    assert tp_call["stop_price"] == pytest.approx(53_000.0)

    position = position_manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None
    assert position.is_open is True
    assert position.size == pytest.approx(0.01)
    assert position.entry_price == pytest.approx(50_000.0)
    assert position.signal_id == "sig-open-long"
    assert position.strategy_name == "flow_strategy"
    assert position.reservation_id == "res-open-long"

    protective_orders = sltp_manager.list_protective_orders(symbol="BTCUSDT")
    assert len(protective_orders) == 2

    assert order_manager.stats.submitted == 3
    assert position_manager.stats.opened == 1
    assert sltp_manager.stats.stop_loss_placed == 1
    assert sltp_manager.stats.take_profit_placed == 1
    assert smart_execution.stats.plans_created == 1
    assert trade_executor.stats.accepted == 1


async def test_risk_reservation_metadata_propagates_through_order_position_and_sltp(
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="reservation-critical-1",
            signal_id="signal-critical-1",
            strategy_name="reservation_strategy",
        ),
    )

    expected_reservation_id = "reservation-critical-1"

    assert fake_event_bus.last_payload("execution.trade_accepted")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("execution.execution_plan_created")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("execution.order_submitted")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("execution.order_filled")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("position.opened")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("execution.stop_loss_placed")["reservation_id"] == expected_reservation_id
    assert fake_event_bus.last_payload("execution.take_profit_placed")["reservation_id"] == expected_reservation_id

    stop_payload = fake_event_bus.last_payload("execution.stop_loss_placed")
    tp_payload = fake_event_bus.last_payload("execution.take_profit_placed")

    assert stop_payload["signal_id"] == "signal-critical-1"
    assert stop_payload["strategy_name"] == "reservation_strategy"
    assert tp_payload["signal_id"] == "signal-critical-1"
    assert tp_payload["strategy_name"] == "reservation_strategy"


async def test_full_open_short_flow_maps_entry_and_protective_sides_correctly(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, position_manager, sltp_manager, _, _ = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            order_intent=OrderIntent.OPEN,
            final_size=0.02,
            final_notional=1_000.0,
            stop_loss=51_000.0,
            take_profit=47_000.0,
            signal_id="sig-open-short",
            reservation_id="res-open-short",
        ),
    )

    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) == 3

    entry_call = create_calls[0].kwargs
    stop_call = create_calls[1].kwargs
    tp_call = create_calls[2].kwargs

    assert entry_call["side"] == "SELL"
    assert entry_call["order_type"] == "MARKET"
    assert entry_call["position_side"] == "SHORT"

    assert stop_call["side"] == "BUY"
    assert stop_call["order_type"] == "STOP_MARKET"
    assert stop_call["position_side"] == "SHORT"
    assert stop_call["stop_price"] == pytest.approx(51_000.0)

    assert tp_call["side"] == "BUY"
    assert tp_call["order_type"] == "TAKE_PROFIT_MARKET"
    assert tp_call["position_side"] == "SHORT"
    assert tp_call["stop_price"] == pytest.approx(47_000.0)

    position = position_manager.get_position(symbol="BTCUSDT", side=PositionSide.SHORT)
    assert position is not None
    assert position.is_open is True
    assert position.size == pytest.approx(0.02)

    protective_orders = sltp_manager.list_protective_orders(symbol="BTCUSDT")
    assert len(protective_orders) == 2


# =============================================================================
# Failure propagation
# =============================================================================


async def test_binance_create_order_failure_emits_order_failed_execution_failed_and_no_position_opened(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    fake_binance_client.create_order_exception = RuntimeError("binance create failed")

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, position_manager, sltp_manager, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            signal_id="sig-failure",
            reservation_id="res-failure",
        ),
    )

    assert fake_event_bus.count("execution.order_failed") == 1
    assert fake_event_bus.count("execution.execution_failed") == 1
    assert fake_event_bus.count("execution.trade_rejected") == 1

    order_failed = fake_event_bus.last_payload("execution.order_failed")
    execution_failed = fake_event_bus.last_payload("execution.execution_failed")

    assert order_failed["reservation_id"] == "res-failure"
    assert execution_failed["reservation_id"] == "res-failure"
    assert "binance create failed" in order_failed["error"]

    assert fake_event_bus.count("position.opened") == 0
    assert position_manager.open_positions == []
    assert sltp_manager.list_protective_orders(symbol="BTCUSDT") == []

    assert trade_executor.stats.failed >= 1 or trade_executor.stats.rejected >= 1


async def test_smart_execution_failure_emits_trade_rejected_and_does_not_call_binance(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.LIMIT)

    # Force bad limit context: no bid/ask/reference price.
    async def bad_market_context_provider(_intent):
        return {
            "tick_size": 0.1,
            "step_size": 0.001,
        }

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=bad_market_context_provider,
    )

    payload = make_signal_confirmed_payload(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        order_intent=OrderIntent.OPEN,
        entry_price=None,
        reservation_id="res-smart-failure",
    )
    payload["entry_price"] = None

    await emit_signal_confirmed(fake_event_bus, payload)

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.execution_failed") == 1
    assert fake_event_bus.count("execution.trade_rejected") == 1

    rejected_payload = fake_event_bus.last_payload("execution.trade_rejected")
    assert rejected_payload["reservation_id"] == "res-smart-failure"


# =============================================================================
# Kill switch
# =============================================================================


async def test_kill_switch_blocks_new_risk_increasing_entries(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, _, _, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await fake_event_bus.emit(
        "risk.kill_switch",
        {
            "reason": "daily_loss_limit",
            "cancel_open_orders": True,
        },
    )

    assert trade_executor.kill_switch_active is True
    assert fake_event_bus.count("execution.kill_switch_handled") == 1

    fake_event_bus.clear()
    fake_binance_client.calls.clear()

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-blocked-by-kill",
        ),
    )

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.trade_rejected") == 1

    rejected_payload = fake_event_bus.last_payload("execution.trade_rejected")
    assert "Kill switch" in rejected_payload["error"] or "daily_loss_limit" in rejected_payload["error"]

    assert trade_executor.stats.kill_switch_rejections >= 1


async def test_manual_resume_releases_kill_switch_and_allows_new_entry(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, _, _, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await fake_event_bus.emit("risk.kill_switch", {"reason": "test"})
    assert trade_executor.kill_switch_active is True

    await fake_event_bus.emit("risk.manual_resume", {"reason": "operator_resume"})
    assert trade_executor.kill_switch_active is False
    assert fake_event_bus.count("execution.kill_switch_released") == 1

    fake_event_bus.clear()
    fake_binance_client.calls.clear()

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-after-resume",
        ),
    )

    assert len(fake_binance_client.calls_for("create_order")) >= 1
    assert fake_event_bus.count("execution.trade_rejected") == 0
    assert fake_event_bus.count("execution.order_submitted") >= 1


async def test_kill_switch_allows_reduce_only_position_reduce_when_config_allows(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(
        smart_mode=ExecutionMode.MARKET,
        kill_switch_allows_reduce_only=True,
    )

    _, position_manager, _, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    # Open position before kill switch.
    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            final_size=0.02,
            final_notional=1_000.0,
            reservation_id="res-before-kill",
        ),
    )

    assert position_manager.has_open_position(symbol="BTCUSDT", side=PositionSide.LONG)

    await fake_event_bus.emit(
        "risk.kill_switch",
        {
            "reason": "test_reduce_during_kill",
            "cancel_open_orders": False,
        },
    )

    fake_event_bus.clear()
    fake_binance_client.calls.clear()

    await fake_event_bus.emit(
        "risk.position_reduce_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "reduce_size": 0.01,
            "reason": "risk_reduce_after_kill",
            "metadata": {},
        },
    )

    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) >= 1

    reduce_call = create_calls[0].kwargs
    assert reduce_call["symbol"] == "BTCUSDT"
    assert reduce_call["side"] == "SELL"
    assert reduce_call["order_type"] == "MARKET"
    assert reduce_call["quantity"] == pytest.approx(0.01)
    assert reduce_call["position_side"] == "LONG"
    assert reduce_call["reduce_only"] is True

    assert fake_event_bus.count("execution.trade_rejected") == 0
    assert fake_event_bus.count("execution.order_submitted") >= 1


async def test_kill_switch_cancels_open_orders_for_known_symbols(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.LIMIT)

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    # LIMIT entry stays NEW in FakeBinanceRestClient, so it remains active.
    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-open-limit",
        ),
    )

    fake_binance_client.calls.clear()

    await fake_event_bus.emit(
        "risk.kill_switch",
        {
            "reason": "cancel_open_orders",
            "cancel_open_orders": True,
        },
    )

    cancel_all_calls = fake_binance_client.calls_for("cancel_all_open_orders")
    assert len(cancel_all_calls) >= 1
    assert cancel_all_calls[0].kwargs["symbol"] == "BTCUSDT"


# =============================================================================
# Close / reduce flows
# =============================================================================


async def test_risk_position_close_requested_builds_market_reduce_only_close_order_and_cancels_sltp(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, position_manager, sltp_manager, _, _ = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            final_size=0.01,
            final_notional=500.0,
            stop_loss=49_000.0,
            take_profit=53_000.0,
            reservation_id="res-before-close",
        ),
    )

    assert position_manager.has_open_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert len(sltp_manager.list_protective_orders(symbol="BTCUSDT", include_terminal=False)) == 2

    fake_event_bus.clear()
    fake_binance_client.calls.clear()

    position = position_manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None

    await fake_event_bus.emit(
        "risk.position_close_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            "position_id": position.position_id,
            "reason": "risk_close",
            "metadata": {},
        },
    )

    create_calls = fake_binance_client.calls_for("create_order")
    cancel_calls = fake_binance_client.calls_for("cancel_order")

    assert len(cancel_calls) == 2
    assert len(create_calls) >= 1

    close_call = create_calls[0].kwargs
    assert close_call["symbol"] == "BTCUSDT"
    assert close_call["side"] == "SELL"
    assert close_call["order_type"] == "MARKET"
    assert close_call["quantity"] == pytest.approx(0.01)
    assert close_call["position_side"] == "LONG"
    assert close_call["reduce_only"] is True

    assert fake_event_bus.count("execution.sltp.cancel_completed") == 2
    assert fake_event_bus.count("execution.order_submitted") >= 1


async def test_risk_position_reduce_requested_uses_min_position_size_when_reduce_size_exceeds_position(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, position_manager, _, _, _ = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            final_size=0.02,
            final_notional=1_000.0,
            reservation_id="res-before-oversize-reduce",
        ),
    )

    assert position_manager.has_open_position(symbol="BTCUSDT", side=PositionSide.LONG)

    fake_event_bus.clear()
    fake_binance_client.calls.clear()

    await fake_event_bus.emit(
        "risk.position_reduce_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "reduce_size": 0.5,
            "reason": "oversize_reduce",
            "metadata": {},
        },
    )

    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) >= 1

    reduce_call = create_calls[0].kwargs
    assert reduce_call["side"] == "SELL"
    assert reduce_call["order_type"] == "MARKET"
    assert reduce_call["quantity"] == pytest.approx(0.02)
    assert reduce_call["reduce_only"] is True


# =============================================================================
# Reservation / permissions / expiration
# =============================================================================


async def test_expired_risk_reservation_rejects_new_entry_before_binance_call(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-expired",
            reservation_expires_at=1.0,
        ),
    )

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.trade_rejected") == 1

    payload = fake_event_bus.last_payload("execution.trade_rejected")
    assert payload["reservation_id"] == "res-expired"
    assert "expired" in payload["error"].lower()


async def test_disable_new_entries_rejects_signal_confirmed_before_binance_call(
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
    market_context_provider,
):
    from conftest import make_signal_confirmed_payload

    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)
    trade_config.allow_new_entries = False
    trade_config.validate()

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await emit_signal_confirmed(
        fake_event_bus,
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-new-disabled",
        ),
    )

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.trade_rejected") == 1

    payload = fake_event_bus.last_payload("execution.trade_rejected")
    assert "disabled" in payload["error"].lower()


# =============================================================================
# State / scheduler / snapshots
# =============================================================================


async def test_cleanup_stale_executions_marks_execution_expired(
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    market_context_provider,
):
    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.LIMIT)
    trade_config.execution_timeout_seconds = 0.001
    trade_config.validate()

    _, _, _, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    # Insert synthetic active execution to test cleanup deterministically.
    from conftest import make_signal_confirmed_payload

    intent = trade_executor._intent_from_mapping(
        make_signal_confirmed_payload(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_intent=OrderIntent.OPEN,
            reservation_id="res-stale",
        )
    )
    intent.created_at = 1.0

    trade_executor._active_executions[intent.execution_id] = intent
    trade_executor._execution_status[intent.execution_id] = ExecutionStatus.SUBMITTED

    await trade_executor.cleanup_stale_executions()

    assert fake_event_bus.count("execution.execution_expired") == 1

    snapshot = trade_executor.snapshot()
    assert snapshot["active_executions"] == 0
    assert snapshot["execution_status"][intent.execution_id] == "expired"


async def test_scheduler_job_can_run_trade_executor_cleanup(
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    market_context_provider,
):
    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    assert "execution.trade_executor.cleanup_stale_executions" in fake_scheduler.job_names()

    await fake_scheduler.run_job("execution.trade_executor.cleanup_stale_executions")

    # No active stale execution, but job must run without errors.
    assert True


async def test_trade_executor_snapshot_contains_runtime_state(
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    market_context_provider,
):
    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    _, _, _, _, trade_executor = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    snapshot = trade_executor.snapshot()

    assert snapshot["service"] == "execution.trade_executor"
    assert snapshot["running"] is True
    assert snapshot["kill_switch_active"] is False
    assert snapshot["active_executions"] == 0
    assert "stats" in snapshot
    assert snapshot["stats"]["accepted"] == 0


async def test_services_stop_cleanly_after_full_stack_start(
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    market_context_provider,
):
    (
        order_config,
        position_config,
        sltp_config,
        smart_config,
        trade_config,
    ) = make_execution_stack_configs(smart_mode=ExecutionMode.MARKET)

    (
        order_manager,
        position_manager,
        sltp_manager,
        _smart_execution,
        trade_executor,
    ) = await build_execution_stack(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
        order_manager_config=order_config,
        position_manager_config=position_config,
        sltp_manager_config=sltp_config,
        smart_execution_config=smart_config,
        trade_executor_config=trade_config,
        market_context_provider=market_context_provider,
    )

    await trade_executor.stop()
    await sltp_manager.stop()
    await position_manager.stop()
    await order_manager.stop()

    assert trade_executor.is_running is False
    assert sltp_manager.is_running is False
    assert position_manager.is_running is False
    assert order_manager.is_running is False

    assert fake_event_bus.count("execution.trade_executor.stopped") == 1
    assert fake_event_bus.count("execution.sltp_manager.stopped") == 1
    assert fake_event_bus.count("execution.position_manager.stopped") == 1
    assert fake_event_bus.count("execution.order_manager.stopped") == 1