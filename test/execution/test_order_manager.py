from __future__ import annotations

import pytest

from execution.enums import OrderSide, OrderStatus, OrderType, TimeInForce, TriggerType, WorkingType
from execution.exceptions import OrderCancelError, OrderSubmitError
from execution.models import OrderRequest
from execution.order_manager import OrderManager

from risk.enums import PositionSide


pytestmark = pytest.mark.asyncio


# =============================================================================
# Helpers
# =============================================================================


async def start_order_manager(
    *,
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
) -> OrderManager:
    manager = OrderManager(
        order_manager_config,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )
    await manager.start()
    return manager


def make_market_open_long_request(
    *,
    quantity: float = 0.01,
    symbol: str = "btcusdt",
    execution_id: str = "exec-open-long",
    signal_id: str = "sig-open-long",
    reservation_id: str = "res-open-long",
) -> OrderRequest:
    return OrderRequest(
        execution_id=execution_id,
        exchange="binance",
        market_type="usdm_futures",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=quantity,
        position_side=PositionSide.LONG,
        reduce_only=False,
        close_position=False,
        signal_id=signal_id,
        strategy_name="test_strategy",
        reservation_id=reservation_id,
        metadata={
            "test_case": "market_open_long",
            "final_leverage": 5.0,
            "final_margin": 100.0,
            "final_notional": 500.0,
            "final_risk_amount": 50.0,
        },
    )


def make_limit_reduce_long_request(
    *,
    quantity: float = 0.01,
    price: float = 50_000.0,
    symbol: str = "BTCUSDT",
    execution_id: str = "exec-reduce-long",
    signal_id: str = "sig-reduce-long",
    reservation_id: str = "res-reduce-long",
) -> OrderRequest:
    return OrderRequest(
        execution_id=execution_id,
        exchange="binance",
        market_type="usdm_futures",
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        reduce_only=True,
        close_position=False,
        signal_id=signal_id,
        strategy_name="test_strategy",
        reservation_id=reservation_id,
        trigger_type=TriggerType.RISK_REDUCE,
        metadata={
            "test_case": "limit_reduce_long",
        },
    )


# =============================================================================
# Lifecycle / registration
# =============================================================================


async def test_order_manager_start_registers_eventbus_subscriptions_and_scheduler_jobs(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    assert manager.is_running is True

    subscription_topics = {subscription.topic for subscription in fake_event_bus.subscriptions}
    assert "execution.order_submit_requested" in subscription_topics
    assert "execution.order_cancel_requested" in subscription_topics
    assert "execution.order_replace_requested" in subscription_topics
    assert "exchange.order.submitted" in subscription_topics
    assert "exchange.order.cancelled" in subscription_topics
    assert "exchange.open_orders.snapshot" in subscription_topics
    assert "risk.kill_switch" in subscription_topics

    assert "execution.order_manager.reconcile_orders" in fake_scheduler.job_names()
    assert "execution.order_manager.sync_open_orders" in fake_scheduler.job_names()

    assert fake_event_bus.count("execution.order_manager.started") == 1


async def test_order_manager_stop_unregisters_subscriptions(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    assert fake_event_bus.subscriptions

    await manager.stop()

    assert manager.is_running is False
    assert fake_event_bus.subscriptions == []
    assert fake_event_bus.count("execution.order_manager.stopped") == 1


# =============================================================================
# Submit order behavior
# =============================================================================


async def test_submit_market_open_long_maps_to_binance_params_and_emits_filled_lifecycle(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = make_market_open_long_request(symbol="btcusdt")

    result = await manager.submit_order(request)

    assert result.status is OrderStatus.FILLED
    assert result.symbol == "BTCUSDT"
    assert result.side is OrderSide.BUY
    assert result.order_type is OrderType.MARKET
    assert result.executed_quantity == pytest.approx(0.01)
    assert result.avg_price == pytest.approx(50_000.0)

    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) == 1

    params = create_calls[0].kwargs
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "BUY"
    assert params["order_type"] == "MARKET"
    assert params["quantity"] == pytest.approx(0.01)
    assert params["position_side"] == "LONG"
    assert params["reduce_only"] is None
    assert params["close_position"] is None
    assert params["new_client_order_id"]
    assert params["new_order_resp_type"] == "RESULT"

    state = manager.get_order_state(
        order_id=result.order_id,
        client_order_id=result.client_order_id,
    )
    assert state is not None
    assert state.status is OrderStatus.FILLED
    assert state.execution_id == "exec-open-long"
    assert state.signal_id == "sig-open-long"
    assert state.reservation_id == "res-open-long"

    assert fake_event_bus.count("execution.order_submit_started") == 1
    assert fake_event_bus.count("execution.order_submitted") == 1
    assert fake_event_bus.count("execution.order_acknowledged") == 1
    assert fake_event_bus.count("execution.order_filled") == 1

    filled_payload = fake_event_bus.last_payload("execution.order_filled")
    assert filled_payload["symbol"] == "BTCUSDT"
    assert filled_payload["execution_id"] == "exec-open-long"
    assert filled_payload["signal_id"] == "sig-open-long"
    assert filled_payload["reservation_id"] == "res-open-long"
    assert filled_payload["status"] == "FILLED"
    assert filled_payload["executed_quantity"] == pytest.approx(0.01)

    assert manager.stats.submitted == 1
    assert manager.stats.filled == 1
    assert manager.stats.active_orders == 0


async def test_submit_limit_reduce_only_order_preserves_reduce_only_and_time_in_force(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=2001,
        client_order_id="reduce-client-1",
        side="SELL",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_100.0,
        avg_price=0.0,
        position_side="LONG",
        reduce_only=True,
        close_position=False,
    )

    request = make_limit_reduce_long_request(price=50_100.0)
    request.client_order_id = "reduce-client-1"

    result = await manager.submit_order(request)

    assert result.status is OrderStatus.NEW
    assert result.reduce_only is True
    assert result.close_position is False

    create_calls = fake_binance_client.calls_for("create_order")
    assert len(create_calls) == 1

    params = create_calls[0].kwargs
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "SELL"
    assert params["order_type"] == "LIMIT"
    assert params["quantity"] == pytest.approx(0.01)
    assert params["price"] == pytest.approx(50_100.0)
    assert params["position_side"] == "LONG"
    assert params["time_in_force"] == "GTC"
    assert params["reduce_only"] is True
    assert params["new_client_order_id"] == "reduce-client-1"

    assert fake_event_bus.count("execution.order_submitted") == 1
    assert fake_event_bus.count("execution.order_filled") == 0
    assert fake_event_bus.count("execution.order_failed") == 0

    state = manager.get_order_state(client_order_id="reduce-client-1")
    assert state is not None
    assert state.is_open is True
    assert state.reduce_only is True


async def test_submit_protective_stop_market_close_position_omits_quantity_and_emits_submitted(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = OrderRequest(
        execution_id="exec-sltp-stop",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=None,
        position_side=PositionSide.LONG,
        reduce_only=False,
        close_position=True,
        client_order_id="sl-close-position-1",
        stop_price=49_000.0,
        working_type=WorkingType.MARK_PRICE,
        price_protect=True,
        trigger_type=TriggerType.STOP_LOSS,
        signal_id="sig-1",
        strategy_name="test_strategy",
        reservation_id="res-1",
    )

    result = await manager.submit_order(request)

    assert result.status is OrderStatus.NEW

    params = fake_binance_client.calls_for("create_order")[0].kwargs
    assert params["order_type"] == "STOP_MARKET"
    assert params["quantity"] is None
    assert params["position_side"] == "LONG"
    assert params["reduce_only"] is None
    assert params["close_position"] is True
    assert params["stop_price"] == pytest.approx(49_000.0)
    assert params["working_type"] == "MARK_PRICE"
    assert params["price_protect"] is True

    submitted_payload = fake_event_bus.last_payload("execution.order_submitted")
    assert submitted_payload["client_order_id"] == "sl-close-position-1"
    assert submitted_payload["close_position"] is True
    assert submitted_payload["stop_price"] == pytest.approx(49_000.0)


async def test_submit_close_position_with_quantity_fails_before_binance_call(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = OrderRequest(
        execution_id="exec-invalid-close-position",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        position_side=PositionSide.LONG,
        close_position=True,
        stop_price=49_000.0,
        trigger_type=TriggerType.STOP_LOSS,
    )

    with pytest.raises(Exception):
        await manager.submit_order(request)

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.order_failed") == 0


async def test_submit_stop_market_without_stop_price_fails_before_binance_call(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = OrderRequest(
        execution_id="exec-invalid-stop",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        position_side=PositionSide.LONG,
        reduce_only=True,
        stop_price=None,
        trigger_type=TriggerType.STOP_LOSS,
    )

    with pytest.raises(Exception):
        await manager.submit_order(request)

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.order_failed") == 0


async def test_submit_protective_reduce_order_without_reduce_only_or_close_position_is_rejected(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = OrderRequest(
        execution_id="exec-invalid-protective",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_MARKET,
        quantity=0.01,
        position_side=PositionSide.LONG,
        reduce_only=False,
        close_position=False,
        stop_price=49_000.0,
        trigger_type=TriggerType.STOP_LOSS,
    )

    with pytest.raises(Exception):
        await manager.submit_order(request)

    assert fake_binance_client.calls_for("create_order") == []
    assert fake_event_bus.count("execution.order_failed") == 0


async def test_submit_binance_exception_emits_order_failed_and_does_not_store_state(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.create_order_exception = RuntimeError("binance unavailable")

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = make_market_open_long_request()

    with pytest.raises(OrderSubmitError):
        await manager.submit_order(request)

    assert len(fake_binance_client.calls_for("create_order")) == 1
    assert fake_event_bus.count("execution.order_failed") == 1

    failed_payload = fake_event_bus.last_payload("execution.order_failed")
    assert failed_payload["symbol"] == "BTCUSDT"
    assert failed_payload["execution_id"] == "exec-open-long"
    assert failed_payload["signal_id"] == "sig-open-long"
    assert failed_payload["reservation_id"] == "res-open-long"
    assert failed_payload["failure_stage"] == "submit"
    assert "binance unavailable" in failed_payload["error"]

    assert manager.list_orders(include_terminal=True) == []
    assert manager.stats.failed == 1


async def test_submit_rejected_exchange_response_emits_order_rejected(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=3001,
        client_order_id="rejected-client-1",
        side="BUY",
        order_type="MARKET",
        status="REJECTED",
        orig_qty=0.01,
        executed_qty=0.0,
        price=0.0,
        avg_price=0.0,
        position_side="LONG",
    )

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = make_market_open_long_request()
    request.client_order_id = "rejected-client-1"

    result = await manager.submit_order(request)

    assert result.status is OrderStatus.REJECTED
    assert fake_event_bus.count("execution.order_submitted") == 1
    assert fake_event_bus.count("execution.order_rejected") == 1
    assert fake_event_bus.count("execution.order_filled") == 0

    rejected_payload = fake_event_bus.last_payload("execution.order_rejected")
    assert rejected_payload["client_order_id"] == "rejected-client-1"
    assert rejected_payload["reservation_id"] == "res-open-long"

    state = manager.get_order_state(client_order_id="rejected-client-1")
    assert state is not None
    assert state.is_terminal is True


async def test_submit_partially_filled_exchange_response_emits_partially_filled(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=3002,
        client_order_id="partial-client-1",
        side="BUY",
        order_type="LIMIT",
        status="PARTIALLY_FILLED",
        orig_qty=0.02,
        executed_qty=0.01,
        price=50_000.0,
        avg_price=50_000.0,
        position_side="LONG",
    )

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = OrderRequest(
        execution_id="exec-partial",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.02,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="partial-client-1",
        signal_id="sig-partial",
        strategy_name="test_strategy",
        reservation_id="res-partial",
    )

    result = await manager.submit_order(request)

    assert result.status is OrderStatus.PARTIALLY_FILLED
    assert result.fill_ratio == pytest.approx(0.5)

    assert fake_event_bus.count("execution.order_partially_filled") == 1
    assert fake_event_bus.count("execution.order_filled") == 0

    payload = fake_event_bus.last_payload("execution.order_partially_filled")
    assert payload["executed_quantity"] == pytest.approx(0.01)
    assert payload["original_quantity"] == pytest.approx(0.02)
    assert payload["fill_ratio"] == pytest.approx(0.5)

    state = manager.get_order_state(client_order_id="partial-client-1")
    assert state is not None
    assert state.is_open is True
    assert state.fill_ratio == pytest.approx(0.5)


# =============================================================================
# Cancel / replace behavior
# =============================================================================


async def test_cancel_order_calls_binance_and_emits_cancelled(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=4001,
        client_order_id="cancel-client-1",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_000.0,
        avg_price=0.0,
        position_side="LONG",
    )

    request = OrderRequest(
        execution_id="exec-cancel",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="cancel-client-1",
        signal_id="sig-cancel",
        reservation_id="res-cancel",
    )

    submitted = await manager.submit_order(request)
    assert submitted.status is OrderStatus.NEW

    result = await manager.cancel_order(
        symbol="BTCUSDT",
        order_id=submitted.order_id,
        client_order_id=submitted.client_order_id,
        reason="test_cancel",
    )

    assert result.status is OrderStatus.CANCELED

    cancel_calls = fake_binance_client.calls_for("cancel_order")
    assert len(cancel_calls) == 1
    assert cancel_calls[0].kwargs["symbol"] == "BTCUSDT"
    assert cancel_calls[0].kwargs["order_id"] == int(submitted.order_id)
    assert cancel_calls[0].kwargs["orig_client_order_id"] == submitted.client_order_id

    assert fake_event_bus.count("execution.order_cancelled") == 1

    cancelled_payload = fake_event_bus.last_payload("execution.order_cancelled")
    assert cancelled_payload["status"] == "CANCELED"
    assert cancelled_payload["client_order_id"] == "cancel-client-1"

    state = manager.get_order_state(client_order_id="cancel-client-1")
    assert state is not None
    assert state.status is OrderStatus.CANCELED
    assert state.is_terminal is True


async def test_cancel_terminal_order_raises_without_second_binance_cancel(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    request = make_market_open_long_request()
    result = await manager.submit_order(request)
    assert result.status is OrderStatus.FILLED

    with pytest.raises(OrderCancelError):
        await manager.cancel_order(
            symbol="BTCUSDT",
            order_id=result.order_id,
            client_order_id=result.client_order_id,
        )

    assert fake_binance_client.calls_for("cancel_order") == []


async def test_cancel_order_binance_failure_emits_order_failed(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.cancel_order_exception = RuntimeError("cancel failed")

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    with pytest.raises(OrderCancelError):
        await manager.cancel_order(
            symbol="BTCUSDT",
            order_id=123,
            client_order_id="client-123",
            reason="test_failure",
        )

    assert fake_event_bus.count("execution.order_failed") == 1

    payload = fake_event_bus.last_payload("execution.order_failed")
    assert payload["symbol"] == "BTCUSDT"
    assert payload["order_id"] == "123"
    assert payload["client_order_id"] == "client-123"
    assert payload["failure_stage"] == "cancel"
    assert "cancel failed" in payload["error"]


async def test_replace_order_is_cancel_then_submit(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    new_request = make_limit_reduce_long_request(
        price=50_250.0,
        execution_id="exec-replace-new",
    )
    new_request.client_order_id = "replace-new-client"

    result = await manager.replace_order(
        existing_symbol="BTCUSDT",
        existing_order_id=555,
        existing_client_order_id="replace-old-client",
        new_request=new_request,
        reason="repricing",
    )

    assert result.client_order_id == "replace-new-client"

    assert len(fake_binance_client.calls_for("cancel_order")) == 1
    assert len(fake_binance_client.calls_for("create_order")) == 1

    create_params = fake_binance_client.calls_for("create_order")[0].kwargs
    assert create_params["price"] == pytest.approx(50_250.0)
    assert create_params["reduce_only"] is True


async def test_cancel_all_orders_marks_local_open_orders_cancelled_and_emits_orders_cancelled(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=6001,
        client_order_id="open-client-1",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_000.0,
        avg_price=0.0,
        position_side="LONG",
    )

    request = OrderRequest(
        execution_id="exec-open-order",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="open-client-1",
    )
    await manager.submit_order(request)

    await manager.cancel_all_orders(symbol="BTCUSDT", reason="kill_switch")

    assert len(fake_binance_client.calls_for("cancel_all_open_orders")) == 1
    assert fake_binance_client.calls_for("cancel_all_open_orders")[0].kwargs["symbol"] == "BTCUSDT"

    assert fake_event_bus.count("execution.orders_cancelled") == 1

    state = manager.get_order_state(client_order_id="open-client-1")
    assert state is not None
    assert state.status is OrderStatus.CANCELED


# =============================================================================
# Fetch / sync / reconciliation
# =============================================================================


async def test_fetch_order_updates_local_state(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    result = await manager.fetch_order(symbol="BTCUSDT", order_id=777)

    assert result.status is OrderStatus.FILLED
    assert result.order_id == "777"

    state = manager.get_order_state(order_id="777")
    assert state is not None
    assert state.status is OrderStatus.FILLED


async def test_sync_open_orders_adds_exchange_open_orders_and_emits_sync_completed(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.open_orders = [
        fake_binance_client._normalized_order(
            symbol="BTCUSDT",
            order_id=8001,
            client_order_id="sync-client-1",
            side="BUY",
            order_type="LIMIT",
            status="NEW",
            orig_qty=0.01,
            executed_qty=0.0,
            price=50_000.0,
            avg_price=0.0,
            position_side="LONG",
        ),
        fake_binance_client._normalized_order(
            symbol="ETHUSDT",
            order_id=8002,
            client_order_id="sync-client-2",
            side="SELL",
            order_type="LIMIT",
            status="NEW",
            orig_qty=0.2,
            executed_qty=0.0,
            price=3_000.0,
            avg_price=0.0,
            position_side="SHORT",
        ),
    ]

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.sync_open_orders()

    assert fake_event_bus.count("execution.order_sync_completed") == 1
    assert manager.get_order_state(client_order_id="sync-client-1") is not None
    assert manager.get_order_state(client_order_id="sync-client-2") is not None
    assert len(manager.active_orders) == 2


async def test_sync_open_orders_marks_missing_local_open_order_as_sync_required(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=8101,
        client_order_id="missing-local-client",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_000.0,
        avg_price=0.0,
        position_side="LONG",
    )

    request = OrderRequest(
        execution_id="exec-missing-local",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="missing-local-client",
    )

    await manager.submit_order(request)
    fake_binance_client.open_orders = []

    await manager.sync_open_orders(symbol="BTCUSDT")

    state = manager.get_order_state(client_order_id="missing-local-client")
    assert state is not None
    assert state.is_open is True
    assert state.metadata["sync_required"] is True
    assert "sync_required_at" in state.metadata


async def test_reconcile_orders_runs_sync_and_emits_reconciled(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.open_orders = [
        fake_binance_client._normalized_order(
            symbol="BTCUSDT",
            order_id=8201,
            client_order_id="reconcile-client",
            side="BUY",
            order_type="LIMIT",
            status="NEW",
            orig_qty=0.01,
            executed_qty=0.0,
            price=50_000.0,
            avg_price=0.0,
            position_side="LONG",
        )
    ]

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.reconcile_orders()

    assert manager.stats.reconciliation_runs == 1
    assert fake_event_bus.count("execution.order_sync_completed") == 1
    assert fake_event_bus.count("execution.order_reconciled") == 1


async def test_scheduler_job_can_run_reconciliation(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.open_orders = [
        fake_binance_client._normalized_order(
            symbol="BTCUSDT",
            order_id=8301,
            client_order_id="scheduler-client",
            side="BUY",
            order_type="LIMIT",
            status="NEW",
            orig_qty=0.01,
            executed_qty=0.0,
            price=50_000.0,
            avg_price=0.0,
            position_side="LONG",
        )
    ]

    await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_scheduler.run_job("execution.order_manager.reconcile_orders")

    assert fake_event_bus.count("execution.order_reconciled") == 1


async def test_sync_open_orders_failure_emits_order_sync_failed(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.get_open_orders_exception = RuntimeError("open order sync failed")

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.sync_open_orders()

    assert fake_event_bus.count("execution.order_sync_failed") == 1
    assert manager.stats.reconciliation_failures == 1
    assert manager.stats.failed == 1

    payload = fake_event_bus.last_payload("execution.order_sync_failed")
    assert "open order sync failed" in payload["error"]


# =============================================================================
# Event handlers
# =============================================================================


async def test_event_order_submit_requested_goes_through_submit_flow(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "execution.order_submit_requested",
        {
            "execution_id": "exec-event-submit",
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": 0.01,
            "position_side": "LONG",
            "signal_id": "sig-event-submit",
            "reservation_id": "res-event-submit",
            "metadata": {"from_event": True},
        },
    )

    assert len(fake_binance_client.calls_for("create_order")) == 1
    assert fake_event_bus.count("execution.order_submitted") == 1
    assert fake_event_bus.count("execution.order_filled") == 1

    payload = fake_event_bus.last_payload("execution.order_filled")
    assert payload["execution_id"] == "exec-event-submit"
    assert payload["signal_id"] == "sig-event-submit"
    assert payload["reservation_id"] == "res-event-submit"


async def test_event_order_cancel_requested_goes_through_cancel_flow(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "execution.order_cancel_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "order_id": 1234,
            "client_order_id": "event-cancel-client",
            "reason": "test_event_cancel",
        },
    )

    assert len(fake_binance_client.calls_for("cancel_order")) == 1
    assert fake_event_bus.count("execution.order_cancelled") == 1


async def test_event_exchange_order_update_updates_state_and_emits_lifecycle(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "exchange.order.submitted",
        fake_binance_client._normalized_order(
            symbol="BTCUSDT",
            order_id=9001,
            client_order_id="exchange-update-client",
            side="BUY",
            order_type="MARKET",
            status="FILLED",
            orig_qty=0.01,
            executed_qty=0.01,
            price=0.0,
            avg_price=50_000.0,
            position_side="LONG",
        ),
    )

    assert fake_event_bus.count("execution.order_status_updated") == 1
    assert fake_event_bus.count("execution.order_filled") == 1

    state = manager.get_order_state(client_order_id="exchange-update-client")
    assert state is not None
    assert state.status is OrderStatus.FILLED


async def test_event_open_orders_snapshot_applies_snapshot_items(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "exchange.open_orders.snapshot",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": None,
            "orders": [
                fake_binance_client._normalized_order(
                    symbol="BTCUSDT",
                    order_id=9101,
                    client_order_id="snapshot-client-1",
                    side="BUY",
                    order_type="LIMIT",
                    status="NEW",
                    orig_qty=0.01,
                    executed_qty=0.0,
                    price=50_000.0,
                    avg_price=0.0,
                    position_side="LONG",
                )
            ],
        },
    )

    state = manager.get_order_state(client_order_id="snapshot-client-1")
    assert state is not None
    assert state.status is OrderStatus.NEW


async def test_event_risk_kill_switch_cancels_open_orders_for_active_symbols(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=9201,
        client_order_id="kill-client-1",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_000.0,
        avg_price=0.0,
        position_side="LONG",
    )

    request = OrderRequest(
        execution_id="exec-kill",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="kill-client-1",
    )

    await manager.submit_order(request)

    await fake_event_bus.emit(
        "risk.kill_switch",
        {
            "reason": "test_kill_switch",
            "cancel_open_orders": True,
        },
    )

    assert len(fake_binance_client.calls_for("cancel_all_open_orders")) == 1
    assert fake_binance_client.calls_for("cancel_all_open_orders")[0].kwargs["symbol"] == "BTCUSDT"
    assert fake_event_bus.count("execution.orders_cancelled") == 1


# =============================================================================
# User trades / fills
# =============================================================================


async def test_load_user_trades_as_fills_normalizes_binance_user_trades(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    fake_binance_client.user_trades = [
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "id": 1,
            "order_id": 123,
            "side": "BUY",
            "position_side": "LONG",
            "price": 50_000.0,
            "qty": 0.01,
            "quote_qty": 500.0,
            "realized_pnl": 0.0,
            "margin_asset": "USDT",
            "commission": 0.2,
            "commission_asset": "USDT",
            "time": 1_700_000_000_000,
            "buyer": True,
            "maker": False,
        },
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "ETHUSDT",
            "id": 2,
            "order_id": 124,
            "side": "BUY",
            "position_side": "LONG",
            "price": 3_000.0,
            "qty": 0.1,
            "quote_qty": 300.0,
            "realized_pnl": 0.0,
            "commission": 0.1,
            "commission_asset": "USDT",
            "time": 1_700_000_000_000,
            "buyer": True,
            "maker": False,
        },
    ]

    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fills = await manager.load_user_trades_as_fills(
        symbol="BTCUSDT",
        order_id=123,
        execution_id="exec-user-trades",
        signal_id="sig-user-trades",
        strategy_name="test_strategy",
        reservation_id="res-user-trades",
    )

    assert len(fills) == 1
    fill = fills[0]

    assert fill.symbol == "BTCUSDT"
    assert fill.side is OrderSide.BUY
    assert fill.position_side is PositionSide.LONG
    assert fill.quantity == pytest.approx(0.01)
    assert fill.price == pytest.approx(50_000.0)
    assert fill.notional == pytest.approx(500.0)
    assert fill.commission == pytest.approx(0.2)
    assert fill.execution_id == "exec-user-trades"
    assert fill.signal_id == "sig-user-trades"
    assert fill.reservation_id == "res-user-trades"


# =============================================================================
# Snapshot / diagnostics
# =============================================================================


async def test_snapshot_contains_stats_and_active_orders(
    order_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_order_manager(
        order_manager_config=order_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fake_binance_client.next_create_order_response = fake_binance_client._normalized_order(
        symbol="BTCUSDT",
        order_id=9301,
        client_order_id="snapshot-active-client",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        orig_qty=0.01,
        executed_qty=0.0,
        price=50_000.0,
        avg_price=0.0,
        position_side="LONG",
    )

    request = OrderRequest(
        execution_id="exec-snapshot",
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=50_000.0,
        position_side=PositionSide.LONG,
        time_in_force=TimeInForce.GTC,
        client_order_id="snapshot-active-client",
    )

    await manager.submit_order(request)

    snapshot = manager.snapshot()

    assert snapshot["service"] == "execution.order_manager"
    assert snapshot["running"] is True
    assert snapshot["orders_count"] == 1
    assert snapshot["active_orders_count"] == 1
    assert snapshot["stats"]["submitted"] == 1
    assert snapshot["stats"]["active_orders"] == 1