from __future__ import annotations

import pytest

from execution.enums import OrderSide
from execution.models import OrderFill, PositionSnapshot
from execution.position_manager import PositionManager

from risk.enums import PositionSide


pytestmark = pytest.mark.asyncio


# =============================================================================
# Helpers
# =============================================================================


async def start_position_manager(
    *,
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
) -> PositionManager:
    manager = PositionManager(
        position_manager_config,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )
    await manager.start()
    return manager


def make_fill(
    *,
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    position_side: PositionSide | None = PositionSide.LONG,
    quantity: float = 0.01,
    price: float = 50_000.0,
    order_id: str = "order-1",
    client_order_id: str = "client-1",
    execution_id: str = "exec-1",
    signal_id: str = "sig-1",
    strategy_name: str = "test_strategy",
    reservation_id: str = "res-1",
    realized_pnl: float | None = None,
    metadata: dict | None = None,
) -> OrderFill:
    return OrderFill(
        exchange="binance",
        market_type="usdm_futures",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_id=order_id,
        client_order_id=client_order_id,
        position_side=position_side,
        quote_quantity=quantity * price,
        commission=0.1,
        commission_asset="USDT",
        realized_pnl=realized_pnl,
        maker=False,
        execution_id=execution_id,
        signal_id=signal_id,
        strategy_name=strategy_name,
        reservation_id=reservation_id,
        fill_time=1_700_000_000_000,
        metadata={
            "final_leverage": 5.0,
            "final_margin": 100.0,
            "final_notional": quantity * price,
            "final_risk_amount": 50.0,
            "stop_loss": 49_000.0,
            "take_profit": 53_000.0,
            "tier": "t2",
            **(metadata or {}),
        },
    )


# =============================================================================
# Lifecycle / registration
# =============================================================================


async def test_position_manager_start_registers_subscriptions_and_scheduler_jobs(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    assert manager.is_running is True

    subscription_topics = {subscription.topic for subscription in fake_event_bus.subscriptions}
    assert "execution.order_filled" in subscription_topics
    assert "execution.order_partially_filled" in subscription_topics
    assert "exchange.positions.snapshot" in subscription_topics
    assert "risk.kill_switch" in subscription_topics

    assert "execution.position_manager.reconcile_positions" in fake_scheduler.job_names()
    assert "execution.position_manager.sync_positions" in fake_scheduler.job_names()

    assert fake_event_bus.count("execution.position_manager.started") == 1


async def test_position_manager_stop_unregisters_subscriptions(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    assert fake_event_bus.subscriptions

    await manager.stop()

    assert manager.is_running is False
    assert fake_event_bus.subscriptions == []
    assert fake_event_bus.count("execution.position_manager.stopped") == 1


# =============================================================================
# Direct fill application
# =============================================================================


async def test_apply_buy_fill_opens_long_position_and_emits_position_opened(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fill = make_fill(
        side=OrderSide.BUY,
        position_side=PositionSide.LONG,
        quantity=0.01,
        price=50_000.0,
    )

    update = await manager.apply_fill(fill)

    assert update is not None
    assert update.update_type == "opened"
    assert update.symbol == "BTCUSDT"
    assert update.side is PositionSide.LONG
    assert update.size == pytest.approx(0.01)
    assert update.entry_price == pytest.approx(50_000.0)
    assert update.notional_value == pytest.approx(500.0)
    assert update.leverage == pytest.approx(5.0)
    assert update.margin_used == pytest.approx(100.0)
    assert update.risk_amount == pytest.approx(50.0)
    assert update.stop_loss == pytest.approx(49_000.0)
    assert update.take_profit == pytest.approx(53_000.0)
    assert update.signal_id == "sig-1"
    assert update.strategy_name == "test_strategy"
    assert update.reservation_id == "res-1"

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None
    assert position.is_open is True
    assert position.side is PositionSide.LONG
    assert position.size == pytest.approx(0.01)
    assert position.entry_price == pytest.approx(50_000.0)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 0
    assert fake_event_bus.count("position.closed") == 0

    payload = fake_event_bus.last_payload("position.opened")
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "long"
    assert payload["size"] == pytest.approx(0.01)
    assert payload["entry_price"] == pytest.approx(50_000.0)
    assert payload["notional_value"] == pytest.approx(500.0)
    assert payload["leverage"] == pytest.approx(5.0)
    assert payload["margin_used"] == pytest.approx(100.0)
    assert payload["risk_amount"] == pytest.approx(50.0)
    assert payload["stop_loss"] == pytest.approx(49_000.0)
    assert payload["take_profit"] == pytest.approx(53_000.0)
    assert payload["strategy_name"] == "test_strategy"
    assert payload["signal_id"] == "sig-1"
    assert payload["reservation_id"] == "res-1"
    assert payload["position_id"]

    assert manager.stats.opened == 1
    assert manager.stats.open_positions == 1


async def test_second_buy_fill_increases_long_and_recalculates_average_entry(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="order-1",
            client_order_id="client-1",
            quantity=0.01,
            price=50_000.0,
        )
    )
    await manager.apply_fill(
        make_fill(
            order_id="order-2",
            client_order_id="client-2",
            quantity=0.01,
            price=51_000.0,
            metadata={
                "final_notional": 510.0,
            },
        )
    )

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)

    assert position is not None
    assert position.size == pytest.approx(0.02)
    assert position.entry_price == pytest.approx(50_500.0)
    assert position.mark_price == pytest.approx(51_000.0)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 1
    assert fake_event_bus.count("position.closed") == 0

    payload = fake_event_bus.last_payload("position.updated")
    assert payload["update_type"] == "updated"
    assert payload["previous_size"] == pytest.approx(0.01)
    assert payload["size"] == pytest.approx(0.02)

    assert manager.stats.opened == 1
    assert manager.stats.updated == 1


async def test_sell_fill_reduces_existing_long_position(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="open-order",
            client_order_id="open-client",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            quantity=0.02,
            price=50_000.0,
        )
    )

    update = await manager.apply_fill(
        make_fill(
            order_id="reduce-order",
            client_order_id="reduce-client",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            quantity=0.01,
            price=51_000.0,
            realized_pnl=10.0,
            metadata={
                "final_notional": 510.0,
            },
        )
    )

    assert update is not None
    assert update.update_type == "reduced"
    assert update.previous_size == pytest.approx(0.02)
    assert update.size == pytest.approx(0.01)
    assert update.realized_pnl == pytest.approx(10.0)

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)

    assert position is not None
    assert position.is_open is True
    assert position.size == pytest.approx(0.01)
    assert position.side is PositionSide.LONG
    assert position.realized_pnl == pytest.approx(10.0)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 1
    assert fake_event_bus.count("position.closed") == 0

    payload = fake_event_bus.last_payload("position.updated")
    assert payload["update_type"] == "reduced"
    assert payload["realized_pnl"] == pytest.approx(10.0)

    assert manager.stats.reduced == 1


async def test_sell_fill_closes_existing_long_position_and_emits_position_closed(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="open-order",
            client_order_id="open-client",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            quantity=0.01,
            price=50_000.0,
        )
    )

    update = await manager.apply_fill(
        make_fill(
            order_id="close-order",
            client_order_id="close-client",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            quantity=0.01,
            price=51_000.0,
            realized_pnl=10.0,
        )
    )

    assert update is not None
    assert update.update_type == "closed"
    assert update.previous_size == pytest.approx(0.01)
    assert update.size == pytest.approx(0.0)
    assert update.side is None

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None
    assert position.is_flat is True
    assert position.size == pytest.approx(0.0)
    assert position.side is None
    assert position.closed_at is not None

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.closed") == 1

    payload = fake_event_bus.last_payload("position.closed")
    assert payload["update_type"] == "closed"
    assert payload["side"] is None
    assert payload["previous_side"] == "long"
    assert payload["size"] == pytest.approx(0.0)
    assert payload["previous_size"] == pytest.approx(0.01)
    assert payload["reservation_id"] == "res-1"

    assert manager.stats.closed == 1
    assert manager.stats.open_positions == 0


async def test_fill_larger_than_existing_position_reverses_position(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="open-long",
            client_order_id="open-long-client",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            quantity=0.01,
            price=50_000.0,
        )
    )

    update = await manager.apply_fill(
        make_fill(
            order_id="reverse-to-short",
            client_order_id="reverse-to-short-client",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            quantity=0.03,
            price=49_000.0,
        )
    )

    assert update is not None
    assert update.update_type == "reversed"
    assert update.previous_size == pytest.approx(0.01)
    assert update.size == pytest.approx(0.02)
    assert update.side is PositionSide.SHORT

    short_position = manager.get_position(symbol="BTCUSDT", side=PositionSide.SHORT)
    assert short_position is not None
    assert short_position.side is PositionSide.SHORT
    assert short_position.size == pytest.approx(0.02)
    assert short_position.entry_price == pytest.approx(49_000.0)

    assert fake_event_bus.count("position.updated") == 1
    assert manager.stats.reversed == 1


async def test_short_position_opens_with_sell_fill_and_closes_with_buy_fill(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    open_update = await manager.apply_fill(
        make_fill(
            order_id="short-open",
            client_order_id="short-open-client",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            quantity=0.02,
            price=50_000.0,
        )
    )

    assert open_update is not None
    assert open_update.update_type == "opened"
    assert open_update.side is PositionSide.SHORT

    close_update = await manager.apply_fill(
        make_fill(
            order_id="short-close",
            client_order_id="short-close-client",
            side=OrderSide.BUY,
            position_side=PositionSide.LONG,
            quantity=0.02,
            price=49_000.0,
            realized_pnl=20.0,
        )
    )

    assert close_update is not None
    assert close_update.update_type == "closed"
    assert close_update.previous_side is PositionSide.SHORT
    assert close_update.side is None
    assert close_update.realized_pnl == pytest.approx(20.0)

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.SHORT)
    assert position is not None
    assert position.is_flat is True

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.closed") == 1


# =============================================================================
# Cumulative fill dedup / delta logic
# =============================================================================


async def test_cumulative_partial_then_filled_applies_only_delta_quantity(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    partial_fill = make_fill(
        order_id="cumulative-order",
        client_order_id="cumulative-client",
        quantity=0.01,
        price=50_000.0,
    )

    full_fill = make_fill(
        order_id="cumulative-order",
        client_order_id="cumulative-client",
        quantity=0.02,
        price=50_000.0,
    )

    update_1 = await manager.apply_fill(partial_fill)
    update_2 = await manager.apply_fill(full_fill)

    assert update_1 is not None
    assert update_2 is not None

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)

    assert position is not None
    assert position.size == pytest.approx(0.02)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 1

    payload = fake_event_bus.last_payload("position.updated")
    assert payload["previous_size"] == pytest.approx(0.01)
    assert payload["size"] == pytest.approx(0.02)


async def test_duplicate_same_cumulative_fill_is_ignored(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    fill = make_fill(
        order_id="dup-order",
        client_order_id="dup-client",
        quantity=0.01,
        price=50_000.0,
    )

    update_1 = await manager.apply_fill(fill)
    update_2 = await manager.apply_fill(fill)

    assert update_1 is not None
    assert update_2 is None

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)

    assert position is not None
    assert position.size == pytest.approx(0.01)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 0


async def test_smaller_old_cumulative_fill_after_larger_fill_is_ignored(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    larger_fill = make_fill(
        order_id="out-of-order",
        client_order_id="out-of-order-client",
        quantity=0.02,
        price=50_000.0,
    )
    old_smaller_fill = make_fill(
        order_id="out-of-order",
        client_order_id="out-of-order-client",
        quantity=0.01,
        price=50_000.0,
    )

    update_1 = await manager.apply_fill(larger_fill)
    update_2 = await manager.apply_fill(old_smaller_fill)

    assert update_1 is not None
    assert update_2 is None

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None
    assert position.size == pytest.approx(0.02)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 0


# =============================================================================
# Event handlers
# =============================================================================


async def test_event_execution_order_filled_opens_position_from_payload(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
    make_order_filled_payload=None,
):
    from conftest import make_order_filled_payload as build_payload

    await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "execution.order_filled",
        build_payload(
            symbol="BTCUSDT",
            side="BUY",
            position_side="LONG",
            executed_qty=0.01,
            avg_price=50_000.0,
            order_id="event-filled-order",
            client_order_id="event-filled-client",
            execution_id="event-exec",
            signal_id="event-sig",
            reservation_id="event-res",
        ),
    )

    assert fake_event_bus.count("position.opened") == 1

    payload = fake_event_bus.last_payload("position.opened")
    assert payload["symbol"] == "BTCUSDT"
    assert payload["side"] == "long"
    assert payload["size"] == pytest.approx(0.01)
    assert payload["entry_price"] == pytest.approx(50_000.0)
    assert payload["execution_id"] if "execution_id" in payload else True
    assert payload["signal_id"] == "event-sig"
    assert payload["reservation_id"] == "event-res"


async def test_event_execution_order_partially_filled_then_filled_applies_delta_only(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    from conftest import make_order_filled_payload as build_payload

    await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "execution.order_partially_filled",
        build_payload(
            order_id="event-cumulative",
            client_order_id="event-cumulative-client",
            executed_qty=0.01,
            orig_qty=0.02,
            status="PARTIALLY_FILLED",
        ),
    )

    await fake_event_bus.emit(
        "execution.order_filled",
        build_payload(
            order_id="event-cumulative",
            client_order_id="event-cumulative-client",
            executed_qty=0.02,
            orig_qty=0.02,
            status="FILLED",
        ),
    )

    position = None
    for item in fake_event_bus.payloads("position.opened") + fake_event_bus.payloads("position.updated"):
        if item["symbol"] == "BTCUSDT":
            position = item

    assert position is not None
    assert position["size"] == pytest.approx(0.02)

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.updated") == 1


async def test_event_missing_valid_price_emits_position_sync_required(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    from conftest import make_order_filled_payload as build_payload

    await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    payload = build_payload(
        avg_price=0.0,
        executed_qty=0.01,
    )
    payload["price"] = 0.0
    payload["avg_price"] = 0.0
    payload["cum_quote"] = 0.0
    payload["cumulative_quote_quantity"] = 0.0

    await fake_event_bus.emit("execution.order_filled", payload)

    assert fake_event_bus.count("position.opened") == 0
    assert fake_event_bus.count("position.sync_required") == 1

    sync_payload = fake_event_bus.last_payload("position.sync_required")
    assert sync_payload["symbol"] == "BTCUSDT"
    assert sync_payload["reason"] == "failed_to_apply_order_filled"


async def test_risk_kill_switch_emits_position_kill_switch_snapshot(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="kill-snapshot-order",
            client_order_id="kill-snapshot-client",
            quantity=0.01,
            price=50_000.0,
        )
    )

    await fake_event_bus.emit(
        "risk.kill_switch",
        {
            "reason": "daily_loss_limit",
        },
    )

    assert fake_event_bus.count("position.kill_switch_snapshot") == 1

    payload = fake_event_bus.last_payload("position.kill_switch_snapshot")
    assert payload["open_positions"] == 1
    assert payload["gross_notional"] == pytest.approx(500.0)
    assert payload["reason"] == "daily_loss_limit"


# =============================================================================
# Reconciliation / snapshots
# =============================================================================


async def test_apply_exchange_position_snapshot_opens_position(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    snapshot = PositionSnapshot.from_exchange_position(
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "position_amt": 0.01,
            "entry_price": 50_000.0,
            "break_even_price": 50_000.0,
            "mark_price": 50_500.0,
            "unrealized_profit": 5.0,
            "liquidation_price": 40_000.0,
            "leverage": 5,
            "margin_type": "isolated",
            "isolated_margin": 101.0,
            "notional": 505.0,
            "update_time": 1_700_000_000_000,
        }
    )

    update = await manager.apply_position_snapshot(snapshot)

    assert update is not None
    assert update.update_type == "opened"
    assert update.side is PositionSide.LONG
    assert update.size == pytest.approx(0.01)
    assert update.entry_price == pytest.approx(50_000.0)
    assert update.mark_price == pytest.approx(50_500.0)
    assert update.unrealized_pnl == pytest.approx(5.0)

    assert fake_event_bus.count("position.opened") == 1


async def test_apply_flat_exchange_snapshot_closes_existing_position(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="open-before-flat-snapshot",
            client_order_id="open-before-flat-snapshot-client",
            quantity=0.01,
            price=50_000.0,
        )
    )

    snapshot = PositionSnapshot.from_exchange_position(
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "position_side": "LONG",
            "position_amt": 0.0,
            "entry_price": 0.0,
            "break_even_price": 0.0,
            "mark_price": 50_500.0,
            "unrealized_profit": 0.0,
            "liquidation_price": 0.0,
            "leverage": 5,
            "margin_type": "isolated",
            "isolated_margin": 0.0,
            "notional": 0.0,
            "update_time": 1_700_000_000_000,
        }
    )

    update = await manager.apply_position_snapshot(snapshot)

    assert update is not None
    assert update.update_type == "closed"
    assert update.previous_size == pytest.approx(0.01)
    assert update.size == pytest.approx(0.0)

    assert fake_event_bus.count("position.closed") == 1


async def test_sync_positions_uses_binance_get_positions_and_emits_sync_completed(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    from conftest import make_binance_position_snapshot

    fake_binance_client.positions = [
        make_binance_position_snapshot(
            symbol="BTCUSDT",
            position_side="LONG",
            position_amt=0.01,
            entry_price=50_000.0,
            mark_price=50_500.0,
            unrealized_profit=5.0,
        )
    ]

    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    snapshots = await manager.sync_positions()

    assert len(snapshots) == 1
    assert len(fake_binance_client.calls_for("get_positions")) == 1

    assert fake_event_bus.count("position.opened") == 1
    assert fake_event_bus.count("position.sync_completed") == 1

    position = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    assert position is not None
    assert position.size == pytest.approx(0.01)
    assert position.unrealized_pnl == pytest.approx(5.0)


async def test_sync_positions_failure_emits_sync_required_and_raises_sync_error(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    from execution.exceptions import PositionSyncError

    fake_binance_client.get_positions_exception = RuntimeError("position endpoint failed")

    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    with pytest.raises(PositionSyncError):
        await manager.sync_positions()

    assert fake_event_bus.count("position.sync_required") == 1
    assert manager.stats.reconciliation_failures == 1

    payload = fake_event_bus.last_payload("position.sync_required")
    assert "position endpoint failed" in payload["error"]


async def test_reconcile_positions_runs_sync_and_emits_reconciled(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    from conftest import make_binance_position_snapshot

    fake_binance_client.positions = [
        make_binance_position_snapshot(
            symbol="BTCUSDT",
            position_side="LONG",
            position_amt=0.01,
        )
    ]

    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.reconcile_positions()

    assert manager.stats.reconciliation_runs == 1
    assert fake_event_bus.count("position.sync_completed") == 1
    assert fake_event_bus.count("position.reconciled") == 1


async def test_scheduler_job_can_run_position_reconciliation(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    fake_binance_client,
    exchange_clients,
):
    from conftest import make_binance_position_snapshot

    fake_binance_client.positions = [
        make_binance_position_snapshot(
            symbol="BTCUSDT",
            position_side="LONG",
            position_amt=0.01,
        )
    ]

    await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_scheduler.run_job("execution.position_manager.reconcile_positions")

    assert fake_event_bus.count("position.reconciled") == 1


async def test_event_exchange_positions_snapshot_applies_items(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    from conftest import make_binance_position_snapshot

    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await fake_event_bus.emit(
        "exchange.positions.snapshot",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "positions": [
                make_binance_position_snapshot(
                    symbol="BTCUSDT",
                    position_side="LONG",
                    position_amt=0.01,
                ),
                make_binance_position_snapshot(
                    symbol="ETHUSDT",
                    position_side="SHORT",
                    position_amt=-0.2,
                    entry_price=3_000.0,
                    mark_price=2_990.0,
                ),
            ],
        },
    )

    btc = manager.get_position(symbol="BTCUSDT", side=PositionSide.LONG)
    eth = manager.get_position(symbol="ETHUSDT", side=PositionSide.SHORT)

    assert btc is not None
    assert eth is not None
    assert btc.size == pytest.approx(0.01)
    assert eth.size == pytest.approx(0.2)

    assert fake_event_bus.count("position.opened") == 2


# =============================================================================
# Read APIs / exposure / snapshots
# =============================================================================


async def test_list_positions_and_has_open_position_filter_correctly(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            symbol="BTCUSDT",
            order_id="btc-order",
            client_order_id="btc-client",
            quantity=0.01,
            price=50_000.0,
        )
    )
    await manager.apply_fill(
        make_fill(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            order_id="eth-order",
            client_order_id="eth-client",
            quantity=0.2,
            price=3_000.0,
            metadata={
                "final_notional": 600.0,
            },
        )
    )

    assert manager.has_open_position(symbol="BTCUSDT", side=PositionSide.LONG) is True
    assert manager.has_open_position(symbol="BTCUSDT", side=PositionSide.SHORT) is False
    assert manager.has_open_position(symbol="ETHUSDT", side=PositionSide.SHORT) is True

    all_open = manager.list_positions()
    btc_only = manager.list_positions(symbol="BTCUSDT")

    assert len(all_open) == 2
    assert len(btc_only) == 1
    assert btc_only[0].symbol == "BTCUSDT"


async def test_calculate_exposure_aggregates_open_positions(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            symbol="BTCUSDT",
            order_id="btc-exposure",
            client_order_id="btc-exposure-client",
            quantity=0.01,
            price=50_000.0,
            metadata={
                "final_margin": 100.0,
                "final_notional": 500.0,
            },
        )
    )
    await manager.apply_fill(
        make_fill(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            position_side=PositionSide.SHORT,
            order_id="eth-exposure",
            client_order_id="eth-exposure-client",
            quantity=0.2,
            price=3_000.0,
            metadata={
                "final_margin": 120.0,
                "final_notional": 600.0,
            },
        )
    )

    exposure = manager.calculate_exposure()

    assert exposure["open_positions"] == 2
    assert exposure["gross_notional"] == pytest.approx(1100.0)
    assert exposure["margin_used"] == pytest.approx(220.0)
    assert exposure["by_symbol"]["BTCUSDT"] == pytest.approx(500.0)
    assert exposure["by_symbol"]["ETHUSDT"] == pytest.approx(600.0)
    assert exposure["by_side"]["long"] == pytest.approx(500.0)
    assert exposure["by_side"]["short"] == pytest.approx(600.0)


async def test_snapshot_contains_risk_compatible_portfolio_position_payload(
    position_manager_config,
    fake_event_bus,
    fake_scheduler,
    exchange_clients,
):
    manager = await start_position_manager(
        position_manager_config=position_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        exchange_clients=exchange_clients,
    )

    await manager.apply_fill(
        make_fill(
            order_id="snapshot-order",
            client_order_id="snapshot-client",
            quantity=0.01,
            price=50_000.0,
        )
    )

    snapshot = manager.snapshot()

    assert snapshot["service"] == "execution.position_manager"
    assert snapshot["running"] is True
    assert snapshot["positions_count"] == 1
    assert snapshot["open_positions_count"] == 1
    assert snapshot["stats"]["opened"] == 1

    positions = snapshot["positions"]
    assert len(positions) == 1

    position_payload = positions[0]

    required_risk_keys = {
        "position_id",
        "exchange",
        "market_type",
        "symbol",
        "side",
        "size",
        "entry_price",
        "mark_price",
        "notional_value",
        "leverage",
        "margin_used",
        "risk_amount",
        "stop_loss",
        "take_profit",
        "tier",
        "strategy_name",
        "signal_id",
        "reservation_id",
        "realized_pnl",
        "unrealized_pnl",
        "opened_at",
        "updated_at",
        "closed_at",
        "metadata",
    }

    assert required_risk_keys.issubset(position_payload.keys())
    assert position_payload["symbol"] == "BTCUSDT"
    assert position_payload["side"] == "long"
    assert position_payload["size"] == pytest.approx(0.01)
    assert position_payload["entry_price"] == pytest.approx(50_000.0)
    assert position_payload["strategy_name"] == "test_strategy"
    assert position_payload["signal_id"] == "sig-1"
    assert position_payload["reservation_id"] == "res-1"