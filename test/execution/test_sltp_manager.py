from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from execution.enums import OrderSide, OrderStatus, OrderType, SLTPType, TriggerType
from execution.models import OrderRequest, OrderResult
from execution.sl_tp_manager import SLTPManager

from risk.enums import PositionSide


pytestmark = pytest.mark.asyncio


# =============================================================================
# Fake OrderManager for SLTPManager tests
# =============================================================================


class FakeOrderManagerForSLTP:
    def __init__(self) -> None:
        self.submitted_requests: list[OrderRequest] = []
        self.cancel_requests: list[dict[str, Any]] = []
        self.cancel_all_requests: list[dict[str, Any]] = []

        self.submit_exception: Exception | None = None
        self.cancel_exception: Exception | None = None

        self.next_order_id = 10_000
        self.force_submit_status: OrderStatus = OrderStatus.NEW

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        request.validate()

        if self.submit_exception is not None:
            raise self.submit_exception

        self.submitted_requests.append(request)
        self.next_order_id += 1

        return OrderResult(
            exchange=request.exchange,
            market_type=request.market_type,
            symbol=request.symbol,
            order_id=str(self.next_order_id),
            client_order_id=request.client_order_id or f"fake-sltp-{self.next_order_id}",
            status=self.force_submit_status,
            side=request.side,
            order_type=request.order_type,
            price=request.price,
            avg_price=None,
            original_quantity=request.quantity or 0.0,
            executed_quantity=0.0,
            cumulative_quote_quantity=0.0,
            position_side=request.position_side.value if request.position_side else None,
            reduce_only=request.reduce_only,
            close_position=request.close_position,
            stop_price=request.stop_price,
            working_type=request.working_type.value if request.working_type else None,
            execution_id=request.execution_id,
            leg_id=request.leg_id,
            signal_id=request.signal_id,
            strategy_name=request.strategy_name,
            reservation_id=request.reservation_id,
            metadata=dict(request.metadata),
        )

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: str | int | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OrderResult:
        if self.cancel_exception is not None:
            raise self.cancel_exception

        payload = {
            "symbol": symbol,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "exchange": exchange,
            "reason": reason,
            "metadata": dict(metadata or {}),
        }
        self.cancel_requests.append(payload)

        return OrderResult(
            exchange=exchange or "binance",
            market_type="usdm_futures",
            symbol=symbol,
            order_id=str(order_id) if order_id is not None else None,
            client_order_id=client_order_id,
            status=OrderStatus.CANCELED,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            original_quantity=0.0,
            executed_quantity=0.0,
            cumulative_quote_quantity=0.0,
            metadata=dict(metadata or {}),
        )

    async def cancel_all_orders(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "reason": reason,
        }
        self.cancel_all_requests.append(payload)
        return {
            "symbol": symbol,
            "exchange": exchange or "binance",
            "status": "ok",
        }


# =============================================================================
# Helpers
# =============================================================================


async def start_sltp_manager(
    *,
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
    order_manager: FakeOrderManagerForSLTP | None = None,
) -> tuple[SLTPManager, FakeOrderManagerForSLTP]:
    order_manager = order_manager or FakeOrderManagerForSLTP()

    manager = SLTPManager(
        sltp_manager_config,
        order_manager=order_manager,
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
    )
    await manager.start()
    return manager, order_manager


def make_position_payload(
    *,
    symbol: str = "BTCUSDT",
    side: PositionSide = PositionSide.LONG,
    size: float = 0.01,
    entry_price: float = 50_000.0,
    mark_price: float = 50_000.0,
    stop_loss: float | None = 49_000.0,
    take_profit: float | None = 53_000.0,
    position_id: str = "pos-1",
    signal_id: str = "sig-1",
    strategy_name: str = "test_strategy",
    reservation_id: str = "res-1",
) -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": symbol,
        "side": side.value,
        "size": size,
        "entry_price": entry_price,
        "mark_price": mark_price,
        "notional_value": size * mark_price,
        "leverage": 5.0,
        "margin_used": size * mark_price / 5.0,
        "risk_amount": 50.0,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "tier": "t2",
        "strategy_name": strategy_name,
        "signal_id": signal_id,
        "reservation_id": reservation_id,
        "position_id": position_id,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "metadata": {},
    }


# =============================================================================
# Lifecycle / registration
# =============================================================================


async def test_sltp_manager_start_registers_subscriptions_and_scheduler_job(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    assert manager.is_running is True

    subscription_topics = {subscription.topic for subscription in fake_event_bus.subscriptions}

    assert "position.opened" in subscription_topics
    assert "position.updated" in subscription_topics
    assert "position.closed" in subscription_topics
    assert "risk.stop_update_requested" in subscription_topics
    assert "risk.take_profit_update_requested" in subscription_topics
    assert "risk.trailing_stop_requested" in subscription_topics
    assert "risk.position_close_requested" in subscription_topics
    assert "execution.order_filled" in subscription_topics
    assert "execution.order_cancelled" in subscription_topics
    assert "execution.order_rejected" in subscription_topics
    assert "execution.order_failed" in subscription_topics

    assert "execution.sltp_manager.reconcile_protective_orders" in fake_scheduler.job_names()
    assert fake_event_bus.count("execution.sltp_manager.started") == 1


async def test_sltp_manager_stop_unregisters_subscriptions(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    assert fake_event_bus.subscriptions

    await manager.stop()

    assert manager.is_running is False
    assert fake_event_bus.subscriptions == []
    assert fake_event_bus.count("execution.sltp_manager.stopped") == 1


# =============================================================================
# position.opened -> protective orders
# =============================================================================


async def test_position_opened_long_places_stop_loss_and_take_profit(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            side=PositionSide.LONG,
            size=0.01,
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    assert len(order_manager.submitted_requests) == 2

    stop_request = order_manager.submitted_requests[0]
    tp_request = order_manager.submitted_requests[1]

    assert stop_request.symbol == "BTCUSDT"
    assert stop_request.side is OrderSide.SELL
    assert stop_request.order_type is OrderType.STOP_MARKET
    assert stop_request.position_side is PositionSide.LONG
    assert stop_request.stop_price == pytest.approx(49_000.0)
    assert stop_request.close_position is True
    assert stop_request.reduce_only is False
    assert stop_request.quantity is None
    assert stop_request.trigger_type is TriggerType.STOP_LOSS
    assert stop_request.working_type.value == "MARK_PRICE"
    assert stop_request.price_protect is True

    assert tp_request.symbol == "BTCUSDT"
    assert tp_request.side is OrderSide.SELL
    assert tp_request.order_type is OrderType.TAKE_PROFIT_MARKET
    assert tp_request.position_side is PositionSide.LONG
    assert tp_request.stop_price == pytest.approx(53_000.0)
    assert tp_request.close_position is False
    assert tp_request.reduce_only is True
    assert tp_request.quantity == pytest.approx(0.01)
    assert tp_request.trigger_type is TriggerType.TAKE_PROFIT

    assert fake_event_bus.count("execution.sltp.place_requested") == 2
    assert fake_event_bus.count("execution.stop_loss_placed") == 1
    assert fake_event_bus.count("execution.take_profit_placed") == 1
    assert fake_event_bus.count("execution.sltp.place_completed") == 1

    stop_payload = fake_event_bus.last_payload("execution.stop_loss_placed")
    assert stop_payload["symbol"] == "BTCUSDT"
    assert stop_payload["position_side"] == "long"
    assert stop_payload["stop_loss"] == pytest.approx(49_000.0)
    assert stop_payload["reservation_id"] == "res-1"

    tracked = manager.list_protective_orders(symbol="BTCUSDT")
    assert len(tracked) == 2
    assert {state.sltp_type for state in tracked} == {SLTPType.STOP_LOSS, SLTPType.TAKE_PROFIT}


async def test_position_opened_short_places_buy_protective_orders(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            side=PositionSide.SHORT,
            size=0.02,
            entry_price=50_000.0,
            stop_loss=51_000.0,
            take_profit=47_000.0,
            position_id="short-pos-1",
        ),
    )

    assert len(order_manager.submitted_requests) == 2

    stop_request = order_manager.submitted_requests[0]
    tp_request = order_manager.submitted_requests[1]

    assert stop_request.side is OrderSide.BUY
    assert stop_request.position_side is PositionSide.SHORT
    assert stop_request.order_type is OrderType.STOP_MARKET
    assert stop_request.stop_price == pytest.approx(51_000.0)

    assert tp_request.side is OrderSide.BUY
    assert tp_request.position_side is PositionSide.SHORT
    assert tp_request.order_type is OrderType.TAKE_PROFIT_MARKET
    assert tp_request.stop_price == pytest.approx(47_000.0)
    assert tp_request.quantity == pytest.approx(0.02)


async def test_position_opened_without_stop_or_take_profit_does_nothing(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            stop_loss=None,
            take_profit=None,
        ),
    )

    assert order_manager.submitted_requests == []
    assert manager.list_protective_orders(symbol="BTCUSDT") == []
    assert fake_event_bus.count("execution.stop_loss_placed") == 0
    assert fake_event_bus.count("execution.take_profit_placed") == 0


async def test_position_opened_with_only_stop_loss_places_only_stop(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    assert len(order_manager.submitted_requests) == 1
    assert order_manager.submitted_requests[0].order_type is OrderType.STOP_MARKET

    tracked = manager.list_protective_orders(symbol="BTCUSDT")
    assert len(tracked) == 1
    assert tracked[0].sltp_type is SLTPType.STOP_LOSS


async def test_position_opened_with_only_take_profit_places_only_take_profit(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            stop_loss=None,
            take_profit=53_000.0,
        ),
    )

    assert len(order_manager.submitted_requests) == 1
    assert order_manager.submitted_requests[0].order_type is OrderType.TAKE_PROFIT_MARKET

    tracked = manager.list_protective_orders(symbol="BTCUSDT")
    assert len(tracked) == 1
    assert tracked[0].sltp_type is SLTPType.TAKE_PROFIT


# =============================================================================
# Cancel on position.closed / close requested
# =============================================================================


async def test_position_closed_cancels_tracked_protective_orders(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-cancel-1",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    assert len(manager.list_protective_orders(symbol="BTCUSDT", include_terminal=False)) == 2

    await fake_event_bus.emit(
        "position.closed",
        {
            **make_position_payload(
                position_id="pos-cancel-1",
                stop_loss=49_000.0,
                take_profit=53_000.0,
            ),
            "update_type": "closed",
            "size": 0.0,
            "previous_side": "long",
        },
    )

    assert len(order_manager.cancel_requests) == 2
    assert all(item["symbol"] == "BTCUSDT" for item in order_manager.cancel_requests)
    assert all(item["reason"] == "position_closed" for item in order_manager.cancel_requests)

    assert fake_event_bus.count("execution.sltp.cancel_completed") == 2

    tracked = manager.list_protective_orders(symbol="BTCUSDT", include_terminal=True)
    assert len(tracked) == 2
    assert all(state.status is OrderStatus.CANCELED for state in tracked)


async def test_risk_position_close_requested_cancels_protective_orders_before_close_flow(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-risk-close-1",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    await fake_event_bus.emit(
        "risk.position_close_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "position_id": "pos-risk-close-1",
            "reason": "risk_close",
        },
    )

    assert len(order_manager.cancel_requests) == 2
    assert all(item["reason"] == "risk_position_close_requested" for item in order_manager.cancel_requests)

    tracked = manager.list_protective_orders(symbol="BTCUSDT", include_terminal=True)
    assert all(state.status is OrderStatus.CANCELED for state in tracked)


async def test_cancel_protective_orders_filters_by_sltp_type_only(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-filter-1",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    cancelled = await manager.cancel_protective_orders(
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        position_id="pos-filter-1",
        sltp_type=SLTPType.STOP_LOSS,
        reason="cancel_stop_only",
    )

    assert len(cancelled) == 1
    assert cancelled[0].sltp_type is SLTPType.STOP_LOSS
    assert len(order_manager.cancel_requests) == 1
    assert order_manager.cancel_requests[0]["reason"] == "cancel_stop_only"

    tracked = manager.list_protective_orders(symbol="BTCUSDT", include_terminal=True)
    stop = [state for state in tracked if state.sltp_type is SLTPType.STOP_LOSS][0]
    tp = [state for state in tracked if state.sltp_type is SLTPType.TAKE_PROFIT][0]

    assert stop.status is OrderStatus.CANCELED
    assert tp.status is OrderStatus.NEW


# =============================================================================
# Risk update requests
# =============================================================================


async def test_risk_stop_update_requested_replaces_existing_stop_loss(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-stop-update-1",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    assert len(order_manager.submitted_requests) == 1

    await fake_event_bus.emit(
        "risk.stop_update_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            "stop_loss": 49_500.0,
            "position_id": "pos-stop-update-1",
            "signal_id": "sig-1",
            "strategy_name": "test_strategy",
            "reservation_id": "res-1",
            "metadata": {"reason": "move_stop"},
        },
    )

    assert len(order_manager.cancel_requests) == 1
    assert order_manager.cancel_requests[0]["reason"] == "update_stop_loss"

    assert len(order_manager.submitted_requests) == 2

    new_stop = order_manager.submitted_requests[-1]
    assert new_stop.order_type is OrderType.STOP_MARKET
    assert new_stop.stop_price == pytest.approx(49_500.0)
    assert new_stop.trigger_type is TriggerType.STOP_LOSS
    assert new_stop.close_position is True

    assert fake_event_bus.count("execution.sltp.update_completed") == 1

    tracked = manager.list_protective_orders(symbol="BTCUSDT", include_terminal=True)
    assert len(tracked) == 2
    assert any(state.status is OrderStatus.CANCELED for state in tracked)
    assert any(state.status is OrderStatus.NEW and state.stop_price == pytest.approx(49_500.0) for state in tracked)


async def test_risk_take_profit_update_requested_replaces_existing_take_profit(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-tp-update-1",
            stop_loss=None,
            take_profit=53_000.0,
        ),
    )

    await fake_event_bus.emit(
        "risk.take_profit_update_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            "take_profit": 54_000.0,
            "position_id": "pos-tp-update-1",
            "signal_id": "sig-1",
            "strategy_name": "test_strategy",
            "reservation_id": "res-1",
            "metadata": {"reason": "move_take_profit"},
        },
    )

    assert len(order_manager.cancel_requests) == 1
    assert order_manager.cancel_requests[0]["reason"] == "update_take_profit"

    assert len(order_manager.submitted_requests) == 2

    new_tp = order_manager.submitted_requests[-1]
    assert new_tp.order_type is OrderType.TAKE_PROFIT_MARKET
    assert new_tp.stop_price == pytest.approx(54_000.0)
    assert new_tp.trigger_type is TriggerType.TAKE_PROFIT
    assert new_tp.reduce_only is True
    assert new_tp.quantity == pytest.approx(0.01)

    assert fake_event_bus.count("execution.sltp.update_completed") == 1


async def test_risk_trailing_stop_requested_places_trailing_stop_with_clamped_callback_rate_low(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "risk.trailing_stop_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            "callback_rate": 0.01,
            "activation_price": 51_000.0,
            "position_id": "pos-trailing-low",
            "signal_id": "sig-1",
            "strategy_name": "test_strategy",
            "reservation_id": "res-1",
        },
    )

    assert len(order_manager.submitted_requests) == 1

    request = order_manager.submitted_requests[0]
    assert request.order_type is OrderType.TRAILING_STOP_MARKET
    assert request.side is OrderSide.SELL
    assert request.position_side is PositionSide.LONG
    assert request.reduce_only is True
    assert request.close_position is False
    assert request.activation_price == pytest.approx(51_000.0)
    assert request.callback_rate == pytest.approx(sltp_manager_config.min_trailing_callback_rate)
    assert request.trigger_type is TriggerType.TRAILING_STOP

    assert fake_event_bus.count("execution.trailing_stop_updated") == 1


async def test_risk_trailing_stop_requested_places_trailing_stop_with_clamped_callback_rate_high(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "risk.trailing_stop_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "short",
            "size": 0.02,
            "callback_rate": 50.0,
            "activation_price": 49_000.0,
            "position_id": "pos-trailing-high",
            "signal_id": "sig-1",
            "strategy_name": "test_strategy",
            "reservation_id": "res-1",
        },
    )

    assert len(order_manager.submitted_requests) == 1

    request = order_manager.submitted_requests[0]
    assert request.order_type is OrderType.TRAILING_STOP_MARKET
    assert request.side is OrderSide.BUY
    assert request.position_side is PositionSide.SHORT
    assert request.callback_rate == pytest.approx(sltp_manager_config.max_trailing_callback_rate)
    assert request.quantity == pytest.approx(0.02)


async def test_invalid_stop_update_payload_emits_update_failed(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "risk.stop_update_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            # missing stop_loss
            "position_id": "pos-invalid-stop",
        },
    )

    assert order_manager.submitted_requests == []
    assert fake_event_bus.count("execution.sltp.update_failed") == 1

    payload = fake_event_bus.last_payload("execution.sltp.update_failed")
    assert payload["update_type"] == "stop_loss"
    assert "stop_loss" in payload["error"]


async def test_invalid_trailing_stop_payload_emits_update_failed(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "risk.trailing_stop_requested",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "side": "long",
            "size": 0.01,
            # missing callback_rate
            "position_id": "pos-invalid-trailing",
        },
    )

    assert order_manager.submitted_requests == []
    assert fake_event_bus.count("execution.sltp.update_failed") == 1

    payload = fake_event_bus.last_payload("execution.sltp.update_failed")
    assert payload["update_type"] == "trailing_stop"
    assert "callback_rate" in payload["error"]


# =============================================================================
# Order lifecycle updates for protective states
# =============================================================================


async def test_execution_order_filled_marks_stop_loss_as_triggered(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-trigger-stop",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    stop_state = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert stop_state.sltp_type is SLTPType.STOP_LOSS

    await fake_event_bus.emit(
        "execution.order_filled",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "order_id": stop_state.order_id,
            "client_order_id": stop_state.client_order_id,
            "status": "FILLED",
            "metadata": {},
        },
    )

    tracked = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert tracked.status is OrderStatus.FILLED

    assert fake_event_bus.count("execution.stop_loss_triggered") == 1

    payload = fake_event_bus.last_payload("execution.stop_loss_triggered")
    assert payload["sltp_type"] == "stop_loss"
    assert payload["order_id"] == stop_state.order_id
    assert payload["client_order_id"] == stop_state.client_order_id


async def test_execution_order_filled_marks_take_profit_as_triggered(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-trigger-tp",
            stop_loss=None,
            take_profit=53_000.0,
        ),
    )

    tp_state = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert tp_state.sltp_type is SLTPType.TAKE_PROFIT

    await fake_event_bus.emit(
        "execution.order_filled",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "order_id": tp_state.order_id,
            "client_order_id": tp_state.client_order_id,
            "status": "FILLED",
            "metadata": {},
        },
    )

    tracked = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert tracked.status is OrderStatus.FILLED

    assert fake_event_bus.count("execution.take_profit_triggered") == 1


async def test_execution_order_cancelled_updates_protective_state_to_cancelled(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-cancel-update",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    state = manager.list_protective_orders(symbol="BTCUSDT")[0]

    await fake_event_bus.emit(
        "execution.order_cancelled",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "order_id": state.order_id,
            "client_order_id": state.client_order_id,
            "status": "CANCELED",
        },
    )

    tracked = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert tracked.status is OrderStatus.CANCELED


async def test_execution_order_rejected_updates_protective_state_to_rejected(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-rejected-update",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    state = manager.list_protective_orders(symbol="BTCUSDT")[0]

    await fake_event_bus.emit(
        "execution.order_rejected",
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "order_id": state.order_id,
            "client_order_id": state.client_order_id,
            "status": "REJECTED",
        },
    )

    tracked = manager.list_protective_orders(symbol="BTCUSDT")[0]
    assert tracked.status is OrderStatus.REJECTED


# =============================================================================
# Failure paths
# =============================================================================


async def test_position_opened_submit_failure_emits_sltp_place_failed(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    order_manager = FakeOrderManagerForSLTP()
    order_manager.submit_exception = RuntimeError("submit protective failed")

    _, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        order_manager=order_manager,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-submit-failure",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    assert len(order_manager.submitted_requests) == 0
    assert fake_event_bus.count("execution.sltp.place_failed") == 1

    payload = fake_event_bus.last_payload("execution.sltp.place_failed")
    assert payload["symbol"] == "BTCUSDT"
    assert "submit protective failed" in payload["error"]


async def test_cancel_protective_order_failure_emits_cancel_failed(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    order_manager = FakeOrderManagerForSLTP()

    manager, order_manager = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        order_manager=order_manager,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-cancel-failure",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    order_manager.cancel_exception = RuntimeError("cancel protective failed")

    cancelled = await manager.cancel_protective_orders(
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        position_id="pos-cancel-failure",
        reason="test_cancel_failure",
    )

    assert cancelled == []
    assert fake_event_bus.count("execution.sltp.cancel_failed") == 1

    payload = fake_event_bus.last_payload("execution.sltp.cancel_failed")
    assert payload["symbol"] == "BTCUSDT"
    assert "cancel protective failed" in payload["error"]


async def test_position_closed_cancel_failure_does_not_raise_to_eventbus(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    order_manager = FakeOrderManagerForSLTP()

    await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        order_manager=order_manager,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-close-cancel-failure",
            stop_loss=49_000.0,
            take_profit=None,
        ),
    )

    order_manager.cancel_exception = RuntimeError("cancel failed from closed")

    await fake_event_bus.emit(
        "position.closed",
        {
            **make_position_payload(
                position_id="pos-close-cancel-failure",
                stop_loss=49_000.0,
                take_profit=None,
            ),
            "update_type": "closed",
            "size": 0.0,
        },
    )

    assert fake_event_bus.count("execution.sltp.cancel_failed") == 1


# =============================================================================
# Reconciliation / snapshot / diagnostics
# =============================================================================


async def test_reconcile_protective_orders_emits_active_count(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-reconcile",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    await manager.reconcile_protective_orders()

    assert fake_event_bus.count("execution.sltp.reconciled") == 1

    payload = fake_event_bus.last_payload("execution.sltp.reconciled")
    assert payload["active_protective_orders"] == 2
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"


async def test_scheduler_job_can_run_sltp_reconciliation(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_scheduler.run_job("execution.sltp_manager.reconcile_protective_orders")

    assert fake_event_bus.count("execution.sltp.reconciled") == 1


async def test_position_updated_resizes_tracked_protective_orders(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-resize",
            size=0.01,
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    await fake_event_bus.emit(
        "position.updated",
        {
            **make_position_payload(
                position_id="pos-resize",
                size=0.02,
                stop_loss=49_000.0,
                take_profit=53_000.0,
            ),
            "update_type": "updated",
        },
    )

    tracked = manager.list_protective_orders(symbol="BTCUSDT", include_terminal=False)

    assert len(tracked) == 2
    assert all(state.quantity == pytest.approx(0.02) or state.close_position for state in tracked)
    assert fake_event_bus.count("execution.sltp.resized") == 1


async def test_snapshot_contains_tracked_protective_orders_and_stats(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            position_id="pos-snapshot",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )

    snapshot = manager.snapshot()

    assert snapshot["service"] == "execution.sltp_manager"
    assert snapshot["running"] is True
    assert snapshot["positions_tracked"] == 1
    assert len(snapshot["protective_orders"]) == 2
    assert snapshot["stats"]["stop_loss_placed"] == 1
    assert snapshot["stats"]["take_profit_placed"] == 1
    assert snapshot["stats"]["active_protective_orders"] == 2

    order_payload = snapshot["protective_orders"][0]
    required_keys = {
        "exchange",
        "market_type",
        "symbol",
        "sltp_type",
        "status",
        "order_id",
        "client_order_id",
        "position_id",
        "position_side",
        "price",
        "stop_price",
        "quantity",
        "reduce_only",
        "close_position",
        "created_at",
        "updated_at",
        "metadata",
    }
    assert required_keys.issubset(order_payload.keys())


async def test_list_protective_orders_filters_symbol_position_and_terminal_status(
    sltp_manager_config,
    fake_event_bus,
    fake_scheduler,
):
    manager, _ = await start_sltp_manager(
        sltp_manager_config=sltp_manager_config,
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
    )

    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            symbol="BTCUSDT",
            position_id="btc-pos",
            stop_loss=49_000.0,
            take_profit=53_000.0,
        ),
    )
    await fake_event_bus.emit(
        "position.opened",
        make_position_payload(
            symbol="ETHUSDT",
            position_id="eth-pos",
            size=0.2,
            entry_price=3_000.0,
            mark_price=3_000.0,
            stop_loss=2_900.0,
            take_profit=3_200.0,
        ),
    )

    btc_orders = manager.list_protective_orders(symbol="BTCUSDT")
    eth_orders = manager.list_protective_orders(symbol="ETHUSDT")
    btc_position_orders = manager.list_protective_orders(position_id="btc-pos")

    assert len(btc_orders) == 2
    assert len(eth_orders) == 2
    assert len(btc_position_orders) == 2

    await manager.cancel_protective_orders(
        symbol="BTCUSDT",
        position_id="btc-pos",
        position_side=PositionSide.LONG,
        sltp_type=SLTPType.STOP_LOSS,
        reason="filter_test",
    )

    active_btc = manager.list_protective_orders(
        symbol="BTCUSDT",
        include_terminal=False,
    )
    all_btc = manager.list_protective_orders(
        symbol="BTCUSDT",
        include_terminal=True,
    )

    assert len(active_btc) == 1
    assert len(all_btc) == 2