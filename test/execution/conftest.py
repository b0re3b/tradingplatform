from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pytest

from execution.config import (
    OrderManagerConfig,
    PositionManagerConfig,
    SLTPManagerConfig,
    SmartExecutionConfig,
    TradeExecutorConfig,
)
from execution.enums import ExecutionMode, OrderStatus
from risk.enums import MarginMode, OrderIntent, PositionSide, RiskMode, TradeTier


# =============================================================================
# Fake EventBus
# =============================================================================


@dataclass(slots=True)
class FakeEvent:
    topic: str
    payload: dict[str, Any]
    source: str | None = None
    priority: Any | None = None


@dataclass(slots=True)
class FakeSubscription:
    topic: str
    handler: Callable[..., Any]
    name: str | None = None


class FakeEventBus:
    """
    Minimal async-compatible EventBus for execution package tests.

    Supports:
    - subscribe(topic, handler, name=...)
    - unsubscribe(subscription)
    - emit(topic, payload, priority=..., source=...)

    Wildcard support:
    - exact topic match
    - prefix wildcard like "execution.*"
    """

    def __init__(self) -> None:
        self.subscriptions: list[FakeSubscription] = []
        self.emitted: list[FakeEvent] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> FakeSubscription:
        subscription = FakeSubscription(topic=topic, handler=handler, name=name)
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: FakeSubscription) -> None:
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    async def emit(
        self,
        topic: str,
        payload: Mapping[str, Any] | None = None,
        *,
        priority: Any | None = None,
        source: str | None = None,
        **_: Any,
    ) -> None:
        event = FakeEvent(
            topic=topic,
            payload=dict(payload or {}),
            source=source,
            priority=priority,
        )
        self.emitted.append(event)

        for subscription in list(self.subscriptions):
            if not self._matches(subscription.topic, topic):
                continue

            result = subscription.handler(event)
            if asyncio.iscoroutine(result):
                await result

    def clear(self) -> None:
        self.emitted.clear()

    def topics(self) -> list[str]:
        return [event.topic for event in self.emitted]

    def payloads(self, topic: str) -> list[dict[str, Any]]:
        return [event.payload for event in self.emitted if event.topic == topic]

    def last_payload(self, topic: str) -> dict[str, Any]:
        payloads = self.payloads(topic)
        assert payloads, f"No emitted payloads for topic={topic!r}"
        return payloads[-1]

    def count(self, topic: str) -> int:
        return len(self.payloads(topic))

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic == prefix or topic.startswith(prefix + ".")

        return False


# =============================================================================
# Fake Scheduler
# =============================================================================


@dataclass(slots=True)
class FakeSchedulerJob:
    name: str
    callback: Callable[..., Any]
    interval_seconds: float
    run_immediately: bool = False


class FakeScheduler:
    """
    Minimal Scheduler fake.

    It records jobs but does not run background loops.
    Tests can call run_job(name) explicitly.
    """

    def __init__(self) -> None:
        self.jobs: list[FakeSchedulerJob] = []

    def add_interval_job(
        self,
        callback: Callable[..., Any],
        *,
        interval_seconds: float,
        name: str,
        run_immediately: bool = False,
        **_: Any,
    ) -> FakeSchedulerJob:
        job = FakeSchedulerJob(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
            run_immediately=run_immediately,
        )
        self.jobs.append(job)
        return job

    async def run_job(self, name: str) -> None:
        for job in self.jobs:
            if job.name != name:
                continue

            result = job.callback()
            if asyncio.iscoroutine(result):
                await result
            return

        raise AssertionError(f"Scheduler job not found: {name}")

    def job_names(self) -> list[str]:
        return [job.name for job in self.jobs]


# =============================================================================
# Fake Binance USD-M Futures REST client
# =============================================================================


@dataclass(slots=True)
class FakeBinanceCall:
    method: str
    kwargs: dict[str, Any]


class FakeBinanceRestClient:
    """
    BinanceRestClient fake for execution tests.

    It simulates the methods used by:
    - OrderManager
    - PositionManager

    Tests can control behavior by setting:
    - next_create_order_response
    - create_order_exception
    - open_orders
    - positions
    - user_trades
    """

    EXCHANGE = "binance"
    SOURCE = "binance_futures_rest"

    def __init__(self) -> None:
        self.calls: list[FakeBinanceCall] = []

        self.next_order_id = 1000

        self.next_create_order_response: dict[str, Any] | None = None
        self.create_order_exception: Exception | None = None

        self.cancel_order_exception: Exception | None = None
        self.get_order_exception: Exception | None = None
        self.get_open_orders_exception: Exception | None = None
        self.get_positions_exception: Exception | None = None

        self.open_orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.user_trades: list[dict[str, Any]] = []

        self.cancelled_orders: list[dict[str, Any]] = []
        self.cancel_all_requests: list[dict[str, Any]] = []

    async def create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float | None = None,
        price: float | None = None,
        position_side: str | None = None,
        time_in_force: str | None = None,
        reduce_only: bool | None = None,
        new_client_order_id: str | None = None,
        stop_price: float | None = None,
        close_position: bool | None = None,
        activation_price: float | None = None,
        callback_rate: float | None = None,
        working_type: str | None = None,
        price_protect: bool | None = None,
        new_order_resp_type: str | None = "RESULT",
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "position_side": position_side,
            "time_in_force": time_in_force,
            "reduce_only": reduce_only,
            "new_client_order_id": new_client_order_id,
            "stop_price": stop_price,
            "close_position": close_position,
            "activation_price": activation_price,
            "callback_rate": callback_rate,
            "working_type": working_type,
            "price_protect": price_protect,
            "new_order_resp_type": new_order_resp_type,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("create_order", kwargs))

        if self.create_order_exception is not None:
            raise self.create_order_exception

        if self.next_create_order_response is not None:
            response = dict(self.next_create_order_response)
            self.next_create_order_response = None
            return response

        self.next_order_id += 1

        executed_qty = quantity or 0.0
        avg_price = price or stop_price or activation_price or 50_000.0

        status = "FILLED" if order_type == "MARKET" else "NEW"

        if order_type in {
            "STOP_MARKET",
            "TAKE_PROFIT_MARKET",
            "TRAILING_STOP_MARKET",
        }:
            status = "NEW"
            executed_qty = 0.0

        return self._normalized_order(
            symbol=symbol,
            order_id=self.next_order_id,
            client_order_id=new_client_order_id or f"fake-{self.next_order_id}",
            side=side,
            order_type=order_type,
            status=status,
            orig_qty=quantity or 0.0,
            executed_qty=executed_qty,
            price=price or 0.0,
            avg_price=avg_price if executed_qty else 0.0,
            position_side=position_side,
            reduce_only=reduce_only,
            close_position=close_position,
            stop_price=stop_price,
            working_type=working_type,
        )

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "symbol": symbol,
            "order_id": order_id,
            "orig_client_order_id": orig_client_order_id,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("cancel_order", kwargs))

        if self.cancel_order_exception is not None:
            raise self.cancel_order_exception

        payload = self._normalized_order(
            symbol=symbol,
            order_id=order_id or self.next_order_id,
            client_order_id=orig_client_order_id or f"fake-{order_id or self.next_order_id}",
            side="BUY",
            order_type="LIMIT",
            status="CANCELED",
            orig_qty=0.01,
            executed_qty=0.0,
            price=50_000.0,
            avg_price=0.0,
        )
        self.cancelled_orders.append(payload)
        return payload

    async def cancel_all_open_orders(
        self,
        *,
        symbol: str,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "symbol": symbol,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("cancel_all_open_orders", kwargs))
        self.cancel_all_requests.append(kwargs)

        return {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": symbol,
            "code": 200,
            "message": "success",
            "timestamp": 1_700_000_000_000,
        }

    async def get_order(
        self,
        *,
        symbol: str,
        order_id: int | None = None,
        orig_client_order_id: str | None = None,
        recv_window: int | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "symbol": symbol,
            "order_id": order_id,
            "orig_client_order_id": orig_client_order_id,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("get_order", kwargs))

        if self.get_order_exception is not None:
            raise self.get_order_exception

        return self._normalized_order(
            symbol=symbol,
            order_id=order_id or self.next_order_id,
            client_order_id=orig_client_order_id or f"fake-{order_id or self.next_order_id}",
            side="BUY",
            order_type="MARKET",
            status="FILLED",
            orig_qty=0.01,
            executed_qty=0.01,
            price=0.0,
            avg_price=50_000.0,
            position_side="LONG",
        )

    async def get_open_orders(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        kwargs = {
            "symbol": symbol,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("get_open_orders", kwargs))

        if self.get_open_orders_exception is not None:
            raise self.get_open_orders_exception

        if symbol is None:
            return list(self.open_orders)

        return [
            order
            for order in self.open_orders
            if order.get("symbol") == symbol
        ]

    async def get_user_trades(
        self,
        *,
        symbol: str,
        limit: int = 500,
        order_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        kwargs = {
            "symbol": symbol,
            "limit": limit,
            "order_id": order_id,
            "start_time": start_time,
            "end_time": end_time,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("get_user_trades", kwargs))

        trades = [
            trade
            for trade in self.user_trades
            if trade.get("symbol") == symbol
        ]

        if order_id is not None:
            trades = [
                trade
                for trade in trades
                if trade.get("order_id") == order_id or trade.get("orderId") == order_id
            ]

        return trades[:limit]

    async def get_positions(
        self,
        *,
        symbol: str | None = None,
        recv_window: int | None = None,
    ) -> list[dict[str, Any]]:
        kwargs = {
            "symbol": symbol,
            "recv_window": recv_window,
        }
        self.calls.append(FakeBinanceCall("get_positions", kwargs))

        if self.get_positions_exception is not None:
            raise self.get_positions_exception

        if symbol is None:
            return list(self.positions)

        return [
            position
            for position in self.positions
            if position.get("symbol") == symbol
        ]

    def calls_for(self, method: str) -> list[FakeBinanceCall]:
        return [call for call in self.calls if call.method == method]

    @staticmethod
    def _normalized_order(
        *,
        symbol: str,
        order_id: int,
        client_order_id: str,
        side: str,
        order_type: str,
        status: str,
        orig_qty: float,
        executed_qty: float,
        price: float,
        avg_price: float,
        position_side: str | None = None,
        reduce_only: bool | None = None,
        close_position: bool | None = None,
        stop_price: float | None = None,
        working_type: str | None = None,
    ) -> dict[str, Any]:
        return {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": symbol,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "price": price,
            "avg_price": avg_price,
            "orig_qty": orig_qty,
            "executed_qty": executed_qty,
            "cum_qty": executed_qty,
            "cum_quote": executed_qty * avg_price if avg_price else 0.0,
            "cumulative_quote_qty": executed_qty * avg_price if avg_price else 0.0,
            "status": status,
            "time_in_force": "GTC",
            "type": order_type,
            "orig_type": order_type,
            "side": side,
            "position_side": position_side,
            "reduce_only": reduce_only,
            "close_position": close_position,
            "stop_price": stop_price,
            "working_type": working_type,
            "price_protect": None,
            "update_time": 1_700_000_000_000,
            "time": 1_700_000_000_000,
        }


# =============================================================================
# Common payload builders
# =============================================================================


def make_signal_confirmed_payload(
    *,
    symbol: str = "BTCUSDT",
    side: PositionSide = PositionSide.LONG,
    order_intent: OrderIntent = OrderIntent.OPEN,
    final_size: float = 0.01,
    final_leverage: float = 5.0,
    final_tier: TradeTier = TradeTier.T2,
    final_risk_amount: float = 50.0,
    final_margin: float = 100.0,
    final_notional: float = 500.0,
    entry_price: float = 50_000.0,
    stop_loss: float | None = 49_000.0,
    take_profit: float | None = 53_000.0,
    signal_id: str = "sig-1",
    strategy_name: str = "test_strategy",
    reservation_id: str = "res-1",
    reservation_expires_at: float | None = None,
    risk_mode: RiskMode = RiskMode.NORMAL,
    margin_mode: MarginMode = MarginMode.ISOLATED,
    reduce_only: bool | None = None,
    close_position: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": symbol,
        "side": side.value,
        "order_intent": order_intent.value,
        "final_size": final_size,
        "final_leverage": final_leverage,
        "final_tier": final_tier.value,
        "final_risk_amount": final_risk_amount,
        "final_margin": final_margin,
        "final_notional": final_notional,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal_id": signal_id,
        "strategy_name": strategy_name,
        "reservation_id": reservation_id,
        "reservation_expires_at": reservation_expires_at,
        "risk_mode": risk_mode.value,
        "margin_mode": margin_mode.value,
        "reduce_only": bool(order_intent.reduces_risk) if reduce_only is None else reduce_only,
        "close_position": close_position,
        "metadata": dict(metadata or {}),
    }


def make_order_filled_payload(
    *,
    symbol: str = "BTCUSDT",
    side: str = "BUY",
    position_side: str = "LONG",
    executed_qty: float = 0.01,
    avg_price: float = 50_000.0,
    orig_qty: float | None = None,
    order_id: str = "1001",
    client_order_id: str = "client-1001",
    execution_id: str = "exec-1",
    signal_id: str = "sig-1",
    strategy_name: str = "test_strategy",
    reservation_id: str = "res-1",
    reduce_only: bool = False,
    close_position: bool = False,
    stop_loss: float | None = 49_000.0,
    take_profit: float | None = 53_000.0,
    final_leverage: float = 5.0,
    final_margin: float = 100.0,
    final_notional: float = 500.0,
    final_risk_amount: float = 50.0,
    tier: str = "t2",
    status: str = "FILLED",
) -> dict[str, Any]:
    orig_qty = executed_qty if orig_qty is None else orig_qty

    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": symbol,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "execution_id": execution_id,
        "signal_id": signal_id,
        "strategy_name": strategy_name,
        "reservation_id": reservation_id,
        "status": status,
        "side": side,
        "order_type": "MARKET",
        "position_side": position_side,
        "price": 0.0,
        "avg_price": avg_price,
        "original_quantity": orig_qty,
        "orig_qty": orig_qty,
        "executed_quantity": executed_qty,
        "executed_qty": executed_qty,
        "cumulative_quote_quantity": executed_qty * avg_price,
        "cum_quote": executed_qty * avg_price,
        "fill_ratio": executed_qty / orig_qty if orig_qty else 0.0,
        "reduce_only": reduce_only,
        "close_position": close_position,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "final_leverage": final_leverage,
        "final_margin": final_margin,
        "final_notional": final_notional,
        "final_risk_amount": final_risk_amount,
        "tier": tier,
        "exchange_time": 1_700_000_000_000,
        "timestamp": 1_700_000_000_000,
        "metadata": {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "final_leverage": final_leverage,
            "final_margin": final_margin,
            "final_notional": final_notional,
            "final_risk_amount": final_risk_amount,
            "tier": tier,
        },
    }


def make_position_opened_payload(
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
        "tier": TradeTier.T2.value,
        "strategy_name": strategy_name,
        "signal_id": signal_id,
        "reservation_id": reservation_id,
        "position_id": position_id,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "metadata": {},
    }


def make_binance_position_snapshot(
    *,
    symbol: str = "BTCUSDT",
    position_side: str = "LONG",
    position_amt: float = 0.01,
    entry_price: float = 50_000.0,
    mark_price: float = 50_500.0,
    leverage: int = 5,
    unrealized_profit: float = 5.0,
) -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": symbol,
        "position_side": position_side,
        "position_amt": position_amt,
        "entry_price": entry_price,
        "break_even_price": entry_price,
        "mark_price": mark_price,
        "unrealized_profit": unrealized_profit,
        "liquidation_price": 40_000.0,
        "leverage": leverage,
        "max_notional_value": 1_000_000.0,
        "margin_type": "isolated",
        "isolated_margin": abs(position_amt * mark_price / leverage),
        "is_auto_add_margin": False,
        "update_time": 1_700_000_000_000,
        "notional": position_amt * mark_price,
        "isolated_wallet": abs(position_amt * mark_price / leverage),
    }


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture
def fake_scheduler() -> FakeScheduler:
    return FakeScheduler()


@pytest.fixture
def fake_binance_client() -> FakeBinanceRestClient:
    return FakeBinanceRestClient()


@pytest.fixture
def exchange_clients(fake_binance_client: FakeBinanceRestClient) -> dict[str, FakeBinanceRestClient]:
    return {"binance": fake_binance_client}


@pytest.fixture
def order_manager_config() -> OrderManagerConfig:
    config = OrderManagerConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        submit_retries=0,
        cancel_retries=0,
        reconcile_enabled=True,
        generate_client_order_id=True,
        emit_acknowledged_events=True,
        emit_partially_filled_events=True,
    )
    config.validate()
    return config


@pytest.fixture
def position_manager_config() -> PositionManagerConfig:
    config = PositionManagerConfig(
        default_exchange="binance",
        default_market_type="usdm_futures",
        reconcile_enabled=True,
        emit_unchanged_snapshots=False,
        emit_pnl_updates=True,
    )
    config.validate()
    return config


@pytest.fixture
def sltp_manager_config() -> SLTPManagerConfig:
    config = SLTPManagerConfig(
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
    config.validate()
    return config


@pytest.fixture
def smart_execution_config() -> SmartExecutionConfig:
    config = SmartExecutionConfig(
        enabled=True,
        default_mode=ExecutionMode.SMART,
        fallback_mode=ExecutionMode.MARKET,
        prefer_limit_for_entries=False,
        prefer_market_for_exits=True,
        allow_order_splitting=True,
        max_split_count=5,
        min_leg_notional=5.0,
        twap_enabled=True,
    )
    config.validate()
    return config


@pytest.fixture
def trade_executor_config() -> TradeExecutorConfig:
    config = TradeExecutorConfig(
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
        kill_switch_allows_reduce_only=True,
        max_concurrent_executions=10,
        per_symbol_execution_lock=True,
    )
    config.validate()
    return config


@pytest.fixture
def default_market_context() -> dict[str, Any]:
    return {
        "bid": 49_990.0,
        "ask": 50_000.0,
        "mark_price": 49_995.0,
        "last_price": 49_998.0,
        "tick_size": 0.1,
        "step_size": 0.001,
        "min_notional": 5.0,
        "available_depth_notional": 100_000.0,
        "expected_slippage_bps": 1.0,
    }


@pytest.fixture
def market_context_provider(default_market_context: dict[str, Any]):
    async def _provider(_: Any) -> dict[str, Any]:
        return dict(default_market_context)

    return _provider


@pytest.fixture
def emitted_topics(fake_event_bus: FakeEventBus):
    def _topics() -> list[str]:
        return fake_event_bus.topics()

    return _topics


@pytest.fixture
def emitted_payloads(fake_event_bus: FakeEventBus):
    def _payloads(topic: str) -> list[dict[str, Any]]:
        return fake_event_bus.payloads(topic)

    return _payloads


# =============================================================================
# Optional pytest markers / async config helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _clear_global_asyncio_noise() -> None:
    """
    Placeholder fixture for future cleanup hooks.

    Kept intentionally empty. It documents that tests should not rely on
    unmanaged background tasks.
    """
    return None