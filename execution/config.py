from __future__ import annotations

from dataclasses import dataclass, field

from execution.enums import ExecutionMode, TimeInForce, WorkingType
from execution.exceptions import ExecutionConfigurationError


def _require_positive_float(value: float | int, field_name: str) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionConfigurationError(f"{field_name} must be a number") from exc

    if value_f <= 0:
        raise ExecutionConfigurationError(f"{field_name} must be > 0")

    return value_f


def _require_non_negative_float(value: float | int, field_name: str) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionConfigurationError(f"{field_name} must be a number") from exc

    if value_f < 0:
        raise ExecutionConfigurationError(f"{field_name} must be >= 0")

    return value_f


def _require_positive_int(value: int, field_name: str) -> int:
    try:
        value_i = int(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionConfigurationError(f"{field_name} must be an integer") from exc

    if value_i <= 0:
        raise ExecutionConfigurationError(f"{field_name} must be > 0")

    return value_i


@dataclass(slots=True)
class OrderManagerConfig:
    """
    Config for execution.order_manager.OrderManager.

    OrderManager is the Binance REST order bridge:
    - submit/cancel orders;
    - normalize exchange order responses;
    - reconcile open orders;
    - emit execution.order_* events required by RiskManager.
    """

    enabled: bool = True

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"

    default_time_in_force: TimeInForce = TimeInForce.GTC
    post_only_time_in_force: TimeInForce = TimeInForce.GTX

    new_order_response_type: str = "RESULT"

    submit_timeout_seconds: float = 10.0
    cancel_timeout_seconds: float = 10.0
    fetch_timeout_seconds: float = 10.0

    submit_retries: int = 2
    cancel_retries: int = 2
    retry_delay_seconds: float = 0.25

    reconcile_enabled: bool = True
    reconcile_interval_seconds: float = 15.0
    open_order_sync_interval_seconds: float = 20.0

    emit_acknowledged_events: bool = True
    emit_partially_filled_events: bool = True

    generate_client_order_id: bool = True
    client_order_id_prefix: str = "ts"
    max_client_order_id_length: int = 36

    allow_market_orders: bool = True
    allow_limit_orders: bool = True
    allow_stop_orders: bool = True
    allow_trailing_stop_orders: bool = True

    require_reduce_only_for_close: bool = True
    require_reduce_only_for_protective_orders: bool = True

    fail_on_unknown_order_status: bool = False

    def validate(self) -> None:
        if not self.default_exchange:
            raise ExecutionConfigurationError("order_manager.default_exchange is required")

        if not self.default_market_type:
            raise ExecutionConfigurationError("order_manager.default_market_type is required")

        self.submit_timeout_seconds = _require_positive_float(
            self.submit_timeout_seconds,
            "order_manager.submit_timeout_seconds",
        )
        self.cancel_timeout_seconds = _require_positive_float(
            self.cancel_timeout_seconds,
            "order_manager.cancel_timeout_seconds",
        )
        self.fetch_timeout_seconds = _require_positive_float(
            self.fetch_timeout_seconds,
            "order_manager.fetch_timeout_seconds",
        )
        self.submit_retries = _require_positive_int(
            self.submit_retries + 1,
            "order_manager.submit_retries_plus_one",
        ) - 1
        self.cancel_retries = _require_positive_int(
            self.cancel_retries + 1,
            "order_manager.cancel_retries_plus_one",
        ) - 1
        self.retry_delay_seconds = _require_non_negative_float(
            self.retry_delay_seconds,
            "order_manager.retry_delay_seconds",
        )
        self.reconcile_interval_seconds = _require_positive_float(
            self.reconcile_interval_seconds,
            "order_manager.reconcile_interval_seconds",
        )
        self.open_order_sync_interval_seconds = _require_positive_float(
            self.open_order_sync_interval_seconds,
            "order_manager.open_order_sync_interval_seconds",
        )
        self.max_client_order_id_length = _require_positive_int(
            self.max_client_order_id_length,
            "order_manager.max_client_order_id_length",
        )

        if self.max_client_order_id_length > 36:
            raise ExecutionConfigurationError(
                "order_manager.max_client_order_id_length must be <= 36 for Binance client order ids"
            )

        if not self.client_order_id_prefix:
            raise ExecutionConfigurationError("order_manager.client_order_id_prefix is required")

        if not self.new_order_response_type:
            raise ExecutionConfigurationError("order_manager.new_order_response_type is required")


@dataclass(slots=True)
class PositionManagerConfig:
    """
    Config for execution.position_manager.PositionManager.

    PositionManager owns local execution-side position state and emits
    position.opened / position.updated / position.closed events for RiskManager.
    """

    enabled: bool = True

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"

    reconcile_enabled: bool = True
    reconcile_interval_seconds: float = 20.0
    position_sync_interval_seconds: float = 20.0

    stale_position_seconds: float = 60.0

    emit_unchanged_snapshots: bool = False
    emit_pnl_updates: bool = True

    min_position_size_epsilon: float = 1e-12
    min_notional_epsilon: float = 1e-9

    require_signal_metadata_for_open: bool = False
    fail_on_unknown_position_side: bool = False

    def validate(self) -> None:
        if not self.default_exchange:
            raise ExecutionConfigurationError("position_manager.default_exchange is required")

        if not self.default_market_type:
            raise ExecutionConfigurationError("position_manager.default_market_type is required")

        self.reconcile_interval_seconds = _require_positive_float(
            self.reconcile_interval_seconds,
            "position_manager.reconcile_interval_seconds",
        )
        self.position_sync_interval_seconds = _require_positive_float(
            self.position_sync_interval_seconds,
            "position_manager.position_sync_interval_seconds",
        )
        self.stale_position_seconds = _require_positive_float(
            self.stale_position_seconds,
            "position_manager.stale_position_seconds",
        )
        self.min_position_size_epsilon = _require_positive_float(
            self.min_position_size_epsilon,
            "position_manager.min_position_size_epsilon",
        )
        self.min_notional_epsilon = _require_positive_float(
            self.min_notional_epsilon,
            "position_manager.min_notional_epsilon",
        )


@dataclass(slots=True)
class SLTPManagerConfig:
    """
    Config for execution.sl_tp_manager.SLTPManager.

    SLTPManager manages protective reduce-only / close-position orders through
    OrderManager. It should not call BinanceRestClient directly.
    """

    enabled: bool = True

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"

    auto_place_on_position_opened: bool = True
    auto_cancel_on_position_closed: bool = True
    auto_resize_on_position_updated: bool = True

    use_close_position_for_full_stop: bool = True
    use_close_position_for_full_take_profit: bool = False

    require_reduce_only: bool = True

    default_working_type: WorkingType = WorkingType.MARK_PRICE
    price_protect: bool = True

    stop_loss_order_timeout_seconds: float = 10.0
    take_profit_order_timeout_seconds: float = 10.0
    cancel_timeout_seconds: float = 10.0

    protective_order_retries: int = 2
    retry_delay_seconds: float = 0.25

    reconcile_enabled: bool = True
    reconcile_interval_seconds: float = 30.0

    trailing_stop_enabled: bool = True
    min_trailing_callback_rate: float = 0.1
    max_trailing_callback_rate: float = 5.0

    breakeven_enabled: bool = True
    breakeven_after_r_multiple: float = 1.0

    def validate(self) -> None:
        if not self.default_exchange:
            raise ExecutionConfigurationError("sltp_manager.default_exchange is required")

        if not self.default_market_type:
            raise ExecutionConfigurationError("sltp_manager.default_market_type is required")

        self.stop_loss_order_timeout_seconds = _require_positive_float(
            self.stop_loss_order_timeout_seconds,
            "sltp_manager.stop_loss_order_timeout_seconds",
        )
        self.take_profit_order_timeout_seconds = _require_positive_float(
            self.take_profit_order_timeout_seconds,
            "sltp_manager.take_profit_order_timeout_seconds",
        )
        self.cancel_timeout_seconds = _require_positive_float(
            self.cancel_timeout_seconds,
            "sltp_manager.cancel_timeout_seconds",
        )
        self.protective_order_retries = _require_positive_int(
            self.protective_order_retries + 1,
            "sltp_manager.protective_order_retries_plus_one",
        ) - 1
        self.retry_delay_seconds = _require_non_negative_float(
            self.retry_delay_seconds,
            "sltp_manager.retry_delay_seconds",
        )
        self.reconcile_interval_seconds = _require_positive_float(
            self.reconcile_interval_seconds,
            "sltp_manager.reconcile_interval_seconds",
        )
        self.min_trailing_callback_rate = _require_positive_float(
            self.min_trailing_callback_rate,
            "sltp_manager.min_trailing_callback_rate",
        )
        self.max_trailing_callback_rate = _require_positive_float(
            self.max_trailing_callback_rate,
            "sltp_manager.max_trailing_callback_rate",
        )
        self.breakeven_after_r_multiple = _require_positive_float(
            self.breakeven_after_r_multiple,
            "sltp_manager.breakeven_after_r_multiple",
        )

        if self.min_trailing_callback_rate > self.max_trailing_callback_rate:
            raise ExecutionConfigurationError(
                "sltp_manager.min_trailing_callback_rate must be <= max_trailing_callback_rate"
            )


@dataclass(slots=True)
class SmartExecutionConfig:
    """
    Config for execution.smart_execution.SmartExecution.

    SmartExecution builds ExecutionPlan from risk-approved ExecutionIntent.
    It does not decide whether risk is allowed and does not submit orders.
    """

    enabled: bool = True

    default_mode: ExecutionMode = ExecutionMode.SMART
    fallback_mode: ExecutionMode = ExecutionMode.MARKET

    prefer_limit_for_entries: bool = False
    prefer_market_for_exits: bool = True

    max_slippage_bps: float = 10.0
    max_spread_bps: float = 8.0
    max_price_deviation_bps: float = 15.0

    allow_order_splitting: bool = True
    min_split_count: int = 1
    max_split_count: int = 5
    min_leg_notional: float = 5.0

    twap_enabled: bool = False
    twap_duration_seconds: float = 60.0
    twap_slice_interval_seconds: float = 5.0

    liquidity_aware_enabled: bool = True
    orderbook_depth_levels: int = 20
    min_depth_notional_multiplier: float = 3.0

    limit_price_offset_bps: float = 1.0
    post_only_price_offset_bps: float = 1.0

    plan_timeout_seconds: float = 5.0

    def validate(self) -> None:
        self.max_slippage_bps = _require_non_negative_float(
            self.max_slippage_bps,
            "smart_execution.max_slippage_bps",
        )
        self.max_spread_bps = _require_non_negative_float(
            self.max_spread_bps,
            "smart_execution.max_spread_bps",
        )
        self.max_price_deviation_bps = _require_non_negative_float(
            self.max_price_deviation_bps,
            "smart_execution.max_price_deviation_bps",
        )
        self.min_split_count = _require_positive_int(
            self.min_split_count,
            "smart_execution.min_split_count",
        )
        self.max_split_count = _require_positive_int(
            self.max_split_count,
            "smart_execution.max_split_count",
        )
        self.min_leg_notional = _require_positive_float(
            self.min_leg_notional,
            "smart_execution.min_leg_notional",
        )
        self.twap_duration_seconds = _require_positive_float(
            self.twap_duration_seconds,
            "smart_execution.twap_duration_seconds",
        )
        self.twap_slice_interval_seconds = _require_positive_float(
            self.twap_slice_interval_seconds,
            "smart_execution.twap_slice_interval_seconds",
        )
        self.orderbook_depth_levels = _require_positive_int(
            self.orderbook_depth_levels,
            "smart_execution.orderbook_depth_levels",
        )
        self.min_depth_notional_multiplier = _require_positive_float(
            self.min_depth_notional_multiplier,
            "smart_execution.min_depth_notional_multiplier",
        )
        self.limit_price_offset_bps = _require_non_negative_float(
            self.limit_price_offset_bps,
            "smart_execution.limit_price_offset_bps",
        )
        self.post_only_price_offset_bps = _require_non_negative_float(
            self.post_only_price_offset_bps,
            "smart_execution.post_only_price_offset_bps",
        )
        self.plan_timeout_seconds = _require_positive_float(
            self.plan_timeout_seconds,
            "smart_execution.plan_timeout_seconds",
        )

        if self.min_split_count > self.max_split_count:
            raise ExecutionConfigurationError(
                "smart_execution.min_split_count must be <= max_split_count"
            )


@dataclass(slots=True)
class TradeExecutorConfig:
    """
    Config for execution.trade_executor.TradeExecutor.

    TradeExecutor is the final orchestrator:
    - listens to signal.confirmed;
    - handles risk.position_close_requested / risk.position_reduce_requested;
    - handles risk.kill_switch;
    - coordinates SmartExecution, OrderManager, PositionManager and SLTPManager.
    """

    enabled: bool = True

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"

    auto_subscribe: bool = True
    register_scheduler_jobs: bool = True

    allow_new_entries: bool = True
    allow_position_reductions: bool = True
    allow_position_closes: bool = True

    reject_expired_risk_reservations: bool = True
    reservation_grace_seconds: float = 2.0

    kill_switch_blocks_new_entries: bool = True
    kill_switch_cancels_open_orders: bool = True
    kill_switch_allows_reduce_only: bool = True

    execution_timeout_seconds: float = 30.0
    close_timeout_seconds: float = 30.0
    reduce_timeout_seconds: float = 30.0

    max_concurrent_executions: int = 10
    per_symbol_execution_lock: bool = True

    emit_lifecycle_events: bool = True
    emit_rejected_events: bool = True

    def validate(self) -> None:
        if not self.default_exchange:
            raise ExecutionConfigurationError("trade_executor.default_exchange is required")

        if not self.default_market_type:
            raise ExecutionConfigurationError("trade_executor.default_market_type is required")

        self.reservation_grace_seconds = _require_non_negative_float(
            self.reservation_grace_seconds,
            "trade_executor.reservation_grace_seconds",
        )
        self.execution_timeout_seconds = _require_positive_float(
            self.execution_timeout_seconds,
            "trade_executor.execution_timeout_seconds",
        )
        self.close_timeout_seconds = _require_positive_float(
            self.close_timeout_seconds,
            "trade_executor.close_timeout_seconds",
        )
        self.reduce_timeout_seconds = _require_positive_float(
            self.reduce_timeout_seconds,
            "trade_executor.reduce_timeout_seconds",
        )
        self.max_concurrent_executions = _require_positive_int(
            self.max_concurrent_executions,
            "trade_executor.max_concurrent_executions",
        )


@dataclass(slots=True)
class ExecutionConfig:
    """
    Root config for the execution package.

    Binance USD-M Futures is the first-class execution target. The package is
    still structured through config/defaults so a future exchange adapter can be
    injected without rewriting the execution domain models.
    """

    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"

    service_name: str = "execution"

    trade_executor: TradeExecutorConfig = field(default_factory=TradeExecutorConfig)
    order_manager: OrderManagerConfig = field(default_factory=OrderManagerConfig)
    position_manager: PositionManagerConfig = field(default_factory=PositionManagerConfig)
    sltp_manager: SLTPManagerConfig = field(default_factory=SLTPManagerConfig)
    smart_execution: SmartExecutionConfig = field(default_factory=SmartExecutionConfig)

    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.default_exchange:
            raise ExecutionConfigurationError("execution.default_exchange is required")

        if not self.default_market_type:
            raise ExecutionConfigurationError("execution.default_market_type is required")

        if not self.service_name:
            raise ExecutionConfigurationError("execution.service_name is required")

        if self.default_exchange != "binance":
            raise ExecutionConfigurationError(
                "execution.default_exchange must be 'binance' for the current Binance-first implementation"
            )

        if self.default_market_type != "usdm_futures":
            raise ExecutionConfigurationError(
                "execution.default_market_type must be 'usdm_futures' for Binance USD-M Futures execution"
            )

        self._sync_child_defaults()

        self.trade_executor.validate()
        self.order_manager.validate()
        self.position_manager.validate()
        self.sltp_manager.validate()
        self.smart_execution.validate()

    def _sync_child_defaults(self) -> None:
        """
        Keep child configs aligned with the root execution target.
        """
        self.trade_executor.default_exchange = self.default_exchange
        self.trade_executor.default_market_type = self.default_market_type

        self.order_manager.default_exchange = self.default_exchange
        self.order_manager.default_market_type = self.default_market_type

        self.position_manager.default_exchange = self.default_exchange
        self.position_manager.default_market_type = self.default_market_type

        self.sltp_manager.default_exchange = self.default_exchange
        self.sltp_manager.default_market_type = self.default_market_type


__all__ = [
    "ExecutionConfig",
    "TradeExecutorConfig",
    "OrderManagerConfig",
    "PositionManagerConfig",
    "SLTPManagerConfig",
    "SmartExecutionConfig",
]