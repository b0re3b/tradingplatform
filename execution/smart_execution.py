from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from core.logger import get_logger
from execution.config import SmartExecutionConfig
from execution.enums import ExecutionMode, ExecutionStatus, OrderType, TimeInForce, TriggerType
from execution.exceptions import ExecutionPlanError, ExecutionPlanValidationError, SmartExecutionError
from execution.models import ExecutionIntent, ExecutionLeg, ExecutionPlan, SmartExecutionStats
from execution.utils import (
    calculate_spread_bps,
    now_ts,
    require_positive_number,
    round_price,
    round_quantity,
)
from risk.enums import OrderIntent


class SmartExecution:
    """
    Execution planning layer.

    Responsibilities:
    - convert risk-approved ExecutionIntent into ExecutionPlan;
    - choose technical order placement mode: market, limit, post-only, smart,
      liquidity-aware or TWAP;
    - optionally split large orders into several legs;
    - respect execution constraints such as max slippage, max spread, min leg
      notional and risk-approved size;
    - never approve risk;
    - never recalculate position size/leverage/tier;
    - never submit orders directly.

    SmartExecution output is ExecutionPlan. OrderManager is responsible for
    actual exchange order submission.
    """

    def __init__(
        self,
        config: SmartExecutionConfig,
        *,
        service_name: str = "execution.smart_execution",
    ) -> None:
        self._config = config
        self._config.validate()

        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="smart_execution",
        )

        self._service_name = service_name
        self._stats = SmartExecutionStats()

    @property
    def stats(self) -> SmartExecutionStats:
        return self._stats

    async def build_execution_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
        mode: ExecutionMode | None = None,
    ) -> ExecutionPlan:
        """
        Build execution plan from risk-approved ExecutionIntent.

        market_context is optional and may contain:
        - bid
        - ask
        - mark_price
        - last_price
        - tick_size
        - step_size
        - min_notional
        - available_depth_notional
        - preferred_limit_price
        """
        try:
            intent.validate()

            selected_mode = mode or self.choose_execution_mode(
                intent,
                market_context=market_context,
            )

            if selected_mode is ExecutionMode.MARKET:
                plan = self._build_market_plan(intent, market_context=market_context)

            elif selected_mode is ExecutionMode.LIMIT:
                plan = self._build_limit_plan(
                    intent,
                    market_context=market_context,
                    post_only=False,
                )

            elif selected_mode is ExecutionMode.POST_ONLY:
                plan = self._build_limit_plan(
                    intent,
                    market_context=market_context,
                    post_only=True,
                )

            elif selected_mode is ExecutionMode.TWAP:
                plan = self._build_twap_plan(intent, market_context=market_context)

            elif selected_mode is ExecutionMode.LIQUIDITY_AWARE:
                plan = self._build_liquidity_aware_plan(
                    intent,
                    market_context=market_context,
                )

            elif selected_mode is ExecutionMode.SMART:
                plan = self._build_smart_plan(intent, market_context=market_context)

            else:
                raise ExecutionPlanError(f"Unsupported execution mode: {selected_mode!r}")

            plan.validate()
            self._stats.register_plan(plan)

            self._logger.info(
                "Execution plan created | execution_id=%s symbol=%s mode=%s legs=%s qty=%s",
                intent.execution_id,
                intent.symbol,
                plan.mode.value,
                len(plan.legs),
                plan.total_quantity,
            )

            return plan

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stats.register_failure(str(exc))
            self._logger.exception(
                "Failed to build execution plan | execution_id=%s symbol=%s",
                intent.execution_id,
                intent.symbol,
            )
            raise SmartExecutionError(f"Failed to build execution plan: {exc}") from exc

    def choose_execution_mode(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> ExecutionMode:
        """
        Choose execution mode.

        This is not a risk decision. It only selects technical placement.
        """
        if not self._config.enabled:
            return self._config.fallback_mode

        if intent.reduces_risk and self._config.prefer_market_for_exits:
            return ExecutionMode.MARKET

        if intent.close_position:
            return ExecutionMode.MARKET

        context = dict(market_context or {})

        spread_bps = self.estimate_spread_bps(context)
        if spread_bps is not None and spread_bps > self._config.max_spread_bps:
            if self._config.prefer_limit_for_entries:
                return ExecutionMode.LIMIT
            return self._config.fallback_mode

        if self._config.default_mode is ExecutionMode.LIQUIDITY_AWARE:
            if self._has_sufficient_depth(intent, context):
                return ExecutionMode.LIQUIDITY_AWARE
            return self._config.fallback_mode

        if self._config.default_mode is ExecutionMode.TWAP:
            if self._config.twap_enabled and self._should_split_order(intent, context):
                return ExecutionMode.TWAP
            return self._config.fallback_mode

        if self._config.default_mode is ExecutionMode.SMART:
            return ExecutionMode.SMART

        return self._config.default_mode

    def estimate_spread_bps(
        self,
        market_context: Mapping[str, Any] | None,
    ) -> float | None:
        context = dict(market_context or {})

        bid = self._float_or_none(context.get("bid"))
        ask = self._float_or_none(context.get("ask"))

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None

        try:
            return calculate_spread_bps(bid=bid, ask=ask)
        except ValueError:
            return None

    def estimate_slippage_bps(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> float | None:
        """
        Conservative rough slippage estimate.

        If market_context already contains expected_slippage_bps, use it.
        Otherwise approximate from order notional vs available depth.
        """
        context = dict(market_context or {})

        explicit = self._float_or_none(context.get("expected_slippage_bps"))
        if explicit is not None:
            return max(0.0, explicit)

        available_depth = self._float_or_none(context.get("available_depth_notional"))
        if available_depth is None or available_depth <= 0:
            return None

        pressure = intent.final_notional / available_depth
        return max(0.0, pressure * 10.0)

    def should_use_market_order(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> bool:
        if intent.reduces_risk and self._config.prefer_market_for_exits:
            return True

        slippage_bps = self.estimate_slippage_bps(intent, market_context=market_context)
        if slippage_bps is not None and slippage_bps > self._config.max_slippage_bps:
            return False

        spread_bps = self.estimate_spread_bps(market_context)
        if spread_bps is not None and spread_bps > self._config.max_spread_bps:
            return False

        return self._config.default_mode is ExecutionMode.MARKET

    def should_use_post_only(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> bool:
        if intent.reduces_risk:
            return False

        if self._config.default_mode is ExecutionMode.POST_ONLY:
            return True

        spread_bps = self.estimate_spread_bps(market_context)
        return bool(
            spread_bps is not None
            and spread_bps <= self._config.max_spread_bps
            and self._config.prefer_limit_for_entries
        )

    def split_order(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
    ) -> list[float]:
        """
        Split final_size into quantities for multiple execution legs.

        Never increases total quantity above risk-approved final_size.
        """
        context = dict(market_context or {})
        quantity = require_positive_number(intent.final_size, "intent.final_size")

        split_count = self._resolve_split_count(intent, context)

        if split_count <= 1:
            return [quantity]

        step_size = self._float_or_none(context.get("step_size"))

        base_qty = quantity / split_count
        quantities: list[float] = []

        remaining = quantity

        for index in range(split_count):
            if index == split_count - 1:
                leg_qty = remaining
            else:
                leg_qty = round_quantity(base_qty, step_size)
                remaining -= leg_qty

            if leg_qty <= 0:
                continue

            quantities.append(leg_qty)

        total = sum(quantities)

        if total > quantity:
            scale = quantity / total
            quantities = [qty * scale for qty in quantities]

        return [qty for qty in quantities if qty > 0]

    def choose_limit_price(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None = None,
        post_only: bool = False,
    ) -> float:
        """
        Choose limit price from context.

        Uses preferred_limit_price if provided. Otherwise derives from bid/ask.
        """
        context = dict(market_context or {})

        preferred = self._float_or_none(context.get("preferred_limit_price"))
        tick_size = self._float_or_none(context.get("tick_size"))

        if preferred is not None and preferred > 0:
            return round_price(preferred, tick_size)

        bid = self._float_or_none(context.get("bid"))
        ask = self._float_or_none(context.get("ask"))
        mark_price = self._float_or_none(context.get("mark_price"))
        last_price = self._float_or_none(context.get("last_price"))
        reference = mark_price or last_price or intent.entry_price

        if bid is None or ask is None:
            if reference is None:
                raise ExecutionPlanError("Cannot choose limit price without bid/ask or reference price")
            return round_price(reference, tick_size)

        offset_bps = (
            self._config.post_only_price_offset_bps
            if post_only
            else self._config.limit_price_offset_bps
        )

        offset = offset_bps / 10_000.0

        if intent.order_side.is_buy:
            if post_only:
                price = bid * (1.0 - offset)
            else:
                price = ask * (1.0 - offset)
        else:
            if post_only:
                price = ask * (1.0 + offset)
            else:
                price = bid * (1.0 + offset)

        return round_price(price, tick_size)

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self._service_name,
            "enabled": self._config.enabled,
            "default_mode": self._config.default_mode.value,
            "fallback_mode": self._config.fallback_mode.value,
            "stats": self._stats.snapshot(),
        }

    # ------------------------------------------------------------------
    # Plan builders
    # ------------------------------------------------------------------

    def _build_market_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
    ) -> ExecutionPlan:
        quantity = self._rounded_quantity(intent.final_size, market_context)

        leg = ExecutionLeg(
            symbol=intent.symbol,
            side=intent.order_side,
            order_type=OrderType.MARKET,
            quantity=quantity if not intent.close_position else None,
            position_side=intent.side,
            reduce_only=intent.is_reduce_only and not intent.close_position,
            close_position=intent.close_position,
            trigger_type=(
                TriggerType.RISK_CLOSE
                if intent.order_intent is OrderIntent.CLOSE
                else TriggerType.RISK_REDUCE
                if intent.reduces_risk
                else TriggerType.NONE
            ),
            metadata={
                "execution_mode": ExecutionMode.MARKET.value,
                "source": "smart_execution",
            },
        )

        return ExecutionPlan(
            intent=intent,
            mode=ExecutionMode.MARKET,
            legs=[leg],
            status=ExecutionStatus.PLANNED,
            metadata=self._plan_metadata(intent, market_context, mode=ExecutionMode.MARKET),
        )

    def _build_limit_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
        post_only: bool,
    ) -> ExecutionPlan:
        quantity = self._rounded_quantity(intent.final_size, market_context)
        price = self.choose_limit_price(intent, market_context=market_context, post_only=post_only)

        leg = ExecutionLeg(
            symbol=intent.symbol,
            side=intent.order_side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            position_side=intent.side,
            time_in_force=TimeInForce.GTX if post_only else TimeInForce.GTC,
            reduce_only=intent.is_reduce_only,
            close_position=False,
            trigger_type=(
                TriggerType.RISK_CLOSE
                if intent.order_intent is OrderIntent.CLOSE
                else TriggerType.RISK_REDUCE
                if intent.reduces_risk
                else TriggerType.NONE
            ),
            metadata={
                "execution_mode": ExecutionMode.POST_ONLY.value if post_only else ExecutionMode.LIMIT.value,
                "source": "smart_execution",
                "post_only": post_only,
            },
        )

        mode = ExecutionMode.POST_ONLY if post_only else ExecutionMode.LIMIT

        return ExecutionPlan(
            intent=intent,
            mode=mode,
            legs=[leg],
            status=ExecutionStatus.PLANNED,
            metadata=self._plan_metadata(intent, market_context, mode=mode),
        )

    def _build_smart_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
    ) -> ExecutionPlan:
        if intent.reduces_risk and self._config.prefer_market_for_exits:
            return self._build_market_plan(intent, market_context=market_context)

        if self.should_use_post_only(intent, market_context=market_context):
            return self._build_limit_plan(
                intent,
                market_context=market_context,
                post_only=True,
            )

        slippage_bps = self.estimate_slippage_bps(intent, market_context=market_context)
        spread_bps = self.estimate_spread_bps(market_context)

        if (
            slippage_bps is not None
            and slippage_bps <= self._config.max_slippage_bps
            and (spread_bps is None or spread_bps <= self._config.max_spread_bps)
            and not self._config.prefer_limit_for_entries
        ):
            return self._build_market_plan(intent, market_context=market_context)

        if self._config.allow_order_splitting and self._should_split_order(intent, dict(market_context or {})):
            return self._build_liquidity_aware_plan(intent, market_context=market_context)

        return self._build_limit_plan(
            intent,
            market_context=market_context,
            post_only=False,
        )

    def _build_liquidity_aware_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
    ) -> ExecutionPlan:
        if not self._config.liquidity_aware_enabled:
            return self._build_market_plan(intent, market_context=market_context)

        quantities = self.split_order(intent, market_context=market_context)
        context = dict(market_context or {})
        tick_size = self._float_or_none(context.get("tick_size"))

        legs: list[ExecutionLeg] = []

        for index, quantity in enumerate(quantities):
            if intent.reduces_risk:
                order_type = OrderType.MARKET
                price = None
                time_in_force = None
            else:
                order_type = OrderType.LIMIT
                price = self.choose_limit_price(
                    intent,
                    market_context=market_context,
                    post_only=False,
                )
                if price is not None:
                    price = round_price(price, tick_size)
                time_in_force = TimeInForce.GTC

            legs.append(
                ExecutionLeg(
                    symbol=intent.symbol,
                    side=intent.order_side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    position_side=intent.side,
                    time_in_force=time_in_force,
                    reduce_only=intent.is_reduce_only,
                    close_position=False,
                    trigger_type=(
                        TriggerType.RISK_REDUCE if intent.reduces_risk else TriggerType.NONE
                    ),
                    sequence=index,
                    metadata={
                        "execution_mode": ExecutionMode.LIQUIDITY_AWARE.value,
                        "source": "smart_execution",
                        "split_index": index,
                        "split_count": len(quantities),
                    },
                )
            )

        return ExecutionPlan(
            intent=intent,
            mode=ExecutionMode.LIQUIDITY_AWARE,
            legs=legs,
            status=ExecutionStatus.PLANNED,
            metadata=self._plan_metadata(intent, market_context, mode=ExecutionMode.LIQUIDITY_AWARE),
        )

    def _build_twap_plan(
        self,
        intent: ExecutionIntent,
        *,
        market_context: Mapping[str, Any] | None,
    ) -> ExecutionPlan:
        if not self._config.twap_enabled:
            return self._build_smart_plan(intent, market_context=market_context)

        quantities = self.split_order(intent, market_context=market_context)

        legs: list[ExecutionLeg] = []
        interval = self._config.twap_slice_interval_seconds

        for index, quantity in enumerate(quantities):
            legs.append(
                ExecutionLeg(
                    symbol=intent.symbol,
                    side=intent.order_side,
                    order_type=OrderType.MARKET if intent.reduces_risk else OrderType.LIMIT,
                    quantity=quantity,
                    price=(
                        None
                        if intent.reduces_risk
                        else self.choose_limit_price(intent, market_context=market_context)
                    ),
                    position_side=intent.side,
                    time_in_force=None if intent.reduces_risk else TimeInForce.GTC,
                    reduce_only=intent.is_reduce_only,
                    close_position=False,
                    trigger_type=TriggerType.RISK_REDUCE if intent.reduces_risk else TriggerType.NONE,
                    sequence=index,
                    metadata={
                        "execution_mode": ExecutionMode.TWAP.value,
                        "source": "smart_execution",
                        "split_index": index,
                        "split_count": len(quantities),
                        "delay_seconds": index * interval,
                    },
                )
            )

        return ExecutionPlan(
            intent=intent,
            mode=ExecutionMode.TWAP,
            legs=legs,
            status=ExecutionStatus.PLANNED,
            metadata=self._plan_metadata(intent, market_context, mode=ExecutionMode.TWAP),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rounded_quantity(
        self,
        quantity: float,
        market_context: Mapping[str, Any] | None,
    ) -> float:
        context = dict(market_context or {})
        step_size = self._float_or_none(context.get("step_size"))
        rounded = round_quantity(quantity, step_size)

        if rounded <= 0:
            raise ExecutionPlanValidationError("Rounded quantity must be > 0")

        return rounded

    def _resolve_split_count(
        self,
        intent: ExecutionIntent,
        context: Mapping[str, Any],
    ) -> int:
        if not self._config.allow_order_splitting:
            return 1

        if intent.close_position:
            return 1

        if intent.final_notional <= 0:
            return 1

        available_depth = self._float_or_none(context.get("available_depth_notional"))

        if available_depth is None or available_depth <= 0:
            if intent.final_notional <= self._config.min_leg_notional * 2:
                return 1
            return min(2, self._config.max_split_count)

        target_leg_notional = max(
            self._config.min_leg_notional,
            available_depth / self._config.min_depth_notional_multiplier,
        )

        split_count = math.ceil(intent.final_notional / target_leg_notional)
        split_count = max(self._config.min_split_count, split_count)
        split_count = min(split_count, self._config.max_split_count)

        if split_count <= 1:
            return 1

        leg_notional = intent.final_notional / split_count
        if leg_notional < self._config.min_leg_notional:
            split_count = max(1, math.floor(intent.final_notional / self._config.min_leg_notional))

        return max(1, min(split_count, self._config.max_split_count))

    def _should_split_order(
        self,
        intent: ExecutionIntent,
        context: Mapping[str, Any],
    ) -> bool:
        if not self._config.allow_order_splitting:
            return False

        if intent.close_position:
            return False

        if intent.final_notional < self._config.min_leg_notional * 2:
            return False

        split_count = self._resolve_split_count(intent, context)
        return split_count > 1

    def _has_sufficient_depth(
        self,
        intent: ExecutionIntent,
        context: Mapping[str, Any],
    ) -> bool:
        available_depth = self._float_or_none(context.get("available_depth_notional"))

        if available_depth is None or available_depth <= 0:
            return False

        required_depth = intent.final_notional * self._config.min_depth_notional_multiplier
        return available_depth >= required_depth

    def _plan_metadata(
        self,
        intent: ExecutionIntent,
        market_context: Mapping[str, Any] | None,
        *,
        mode: ExecutionMode,
    ) -> dict[str, Any]:
        context = dict(market_context or {})

        spread_bps = self.estimate_spread_bps(context)
        slippage_bps = self.estimate_slippage_bps(intent, market_context=context)

        return {
            "service": self._service_name,
            "mode": mode.value,
            "created_at": now_ts(),
            "risk_approved": True,
            "risk_mode": intent.risk_mode.value,
            "final_size": intent.final_size,
            "final_leverage": intent.final_leverage,
            "final_notional": intent.final_notional,
            "final_margin": intent.final_margin,
            "final_risk_amount": intent.final_risk_amount,
            "reservation_id": intent.reservation_id,
            "estimated_spread_bps": spread_bps,
            "estimated_slippage_bps": slippage_bps,
            "max_slippage_bps": self._config.max_slippage_bps,
            "max_spread_bps": self._config.max_spread_bps,
            "market_context_keys": sorted(context.keys()),
        }

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None or value == "":
            return None

        try:
            value_f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(value_f):
            return None

        return value_f


__all__ = [
    "SmartExecution",
]