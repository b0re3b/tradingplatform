from __future__ import annotations

import pytest

from execution.config import SmartExecutionConfig
from execution.enums import ExecutionMode, OrderSide, OrderType, TimeInForce, TriggerType
from execution.exceptions import SmartExecutionError
from execution.models import ExecutionIntent
from execution.smart_execution import SmartExecution

from risk.enums import MarginMode, OrderIntent, PositionSide, RiskMode, TradeTier


pytestmark = pytest.mark.asyncio


# =============================================================================
# Helpers
# =============================================================================


def make_intent(
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
    entry_price: float | None = 50_000.0,
    stop_loss: float | None = 49_000.0,
    take_profit: float | None = 53_000.0,
    signal_id: str = "sig-1",
    strategy_name: str = "test_strategy",
    reservation_id: str = "res-1",
    risk_mode: RiskMode = RiskMode.NORMAL,
    margin_mode: MarginMode = MarginMode.ISOLATED,
    reduce_only: bool | None = None,
    close_position: bool = False,
    metadata: dict | None = None,
) -> ExecutionIntent:
    return ExecutionIntent(
        exchange="binance",
        market_type="usdm_futures",
        symbol=symbol,
        side=side,
        order_intent=order_intent,
        final_size=final_size,
        final_leverage=final_leverage,
        final_tier=final_tier,
        final_risk_amount=final_risk_amount,
        final_margin=final_margin,
        final_notional=final_notional,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_id=signal_id,
        strategy_name=strategy_name,
        reservation_id=reservation_id,
        risk_mode=risk_mode,
        margin_mode=margin_mode,
        reduce_only=bool(order_intent.reduces_risk) if reduce_only is None else reduce_only,
        close_position=close_position,
        metadata=dict(metadata or {}),
    )


def make_context(
    *,
    bid: float | None = 49_990.0,
    ask: float | None = 50_000.0,
    mark_price: float | None = 49_995.0,
    last_price: float | None = 49_998.0,
    tick_size: float | None = 0.1,
    step_size: float | None = 0.001,
    min_notional: float | None = 5.0,
    available_depth_notional: float | None = 100_000.0,
    expected_slippage_bps: float | None = 1.0,
    preferred_limit_price: float | None = None,
) -> dict:
    context = {
        "bid": bid,
        "ask": ask,
        "mark_price": mark_price,
        "last_price": last_price,
        "tick_size": tick_size,
        "step_size": step_size,
        "min_notional": min_notional,
        "available_depth_notional": available_depth_notional,
        "expected_slippage_bps": expected_slippage_bps,
        "preferred_limit_price": preferred_limit_price,
    }
    return {key: value for key, value in context.items() if value is not None}


def make_smart_execution(
    *,
    default_mode: ExecutionMode = ExecutionMode.SMART,
    fallback_mode: ExecutionMode = ExecutionMode.MARKET,
    prefer_limit_for_entries: bool = False,
    prefer_market_for_exits: bool = True,
    allow_order_splitting: bool = True,
    min_split_count: int = 1,
    max_split_count: int = 5,
    min_leg_notional: float = 5.0,
    twap_enabled: bool = True,
    liquidity_aware_enabled: bool = True,
    max_slippage_bps: float = 10.0,
    max_spread_bps: float = 8.0,
) -> SmartExecution:
    config = SmartExecutionConfig(
        enabled=True,
        default_mode=default_mode,
        fallback_mode=fallback_mode,
        prefer_limit_for_entries=prefer_limit_for_entries,
        prefer_market_for_exits=prefer_market_for_exits,
        allow_order_splitting=allow_order_splitting,
        min_split_count=min_split_count,
        max_split_count=max_split_count,
        min_leg_notional=min_leg_notional,
        twap_enabled=twap_enabled,
        liquidity_aware_enabled=liquidity_aware_enabled,
        max_slippage_bps=max_slippage_bps,
        max_spread_bps=max_spread_bps,
    )
    config.validate()
    return SmartExecution(config)


# =============================================================================
# Basic MARKET / side mapping
# =============================================================================


async def test_market_open_long_plan_builds_single_buy_market_leg():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)
    intent = make_intent(
        side=PositionSide.LONG,
        order_intent=OrderIntent.OPEN,
        final_size=0.01,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=make_context(),
        mode=ExecutionMode.MARKET,
    )

    assert plan.mode is ExecutionMode.MARKET
    assert len(plan.legs) == 1
    assert plan.total_quantity == pytest.approx(0.01)

    leg = plan.legs[0]
    assert leg.symbol == "BTCUSDT"
    assert leg.side is OrderSide.BUY
    assert leg.order_type is OrderType.MARKET
    assert leg.quantity == pytest.approx(0.01)
    assert leg.position_side is PositionSide.LONG
    assert leg.reduce_only is False
    assert leg.close_position is False
    assert leg.trigger_type is TriggerType.NONE

    assert plan.metadata["risk_approved"] is True
    assert plan.metadata["final_size"] == pytest.approx(0.01)
    assert plan.metadata["final_leverage"] == pytest.approx(5.0)
    assert plan.metadata["reservation_id"] == "res-1"

    assert smart.stats.plans_created == 1
    assert smart.stats.market_plans_created == 1


async def test_market_open_short_plan_builds_single_sell_market_leg():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)
    intent = make_intent(
        side=PositionSide.SHORT,
        order_intent=OrderIntent.OPEN,
        final_size=0.02,
        final_notional=1_000.0,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=make_context(),
        mode=ExecutionMode.MARKET,
    )

    leg = plan.legs[0]

    assert leg.side is OrderSide.SELL
    assert leg.order_type is OrderType.MARKET
    assert leg.quantity == pytest.approx(0.02)
    assert leg.position_side is PositionSide.SHORT
    assert leg.reduce_only is False


async def test_reduce_long_plan_uses_sell_market_reduce_only_when_prefer_market_for_exits_true():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_market_for_exits=True,
    )
    intent = make_intent(
        side=PositionSide.LONG,
        order_intent=OrderIntent.REDUCE,
        final_size=0.005,
        final_notional=250.0,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    plan = await smart.build_execution_plan(intent, market_context=make_context())

    assert plan.mode is ExecutionMode.MARKET
    assert len(plan.legs) == 1

    leg = plan.legs[0]
    assert leg.side is OrderSide.SELL
    assert leg.order_type is OrderType.MARKET
    assert leg.reduce_only is True
    assert leg.close_position is False
    assert leg.trigger_type is TriggerType.RISK_REDUCE


async def test_close_short_plan_uses_buy_market_reduce_only():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_market_for_exits=True,
    )
    intent = make_intent(
        side=PositionSide.SHORT,
        order_intent=OrderIntent.CLOSE,
        final_size=0.02,
        final_notional=1_000.0,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    plan = await smart.build_execution_plan(intent, market_context=make_context())

    assert plan.mode is ExecutionMode.MARKET

    leg = plan.legs[0]
    assert leg.side is OrderSide.BUY
    assert leg.order_type is OrderType.MARKET
    assert leg.reduce_only is True
    assert leg.trigger_type is TriggerType.RISK_CLOSE


async def test_close_position_true_builds_market_reduce_only_leg_without_binance_close_position_flag():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)
    intent = make_intent(
        side=PositionSide.LONG,
        order_intent=OrderIntent.CLOSE,
        final_size=0.01,
        final_notional=500.0,
        risk_mode=RiskMode.REDUCE_ONLY,
        reduce_only=True,
        close_position=True,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=make_context(),
        mode=ExecutionMode.MARKET,
    )

    leg = plan.legs[0]

    assert leg.order_type is OrderType.MARKET
    assert leg.side is OrderSide.SELL
    assert leg.quantity == pytest.approx(0.01)
    assert leg.reduce_only is True
    assert leg.close_position is False
    assert leg.metadata["intent_close_position"] is True


# =============================================================================
# Mode selection
# =============================================================================


def test_choose_execution_mode_market_default_returns_market_for_clean_context():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)
    intent = make_intent()

    mode = smart.choose_execution_mode(intent, market_context=make_context())

    assert mode is ExecutionMode.MARKET


def test_choose_execution_mode_reduce_intent_prefers_market_for_exits():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIMIT,
        prefer_market_for_exits=True,
    )
    intent = make_intent(
        order_intent=OrderIntent.REDUCE,
        reduce_only=True,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    mode = smart.choose_execution_mode(intent, market_context=make_context())

    assert mode is ExecutionMode.MARKET


def test_choose_execution_mode_high_spread_with_prefer_limit_returns_limit():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_limit_for_entries=True,
        max_spread_bps=1.0,
    )
    intent = make_intent()

    # Wide spread: 100 bps approx.
    context = make_context(bid=49_500.0, ask=50_000.0)

    mode = smart.choose_execution_mode(intent, market_context=context)

    assert mode is ExecutionMode.LIMIT


def test_choose_execution_mode_liquidity_aware_requires_sufficient_depth():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIQUIDITY_AWARE,
        fallback_mode=ExecutionMode.MARKET,
    )
    intent = make_intent(final_notional=10_000.0)

    insufficient = make_context(available_depth_notional=15_000.0)
    sufficient = make_context(available_depth_notional=100_000.0)

    assert smart.choose_execution_mode(intent, market_context=insufficient) is ExecutionMode.MARKET
    assert smart.choose_execution_mode(intent, market_context=sufficient) is ExecutionMode.LIQUIDITY_AWARE


def test_choose_execution_mode_twap_requires_enabled_and_split_needed():
    smart = make_smart_execution(
        default_mode=ExecutionMode.TWAP,
        fallback_mode=ExecutionMode.MARKET,
        twap_enabled=True,
        max_split_count=5,
        min_leg_notional=100.0,
    )
    intent = make_intent(final_size=1.0, final_notional=50_000.0)

    mode = smart.choose_execution_mode(
        intent,
        market_context=make_context(available_depth_notional=5_000.0),
    )

    assert mode is ExecutionMode.TWAP


def test_choose_execution_mode_twap_disabled_falls_back_to_market():
    smart = make_smart_execution(
        default_mode=ExecutionMode.TWAP,
        fallback_mode=ExecutionMode.MARKET,
        twap_enabled=False,
    )
    intent = make_intent(final_size=1.0, final_notional=50_000.0)

    mode = smart.choose_execution_mode(
        intent,
        market_context=make_context(available_depth_notional=5_000.0),
    )

    assert mode is ExecutionMode.MARKET


# =============================================================================
# LIMIT / POST_ONLY price logic
# =============================================================================


async def test_limit_buy_uses_ask_minus_offset_and_rounds_to_tick():
    smart = make_smart_execution(default_mode=ExecutionMode.LIMIT)
    intent = make_intent(side=PositionSide.LONG, order_intent=OrderIntent.OPEN)

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIMIT,
    )

    leg = plan.legs[0]

    assert plan.mode is ExecutionMode.LIMIT
    assert leg.order_type is OrderType.LIMIT
    assert leg.side is OrderSide.BUY
    assert leg.time_in_force is TimeInForce.GTC

    # default limit_price_offset_bps = 1 bps
    assert leg.price == pytest.approx(49_995.0)


async def test_limit_sell_uses_bid_plus_offset_and_rounds_to_tick():
    smart = make_smart_execution(default_mode=ExecutionMode.LIMIT)
    intent = make_intent(side=PositionSide.SHORT, order_intent=OrderIntent.OPEN)

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIMIT,
    )

    leg = plan.legs[0]

    assert leg.order_type is OrderType.LIMIT
    assert leg.side is OrderSide.SELL
    assert leg.time_in_force is TimeInForce.GTC

    # 49_990 * 1.0001 = 49_994.999 => rounded to 49_995.0
    assert leg.price == pytest.approx(49_995.0)


async def test_post_only_buy_uses_bid_minus_offset_and_gtx():
    smart = make_smart_execution(default_mode=ExecutionMode.POST_ONLY)
    intent = make_intent(side=PositionSide.LONG)

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.POST_ONLY,
    )

    leg = plan.legs[0]

    assert plan.mode is ExecutionMode.POST_ONLY
    assert leg.order_type is OrderType.LIMIT
    assert leg.side is OrderSide.BUY
    assert leg.time_in_force is TimeInForce.GTX
    assert leg.price == pytest.approx(49_985.0)


async def test_post_only_sell_uses_ask_plus_offset_and_gtx():
    smart = make_smart_execution(default_mode=ExecutionMode.POST_ONLY)
    intent = make_intent(side=PositionSide.SHORT)

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.POST_ONLY,
    )

    leg = plan.legs[0]

    assert plan.mode is ExecutionMode.POST_ONLY
    assert leg.order_type is OrderType.LIMIT
    assert leg.side is OrderSide.SELL
    assert leg.time_in_force is TimeInForce.GTX
    assert leg.price == pytest.approx(50_005.0)


async def test_preferred_limit_price_overrides_bid_ask_price_selection():
    smart = make_smart_execution(default_mode=ExecutionMode.LIMIT)
    intent = make_intent(side=PositionSide.LONG)

    context = make_context(
        preferred_limit_price=49_876.543,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIMIT,
    )

    assert plan.legs[0].price == pytest.approx(49_876.5)


async def test_limit_price_falls_back_to_entry_price_when_bid_ask_missing():
    smart = make_smart_execution(default_mode=ExecutionMode.LIMIT)
    intent = make_intent(entry_price=50_123.456)

    context = make_context(
        bid=None,
        ask=None,
        mark_price=None,
        last_price=None,
        tick_size=0.1,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIMIT,
    )

    assert plan.legs[0].price == pytest.approx(50_123.5)


async def test_limit_plan_without_bid_ask_or_reference_price_fails():
    smart = make_smart_execution(default_mode=ExecutionMode.LIMIT)
    intent = make_intent(entry_price=None)

    context = make_context(
        bid=None,
        ask=None,
        mark_price=None,
        last_price=None,
        preferred_limit_price=None,
    )

    with pytest.raises(SmartExecutionError):
        await smart.build_execution_plan(
            intent,
            market_context=context,
            mode=ExecutionMode.LIMIT,
        )

    assert smart.stats.plans_failed == 1
    assert smart.stats.plans_created == 0


# =============================================================================
# Quantity rounding / split constraints
# =============================================================================


async def test_quantity_is_rounded_down_to_step_size_without_exceeding_final_size():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)
    intent = make_intent(final_size=0.0109, final_notional=545.0)

    context = make_context(step_size=0.001)

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.MARKET,
    )

    leg = plan.legs[0]

    assert leg.quantity == pytest.approx(0.01)
    assert leg.quantity <= intent.final_size


def test_split_order_does_not_exceed_final_size_and_respects_max_split_count():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIQUIDITY_AWARE,
        max_split_count=5,
        min_leg_notional=5.0,
    )
    intent = make_intent(
        final_size=1.0,
        final_notional=50_000.0,
    )

    context = make_context(
        step_size=0.001,
        available_depth_notional=5_000.0,
    )

    quantities = smart.split_order(intent, market_context=context)

    assert 1 < len(quantities) <= 5
    assert all(quantity > 0 for quantity in quantities)
    assert sum(quantities) <= intent.final_size + 1e-12
    assert sum(quantities) == pytest.approx(intent.final_size, abs=1e-12)


async def test_liquidity_aware_plan_splits_order_into_multiple_legs():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIQUIDITY_AWARE,
        max_split_count=5,
        min_leg_notional=5.0,
    )
    intent = make_intent(
        side=PositionSide.LONG,
        final_size=1.0,
        final_notional=50_000.0,
    )

    context = make_context(
        available_depth_notional=5_000.0,
        step_size=0.001,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIQUIDITY_AWARE,
    )

    total_quantity = sum(leg.quantity or 0 for leg in plan.legs)

    assert plan.mode is ExecutionMode.LIQUIDITY_AWARE
    assert 1 < len(plan.legs) <= 5
    assert total_quantity <= intent.final_size + 1e-12
    assert total_quantity == pytest.approx(intent.final_size, abs=1e-12)

    for index, leg in enumerate(plan.legs):
        assert leg.sequence == index
        assert leg.symbol == "BTCUSDT"
        assert leg.side is OrderSide.BUY
        assert leg.position_side is PositionSide.LONG
        assert leg.quantity is not None
        assert leg.quantity > 0
        assert leg.metadata["execution_mode"] == "liquidity_aware"
        assert leg.metadata["split_index"] == index
        assert leg.metadata["split_count"] == len(plan.legs)

    assert smart.stats.split_plans_created == 1


async def test_liquidity_aware_reduce_intent_uses_market_reduce_only_legs():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIQUIDITY_AWARE,
        max_split_count=5,
    )
    intent = make_intent(
        side=PositionSide.LONG,
        order_intent=OrderIntent.REDUCE,
        final_size=1.0,
        final_notional=50_000.0,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    context = make_context(
        available_depth_notional=5_000.0,
        step_size=0.001,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.LIQUIDITY_AWARE,
    )

    assert plan.mode is ExecutionMode.LIQUIDITY_AWARE
    assert len(plan.legs) > 1

    for leg in plan.legs:
        assert leg.order_type is OrderType.MARKET
        assert leg.side is OrderSide.SELL
        assert leg.reduce_only is True
        assert leg.trigger_type is TriggerType.RISK_REDUCE


def test_split_order_returns_single_quantity_when_splitting_disabled():
    smart = make_smart_execution(
        allow_order_splitting=False,
        max_split_count=5,
    )
    intent = make_intent(final_size=1.0, final_notional=50_000.0)

    quantities = smart.split_order(
        intent,
        market_context=make_context(available_depth_notional=1_000.0),
    )

    assert quantities == [1.0]


def test_split_order_returns_single_quantity_for_close_position():
    smart = make_smart_execution(
        allow_order_splitting=True,
        max_split_count=5,
    )
    intent = make_intent(
        order_intent=OrderIntent.CLOSE,
        final_size=1.0,
        final_notional=50_000.0,
        reduce_only=True,
        close_position=True,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    quantities = smart.split_order(
        intent,
        market_context=make_context(available_depth_notional=1_000.0),
    )

    assert quantities == [1.0]


# =============================================================================
# TWAP behavior
# =============================================================================


async def test_twap_plan_splits_order_and_adds_increasing_delay_metadata():
    smart = make_smart_execution(
        default_mode=ExecutionMode.TWAP,
        twap_enabled=True,
        max_split_count=5,
        min_leg_notional=5.0,
    )
    intent = make_intent(
        final_size=1.0,
        final_notional=50_000.0,
    )

    context = make_context(
        available_depth_notional=5_000.0,
        step_size=0.001,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=context,
        mode=ExecutionMode.TWAP,
    )

    assert plan.mode is ExecutionMode.TWAP
    assert 1 < len(plan.legs) <= 5

    delays = [leg.metadata["delay_seconds"] for leg in plan.legs]

    assert delays == sorted(delays)
    assert delays[0] == 0
    assert all(delays[index] < delays[index + 1] for index in range(len(delays) - 1))

    for index, leg in enumerate(plan.legs):
        assert leg.sequence == index
        assert leg.metadata["execution_mode"] == "twap"
        assert leg.metadata["split_index"] == index
        assert leg.metadata["split_count"] == len(plan.legs)

    assert smart.stats.plans_created == 1
    assert smart.stats.split_plans_created == 1


async def test_twap_disabled_falls_back_to_smart_plan():
    smart = make_smart_execution(
        default_mode=ExecutionMode.TWAP,
        twap_enabled=False,
        max_split_count=5,
    )
    intent = make_intent(
        final_size=1.0,
        final_notional=50_000.0,
    )

    plan = await smart.build_execution_plan(
        intent,
        market_context=make_context(available_depth_notional=5_000.0),
        mode=ExecutionMode.TWAP,
    )

    assert plan.mode in {
        ExecutionMode.MARKET,
        ExecutionMode.LIMIT,
        ExecutionMode.POST_ONLY,
        ExecutionMode.LIQUIDITY_AWARE,
    }
    assert plan.mode is not ExecutionMode.TWAP


# =============================================================================
# SMART behavior
# =============================================================================


async def test_smart_plan_prefers_post_only_when_configured_for_limit_entries_and_spread_is_ok():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_limit_for_entries=True,
        max_spread_bps=10.0,
    )
    intent = make_intent(
        side=PositionSide.LONG,
        order_intent=OrderIntent.OPEN,
    )

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
    )

    plan = await smart.build_execution_plan(intent, market_context=context)

    assert plan.mode is ExecutionMode.POST_ONLY
    assert plan.legs[0].order_type is OrderType.LIMIT
    assert plan.legs[0].time_in_force is TimeInForce.GTX


async def test_smart_plan_uses_market_when_slippage_and_spread_are_acceptable():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_limit_for_entries=False,
        max_slippage_bps=10.0,
        max_spread_bps=8.0,
    )
    intent = make_intent(final_size=0.01, final_notional=500.0)

    context = make_context(
        bid=49_990.0,
        ask=50_000.0,
        expected_slippage_bps=1.0,
    )

    plan = await smart.build_execution_plan(intent, market_context=context)

    assert plan.mode is ExecutionMode.MARKET
    assert plan.legs[0].order_type is OrderType.MARKET


async def test_smart_plan_uses_liquidity_aware_when_slippage_high_and_split_needed():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_limit_for_entries=False,
        allow_order_splitting=True,
        max_slippage_bps=5.0,
        max_split_count=5,
    )
    intent = make_intent(
        final_size=1.0,
        final_notional=50_000.0,
    )

    context = make_context(
        available_depth_notional=5_000.0,
        expected_slippage_bps=20.0,
    )

    plan = await smart.build_execution_plan(intent, market_context=context)

    assert plan.mode is ExecutionMode.LIQUIDITY_AWARE
    assert len(plan.legs) > 1


async def test_smart_plan_falls_back_to_limit_when_market_unfavorable_and_no_split_needed():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        prefer_limit_for_entries=False,
        allow_order_splitting=True,
        max_slippage_bps=5.0,
    )
    intent = make_intent(
        final_size=0.01,
        final_notional=500.0,
    )

    context = make_context(
        available_depth_notional=100_000.0,
        expected_slippage_bps=20.0,
    )

    plan = await smart.build_execution_plan(intent, market_context=context)

    assert plan.mode is ExecutionMode.LIMIT
    assert plan.legs[0].order_type is OrderType.LIMIT


# =============================================================================
# Estimates / constraints
# =============================================================================


def test_estimate_spread_bps_returns_none_for_missing_or_invalid_book():
    smart = make_smart_execution()

    assert smart.estimate_spread_bps({}) is None
    assert smart.estimate_spread_bps({"bid": None, "ask": 50_000.0}) is None
    assert smart.estimate_spread_bps({"bid": 50_000.0, "ask": 49_000.0}) is None


def test_estimate_spread_bps_calculates_valid_spread():
    smart = make_smart_execution()

    spread = smart.estimate_spread_bps(
        {
            "bid": 49_990.0,
            "ask": 50_000.0,
        }
    )

    assert spread == pytest.approx(2.0002, rel=1e-3)


def test_estimate_slippage_uses_explicit_context_value():
    smart = make_smart_execution()
    intent = make_intent(final_notional=1_000.0)

    slippage = smart.estimate_slippage_bps(
        intent,
        market_context={"expected_slippage_bps": 7.5},
    )

    assert slippage == pytest.approx(7.5)


def test_estimate_slippage_approximates_from_available_depth():
    smart = make_smart_execution()
    intent = make_intent(final_notional=10_000.0)

    slippage = smart.estimate_slippage_bps(
        intent,
        market_context={"available_depth_notional": 20_000.0},
    )

    assert slippage == pytest.approx(5.0)


def test_should_use_market_order_false_when_slippage_above_limit():
    smart = make_smart_execution(
        default_mode=ExecutionMode.MARKET,
        max_slippage_bps=5.0,
    )
    intent = make_intent()

    result = smart.should_use_market_order(
        intent,
        market_context={"expected_slippage_bps": 20.0},
    )

    assert result is False


def test_should_use_post_only_false_for_reduce_intent():
    smart = make_smart_execution(
        default_mode=ExecutionMode.POST_ONLY,
        prefer_limit_for_entries=True,
    )
    intent = make_intent(
        order_intent=OrderIntent.REDUCE,
        reduce_only=True,
        risk_mode=RiskMode.REDUCE_ONLY,
    )

    assert smart.should_use_post_only(intent, market_context=make_context()) is False


# =============================================================================
# Validation / failure paths
# =============================================================================


async def test_invalid_intent_size_fails_and_updates_failed_stats():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)

    intent = make_intent(final_size=0.0)

    with pytest.raises(SmartExecutionError):
        await smart.build_execution_plan(intent, market_context=make_context())

    assert smart.stats.plans_created == 0
    assert smart.stats.plans_failed == 1
    assert smart.stats.last_error is not None


async def test_invalid_rounded_quantity_fails_when_step_size_rounds_to_zero():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)

    intent = make_intent(
        final_size=0.0001,
        final_notional=5.0,
    )

    context = make_context(step_size=0.001)

    with pytest.raises(SmartExecutionError):
        await smart.build_execution_plan(
            intent,
            market_context=context,
            mode=ExecutionMode.MARKET,
        )

    assert smart.stats.plans_failed == 1


async def test_execution_plan_validation_fails_when_no_legs_possible():
    smart = make_smart_execution(
        default_mode=ExecutionMode.LIQUIDITY_AWARE,
        max_split_count=5,
    )

    intent = make_intent(
        final_size=0.0001,
        final_notional=10.0,
    )

    context = make_context(step_size=0.001, available_depth_notional=1.0)

    with pytest.raises(SmartExecutionError):
        await smart.build_execution_plan(
            intent,
            market_context=context,
            mode=ExecutionMode.LIQUIDITY_AWARE,
        )

    assert smart.stats.plans_failed == 1


async def test_build_execution_plan_rejects_unsupported_execution_market():
    smart = make_smart_execution(default_mode=ExecutionMode.MARKET)

    intent = ExecutionIntent(
        exchange="bybit",
        market_type="linear",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        order_intent=OrderIntent.OPEN,
        final_size=0.01,
        final_leverage=5.0,
        final_tier=TradeTier.T2,
        final_risk_amount=50.0,
        final_margin=100.0,
        final_notional=500.0,
    )

    with pytest.raises(SmartExecutionError):
        await smart.build_execution_plan(intent, market_context=make_context())

    assert smart.stats.plans_failed == 1


def test_snapshot_contains_config_and_stats():
    smart = make_smart_execution(
        default_mode=ExecutionMode.SMART,
        fallback_mode=ExecutionMode.MARKET,
    )

    snapshot = smart.snapshot()

    assert snapshot["service"] == "execution.smart_execution"
    assert snapshot["enabled"] is True
    assert snapshot["default_mode"] == "smart"
    assert snapshot["fallback_mode"] == "market"
    assert snapshot["stats"]["plans_created"] == 0
    assert snapshot["stats"]["plans_failed"] == 0