# test/fundingenginetest/test_funding_extreme_reversal_strategy_flow.py

from __future__ import annotations

from typing import Any

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingDivergenceType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)

from strategy.strategies.funding.base import (
    FundingSetupStatus,
    FundingStrategyDirection,
)


# =============================================================================
# Local constants / helpers
# =============================================================================


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_EXCHANGE = "binance"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_EXCHANGE_SYMBOL = "BTCUSDT"


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _pressure_level_value(name: str, fallback: str) -> str:
    value = getattr(FundingPressureLevel, name, None)
    return _enum_value(value) if value is not None else fallback


MEDIUM_PRESSURE_LEVEL = _pressure_level_value(
    "MEDIUM",
    _pressure_level_value("MODERATE", "moderate"),
)


def _scope(
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> dict[str, str]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
    }


def _last_record(event_bus_spy: Any) -> Any:
    assert event_bus_spy.emitted, "Expected at least one emitted strategy event"
    return event_bus_spy.emitted[-1]


def _records_for_topic(event_bus_spy: Any, topic: str) -> list[Any]:
    return [record for record in event_bus_spy.emitted if record.topic == topic]


def _last_record_for_topic(event_bus_spy: Any, topic: str) -> Any:
    records = _records_for_topic(event_bus_spy, topic)
    assert records, f"No records for topic={topic!r}; emitted={[r.topic for r in event_bus_spy.emitted]}"
    return records[-1]


def _assert_no_events(event_bus_spy: Any) -> None:
    assert event_bus_spy.emitted == []


def _state(
    strategy: Any,
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str | FundingTimeframe = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> Any:
    return strategy.get_state(
        symbol,
        exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


def _assert_state_scope(
    state: Any,
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> None:
    expected_scope = _scope(
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )

    assert state.symbol == symbol
    assert state.exchange == exchange
    assert state.market_type == market_type
    assert state.timeframe.value == timeframe
    assert state.exchange_symbol == exchange_symbol
    assert state.scope.to_dict() == expected_scope
    assert state.key == f"{exchange}:{market_type}:{symbol}:{timeframe}"
    assert state.legacy_key == f"{symbol}:{exchange}"


def _assert_state(
    strategy: Any,
    *,
    status: FundingSetupStatus,
    direction: FundingStrategyDirection | None = None,
    reason: str | None = None,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> Any:
    state = _state(
        strategy,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )
    _assert_state_scope(
        state,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )

    assert state.status == status

    if direction is not None:
        assert state.direction == direction

    if reason is not None:
        assert state.reason == reason

    return state


def _assert_event_scope(
    record: Any,
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> None:
    expected_scope = _scope(
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )
    payload = record.payload
    headers = record.headers

    assert payload["symbol"] == symbol
    assert payload["exchange"] == exchange
    assert payload["market_type"] == market_type
    assert payload["timeframe"] == timeframe
    assert payload["exchange_symbol"] == exchange_symbol
    assert payload["scope"] == expected_scope

    assert headers["symbol"] == symbol
    assert headers["exchange"] == exchange
    assert headers["market_type"] == market_type
    assert headers["timeframe"] == timeframe
    assert headers["exchange_symbol"] == exchange_symbol
    assert headers["scope"] == expected_scope

    assert payload["state"]["scope"] == expected_scope
    assert payload["state"]["key"] == f"{exchange}:{market_type}:{symbol}:{timeframe}"
    assert payload["state"]["legacy_key"] == f"{symbol}:{exchange}"

    assert payload["funding_context"]["scope"] == expected_scope
    assert payload["analytics_context"]["scope"] == expected_scope


def _assert_last_event(
    event_bus_spy: Any,
    *,
    topic: str,
    event_kind: str,
    direction: FundingStrategyDirection | None = None,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> dict[str, Any]:
    record = _last_record_for_topic(event_bus_spy, topic)
    assert record.topic == topic

    payload = record.payload
    assert payload["event_kind"] == event_kind
    assert payload["strategy"] == "funding_extreme_reversal"
    assert payload["strategy_name"] == "funding_extreme_reversal"
    assert payload["strategy_namespace"] == "strategy.funding.extreme_reversal"
    assert payload["strategy_family"] == "funding"
    assert payload["strategy_variant"] == "extreme_reversal"
    assert payload["signal_class"] == "contrarian_reversal"

    _assert_event_scope(
        record,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )

    if direction is not None:
        assert payload["direction"] == direction.value

    if event_kind == "confirmed":
        assert payload["is_tradeable"] is True

    if event_kind == "setup":
        assert payload.get("is_tradeable") is not True

    return payload


def _copy_context(context: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in context.items()}


def _with_scope(
    payload: dict[str, Any],
    *,
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> dict[str, Any]:
    updated = dict(payload)
    updated["symbol"] = symbol
    updated["exchange"] = exchange
    updated["market_type"] = market_type
    updated["timeframe"] = timeframe
    updated["exchange_symbol"] = exchange_symbol

    metadata = dict(updated.get("metadata") or {})
    metadata["scope"] = {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
        "exchange_symbol": exchange_symbol,
    }
    metadata["exchange_symbol"] = exchange_symbol
    updated["metadata"] = metadata
    return updated


async def _seed_crowded_longs_context(
    strategy: Any,
    *,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            crowded_longs_context["regime"],
        )
    )
    await strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            crowded_longs_context["pressure"],
        )
    )


async def _seed_crowded_shorts_context(
    strategy: Any,
    *,
    make_test_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    await strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            crowded_shorts_context["regime"],
        )
    )
    await strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            crowded_shorts_context["pressure"],
        )
    )


async def _create_short_reversal_setup(
    strategy: Any,
    *,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> Any:
    await _seed_crowded_longs_context(
        strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    await strategy.on_extreme(
        make_test_event(
            "analytics.funding.extreme",
            crowded_longs_context["extreme"],
        )
    )
    return _state(
        strategy,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


async def _create_long_reversal_setup(
    strategy: Any,
    *,
    make_test_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
    symbol: str = DEFAULT_SYMBOL,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    timeframe: str = DEFAULT_TIMEFRAME,
    exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
) -> Any:
    await _seed_crowded_shorts_context(
        strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=crowded_shorts_context,
    )
    await strategy.on_extreme(
        make_test_event(
            "analytics.funding.extreme",
            crowded_shorts_context["extreme"],
        )
    )
    return _state(
        strategy,
        symbol=symbol,
        exchange=exchange,
        market_type=market_type,
        timeframe=timeframe,
        exchange_symbol=exchange_symbol,
    )


# =============================================================================
# Setup creation: happy paths
# =============================================================================


@pytest.mark.asyncio
async def test_positive_funding_extreme_creates_short_reversal_setup_with_full_scope_contract(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    state = await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )

    _assert_state_scope(state)
    assert state.status == FundingSetupStatus.SETUP_DETECTED
    assert state.direction == FundingStrategyDirection.SHORT
    assert state.setup_type == extreme_reversal_strategy.config.bearish_setup_type
    assert state.score > 0.0
    assert state.confidence > 0.0
    assert state.expires_at is not None
    assert state.setup_event_time is not None

    assert extreme_reversal_strategy.config.tag_extreme in state.tags
    assert extreme_reversal_strategy.config.tag_reversal in state.tags
    assert extreme_reversal_strategy.config.tag_crowding in state.tags
    assert state.metadata["scope"] == state.scope.to_dict()
    assert state.metadata["is_reversal_risk"] is True

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "extreme"
    assert payload["setup_type"] == extreme_reversal_strategy.config.bearish_setup_type
    assert payload["metadata"]["scope"] == state.scope.to_dict()


@pytest.mark.asyncio
async def test_negative_funding_extreme_creates_long_reversal_setup_with_full_scope_contract(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    state = await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=crowded_shorts_context,
    )

    _assert_state_scope(state)
    assert state.status == FundingSetupStatus.SETUP_DETECTED
    assert state.direction == FundingStrategyDirection.LONG
    assert state.setup_type == extreme_reversal_strategy.config.bullish_setup_type
    assert state.score > 0.0
    assert state.confidence > 0.0
    assert state.expires_at is not None

    assert extreme_reversal_strategy.config.tag_extreme in state.tags
    assert extreme_reversal_strategy.config.tag_reversal in state.tags
    assert extreme_reversal_strategy.config.tag_crowding in state.tags
    assert state.metadata["scope"] == state.scope.to_dict()

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "extreme"
    assert payload["setup_type"] == extreme_reversal_strategy.config.bullish_setup_type


# =============================================================================
# Setup creation: context filters
# =============================================================================


@pytest.mark.asyncio
async def test_extreme_event_does_not_create_setup_without_regime_context(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await extreme_reversal_strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            crowded_longs_context["pressure"],
        )
    )
    await extreme_reversal_strategy.on_extreme(
        make_test_event(
            "analytics.funding.extreme",
            crowded_longs_context["extreme"],
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    assert state.last_pressure is not None
    assert state.last_extreme is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_extreme_event_does_not_create_setup_without_pressure_context(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await extreme_reversal_strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            crowded_longs_context["regime"],
        )
    )
    await extreme_reversal_strategy.on_extreme(
        make_test_event(
            "analytics.funding.extreme",
            crowded_longs_context["extreme"],
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    assert state.last_regime is not None
    assert state.last_extreme is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_extreme_below_min_severity_is_ignored(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    context = _copy_context(crowded_longs_context)
    context["extreme"]["severity"] = extreme_reversal_strategy.config.min_extreme_severity - 0.01

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_low_regime_confidence_blocks_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    context = _copy_context(crowded_longs_context)
    context["regime"]["confidence"] = extreme_reversal_strategy.config.min_regime_confidence - 0.01

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_low_pressure_score_blocks_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    context = _copy_context(crowded_longs_context)
    context["pressure"]["pressure_score"] = extreme_reversal_strategy.config.min_pressure_score - 0.01

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_level_must_be_high_or_extreme_when_required(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.require_high_pressure_level = True

    context = _copy_context(crowded_longs_context)
    context["pressure"]["level"] = FundingPressureLevel.LOW.value

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_setup_requires_reversal_risk_when_enabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.require_reversal_risk = True

    context = _copy_context(crowded_longs_context)
    context["extreme"]["is_reversal_risk"] = False
    context["extreme"]["is_squeeze_risk"] = True

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_setup_allows_missing_reversal_risk_when_filter_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.require_reversal_risk = False

    context = _copy_context(crowded_longs_context)
    context["extreme"]["is_reversal_risk"] = False
    context["extreme"]["is_squeeze_risk"] = True

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.SHORT,
    )


@pytest.mark.asyncio
async def test_setup_requires_squeeze_or_mean_reversion_probability(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.require_squeeze_risk_or_reversion_probability = True

    context = _copy_context(crowded_longs_context)
    context["extreme"]["is_squeeze_risk"] = False
    context["pressure"]["squeeze_probability"] = (
        extreme_reversal_strategy.config.min_squeeze_probability - 0.01
    )
    context["pressure"]["mean_reversion_probability"] = (
        extreme_reversal_strategy.config.min_mean_reversion_probability - 0.01
    )

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_setup_allows_no_squeeze_risk_when_mean_reversion_probability_is_high(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.require_squeeze_risk_or_reversion_probability = True

    context = _copy_context(crowded_longs_context)
    context["extreme"]["is_squeeze_risk"] = False
    context["pressure"]["squeeze_probability"] = (
        extreme_reversal_strategy.config.min_squeeze_probability - 0.01
    )
    context["pressure"]["mean_reversion_probability"] = (
        extreme_reversal_strategy.config.min_mean_reversion_probability + 0.10
    )

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.SHORT,
    )


@pytest.mark.asyncio
async def test_wrong_pressure_direction_blocks_positive_extreme_short_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    context = _copy_context(crowded_longs_context)
    context["pressure"]["direction"] = FundingPressureDirection.SHORT.value

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_wrong_bias_blocks_positive_extreme_short_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    context = _copy_context(crowded_longs_context)
    context["regime"]["bias"] = FundingBias.SHORT_BIAS.value

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=context,
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


# =============================================================================
# Full futures scope isolation
# =============================================================================


@pytest.mark.asyncio
async def test_same_symbol_different_market_types_create_independent_reversal_setups(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    usdm_context = _copy_context(crowded_longs_context)
    coinm_context = _copy_context(crowded_shorts_context)

    for key in coinm_context:
        coinm_context[key] = _with_scope(
            coinm_context[key],
            market_type="coinm_futures",
            exchange_symbol="BTCUSD_PERP",
        )

    usdm_state = await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=usdm_context,
    )
    coinm_state = await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=coinm_context,
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    assert usdm_state is not coinm_state
    assert usdm_state.key == "binance:usdm_futures:BTCUSDT:1h"
    assert coinm_state.key == "binance:coinm_futures:BTCUSDT:1h"
    assert usdm_state.direction == FundingStrategyDirection.SHORT
    assert coinm_state.direction == FundingStrategyDirection.LONG
    assert extreme_reversal_strategy.stats()["states_total"] == 2

    records = _records_for_topic(event_bus_spy, "strategy.funding.extreme_reversal.setup")
    assert len(records) == 2
    assert records[0].payload["scope"]["market_type"] == "usdm_futures"
    assert records[1].payload["scope"]["market_type"] == "coinm_futures"
    assert records[1].payload["exchange_symbol"] == "BTCUSD_PERP"


@pytest.mark.asyncio
async def test_opposite_signal_on_different_timeframe_does_not_invalidate_existing_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            timeframe="4h",
            score=0.80,
            confidence=0.90,
            signal_origin="extreme_squeeze",
            bias=FundingBias.SHORT_BIAS,
            signal_type=FundingSignalType.REVERSION_SETUP,
        )
    )

    one_hour_state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    four_hour_state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
        timeframe="4h",
    )

    assert one_hour_state is not four_hour_state
    assert one_hour_state.reason != "opposite_funding_signal_invalidated_reversal_setup"
    assert len(four_hour_state.recent_signals) == 1
    _assert_no_events(event_bus_spy)


# =============================================================================
# Extreme type score weighting
# =============================================================================


@pytest.mark.asyncio
async def test_extreme_type_score_ordering_global_greater_than_percentile_greater_than_zscore_greater_than_local(
    make_test_event: Any,
    event_bus_spy: Any,
    scheduler_spy: Any,
    parquet_storage_spy: Any,
    extreme_reversal_config: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    from strategy.strategies.funding.funding_extreme_reversal_strategy import (
        FundingExtremeReversalStrategy,
    )

    extreme_types = [
        FundingExtremeType.GLOBAL_HIGH,
        FundingExtremeType.PERCENTILE_HIGH,
        FundingExtremeType.ZSCORE_HIGH,
        FundingExtremeType.LOCAL_HIGH,
    ]

    scores: list[float] = []

    for index, extreme_type in enumerate(extreme_types):
        strategy = FundingExtremeReversalStrategy(
            event_bus=event_bus_spy,
            scheduler=scheduler_spy,
            parquet_storage=parquet_storage_spy,
            config=extreme_reversal_config,
        )
        context = _copy_context(crowded_longs_context)

        symbol = f"BTCUSDT{index}"
        exchange_symbol = f"BTCUSDT-{index}"

        for key in context:
            context[key] = _with_scope(
                context[key],
                symbol=symbol,
                exchange_symbol=exchange_symbol,
            )

        context["extreme"]["extreme_type"] = extreme_type.value
        context["extreme"]["severity"] = 0.80

        state = await _create_short_reversal_setup(
            strategy,
            make_test_event=make_test_event,
            crowded_longs_context=context,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
        )
        scores.append(state.score)

    assert scores[0] > scores[1] > scores[2] > scores[3]


# =============================================================================
# Flip confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_flip_confirms_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_to_negative_flip_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_flip(make_positive_to_negative_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="flip_confirmed_reversal_setup",
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_flip in state.tags
    assert state.metadata["confirmation_source"] == "flip"
    assert state.metadata["scope"] == state.scope.to_dict()
    assert state.confirmed_at is not None
    assert state.confirmation_event_time is not None

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "flip"
    assert payload["metadata"]["confirmation_source"] == "flip"


@pytest.mark.asyncio
async def test_flip_confirms_long_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_to_positive_flip_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=crowded_shorts_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_flip(make_negative_to_positive_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="flip_confirmed_reversal_setup",
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_flip in state.tags
    assert state.metadata["confirmation_source"] == "flip"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "flip"


@pytest.mark.asyncio
async def test_opposite_flip_invalidates_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_to_positive_flip_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_flip(make_negative_to_positive_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="opposite_flip_invalidated_reversal_setup",
    )
    assert state.cooldown_until is not None
    assert state.metadata["invalidation_source"] == "flip"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "flip"
    assert payload["metadata"]["invalidation_source"] == "flip"


@pytest.mark.asyncio
async def test_flip_does_nothing_when_no_active_setup_exists(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_positive_to_negative_flip_event: Any,
) -> None:
    await extreme_reversal_strategy.on_flip(make_positive_to_negative_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    assert state.last_flip is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_flip_confirmation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_to_negative_flip_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.allow_flip_confirmation = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_flip(make_positive_to_negative_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_flip not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_flip_invalidation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_to_positive_flip_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_opposite_flip = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_flip(make_negative_to_positive_flip_event())

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.reason != "opposite_flip_invalidated_reversal_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Pressure confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_pressure_release_confirms_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    released_pressure = dict(crowded_longs_context["pressure"])
    released_pressure["pressure_score"] = (
        crowded_longs_context["pressure"]["pressure_score"]
        - extreme_reversal_strategy.config.pressure_release_min_score_drop
        - 0.05
    )
    released_pressure["level"] = MEDIUM_PRESSURE_LEVEL
    released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value

    await extreme_reversal_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", released_pressure)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="pressure_release_confirmed_reversal_setup",
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_release in state.tags
    assert state.metadata["confirmation_source"] == "pressure_release"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "pressure_release"
    assert payload["metadata"]["confirmation_source"] == "pressure_release"


@pytest.mark.asyncio
async def test_pressure_neutralization_invalidates_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    neutral_pressure = dict(crowded_longs_context["pressure"])
    neutral_pressure["direction"] = FundingPressureDirection.NEUTRAL.value
    neutral_pressure["level"] = FundingPressureLevel.LOW.value
    neutral_pressure["pressure_score"] = 0.01

    await extreme_reversal_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", neutral_pressure)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="pressure_context_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "pressure"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "pressure"


@pytest.mark.asyncio
async def test_pressure_release_confirmation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.allow_pressure_release_confirmation = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    released_pressure = dict(crowded_longs_context["pressure"])
    released_pressure["pressure_score"] = 0.01
    released_pressure["level"] = MEDIUM_PRESSURE_LEVEL
    released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value

    await extreme_reversal_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", released_pressure)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_release not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_neutralization_invalidation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_pressure_neutralization = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    neutral_pressure = dict(crowded_longs_context["pressure"])
    neutral_pressure["direction"] = FundingPressureDirection.NEUTRAL.value
    neutral_pressure["level"] = FundingPressureLevel.LOW.value
    neutral_pressure["pressure_score"] = 0.01

    await extreme_reversal_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", neutral_pressure)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.reason != "pressure_context_invalidated_reversal_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Divergence confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_aligned_bearish_divergence_confirms_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bearish_divergence_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bearish_divergence_event(confidence=0.90)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="divergence_confirmed_reversal_setup",
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_divergence in state.tags
    assert extreme_reversal_strategy.config.tag_divergence in state.tags
    assert state.metadata["confirmation_source"] == "divergence"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "divergence"


@pytest.mark.asyncio
async def test_aligned_bullish_divergence_confirms_long_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bullish_divergence_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=crowded_shorts_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bullish_divergence_event(confidence=0.90)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="divergence_confirmed_reversal_setup",
    )
    assert state.metadata["confirmation_source"] == "divergence"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "divergence"


@pytest.mark.asyncio
async def test_opposite_divergence_invalidates_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bullish_divergence_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bullish_divergence_event(confidence=0.90)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="opposite_divergence_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "divergence"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "divergence"


@pytest.mark.asyncio
async def test_low_confidence_divergence_does_not_confirm_setup_but_is_stored(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bearish_divergence_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bearish_divergence_event(
            confidence=extreme_reversal_strategy.config.min_divergence_confidence - 0.01
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.last_divergence is not None
    assert extreme_reversal_strategy.config.tag_confirmed_by_divergence not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_divergence_confirmation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bearish_divergence_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.allow_divergence_confirmation = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bearish_divergence_event(confidence=0.90)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_divergence not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_divergence_invalidation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bullish_divergence_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_opposite_divergence = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_divergence(
        make_bullish_divergence_event(confidence=0.90)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.reason != "opposite_divergence_invalidated_reversal_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Funding signal confirmation / invalidation / signal origin
# =============================================================================


@pytest.mark.asyncio
async def test_funding_signal_confirms_short_reversal_setup_and_tracks_origin(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=0.85,
            signal_origin="extreme_reversion",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="funding_signal_confirmed_reversal_setup",
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_signal in state.tags
    assert extreme_reversal_strategy.config.tag_signal in state.tags
    assert state.metadata["confirmation_source"] == "funding_signal"
    assert state.metadata["signal_origin"] == "extreme_reversion"
    assert "extreme_reversion" in state.last_signals_by_origin
    assert len(state.recent_signals) == 1

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_signal"
    assert payload["metadata"]["signal_origin"] == "extreme_reversion"


@pytest.mark.asyncio
async def test_funding_signal_confirms_long_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=crowded_shorts_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=0.85,
            signal_origin="pressure_reversion",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="funding_signal_confirmed_reversal_setup",
    )
    assert state.metadata["confirmation_source"] == "funding_signal"
    assert state.metadata["signal_origin"] == "pressure_reversion"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "funding_signal"


@pytest.mark.asyncio
async def test_opposite_funding_signal_invalidates_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=0.85,
            signal_origin="extreme_squeeze",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="opposite_funding_signal_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "funding_signal"
    assert state.metadata["signal_origin"] == "extreme_squeeze"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_signal"


@pytest.mark.asyncio
async def test_low_confidence_funding_signal_does_not_confirm_setup_but_is_stored(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=extreme_reversal_strategy.config.min_signal_confidence - 0.01,
            signal_origin="extreme_reversion",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_signal not in state.tags
    assert "extreme_reversion" in state.last_signals_by_origin
    assert len(state.recent_signals) == 1
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_low_abs_score_funding_signal_does_not_confirm_setup_but_is_stored(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-(extreme_reversal_strategy.config.min_signal_abs_score - 0.01),
            confidence=0.85,
            signal_origin="extreme_reversion",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_signal not in state.tags
    assert "extreme_reversion" in state.last_signals_by_origin
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_funding_signal_confirmation_can_be_disabled_but_signal_history_is_kept(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.allow_signal_confirmation = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=0.85,
            signal_origin="extreme_reversion",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert extreme_reversal_strategy.config.tag_confirmed_by_signal not in state.tags
    assert "extreme_reversion" in state.last_signals_by_origin
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_funding_signal_invalidation_can_be_disabled_but_signal_history_is_kept(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_opposite_signal = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=0.85,
            signal_origin="extreme_squeeze",
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.reason != "opposite_funding_signal_invalidated_reversal_setup"
    assert "extreme_squeeze" in state.last_signals_by_origin
    _assert_no_events(event_bus_spy)


# =============================================================================
# Regime invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_regime_conflict_invalidates_short_reversal_setup(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    conflicting_regime = dict(crowded_longs_context["regime"])
    conflicting_regime["bias"] = FundingBias.SHORT_BIAS.value
    conflicting_regime["regime"] = FundingRegime.NEGATIVE.value
    conflicting_regime["confidence"] = 0.90

    await extreme_reversal_strategy.on_regime(
        make_test_event("analytics.funding.regime", conflicting_regime)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="regime_context_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "regime"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "regime"


@pytest.mark.asyncio
async def test_regime_conflict_invalidation_can_be_disabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_regime_conflict = False

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    conflicting_regime = dict(crowded_longs_context["regime"])
    conflicting_regime["bias"] = FundingBias.SHORT_BIAS.value
    conflicting_regime["regime"] = FundingRegime.NEGATIVE.value
    conflicting_regime["confidence"] = 0.90

    await extreme_reversal_strategy.on_regime(
        make_test_event("analytics.funding.regime", conflicting_regime)
    )

    _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_neutral_regime_invalidates_when_enabled(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.invalidate_on_regime_neutral = True

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    neutral_regime = dict(crowded_longs_context["regime"])
    neutral_regime["regime"] = FundingRegime.NEUTRAL.value
    neutral_regime["bias"] = FundingBias.NEUTRAL.value
    neutral_regime["confidence"] = 0.90

    await extreme_reversal_strategy.on_regime(
        make_test_event("analytics.funding.regime", neutral_regime)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="regime_context_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "regime"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "regime"


# =============================================================================
# Stale events / cooldown / active setup protection
# =============================================================================


@pytest.mark.asyncio
async def test_stale_extreme_event_is_ignored(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _seed_crowded_longs_context(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )

    stale_extreme = dict(crowded_longs_context["extreme"])
    stale_extreme["event_time"] = "2000-01-01T00:00:00+00:00"

    await extreme_reversal_strategy.on_extreme(
        make_test_event("analytics.funding.extreme", stale_extreme)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    assert state.last_extreme is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_cooldown_blocks_new_extreme_setup_for_same_full_scope_only(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
    crowded_shorts_context: dict[str, dict[str, Any]],
) -> None:
    state = _state(extreme_reversal_strategy)
    extreme_reversal_strategy.set_cooldown(
        state,
        cooldown_sec=60.0,
        reason="pytest_cooldown",
    )

    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )

    same_scope_state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.NEUTRAL,
        reason="pytest_cooldown",
    )
    assert same_scope_state.cooldown_until is not None
    _assert_no_events(event_bus_spy)

    coinm_context = _copy_context(crowded_shorts_context)
    for key in coinm_context:
        coinm_context[key] = _with_scope(
            coinm_context[key],
            market_type="coinm_futures",
            exchange_symbol="BTCUSD_PERP",
        )

    await _create_long_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_shorts_context=coinm_context,
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )

    coinm_state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
        market_type="coinm_futures",
        exchange_symbol="BTCUSD_PERP",
    )
    assert coinm_state is not same_scope_state
    assert extreme_reversal_strategy.stats()["states_total"] == 2


@pytest.mark.asyncio
async def test_extreme_event_does_not_replace_active_setup_by_default(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    first_state = _state(extreme_reversal_strategy)
    first_created_at = first_state.created_at
    first_score = first_state.score

    event_bus_spy.emitted.clear()

    stronger_extreme = dict(crowded_longs_context["extreme"])
    stronger_extreme["severity"] = 1.0

    await extreme_reversal_strategy.on_extreme(
        make_test_event("analytics.funding.extreme", stronger_extreme)
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.created_at == first_created_at
    assert state.score == pytest.approx(first_score)
    _assert_no_events(event_bus_spy)


# =============================================================================
# analytics.funding.updated integration
# =============================================================================


@pytest.mark.asyncio
async def test_funding_updated_can_create_setup_from_atomic_context_with_full_scope(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_funding_updated_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    event = make_funding_updated_event(
        regime_state=crowded_longs_context["regime"],
        pressure_state=crowded_longs_context["pressure"],
        extreme_event=crowded_longs_context["extreme"],
    )

    await extreme_reversal_strategy.on_funding_updated(event)

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_extreme is not None
    assert state.last_funding_updated_payload is not None

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_updated"


@pytest.mark.asyncio
async def test_funding_updated_setup_creation_can_be_disabled_but_context_is_kept(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_funding_updated_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    extreme_reversal_strategy.config.allow_updated_context_setup = False

    event = make_funding_updated_event(
        regime_state=crowded_longs_context["regime"],
        pressure_state=crowded_longs_context["pressure"],
        extreme_event=crowded_longs_context["extreme"],
    )

    await extreme_reversal_strategy.on_funding_updated(event)

    state = _state(extreme_reversal_strategy)
    assert state.status == FundingSetupStatus.IDLE
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_extreme is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_funding_updated_can_confirm_active_setup_from_recent_aligned_signal(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    make_funding_updated_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    extreme_reversal_strategy.config.allow_signal_confirmation = False

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_origin="extreme_reversion",
            score=-0.86,
            confidence=0.91,
            bias=FundingBias.LONG_BIAS,
            signal_type=FundingSignalType.REVERSION_SETUP,
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert "extreme_reversion" in state.last_signals_by_origin

    event_bus_spy.emitted.clear()
    extreme_reversal_strategy.config.allow_signal_confirmation = True

    await extreme_reversal_strategy.on_funding_updated(
        make_funding_updated_event(
            regime_state=crowded_longs_context["regime"],
            pressure_state=crowded_longs_context["pressure"],
            extreme_event=crowded_longs_context["extreme"],
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="atomic_update_signal_confirmed_reversal_setup",
    )
    assert state.metadata["confirmation_source"] == "funding_updated.signal"
    assert state.metadata["signal_origin"] == "extreme_reversion"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_updated"


@pytest.mark.asyncio
async def test_funding_updated_can_invalidate_active_setup_from_recent_opposite_signal(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    make_funding_updated_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    extreme_reversal_strategy.config.invalidate_on_opposite_signal = False

    await extreme_reversal_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_origin="extreme_squeeze",
            score=0.86,
            confidence=0.91,
            bias=FundingBias.SHORT_BIAS,
            signal_type=FundingSignalType.REVERSION_SETUP,
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.SHORT,
    )
    assert "extreme_squeeze" in state.last_signals_by_origin

    event_bus_spy.emitted.clear()
    extreme_reversal_strategy.config.invalidate_on_opposite_signal = True

    await extreme_reversal_strategy.on_funding_updated(
        make_funding_updated_event(
            regime_state=crowded_longs_context["regime"],
            pressure_state=crowded_longs_context["pressure"],
            extreme_event=crowded_longs_context["extreme"],
        )
    )

    state = _assert_state(
        extreme_reversal_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.SHORT,
        reason="atomic_update_opposite_signal_invalidated_reversal_setup",
    )
    assert state.metadata["invalidation_source"] == "funding_updated.signal"
    assert state.metadata["signal_origin"] == "extreme_squeeze"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.extreme_reversal.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_updated"


@pytest.mark.asyncio
async def test_funding_updated_active_state_keeps_single_state_and_context_consistent(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_updated_event: Any,
    crowded_longs_context: dict[str, dict[str, Any]],
) -> None:
    await _create_short_reversal_setup(
        extreme_reversal_strategy,
        make_test_event=make_test_event,
        crowded_longs_context=crowded_longs_context,
    )
    event_bus_spy.emitted.clear()

    released_pressure = dict(crowded_longs_context["pressure"])
    released_pressure["pressure_score"] = 0.01
    released_pressure["level"] = MEDIUM_PRESSURE_LEVEL
    released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value

    event = make_funding_updated_event(
        regime_state=crowded_longs_context["regime"],
        pressure_state=released_pressure,
        extreme_event=crowded_longs_context["extreme"],
    )

    await extreme_reversal_strategy.on_funding_updated(event)

    state = _state(extreme_reversal_strategy)
    assert state.status in {
        FundingSetupStatus.SETUP_DETECTED,
        FundingSetupStatus.CONFIRMED,
        FundingSetupStatus.COOLDOWN,
    }
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_extreme is not None
    assert len(extreme_reversal_strategy.get_all_states()) == 1


# =============================================================================
# Guard clauses / error-tolerant behavior
# =============================================================================


@pytest.mark.asyncio
async def test_handlers_ignore_events_without_symbol(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
) -> None:
    await extreme_reversal_strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "timeframe": "1h",
                "regime": FundingRegime.POSITIVE.value,
                "bias": FundingBias.LONG_BIAS.value,
                "confidence": 0.80,
            },
        )
    )
    await extreme_reversal_strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "timeframe": "1h",
                "direction": FundingPressureDirection.LONG.value,
                "level": FundingPressureLevel.HIGH.value,
                "pressure_score": 0.80,
            },
        )
    )
    await extreme_reversal_strategy.on_extreme(
        make_test_event(
            "analytics.funding.extreme",
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "timeframe": "1h",
                "severity": 0.90,
            },
        )
    )

    assert extreme_reversal_strategy.get_all_states() == {}
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_handler_returns_without_state_change_when_lock_timeout(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_positive_extreme_event: Any,
) -> None:
    lock = await extreme_reversal_strategy.acquire_symbol_lock("BTCUSDT", "binance")
    assert lock is not None

    try:
        await extreme_reversal_strategy.on_extreme(make_positive_extreme_event())
    finally:
        extreme_reversal_strategy.release_symbol_lock(lock)

    state = _state(extreme_reversal_strategy)
    assert state.status == FundingSetupStatus.IDLE
    assert state.last_extreme is None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_malformed_event_is_caught_and_does_not_raise(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
) -> None:
    event = make_test_event(
        "analytics.funding.extreme",
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "market_type": "usdm_futures",
            "timeframe": "1h",
            "exchange_symbol": "BTCUSDT",
            "severity": object(),
            "extreme_type": object(),
            "event_time": object(),
        },
    )

    await extreme_reversal_strategy.on_extreme(event)

    state = _state(extreme_reversal_strategy)
    assert state.status in {
        FundingSetupStatus.IDLE,
        FundingSetupStatus.SETUP_DETECTED,
        FundingSetupStatus.COOLDOWN,
    }
    assert isinstance(event_bus_spy.emitted, list)


# =============================================================================
# Registration / stats
# =============================================================================


def test_extreme_reversal_registers_expected_subscriptions(
    extreme_reversal_strategy: Any,
    event_bus_spy: Any,
) -> None:
    extreme_reversal_strategy.register()

    patterns = {subscription.pattern for subscription in event_bus_spy.subscribed}

    assert patterns == {
        "analytics.funding.updated",
        "analytics.funding.signal",
        "analytics.funding.regime",
        "analytics.funding.pressure",
        "analytics.funding.extreme",
        "analytics.funding.flip",
        "analytics.funding.divergence",
    }

    names = {subscription.name for subscription in event_bus_spy.subscribed}
    assert "funding_extreme_reversal.on_regime" in names
    assert "funding_extreme_reversal.on_pressure" in names
    assert "funding_extreme_reversal.on_extreme" in names
    assert "funding_extreme_reversal.on_flip" in names
    assert "funding_extreme_reversal.on_divergence" in names


def test_extreme_reversal_stats_report_strategy_identity(
    extreme_reversal_strategy: Any,
) -> None:
    stats = extreme_reversal_strategy.stats()

    assert stats["strategy"] == "funding_extreme_reversal"
    assert stats["namespace"] == "strategy.funding.extreme_reversal"
    assert stats["registered"] is False
    assert stats["running"] is False
    assert stats["states_total"] == 0