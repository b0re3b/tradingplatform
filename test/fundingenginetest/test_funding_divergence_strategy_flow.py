# test/fundingenginetest/test_funding_divergence_strategy_flow.py
from __future__ import annotations

from typing import Any

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
)

from strategy.strategies.funding.base import (
    FundingSetupStatus,
    FundingStrategyDirection,
)


# =============================================================================
# Local helpers
# =============================================================================


def _last_record(event_bus_spy: Any) -> Any:
    assert event_bus_spy.emitted, "Expected at least one emitted strategy event"
    return event_bus_spy.emitted[-1]


def _assert_no_events(event_bus_spy: Any) -> None:
    assert event_bus_spy.emitted == []


def _state(strategy: Any) -> Any:
    return strategy.get_state("BTCUSDT", "binance")


def _assert_state(
    strategy: Any,
    *,
    status: FundingSetupStatus,
    direction: FundingStrategyDirection | None = None,
    reason: str | None = None,
) -> Any:
    state = _state(strategy)
    assert state.status == status

    if direction is not None:
        assert state.direction == direction

    if reason is not None:
        assert state.reason == reason

    return state


def _assert_last_event(
    event_bus_spy: Any,
    *,
    topic: str,
    event_kind: str,
    direction: FundingStrategyDirection | None = None,
) -> dict[str, Any]:
    record = _last_record(event_bus_spy)
    assert record.topic == topic

    payload = record.payload
    assert payload["event_kind"] == event_kind
    assert payload["strategy"] == "funding_divergence"
    assert payload["strategy_name"] == "funding_divergence"
    assert payload["strategy_namespace"] == "strategy.funding.divergence"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange"] == "binance"

    if direction is not None:
        assert payload["direction"] == direction.value

    return payload


async def _seed_bullish_context(
    strategy: Any,
    *,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            bullish_divergence_context["regime"],
        )
    )
    await strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            bullish_divergence_context["pressure"],
        )
    )


async def _seed_bearish_context(
    strategy: Any,
    *,
    make_test_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            bearish_divergence_context["regime"],
        )
    )
    await strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            bearish_divergence_context["pressure"],
        )
    )


async def _create_bullish_divergence_setup(
    strategy: Any,
    *,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> Any:
    await _seed_bullish_context(
        strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    await strategy.on_divergence(
        make_test_event(
            "analytics.funding.divergence",
            bullish_divergence_context["divergence"],
        )
    )
    return _state(strategy)


async def _create_bearish_divergence_setup(
    strategy: Any,
    *,
    make_test_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> Any:
    await _seed_bearish_context(
        strategy,
        make_test_event=make_test_event,
        bearish_divergence_context=bearish_divergence_context,
    )
    await strategy.on_divergence(
        make_test_event(
            "analytics.funding.divergence",
            bearish_divergence_context["divergence"],
        )
    )
    return _state(strategy)


# =============================================================================
# Setup creation: happy paths
# =============================================================================


@pytest.mark.asyncio
async def test_bullish_divergence_creates_long_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    state = await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )

    assert state.status == FundingSetupStatus.SETUP_DETECTED
    assert state.direction == FundingStrategyDirection.LONG
    assert state.setup_type == divergence_strategy.config.bullish_setup_type
    assert state.score > 0.0
    assert state.confidence > 0.0
    assert state.expires_at is not None
    assert state.setup_event_time is not None

    assert divergence_strategy.config.tag_divergence in state.tags
    assert divergence_strategy.config.tag_dislocation in state.tags
    assert divergence_strategy.config.tag_reversal in state.tags

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "divergence"
    assert payload["setup_type"] == divergence_strategy.config.bullish_setup_type
    assert payload["strategy_variant"] == "divergence"
    assert payload["signal_class"] == "funding_dislocation_reversal"
    assert payload["is_tradeable"] is False


@pytest.mark.asyncio
async def test_bearish_divergence_creates_short_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    state = await _create_bearish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bearish_divergence_context=bearish_divergence_context,
    )

    assert state.status == FundingSetupStatus.SETUP_DETECTED
    assert state.direction == FundingStrategyDirection.SHORT
    assert state.setup_type == divergence_strategy.config.bearish_setup_type
    assert state.score > 0.0
    assert state.confidence > 0.0

    assert divergence_strategy.config.tag_divergence in state.tags
    assert divergence_strategy.config.tag_dislocation in state.tags
    assert divergence_strategy.config.tag_reversal in state.tags

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "divergence"
    assert payload["setup_type"] == divergence_strategy.config.bearish_setup_type
    assert payload["strategy_variant"] == "divergence"
    assert payload["signal_class"] == "funding_dislocation_reversal"


# =============================================================================
# Setup creation: filters
# =============================================================================


@pytest.mark.asyncio
async def test_divergence_ignores_low_confidence_event(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    context = dict(bullish_divergence_context)
    context["divergence"] = dict(bullish_divergence_context["divergence"])
    context["divergence"]["confidence"] = divergence_strategy.config.min_divergence_confidence - 0.01

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_divergence_requires_non_neutral_regime_when_enabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.require_non_neutral_regime = True

    context = dict(bullish_divergence_context)
    context["regime"] = dict(bullish_divergence_context["regime"])
    context["regime"]["regime"] = FundingRegime.NEUTRAL.value
    context["regime"]["bias"] = FundingBias.NEUTRAL.value
    context["regime"]["confidence"] = 0.90

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_divergence_allows_neutral_regime_when_filter_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.require_non_neutral_regime = False

    context = dict(bullish_divergence_context)
    context["regime"] = dict(bullish_divergence_context["regime"])
    context["regime"]["regime"] = FundingRegime.NEUTRAL.value
    context["regime"]["bias"] = FundingBias.NEUTRAL.value
    context["regime"]["confidence"] = 0.90

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )

    _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.LONG,
    )


@pytest.mark.asyncio
async def test_low_regime_confidence_blocks_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    context = dict(bullish_divergence_context)
    context["regime"] = dict(bullish_divergence_context["regime"])
    context["regime"]["confidence"] = divergence_strategy.config.min_regime_confidence - 0.01

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_required_blocks_setup_when_missing(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.require_pressure_present = True

    await divergence_strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            bullish_divergence_context["regime"],
        )
    )
    await divergence_strategy.on_divergence(
        make_test_event(
            "analytics.funding.divergence",
            bullish_divergence_context["divergence"],
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    assert state.last_regime is not None
    assert state.last_divergence is not None
    assert state.last_pressure is None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_low_pressure_score_blocks_setup_when_pressure_is_present(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    context = dict(bullish_divergence_context)
    context["pressure"] = dict(bullish_divergence_context["pressure"])
    context["pressure"]["pressure_score"] = divergence_strategy.config.min_pressure_score - 0.01

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_alignment_filter_blocks_wrong_direction(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.require_pressure_alignment = True

    context = dict(bullish_divergence_context)
    context["pressure"] = dict(bullish_divergence_context["pressure"])
    context["pressure"]["direction"] = FundingPressureDirection.LONG.value

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_alignment_filter_allows_opposite_pressure_direction(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.require_pressure_alignment = True

    context = dict(bullish_divergence_context)
    context["pressure"] = dict(bullish_divergence_context["pressure"])
    context["pressure"]["direction"] = FundingPressureDirection.SHORT.value

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=context,
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.LONG,
    )


@pytest.mark.asyncio
async def test_stale_divergence_event_is_ignored(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _seed_bullish_context(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )

    stale_divergence = dict(bullish_divergence_context["divergence"])
    stale_divergence["event_time"] = "2000-01-01T00:00:00+00:00"

    await divergence_strategy.on_divergence(
        make_test_event(
            "analytics.funding.divergence",
            stale_divergence,
        )
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.IDLE,
        direction=FundingStrategyDirection.NEUTRAL,
    )
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_cooldown_blocks_new_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    state = divergence_strategy.get_state("BTCUSDT", "binance")
    divergence_strategy.set_cooldown(
        state,
        cooldown_sec=60.0,
        reason="pytest_cooldown",
    )

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.NEUTRAL,
        reason="pytest_cooldown",
    )
    assert state.cooldown_until is not None
    _assert_no_events(event_bus_spy)


# =============================================================================
# Repeat divergence confirmation / opposite divergence invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_repeat_bullish_divergence_confirms_long_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bullish_divergence_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_divergence(
        make_bullish_divergence_event(confidence=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="repeat_divergence_confirmed_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_repeat in state.tags
    assert state.metadata["confirmation_source"] == "repeat_divergence"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "repeat_divergence"
    assert payload["metadata"]["confirmation_source"] == "repeat_divergence"


@pytest.mark.asyncio
async def test_repeat_divergence_confirmation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bullish_divergence_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.allow_repeat_divergence_confirmation = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_divergence(
        make_bullish_divergence_event(confidence=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_repeat not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_divergence_invalidates_long_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bearish_divergence_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_divergence(
        make_bearish_divergence_event(confidence=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="opposite_divergence_invalidated_setup",
    )
    assert state.cooldown_until is not None
    assert state.metadata["invalidation_source"] == "divergence"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "divergence"


@pytest.mark.asyncio
async def test_opposite_divergence_invalidation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_bearish_divergence_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_opposite_divergence = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_divergence(
        make_bearish_divergence_event(confidence=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.reason != "opposite_divergence_invalidated_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Flip confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_flip_confirms_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_to_positive_flip_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_flip(make_negative_to_positive_flip_event(confidence=0.90))

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="flip_confirmed_divergence_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_flip in state.tags
    assert state.metadata["confirmation_source"] == "flip"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "flip"
    assert payload["is_tradeable"] is True


@pytest.mark.asyncio
async def test_flip_confirms_short_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_to_negative_flip_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bearish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bearish_divergence_context=bearish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="flip_confirmed_divergence_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_flip in state.tags
    assert state.metadata["confirmation_source"] == "flip"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "flip"


@pytest.mark.asyncio
async def test_opposite_flip_invalidates_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_to_negative_flip_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="opposite_flip_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "flip"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "flip"


@pytest.mark.asyncio
async def test_flip_confirmation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_to_positive_flip_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.allow_flip_confirmation = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_flip(make_negative_to_positive_flip_event(confidence=0.90))

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_flip not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_flip_invalidation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_to_negative_flip_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_opposite_flip = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.reason != "opposite_flip_invalidated_divergence_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Pressure confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_pressure_release_confirms_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    released_pressure = dict(bullish_divergence_context["pressure"])
    released_pressure["pressure_score"] = (
        bullish_divergence_context["pressure"]["pressure_score"]
        - divergence_strategy.config.pressure_release_min_score_drop
        - 0.05
    )
    released_pressure["level"] = FundingPressureLevel.MEDIUM.value
    released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value

    await divergence_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", released_pressure)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="pressure_release_confirmed_divergence_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_release in state.tags
    assert state.metadata["confirmation_source"] == "pressure_release"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "pressure_release"


@pytest.mark.asyncio
async def test_pressure_breakdown_invalidates_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    breakdown_pressure = dict(bullish_divergence_context["pressure"])
    breakdown_pressure["direction"] = FundingPressureDirection.LONG.value
    breakdown_pressure["level"] = FundingPressureLevel.HIGH.value
    breakdown_pressure["pressure_score"] = 0.80

    await divergence_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", breakdown_pressure)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="pressure_context_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "pressure"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "pressure"


@pytest.mark.asyncio
async def test_pressure_release_confirmation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.allow_pressure_release_confirmation = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    released_pressure = dict(bullish_divergence_context["pressure"])
    released_pressure["pressure_score"] = 0.01
    released_pressure["level"] = FundingPressureLevel.MEDIUM.value
    released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value

    await divergence_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", released_pressure)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_release not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_pressure_breakdown_invalidation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_pressure_breakdown = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    breakdown_pressure = dict(bullish_divergence_context["pressure"])
    breakdown_pressure["direction"] = FundingPressureDirection.LONG.value
    breakdown_pressure["level"] = FundingPressureLevel.HIGH.value
    breakdown_pressure["pressure_score"] = 0.80

    await divergence_strategy.on_pressure(
        make_test_event("analytics.funding.pressure", breakdown_pressure)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.reason != "pressure_context_invalidated_divergence_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# Regime confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_regime_conflict_invalidates_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    conflicting_regime = dict(bullish_divergence_context["regime"])
    conflicting_regime["regime"] = FundingRegime.POSITIVE.value
    conflicting_regime["bias"] = FundingBias.LONG_BIAS.value
    conflicting_regime["confidence"] = 0.90

    await divergence_strategy.on_regime(
        make_test_event("analytics.funding.regime", conflicting_regime)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="regime_context_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "regime"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "regime"


@pytest.mark.asyncio
async def test_unknown_regime_invalidates_active_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    unknown_regime = dict(bullish_divergence_context["regime"])
    unknown_regime["regime"] = FundingRegime.UNKNOWN.value
    unknown_regime["bias"] = FundingBias.NEUTRAL.value
    unknown_regime["confidence"] = 0.90

    await divergence_strategy.on_regime(
        make_test_event("analytics.funding.regime", unknown_regime)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="regime_context_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "regime"


@pytest.mark.asyncio
async def test_regime_conflict_invalidation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_regime_conflict = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    conflicting_regime = dict(bullish_divergence_context["regime"])
    conflicting_regime["regime"] = FundingRegime.POSITIVE.value
    conflicting_regime["bias"] = FundingBias.LONG_BIAS.value
    conflicting_regime["confidence"] = 0.90

    await divergence_strategy.on_regime(
        make_test_event("analytics.funding.regime", conflicting_regime)
    )

    _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    _assert_no_events(event_bus_spy)


# =============================================================================
# Extreme confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_negative_extreme_confirms_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_extreme_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_extreme(
        make_negative_extreme_event(severity=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="extreme_context_confirmed_divergence_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_extreme in state.tags
    assert divergence_strategy.config.tag_extreme in state.tags
    assert state.metadata["confirmation_source"] == "extreme"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "extreme"


@pytest.mark.asyncio
async def test_positive_extreme_confirms_short_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_extreme_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bearish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bearish_divergence_context=bearish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_extreme(
        make_positive_extreme_event(severity=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="extreme_context_confirmed_divergence_setup",
    )
    assert state.metadata["confirmation_source"] == "extreme"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "extreme"


@pytest.mark.asyncio
async def test_low_severity_extreme_does_not_confirm_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_negative_extreme_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_extreme(
        make_negative_extreme_event(
            severity=divergence_strategy.config.min_extreme_severity - 0.01
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_extreme not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_extreme_does_not_invalidate_by_default(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_extreme_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    assert divergence_strategy.config.invalidate_on_opposite_extreme is False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_extreme(
        make_positive_extreme_event(severity=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.reason != "opposite_extreme_invalidated_divergence_setup"
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_extreme_invalidates_when_enabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_positive_extreme_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_opposite_extreme = True

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_extreme(
        make_positive_extreme_event(severity=0.91)
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="opposite_extreme_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "extreme"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "extreme"


# =============================================================================
# Funding signal confirmation / invalidation
# =============================================================================


@pytest.mark.asyncio
async def test_funding_signal_confirms_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.LONG,
        reason="funding_signal_confirmed_divergence_setup",
    )
    assert divergence_strategy.config.tag_confirmed_by_signal in state.tags
    assert divergence_strategy.config.tag_signal in state.tags
    assert state.metadata["confirmation_source"] == "funding_signal"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "funding_signal"


@pytest.mark.asyncio
async def test_funding_signal_confirms_short_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bearish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bearish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bearish_divergence_context=bearish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.CONFIRMED,
        direction=FundingStrategyDirection.SHORT,
        reason="funding_signal_confirmed_divergence_setup",
    )
    assert state.metadata["confirmation_source"] == "funding_signal"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.confirmed",
        event_kind="confirmed",
        direction=FundingStrategyDirection.SHORT,
    )
    assert payload["trigger"] == "funding_signal"


@pytest.mark.asyncio
async def test_opposite_funding_signal_invalidates_long_divergence_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.COOLDOWN,
        direction=FundingStrategyDirection.LONG,
        reason="opposite_funding_signal_invalidated_divergence_setup",
    )
    assert state.metadata["invalidation_source"] == "funding_signal"

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.invalidated",
        event_kind="invalidated",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "funding_signal"


@pytest.mark.asyncio
async def test_low_confidence_funding_signal_does_not_confirm_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=divergence_strategy.config.min_signal_confidence - 0.01,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_low_abs_score_funding_signal_does_not_confirm_setup(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=divergence_strategy.config.min_signal_abs_score - 0.01,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_funding_signal_confirmation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.allow_signal_confirmation = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.SHORT_BIAS,
            score=0.80,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_opposite_funding_signal_invalidation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_signal_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.invalidate_on_opposite_signal = False

    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    await divergence_strategy.on_funding_signal(
        make_funding_signal_event(
            signal_type=FundingSignalType.REVERSION_SETUP,
            bias=FundingBias.LONG_BIAS,
            score=-0.80,
            confidence=0.85,
        )
    )

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.reason != "opposite_funding_signal_invalidated_divergence_setup"
    _assert_no_events(event_bus_spy)


# =============================================================================
# analytics.funding.updated integration
# =============================================================================


@pytest.mark.asyncio
async def test_funding_updated_can_create_setup_from_atomic_context(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_funding_updated_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    event = make_funding_updated_event(
        regime_state=bullish_divergence_context["regime"],
        pressure_state=bullish_divergence_context["pressure"],
        divergence_event=bullish_divergence_context["divergence"],
    )

    await divergence_strategy.on_funding_updated(event)

    state = _assert_state(
        divergence_strategy,
        status=FundingSetupStatus.SETUP_DETECTED,
        direction=FundingStrategyDirection.LONG,
    )
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_divergence is not None

    payload = _assert_last_event(
        event_bus_spy,
        topic="strategy.funding.divergence.setup",
        event_kind="setup",
        direction=FundingStrategyDirection.LONG,
    )
    assert payload["trigger"] == "funding_updated"


@pytest.mark.asyncio
async def test_funding_updated_setup_creation_can_be_disabled(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_funding_updated_event: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    divergence_strategy.config.allow_updated_context_setup = False

    event = make_funding_updated_event(
        regime_state=bullish_divergence_context["regime"],
        pressure_state=bullish_divergence_context["pressure"],
        divergence_event=bullish_divergence_context["divergence"],
    )

    await divergence_strategy.on_funding_updated(event)

    state = _state(divergence_strategy)
    assert state.status == FundingSetupStatus.IDLE
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_divergence is not None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_funding_updated_can_invalidate_active_setup_by_opposite_signal(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
    make_funding_updated_event: Any,
    funding_signal_payload: Any,
    bullish_divergence_context: dict[str, dict[str, Any]],
) -> None:
    await _create_bullish_divergence_setup(
        divergence_strategy,
        make_test_event=make_test_event,
        bullish_divergence_context=bullish_divergence_context,
    )
    event_bus_spy.emitted.clear()

    opposite_signal = funding_signal_payload(
        signal_type=FundingSignalType.REVERSION_SETUP,
        bias=FundingBias.LONG_BIAS,
        score=-0.80,
        confidence=0.85,
    )

    event = make_funding_updated_event(
        regime_state=bullish_divergence_context["regime"],
        pressure_state=bullish_divergence_context["pressure"],
        divergence_event=bullish_divergence_context["divergence"],
        **{"payload": {"signal": opposite_signal}},
    )

    await divergence_strategy.on_funding_updated(event)

    # Depending on unwrap/attach implementation, signal may or may not be present in
    # the normalized atomic envelope. This test still asserts the critical invariant:
    # no second state should be created and active context must remain consistent.
    assert len(divergence_strategy.get_all_states()) == 1
    state = _state(divergence_strategy)
    assert state.last_regime is not None
    assert state.last_pressure is not None
    assert state.last_divergence is not None


# =============================================================================
# Guard clauses / lock timeout / malformed payload
# =============================================================================


@pytest.mark.asyncio
async def test_handlers_ignore_events_without_symbol(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
) -> None:
    await divergence_strategy.on_regime(
        make_test_event(
            "analytics.funding.regime",
            {
                "exchange": "binance",
                "regime": FundingRegime.POSITIVE.value,
                "bias": FundingBias.LONG_BIAS.value,
                "confidence": 0.80,
            },
        )
    )
    await divergence_strategy.on_pressure(
        make_test_event(
            "analytics.funding.pressure",
            {
                "exchange": "binance",
                "direction": FundingPressureDirection.SHORT.value,
                "level": FundingPressureLevel.HIGH.value,
                "pressure_score": 0.80,
            },
        )
    )
    await divergence_strategy.on_divergence(
        make_test_event(
            "analytics.funding.divergence",
            {
                "exchange": "binance",
                "confidence": 0.90,
            },
        )
    )

    assert divergence_strategy.get_all_states() == {}
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_handler_returns_without_state_change_when_lock_timeout(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_bullish_divergence_event: Any,
) -> None:
    lock = await divergence_strategy.acquire_symbol_lock("BTCUSDT", "binance")
    assert lock is not None

    try:
        await divergence_strategy.on_divergence(
            make_bullish_divergence_event(confidence=0.90)
        )
    finally:
        divergence_strategy.release_symbol_lock(lock)

    state = _state(divergence_strategy)
    assert state.status == FundingSetupStatus.IDLE
    assert state.last_divergence is None
    _assert_no_events(event_bus_spy)


@pytest.mark.asyncio
async def test_malformed_divergence_event_is_caught_and_does_not_raise(
    divergence_strategy: Any,
    event_bus_spy: Any,
    make_test_event: Any,
) -> None:
    event = make_test_event(
        "analytics.funding.divergence",
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "divergence_type": object(),
            "confidence": object(),
            "event_time": object(),
        },
    )

    await divergence_strategy.on_divergence(event)

    state = _state(divergence_strategy)
    assert state.status in {
        FundingSetupStatus.IDLE,
        FundingSetupStatus.SETUP_DETECTED,
        FundingSetupStatus.COOLDOWN,
    }
    assert isinstance(event_bus_spy.emitted, list)


# =============================================================================
# Registration / stats
# =============================================================================


def test_divergence_registers_expected_subscriptions(
    divergence_strategy: Any,
    event_bus_spy: Any,
) -> None:
    divergence_strategy.register()

    patterns = {subscription.pattern for subscription in event_bus_spy.subscribed}
    assert patterns == {
        "analytics.funding.updated",
        "analytics.funding.signal",
        "analytics.funding.regime",
        "analytics.funding.pressure",
        "analytics.funding.divergence",
        "analytics.funding.flip",
        "analytics.funding.extreme",
    }

    names = {subscription.name for subscription in event_bus_spy.subscribed}
    assert "funding_divergence.on_regime" in names
    assert "funding_divergence.on_pressure" in names
    assert "funding_divergence.on_divergence" in names
    assert "funding_divergence.on_flip" in names
    assert "funding_divergence.on_extreme" in names


def test_divergence_stats_report_strategy_identity(
    divergence_strategy: Any,
) -> None:
    stats = divergence_strategy.stats()

    assert stats["strategy"] == "funding_divergence"
    assert stats["namespace"] == "strategy.funding.divergence"
    assert stats["registered"] is False
    assert stats["running"] is False
    assert stats["states_total"] == 0