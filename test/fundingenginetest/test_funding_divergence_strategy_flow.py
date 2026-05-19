# # test/fundingenginetest/test_funding_divergence_strategy_flow.py
#
# from __future__ import annotations
#
# from typing import Any
#
# import pytest
#
# from analytics.funding.enums import (
#     FundingBias,
#     FundingDivergenceType,
#     FundingExtremeType,
#     FundingFlipType,
#     FundingPressureDirection,
#     FundingPressureLevel,
#     FundingRegime,
#     FundingSignalType,
#     FundingTimeframe,
# )
#
# from strategy.strategies.funding.base import (
#     FundingSetupStatus,
#     FundingStrategyDirection,
# )
#
#
# # =============================================================================
# # Local helpers
# # =============================================================================
#
#
# DEFAULT_SYMBOL = "BTCUSDT"
# DEFAULT_EXCHANGE = "binance"
# DEFAULT_MARKET_TYPE = "usdm_futures"
# DEFAULT_TIMEFRAME = "1h"
# DEFAULT_EXCHANGE_SYMBOL = "BTCUSDT"
#
#
# def _scope(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> dict[str, str]:
#     return {
#         "exchange": exchange,
#         "market_type": market_type,
#         "symbol": symbol,
#         "timeframe": timeframe,
#         "exchange_symbol": exchange_symbol,
#     }
#
#
# def _last_record(event_bus_spy: Any) -> Any:
#     assert event_bus_spy.emitted, "Expected at least one emitted strategy event"
#     return event_bus_spy.emitted[-1]
#
#
# def _records_for_topic(event_bus_spy: Any, topic: str) -> list[Any]:
#     return [record for record in event_bus_spy.emitted if record.topic == topic]
#
#
# def _last_record_for_topic(event_bus_spy: Any, topic: str) -> Any:
#     records = _records_for_topic(event_bus_spy, topic)
#     assert records, f"No records for topic={topic!r}; emitted={[r.topic for r in event_bus_spy.emitted]}"
#     return records[-1]
#
#
# def _assert_no_events(event_bus_spy: Any) -> None:
#     assert event_bus_spy.emitted == []
#
#
# def _state(
#     strategy: Any,
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str | FundingTimeframe = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> Any:
#     return strategy.get_state(
#         symbol,
#         exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#
# def _assert_state_scope(
#     state: Any,
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> None:
#     expected_scope = _scope(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#     assert state.symbol == symbol
#     assert state.exchange == exchange
#     assert state.market_type == market_type
#     assert state.timeframe.value == timeframe
#     assert state.exchange_symbol == exchange_symbol
#     assert state.scope.to_dict() == expected_scope
#     assert state.key == f"{exchange}:{market_type}:{symbol}:{timeframe}"
#     assert state.legacy_key == f"{symbol}:{exchange}"
#
#
# def _assert_state(
#     strategy: Any,
#     *,
#     status: FundingSetupStatus,
#     direction: FundingStrategyDirection | None = None,
#     reason: str | None = None,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> Any:
#     state = _state(
#         strategy,
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#     _assert_state_scope(
#         state,
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#     assert state.status == status
#
#     if direction is not None:
#         assert state.direction == direction
#
#     if reason is not None:
#         assert state.reason == reason
#
#     return state
#
#
# def _assert_event_scope(
#     record: Any,
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> None:
#     expected_scope = _scope(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#     payload = record.payload
#     headers = record.headers
#
#     assert payload["symbol"] == symbol
#     assert payload["exchange"] == exchange
#     assert payload["market_type"] == market_type
#     assert payload["timeframe"] == timeframe
#     assert payload["exchange_symbol"] == exchange_symbol
#     assert payload["scope"] == expected_scope
#
#     assert headers["symbol"] == symbol
#     assert headers["exchange"] == exchange
#     assert headers["market_type"] == market_type
#     assert headers["timeframe"] == timeframe
#     assert headers["exchange_symbol"] == exchange_symbol
#     assert headers["scope"] == expected_scope
#
#     assert payload["state"]["scope"] == expected_scope
#     assert payload["state"]["key"] == f"{exchange}:{market_type}:{symbol}:{timeframe}"
#     assert payload["state"]["legacy_key"] == f"{symbol}:{exchange}"
#
#     assert payload["funding_context"]["scope"] == expected_scope
#     assert payload["analytics_context"]["scope"] == expected_scope
#
#
# def _assert_last_event(
#     event_bus_spy: Any,
#     *,
#     topic: str,
#     event_kind: str,
#     direction: FundingStrategyDirection | None = None,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> dict[str, Any]:
#     record = _last_record_for_topic(event_bus_spy, topic)
#     assert record.topic == topic
#
#     payload = record.payload
#     assert payload["event_kind"] == event_kind
#     assert payload["strategy"] == "funding_divergence"
#     assert payload["strategy_name"] == "funding_divergence"
#     assert payload["strategy_namespace"] == "strategy.funding.divergence"
#     assert payload["strategy_family"] == "funding"
#     assert payload["strategy_variant"] == "divergence"
#     assert payload["signal_class"] == "directional_dislocation"
#
#     _assert_event_scope(
#         record,
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#     if direction is not None:
#         assert payload["direction"] == direction.value
#
#     if event_kind == "confirmed":
#         assert payload["is_tradeable"] is True
#
#     if event_kind == "setup":
#         assert payload.get("is_tradeable") is not True
#
#     return payload
#
#
# def _copy_context(context: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
#     return {key: dict(value) for key, value in context.items()}
#
#
# def _with_scope(
#     payload: dict[str, Any],
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> dict[str, Any]:
#     updated = dict(payload)
#     updated["symbol"] = symbol
#     updated["exchange"] = exchange
#     updated["market_type"] = market_type
#     updated["timeframe"] = timeframe
#     updated["exchange_symbol"] = exchange_symbol
#     metadata = dict(updated.get("metadata") or {})
#     metadata["scope"] = {
#         "exchange": exchange,
#         "market_type": market_type,
#         "symbol": symbol,
#         "timeframe": timeframe,
#     }
#     metadata["exchange_symbol"] = exchange_symbol
#     updated["metadata"] = metadata
#     return updated
#
#
# async def _seed_bullish_context(
#     strategy: Any,
#     *,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await strategy.on_regime(
#         make_test_event(
#             "analytics.funding.regime",
#             bullish_divergence_context["regime"],
#         )
#     )
#     await strategy.on_pressure(
#         make_test_event(
#             "analytics.funding.pressure",
#             bullish_divergence_context["pressure"],
#         )
#     )
#
#
# async def _seed_bearish_context(
#     strategy: Any,
#     *,
#     make_test_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await strategy.on_regime(
#         make_test_event(
#             "analytics.funding.regime",
#             bearish_divergence_context["regime"],
#         )
#     )
#     await strategy.on_pressure(
#         make_test_event(
#             "analytics.funding.pressure",
#             bearish_divergence_context["pressure"],
#         )
#     )
#
#
# async def _create_bullish_divergence_setup(
#     strategy: Any,
#     *,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> Any:
#     await _seed_bullish_context(
#         strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     await strategy.on_divergence(
#         make_test_event(
#             "analytics.funding.divergence",
#             bullish_divergence_context["divergence"],
#         )
#     )
#     return _state(
#         strategy,
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#
# async def _create_bearish_divergence_setup(
#     strategy: Any,
#     *,
#     make_test_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: str = DEFAULT_TIMEFRAME,
#     exchange_symbol: str = DEFAULT_EXCHANGE_SYMBOL,
# ) -> Any:
#     await _seed_bearish_context(
#         strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=bearish_divergence_context,
#     )
#     await strategy.on_divergence(
#         make_test_event(
#             "analytics.funding.divergence",
#             bearish_divergence_context["divergence"],
#         )
#     )
#     return _state(
#         strategy,
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#
# # =============================================================================
# # Setup creation: happy paths
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_bullish_divergence_creates_long_setup_with_full_scope_contract(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     state = await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#
#     _assert_state_scope(state)
#     assert state.status == FundingSetupStatus.SETUP_DETECTED
#     assert state.direction == FundingStrategyDirection.LONG
#     assert state.setup_type == divergence_strategy.config.bullish_setup_type
#     assert state.score > 0.0
#     assert state.confidence > 0.0
#     assert state.expires_at is not None
#     assert state.setup_event_time is not None
#
#     assert divergence_strategy.config.tag_divergence in state.tags
#     assert divergence_strategy.config.tag_dislocation in state.tags
#     assert divergence_strategy.config.tag_reversal in state.tags
#     assert state.metadata["scope"] == state.scope.to_dict()
#     assert state.metadata["funding_context"]["scope"]["market_type"] == DEFAULT_MARKET_TYPE
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.setup",
#         event_kind="setup",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "divergence"
#     assert payload["setup_type"] == divergence_strategy.config.bullish_setup_type
#     assert payload["metadata"]["scope"] == state.scope.to_dict()
#
#
# @pytest.mark.asyncio
# async def test_bearish_divergence_creates_short_setup_with_full_scope_contract(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     state = await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=bearish_divergence_context,
#     )
#
#     _assert_state_scope(state)
#     assert state.status == FundingSetupStatus.SETUP_DETECTED
#     assert state.direction == FundingStrategyDirection.SHORT
#     assert state.setup_type == divergence_strategy.config.bearish_setup_type
#     assert state.score > 0.0
#     assert state.confidence > 0.0
#
#     assert divergence_strategy.config.tag_divergence in state.tags
#     assert divergence_strategy.config.tag_dislocation in state.tags
#     assert divergence_strategy.config.tag_reversal in state.tags
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.setup",
#         event_kind="setup",
#         direction=FundingStrategyDirection.SHORT,
#     )
#     assert payload["trigger"] == "divergence"
#     assert payload["setup_type"] == divergence_strategy.config.bearish_setup_type
#
#
# # =============================================================================
# # Setup creation: filters
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_divergence_ignores_low_confidence_event_without_emitting(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     context = _copy_context(bullish_divergence_context)
#     context["divergence"]["confidence"] = divergence_strategy.config.min_divergence_confidence - 0.01
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     assert state.last_regime is not None
#     assert state.last_pressure is not None
#     assert state.last_divergence is not None
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_divergence_requires_non_neutral_regime_when_enabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.require_non_neutral_regime = True
#
#     context = _copy_context(bullish_divergence_context)
#     context["regime"]["regime"] = FundingRegime.NEUTRAL.value
#     context["regime"]["bias"] = FundingBias.NEUTRAL.value
#     context["regime"]["confidence"] = 0.90
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_divergence_allows_neutral_regime_when_filter_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.require_non_neutral_regime = False
#
#     context = _copy_context(bullish_divergence_context)
#     context["regime"]["regime"] = FundingRegime.NEUTRAL.value
#     context["regime"]["bias"] = FundingBias.NEUTRAL.value
#     context["regime"]["confidence"] = 0.90
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.setup",
#         event_kind="setup",
#         direction=FundingStrategyDirection.LONG,
#     )
#
#
# @pytest.mark.asyncio
# async def test_low_regime_confidence_blocks_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     context = _copy_context(bullish_divergence_context)
#     context["regime"]["confidence"] = divergence_strategy.config.min_regime_confidence - 0.01
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_pressure_required_blocks_setup_when_missing(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.require_pressure_present = True
#
#     await divergence_strategy.on_regime(
#         make_test_event(
#             "analytics.funding.regime",
#             bullish_divergence_context["regime"],
#         )
#     )
#     await divergence_strategy.on_divergence(
#         make_test_event(
#             "analytics.funding.divergence",
#             bullish_divergence_context["divergence"],
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     assert state.last_regime is not None
#     assert state.last_divergence is not None
#     assert state.last_pressure is None
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_low_pressure_score_blocks_setup_when_pressure_is_present(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     context = _copy_context(bullish_divergence_context)
#     context["pressure"]["pressure_score"] = divergence_strategy.config.min_pressure_score - 0.01
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_pressure_alignment_filter_blocks_wrong_direction(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.require_pressure_alignment = True
#
#     context = _copy_context(bullish_divergence_context)
#     context["pressure"]["direction"] = FundingPressureDirection.LONG.value
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_pressure_alignment_filter_allows_opposite_pressure_direction(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.require_pressure_alignment = True
#
#     context = _copy_context(bullish_divergence_context)
#     context["pressure"]["direction"] = FundingPressureDirection.SHORT.value
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=context,
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.setup",
#         event_kind="setup",
#         direction=FundingStrategyDirection.LONG,
#     )
#
#
# @pytest.mark.asyncio
# async def test_stale_divergence_event_is_ignored_without_mutating_setup_status(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _seed_bullish_context(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#
#     stale_divergence = dict(bullish_divergence_context["divergence"])
#     stale_divergence["event_time"] = "2000-01-01T00:00:00+00:00"
#
#     await divergence_strategy.on_divergence(
#         make_test_event(
#             "analytics.funding.divergence",
#             stale_divergence,
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#     )
#     assert state.last_divergence is not None
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_cooldown_blocks_new_divergence_setup_for_same_full_scope_only(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     state = _state(divergence_strategy)
#     divergence_strategy.set_cooldown(
#         state,
#         cooldown_sec=60.0,
#         reason="pytest_cooldown",
#     )
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#
#     same_scope_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.NEUTRAL,
#         reason="pytest_cooldown",
#     )
#     assert same_scope_state.cooldown_until is not None
#     _assert_no_events(event_bus_spy)
#
#     coinm_context = _copy_context(bearish_divergence_context)
#     for key in coinm_context:
#         coinm_context[key] = _with_scope(
#             coinm_context[key],
#             market_type="coinm_futures",
#             exchange_symbol="BTCUSD_PERP",
#         )
#
#     await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=coinm_context,
#         market_type="coinm_futures",
#         exchange_symbol="BTCUSD_PERP",
#     )
#
#     coinm_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.SHORT,
#         market_type="coinm_futures",
#         exchange_symbol="BTCUSD_PERP",
#     )
#     assert coinm_state is not same_scope_state
#     assert divergence_strategy.stats()["states_total"] == 2
#
#
# # =============================================================================
# # Full futures scope isolation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_same_symbol_different_market_types_create_independent_setups(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     usdm_context = _copy_context(bullish_divergence_context)
#     coinm_context = _copy_context(bearish_divergence_context)
#
#     for key in coinm_context:
#         coinm_context[key] = _with_scope(
#             coinm_context[key],
#             market_type="coinm_futures",
#             exchange_symbol="BTCUSD_PERP",
#         )
#
#     usdm_state = await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=usdm_context,
#     )
#     coinm_state = await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=coinm_context,
#         market_type="coinm_futures",
#         exchange_symbol="BTCUSD_PERP",
#     )
#
#     assert usdm_state is not coinm_state
#     assert usdm_state.key == "binance:usdm_futures:BTCUSDT:1h"
#     assert coinm_state.key == "binance:coinm_futures:BTCUSDT:1h"
#     assert usdm_state.direction == FundingStrategyDirection.LONG
#     assert coinm_state.direction == FundingStrategyDirection.SHORT
#     assert divergence_strategy.stats()["states_total"] == 2
#
#     usdm_payload = _records_for_topic(event_bus_spy, "strategy.funding.divergence.setup")[0].payload
#     coinm_payload = _records_for_topic(event_bus_spy, "strategy.funding.divergence.setup")[1].payload
#
#     assert usdm_payload["scope"]["market_type"] == "usdm_futures"
#     assert coinm_payload["scope"]["market_type"] == "coinm_futures"
#     assert coinm_payload["exchange_symbol"] == "BTCUSD_PERP"
#
#
# @pytest.mark.asyncio
# async def test_opposite_divergence_on_different_market_type_does_not_invalidate_existing_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_bearish_divergence_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_divergence(
#         make_bearish_divergence_event(
#             market_type="coinm_futures",
#             exchange_symbol="BTCUSD_PERP",
#             confidence=0.91,
#         )
#     )
#
#     usdm_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     coinm_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#         market_type="coinm_futures",
#         exchange_symbol="BTCUSD_PERP",
#     )
#
#     assert usdm_state is not coinm_state
#     assert usdm_state.reason != "opposite_divergence_invalidated_setup"
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_signal_on_different_timeframe_does_not_invalidate_existing_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             timeframe="4h",
#             score=-0.80,
#             confidence=0.90,
#             signal_origin="divergence",
#             bias=FundingBias.LONG_BIAS,
#             signal_type=FundingSignalType.REVERSION_SETUP,
#         )
#     )
#
#     one_hour_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     four_hour_state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.IDLE,
#         direction=FundingStrategyDirection.NEUTRAL,
#         timeframe="4h",
#     )
#
#     assert one_hour_state is not four_hour_state
#     assert one_hour_state.reason != "opposite_funding_signal_invalidated_divergence_setup"
#     assert len(four_hour_state.recent_signals) == 1
#     _assert_no_events(event_bus_spy)
#
#
# # =============================================================================
# # Repeat divergence confirmation / opposite divergence invalidation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_repeat_bullish_divergence_confirms_long_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_bullish_divergence_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_divergence(
#         make_bullish_divergence_event(confidence=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="repeat_divergence_confirmed_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_repeat in state.tags
#     assert state.metadata["confirmation_source"] == "repeat_divergence"
#     assert state.metadata["scope"] == state.scope.to_dict()
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "repeat_divergence"
#     assert payload["metadata"]["confirmation_source"] == "repeat_divergence"
#
#
# @pytest.mark.asyncio
# async def test_repeat_divergence_confirmation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_bullish_divergence_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.allow_repeat_divergence_confirmation = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_divergence(
#         make_bullish_divergence_event(confidence=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_repeat not in state.tags
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_divergence_invalidates_long_setup_on_same_scope(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_bearish_divergence_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_divergence(
#         make_bearish_divergence_event(confidence=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="opposite_divergence_invalidated_setup",
#     )
#     assert state.cooldown_until is not None
#     assert state.metadata["invalidation_source"] == "divergence"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "divergence"
#
#
# @pytest.mark.asyncio
# async def test_opposite_divergence_invalidation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_bearish_divergence_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_opposite_divergence = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_divergence(
#         make_bearish_divergence_event(confidence=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.reason != "opposite_divergence_invalidated_setup"
#     _assert_no_events(event_bus_spy)
#
#
# # =============================================================================
# # Flip confirmation / invalidation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_flip_confirms_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_negative_to_positive_flip_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_flip(make_negative_to_positive_flip_event(confidence=0.90))
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="flip_confirmed_divergence_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_flip in state.tags
#     assert state.metadata["confirmation_source"] == "flip"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "flip"
#
#
# @pytest.mark.asyncio
# async def test_flip_confirms_short_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_to_negative_flip_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=bearish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.SHORT,
#         reason="flip_confirmed_divergence_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_flip in state.tags
#     assert state.metadata["confirmation_source"] == "flip"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.SHORT,
#     )
#     assert payload["trigger"] == "flip"
#
#
# @pytest.mark.asyncio
# async def test_opposite_flip_invalidates_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_to_negative_flip_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="opposite_flip_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "flip"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "flip"
#
#
# @pytest.mark.asyncio
# async def test_flip_confirmation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_negative_to_positive_flip_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.allow_flip_confirmation = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_flip(make_negative_to_positive_flip_event(confidence=0.90))
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_flip not in state.tags
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_flip_invalidation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_to_negative_flip_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_opposite_flip = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_flip(make_positive_to_negative_flip_event(confidence=0.90))
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.reason != "opposite_flip_invalidated_divergence_setup"
#     _assert_no_events(event_bus_spy)
#
#
# # =============================================================================
# # Pressure confirmation / invalidation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_pressure_release_confirms_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     released_pressure = dict(bullish_divergence_context["pressure"])
#     released_pressure["pressure_score"] = (
#         bullish_divergence_context["pressure"]["pressure_score"]
#         - divergence_strategy.config.pressure_release_min_score_drop
#         - 0.05
#     )
#     released_pressure["level"] = FundingPressureLevel.MODERATE.value
#     released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value
#
#     await divergence_strategy.on_pressure(
#         make_test_event("analytics.funding.pressure", released_pressure)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="pressure_release_confirmed_divergence_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_release in state.tags
#     assert state.metadata["confirmation_source"] == "pressure_release"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "pressure_release"
#
#
# @pytest.mark.asyncio
# async def test_pressure_breakdown_invalidates_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     breakdown_pressure = dict(bullish_divergence_context["pressure"])
#     breakdown_pressure["direction"] = FundingPressureDirection.LONG.value
#     breakdown_pressure["level"] = FundingPressureLevel.HIGH.value
#     breakdown_pressure["pressure_score"] = 0.80
#
#     await divergence_strategy.on_pressure(
#         make_test_event("analytics.funding.pressure", breakdown_pressure)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="pressure_context_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "pressure"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "pressure"
#
#
# @pytest.mark.asyncio
# async def test_pressure_release_confirmation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.allow_pressure_release_confirmation = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     released_pressure = dict(bullish_divergence_context["pressure"])
#     released_pressure["pressure_score"] = 0.01
#     released_pressure["level"] = FundingPressureLevel.MODERATE.value
#     released_pressure["direction"] = FundingPressureDirection.NEUTRAL.value
#
#     await divergence_strategy.on_pressure(
#         make_test_event("analytics.funding.pressure", released_pressure)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_release not in state.tags
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_pressure_breakdown_invalidation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_pressure_breakdown = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     breakdown_pressure = dict(bullish_divergence_context["pressure"])
#     breakdown_pressure["direction"] = FundingPressureDirection.LONG.value
#     breakdown_pressure["level"] = FundingPressureLevel.HIGH.value
#     breakdown_pressure["pressure_score"] = 0.80
#
#     await divergence_strategy.on_pressure(
#         make_test_event("analytics.funding.pressure", breakdown_pressure)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.reason != "pressure_context_invalidated_divergence_setup"
#     _assert_no_events(event_bus_spy)
#
#
# # =============================================================================
# # Regime confirmation / invalidation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_regime_conflict_invalidates_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     conflicting_regime = dict(bullish_divergence_context["regime"])
#     conflicting_regime["regime"] = FundingRegime.POSITIVE.value
#     conflicting_regime["bias"] = FundingBias.LONG_BIAS.value
#     conflicting_regime["confidence"] = 0.90
#
#     await divergence_strategy.on_regime(
#         make_test_event("analytics.funding.regime", conflicting_regime)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="regime_context_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "regime"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "regime"
#
#
# @pytest.mark.asyncio
# async def test_unknown_regime_invalidates_active_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     unknown_regime = dict(bullish_divergence_context["regime"])
#     unknown_regime["regime"] = FundingRegime.UNKNOWN.value
#     unknown_regime["bias"] = FundingBias.NEUTRAL.value
#     unknown_regime["confidence"] = 0.90
#
#     await divergence_strategy.on_regime(
#         make_test_event("analytics.funding.regime", unknown_regime)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="regime_context_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "regime"
#
#
# @pytest.mark.asyncio
# async def test_regime_conflict_invalidation_can_be_disabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_regime_conflict = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     conflicting_regime = dict(bullish_divergence_context["regime"])
#     conflicting_regime["regime"] = FundingRegime.POSITIVE.value
#     conflicting_regime["bias"] = FundingBias.LONG_BIAS.value
#     conflicting_regime["confidence"] = 0.90
#
#     await divergence_strategy.on_regime(
#         make_test_event("analytics.funding.regime", conflicting_regime)
#     )
#
#     _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     _assert_no_events(event_bus_spy)
#
#
# # =============================================================================
# # Extreme confirmation / invalidation
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_negative_extreme_confirms_long_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_negative_extreme_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_extreme(
#         make_negative_extreme_event(severity=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="extreme_context_confirmed_divergence_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_extreme in state.tags
#     assert divergence_strategy.config.tag_extreme in state.tags
#     assert state.metadata["confirmation_source"] == "extreme"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "extreme"
#
#
# @pytest.mark.asyncio
# async def test_positive_extreme_confirms_short_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_extreme_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=bearish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_extreme(
#         make_positive_extreme_event(severity=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.SHORT,
#         reason="extreme_context_confirmed_divergence_setup",
#     )
#     assert state.metadata["confirmation_source"] == "extreme"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.SHORT,
#     )
#     assert payload["trigger"] == "extreme"
#
#
# @pytest.mark.asyncio
# async def test_low_severity_extreme_does_not_confirm_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_negative_extreme_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_extreme(
#         make_negative_extreme_event(
#             severity=divergence_strategy.config.min_extreme_severity - 0.01
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_extreme not in state.tags
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_extreme_does_not_invalidate_by_default(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_extreme_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     assert divergence_strategy.config.invalidate_on_opposite_extreme is False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_extreme(
#         make_positive_extreme_event(severity=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.reason != "opposite_extreme_invalidated_divergence_setup"
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_extreme_invalidates_when_enabled(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_positive_extreme_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_opposite_extreme = True
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_extreme(
#         make_positive_extreme_event(severity=0.91)
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="opposite_extreme_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "extreme"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "extreme"
#
#
# # =============================================================================
# # Funding signal confirmation / invalidation / signal origin
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_funding_signal_confirms_long_divergence_setup_and_tracks_origin(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.SHORT_BIAS,
#             score=0.80,
#             confidence=0.85,
#             signal_origin="pressure_reversion",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="funding_signal_confirmed_divergence_setup",
#     )
#     assert divergence_strategy.config.tag_confirmed_by_signal in state.tags
#     assert divergence_strategy.config.tag_signal in state.tags
#     assert state.metadata["confirmation_source"] == "funding_signal"
#     assert state.metadata["signal_origin"] == "pressure_reversion"
#     assert "pressure_reversion" in state.last_signals_by_origin
#     assert len(state.recent_signals) == 1
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_signal"
#     assert payload["metadata"]["signal_origin"] == "pressure_reversion"
#
#
# @pytest.mark.asyncio
# async def test_funding_signal_confirms_short_divergence_setup(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bearish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bearish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bearish_divergence_context=bearish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.LONG_BIAS,
#             score=-0.80,
#             confidence=0.85,
#             signal_origin="divergence",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.SHORT,
#         reason="funding_signal_confirmed_divergence_setup",
#     )
#     assert state.metadata["confirmation_source"] == "funding_signal"
#     assert state.metadata["signal_origin"] == "divergence"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.SHORT,
#     )
#     assert payload["trigger"] == "funding_signal"
#
#
# @pytest.mark.asyncio
# async def test_opposite_funding_signal_invalidates_long_divergence_setup_on_same_scope(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.LONG_BIAS,
#             score=-0.80,
#             confidence=0.85,
#             signal_origin="divergence",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="opposite_funding_signal_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "funding_signal"
#     assert state.metadata["signal_origin"] == "divergence"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_signal"
#
#
# @pytest.mark.asyncio
# async def test_low_confidence_funding_signal_does_not_confirm_setup_but_is_stored(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.SHORT_BIAS,
#             score=0.80,
#             confidence=divergence_strategy.config.min_signal_confidence - 0.01,
#             signal_origin="pressure_reversion",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
#     assert "pressure_reversion" in state.last_signals_by_origin
#     assert len(state.recent_signals) == 1
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_low_abs_score_funding_signal_does_not_confirm_setup_but_is_stored(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.SHORT_BIAS,
#             score=divergence_strategy.config.min_signal_abs_score - 0.01,
#             confidence=0.85,
#             signal_origin="pressure_reversion",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
#     assert "pressure_reversion" in state.last_signals_by_origin
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_funding_signal_confirmation_can_be_disabled_but_signal_history_is_kept(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.allow_signal_confirmation = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.SHORT_BIAS,
#             score=0.80,
#             confidence=0.85,
#             signal_origin="pressure_reversion",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert divergence_strategy.config.tag_confirmed_by_signal not in state.tags
#     assert "pressure_reversion" in state.last_signals_by_origin
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_opposite_funding_signal_invalidation_can_be_disabled_but_signal_history_is_kept(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.invalidate_on_opposite_signal = False
#
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_type=FundingSignalType.REVERSION_SETUP,
#             bias=FundingBias.LONG_BIAS,
#             score=-0.80,
#             confidence=0.85,
#             signal_origin="divergence",
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.reason != "opposite_funding_signal_invalidated_divergence_setup"
#     assert "divergence" in state.last_signals_by_origin
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_signal_origin_weighting_prefers_better_recent_signal_during_atomic_update(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     make_funding_updated_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     # Stored but too weak / lower origin weight.
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_origin="pressure",
#             score=0.40,
#             confidence=0.50,
#             bias=FundingBias.SHORT_BIAS,
#             signal_type=FundingSignalType.SQUEEZE_WARNING,
#         )
#     )
#     event_bus_spy.emitted.clear()
#
#     # Stronger preferred confirmation origin.
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_origin="divergence",
#             score=0.83,
#             confidence=0.92,
#             bias=FundingBias.SHORT_BIAS,
#             signal_type=FundingSignalType.DIVERGENCE_DETECTED,
#         )
#     )
#     event_bus_spy.emitted.clear()
#
#     event = make_funding_updated_event(
#         regime_state=bullish_divergence_context["regime"],
#         pressure_state=bullish_divergence_context["pressure"],
#         divergence_event=bullish_divergence_context["divergence"],
#     )
#
#     await divergence_strategy.on_funding_updated(event)
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="atomic_update_signal_confirmed_divergence_setup",
#     )
#     assert state.metadata["signal_origin"] == "divergence"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_updated"
#     assert payload["metadata"]["signal_origin"] == "divergence"
#
#
# # =============================================================================
# # Divergence type scoring
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_liquidation_divergence_scores_higher_than_price_only_divergence_with_same_context(
#     make_test_event: Any,
#     event_bus_spy: Any,
#     scheduler_spy: Any,
#     parquet_storage_spy: Any,
#     divergence_config: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     from strategy.strategies.funding.funding_divergence_strategy import FundingDivergenceStrategy
#
#     price_strategy = FundingDivergenceStrategy(
#         event_bus=event_bus_spy,
#         scheduler=scheduler_spy,
#         parquet_storage=parquet_storage_spy,
#         config=divergence_config,
#     )
#
#     price_context = _copy_context(bullish_divergence_context)
#     price_context["divergence"]["divergence_type"] = FundingDivergenceType.PRICE_UP_FUNDING_DOWN.value
#     price_context["divergence"]["confidence"] = 0.80
#
#     price_state = await _create_bullish_divergence_setup(
#         price_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=price_context,
#     )
#     price_score = price_state.score
#
#     event_bus_spy.emitted.clear()
#
#     liquidation_strategy = FundingDivergenceStrategy(
#         event_bus=event_bus_spy,
#         scheduler=scheduler_spy,
#         parquet_storage=parquet_storage_spy,
#         config=divergence_config,
#     )
#
#     liquidation_context = _copy_context(bullish_divergence_context)
#     liquidation_context["divergence"]["divergence_type"] = (
#         FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING.value
#     )
#     liquidation_context["divergence"]["confidence"] = 0.80
#     liquidation_context["divergence"]["short_liquidations"] = 250_000.0
#
#     liquidation_state = await _create_bullish_divergence_setup(
#         liquidation_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=liquidation_context,
#     )
#
#     assert liquidation_state.score > price_score
#     assert liquidation_strategy.config.tag_liquidation in liquidation_state.tags
#     assert liquidation_state.metadata["divergence_type_bonus"] > price_state.metadata["divergence_type_bonus"]
#
#
# # =============================================================================
# # analytics.funding.updated integration
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_funding_updated_can_create_setup_from_atomic_context_with_full_scope(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_funding_updated_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     event = make_funding_updated_event(
#         regime_state=bullish_divergence_context["regime"],
#         pressure_state=bullish_divergence_context["pressure"],
#         divergence_event=bullish_divergence_context["divergence"],
#     )
#
#     await divergence_strategy.on_funding_updated(event)
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.SETUP_DETECTED,
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert state.last_regime is not None
#     assert state.last_pressure is not None
#     assert state.last_divergence is not None
#     assert state.last_funding_updated_payload is not None
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.setup",
#         event_kind="setup",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_updated"
#
#
# @pytest.mark.asyncio
# async def test_funding_updated_setup_creation_can_be_disabled_but_context_is_kept(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_funding_updated_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     divergence_strategy.config.allow_updated_context_setup = False
#
#     event = make_funding_updated_event(
#         regime_state=bullish_divergence_context["regime"],
#         pressure_state=bullish_divergence_context["pressure"],
#         divergence_event=bullish_divergence_context["divergence"],
#     )
#
#     await divergence_strategy.on_funding_updated(event)
#
#     state = _state(divergence_strategy)
#     assert state.status == FundingSetupStatus.IDLE
#     assert state.last_regime is not None
#     assert state.last_pressure is not None
#     assert state.last_divergence is not None
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_funding_updated_can_confirm_active_setup_from_recent_aligned_signal(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     make_funding_updated_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_origin="pressure_reversion",
#             score=0.86,
#             confidence=0.91,
#             bias=FundingBias.SHORT_BIAS,
#             signal_type=FundingSignalType.REVERSION_SETUP,
#         )
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_updated(
#         make_funding_updated_event(
#             regime_state=bullish_divergence_context["regime"],
#             pressure_state=bullish_divergence_context["pressure"],
#             divergence_event=bullish_divergence_context["divergence"],
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.CONFIRMED,
#         direction=FundingStrategyDirection.LONG,
#         reason="atomic_update_signal_confirmed_divergence_setup",
#     )
#     assert state.metadata["confirmation_source"] == "funding_updated.signal"
#     assert state.metadata["signal_origin"] == "pressure_reversion"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.confirmed",
#         event_kind="confirmed",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_updated"
#
#
# @pytest.mark.asyncio
# async def test_funding_updated_can_invalidate_active_setup_from_recent_opposite_signal(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
#     make_funding_signal_event: Any,
#     make_funding_updated_event: Any,
#     bullish_divergence_context: dict[str, dict[str, Any]],
# ) -> None:
#     await _create_bullish_divergence_setup(
#         divergence_strategy,
#         make_test_event=make_test_event,
#         bullish_divergence_context=bullish_divergence_context,
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_signal(
#         make_funding_signal_event(
#             signal_origin="divergence",
#             score=-0.86,
#             confidence=0.91,
#             bias=FundingBias.LONG_BIAS,
#             signal_type=FundingSignalType.DIVERGENCE_DETECTED,
#         )
#     )
#     event_bus_spy.emitted.clear()
#
#     await divergence_strategy.on_funding_updated(
#         make_funding_updated_event(
#             regime_state=bullish_divergence_context["regime"],
#             pressure_state=bullish_divergence_context["pressure"],
#             divergence_event=bullish_divergence_context["divergence"],
#         )
#     )
#
#     state = _assert_state(
#         divergence_strategy,
#         status=FundingSetupStatus.COOLDOWN,
#         direction=FundingStrategyDirection.LONG,
#         reason="atomic_update_opposite_signal_invalidated_divergence_setup",
#     )
#     assert state.metadata["invalidation_source"] == "funding_updated.signal"
#     assert state.metadata["signal_origin"] == "divergence"
#
#     payload = _assert_last_event(
#         event_bus_spy,
#         topic="strategy.funding.divergence.invalidated",
#         event_kind="invalidated",
#         direction=FundingStrategyDirection.LONG,
#     )
#     assert payload["trigger"] == "funding_updated"
#
#
# # =============================================================================
# # Guard clauses / lock timeout / malformed payload
# # =============================================================================
#
#
# @pytest.mark.asyncio
# async def test_handlers_ignore_events_without_symbol(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
# ) -> None:
#     await divergence_strategy.on_regime(
#         make_test_event(
#             "analytics.funding.regime",
#             {
#                 "exchange": "binance",
#                 "market_type": "usdm_futures",
#                 "timeframe": "1h",
#                 "regime": FundingRegime.POSITIVE.value,
#                 "bias": FundingBias.LONG_BIAS.value,
#                 "confidence": 0.80,
#             },
#         )
#     )
#     await divergence_strategy.on_pressure(
#         make_test_event(
#             "analytics.funding.pressure",
#             {
#                 "exchange": "binance",
#                 "market_type": "usdm_futures",
#                 "timeframe": "1h",
#                 "direction": FundingPressureDirection.SHORT.value,
#                 "level": FundingPressureLevel.HIGH.value,
#                 "pressure_score": 0.80,
#             },
#         )
#     )
#     await divergence_strategy.on_divergence(
#         make_test_event(
#             "analytics.funding.divergence",
#             {
#                 "exchange": "binance",
#                 "market_type": "usdm_futures",
#                 "timeframe": "1h",
#                 "confidence": 0.90,
#             },
#         )
#     )
#
#     assert divergence_strategy.get_all_states() == {}
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_handler_returns_without_state_change_when_lock_timeout(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_bullish_divergence_event: Any,
# ) -> None:
#     lock = await divergence_strategy.acquire_symbol_lock("BTCUSDT", "binance")
#     assert lock is not None
#
#     try:
#         await divergence_strategy.on_divergence(
#             make_bullish_divergence_event(confidence=0.90)
#         )
#     finally:
#         divergence_strategy.release_symbol_lock(lock)
#
#     state = _state(divergence_strategy)
#     assert state.status == FundingSetupStatus.IDLE
#     assert state.last_divergence is None
#     _assert_no_events(event_bus_spy)
#
#
# @pytest.mark.asyncio
# async def test_malformed_divergence_event_is_caught_and_does_not_raise(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
#     make_test_event: Any,
# ) -> None:
#     event = make_test_event(
#         "analytics.funding.divergence",
#         {
#             "symbol": "BTCUSDT",
#             "exchange": "binance",
#             "market_type": "usdm_futures",
#             "timeframe": "1h",
#             "exchange_symbol": "BTCUSDT",
#             "divergence_type": object(),
#             "confidence": object(),
#             "event_time": object(),
#         },
#     )
#
#     await divergence_strategy.on_divergence(event)
#
#     state = _state(divergence_strategy)
#     assert state.status in {
#         FundingSetupStatus.IDLE,
#         FundingSetupStatus.SETUP_DETECTED,
#         FundingSetupStatus.COOLDOWN,
#     }
#     assert isinstance(event_bus_spy.emitted, list)
#
#
# # =============================================================================
# # Registration / stats
# # =============================================================================
#
#
# def test_divergence_registers_expected_subscriptions(
#     divergence_strategy: Any,
#     event_bus_spy: Any,
# ) -> None:
#     divergence_strategy.register()
#
#     patterns = {subscription.pattern for subscription in event_bus_spy.subscribed}
#     assert patterns == {
#         "analytics.funding.updated",
#         "analytics.funding.signal",
#         "analytics.funding.regime",
#         "analytics.funding.pressure",
#         "analytics.funding.divergence",
#         "analytics.funding.flip",
#         "analytics.funding.extreme",
#     }
#
#     names = {subscription.name for subscription in event_bus_spy.subscribed}
#     assert "funding_divergence.on_regime" in names
#     assert "funding_divergence.on_pressure" in names
#     assert "funding_divergence.on_divergence" in names
#     assert "funding_divergence.on_flip" in names
#     assert "funding_divergence.on_extreme" in names
#
#
# def test_divergence_stats_report_strategy_identity(
#     divergence_strategy: Any,
# ) -> None:
#     stats = divergence_strategy.stats()
#
#     assert stats["strategy"] == "funding_divergence"
#     assert stats["namespace"] == "strategy.funding.divergence"
#     assert stats["registered"] is False
#     assert stats["running"] is False
#     assert stats["states_total"] == 0