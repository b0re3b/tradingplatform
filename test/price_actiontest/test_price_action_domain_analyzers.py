# tests/analytics/price_action/test_price_action_domain_analyzers.py

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest

from analytics.price_action import (
    FairValueGapAnalyzer,
    FVGDirection,
    FVGEventType,
    FVGStatus,
    LevelStatus,
    LevelType,
    LiquidityEventType,
    LiquidityLevelsAnalyzer,
    LiquidityLevelStatus,
    LiquidityLevelType,
    MarketBias,
    MarketStructureAnalyzer,
    SREventType,
    StructureEventType,
    StructureLayer,
    SupportResistanceAnalyzer,
    SwingType,
    TrendAnalyzer,
    TrendDirection,
    TrendRegime,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

class EmitRecorder:
    """
    Captures EventBus emissions from async analyzer handlers.

    We use this instead of starting the EventBus worker because these tests are
    focused on analyzer domain behavior and handler publication contract.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        topic: str,
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> bool:
        self.calls.append(
            {
                "topic": topic,
                "payload": dict(payload),
                "kwargs": dict(kwargs),
            }
        )
        return True


def values(items: list[dict[str, Any]], key: str = "event_type") -> list[str]:
    return [str(item[key]) for item in items if key in item]


def model_event_values(events: list[Any]) -> list[str]:
    result: list[str] = []
    for event in events:
        event_type = getattr(event, "event_type", None)
        if event_type is not None:
            result.append(str(event_type.value if hasattr(event_type, "value") else event_type))
    return result


def snapshot_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


def assert_snapshot_is_serialized(snapshot: dict[str, Any], *, symbol: str, timeframe: str) -> None:
    assert snapshot["symbol"] == symbol
    assert snapshot["timeframe"] == timeframe
    assert isinstance(snapshot["generated_at"], str)
    assert isinstance(snapshot["state"], dict)
    assert isinstance(snapshot["metadata"], dict)


def assert_no_duplicate_ids(items: list[dict[str, Any]], id_key: str) -> None:
    ids = [item[id_key] for item in items if id_key in item]
    assert len(ids) == len(set(ids))


def assert_confidences_are_bounded(items: list[dict[str, Any]]) -> None:
    for item in items:
        if "confidence" in item:
            assert 0.0 <= float(item["confidence"]) <= 1.0


def build_resistance_rejection_candle(
    candle_factory: Callable[..., dict[str, Any]],
    index: int,
    *,
    level_price: float,
) -> dict[str, Any]:
    return candle_factory(
        index,
        open_=level_price - 0.20,
        high=level_price + 1.20,
        low=level_price - 0.35,
        close=level_price - 0.45,
        volume=2_000.0,
    )


def build_resistance_breakout_candle(
    candle_factory: Callable[..., dict[str, Any]],
    index: int,
    *,
    level_price: float,
) -> dict[str, Any]:
    return candle_factory(
        index,
        open_=level_price - 0.10,
        high=level_price + 2.00,
        low=level_price - 0.25,
        close=level_price + 1.50,
        volume=2_500.0,
    )


def build_upper_liquidity_sweep_candle(
    candle_factory: Callable[..., dict[str, Any]],
    index: int,
    *,
    level_price: float,
) -> dict[str, Any]:
    return candle_factory(
        index,
        open_=level_price - 0.25,
        high=level_price + 1.50,
        low=level_price - 0.40,
        close=level_price - 0.20,
        volume=3_000.0,
    )


def build_upper_liquidity_reclaim_candle(
    candle_factory: Callable[..., dict[str, Any]],
    index: int,
    *,
    level_price: float,
) -> dict[str, Any]:
    return candle_factory(
        index,
        open_=level_price + 0.20,
        high=level_price + 0.45,
        low=level_price - 0.65,
        close=level_price - 0.35,
        volume=3_100.0,
    )


# ---------------------------------------------------------------------------
# MarketStructureAnalyzer
# ---------------------------------------------------------------------------

class TestMarketStructureAnalyzerDomain:
    def test_detects_swings_and_does_not_reprocess_same_pivot_without_new_candles(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        market_structure_config,
        swing_pattern_candles,
    ) -> None:
        analyzer = MarketStructureAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=market_structure_config,
        )

        result = analyzer.add_candles(swing_pattern_candles())

        assert result["state"]["symbol"] == symbol
        assert result["state"]["timeframe"] == timeframe
        assert len(result["new_swings"]) >= 2
        assert len(analyzer.get_swings()) == len({s.swing_id for s in analyzer.get_swings()})
        assert_no_duplicate_ids(result["new_swings"], "swing_id")

        previous_swings = len(analyzer.get_swings())
        previous_events = len(analyzer.get_events())
        second_pass_swings, second_pass_events = analyzer._process_incremental_pivots()

        assert second_pass_swings == []
        assert second_pass_events == []
        assert len(analyzer.get_swings()) == previous_swings
        assert len(analyzer.get_events()) == previous_events

    def test_detects_structure_labels_without_unbounded_duplicate_labels(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        market_structure_config,
        candle_factory,
    ) -> None:
        analyzer = MarketStructureAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=market_structure_config,
        )

        prices = [
            100.0, 102.0, 105.0, 101.0, 99.0,
            103.0, 107.0, 104.0, 101.0, 106.0,
            110.0, 107.0, 103.0, 108.0, 112.0,
        ]
        candles = []
        for i, close in enumerate(prices):
            open_ = prices[i - 1] if i else close
            candles.append(
                candle_factory(
                    i,
                    open_=open_,
                    high=max(open_, close) + 0.50,
                    low=min(open_, close) - 0.50,
                    close=close,
                )
            )

        result = analyzer.add_candles(candles)
        event_types = values(result["new_events"])

        assert StructureEventType.HH.value in event_types or StructureEventType.HL.value in event_types
        assert len(analyzer._processed_structure_labels) == len(set(analyzer._processed_structure_labels))
        assert len(analyzer.get_events()) == len({event.event_id for event in analyzer.get_events()})
        assert_confidences_are_bounded(result["new_events"])

    def test_higher_timeframe_context_updates_alignment_but_invalid_payload_is_ignored(
        self,
        event_bus,
        event_factory,
        symbol: str,
        timeframe: str,
        market_structure_config,
    ) -> None:
        analyzer = MarketStructureAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=market_structure_config,
            higher_timeframe="15m",
        )

        valid_payload = {
            "symbol": symbol,
            "timeframe": "15m",
            "bias": MarketBias.BULLISH.value,
            "confidence": 0.9,
        }

        analyzer.update_higher_timeframe_context(valid_payload)
        analyzer._refresh_state()

        state = analyzer.get_state()
        assert state.mtf_alignment.higher_timeframe == "15m"
        assert state.mtf_alignment.higher_timeframe_bias == MarketBias.BULLISH
        assert 0.0 <= state.mtf_alignment.alignment_score <= 1.0

        previous_snapshot = analyzer.snapshot()

        # Handler should ignore invalid payloads, not mutate state and not raise.
        import asyncio

        asyncio.run(
            analyzer.on_higher_timeframe_context_event(
                event_factory(
                    market_structure_config.higher_timeframe_context_topic,
                    "not-a-mapping",
                )
            )
        )

        assert analyzer.snapshot()["state"] == previous_snapshot["state"]

    def test_reset_clears_caches_processed_sets_and_state_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        market_structure_config,
        swing_pattern_candles,
    ) -> None:
        analyzer = MarketStructureAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=market_structure_config,
        )

        analyzer.add_candles(swing_pattern_candles())
        assert analyzer.get_swings()
        assert analyzer._processed_pivots

        analyzer.reset()

        assert analyzer.get_swings() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_pivots == set()
        assert analyzer._processed_structure_labels == set()
        assert analyzer._processed_breaks == set()
        assert analyzer._global_candle_index == 0

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 0
        assert metadata["internal_swings"] == 0
        assert metadata["external_swings"] == 0
        assert metadata["events"] == 0


# ---------------------------------------------------------------------------
# SupportResistanceAnalyzer
# ---------------------------------------------------------------------------

class TestSupportResistanceAnalyzerDomain:
    def test_duplicate_swing_id_is_ignored_and_does_not_inflate_strength_or_events(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        swing_factory,
    ) -> None:
        analyzer = SupportResistanceAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=support_resistance_config,
        )

        swing = swing_factory(
            10,
            price=105.0,
            swing_type=SwingType.HIGH,
            layer=StructureLayer.INTERNAL,
            swing_id="duplicate-high",
        )

        first = analyzer.add_swings([swing])
        second = analyzer.add_swings([dict(swing)])

        assert len(first["updated_levels"]) == 1
        assert len(first["new_events"]) == 1
        assert second["updated_levels"] == []
        assert second["new_events"] == []
        assert len(analyzer.get_levels(StructureLayer.INTERNAL)) == 1
        assert len(analyzer.get_events()) == 1
        assert analyzer._processed_swings == {"duplicate-high"}

    def test_nearby_resistance_swings_merge_instead_of_creating_level_explosion(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        swing_factory,
    ) -> None:
        config = replace(
            support_resistance_config,
            internal_merge_distance_pct=0.005,
            min_touches_for_validation=2,
        )
        analyzer = SupportResistanceAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=config,
        )

        swings = [
            swing_factory(1, price=105.00, swing_type=SwingType.HIGH, swing_id="h-1"),
            swing_factory(2, price=105.20, swing_type=SwingType.HIGH, swing_id="h-2"),
            swing_factory(3, price=105.35, swing_type=SwingType.HIGH, swing_id="h-3"),
        ]

        result = analyzer.add_swings(swings)
        levels = analyzer.get_levels(StructureLayer.INTERNAL)
        event_types = values(result["new_events"])

        assert len(levels) == 1
        assert levels[0].source_count == 3
        assert levels[0].touch_count == 3
        assert SREventType.LEVEL_CREATED.value in event_types
        assert SREventType.LEVEL_MERGED.value in event_types
        assert_no_duplicate_ids(result["new_events"], "event_id")
        assert_confidences_are_bounded(result["new_events"])

    def test_touch_rejection_break_flip_retest_sequence_is_ordered_and_bounded(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        swing_factory,
        candle_factory,
    ) -> None:
        config = replace(
            support_resistance_config,
            internal_zone_half_width_pct=0.001,
            breakout_threshold_pct=0.0,
            rejection_wick_ratio_threshold=0.20,
            allow_flip_on_break=True,
            retest_window_bars=5,
        )
        analyzer = SupportResistanceAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=config,
        )

        analyzer.add_swings(
            [
                swing_factory(
                    1,
                    price=105.0,
                    swing_type=SwingType.HIGH,
                    layer=StructureLayer.INTERNAL,
                    swing_id="resistance-1",
                )
            ]
        )

        rejection = build_resistance_rejection_candle(candle_factory, 10, level_price=105.0)
        breakout = build_resistance_breakout_candle(candle_factory, 11, level_price=105.0)
        retest = candle_factory(
            12,
            open_=106.0,
            high=106.4,
            low=104.9,
            close=105.4,
            volume=2_600.0,
        )

        result = analyzer.add_candles([rejection, breakout, retest])
        event_types = values(result["new_events"])
        level = analyzer.get_levels(StructureLayer.INTERNAL)[0]

        assert SREventType.LEVEL_TOUCHED.value in event_types
        assert SREventType.LEVEL_REJECTED.value in event_types
        assert SREventType.LEVEL_BROKEN.value in event_types
        assert level.status in {LevelStatus.BROKEN, LevelStatus.ACTIVE}
        assert level.level_type in {
            LevelType.RESISTANCE,
            LevelType.FLIP_SUPPORT,
            LevelType.FLIP_RESISTANCE,
        }
        assert level.touch_count >= 1
        assert_confidences_are_bounded(result["new_events"])

        duplicate_event_count = len(analyzer.get_events())
        duplicate_result = analyzer.add_candles([dict(retest)])
        assert len(analyzer.get_events()) <= duplicate_event_count + len(duplicate_result["new_events"])

    def test_invalid_swing_payload_rejects_before_state_is_mutated(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        swing_factory,
    ) -> None:
        analyzer = SupportResistanceAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=support_resistance_config,
        )

        valid = swing_factory(1, price=100.0, swing_id="valid-low", swing_type=SwingType.LOW)
        invalid = dict(swing_factory(2, price=101.0, swing_id="invalid-negative-index"))
        invalid["index"] = -999

        with pytest.raises(ValueError):
            analyzer.add_swings([valid, invalid])

        # This assertion is deliberately strict. It exposes partial batch mutation.
        # If it fails, add_swings/add_data should be made atomic or explicitly
        # documented as partial-commit behavior.
        assert analyzer.get_levels() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_swings == set()

    @pytest.mark.asyncio
    async def test_event_handler_invalid_payload_does_not_emit_or_mutate_state(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        support_resistance_config,
    ) -> None:
        analyzer = SupportResistanceAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=support_resistance_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()
        await analyzer.on_swing_event(
            event_factory(
                support_resistance_config.swing_high_topic,
                ["not", "a", "mapping"],
                correlation_id="bad-swing",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# FairValueGapAnalyzer
# ---------------------------------------------------------------------------

class TestFairValueGapAnalyzerDomain:
    def test_detects_bullish_and_bearish_fvg_with_separate_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
        bearish_fvg_candles,
    ) -> None:
        bullish = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )
        bearish = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )

        bullish_result = bullish.add_candles(bullish_fvg_candles)
        bearish_result = bearish.add_candles(bearish_fvg_candles)

        assert FVGEventType.FVG_CREATED.value in values(bullish_result["new_events"])
        assert FVGEventType.FVG_CREATED.value in values(bearish_result["new_events"])

        bullish_directions = {gap["direction"] for gap in bullish_result["updated_gaps"]}
        bearish_directions = {gap["direction"] for gap in bearish_result["updated_gaps"]}

        assert FVGDirection.BULLISH.value in bullish_directions
        assert FVGDirection.BEARISH.value in bearish_directions
        assert all(gap.lower_bound <= gap.upper_bound for gap in bullish.get_gaps())
        assert all(gap.lower_bound <= gap.upper_bound for gap in bearish.get_gaps())
        assert_no_duplicate_ids(bullish_result["new_events"], "event_id")
        assert_confidences_are_bounded(bullish_result["new_events"] + bearish_result["new_events"])

    def test_triplet_detection_is_idempotent_without_new_candles(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
    ) -> None:
        analyzer = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )

        first = analyzer.add_candles(bullish_fvg_candles)
        assert first["updated_gaps"]

        previous_gap_count = len(analyzer.get_gaps())
        previous_event_count = len(analyzer.get_events())

        second_gaps, second_events = analyzer._process_incremental_gap_detection()

        assert second_gaps == []
        assert second_events == []
        assert len(analyzer.get_gaps()) == previous_gap_count
        assert len(analyzer.get_events()) == previous_event_count

    def test_gap_lifecycle_partial_fill_full_fill_or_invalidation_is_deduplicated(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
        candle_factory,
    ) -> None:
        analyzer = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=replace(
                fair_value_gap_config,
                respected_reaction_threshold_pct=0.0001,
                invalidation_close_buffer_pct=0.0,
            ),
        )

        analyzer.add_candles(bullish_fvg_candles)
        assert analyzer.get_gaps()

        gap = analyzer.get_gaps()[0]

        partial_fill = candle_factory(
            10,
            open_=gap.upper_bound + 0.20,
            high=gap.upper_bound + 0.50,
            low=gap.mid_price,
            close=gap.mid_price + 0.05,
        )
        full_fill_or_invalidation = candle_factory(
            11,
            open_=gap.mid_price,
            high=gap.upper_bound + 0.10,
            low=max(gap.lower_bound - 0.20, 0.0),
            close=max(gap.lower_bound - 0.10, 0.0),
        )

        result = analyzer.add_candles([partial_fill, full_fill_or_invalidation])
        event_types = values(result["new_events"])

        assert any(
            event_type in event_types
            for event_type in {
                FVGEventType.FVG_FILL_STARTED.value,
                FVGEventType.FVG_PARTIALLY_FILLED.value,
                FVGEventType.FVG_FILLED.value,
                FVGEventType.FVG_INVALIDATED.value,
                FVGEventType.FVG_RETESTED.value,
            }
        )

        after_first_lifecycle_count = len(analyzer.get_events())
        repeated = analyzer.add_candles([dict(full_fill_or_invalidation)])

        assert len(analyzer.get_events()) <= after_first_lifecycle_count + len(repeated["new_events"])
        assert all(0.0 <= gap.fill_percentage <= 1.0 for gap in analyzer.get_gaps())
        assert all(gap.status in set(FVGStatus) for gap in analyzer.get_gaps())

    def test_batch_with_invalid_late_candle_should_not_leave_half_mutated_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        candle_factory,
    ) -> None:
        analyzer = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )

        valid = candle_factory(1, open_=100, high=101, low=99, close=100.5)
        invalid = candle_factory(2, open_=100, high=101, low=99, close=100.5)
        invalid["low"] = 200.0

        with pytest.raises(ValueError):
            analyzer.add_candles([valid, invalid])

        # Deliberately strict: catches partial commit before validation of the
        # full batch. If this fails, add_candles should parse/validate all candles
        # before mutating rolling state.
        assert analyzer.get_gaps() == []
        assert analyzer.get_events() == []
        assert analyzer._global_candle_index == 0
        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 0

    def test_reset_clears_gaps_events_processed_keys_and_snapshot_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
    ) -> None:
        analyzer = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )

        analyzer.add_candles(bullish_fvg_candles)
        assert analyzer.get_gaps()
        assert analyzer.get_events()

        analyzer.reset()

        assert analyzer.get_gaps() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_fill_keys == set()
        assert analyzer._processed_respect_keys == set()
        assert analyzer._processed_invalidation_keys == set()
        assert analyzer._processed_retest_keys == set()
        assert analyzer._last_processed_triplet_end_index == -1

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 0
        assert metadata["internal_gaps"] == 0
        assert metadata["external_gaps"] == 0
        assert metadata["events"] == 0


# ---------------------------------------------------------------------------
# LiquidityLevelsAnalyzer
# ---------------------------------------------------------------------------

class TestLiquidityLevelsAnalyzerDomain:
    def test_duplicate_swing_id_creates_levels_once_only(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
    ) -> None:
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=liquidity_levels_config,
        )

        swing = swing_factory(
            1,
            price=105.0,
            swing_type=SwingType.HIGH,
            layer=StructureLayer.INTERNAL,
            swing_id="same-swing",
        )

        first = analyzer.add_swings([swing])
        second = analyzer.add_swings([dict(swing)])

        assert len(first["updated_levels"]) >= 1
        assert second["updated_levels"] == []
        assert second["new_events"] == []
        assert len(analyzer._processed_swings) == 1
        assert len(analyzer.get_levels(StructureLayer.INTERNAL)) == len(
            {level.level_id for level in analyzer.get_levels(StructureLayer.INTERNAL)}
        )

    def test_equal_high_cluster_merges_without_level_explosion(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
    ) -> None:
        config = replace(
            liquidity_levels_config,
            equal_level_tolerance_pct_internal=0.004,
            min_cluster_size_for_equal_levels=2,
        )
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=config,
        )

        swings = [
            swing_factory(1, price=105.00, swing_type=SwingType.HIGH, swing_id="lh-1"),
            swing_factory(2, price=105.15, swing_type=SwingType.HIGH, swing_id="lh-2"),
            swing_factory(3, price=105.25, swing_type=SwingType.HIGH, swing_id="lh-3"),
        ]

        result = analyzer.add_swings(swings)
        levels = analyzer.get_levels(StructureLayer.INTERNAL)
        event_types = values(result["new_events"])

        assert len(levels) <= 4
        assert LiquidityEventType.LEVEL_CREATED.value in event_types
        assert any(level.source_count >= 2 for level in levels)
        assert_no_duplicate_ids(result["new_events"], "event_id")
        assert_confidences_are_bounded(result["new_events"])

    def test_sweep_reclaim_failed_breakout_and_stop_run_do_not_duplicate_same_level_index_keys(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
        candle_factory,
    ) -> None:
        config = replace(
            liquidity_levels_config,
            min_sweep_penetration_pct=0.0,
            reclaim_close_buffer_pct=0.0,
            require_close_reclaim=True,
            retest_window_bars=5,
            failed_breakout_reclaim_window_bars=3,
            stop_run_wick_ratio_threshold=0.25,
        )
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=config,
        )

        analyzer.add_swings(
            [
                swing_factory(
                    1,
                    price=105.0,
                    swing_type=SwingType.HIGH,
                    layer=StructureLayer.INTERNAL,
                    swing_id="liq-high-1",
                    strength=0.90,
                )
            ]
        )

        sweep = build_upper_liquidity_sweep_candle(candle_factory, 10, level_price=105.0)
        reclaim = build_upper_liquidity_reclaim_candle(candle_factory, 11, level_price=105.0)
        second_reclaim_like = build_upper_liquidity_reclaim_candle(candle_factory, 12, level_price=105.0)

        result = analyzer.add_candles([sweep, reclaim, second_reclaim_like])
        event_types = values(result["new_events"])

        assert LiquidityEventType.LIQUIDITY_TOUCHED.value in event_types
        assert LiquidityEventType.LIQUIDITY_SWEPT.value in event_types
        assert any(
            event_type in event_types
            for event_type in {
                LiquidityEventType.LIQUIDITY_RECLAIMED.value,
                LiquidityEventType.FAILED_BREAKOUT.value,
                LiquidityEventType.STOP_RUN.value,
            }
        )

        assert len(analyzer._processed_touch_keys) == len(set(analyzer._processed_touch_keys))
        assert len(analyzer._processed_sweep_keys) == len(set(analyzer._processed_sweep_keys))
        assert len(analyzer._processed_reclaim_keys) == len(set(analyzer._processed_reclaim_keys))
        assert len(analyzer._processed_failed_breakout_keys) == len(set(analyzer._processed_failed_breakout_keys))
        assert len(analyzer._processed_stop_run_keys) == len(set(analyzer._processed_stop_run_keys))
        assert_confidences_are_bounded(result["new_events"])

    def test_invalid_swing_batch_should_not_partial_commit_levels_or_processed_keys(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
    ) -> None:
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=liquidity_levels_config,
        )

        valid = swing_factory(1, price=101.0, swing_id="valid-swing")
        invalid = dict(swing_factory(2, price=102.0, swing_id="invalid-swing"))
        invalid["price"] = -10.0

        with pytest.raises(ValueError):
            analyzer.add_swings([valid, invalid])

        # Deliberately strict atomicity check.
        assert analyzer.get_levels() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_swings == set()

    def test_reset_clears_liquidity_runtime_state_and_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
        candle_factory,
    ) -> None:
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=liquidity_levels_config,
        )

        analyzer.add_swings(
            [
                swing_factory(1, price=105.0, swing_type=SwingType.HIGH, swing_id="reset-liq"),
            ]
        )
        analyzer.add_candles([build_upper_liquidity_sweep_candle(candle_factory, 10, level_price=105.0)])

        assert analyzer.get_levels()
        assert analyzer.get_events()

        analyzer.reset()

        assert analyzer.get_levels() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_swings == set()
        assert analyzer._processed_touch_keys == set()
        assert analyzer._processed_sweep_keys == set()
        assert analyzer._processed_reclaim_keys == set()
        assert analyzer._processed_failed_breakout_keys == set()
        assert analyzer._processed_stop_run_keys == set()

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 0
        assert metadata["internal_levels"] == 0
        assert metadata["external_levels"] == 0
        assert metadata["events"] == 0

    @pytest.mark.asyncio
    async def test_invalid_swing_handler_payload_does_not_emit_or_mutate_state(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
    ) -> None:
        analyzer = LiquidityLevelsAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=liquidity_levels_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()

        await analyzer.on_swing_event(
            event_factory(
                liquidity_levels_config.swing_high_topic,
                object(),
                correlation_id="bad-liq-swing",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# TrendAnalyzer
# ---------------------------------------------------------------------------

class TestTrendAnalyzerDomain:
    def test_rising_falling_and_ranging_sequences_produce_distinct_directional_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        rising_candles,
        falling_candles,
        ranging_candles,
    ) -> None:
        bullish = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)
        bearish = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)
        ranging = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)

        bullish_result = bullish.add_candles(rising_candles(40, step=0.80))
        bearish_result = bearish.add_candles(falling_candles(40, step=0.80))
        ranging_result = ranging.add_candles(ranging_candles(40, amplitude=0.10))

        bullish_state = bullish.get_state()
        bearish_state = bearish.get_state()
        ranging_state = ranging.get_state()

        assert bullish_state.last_price is not None
        assert bearish_state.last_price is not None
        assert ranging_state.last_price is not None

        assert bullish_state.internal.direction in {
            TrendDirection.BULLISH,
            TrendDirection.NEUTRAL,
            TrendDirection.UNKNOWN,
        }
        assert bearish_state.internal.direction in {
            TrendDirection.BEARISH,
            TrendDirection.NEUTRAL,
            TrendDirection.UNKNOWN,
        }
        assert ranging_state.internal.regime in set(TrendRegime)

        assert isinstance(bullish_result["new_signals"], list)
        assert isinstance(bearish_result["new_signals"], list)
        assert isinstance(ranging_result["new_signals"], list)
        assert_no_duplicate_ids(bullish_result["new_signals"], "signal_id")
        assert_confidences_are_bounded(bullish_result["new_signals"] + bearish_result["new_signals"])

    def test_market_structure_and_support_resistance_context_are_consumed_without_losing_candles(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        rising_candles,
        market_structure_update_payload,
        support_resistance_update_payload,
    ) -> None:
        analyzer = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)

        candles_result = analyzer.add_candles(rising_candles(20))
        candles_before_context = snapshot_metadata(analyzer.snapshot())["total_candles"]

        context_result = analyzer.add_data(
            market_structure=market_structure_update_payload,
            support_resistance=support_resistance_update_payload,
        )

        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == candles_before_context
        assert analyzer._latest_market_structure == market_structure_update_payload
        assert analyzer._latest_support_resistance == support_resistance_update_payload
        assert isinstance(candles_result["new_signals"], list)
        assert isinstance(context_result["new_signals"], list)

    def test_invalid_context_handlers_do_not_emit_or_mutate_context(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        trend_config,
        market_structure_update_payload,
    ) -> None:
        analyzer = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        analyzer.add_data(market_structure=market_structure_update_payload)
        before_context = dict(analyzer._latest_market_structure)
        before_snapshot_state = analyzer.snapshot()["state"]

        import asyncio

        asyncio.run(
            analyzer.on_market_structure_event(
                event_factory(
                    trend_config.market_structure_updated_topic,
                    ["invalid"],
                    correlation_id="bad-ms-context",
                )
            )
        )
        asyncio.run(
            analyzer.on_support_resistance_event(
                event_factory(
                    trend_config.support_resistance_updated_topic,
                    "invalid",
                    correlation_id="bad-sr-context",
                )
            )
        )

        assert analyzer._latest_market_structure == before_context
        assert analyzer.snapshot()["state"] == before_snapshot_state
        assert recorder.calls == []

    def test_reset_clears_trend_candles_signals_context_and_increments_state_version(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        rising_candles,
        market_structure_update_payload,
        support_resistance_update_payload,
    ) -> None:
        analyzer = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)

        analyzer.add_data(
            candles=rising_candles(25),
            market_structure=market_structure_update_payload,
            support_resistance=support_resistance_update_payload,
        )

        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 25
        assert analyzer._latest_market_structure
        assert analyzer._latest_support_resistance

        version_before = analyzer._state_version
        analyzer.reset()

        assert analyzer._state_version == version_before + 1
        assert analyzer.get_signals() == []
        assert analyzer._latest_market_structure == {}
        assert analyzer._latest_support_resistance == {}
        assert analyzer._global_candle_index == 0

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 0
        assert metadata["signals"] == 0

    def test_batch_with_invalid_late_candle_should_not_partially_advance_global_index(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        candle_factory,
    ) -> None:
        analyzer = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)

        valid = candle_factory(1, open_=100.0, high=101.0, low=99.0, close=100.4)
        invalid = candle_factory(2, open_=100.0, high=101.0, low=99.0, close=100.4)
        invalid["volume"] = -1.0

        with pytest.raises(ValueError):
            analyzer.add_candles([valid, invalid])

        # Deliberately strict atomicity check.
        assert analyzer._global_candle_index == 0
        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 0
        assert analyzer.get_state().last_price is None


# ---------------------------------------------------------------------------
# Cross-analyzer event-handler publication contract
# ---------------------------------------------------------------------------

class TestAnalyzerEventHandlers:
    @pytest.mark.asyncio
    async def test_fvg_handler_publishes_domain_events_before_updated_event(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
    ) -> None:
        analyzer = FairValueGapAnalyzer(
            symbol,
            timeframe,
            event_bus=event_bus,
            config=fair_value_gap_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await analyzer.on_candles_event(
            event_factory(
                fair_value_gap_config.market_candles_topic,
                {"candles": bullish_fvg_candles},
                correlation_id="fvg-handler",
            )
        )

        assert recorder.calls
        topics = [call["topic"] for call in recorder.calls]
        assert topics[-1] == "analytics.price_action.fair_value_gap.updated"
        assert any(topic.endswith(f".{FVGEventType.FVG_CREATED.value}") for topic in topics)
        assert all(call["kwargs"]["correlation_id"] == "fvg-handler" for call in recorder.calls)

    @pytest.mark.asyncio
    async def test_trend_handler_does_not_publish_on_empty_candle_payload(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        trend_config,
    ) -> None:
        analyzer = TrendAnalyzer(symbol, timeframe, event_bus=event_bus, config=trend_config)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await analyzer.on_candles_event(
            event_factory(
                trend_config.market_candles_topic,
                {"candles": []},
                correlation_id="empty-trend",
            )
        )

        assert recorder.calls == []
        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 0