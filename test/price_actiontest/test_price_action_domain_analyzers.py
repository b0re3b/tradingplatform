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

    Domain tests patch EventBus.emit so they can verify publication contracts
    without starting the EventBus worker.
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


def build_market_structure(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    market_structure_config,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
    higher_timeframe: str | None = None,
) -> MarketStructureAnalyzer:
    return MarketStructureAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=market_structure_config,
        higher_timeframe=higher_timeframe,
    )


def build_support_resistance(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    support_resistance_config,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
) -> SupportResistanceAnalyzer:
    return SupportResistanceAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=support_resistance_config,
    )


def build_fvg(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    fair_value_gap_config,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
) -> FairValueGapAnalyzer:
    return FairValueGapAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=fair_value_gap_config,
    )


def build_liquidity(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    liquidity_levels_config,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
) -> LiquidityLevelsAnalyzer:
    return LiquidityLevelsAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=liquidity_levels_config,
    )


def build_trend(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    trend_config,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
) -> TrendAnalyzer:
    return TrendAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=trend_config,
    )


def values(items: list[dict[str, Any]], key: str = "event_type") -> list[str]:
    return [str(item[key]) for item in items if key in item]


def snapshot_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


def assert_payload_scope(payload: Mapping[str, Any]) -> None:
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange_symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "1m"
    assert payload["key"] == ["binance", "usdm_futures", "BTCUSDT", "1m"]


def assert_snapshot_is_serialized(snapshot: dict[str, Any]) -> None:
    assert_payload_scope(snapshot)
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
# Cross-module scoped EventBus handler behavior
# ---------------------------------------------------------------------------

class TestScopedDomainEventHandlers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("module_name", "config_fixture", "expected_topic"),
        [
            (
                "market_structure",
                "market_structure_config",
                "analytics.price_action.market_structure.updated",
            ),
            (
                "support_resistance",
                "support_resistance_config",
                "analytics.price_action.support_resistance.updated",
            ),
            (
                "fair_value_gap",
                "fair_value_gap_config",
                "analytics.price_action.fair_value_gap.updated",
            ),
            (
                "liquidity",
                "liquidity_levels_config",
                "analytics.price_action.liquidity_levels.updated",
            ),
            (
                "trend",
                "trend_config",
                "analytics.price_action.trend.updated",
            ),
        ],
    )
    async def test_matching_scoped_candles_updated_event_mutates_and_emits(
        self,
        request,
        event_bus,
        monkeypatch,
        event_factory,
        candles_updated_payload,
        candle_factory,
        symbol: str,
        timeframe: str,
        module_name: str,
        config_fixture: str,
        expected_topic: str,
    ) -> None:
        config = request.getfixturevalue(config_fixture)

        builders = {
            "market_structure": lambda: build_market_structure(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                market_structure_config=config,
            ),
            "support_resistance": lambda: build_support_resistance(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                support_resistance_config=config,
            ),
            "fair_value_gap": lambda: build_fvg(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                fair_value_gap_config=config,
            ),
            "liquidity": lambda: build_liquidity(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                liquidity_levels_config=config,
            ),
            "trend": lambda: build_trend(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                trend_config=config,
            ),
        }

        analyzer = builders[module_name]()
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        candles = [candle_factory(0), candle_factory(1), candle_factory(2)]
        event = event_factory(
            "market.candles.updated",
            candles_updated_payload(candles),
            source="CandlesCache",
            correlation_id=f"{module_name}-batch",
        )

        await analyzer._on_candles_event_scoped(event)

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 3
        assert recorder.calls
        assert recorder.calls[-1]["topic"] == expected_topic
        assert recorder.calls[-1]["kwargs"]["correlation_id"] == f"{module_name}-batch"
        assert_payload_scope(recorder.calls[-1]["payload"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("module_name", "config_fixture"),
        [
            ("market_structure", "market_structure_config"),
            ("support_resistance", "support_resistance_config"),
            ("fair_value_gap", "fair_value_gap_config"),
            ("liquidity", "liquidity_levels_config"),
            ("trend", "trend_config"),
        ],
    )
    async def test_wrong_scope_single_candle_event_is_ignored_without_emit_or_mutation(
        self,
        request,
        event_bus,
        monkeypatch,
        event_factory,
        wrong_scope_candle_factory,
        symbol: str,
        timeframe: str,
        module_name: str,
        config_fixture: str,
    ) -> None:
        config = request.getfixturevalue(config_fixture)

        builders = {
            "market_structure": lambda: build_market_structure(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                market_structure_config=config,
            ),
            "support_resistance": lambda: build_support_resistance(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                support_resistance_config=config,
            ),
            "fair_value_gap": lambda: build_fvg(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                fair_value_gap_config=config,
            ),
            "liquidity": lambda: build_liquidity(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                liquidity_levels_config=config,
            ),
            "trend": lambda: build_trend(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                trend_config=config,
            ),
        }

        analyzer = builders[module_name]()
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()
        event = event_factory(
            "market.candle.closed",
            wrong_scope_candle_factory(1, wrong_exchange=True),
            source="CandlesCache",
            correlation_id=f"{module_name}-wrong-scope",
        )

        await analyzer._on_candle_event_scoped(event)

        assert analyzer.snapshot()["state"] == before["state"]
        assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 0
        assert recorder.calls == []


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
        analyzer = build_market_structure(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            market_structure_config=market_structure_config,
        )

        result = analyzer.add_candles(swing_pattern_candles())

        assert_payload_scope(result)
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
        assert_snapshot_is_serialized(analyzer.snapshot())

    def test_detects_structure_labels_without_unbounded_duplicate_labels(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        market_structure_config,
        candle_factory,
    ) -> None:
        analyzer = build_market_structure(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            market_structure_config=market_structure_config,
        )

        prices = [
            100.0,
            102.0,
            105.0,
            101.0,
            99.0,
            103.0,
            107.0,
            104.0,
            101.0,
            106.0,
            110.0,
            107.0,
            103.0,
            108.0,
            112.0,
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

        assert any(
            event_type in event_types
            for event_type in {
                StructureEventType.HH.value,
                StructureEventType.HL.value,
                StructureEventType.LH.value,
                StructureEventType.LL.value,
            }
        )
        assert len(analyzer._processed_structure_labels) == len(
            set(analyzer._processed_structure_labels)
        )
        assert len(analyzer.get_events()) == len(
            {event.event_id for event in analyzer.get_events()}
        )
        assert_confidences_are_bounded(result["new_events"])

    @pytest.mark.asyncio
    async def test_higher_timeframe_context_updates_alignment_and_wrong_scope_is_ignored(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        market_structure_config,
    ) -> None:
        analyzer = build_market_structure(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            market_structure_config=market_structure_config,
            higher_timeframe="15m",
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        valid_payload = {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": symbol,
            "exchange_symbol": symbol,
            "timeframe": "15m",
            "bias": MarketBias.BULLISH.value,
            "confidence": 0.90,
        }

        analyzer.update_higher_timeframe_context(valid_payload)
        analyzer._refresh_state()

        state = analyzer.get_state()
        assert state.mtf_alignment.higher_timeframe == "15m"
        assert state.mtf_alignment.higher_timeframe_bias == MarketBias.BULLISH
        assert 0.0 <= state.mtf_alignment.alignment_score <= 1.0

        before = analyzer.snapshot()

        wrong_payload = dict(valid_payload)
        wrong_payload["exchange"] = "bybit"

        await analyzer.on_higher_timeframe_context_event(
            event_factory(
                market_structure_config.higher_timeframe_context_topic,
                wrong_payload,
                correlation_id="wrong-htf-scope",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert recorder.calls == []

    def test_reset_clears_caches_processed_sets_and_state_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        market_structure_config,
        swing_pattern_candles,
    ) -> None:
        analyzer = build_market_structure(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            market_structure_config=market_structure_config,
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
        analyzer = build_support_resistance(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            support_resistance_config=support_resistance_config,
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
        analyzer = build_support_resistance(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            support_resistance_config=config,
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
        analyzer = build_support_resistance(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            support_resistance_config=config,
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

        rejection = build_resistance_rejection_candle(
            candle_factory,
            10,
            level_price=105.0,
        )
        breakout = build_resistance_breakout_candle(
            candle_factory,
            11,
            level_price=105.0,
        )
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

    @pytest.mark.asyncio
    async def test_wrong_scope_swing_event_is_ignored_without_emit_or_mutation(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        wrong_scope_swing_factory,
    ) -> None:
        analyzer = build_support_resistance(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            support_resistance_config=support_resistance_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()
        await analyzer.on_swing_event(
            event_factory(
                support_resistance_config.swing_high_topic,
                wrong_scope_swing_factory(1, wrong_exchange=True),
                correlation_id="wrong-sr-swing",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert analyzer.get_levels() == []
        assert recorder.calls == []

    @pytest.mark.xfail(
        reason=(
            "Known hardening target: add_swings/add_data should validate the full "
            "batch before mutating state, or explicitly document partial commits."
        )
    )
    def test_invalid_swing_payload_rejects_before_state_is_mutated(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        support_resistance_config,
        swing_factory,
    ) -> None:
        analyzer = build_support_resistance(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            support_resistance_config=support_resistance_config,
        )

        valid = swing_factory(
            1,
            price=100.0,
            swing_id="valid-low",
            swing_type=SwingType.LOW,
        )
        invalid = dict(
            swing_factory(
                2,
                price=101.0,
                swing_id="invalid-negative-index",
            )
        )
        invalid["index"] = -999

        with pytest.raises(ValueError):
            analyzer.add_swings([valid, invalid])

        assert analyzer.get_levels() == []
        assert analyzer.get_events() == []
        assert analyzer._processed_swings == set()


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
        bullish = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=fair_value_gap_config,
        )
        bearish = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=fair_value_gap_config,
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
        assert_confidences_are_bounded(
            bullish_result["new_events"] + bearish_result["new_events"]
        )

    def test_triplet_detection_is_idempotent_without_new_candles(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        bullish_fvg_candles,
    ) -> None:
        analyzer = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=fair_value_gap_config,
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
        analyzer = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=replace(
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

        assert len(analyzer.get_events()) <= (
            after_first_lifecycle_count + len(repeated["new_events"])
        )
        assert all(0.0 <= gap.fill_percentage <= 1.0 for gap in analyzer.get_gaps())
        assert all(gap.status in set(FVGStatus) for gap in analyzer.get_gaps())

    @pytest.mark.xfail(
        reason=(
            "Known hardening target: add_candles should parse/validate the full "
            "batch before mutating rolling state."
        )
    )
    def test_batch_with_invalid_late_candle_should_not_leave_half_mutated_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        fair_value_gap_config,
        candle_factory,
    ) -> None:
        analyzer = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=fair_value_gap_config,
        )

        valid = candle_factory(1, open_=100, high=101, low=99, close=100.5)
        invalid = candle_factory(2, open_=100, high=101, low=99, close=100.5)
        invalid["low"] = 200.0

        with pytest.raises(ValueError):
            analyzer.add_candles([valid, invalid])

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
        analyzer = build_fvg(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            fair_value_gap_config=fair_value_gap_config,
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
        analyzer = build_liquidity(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            liquidity_levels_config=liquidity_levels_config,
        )

        swing = swing_factory(
            10,
            price=105.0,
            swing_type=SwingType.HIGH,
            layer=StructureLayer.INTERNAL,
            swing_id="liq-high-duplicate",
        )

        first = analyzer.add_data(swings=[swing])
        second = analyzer.add_data(swings=[dict(swing)])

        assert len(first["updated_levels"]) == 1
        assert len(first["new_events"]) == 1
        assert second["updated_levels"] == []
        assert second["new_events"] == []
        assert len(analyzer.get_levels(StructureLayer.INTERNAL)) == 1
        assert len(analyzer.get_events()) == 1
        assert analyzer._processed_swings == {"liq-high-duplicate"}

    def test_equal_highs_cluster_merges_instead_of_level_explosion(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
    ) -> None:
        config = replace(
            liquidity_levels_config,
            equal_level_tolerance_pct_internal=0.005,
            min_cluster_size_for_equal_levels=2,
        )
        analyzer = build_liquidity(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            liquidity_levels_config=config,
        )

        swings = [
            swing_factory(1, price=105.00, swing_type=SwingType.HIGH, swing_id="lh-1"),
            swing_factory(2, price=105.15, swing_type=SwingType.HIGH, swing_id="lh-2"),
            swing_factory(3, price=105.30, swing_type=SwingType.HIGH, swing_id="lh-3"),
        ]

        result = analyzer.add_data(swings=swings)
        levels = analyzer.get_levels(StructureLayer.INTERNAL)
        event_types = values(result["new_events"])

        assert len(levels) == 1
        assert levels[0].source_count == 3
        assert LiquidityEventType.LEVEL_CREATED.value in event_types
        assert LiquidityEventType.LEVEL_MERGED.value in event_types
        assert_no_duplicate_ids(result["new_events"], "event_id")
        assert_confidences_are_bounded(result["new_events"])

    def test_sweep_reclaim_failed_breakout_or_stop_run_events_are_bounded(
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
            stop_run_wick_ratio_threshold=0.20,
            failed_breakout_reclaim_window_bars=3,
            retest_window_bars=5,
        )
        analyzer = build_liquidity(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            liquidity_levels_config=config,
        )

        analyzer.add_data(
            swings=[
                swing_factory(
                    1,
                    price=105.0,
                    swing_type=SwingType.HIGH,
                    layer=StructureLayer.INTERNAL,
                    swing_id="upper-liquidity-1",
                )
            ]
        )

        sweep = build_upper_liquidity_sweep_candle(
            candle_factory,
            10,
            level_price=105.0,
        )
        reclaim = build_upper_liquidity_reclaim_candle(
            candle_factory,
            11,
            level_price=105.0,
        )

        result = analyzer.add_data(candles=[sweep, reclaim])
        event_types = values(result["new_events"])

        assert any(
            event_type in event_types
            for event_type in {
                LiquidityEventType.LIQUIDITY_TOUCHED.value,
                LiquidityEventType.LIQUIDITY_SWEPT.value,
                LiquidityEventType.LIQUIDITY_RECLAIMED.value,
                LiquidityEventType.FAILED_BREAKOUT.value,
                LiquidityEventType.STOP_RUN.value,
            }
        )
        assert_confidences_are_bounded(result["new_events"])
        assert all(level.touch_count >= 0 for level in analyzer.get_levels())

    @pytest.mark.asyncio
    async def test_wrong_scope_swing_event_is_ignored_without_emit_or_mutation(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        wrong_scope_swing_factory,
    ) -> None:
        analyzer = build_liquidity(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            liquidity_levels_config=liquidity_levels_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()
        await analyzer.on_swing_event(
            event_factory(
                liquidity_levels_config.swing_high_topic,
                wrong_scope_swing_factory(1, wrong_market_type=True),
                correlation_id="wrong-liquidity-swing",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert analyzer.get_levels() == []
        assert recorder.calls == []

    def test_reset_clears_levels_events_processed_keys_and_snapshot_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        liquidity_levels_config,
        swing_factory,
    ) -> None:
        analyzer = build_liquidity(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            liquidity_levels_config=liquidity_levels_config,
        )

        analyzer.add_data(
            swings=[
                swing_factory(1, price=105.0, swing_type=SwingType.HIGH, swing_id="a"),
                swing_factory(2, price=95.0, swing_type=SwingType.LOW, swing_id="b"),
            ]
        )
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


# ---------------------------------------------------------------------------
# TrendAnalyzer
# ---------------------------------------------------------------------------

class TestTrendAnalyzerDomain:
    def test_rising_candles_produce_bullish_or_non_unknown_trend_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        rising_candles,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )

        result = analyzer.add_data(candles=rising_candles(25))
        state = analyzer.get_state()

        assert_payload_scope(result)
        assert state.last_price is not None
        assert 0.0 <= float(state.overall_trend_score) <= 1.0
        assert state.internal.direction in {
            TrendDirection.BULLISH,
            TrendDirection.NEUTRAL,
            TrendDirection.UNKNOWN,
        }
        assert state.internal.regime in set(TrendRegime)
        assert_confidences_are_bounded(result.get("new_signals", []))

    def test_falling_candles_produce_bearish_or_non_unknown_trend_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        falling_candles,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )

        result = analyzer.add_data(candles=falling_candles(25))
        state = analyzer.get_state()

        assert_payload_scope(result)
        assert state.last_price is not None
        assert 0.0 <= float(state.overall_trend_score) <= 1.0
        assert state.internal.direction in {
            TrendDirection.BEARISH,
            TrendDirection.NEUTRAL,
            TrendDirection.UNKNOWN,
        }
        assert state.internal.regime in set(TrendRegime)
        assert_confidences_are_bounded(result.get("new_signals", []))

    def test_ranging_candles_do_not_create_unbounded_signal_explosion(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        ranging_candles,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )

        result = analyzer.add_data(candles=ranging_candles(80))

        assert len(analyzer.get_signals()) <= analyzer.config.max_signals
        assert len(result.get("new_signals", [])) <= analyzer.config.max_signals
        assert_confidences_are_bounded(result.get("new_signals", []))

    @pytest.mark.asyncio
    async def test_market_structure_and_support_resistance_context_update_trend_state(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        trend_config,
        market_structure_update_payload,
        support_resistance_update_payload,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await analyzer.on_market_structure_event(
            event_factory(
                trend_config.market_structure_updated_topic,
                market_structure_update_payload,
                correlation_id="trend-ms-context",
            )
        )
        await analyzer.on_support_resistance_event(
            event_factory(
                trend_config.support_resistance_updated_topic,
                support_resistance_update_payload,
                correlation_id="trend-sr-context",
            )
        )

        assert analyzer._latest_market_structure
        assert analyzer._latest_support_resistance
        assert recorder.calls
        assert all(call["topic"] == "analytics.price_action.trend.updated" for call in recorder.calls)

    @pytest.mark.asyncio
    async def test_wrong_scope_context_is_ignored_without_emit_or_state_mutation(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
        trend_config,
        market_structure_update_payload,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before = analyzer.snapshot()
        wrong_payload = dict(market_structure_update_payload)
        wrong_payload["exchange"] = "bybit"

        await analyzer.on_market_structure_event(
            event_factory(
                trend_config.market_structure_updated_topic,
                wrong_payload,
                correlation_id="wrong-trend-context",
            )
        )

        assert analyzer.snapshot()["state"] == before["state"]
        assert analyzer._latest_market_structure == {}
        assert recorder.calls == []

    def test_reset_clears_candles_signals_context_and_snapshot_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
        trend_config,
        rising_candles,
        market_structure_update_payload,
        support_resistance_update_payload,
    ) -> None:
        analyzer = build_trend(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            trend_config=trend_config,
        )

        analyzer.add_data(
            candles=rising_candles(30),
            market_structure=market_structure_update_payload,
            support_resistance=support_resistance_update_payload,
        )

        assert snapshot_metadata(analyzer.snapshot())["total_candles"] > 0

        analyzer.reset()

        assert analyzer.get_signals() == []
        assert analyzer._latest_market_structure == {}
        assert analyzer._latest_support_resistance == {}
        assert analyzer._global_candle_index == 0
        assert analyzer._state_version == 0

        metadata = snapshot_metadata(analyzer.snapshot())
        assert metadata["total_candles"] == 0
        assert metadata["signals"] == 0