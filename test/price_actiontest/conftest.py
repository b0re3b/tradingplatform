# tests/analytics/price_action/conftest.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.enums import (
    MarketBias,
    StructureLayer,
    SwingType,
)
from analytics.price_action.fair_value_gap import FairValueGapConfig
from analytics.price_action.liquidity_levels import LiquidityLevelsConfig
from analytics.price_action.market_structure import MarketStructureConfig
from analytics.price_action.price_action_analyzer import PriceActionAnalyzerConfig
from analytics.price_action.support_resistance import SupportResistanceConfig
from analytics.price_action.trend import TrendConfig


TEST_SYMBOL = "BTCUSDT"
TEST_TIMEFRAME = "1m"
TEST_START = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Core infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus() -> EventBus:
    """
    Real EventBus instance.

    We prefer the real core EventBus over a dummy mock because price_action modules
    are event-driven and their lifecycle/register/unregister behavior must be
    tested against the actual infrastructure contract.
    """
    return EventBus()


@pytest.fixture
def scheduler(event_bus: EventBus) -> Scheduler:
    """
    Real Scheduler instance bound to the test EventBus.

    Snapshot jobs are disabled in most configs below by default, but this fixture
    is useful for lifecycle tests that verify scheduled snapshot registration.
    """
    return Scheduler(event_bus=event_bus)


@pytest.fixture
def symbol() -> str:
    return TEST_SYMBOL


@pytest.fixture
def timeframe() -> str:
    return TEST_TIMEFRAME


@pytest.fixture
def start_time() -> datetime:
    return TEST_START


# ---------------------------------------------------------------------------
# Candle factories
# ---------------------------------------------------------------------------

@pytest.fixture
def candle_factory(start_time: datetime) -> Callable[..., dict[str, Any]]:
    """
    Factory for valid OHLCV candle payloads.

    Returns dict payloads instead of Candle instances on purpose:
    analyzer public APIs and EventBus handlers usually receive raw mappings and
    must exercise their own parsing/normalization logic.
    """

    def _make(
        index: int,
        *,
        open_: float = 100.0,
        high: float | None = None,
        low: float | None = None,
        close: float | None = None,
        volume: float = 1_000.0,
        timestamp: datetime | str | int | float | None = None,
    ) -> dict[str, Any]:
        resolved_close = close if close is not None else open_ + 0.25
        resolved_high = high if high is not None else max(open_, resolved_close) + 0.50
        resolved_low = low if low is not None else min(open_, resolved_close) - 0.50

        return {
            "timestamp": timestamp or start_time + timedelta(minutes=index),
            "open": float(open_),
            "high": float(resolved_high),
            "low": float(resolved_low),
            "close": float(resolved_close),
            "volume": float(volume),
            "index": index,
        }

    return _make


@pytest.fixture
def rising_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., list[dict[str, Any]]]:
    """
    Controlled bullish sequence.

    Useful for TrendAnalyzer, MarketStructureAnalyzer and breakout scenarios.
    """

    def _make(
        count: int,
        *,
        start: float = 100.0,
        step: float = 0.75,
        index_offset: int = 0,
        volume: float = 1_000.0,
    ) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []

        for i in range(count):
            price = start + i * step
            candles.append(
                candle_factory(
                    index_offset + i,
                    open_=price,
                    high=price + 0.85,
                    low=price - 0.30,
                    close=price + 0.60,
                    volume=volume + i,
                )
            )

        return candles

    return _make


@pytest.fixture
def falling_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., list[dict[str, Any]]]:
    """
    Controlled bearish sequence.

    Useful for TrendAnalyzer, MarketStructureAnalyzer and breakdown scenarios.
    """

    def _make(
        count: int,
        *,
        start: float = 100.0,
        step: float = 0.75,
        index_offset: int = 0,
        volume: float = 1_000.0,
    ) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []

        for i in range(count):
            price = start - i * step
            candles.append(
                candle_factory(
                    index_offset + i,
                    open_=price,
                    high=price + 0.30,
                    low=price - 0.85,
                    close=price - 0.60,
                    volume=volume + i,
                )
            )

        return candles

    return _make


@pytest.fixture
def ranging_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., list[dict[str, Any]]]:
    """
    Sideways/ranging sequence with alternating candles.

    Useful for verifying that TrendAnalyzer does not incorrectly mark a strong
    directional trend on low-displacement data.
    """

    def _make(
        count: int,
        *,
        center: float = 100.0,
        amplitude: float = 0.35,
        index_offset: int = 0,
        volume: float = 1_000.0,
    ) -> list[dict[str, Any]]:
        candles: list[dict[str, Any]] = []

        for i in range(count):
            direction = 1 if i % 2 == 0 else -1
            open_ = center - direction * amplitude * 0.30
            close = center + direction * amplitude * 0.30
            high = center + amplitude
            low = center - amplitude

            candles.append(
                candle_factory(
                    index_offset + i,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )

        return candles

    return _make


@pytest.fixture
def swing_pattern_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., list[dict[str, Any]]]:
    """
    Deterministic pivot-friendly candles.

    The default shape creates visible local highs/lows that are useful for
    MarketStructureAnalyzer tests with small pivot_left/pivot_right settings.
    """

    def _make(
        *,
        prices: Sequence[float] | None = None,
        index_offset: int = 0,
    ) -> list[dict[str, Any]]:
        resolved_prices = list(
            prices
            or [
                100.0,
                101.0,
                103.0,
                101.5,
                100.5,
                99.0,
                100.0,
                102.0,
                104.0,
                102.5,
                101.0,
                98.5,
                99.5,
                101.5,
                105.0,
                103.0,
                101.5,
            ]
        )

        candles: list[dict[str, Any]] = []
        for i, close in enumerate(resolved_prices):
            previous = resolved_prices[i - 1] if i > 0 else close
            open_ = previous
            high = max(open_, close) + 0.40
            low = min(open_, close) - 0.40

            candles.append(
                candle_factory(
                    index_offset + i,
                    open_=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=1_000.0 + i,
                )
            )

        return candles

    return _make


@pytest.fixture
def bullish_fvg_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Three-candle bullish FVG setup.

    Typical bullish gap condition expected by ICT-style logic:
    first candle high is below third candle low, with impulse in between.
    """
    return [
        candle_factory(0, open_=100.0, high=101.0, low=99.4, close=100.6),
        candle_factory(1, open_=100.7, high=104.0, low=100.5, close=103.7),
        candle_factory(2, open_=103.8, high=105.0, low=101.8, close=104.5),
    ]


@pytest.fixture
def bearish_fvg_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Three-candle bearish FVG setup.

    Typical bearish gap condition:
    first candle low is above third candle high, with bearish impulse in between.
    """
    return [
        candle_factory(0, open_=105.0, high=105.8, low=104.0, close=104.4),
        candle_factory(1, open_=104.2, high=104.4, low=100.6, close=101.0),
        candle_factory(2, open_=100.8, high=103.1, low=99.7, close=100.2),
    ]


# ---------------------------------------------------------------------------
# Swing / analytics payload factories
# ---------------------------------------------------------------------------

@pytest.fixture
def swing_factory(start_time: datetime) -> Callable[..., dict[str, Any]]:
    """
    Factory for raw SwingPoint-like mappings.

    Returned shape intentionally mirrors serialized swing events from
    MarketStructureAnalyzer so support/resistance and liquidity tests can use
    it directly through add_swings() or EventBus payloads.
    """

    def _make(
        index: int,
        *,
        price: float = 100.0,
        swing_type: SwingType | str = SwingType.HIGH,
        layer: StructureLayer | str = StructureLayer.INTERNAL,
        strength: float = 0.75,
        swing_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_swing_type = (
            swing_type.value if isinstance(swing_type, SwingType) else str(swing_type)
        )
        resolved_layer = layer.value if isinstance(layer, StructureLayer) else str(layer)

        return {
            "swing_id": swing_id or f"{resolved_layer}-{resolved_swing_type}-{index}",
            "timestamp": start_time + timedelta(minutes=index),
            "price": float(price),
            "swing_type": resolved_swing_type,
            "layer": resolved_layer,
            "index": index,
            "candle_open": float(price - 0.25),
            "candle_high": float(price + 0.50),
            "candle_low": float(price - 0.50),
            "candle_close": float(price + 0.25),
            "strength": float(strength),
            "is_confirmed": True,
            "metadata": {},
        }

    return _make


@pytest.fixture
def market_structure_update_payload() -> dict[str, Any]:
    return {
        "symbol": TEST_SYMBOL,
        "timeframe": TEST_TIMEFRAME,
        "state": {
            "symbol": TEST_SYMBOL,
            "timeframe": TEST_TIMEFRAME,
            "last_price": 101.25,
            "internal": {
                "bias": MarketBias.BULLISH.value,
                "confidence": 0.70,
                "trend_strength": 0.65,
            },
            "external": {
                "bias": MarketBias.BULLISH.value,
                "confidence": 0.62,
                "trend_strength": 0.58,
            },
            "mtf_alignment": {
                "higher_timeframe": "15m",
                "higher_timeframe_bias": MarketBias.BULLISH.value,
                "higher_timeframe_confidence": 0.75,
                "alignment_score": 0.80,
            },
        },
        "new_swings_count": 0,
        "new_events_count": 0,
    }


@pytest.fixture
def support_resistance_update_payload() -> dict[str, Any]:
    return {
        "symbol": TEST_SYMBOL,
        "timeframe": TEST_TIMEFRAME,
        "state": {
            "symbol": TEST_SYMBOL,
            "timeframe": TEST_TIMEFRAME,
            "last_price": 101.25,
            "nearest_support": 99.50,
            "nearest_resistance": 103.00,
        },
        "updated_levels_count": 1,
        "new_events_count": 0,
    }


@pytest.fixture
def event_factory() -> Callable[..., Event]:
    """
    Factory for core Event payloads.

    This keeps Event construction consistent across handler tests.
    """

    def _make(
        topic: str,
        payload: Any,
        *,
        source: str = "pytest",
        correlation_id: str | None = "test-correlation-id",
        headers: dict[str, Any] | None = None,
    ) -> Event:
        return Event(
            topic=topic,
            payload=payload,
            source=source,
            correlation_id=correlation_id,
            headers=headers or {},
        )

    return _make


# ---------------------------------------------------------------------------
# Analyzer configs
# ---------------------------------------------------------------------------

@pytest.fixture
def market_structure_config() -> MarketStructureConfig:
    """
    Test-tuned MarketStructureConfig.

    Small pivot windows make swing detection deterministic and keep tests fast.
    Snapshot publishing is disabled by default; lifecycle tests can override it.
    """
    return MarketStructureConfig(
        pivot_left=1,
        pivot_right=1,
        internal_min_swing_distance_pct=0.0,
        external_min_swing_distance_pct=0.0,
        structure_break_threshold_pct=0.0,
        require_close_break=True,
        max_candles=500,
        max_internal_swings=100,
        max_external_swings=100,
        max_events=200,
        alignment_window=3,
        min_external_strength=0.10,
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=True,
        subscribe_higher_timeframe_context=True,
    )


@pytest.fixture
def support_resistance_config() -> SupportResistanceConfig:
    return SupportResistanceConfig(
        internal_merge_distance_pct=0.0010,
        external_merge_distance_pct=0.0020,
        internal_zone_half_width_pct=0.0010,
        external_zone_half_width_pct=0.0020,
        min_touches_for_validation=2,
        breakout_threshold_pct=0.0001,
        require_close_break=True,
        rejection_wick_ratio_threshold=0.30,
        max_candles=500,
        max_levels_per_layer=100,
        max_events=200,
        retest_window_bars=6,
        allow_flip_on_break=True,
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=True,
        subscribe_market_structure_swings=True,
    )


@pytest.fixture
def fair_value_gap_config() -> FairValueGapConfig:
    return FairValueGapConfig(
        max_candles=500,
        max_gaps_per_layer=100,
        max_events=200,
        min_gap_pct_internal=0.0,
        min_gap_pct_external=0.0,
        merge_distance_pct_internal=0.0,
        merge_distance_pct_external=0.0,
        min_impulse_body_ratio=0.0,
        respected_reaction_threshold_pct=0.0001,
        invalidation_close_buffer_pct=0.0,
        retest_window_bars=8,
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=True,
    )


@pytest.fixture
def liquidity_levels_config() -> LiquidityLevelsConfig:
    return LiquidityLevelsConfig(
        max_candles=500,
        max_levels_per_layer=100,
        max_events=200,
        equal_level_tolerance_pct_internal=0.0020,
        equal_level_tolerance_pct_external=0.0030,
        swing_liquidity_zone_width_pct_internal=0.0010,
        swing_liquidity_zone_width_pct_external=0.0020,
        min_cluster_size_for_equal_levels=2,
        min_sweep_penetration_pct=0.0,
        reclaim_close_buffer_pct=0.0,
        require_close_reclaim=True,
        retest_window_bars=5,
        stop_run_wick_ratio_threshold=0.35,
        failed_breakout_reclaim_window_bars=3,
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=True,
        subscribe_market_structure_swings=True,
    )


@pytest.fixture
def trend_config() -> TrendConfig:
    return TrendConfig(
        max_candles=300,
        max_signals=200,
        short_window=3,
        medium_window=5,
        long_window=8,
        atr_window=3,
        trend_strength_threshold=0.40,
        acceleration_threshold=0.55,
        exhaustion_threshold=0.65,
        reversal_risk_threshold=0.55,
        pullback_depth_threshold=0.0010,
        momentum_slope_threshold=0.0001,
        consolidation_range_threshold=0.0030,
        direction_positive_threshold=0.10,
        direction_negative_threshold=-0.10,
        structure_bias_weight=0.15,
        support_resistance_weight=0.10,
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=True,
        subscribe_market_structure=True,
        subscribe_support_resistance=True,
    )


@pytest.fixture
def price_action_analyzer_config(
    market_structure_config: MarketStructureConfig,
    support_resistance_config: SupportResistanceConfig,
    fair_value_gap_config: FairValueGapConfig,
    liquidity_levels_config: LiquidityLevelsConfig,
    trend_config: TrendConfig,
) -> PriceActionAnalyzerConfig:
    """
    Facade config with all child modules enabled.

    Child configs are injected explicitly so facade tests use the same fast,
    deterministic settings as direct analyzer tests.
    """
    return PriceActionAnalyzerConfig(
        emit_events=True,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=False,
        auto_register_modules=True,
        shutdown_child_modules=True,
        reset_child_modules=True,
        publish_on_module_update=True,
        publish_composite_snapshot_on_module_update=False,
        enable_market_structure=True,
        enable_support_resistance=True,
        enable_fair_value_gap=True,
        enable_liquidity_levels=True,
        enable_trend=True,
        market_structure_config=market_structure_config,
        support_resistance_config=support_resistance_config,
        fair_value_gap_config=fair_value_gap_config,
        liquidity_levels_config=liquidity_levels_config,
        trend_config=trend_config,
    )


@pytest.fixture
def silent_price_action_analyzer_config(
    price_action_analyzer_config: PriceActionAnalyzerConfig,
) -> PriceActionAnalyzerConfig:
    """
    Facade config variant for tests that should not emit EventBus updates.
    """
    price_action_analyzer_config.emit_events = False
    price_action_analyzer_config.publish_on_module_update = False
    price_action_analyzer_config.publish_composite_snapshot_on_module_update = False
    return price_action_analyzer_config


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def event_types() -> Callable[[Iterable[dict[str, Any]]], list[str]]:
    def _extract(events: Iterable[dict[str, Any]]) -> list[str]:
        result: list[str] = []
        for event in events:
            event_type = event.get("event_type")
            if event_type is not None:
                result.append(str(event_type))
        return result

    return _extract


@pytest.fixture
def assert_snapshot_envelope() -> Callable[[dict[str, Any]], None]:
    """
    Shared assertion for analyzer snapshot envelope shape.
    """

    def _assert(snapshot: dict[str, Any]) -> None:
        assert isinstance(snapshot, dict)
        assert "symbol" in snapshot
        assert "timeframe" in snapshot
        assert "state" in snapshot
        assert "metadata" in snapshot
        assert snapshot["symbol"]
        assert snapshot["timeframe"]
        assert isinstance(snapshot["metadata"], dict)

    return _assert