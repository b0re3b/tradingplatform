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


TEST_EXCHANGE = "binance"
TEST_MARKET_TYPE = "usdm_futures"
TEST_SYMBOL = "BTCUSDT"
TEST_EXCHANGE_SYMBOL = "BTCUSDT"
TEST_TIMEFRAME = "1m"

TEST_ALT_EXCHANGE = "bybit"
TEST_ALT_MARKET_TYPE = "linear"
TEST_ALT_SYMBOL = "ETHUSDT"
TEST_ALT_EXCHANGE_SYMBOL = "ETHUSDT"

TEST_SPOT_MARKET_TYPE = "spot"
TEST_START = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared timestamp helpers
# ---------------------------------------------------------------------------

def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_epoch_ms(value: datetime) -> int:
    return int(_ensure_utc(value).timestamp() * 1000)


def _minute_bounds(start_time: datetime, index: int) -> tuple[datetime, datetime]:
    open_time = _ensure_utc(start_time) + timedelta(minutes=index)
    close_time = open_time + timedelta(minutes=1) - timedelta(milliseconds=1)
    return open_time, close_time


def _scope_payload(
    *,
    exchange: str = TEST_EXCHANGE,
    market_type: str = TEST_MARKET_TYPE,
    symbol: str = TEST_SYMBOL,
    exchange_symbol: str = TEST_EXCHANGE_SYMBOL,
    timeframe: str = TEST_TIMEFRAME,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": exchange_symbol,
        "timeframe": timeframe,
        "key": [exchange, market_type, symbol, timeframe],
    }


# ---------------------------------------------------------------------------
# Core infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus() -> EventBus:
    """
    Real EventBus instance.

    Price action modules are EventBus-first, so tests should exercise the real
    core contract instead of a dummy object whenever possible.
    """
    return EventBus()


@pytest.fixture
def scheduler(event_bus: EventBus) -> Scheduler:
    """
    Real Scheduler instance bound to the test EventBus.

    Snapshot jobs are usually disabled in test configs, but lifecycle tests use
    this fixture to verify Scheduler integration.
    """
    return Scheduler(event_bus=event_bus)


# ---------------------------------------------------------------------------
# Scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def exchange() -> str:
    return TEST_EXCHANGE


@pytest.fixture
def market_type() -> str:
    return TEST_MARKET_TYPE


@pytest.fixture
def symbol() -> str:
    return TEST_SYMBOL


@pytest.fixture
def exchange_symbol() -> str:
    return TEST_EXCHANGE_SYMBOL


@pytest.fixture
def timeframe() -> str:
    return TEST_TIMEFRAME


@pytest.fixture
def alt_exchange() -> str:
    return TEST_ALT_EXCHANGE


@pytest.fixture
def alt_market_type() -> str:
    return TEST_ALT_MARKET_TYPE


@pytest.fixture
def alt_symbol() -> str:
    return TEST_ALT_SYMBOL


@pytest.fixture
def alt_exchange_symbol() -> str:
    return TEST_ALT_EXCHANGE_SYMBOL


@pytest.fixture
def spot_market_type() -> str:
    return TEST_SPOT_MARKET_TYPE


@pytest.fixture
def start_time() -> datetime:
    return TEST_START


@pytest.fixture
def price_action_scope(
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return _scope_payload(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
    )


@pytest.fixture
def wrong_exchange_scope(
    alt_exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return _scope_payload(
        exchange=alt_exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
    )


@pytest.fixture
def wrong_market_type_scope(
    exchange: str,
    alt_market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return _scope_payload(
        exchange=exchange,
        market_type=alt_market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
    )


@pytest.fixture
def wrong_symbol_scope(
    exchange: str,
    market_type: str,
    alt_symbol: str,
    alt_exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return _scope_payload(
        exchange=exchange,
        market_type=market_type,
        symbol=alt_symbol,
        exchange_symbol=alt_exchange_symbol,
        timeframe=timeframe,
    )


@pytest.fixture
def spot_scope(
    exchange: str,
    spot_market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return _scope_payload(
        exchange=exchange,
        market_type=spot_market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
    )


# ---------------------------------------------------------------------------
# Candle factories
# ---------------------------------------------------------------------------

@pytest.fixture
def candle_factory(
    start_time: datetime,
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> Callable[..., dict[str, Any]]:
    """
    Factory for valid CandlesCache-like OHLCV payloads.

    The callable is backward-compatible with older tests:
        candle_factory(index, open_=..., high=..., low=..., close=...)

    But every returned candle now includes the full futures scope and data-layer
    identity fields:
        exchange, market_type, symbol, exchange_symbol, timeframe,
        open_time_ms, close_time_ms, timestamp_ms, received_at_ms, is_closed.
    """

    def _make(
        index: int,
        *,
        open_: float = 100.0,
        high: float | None = None,
        low: float | None = None,
        close: float | None = None,
        volume: float = 1_000.0,
        quote_volume: float | None = None,
        trades_count: int | None = None,
        timestamp: datetime | str | int | float | None = None,
        timestamp_ms: int | None = None,
        open_time_ms: int | None = None,
        close_time_ms: int | None = None,
        received_at_ms: int | None = None,
        is_closed: bool = True,
        exchange_: str | None = None,
        market_type_: str | None = None,
        symbol_: str | None = None,
        exchange_symbol_: str | None = None,
        timeframe_: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_close = close if close is not None else open_ + 0.25
        resolved_high = high if high is not None else max(open_, resolved_close) + 0.50
        resolved_low = low if low is not None else min(open_, resolved_close) - 0.50

        open_time, close_time = _minute_bounds(start_time, index)
        resolved_open_time_ms = (
            int(open_time_ms) if open_time_ms is not None else _to_epoch_ms(open_time)
        )
        resolved_close_time_ms = (
            int(close_time_ms) if close_time_ms is not None else _to_epoch_ms(close_time)
        )
        resolved_timestamp_ms = (
            int(timestamp_ms)
            if timestamp_ms is not None
            else resolved_close_time_ms
        )
        resolved_received_at_ms = (
            int(received_at_ms)
            if received_at_ms is not None
            else resolved_timestamp_ms + 25
        )

        resolved_exchange = exchange_ if exchange_ is not None else exchange
        resolved_market_type = market_type_ if market_type_ is not None else market_type
        resolved_symbol = symbol_ if symbol_ is not None else symbol
        resolved_exchange_symbol = (
            exchange_symbol_ if exchange_symbol_ is not None else exchange_symbol
        )
        resolved_timeframe = timeframe_ if timeframe_ is not None else timeframe

        payload_metadata = {
            "open_time_ms": resolved_open_time_ms,
            "close_time_ms": resolved_close_time_ms,
            "timestamp_ms": resolved_timestamp_ms,
            "received_at_ms": resolved_received_at_ms,
            "source": "pytest.candle_factory",
        }
        payload_metadata.update(metadata or {})

        return {
            "exchange": resolved_exchange,
            "market_type": resolved_market_type,
            "symbol": resolved_symbol,
            "exchange_symbol": resolved_exchange_symbol,
            "timeframe": resolved_timeframe,
            "timestamp": timestamp or close_time,
            "timestamp_ms": resolved_timestamp_ms,
            "open_time_ms": resolved_open_time_ms,
            "close_time_ms": resolved_close_time_ms,
            "received_at_ms": resolved_received_at_ms,
            "open": float(open_),
            "high": float(resolved_high),
            "low": float(resolved_low),
            "close": float(resolved_close),
            "volume": float(volume),
            "quote_volume": (
                float(quote_volume)
                if quote_volume is not None
                else float(volume) * float(resolved_close)
            ),
            "trades_count": int(trades_count) if trades_count is not None else 100 + index,
            "is_closed": bool(is_closed),
            "index": int(index),
            "metadata": payload_metadata,
        }

    return _make


@pytest.fixture
def wrong_scope_candle_factory(
    candle_factory: Callable[..., dict[str, Any]],
    alt_exchange: str,
    alt_market_type: str,
    alt_symbol: str,
    alt_exchange_symbol: str,
) -> Callable[..., dict[str, Any]]:
    """
    Factory for intentionally wrong-scope candles.

    Useful for tests that assert scoped handlers ignore or reject data from
    another exchange, market type or symbol.
    """

    def _make(
        index: int,
        *,
        wrong_exchange: bool = False,
        wrong_market_type: bool = False,
        wrong_symbol: bool = False,
        spot: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        if wrong_exchange:
            overrides.setdefault("exchange_", alt_exchange)

        if wrong_market_type:
            overrides.setdefault("market_type_", alt_market_type)

        if spot:
            overrides.setdefault("market_type_", TEST_SPOT_MARKET_TYPE)

        if wrong_symbol:
            overrides.setdefault("symbol_", alt_symbol)
            overrides.setdefault("exchange_symbol_", alt_exchange_symbol)

        return candle_factory(index, **overrides)

    return _make


@pytest.fixture
def candles_updated_payload() -> Callable[..., dict[str, Any]]:
    """
    Build CandlesCache-style batch payloads for market.candles.updated events.
    """

    def _make(
        candles: Sequence[dict[str, Any]],
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        exchange_symbol: str | None = None,
        timeframe: str | None = None,
        source: str = "CandlesCache",
        update_reason: str = "test",
    ) -> dict[str, Any]:
        first = candles[0] if candles else {}

        resolved_exchange = exchange or str(first.get("exchange") or TEST_EXCHANGE)
        resolved_market_type = market_type or str(first.get("market_type") or TEST_MARKET_TYPE)
        resolved_symbol = symbol or str(first.get("symbol") or TEST_SYMBOL)
        resolved_exchange_symbol = (
            exchange_symbol
            or str(first.get("exchange_symbol") or resolved_symbol)
        )
        resolved_timeframe = timeframe or str(first.get("timeframe") or TEST_TIMEFRAME)

        return {
            **_scope_payload(
                exchange=resolved_exchange,
                market_type=resolved_market_type,
                symbol=resolved_symbol,
                exchange_symbol=resolved_exchange_symbol,
                timeframe=resolved_timeframe,
            ),
            "candles": list(candles),
            "count": len(candles),
            "source": source,
            "update_reason": update_reason,
        }

    return _make


@pytest.fixture
def candle_closed_event_payload(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """
    Build a single closed-candle payload exactly like a CandlesCache emission.
    """

    def _make(index: int, **overrides: Any) -> dict[str, Any]:
        overrides.setdefault("is_closed", True)
        return candle_factory(index, **overrides)

    return _make


@pytest.fixture
def rising_candles(
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., list[dict[str, Any]]]:
    """
    Controlled bullish sequence.
    """

    def _make(
        count: int,
        *,
        start: float = 100.0,
        step: float = 0.75,
        index_offset: int = 0,
        volume: float = 1_000.0,
        **scope_overrides: Any,
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
                    **scope_overrides,
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
    """

    def _make(
        count: int,
        *,
        start: float = 100.0,
        step: float = 0.75,
        index_offset: int = 0,
        volume: float = 1_000.0,
        **scope_overrides: Any,
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
                    **scope_overrides,
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
    """

    def _make(
        count: int,
        *,
        center: float = 100.0,
        amplitude: float = 0.35,
        index_offset: int = 0,
        volume: float = 1_000.0,
        **scope_overrides: Any,
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
                    **scope_overrides,
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

    The default shape creates visible local highs/lows for small
    pivot_left/pivot_right settings.
    """

    def _make(
        *,
        prices: Sequence[float] | None = None,
        index_offset: int = 0,
        **scope_overrides: Any,
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
                    **scope_overrides,
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

    Typical bullish gap condition:
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
def swing_factory(
    start_time: datetime,
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> Callable[..., dict[str, Any]]:
    """
    Factory for raw SwingPoint-like mappings.

    The shape mirrors serialized swing events emitted by MarketStructureAnalyzer.
    Full scope is included so SupportResistanceAnalyzer and LiquidityLevelsAnalyzer
    can enforce event-scope filtering.
    """

    def _make(
        index: int,
        *,
        price: float = 100.0,
        swing_type: SwingType | str = SwingType.HIGH,
        layer: StructureLayer | str = StructureLayer.INTERNAL,
        strength: float = 0.75,
        swing_id: str | None = None,
        exchange_: str | None = None,
        market_type_: str | None = None,
        symbol_: str | None = None,
        exchange_symbol_: str | None = None,
        timeframe_: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_swing_type = (
            swing_type.value if isinstance(swing_type, SwingType) else str(swing_type)
        )
        resolved_layer = layer.value if isinstance(layer, StructureLayer) else str(layer)

        resolved_exchange = exchange_ if exchange_ is not None else exchange
        resolved_market_type = market_type_ if market_type_ is not None else market_type
        resolved_symbol = symbol_ if symbol_ is not None else symbol
        resolved_exchange_symbol = (
            exchange_symbol_ if exchange_symbol_ is not None else exchange_symbol
        )
        resolved_timeframe = timeframe_ if timeframe_ is not None else timeframe

        timestamp = start_time + timedelta(minutes=index)

        return {
            **_scope_payload(
                exchange=resolved_exchange,
                market_type=resolved_market_type,
                symbol=resolved_symbol,
                exchange_symbol=resolved_exchange_symbol,
                timeframe=resolved_timeframe,
            ),
            "swing_id": swing_id or f"{resolved_layer}-{resolved_swing_type}-{index}",
            "timestamp": timestamp,
            "timestamp_ms": _to_epoch_ms(timestamp),
            "price": float(price),
            "swing_type": resolved_swing_type,
            "layer": resolved_layer,
            "index": int(index),
            "candle_open": float(price - 0.25),
            "candle_high": float(price + 0.50),
            "candle_low": float(price - 0.50),
            "candle_close": float(price + 0.25),
            "strength": float(strength),
            "is_confirmed": True,
            "metadata": {
                "source": "pytest.swing_factory",
                **dict(metadata or {}),
            },
        }

    return _make


@pytest.fixture
def wrong_scope_swing_factory(
    swing_factory: Callable[..., dict[str, Any]],
    alt_exchange: str,
    alt_market_type: str,
    alt_symbol: str,
    alt_exchange_symbol: str,
) -> Callable[..., dict[str, Any]]:
    def _make(
        index: int,
        *,
        wrong_exchange: bool = False,
        wrong_market_type: bool = False,
        wrong_symbol: bool = False,
        spot: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        if wrong_exchange:
            overrides.setdefault("exchange_", alt_exchange)

        if wrong_market_type:
            overrides.setdefault("market_type_", alt_market_type)

        if spot:
            overrides.setdefault("market_type_", TEST_SPOT_MARKET_TYPE)

        if wrong_symbol:
            overrides.setdefault("symbol_", alt_symbol)
            overrides.setdefault("exchange_symbol_", alt_exchange_symbol)

        return swing_factory(index, **overrides)

    return _make


@pytest.fixture
def market_structure_update_payload(
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return {
        **_scope_payload(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
        ),
        "state": {
            **_scope_payload(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                exchange_symbol=exchange_symbol,
                timeframe=timeframe,
            ),
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
def support_resistance_update_payload(
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    return {
        **_scope_payload(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
        ),
        "state": {
            **_scope_payload(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                exchange_symbol=exchange_symbol,
                timeframe=timeframe,
            ),
            "last_price": 101.25,
            "nearest_support": 99.50,
            "nearest_resistance": 103.00,
        },
        "updated_levels_count": 1,
        "new_events_count": 0,
    }


@pytest.fixture
def child_update_payload_factory(
    exchange: str,
    market_type: str,
    symbol: str,
    exchange_symbol: str,
    timeframe: str,
) -> Callable[..., dict[str, Any]]:
    """
    Generic scoped child update payload factory for facade tests.
    """

    def _make(
        *,
        module_name: str,
        last_price: float = 100.0,
        exchange_: str | None = None,
        market_type_: str | None = None,
        symbol_: str | None = None,
        exchange_symbol_: str | None = None,
        timeframe_: str | None = None,
        state_overrides: dict[str, Any] | None = None,
        payload_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_exchange = exchange_ if exchange_ is not None else exchange
        resolved_market_type = market_type_ if market_type_ is not None else market_type
        resolved_symbol = symbol_ if symbol_ is not None else symbol
        resolved_exchange_symbol = (
            exchange_symbol_ if exchange_symbol_ is not None else exchange_symbol
        )
        resolved_timeframe = timeframe_ if timeframe_ is not None else timeframe

        scope = _scope_payload(
            exchange=resolved_exchange,
            market_type=resolved_market_type,
            symbol=resolved_symbol,
            exchange_symbol=resolved_exchange_symbol,
            timeframe=resolved_timeframe,
        )

        state = {
            **scope,
            "last_price": float(last_price),
            "metadata": {
                "child_module": module_name,
                "source": "pytest.child_update_payload_factory",
            },
        }
        state.update(state_overrides or {})

        payload = {
            **scope,
            "state": state,
            "updated_module": module_name,
            "new_events_count": 0,
        }
        payload.update(payload_overrides or {})
        return payload

    return _make


@pytest.fixture
def event_factory() -> Callable[..., Event]:
    """
    Factory for core Event payloads.
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


@pytest.fixture
def candle_closed_event_factory(
    event_factory: Callable[..., Event],
    candle_factory: Callable[..., dict[str, Any]],
) -> Callable[..., Event]:
    def _make(index: int, **overrides: Any) -> Event:
        payload = candle_factory(index, is_closed=True, **overrides)
        return event_factory(
            "market.candle.closed",
            payload,
            source="CandlesCache",
            correlation_id=f"closed-candle-{index}",
        )

    return _make


@pytest.fixture
def candles_updated_event_factory(
    event_factory: Callable[..., Event],
    candles_updated_payload: Callable[..., dict[str, Any]],
) -> Callable[..., Event]:
    def _make(
        candles: Sequence[dict[str, Any]],
        *,
        correlation_id: str = "candles-updated",
        **payload_overrides: Any,
    ) -> Event:
        return event_factory(
            "market.candles.updated",
            candles_updated_payload(candles, **payload_overrides),
            source="CandlesCache",
            correlation_id=correlation_id,
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
        market_candle_topic="market.candle.closed",
        market_candles_topic="market.candles.updated",
        require_event_scope=True,
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
        market_candle_topic="market.candle.closed",
        market_candles_topic="market.candles.updated",
        require_event_scope=True,
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
        market_candle_topic="market.candle.closed",
        market_candles_topic="market.candles.updated",
        require_event_scope=True,
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
        market_candle_topic="market.candle.closed",
        market_candles_topic="market.candles.updated",
        require_event_scope=True,
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
        market_candle_topic="market.candle.closed",
        market_candles_topic="market.candles.updated",
        require_event_scope=True,
        subscribe_market_structure=True,
        subscribe_support_resistance=True,
    )


@pytest.fixture
def unscoped_market_structure_config(
    market_structure_config: MarketStructureConfig,
) -> MarketStructureConfig:
    market_structure_config.require_event_scope = False
    return market_structure_config


@pytest.fixture
def unscoped_support_resistance_config(
    support_resistance_config: SupportResistanceConfig,
) -> SupportResistanceConfig:
    support_resistance_config.require_event_scope = False
    return support_resistance_config


@pytest.fixture
def unscoped_fair_value_gap_config(
    fair_value_gap_config: FairValueGapConfig,
) -> FairValueGapConfig:
    fair_value_gap_config.require_event_scope = False
    return fair_value_gap_config


@pytest.fixture
def unscoped_liquidity_levels_config(
    liquidity_levels_config: LiquidityLevelsConfig,
) -> LiquidityLevelsConfig:
    liquidity_levels_config.require_event_scope = False
    return liquidity_levels_config


@pytest.fixture
def unscoped_trend_config(
    trend_config: TrendConfig,
) -> TrendConfig:
    trend_config.require_event_scope = False
    return trend_config


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

    Facade does not consume market candles directly. Child modules own data-layer
    candle subscriptions.
    """
    return PriceActionAnalyzerConfig(
        emit_events=True,
        event_namespace="analytics.price_action",
        publish_snapshots=False,
        snapshot_interval_seconds=None,
        subscribe_market_candles=False,
        require_event_scope=True,
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
        assert "exchange" in snapshot
        assert "market_type" in snapshot
        assert "symbol" in snapshot
        assert "exchange_symbol" in snapshot
        assert "timeframe" in snapshot
        assert "state" in snapshot
        assert "metadata" in snapshot
        assert snapshot["exchange"]
        assert snapshot["market_type"]
        assert snapshot["symbol"]
        assert snapshot["timeframe"]
        assert isinstance(snapshot["metadata"], dict)

    return _assert


@pytest.fixture
def assert_scope_matches() -> Callable[[dict[str, Any], dict[str, Any]], None]:
    """
    Assert that a serialized payload carries the expected price-action scope.
    """

    def _assert(payload: dict[str, Any], expected_scope: dict[str, Any]) -> None:
        assert payload["exchange"] == expected_scope["exchange"]
        assert payload["market_type"] == expected_scope["market_type"]
        assert payload["symbol"] == expected_scope["symbol"]
        assert payload["exchange_symbol"] == expected_scope["exchange_symbol"]
        assert payload["timeframe"] == expected_scope["timeframe"]

        if "key" in payload:
            assert list(payload["key"]) == list(expected_scope["key"])

    return _assert


@pytest.fixture
def assert_no_duplicate_ids() -> Callable[[Iterable[dict[str, Any]], str], None]:
    def _assert(items: Iterable[dict[str, Any]], id_key: str) -> None:
        ids = [item[id_key] for item in items if id_key in item]
        assert len(ids) == len(set(ids))

    return _assert


@pytest.fixture
def assert_confidences_are_bounded() -> Callable[[Iterable[dict[str, Any]]], None]:
    def _assert(items: Iterable[dict[str, Any]]) -> None:
        for item in items:
            if "confidence" in item:
                assert 0.0 <= float(item["confidence"]) <= 1.0

    return _assert