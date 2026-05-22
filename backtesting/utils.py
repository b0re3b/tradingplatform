from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from core.event_bus import EventBus

from backtesting.exceptions import BacktestReplayError, BacktestSafetyError
from backtesting.models import ReplayEvent


_TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def timeframe_to_ms(timeframe: str) -> int:
    try:
        return _TIMEFRAME_MS[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def align_down_ms(timestamp_ms: int, timeframe: str) -> int:
    step = timeframe_to_ms(timeframe)
    return timestamp_ms - (timestamp_ms % step)


def default_period(lookback_days: int, *, smallest_timeframe: str = "1m") -> tuple[int, int]:
    end = utc_ms(utc_now())
    end = align_down_ms(end, smallest_timeframe) - 1
    start = end - int(timedelta(days=lookback_days).total_seconds() * 1000)
    return start, end


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "").replace("-", "")


def decimal_from(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


def bps_to_fraction(bps: Decimal) -> Decimal:
    return bps / Decimal("10000")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_backtest_safety(*, env_guard: str = "BACKTEST_MODE") -> None:
    os.environ[env_guard] = "true"
    live_enabled = os.getenv("LIVE_TRADING_ENABLED", "").strip().lower()
    if live_enabled in {"1", "true", "yes", "on"}:
        raise BacktestSafetyError(
            "LIVE_TRADING_ENABLED must not be enabled during backtest."
        )


async def drain_event_bus(
    event_bus: EventBus,
    *,
    require_public_join: bool = False,
    timeout: float | None = None,
    raise_on_timeout: bool = False,
) -> bool:
    """
    Try to wait until EventBus queue is drained.

    Returns:
        True  - queue drained successfully or no drain primitive exists.
        False - timeout happened but caller allowed non-fatal timeout.
    """
    join = getattr(event_bus, "join", None)

    try:
        if callable(join):
            result = join()
            if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                await asyncio.wait_for(result, timeout=timeout)
            return True

        if require_public_join:
            raise BacktestReplayError(
                "EventBus.join() is missing. Add a public join() method that awaits the internal queue."
            )

        queue = getattr(event_bus, "_queue", None)
        if queue is not None and hasattr(queue, "join"):
            await asyncio.wait_for(queue.join(), timeout=timeout)
            return True

        await asyncio.sleep(0)
        return True

    except asyncio.TimeoutError:
        if raise_on_timeout:
            raise
        return False


def sort_events_causally(events: Iterable[ReplayEvent]) -> list[ReplayEvent]:
    topic_rank = {
        "market.trade": 10,
        "market.mark_price": 20,
        "market.funding": 30,
        "market.open_interest": 40,
        "market.candle": 50,
    }
    return sorted(
        events,
        key=lambda item: (
            item.timestamp_ms,
            topic_rank.get(item.topic, 100),
            item.sequence,
        ),
    )


def pct(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return (numerator / denominator) * Decimal("100")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
