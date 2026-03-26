from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import EqualLevel, LiquidityLevel, LiquidityMapSnapshot, StopCluster


@dataclass(slots=True)
class LiquidityTimeframeState:
    """
    State для конкретного symbol + timeframe.
    """

    symbol: str
    timeframe: str

    active_levels: list[LiquidityLevel] = field(default_factory=list)
    equal_levels: list[EqualLevel] = field(default_factory=list)
    stop_clusters: list[StopCluster] = field(default_factory=list)

    last_snapshot: LiquidityMapSnapshot | None = None

    last_candle_open_time: datetime | None = None
    last_candle_close_time: datetime | None = None
    last_update_at: datetime | None = None

    processed_candles: int = 0
    processed_orderbook_updates: int = 0

    def touch(self, ts: datetime | None = None) -> None:
        self.last_update_at = ts or datetime.utcnow()

    def reset(self) -> None:
        self.active_levels.clear()
        self.equal_levels.clear()
        self.stop_clusters.clear()
        self.last_snapshot = None
        self.last_candle_open_time = None
        self.last_candle_close_time = None
        self.last_update_at = None
        self.processed_candles = 0
        self.processed_orderbook_updates = 0


@dataclass(slots=True)
class LiquidityState:
    """
    Загальний state liquidity-модуля.

    Ключ:
        "{symbol}:{timeframe}"
    """

    states: dict[str, LiquidityTimeframeState] = field(default_factory=dict)

    @staticmethod
    def make_key(symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def get(self, symbol: str, timeframe: str) -> LiquidityTimeframeState | None:
        return self.states.get(self.make_key(symbol, timeframe))

    def get_or_create(self, symbol: str, timeframe: str) -> LiquidityTimeframeState:
        key = self.make_key(symbol, timeframe)
        if key not in self.states:
            self.states[key] = LiquidityTimeframeState(symbol=symbol, timeframe=timeframe)
        return self.states[key]

    def remove(self, symbol: str, timeframe: str) -> None:
        self.states.pop(self.make_key(symbol, timeframe), None)

    def clear(self) -> None:
        self.states.clear()

    def keys(self) -> list[str]:
        return list(self.states.keys())

    def values(self) -> list[LiquidityTimeframeState]:
        return list(self.states.values())