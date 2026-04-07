from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

from .enums import LiquidationSide
from .models import LiquidationBufferSnapshot, LiquidationEvent


@dataclass(slots=True)
class SymbolLiquidationState:
    """
    Оперативний state для одного (exchange, symbol).

    Тут немає бізнес-логіки ліквідності.
    Тут лише буфер recent liquidation events, timestamps і cooldown state.
    """

    exchange: str
    symbol: str
    max_events: int = 5000

    events: Deque[LiquidationEvent] = field(init=False)
    long_events_count: int = 0
    short_events_count: int = 0
    total_events_seen: int = 0

    last_event_at: datetime | None = None
    last_long_event_at: datetime | None = None
    last_short_event_at: datetime | None = None
    last_cascade_at: datetime | None = None
    cooldown_until: datetime | None = None

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.max_events)

    def add_event(self, event: LiquidationEvent) -> None:
        """
        Додає liquidation event до буфера.
        Якщо deque переповнюється, старий елемент буде автоматично витіснений.
        """

        if len(self.events) == self.max_events:
            oldest = self.events[0]
            if oldest.side == LiquidationSide.LONG:
                self.long_events_count = max(0, self.long_events_count - 1)
            elif oldest.side == LiquidationSide.SHORT:
                self.short_events_count = max(0, self.short_events_count - 1)

        self.events.append(event)
        self.total_events_seen += 1
        self.last_event_at = event.timestamp

        if event.side == LiquidationSide.LONG:
            self.long_events_count += 1
            self.last_long_event_at = event.timestamp
        elif event.side == LiquidationSide.SHORT:
            self.short_events_count += 1
            self.last_short_event_at = event.timestamp

    def set_cascade_detected(self, detected_at: datetime, cooldown_until: datetime | None) -> None:
        self.last_cascade_at = detected_at
        self.cooldown_until = cooldown_until

    def is_in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and now < self.cooldown_until

    def clear(self) -> None:
        self.events.clear()
        self.long_events_count = 0
        self.short_events_count = 0
        self.last_event_at = None
        self.last_long_event_at = None
        self.last_short_event_at = None
        self.last_cascade_at = None
        self.cooldown_until = None

    def snapshot(self) -> LiquidationBufferSnapshot:
        first_event_at = self.events[0].timestamp if self.events else None
        last_event_at = self.events[-1].timestamp if self.events else None

        return LiquidationBufferSnapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            total_buffered_events=len(self.events),
            long_buffered_events=self.long_events_count,
            short_buffered_events=self.short_events_count,
            first_event_at=first_event_at,
            last_event_at=last_event_at,
            last_cascade_at=self.last_cascade_at,
            cooldown_until=self.cooldown_until,
        )


@dataclass(slots=True)
class LiquidationState:
    """
    Глобальний in-memory state liquidation-модуля.

    Ключ — (exchange, symbol), значення — SymbolLiquidationState.
    """

    max_events_per_symbol: int = 5000
    symbols: dict[tuple[str, str], SymbolLiquidationState] = field(default_factory=dict)

    def get_or_create(self, exchange: str, symbol: str) -> SymbolLiquidationState:
        key = (exchange, symbol)
        if key not in self.symbols:
            self.symbols[key] = SymbolLiquidationState(
                exchange=exchange,
                symbol=symbol,
                max_events=self.max_events_per_symbol,
            )
        return self.symbols[key]

    def add_event(self, event: LiquidationEvent) -> SymbolLiquidationState:
        symbol_state = self.get_or_create(event.exchange, event.symbol)
        symbol_state.add_event(event)
        return symbol_state

    def get(self, exchange: str, symbol: str) -> SymbolLiquidationState | None:
        return self.symbols.get((exchange, symbol))

    def remove(self, exchange: str, symbol: str) -> None:
        self.symbols.pop((exchange, symbol), None)

    def clear(self) -> None:
        self.symbols.clear()

    def snapshots(self) -> list[LiquidationBufferSnapshot]:
        return [state.snapshot() for state in self.symbols.values()]