from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Iterable

from .enums import LiquidationSide
from .models import LiquidationBufferSnapshot, LiquidationEvent
from .utils import build_symbol_key, ensure_utc, prune_events_older_than, utc_now


@dataclass(slots=True)
class SymbolLiquidationState:
    """
    Оперативний in-memory state для одного (exchange, symbol).

    Відповідальність:
    - тримати recent liquidation events у bounded deque;
    - вести counters long/short events;
    - зберігати timestamps останніх подій;
    - зберігати cooldown state після cascade detection.

    Цей клас не має знати про EventBus, Scheduler, logger або trading decisions.
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
        if self.max_events <= 0:
            raise ValueError("max_events must be > 0")

        normalized_exchange, normalized_symbol = build_symbol_key(self.exchange, self.symbol)
        self.exchange = normalized_exchange
        self.symbol = normalized_symbol

        self.events = deque(maxlen=self.max_events)

    @property
    def key(self) -> tuple[str, str]:
        return self.exchange, self.symbol

    @property
    def total_buffered_events(self) -> int:
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        return not self.events

    @property
    def first_event_at(self) -> datetime | None:
        return ensure_utc(self.events[0].timestamp) if self.events else None

    @property
    def buffered_notional_usd(self):
        total = sum((event.notional_usd for event in self.events), start=0)
        return total

    def add_event(self, event: LiquidationEvent) -> None:
        """
        Додає liquidation event до bounded buffer.

        Якщо deque переповнюється, найстаріший event автоматично витісняється,
        а side counters коректно оновлюються.
        """
        if event.symbol_key != self.key:
            raise ValueError(
                f"Event key mismatch: expected={self.key}, got={event.symbol_key}"
            )

        if len(self.events) == self.max_events:
            self._remove_oldest_from_counters()

        self.events.append(event)
        self.total_events_seen += 1

        event_ts = ensure_utc(event.timestamp)
        self.last_event_at = event_ts

        if event.side is LiquidationSide.LONG:
            self.long_events_count += 1
            self.last_long_event_at = event_ts
        elif event.side is LiquidationSide.SHORT:
            self.short_events_count += 1
            self.last_short_event_at = event_ts

    def extend_events(self, events: Iterable[LiquidationEvent]) -> None:
        """
        Додає кілька events послідовно.
        """
        for event in events:
            self.add_event(event)

    def set_cascade_detected(
        self,
        detected_at: datetime,
        cooldown_until: datetime | None,
    ) -> None:
        self.last_cascade_at = ensure_utc(detected_at)
        self.cooldown_until = ensure_utc(cooldown_until) if cooldown_until else None

    def clear_cooldown(self) -> None:
        self.cooldown_until = None

    def is_in_cooldown(self, now: datetime | None = None) -> bool:
        if self.cooldown_until is None:
            return False

        current_time = ensure_utc(now or utc_now())
        return current_time < ensure_utc(self.cooldown_until)

    def get_recent_events(
        self,
        *,
        side: LiquidationSide | None = None,
        limit: int | None = None,
    ) -> list[LiquidationEvent]:
        """
        Повертає recent events у зворотному порядку: від нових до старих.
        """
        result: list[LiquidationEvent] = []

        for event in reversed(self.events):
            if side is not None and event.side is not side:
                continue

            result.append(event)

            if limit is not None and len(result) >= limit:
                break

        return result

    def get_window_events(self, *, min_timestamp: datetime) -> list[LiquidationEvent]:
        """
        Повертає events, що входять у часовий window.
        """
        return prune_events_older_than(
            list(self.events),
            min_timestamp=ensure_utc(min_timestamp),
        )

    def prune_before(self, min_timestamp: datetime) -> int:
        """
        Видаляє buffered events, старіші за min_timestamp.

        Повертає кількість видалених events.
        """
        min_ts = ensure_utc(min_timestamp)
        old_len = len(self.events)

        kept_events = [
            event for event in self.events
            if ensure_utc(event.timestamp) >= min_ts
        ]

        self._rebuild_from_events(kept_events, preserve_total_seen=True)

        return old_len - len(self.events)

    def clear(self, *, reset_total_seen: bool = True) -> None:
        """
        Повністю очищає state для символу.
        """
        self.events.clear()

        self.long_events_count = 0
        self.short_events_count = 0

        if reset_total_seen:
            self.total_events_seen = 0

        self.last_event_at = None
        self.last_long_event_at = None
        self.last_short_event_at = None
        self.last_cascade_at = None
        self.cooldown_until = None

    def snapshot(self) -> LiquidationBufferSnapshot:
        return LiquidationBufferSnapshot(
            exchange=self.exchange,
            symbol=self.symbol,
            total_buffered_events=len(self.events),
            long_buffered_events=self.long_events_count,
            short_buffered_events=self.short_events_count,
            first_event_at=self.first_event_at,
            last_event_at=self.last_event_at,
            last_cascade_at=self.last_cascade_at,
            cooldown_until=self.cooldown_until,
            max_events=self.max_events,
            total_events_seen=self.total_events_seen,
            metadata={
                "is_in_cooldown": self.is_in_cooldown(),
            },
        )

    def _remove_oldest_from_counters(self) -> None:
        if not self.events:
            return

        oldest = self.events[0]

        if oldest.side is LiquidationSide.LONG:
            self.long_events_count = max(0, self.long_events_count - 1)
        elif oldest.side is LiquidationSide.SHORT:
            self.short_events_count = max(0, self.short_events_count - 1)

    def _rebuild_from_events(
        self,
        events: Iterable[LiquidationEvent],
        *,
        preserve_total_seen: bool,
    ) -> None:
        previous_total_seen = self.total_events_seen

        self.events.clear()
        self.long_events_count = 0
        self.short_events_count = 0
        self.last_event_at = None
        self.last_long_event_at = None
        self.last_short_event_at = None

        for event in events:
            if len(self.events) == self.max_events:
                self._remove_oldest_from_counters()

            self.events.append(event)

            event_ts = ensure_utc(event.timestamp)
            self.last_event_at = event_ts

            if event.side is LiquidationSide.LONG:
                self.long_events_count += 1
                self.last_long_event_at = event_ts
            elif event.side is LiquidationSide.SHORT:
                self.short_events_count += 1
                self.last_short_event_at = event_ts

        if preserve_total_seen:
            self.total_events_seen = previous_total_seen
        else:
            self.total_events_seen = len(self.events)


@dataclass(slots=True)
class LiquidationState:
    """
    Глобальний in-memory state liquidation-модуля.

    Ключ:
        (exchange, symbol)

    Значення:
        SymbolLiquidationState

    Цей клас не має залежати від EventBus/Scheduler/logger.
    Runtime-класи самі вирішують, коли публікувати snapshots або запускати cleanup.
    """

    max_events_per_symbol: int = 5000
    symbols: dict[tuple[str, str], SymbolLiquidationState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_events_per_symbol <= 0:
            raise ValueError("max_events_per_symbol must be > 0")

    @property
    def symbols_count(self) -> int:
        return len(self.symbols)

    @property
    def total_buffered_events(self) -> int:
        return sum(state.total_buffered_events for state in self.symbols.values())

    @property
    def total_events_seen(self) -> int:
        return sum(state.total_events_seen for state in self.symbols.values())

    def get_or_create(self, exchange: str, symbol: str) -> SymbolLiquidationState:
        key = build_symbol_key(exchange, symbol)

        if key not in self.symbols:
            self.symbols[key] = SymbolLiquidationState(
                exchange=key[0],
                symbol=key[1],
                max_events=self.max_events_per_symbol,
            )

        return self.symbols[key]

    def add_event(self, event: LiquidationEvent) -> SymbolLiquidationState:
        symbol_state = self.get_or_create(event.exchange, event.symbol)
        symbol_state.add_event(event)
        return symbol_state

    def add_events(self, events: Iterable[LiquidationEvent]) -> None:
        for event in events:
            self.add_event(event)

    def get(self, exchange: str, symbol: str) -> SymbolLiquidationState | None:
        return self.symbols.get(build_symbol_key(exchange, symbol))

    def remove(self, exchange: str, symbol: str) -> None:
        self.symbols.pop(build_symbol_key(exchange, symbol), None)

    def remove_empty(self) -> int:
        """
        Видаляє порожні symbol states.

        Повертає кількість видалених states.
        """
        empty_keys = [
            key for key, symbol_state in self.symbols.items()
            if symbol_state.is_empty
        ]

        for key in empty_keys:
            self.symbols.pop(key, None)

        return len(empty_keys)

    def prune_before(self, min_timestamp: datetime) -> int:
        """
        Видаляє старі events у всіх symbol states.

        Повертає загальну кількість видалених events.
        """
        removed_total = 0

        for symbol_state in self.symbols.values():
            removed_total += symbol_state.prune_before(min_timestamp)

        self.remove_empty()
        return removed_total

    def get_recent_events(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        """
        Повертає recent events з усього state або з конкретного exchange/symbol.
        """
        if limit <= 0:
            return []

        states = self._select_states(exchange=exchange, symbol=symbol)

        events: list[LiquidationEvent] = []
        for symbol_state in states:
            events.extend(symbol_state.get_recent_events(side=side))

        events.sort(key=lambda event: ensure_utc(event.timestamp), reverse=True)
        return events[:limit]

    def snapshots(self) -> list[LiquidationBufferSnapshot]:
        return [state.snapshot() for state in self.symbols.values()]

    def snapshot_by_symbol(
        self,
        exchange: str,
        symbol: str,
    ) -> LiquidationBufferSnapshot | None:
        symbol_state = self.get(exchange, symbol)
        if symbol_state is None:
            return None

        return symbol_state.snapshot()

    def clear(self, *, reset_total_seen: bool = True) -> None:
        if reset_total_seen:
            self.symbols.clear()
            return

        for symbol_state in self.symbols.values():
            symbol_state.clear(reset_total_seen=False)

    def _select_states(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
    ) -> list[SymbolLiquidationState]:
        if exchange is not None and symbol is not None:
            state = self.get(exchange, symbol)
            return [state] if state is not None else []

        if exchange is not None:
            normalized_exchange = exchange.strip().lower()
            return [
                state
                for (state_exchange, _), state in self.symbols.items()
                if state_exchange == normalized_exchange
            ]

        if symbol is not None:
            normalized_symbol = symbol.strip().upper().replace("-", "").replace("/", "")
            return [
                state
                for (_, state_symbol), state in self.symbols.items()
                if state_symbol == normalized_symbol
            ]

        return list(self.symbols.values())