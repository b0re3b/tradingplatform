from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Deque, Iterable

from analytics.liquidations.enums import LiquidationSide
from analytics.liquidations.models import (
    DECIMAL_ZERO,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidationBufferSnapshot,
    LiquidationEvent,
    LiquidationKey,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)
from analytics.liquidations.utils import ensure_utc, prune_events_older_than, utc_now


@dataclass(slots=True)
class SymbolLiquidationState:
    """
    Оперативний in-memory state для одного liquidation scope.

    Canonical scope:
        exchange + market_type + symbol + timeframe

    Відповідальність:
    - тримати recent liquidation events у bounded deque;
    - вести counters long/short events;
    - зберігати timestamps останніх подій;
    - зберігати cooldown state після cascade detection.

    Цей клас не має знати про EventBus, Scheduler, logger або trading decisions.
    """

    exchange: str
    symbol: str
    market_type: str = DEFAULT_MARKET_TYPE
    timeframe: str = DEFAULT_TIMEFRAME
    exchange_symbol: str | None = None
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

        self.exchange = normalize_exchange(self.exchange)
        self.symbol = normalize_symbol(self.symbol)
        self.market_type = normalize_market_type(self.market_type)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.exchange_symbol = normalize_exchange_symbol(
            self.exchange_symbol,
            fallback_symbol=self.symbol,
        )

        self.events = deque(maxlen=self.max_events)

    @property
    def key(self) -> LiquidationKey:
        return make_liquidation_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def liquidation_key(self) -> LiquidationKey:
        return self.key

    @property
    def symbol_key(self) -> tuple[str, str]:
        """
        Backward-compatible legacy key.

        Новий код має використовувати .key.
        """
        return self.exchange, self.symbol

    @property
    def scope(self) -> dict[str, str]:
        scope = liquidation_key_to_dict(self.key)
        scope["exchange_symbol"] = self.exchange_symbol or self.symbol
        return scope

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
    def buffered_notional_usd(self) -> Decimal:
        total = DECIMAL_ZERO
        for event in self.events:
            total += event.notional_usd
        return total

    def add_event(self, event: LiquidationEvent) -> None:
        """
        Додає liquidation event до bounded buffer.

        Якщо deque переповнюється, найстаріший event автоматично витісняється,
        а side counters коректно оновлюються.
        """
        if event.key != self.key:
            raise ValueError(
                "Event key mismatch: "
                f"expected={liquidation_key_to_dict(self.key)}, "
                f"got={liquidation_key_to_dict(event.key)}"
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
            event
            for event in self.events
            if ensure_utc(event.timestamp) >= min_ts
        ]

        self._rebuild_from_events(kept_events, preserve_total_seen=True)

        return old_len - len(self.events)

    def clear(self, *, reset_total_seen: bool = True) -> None:
        """
        Повністю очищає state для scope.
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
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
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
                "scope": liquidation_key_to_dict(self.key),
                "exchange_symbol": self.exchange_symbol,
                "is_in_cooldown": self.is_in_cooldown(),
                "buffered_notional_usd": str(self.buffered_notional_usd),
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
            if event.key != self.key:
                raise ValueError(
                    "Event key mismatch while rebuilding state: "
                    f"expected={liquidation_key_to_dict(self.key)}, "
                    f"got={liquidation_key_to_dict(event.key)}"
                )

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

    Canonical key:
        LiquidationKey = exchange + market_type + symbol + timeframe

    Значення:
        SymbolLiquidationState

    Цей клас не має залежати від EventBus/Scheduler/logger.
    Runtime-класи самі вирішують, коли публікувати snapshots або запускати cleanup.
    """

    max_events_per_symbol: int = 5000
    symbols: dict[LiquidationKey, SymbolLiquidationState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_events_per_symbol <= 0:
            raise ValueError("max_events_per_symbol must be > 0")

    @property
    def symbols_count(self) -> int:
        return len(self.symbols)

    @property
    def scopes_count(self) -> int:
        return len(self.symbols)

    @property
    def total_buffered_events(self) -> int:
        return sum(state.total_buffered_events for state in self.symbols.values())

    @property
    def total_events_seen(self) -> int:
        return sum(state.total_events_seen for state in self.symbols.values())

    def make_key(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> LiquidationKey:
        return make_liquidation_key(
            exchange=exchange,
            market_type=market_type or DEFAULT_MARKET_TYPE,
            symbol=symbol,
            timeframe=timeframe or DEFAULT_TIMEFRAME,
        )

    def get_or_create(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
        exchange_symbol: str | None = None,
    ) -> SymbolLiquidationState:
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        if key not in self.symbols:
            scope = liquidation_key_to_dict(key)
            normalized_symbol = scope["symbol"]

            self.symbols[key] = SymbolLiquidationState(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=normalized_symbol,
                timeframe=scope["timeframe"],
                exchange_symbol=normalize_exchange_symbol(
                    exchange_symbol,
                    fallback_symbol=normalized_symbol,
                ),
                max_events=self.max_events_per_symbol,
            )

        return self.symbols[key]

    def get_or_create_key(
        self,
        key: LiquidationKey,
        *,
        exchange_symbol: str | None = None,
    ) -> SymbolLiquidationState:
        scope = liquidation_key_to_dict(key)

        return self.get_or_create(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
            exchange_symbol=exchange_symbol,
        )

    def add_event(self, event: LiquidationEvent) -> SymbolLiquidationState:
        symbol_state = self.get_or_create(
            exchange=event.exchange,
            market_type=event.market_type,
            symbol=event.symbol,
            timeframe=event.timeframe,
            exchange_symbol=event.exchange_symbol,
        )
        symbol_state.add_event(event)
        return symbol_state

    def add_events(self, events: Iterable[LiquidationEvent]) -> None:
        for event in events:
            self.add_event(event)

    def get(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> SymbolLiquidationState | None:
        return self.symbols.get(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def get_key(self, key: LiquidationKey) -> SymbolLiquidationState | None:
        return self.symbols.get(key)

    def remove(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.symbols.pop(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            ),
            None,
        )

    def remove_key(self, key: LiquidationKey) -> None:
        self.symbols.pop(key, None)

    def remove_empty(self) -> int:
        """
        Видаляє порожні scoped states.

        Повертає кількість видалених states.
        """
        empty_keys = [
            key
            for key, symbol_state in self.symbols.items()
            if symbol_state.is_empty
        ]

        for key in empty_keys:
            self.symbols.pop(key, None)

        return len(empty_keys)

    def prune_before(self, min_timestamp: datetime) -> int:
        """
        Видаляє старі events у всіх scoped states.

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
        market_type: str | None = None,
        timeframe: str | None = None,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        """
        Повертає recent events з усього state або з конкретного scope.

        Якщо передані тільки exchange/symbol без market_type/timeframe,
        повертаються всі matching scopes для цього exchange/symbol.
        """
        if limit <= 0:
            return []

        states = self._select_states(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        events: list[LiquidationEvent] = []
        for symbol_state in states:
            events.extend(symbol_state.get_recent_events(side=side))

        events.sort(key=lambda event: ensure_utc(event.timestamp), reverse=True)
        return events[:limit]

    def get_recent_events_for_key(
        self,
        key: LiquidationKey,
        *,
        side: LiquidationSide | None = None,
        limit: int = 100,
    ) -> list[LiquidationEvent]:
        state = self.get_key(key)
        if state is None:
            return []

        return state.get_recent_events(side=side, limit=limit)

    def snapshots(self) -> list[LiquidationBufferSnapshot]:
        return [state.snapshot() for state in self.symbols.values()]

    def snapshot_by_key(
        self,
        key: LiquidationKey,
    ) -> LiquidationBufferSnapshot | None:
        symbol_state = self.get_key(key)
        if symbol_state is None:
            return None

        return symbol_state.snapshot()

    def snapshot_by_symbol(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> LiquidationBufferSnapshot | None:
        symbol_state = self.get(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        if symbol_state is None:
            return None

        return symbol_state.snapshot()

    def clear(self, *, reset_total_seen: bool = True) -> None:
        if reset_total_seen:
            self.symbols.clear()
            return

        for symbol_state in self.symbols.values():
            symbol_state.clear(reset_total_seen=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "scopes_count": self.scopes_count,
            "symbols_count": self.symbols_count,
            "total_buffered_events": self.total_buffered_events,
            "total_events_seen": self.total_events_seen,
            "max_events_per_symbol": self.max_events_per_symbol,
            "scopes": {
                self._key_to_string(key): state.snapshot().to_dict()
                for key, state in self.symbols.items()
            },
        }

    def _select_states(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[SymbolLiquidationState]:
        normalized_exchange = (
            normalize_exchange(exchange)
            if exchange is not None
            else None
        )
        normalized_symbol = (
            normalize_symbol(symbol)
            if symbol is not None
            else None
        )
        normalized_market_type = (
            normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        selected: list[SymbolLiquidationState] = []

        for key, state in self.symbols.items():
            scope = liquidation_key_to_dict(key)

            if normalized_exchange is not None and scope["exchange"] != normalized_exchange:
                continue

            if normalized_market_type is not None and scope["market_type"] != normalized_market_type:
                continue

            if normalized_symbol is not None and scope["symbol"] != normalized_symbol:
                continue

            if normalized_timeframe is not None and scope["timeframe"] != normalized_timeframe:
                continue

            selected.append(state)

        return selected

    @staticmethod
    def _key_to_string(key: LiquidationKey) -> str:
        scope = liquidation_key_to_dict(key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )


_SHARED_LIQUIDATION_STATE: LiquidationState | None = None


def get_shared_liquidation_state(
    *,
    max_events_per_symbol: int = 5000,
    reset: bool = False,
) -> LiquidationState:
    """Return the package-level shared liquidation state.

    This keeps LiquidationStream and CascadeDetector on the same in-memory
    buffers without adding a separate pipeline module. Factory/bootstrap code can
    still pass an explicit LiquidationState when it wants full ownership.
    """
    global _SHARED_LIQUIDATION_STATE

    if reset or _SHARED_LIQUIDATION_STATE is None:
        _SHARED_LIQUIDATION_STATE = LiquidationState(
            max_events_per_symbol=max_events_per_symbol,
        )

    return _SHARED_LIQUIDATION_STATE


def reset_shared_liquidation_state(
    *,
    max_events_per_symbol: int = 5000,
) -> LiquidationState:
    """Replace and return the package-level shared liquidation state."""
    return get_shared_liquidation_state(
        max_events_per_symbol=max_events_per_symbol,
        reset=True,
    )


__all__ = [
    "SymbolLiquidationState",
    "LiquidationState",
    "get_shared_liquidation_state",
    "reset_shared_liquidation_state",
]
