from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .context import StrategyContext
from .enums import MarketRegime
from .exceptions import FeatureStoreError, StrategyStateError, ValidationError
from .models import (
    CooldownState,
    FeatureSnapshot,
    PortfolioSnapshot,
    RegimeSnapshot,
    StrategySignal,
)


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass(slots=True)
class SymbolState:
    """
    Стан по конкретному символу.
    """

    symbol: str
    last_context: StrategyContext | None = None
    last_signal: StrategySignal | None = None
    active_signals: list[StrategySignal] = field(default_factory=list)

    cooldowns: dict[str, CooldownState] = field(default_factory=dict)
    side_cooldowns: dict[str, CooldownState] = field(default_factory=dict)

    last_signal_by_side: dict[str, StrategySignal] = field(default_factory=dict)
    signal_history: list[StrategySignal] = field(default_factory=list)

    active_conflicts: list[str] = field(default_factory=list)
    last_updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("SymbolState.symbol cannot be empty")
        if self.last_context is not None:
            self.last_context.validate()
        if self.last_signal is not None:
            self.last_signal.validate()
        for signal in self.active_signals:
            signal.validate()
        for signal in self.last_signal_by_side.values():
            signal.validate()
        for signal in self.signal_history:
            signal.validate()
        for cooldown in self.cooldowns.values():
            cooldown.validate()
        for cooldown in self.side_cooldowns.values():
            cooldown.validate()

    def touch(self) -> None:
        self.last_updated_at = utcnow()

    def set_context(self, context: StrategyContext) -> None:
        context.validate()
        if context.symbol != self.symbol:
            raise StrategyStateError(
                f"Context symbol mismatch: expected {self.symbol}, got {context.symbol}"
            )
        self.last_context = context
        self.touch()

    def set_last_signal(self, signal: StrategySignal) -> None:
        signal.validate()
        if signal.symbol != self.symbol:
            raise StrategyStateError(
                f"Signal symbol mismatch: expected {self.symbol}, got {signal.symbol}"
            )
        self.last_signal = signal
        self.touch()

    def add_active_signal(self, signal: StrategySignal) -> None:
        signal.validate()
        if signal.symbol != self.symbol:
            raise StrategyStateError(
                f"Signal symbol mismatch: expected {self.symbol}, got {signal.symbol}"
            )
        self.active_signals.append(signal)
        self.last_signal = signal
        self.touch()

    def remove_inactive_signals(self) -> None:
        self.active_signals = [signal for signal in self.active_signals if signal.is_active]
        self.touch()

    def add_cooldown(self, strategy_name: str, seconds: int, reason: str | None = None) -> None:
        if seconds < 0:
            raise StrategyStateError("Cooldown seconds must be >= 0")

        self.cooldowns[strategy_name] = CooldownState(
            symbol=self.symbol,
            strategy_name=strategy_name,
            until=utcnow() + timedelta(seconds=seconds),
            reason=reason,
        )
        self.touch()

    def is_on_cooldown(self, strategy_name: str, now: datetime | None = None) -> bool:
        cooldown = self.cooldowns.get(strategy_name)
        if cooldown is None:
            return False
        return cooldown.is_active(now)

    def clear_expired_cooldowns(self, now: datetime | None = None) -> None:
        current = now or utcnow()
        self.cooldowns = {
            name: cooldown
            for name, cooldown in self.cooldowns.items()
            if cooldown.is_active(current)
        }
        self.touch()

    def add_side_cooldown(self, side: str, seconds: int, reason: str | None = None) -> None:
        if seconds < 0:
            raise StrategyStateError("Side cooldown seconds must be >= 0")

        self.side_cooldowns[side] = CooldownState(
            symbol=self.symbol,
            strategy_name=f"__side__:{side}",
            until=utcnow() + timedelta(seconds=seconds),
            reason=reason,
        )
        self.touch()

    def is_side_on_cooldown(self, side: str, now: datetime | None = None) -> bool:
        cooldown = self.side_cooldowns.get(side)
        if cooldown is None:
            return False
        return cooldown.is_active(now)

    def clear_expired_side_cooldowns(self, now: datetime | None = None) -> None:
        current = now or utcnow()
        self.side_cooldowns = {
            name: cooldown
            for name, cooldown in self.side_cooldowns.items()
            if cooldown.is_active(current)
        }
        self.touch()

    def remember_signal(self, signal: StrategySignal) -> None:
        signal.validate()
        if signal.symbol != self.symbol:
            raise StrategyStateError(
                f"Signal symbol mismatch: expected {self.symbol}, got {signal.symbol}"
            )

        self.last_signal_by_side[signal.side.value] = signal
        self.signal_history.append(signal)
        self.last_signal = signal
        self.touch()


@dataclass(slots=True)
class MarketRegimeState:
    """
    Окремий state для regime по символах.
    """

    regimes: dict[str, RegimeSnapshot] = field(default_factory=dict)
    updated_at: datetime | None = None

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()
        self.regimes[snapshot.symbol] = snapshot
        self.updated_at = utcnow()

    def get_regime_snapshot(self, symbol: str) -> RegimeSnapshot | None:
        return self.regimes.get(symbol)

    def get_regime(self, symbol: str) -> MarketRegime:
        snapshot = self.regimes.get(symbol)
        if snapshot is None:
            return MarketRegime.UNKNOWN
        return snapshot.regime

    def clear(self) -> None:
        self.regimes.clear()
        self.updated_at = utcnow()


@dataclass(slots=True)
class PortfolioState:
    """
    Локальний state strategy layer для портфеля.
    """

    snapshot: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)
    blocked_symbols: set[str] = field(default_factory=set)
    symbol_signal_counts: dict[str, int] = field(default_factory=dict)
    updated_at: datetime | None = None

    def set_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()
        self.snapshot = snapshot
        self.updated_at = utcnow()

    def block_symbol(self, symbol: str) -> None:
        self.blocked_symbols.add(symbol)
        self.updated_at = utcnow()

    def unblock_symbol(self, symbol: str) -> None:
        self.blocked_symbols.discard(symbol)
        self.updated_at = utcnow()

    def is_symbol_blocked(self, symbol: str) -> bool:
        return symbol in self.blocked_symbols or symbol in self.snapshot.blocked_symbols

    def increment_signal_count(self, symbol: str) -> None:
        self.symbol_signal_counts[symbol] = self.symbol_signal_counts.get(symbol, 0) + 1
        self.updated_at = utcnow()

    def reset_signal_count(self, symbol: str) -> None:
        self.symbol_signal_counts.pop(symbol, None)
        self.updated_at = utcnow()

    def get_signal_count(self, symbol: str) -> int:
        return self.symbol_signal_counts.get(symbol, 0)


class FeatureStore:
    """
    In-memory feature store для strategy layer.

    Зберігає останні нормалізовані feature snapshots по символах.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, FeatureSnapshot]] = {}
        self._updated_at: dict[str, datetime] = {}

    def upsert(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()
        symbol_features = self._store.setdefault(snapshot.symbol, {})
        symbol_features[snapshot.name] = snapshot
        self._updated_at[snapshot.symbol] = utcnow()

    def bulk_upsert(self, snapshots: list[FeatureSnapshot]) -> None:
        for snapshot in snapshots:
            self.upsert(snapshot)

    def get(self, symbol: str, feature_name: str) -> FeatureSnapshot | None:
        return self._store.get(symbol, {}).get(feature_name)

    def get_value(self, symbol: str, feature_name: str, default: Any = None) -> Any:
        snapshot = self.get(symbol, feature_name)
        if snapshot is None:
            return default
        return snapshot.value

    def get_normalized_value(
        self,
        symbol: str,
        feature_name: str,
        default: float | None = None,
    ) -> float | None:
        snapshot = self.get(symbol, feature_name)
        if snapshot is None:
            return default
        return snapshot.normalized_value if snapshot.normalized_value is not None else default

    def get_symbol_features(self, symbol: str) -> dict[str, FeatureSnapshot]:
        return dict(self._store.get(symbol, {}))

    def has_feature(self, symbol: str, feature_name: str) -> bool:
        return feature_name in self._store.get(symbol, {})

    def remove_feature(self, symbol: str, feature_name: str) -> None:
        symbol_features = self._store.get(symbol)
        if not symbol_features:
            return
        symbol_features.pop(feature_name, None)
        self._updated_at[symbol] = utcnow()

    def remove_symbol(self, symbol: str) -> None:
        self._store.pop(symbol, None)
        self._updated_at.pop(symbol, None)

    def clear(self) -> None:
        self._store.clear()
        self._updated_at.clear()

    def prune_stale(self, now: datetime | None = None) -> None:
        current = now or utcnow()
        symbols_to_remove: list[str] = []

        for symbol, features in self._store.items():
            stale_keys = [
                feature_name
                for feature_name, snapshot in features.items()
                if snapshot.is_expired(current)
            ]
            for key in stale_keys:
                features.pop(key, None)

            if not features:
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            self._store.pop(symbol, None)
            self._updated_at.pop(symbol, None)

    def build_context(
        self,
        symbol: str,
        timestamp: datetime | None = None,
    ) -> StrategyContext:
        if not symbol.strip():
            raise FeatureStoreError("symbol cannot be empty")

        context = StrategyContext(
            symbol=symbol,
            timestamp=timestamp or utcnow(),
        )

        for snapshot in self.get_symbol_features(symbol).values():
            context.put_feature(snapshot)

        return context

    @property
    def symbols(self) -> list[str]:
        return list(self._store.keys())


class StrategyState:
    """
    Кореневий контейнер strategy state.

    Об’єднує:
    - symbol states
    - regime state
    - portfolio state
    - feature store
    """

    def __init__(self) -> None:
        self.symbols: dict[str, SymbolState] = {}
        self.regimes = MarketRegimeState()
        self.portfolio = PortfolioState()
        self.feature_store = FeatureStore()
        self.updated_at: datetime | None = None

    def get_or_create_symbol_state(self, symbol: str) -> SymbolState:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        state = self.symbols.get(symbol)
        if state is None:
            state = SymbolState(symbol=symbol)
            self.symbols[symbol] = state
            self.updated_at = utcnow()
        return state

    def get_symbol_state(self, symbol: str) -> SymbolState | None:
        return self.symbols.get(symbol)

    def update_context(self, context: StrategyContext) -> SymbolState:
        context.validate()

        symbol_state = self.get_or_create_symbol_state(context.symbol)
        symbol_state.set_context(context)

        for snapshot in context.feature_map.values():
            self.feature_store.upsert(snapshot)

        self.updated_at = utcnow()
        return symbol_state

    def update_signal(self, signal: StrategySignal, active: bool = True) -> SymbolState:
        signal.validate()

        symbol_state = self.get_or_create_symbol_state(signal.symbol)
        symbol_state.set_last_signal(signal)
        if active:
            symbol_state.add_active_signal(signal)

        self.updated_at = utcnow()
        return symbol_state

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        self.regimes.set_regime(snapshot)
        self.updated_at = utcnow()

    def set_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self.portfolio.set_snapshot(snapshot)
        self.updated_at = utcnow()

    def prune(self) -> None:
        for symbol_state in self.symbols.values():
            symbol_state.clear_expired_cooldowns()
            symbol_state.clear_expired_side_cooldowns()
            symbol_state.remove_inactive_signals()

        self.feature_store.prune_stale()
        self.updated_at = utcnow()

    def clear_symbol(self, symbol: str) -> None:
        self.symbols.pop(symbol, None)
        self.feature_store.remove_symbol(symbol)
        self.updated_at = utcnow()

    def clear(self) -> None:
        self.symbols.clear()
        self.regimes.clear()
        self.portfolio = PortfolioState()
        self.feature_store.clear()
        self.updated_at = utcnow()