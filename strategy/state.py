# trading_system/strategy/state.py

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque

from .enums import SignalSide, SignalStatus, StrategyCategory
from .exceptions import FeatureStoreError, StrategyStateError, ValidationError
from .models import (
    CooldownState,
    FeatureSnapshot,
    PortfolioSnapshot,
    RegimeSnapshot,
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    ensure_aware_utc,
    utcnow,
)


@dataclass(slots=True)
class SignalState:
    """
    Runtime state для strategy signals.

    Відповідає за:
    - останній signal по symbol;
    - active signals;
    - signal history;
    - lookup by strategy/symbol/side;
    - cleanup inactive/expired signals.

    Не виконує risk/execution логіку.
    """

    max_history_size: int = 1000

    active_signals: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_symbol: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_strategy: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_symbol_side: dict[tuple[str, SignalSide], StrategySignal] = field(
        default_factory=dict
    )

    history: Deque[StrategySignal] = field(default_factory=deque)
    rejected_signals: Deque[StrategySignal] = field(default_factory=deque)
    expired_signals: Deque[StrategySignal] = field(default_factory=deque)

    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_history_size <= 0:
            raise ValidationError("SignalState.max_history_size must be > 0")

    def validate(self) -> None:
        if self.max_history_size <= 0:
            raise ValidationError("SignalState.max_history_size must be > 0")

        for signal in self.active_signals.values():
            signal.validate()

        for signal in self.last_signal_by_symbol.values():
            signal.validate()

        for signal in self.last_signal_by_strategy.values():
            signal.validate()

        for signal in self.last_signal_by_symbol_side.values():
            signal.validate()

    def touch(self) -> None:
        self.updated_at = utcnow()

    def remember(
        self,
        signal: StrategySignal,
        *,
        active: bool | None = None,
    ) -> None:
        """
        Save signal into state.

        active=None:
            активність визначається через signal.is_active.
        """
        signal.validate()

        is_active = signal.is_active if active is None else active
        key = self._signal_key(signal)

        if is_active:
            self.active_signals[key] = signal
        else:
            self.active_signals.pop(key, None)

        self.last_signal_by_symbol[signal.symbol] = signal
        self.last_signal_by_strategy[signal.strategy_name] = signal
        self.last_signal_by_symbol_side[(signal.symbol, signal.side)] = signal

        self._append_history(signal)

        if signal.status == SignalStatus.REJECTED:
            self._append_bounded(self.rejected_signals, signal)

        if signal.status == SignalStatus.EXPIRED:
            self._append_bounded(self.expired_signals, signal)

        self.touch()

    def remember_evaluation(self, evaluation: StrategyEvaluation) -> None:
        evaluation.validate()
        if evaluation.signal is not None:
            self.remember(evaluation.signal, active=evaluation.passed)

    def mark_rejected(
        self,
        signal: StrategySignal,
        *,
        reason: str | None = None,
    ) -> None:
        signal.validate()

        if reason:
            signal.add_reason(reason)

        signal.to_rejected()
        self.remember(signal, active=False)

    def mark_expired(
        self,
        signal: StrategySignal,
        *,
        reason: str | None = None,
    ) -> None:
        signal.validate()

        if reason:
            signal.add_reason(reason)

        signal.to_expired()
        self.remember(signal, active=False)

    def remove_active(self, signal: StrategySignal) -> None:
        signal.validate()
        self.active_signals.pop(self._signal_key(signal), None)
        self.touch()

    def remove_active_by_key(
        self,
        *,
        symbol: str,
        strategy_name: str,
        side: SignalSide,
    ) -> None:
        key = self._make_signal_key(
            symbol=symbol,
            strategy_name=strategy_name,
            side=side,
        )
        self.active_signals.pop(key, None)
        self.touch()

    def get_active(self) -> list[StrategySignal]:
        return sorted(
            self.active_signals.values(),
            key=lambda signal: (signal.timestamp, signal.strategy_name),
            reverse=True,
        )

    def get_active_for_symbol(self, symbol: str) -> list[StrategySignal]:
        return [
            signal
            for signal in self.get_active()
            if signal.symbol == symbol
        ]

    def get_active_for_strategy(self, strategy_name: str) -> list[StrategySignal]:
        return [
            signal
            for signal in self.get_active()
            if signal.strategy_name == strategy_name
        ]

    def get_last_for_symbol(self, symbol: str) -> StrategySignal | None:
        return self.last_signal_by_symbol.get(symbol)

    def get_last_for_strategy(self, strategy_name: str) -> StrategySignal | None:
        return self.last_signal_by_strategy.get(strategy_name)

    def get_last_for_symbol_side(
        self,
        symbol: str,
        side: SignalSide,
    ) -> StrategySignal | None:
        return self.last_signal_by_symbol_side.get((symbol, side))

    def history_list(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        limit: int | None = None,
    ) -> list[StrategySignal]:
        items = list(self.history)

        if symbol is not None:
            items = [signal for signal in items if signal.symbol == symbol]

        if strategy_name is not None:
            items = [
                signal
                for signal in items
                if signal.strategy_name == strategy_name
            ]

        items = sorted(
            items,
            key=lambda signal: signal.timestamp,
            reverse=True,
        )

        if limit is not None:
            return items[:limit]

        return items

    def prune_inactive(self) -> None:
        self.active_signals = {
            key: signal
            for key, signal in self.active_signals.items()
            if signal.is_active
        }
        self.touch()

    def prune_older_than(self, cutoff: datetime) -> None:
        cutoff = ensure_aware_utc(cutoff)

        self.history = deque(
            [
                signal
                for signal in self.history
                if signal.timestamp >= cutoff
            ],
            maxlen=self.max_history_size,
        )

        self.rejected_signals = deque(
            [
                signal
                for signal in self.rejected_signals
                if signal.timestamp >= cutoff
            ],
            maxlen=self.max_history_size,
        )

        self.expired_signals = deque(
            [
                signal
                for signal in self.expired_signals
                if signal.timestamp >= cutoff
            ],
            maxlen=self.max_history_size,
        )

        self.touch()

    def clear_symbol(self, symbol: str) -> None:
        self.active_signals = {
            key: signal
            for key, signal in self.active_signals.items()
            if signal.symbol != symbol
        }

        self.last_signal_by_symbol.pop(symbol, None)

        self.last_signal_by_symbol_side = {
            key: signal
            for key, signal in self.last_signal_by_symbol_side.items()
            if key[0] != symbol
        }

        self.history = deque(
            [
                signal
                for signal in self.history
                if signal.symbol != symbol
            ],
            maxlen=self.max_history_size,
        )

        self.touch()

    def clear(self) -> None:
        self.active_signals.clear()
        self.last_signal_by_symbol.clear()
        self.last_signal_by_strategy.clear()
        self.last_signal_by_symbol_side.clear()
        self.history.clear()
        self.rejected_signals.clear()
        self.expired_signals.clear()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "active": len(self.active_signals),
            "history": len(self.history),
            "rejected": len(self.rejected_signals),
            "expired": len(self.expired_signals),
            "symbols": sorted(self.last_signal_by_symbol.keys()),
            "strategies": sorted(self.last_signal_by_strategy.keys()),
            "updated_at": self.updated_at,
        }

    def _append_history(self, signal: StrategySignal) -> None:
        self._append_bounded(self.history, signal)

    def _append_bounded(
        self,
        target: Deque[StrategySignal],
        signal: StrategySignal,
    ) -> None:
        target.append(signal)

        while len(target) > self.max_history_size:
            target.popleft()

    def _signal_key(self, signal: StrategySignal) -> str:
        return self._make_signal_key(
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            side=signal.side,
        )

    @staticmethod
    def _make_signal_key(
        *,
        symbol: str,
        strategy_name: str,
        side: SignalSide,
    ) -> str:
        return f"{symbol}:{strategy_name}:{side.value}"


@dataclass(slots=True)
class StrategyContextStore:
    """
    In-memory context/feature store для strategy layer.

    Відповідає за:
    - збереження останнього StrategyContext по symbol;
    - збереження FeatureSnapshot по symbol/feature;
    - regime snapshots;
    - portfolio snapshot;
    - побудову StrategyContext з feature store.

    Це локальний strategy-layer state, не storage persistence.
    """

    contexts: dict[str, StrategyContext] = field(default_factory=dict)
    features: dict[str, dict[str, FeatureSnapshot]] = field(default_factory=dict)
    regimes: dict[str, RegimeSnapshot] = field(default_factory=dict)
    portfolio: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)

    updated_at_by_symbol: dict[str, datetime] = field(default_factory=dict)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for symbol, context in self.contexts.items():
            if symbol != context.symbol:
                raise ValidationError(
                    f"StrategyContextStore.contexts key '{symbol}' does not match context.symbol '{context.symbol}'"
                )
            context.validate()

        for symbol, feature_map in self.features.items():
            if not symbol.strip():
                raise ValidationError("StrategyContextStore.features contains empty symbol")

            for feature_name, snapshot in feature_map.items():
                if feature_name != snapshot.name:
                    raise ValidationError(
                        f"Feature key '{feature_name}' does not match snapshot.name '{snapshot.name}'"
                    )
                if snapshot.symbol != symbol:
                    raise ValidationError(
                        f"FeatureSnapshot.symbol '{snapshot.symbol}' does not match symbol key '{symbol}'"
                    )
                snapshot.validate()

        for symbol, regime in self.regimes.items():
            if symbol != regime.symbol:
                raise ValidationError(
                    f"Regime key '{symbol}' does not match regime.symbol '{regime.symbol}'"
                )
            regime.validate()

        self.portfolio.validate()

    def touch(self, symbol: str | None = None) -> None:
        now = utcnow()
        self.updated_at = now

        if symbol is not None:
            self.updated_at_by_symbol[symbol] = now

    def set_context(self, context: StrategyContext) -> None:
        context.validate()

        self.contexts[context.symbol] = context

        for snapshot in context.feature_map.values():
            self.upsert_feature(snapshot)

        if context.regime is not None:
            self.set_regime(context.regime)

        if context.portfolio is not None:
            self.set_portfolio(context.portfolio)

        self.touch(context.symbol)

    def get_context(self, symbol: str) -> StrategyContext | None:
        return self.contexts.get(symbol)

    def require_context(self, symbol: str) -> StrategyContext:
        context = self.get_context(symbol)
        if context is None:
            raise FeatureStoreError(f"context for symbol '{symbol}' is not available")
        return context

    def upsert_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()

        symbol_features = self.features.setdefault(snapshot.symbol, {})
        symbol_features[snapshot.name] = snapshot

        self.touch(snapshot.symbol)

    def bulk_upsert_features(self, snapshots: list[FeatureSnapshot]) -> None:
        for snapshot in snapshots:
            self.upsert_feature(snapshot)

    def get_feature(
        self,
        symbol: str,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        return self.features.get(symbol, {}).get(feature_name)

    def require_feature(
        self,
        symbol: str,
        feature_name: str,
    ) -> FeatureSnapshot:
        snapshot = self.get_feature(symbol, feature_name)
        if snapshot is None:
            raise FeatureStoreError(
                f"feature '{feature_name}' for symbol '{symbol}' is not available"
            )
        return snapshot

    def get_feature_value(
        self,
        symbol: str,
        feature_name: str,
        default: Any = None,
    ) -> Any:
        snapshot = self.get_feature(symbol, feature_name)
        if snapshot is None:
            return default
        return snapshot.value

    def get_normalized_feature_value(
        self,
        symbol: str,
        feature_name: str,
        default: float | None = None,
    ) -> float | None:
        snapshot = self.get_feature(symbol, feature_name)
        if snapshot is None:
            return default

        if snapshot.normalized_value is None:
            return default

        return snapshot.normalized_value

    def get_symbol_features(self, symbol: str) -> dict[str, FeatureSnapshot]:
        return dict(self.features.get(symbol, {}))

    def has_feature(self, symbol: str, feature_name: str) -> bool:
        return feature_name in self.features.get(symbol, {})

    def remove_feature(self, symbol: str, feature_name: str) -> None:
        symbol_features = self.features.get(symbol)
        if not symbol_features:
            return

        symbol_features.pop(feature_name, None)

        if not symbol_features:
            self.features.pop(symbol, None)

        self.touch(symbol)

    def remove_symbol(self, symbol: str) -> None:
        self.contexts.pop(symbol, None)
        self.features.pop(symbol, None)
        self.regimes.pop(symbol, None)
        self.updated_at_by_symbol.pop(symbol, None)
        self.touch()

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()
        self.regimes[snapshot.symbol] = snapshot
        self.touch(snapshot.symbol)

    def get_regime_snapshot(self, symbol: str) -> RegimeSnapshot | None:
        return self.regimes.get(symbol)

    def set_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()
        self.portfolio = snapshot
        self.touch()

    def build_context(
        self,
        symbol: str,
        *,
        timestamp: datetime | None = None,
        include_regime: bool = True,
        include_portfolio: bool = True,
    ) -> StrategyContext:
        if not symbol.strip():
            raise FeatureStoreError("symbol cannot be empty")

        context = StrategyContext(
            symbol=symbol,
            timestamp=ensure_aware_utc(timestamp or utcnow()),
            regime=self.regimes.get(symbol) if include_regime else None,
            portfolio=self.portfolio if include_portfolio else None,
        )

        for snapshot in self.get_symbol_features(symbol).values():
            context.put_feature(snapshot)

        return context

    def prune_expired_features(self, now: datetime | None = None) -> int:
        current = ensure_aware_utc(now or utcnow())
        removed = 0
        symbols_to_remove: list[str] = []

        for symbol, feature_map in self.features.items():
            expired_names = [
                feature_name
                for feature_name, snapshot in feature_map.items()
                if snapshot.is_expired(current)
            ]

            for feature_name in expired_names:
                feature_map.pop(feature_name, None)
                removed += 1

            if not feature_map:
                symbols_to_remove.append(symbol)

        for symbol in symbols_to_remove:
            self.features.pop(symbol, None)

        if removed:
            self.touch()

        return removed

    def clear(self) -> None:
        self.contexts.clear()
        self.features.clear()
        self.regimes.clear()
        self.portfolio = PortfolioSnapshot()
        self.updated_at_by_symbol.clear()
        self.touch()

    @property
    def symbols(self) -> list[str]:
        return sorted(
            set(self.contexts.keys())
            | set(self.features.keys())
            | set(self.regimes.keys())
        )

    def summary(self) -> dict[str, Any]:
        return {
            "contexts": len(self.contexts),
            "feature_symbols": len(self.features),
            "features_total": sum(len(features) for features in self.features.values()),
            "regimes": len(self.regimes),
            "symbols": self.symbols,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class StrategyCooldownState:
    """
    Cooldown state для strategy layer.

    Підтримує:
    - cooldown по strategy+symbol;
    - cooldown по symbol+side;
    - global symbol cooldown;
    - cleanup expired cooldowns.
    """

    strategy_cooldowns: dict[tuple[str, str], CooldownState] = field(
        default_factory=dict
    )
    side_cooldowns: dict[tuple[str, SignalSide], CooldownState] = field(
        default_factory=dict
    )
    symbol_cooldowns: dict[str, CooldownState] = field(default_factory=dict)

    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for cooldown in self.strategy_cooldowns.values():
            cooldown.validate()

        for cooldown in self.side_cooldowns.values():
            cooldown.validate()

        for cooldown in self.symbol_cooldowns.values():
            cooldown.validate()

    def touch(self) -> None:
        self.updated_at = utcnow()

    def add_strategy_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")
        if not strategy_name.strip():
            raise StrategyStateError("strategy_name cannot be empty")
        if seconds < 0:
            raise StrategyStateError("cooldown seconds must be >= 0")

        until = utcnow() + timedelta(seconds=seconds)

        self.strategy_cooldowns[(symbol, strategy_name)] = CooldownState(
            symbol=symbol,
            strategy_name=strategy_name,
            until=until,
            reason=reason,
        )
        self.touch()

    def add_side_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")
        if seconds < 0:
            raise StrategyStateError("side cooldown seconds must be >= 0")

        until = utcnow() + timedelta(seconds=seconds)

        self.side_cooldowns[(symbol, side)] = CooldownState(
            symbol=symbol,
            strategy_name=f"__side__:{side.value}",
            until=until,
            reason=reason,
        )
        self.touch()

    def add_symbol_cooldown(
        self,
        *,
        symbol: str,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")
        if seconds < 0:
            raise StrategyStateError("symbol cooldown seconds must be >= 0")

        until = utcnow() + timedelta(seconds=seconds)

        self.symbol_cooldowns[symbol] = CooldownState(
            symbol=symbol,
            strategy_name="__symbol__",
            until=until,
            reason=reason,
        )
        self.touch()

    def is_strategy_on_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str,
        now: datetime | None = None,
    ) -> bool:
        cooldown = self.strategy_cooldowns.get((symbol, strategy_name))
        if cooldown is None:
            return False
        return cooldown.is_active(now)

    def is_side_on_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
        now: datetime | None = None,
    ) -> bool:
        cooldown = self.side_cooldowns.get((symbol, side))
        if cooldown is None:
            return False
        return cooldown.is_active(now)

    def is_symbol_on_cooldown(
        self,
        *,
        symbol: str,
        now: datetime | None = None,
    ) -> bool:
        cooldown = self.symbol_cooldowns.get(symbol)
        if cooldown is None:
            return False
        return cooldown.is_active(now)

    def is_blocked(
        self,
        *,
        symbol: str,
        strategy_name: str | None = None,
        side: SignalSide | None = None,
        now: datetime | None = None,
    ) -> bool:
        if self.is_symbol_on_cooldown(symbol=symbol, now=now):
            return True

        if strategy_name is not None and self.is_strategy_on_cooldown(
            symbol=symbol,
            strategy_name=strategy_name,
            now=now,
        ):
            return True

        if side is not None and self.is_side_on_cooldown(
            symbol=symbol,
            side=side,
            now=now,
        ):
            return True

        return False

    def clear_strategy_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str,
    ) -> None:
        self.strategy_cooldowns.pop((symbol, strategy_name), None)
        self.touch()

    def clear_side_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
    ) -> None:
        self.side_cooldowns.pop((symbol, side), None)
        self.touch()

    def clear_symbol_cooldown(self, symbol: str) -> None:
        self.symbol_cooldowns.pop(symbol, None)
        self.touch()

    def clear_symbol(self, symbol: str) -> None:
        self.symbol_cooldowns.pop(symbol, None)

        self.strategy_cooldowns = {
            key: cooldown
            for key, cooldown in self.strategy_cooldowns.items()
            if key[0] != symbol
        }

        self.side_cooldowns = {
            key: cooldown
            for key, cooldown in self.side_cooldowns.items()
            if key[0] != symbol
        }

        self.touch()

    def prune_expired(self, now: datetime | None = None) -> int:
        current = ensure_aware_utc(now or utcnow())

        before = self.count()

        self.strategy_cooldowns = {
            key: cooldown
            for key, cooldown in self.strategy_cooldowns.items()
            if cooldown.is_active(current)
        }

        self.side_cooldowns = {
            key: cooldown
            for key, cooldown in self.side_cooldowns.items()
            if cooldown.is_active(current)
        }

        self.symbol_cooldowns = {
            symbol: cooldown
            for symbol, cooldown in self.symbol_cooldowns.items()
            if cooldown.is_active(current)
        }

        removed = before - self.count()

        if removed:
            self.touch()

        return removed

    def count(self) -> int:
        return (
            len(self.strategy_cooldowns)
            + len(self.side_cooldowns)
            + len(self.symbol_cooldowns)
        )

    def clear(self) -> None:
        self.strategy_cooldowns.clear()
        self.side_cooldowns.clear()
        self.symbol_cooldowns.clear()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_cooldowns": len(self.strategy_cooldowns),
            "side_cooldowns": len(self.side_cooldowns),
            "symbol_cooldowns": len(self.symbol_cooldowns),
            "total": self.count(),
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class StrategyMetricsState:
    """
    In-memory metrics для strategy layer.

    Це lightweight runtime metrics, не Prometheus exporter.
    Prometheus/Grafana можна будувати окремо на основі цих counters.
    """

    evaluations_total: int = 0
    evaluations_passed: int = 0
    evaluations_failed: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    signals_expired: int = 0
    applicability_skipped: int = 0
    errors_total: int = 0

    evaluations_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    signals_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    signals_by_symbol: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    signals_by_category: dict[StrategyCategory, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errors_by_strategy: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    last_evaluation_at: dict[str, datetime] = field(default_factory=dict)
    last_signal_at: dict[str, datetime] = field(default_factory=dict)
    last_error_at: dict[str, datetime] = field(default_factory=dict)

    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = ensure_aware_utc(self.started_at)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def record_evaluation(self, evaluation: StrategyEvaluation) -> None:
        evaluation.validate()

        self.evaluations_total += 1
        self.evaluations_by_strategy[evaluation.strategy_name] += 1
        self.last_evaluation_at[evaluation.strategy_name] = evaluation.timestamp

        if evaluation.passed:
            self.evaluations_passed += 1
        else:
            self.evaluations_failed += 1

        if evaluation.signal is not None:
            self.record_signal(evaluation.signal)

        self.touch()

    def record_signal(self, signal: StrategySignal) -> None:
        signal.validate()

        self.signals_generated += 1
        self.signals_by_strategy[signal.strategy_name] += 1
        self.signals_by_symbol[signal.symbol] += 1
        self.signals_by_category[signal.category] += 1
        self.last_signal_at[signal.strategy_name] = signal.timestamp

        if signal.status == SignalStatus.REJECTED:
            self.signals_rejected += 1

        if signal.status == SignalStatus.EXPIRED:
            self.signals_expired += 1

        self.touch()

    def record_rejected_signal(self, signal: StrategySignal) -> None:
        signal.validate()
        self.signals_rejected += 1
        self.signals_by_strategy[signal.strategy_name] += 1
        self.signals_by_symbol[signal.symbol] += 1
        self.signals_by_category[signal.category] += 1
        self.touch()

    def record_expired_signal(self, signal: StrategySignal) -> None:
        signal.validate()
        self.signals_expired += 1
        self.touch()

    def record_applicability_skip(
        self,
        *,
        strategy_name: str | None = None,
    ) -> None:
        self.applicability_skipped += 1

        if strategy_name:
            self.evaluations_by_strategy[strategy_name] += 0

        self.touch()

    def record_error(
        self,
        *,
        strategy_name: str | None = None,
    ) -> None:
        self.errors_total += 1
        now = utcnow()

        if strategy_name:
            self.errors_by_strategy[strategy_name] += 1
            self.last_error_at[strategy_name] = now

        self.touch()

    @property
    def pass_rate(self) -> float:
        if self.evaluations_total == 0:
            return 0.0
        return self.evaluations_passed / self.evaluations_total

    @property
    def error_rate(self) -> float:
        if self.evaluations_total == 0:
            return 0.0
        return self.errors_total / self.evaluations_total

    def reset(self) -> None:
        self.evaluations_total = 0
        self.evaluations_passed = 0
        self.evaluations_failed = 0
        self.signals_generated = 0
        self.signals_rejected = 0
        self.signals_expired = 0
        self.applicability_skipped = 0
        self.errors_total = 0

        self.evaluations_by_strategy.clear()
        self.signals_by_strategy.clear()
        self.signals_by_symbol.clear()
        self.signals_by_category.clear()
        self.errors_by_strategy.clear()

        self.last_evaluation_at.clear()
        self.last_signal_at.clear()
        self.last_error_at.clear()

        self.started_at = utcnow()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "evaluations_total": self.evaluations_total,
            "evaluations_passed": self.evaluations_passed,
            "evaluations_failed": self.evaluations_failed,
            "pass_rate": self.pass_rate,
            "signals_generated": self.signals_generated,
            "signals_rejected": self.signals_rejected,
            "signals_expired": self.signals_expired,
            "applicability_skipped": self.applicability_skipped,
            "errors_total": self.errors_total,
            "error_rate": self.error_rate,
            "evaluations_by_strategy": dict(self.evaluations_by_strategy),
            "signals_by_strategy": dict(self.signals_by_strategy),
            "signals_by_symbol": dict(self.signals_by_symbol),
            "signals_by_category": {
                str(category): count
                for category, count in self.signals_by_category.items()
            },
            "errors_by_strategy": dict(self.errors_by_strategy),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class StrategyRuntimeState:
    """
    Root runtime state container для strategy package.

    Об'єднує:
    - SignalState;
    - StrategyContextStore;
    - StrategyCooldownState;
    - StrategyMetricsState;
    - runtime blocked symbols;
    - lifecycle timestamps.

    Не має EventBus subscription і не виконує торгову логіку.
    Його використовують StrategyEngine, SignalProcessor, PortfolioCoordinator
    та domain strategies як shared in-memory runtime state.
    """

    signals: SignalState = field(default_factory=SignalState)
    contexts: StrategyContextStore = field(default_factory=StrategyContextStore)
    cooldowns: StrategyCooldownState = field(default_factory=StrategyCooldownState)
    metrics: StrategyMetricsState = field(default_factory=StrategyMetricsState)

    blocked_symbols: set[str] = field(default_factory=set)
    active_symbols: set[str] = field(default_factory=set)

    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = ensure_aware_utc(self.started_at)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def validate(self) -> None:
        self.signals.validate()
        self.contexts.validate()
        self.cooldowns.validate()

        if any(not symbol.strip() for symbol in self.blocked_symbols):
            raise ValidationError("StrategyRuntimeState.blocked_symbols contains empty symbol")

        if any(not symbol.strip() for symbol in self.active_symbols):
            raise ValidationError("StrategyRuntimeState.active_symbols contains empty symbol")

    def update_context(self, context: StrategyContext) -> None:
        context.validate()

        self.contexts.set_context(context)
        self.active_symbols.add(context.symbol)
        self.touch()

    def build_context(
        self,
        symbol: str,
        *,
        timestamp: datetime | None = None,
        include_regime: bool = True,
        include_portfolio: bool = True,
    ) -> StrategyContext:
        return self.contexts.build_context(
            symbol,
            timestamp=timestamp,
            include_regime=include_regime,
            include_portfolio=include_portfolio,
        )

    def update_signal(
        self,
        signal: StrategySignal,
        *,
        active: bool | None = None,
    ) -> None:
        signal.validate()

        self.signals.remember(signal, active=active)
        self.active_symbols.add(signal.symbol)
        self.touch()

    def update_evaluation(self, evaluation: StrategyEvaluation) -> None:
        evaluation.validate()

        self.metrics.record_evaluation(evaluation)

        if evaluation.signal is not None:
            self.signals.remember(
                evaluation.signal,
                active=evaluation.passed,
            )
            self.active_symbols.add(evaluation.symbol)

        self.touch()

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()

        self.contexts.set_regime(snapshot)
        self.active_symbols.add(snapshot.symbol)
        self.touch()

    def set_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()

        self.contexts.set_portfolio(snapshot)
        self.touch()

    def block_symbol(self, symbol: str) -> None:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        self.blocked_symbols.add(symbol)
        self.touch()

    def unblock_symbol(self, symbol: str) -> None:
        self.blocked_symbols.discard(symbol)
        self.touch()

    def is_symbol_blocked(self, symbol: str) -> bool:
        return (
            symbol in self.blocked_symbols
            or symbol in self.contexts.portfolio.blocked_symbols
        )

    def add_strategy_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        self.cooldowns.add_strategy_cooldown(
            symbol=symbol,
            strategy_name=strategy_name,
            seconds=seconds,
            reason=reason,
        )
        self.touch()

    def add_side_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        self.cooldowns.add_side_cooldown(
            symbol=symbol,
            side=side,
            seconds=seconds,
            reason=reason,
        )
        self.touch()

    def add_symbol_cooldown(
        self,
        *,
        symbol: str,
        seconds: int,
        reason: str | None = None,
    ) -> None:
        self.cooldowns.add_symbol_cooldown(
            symbol=symbol,
            seconds=seconds,
            reason=reason,
        )
        self.touch()

    def is_blocked_by_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str | None = None,
        side: SignalSide | None = None,
        now: datetime | None = None,
    ) -> bool:
        return self.cooldowns.is_blocked(
            symbol=symbol,
            strategy_name=strategy_name,
            side=side,
            now=now,
        )

    def prune(
        self,
        *,
        max_signal_age_seconds: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, int]:
        current = ensure_aware_utc(now or utcnow())

        removed_cooldowns = self.cooldowns.prune_expired(current)
        removed_features = self.contexts.prune_expired_features(current)

        self.signals.prune_inactive()

        removed_old_signals = 0
        if max_signal_age_seconds is not None:
            if max_signal_age_seconds <= 0:
                raise StrategyStateError("max_signal_age_seconds must be > 0")

            before = len(self.signals.history)
            cutoff = current - timedelta(seconds=max_signal_age_seconds)
            self.signals.prune_older_than(cutoff)
            removed_old_signals = before - len(self.signals.history)

        self.touch()

        return {
            "removed_cooldowns": removed_cooldowns,
            "removed_features": removed_features,
            "removed_old_signals": removed_old_signals,
        }

    def clear_symbol(self, symbol: str) -> None:
        self.contexts.remove_symbol(symbol)
        self.signals.clear_symbol(symbol)
        self.cooldowns.clear_symbol(symbol)
        self.blocked_symbols.discard(symbol)
        self.active_symbols.discard(symbol)
        self.touch()

    def clear(self) -> None:
        self.signals.clear()
        self.contexts.clear()
        self.cooldowns.clear()
        self.metrics.reset()
        self.blocked_symbols.clear()
        self.active_symbols.clear()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "signals": self.signals.summary(),
            "contexts": self.contexts.summary(),
            "cooldowns": self.cooldowns.summary(),
            "metrics": self.metrics.summary(),
            "blocked_symbols": sorted(self.blocked_symbols),
            "active_symbols": sorted(self.active_symbols),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }