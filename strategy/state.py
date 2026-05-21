# trading_system/strategy/state.py

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque

from strategy.enums import SignalSide, SignalStatus, StrategyCategory, Timeframe
from strategy.exceptions import StrategyStateError, ValidationError
from strategy.models import (
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
    - lookup by signal_id;
    - active signals;
    - last signal by symbol/strategy/symbol+side;
    - signal history;
    - rejected/expired/terminal buckets;
    - status updates from StrategyEventHandler.

    Не містить EventBus logic, risk logic або execution logic.
    """

    max_history_size: int = 1000

    active_signals: dict[str, StrategySignal] = field(default_factory=dict)

    signal_by_id: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_symbol: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_strategy: dict[str, StrategySignal] = field(default_factory=dict)
    last_signal_by_symbol_side: dict[tuple[str, SignalSide], StrategySignal] = field(
        default_factory=dict
    )

    history: Deque[StrategySignal] = field(default_factory=deque)
    rejected_signals: Deque[StrategySignal] = field(default_factory=deque)
    expired_signals: Deque[StrategySignal] = field(default_factory=deque)
    confirmed_signals: Deque[StrategySignal] = field(default_factory=deque)
    executed_signals: Deque[StrategySignal] = field(default_factory=deque)
    failed_signals: Deque[StrategySignal] = field(default_factory=deque)
    cancelled_signals: Deque[StrategySignal] = field(default_factory=deque)

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

        for signal in self.signal_by_id.values():
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

        signal_id = self._signal_id(signal)
        if signal_id:
            self.signal_by_id[signal_id] = signal

        key = self._signal_key(signal)
        is_active = signal.is_active if active is None else active

        if is_active:
            self.active_signals[key] = signal
        else:
            self.active_signals.pop(key, None)

        self.last_signal_by_symbol[signal.symbol] = signal
        self.last_signal_by_strategy[signal.strategy_name] = signal
        self.last_signal_by_symbol_side[(signal.symbol, signal.side)] = signal

        self._append_history(signal)
        self._append_status_bucket(signal)
        self.touch()

    def remember_evaluation(self, evaluation: StrategyEvaluation) -> None:
        evaluation.validate()
        if evaluation.signal is not None:
            self.remember(evaluation.signal, active=evaluation.passed)

    def get_by_signal_id(self, signal_id: str | None) -> StrategySignal | None:
        if not isinstance(signal_id, str) or not signal_id.strip():
            return None
        return self.signal_by_id.get(signal_id.strip())

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

    def find_signal(
        self,
        *,
        signal_id: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        side: SignalSide | None = None,
    ) -> StrategySignal | None:
        """
        Best-effort lookup used by StrategyEventHandler.

        Priority:
        1. signal_id;
        2. exact symbol + strategy_name + side;
        3. exact symbol + strategy_name;
        4. symbol + side;
        5. symbol;
        6. strategy_name.
        """
        signal = self.get_by_signal_id(signal_id)
        if signal is not None:
            return signal

        if symbol and strategy_name and side is not None:
            for candidate in self.signal_by_id.values():
                if (
                    candidate.symbol == symbol
                    and candidate.strategy_name == strategy_name
                    and candidate.side is side
                ):
                    return candidate

        if symbol and strategy_name:
            for candidate in self.signal_by_id.values():
                if candidate.symbol == symbol and candidate.strategy_name == strategy_name:
                    return candidate

        if symbol and side is not None:
            signal = self.get_last_for_symbol_side(symbol, side)
            if signal is not None:
                return signal

        if symbol:
            signal = self.get_last_for_symbol(symbol)
            if signal is not None:
                return signal

        if strategy_name:
            return self.get_last_for_strategy(strategy_name)

        return None

    def mark_status_by_signal_id(
        self,
        signal_id: str,
        *,
        status: SignalStatus,
        reason: str | None = None,
        active: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategySignal | None:
        signal = self.get_by_signal_id(signal_id)
        if signal is None:
            return None

        return self.mark_status(
            signal,
            status=status,
            reason=reason,
            active=active,
            metadata=metadata,
        )

    def mark_status(
        self,
        signal: StrategySignal,
        *,
        status: SignalStatus,
        reason: str | None = None,
        active: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategySignal:
        """
        Mutate signal status and re-index it.

        This method is intentionally state-only. StrategyEventHandler decides
        when to call it after signal.confirmed / risk.position_blocked /
        execution.* events.
        """
        signal.validate()

        signal.status = status

        if reason:
            self._add_reason(signal, reason)

        if metadata:
            signal.metadata.update(metadata)

        if active is None:
            active = signal.is_active

        self.remember(signal, active=active)
        return signal

    def mark_status_from_payload(
        self,
        *,
        payload: dict[str, Any],
        status: SignalStatus,
        default_reason: str,
    ) -> StrategySignal | None:
        """
        Convenience method for StrategyEventHandler.

        Expected payload can contain:
        - signal_id;
        - symbol;
        - strategy_name;
        - side;
        - reason;
        - metadata.
        """
        signal_id = payload.get("signal_id")
        symbol = payload.get("symbol")
        strategy_name = payload.get("strategy_name")
        raw_side = payload.get("side")

        side = self._parse_side(raw_side)

        signal = self.find_signal(
            signal_id=signal_id if isinstance(signal_id, str) else None,
            symbol=symbol if isinstance(symbol, str) else None,
            strategy_name=strategy_name if isinstance(strategy_name, str) else None,
            side=side,
        )

        if signal is None:
            return None

        reason = payload.get("reason")
        reason_text = reason if isinstance(reason, str) and reason.strip() else default_reason

        event_metadata = payload.get("metadata")
        metadata = event_metadata if isinstance(event_metadata, dict) else {}

        return self.mark_status(
            signal,
            status=status,
            reason=reason_text,
            active=status in {SignalStatus.NEW, SignalStatus.PENDING, SignalStatus.CONFIRMED},
            metadata={
                "last_status_event_payload": dict(payload),
                **metadata,
            },
        )

    def mark_rejected(
        self,
        signal: StrategySignal,
        *,
        reason: str | None = None,
    ) -> None:
        if reason:
            self._add_reason(signal, reason)

        if hasattr(signal, "to_rejected"):
            signal.to_rejected()
        else:
            signal.status = SignalStatus.REJECTED

        self.remember(signal, active=False)

    def mark_expired(
        self,
        signal: StrategySignal,
        *,
        reason: str | None = None,
    ) -> None:
        if reason:
            self._add_reason(signal, reason)

        if hasattr(signal, "to_expired"):
            signal.to_expired()
        else:
            signal.status = SignalStatus.EXPIRED

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

    def history_list(
        self,
        *,
        symbol: str | None = None,
        strategy_name: str | None = None,
        signal_id: str | None = None,
        limit: int | None = None,
    ) -> list[StrategySignal]:
        items = list(self.history)

        if signal_id is not None:
            items = [
                signal
                for signal in items
                if self._signal_id(signal) == signal_id
            ]

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

    def prune_older_than(self, cutoff: datetime) -> dict[str, int]:
        cutoff = ensure_aware_utc(cutoff)

        before_history = len(self.history)
        before_rejected = len(self.rejected_signals)
        before_expired = len(self.expired_signals)
        before_confirmed = len(self.confirmed_signals)
        before_executed = len(self.executed_signals)
        before_failed = len(self.failed_signals)
        before_cancelled = len(self.cancelled_signals)

        self.history = self._filtered_deque(self.history, cutoff)
        self.rejected_signals = self._filtered_deque(self.rejected_signals, cutoff)
        self.expired_signals = self._filtered_deque(self.expired_signals, cutoff)
        self.confirmed_signals = self._filtered_deque(self.confirmed_signals, cutoff)
        self.executed_signals = self._filtered_deque(self.executed_signals, cutoff)
        self.failed_signals = self._filtered_deque(self.failed_signals, cutoff)
        self.cancelled_signals = self._filtered_deque(self.cancelled_signals, cutoff)

        self._rebuild_indexes_from_history()
        self.prune_inactive()
        self.touch()

        return {
            "history": before_history - len(self.history),
            "rejected": before_rejected - len(self.rejected_signals),
            "expired": before_expired - len(self.expired_signals),
            "confirmed": before_confirmed - len(self.confirmed_signals),
            "executed": before_executed - len(self.executed_signals),
            "failed": before_failed - len(self.failed_signals),
            "cancelled": before_cancelled - len(self.cancelled_signals),
        }

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

        self.signal_by_id = {
            signal_id: signal
            for signal_id, signal in self.signal_by_id.items()
            if signal.symbol != symbol
        }

        self.history = self._remove_symbol_from_deque(self.history, symbol)
        self.rejected_signals = self._remove_symbol_from_deque(self.rejected_signals, symbol)
        self.expired_signals = self._remove_symbol_from_deque(self.expired_signals, symbol)
        self.confirmed_signals = self._remove_symbol_from_deque(self.confirmed_signals, symbol)
        self.executed_signals = self._remove_symbol_from_deque(self.executed_signals, symbol)
        self.failed_signals = self._remove_symbol_from_deque(self.failed_signals, symbol)
        self.cancelled_signals = self._remove_symbol_from_deque(self.cancelled_signals, symbol)

        self.touch()

    def clear(self) -> None:
        self.active_signals.clear()
        self.signal_by_id.clear()
        self.last_signal_by_symbol.clear()
        self.last_signal_by_strategy.clear()
        self.last_signal_by_symbol_side.clear()
        self.history.clear()
        self.rejected_signals.clear()
        self.expired_signals.clear()
        self.confirmed_signals.clear()
        self.executed_signals.clear()
        self.failed_signals.clear()
        self.cancelled_signals.clear()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "active": len(self.active_signals),
            "signal_by_id": len(self.signal_by_id),
            "history": len(self.history),
            "rejected": len(self.rejected_signals),
            "expired": len(self.expired_signals),
            "confirmed": len(self.confirmed_signals),
            "executed": len(self.executed_signals),
            "failed": len(self.failed_signals),
            "cancelled": len(self.cancelled_signals),
            "symbols": sorted(self.last_signal_by_symbol.keys()),
            "strategies": sorted(self.last_signal_by_strategy.keys()),
            "updated_at": self.updated_at,
        }

    def _append_history(self, signal: StrategySignal) -> None:
        self._append_bounded(self.history, signal)

    def _append_status_bucket(self, signal: StrategySignal) -> None:
        if signal.status == SignalStatus.REJECTED:
            self._append_bounded(self.rejected_signals, signal)
        elif signal.status == SignalStatus.EXPIRED:
            self._append_bounded(self.expired_signals, signal)
        elif signal.status == SignalStatus.CONFIRMED:
            self._append_bounded(self.confirmed_signals, signal)
        elif signal.status == SignalStatus.EXECUTED:
            self._append_bounded(self.executed_signals, signal)
        elif signal.status == SignalStatus.FAILED:
            self._append_bounded(self.failed_signals, signal)
        elif signal.status == SignalStatus.CANCELLED:
            self._append_bounded(self.cancelled_signals, signal)

    def _append_bounded(
        self,
        target: Deque[StrategySignal],
        signal: StrategySignal,
    ) -> None:
        target.append(signal)

        while len(target) > self.max_history_size:
            target.popleft()

    def _filtered_deque(
        self,
        source: Deque[StrategySignal],
        cutoff: datetime,
    ) -> Deque[StrategySignal]:
        return deque(
            [
                signal
                for signal in source
                if signal.timestamp >= cutoff
            ],
            maxlen=self.max_history_size,
        )

    def _remove_symbol_from_deque(
        self,
        source: Deque[StrategySignal],
        symbol: str,
    ) -> Deque[StrategySignal]:
        return deque(
            [
                signal
                for signal in source
                if signal.symbol != symbol
            ],
            maxlen=self.max_history_size,
        )

    def _rebuild_indexes_from_history(self) -> None:
        self.active_signals.clear()
        self.signal_by_id.clear()
        self.last_signal_by_symbol.clear()
        self.last_signal_by_strategy.clear()
        self.last_signal_by_symbol_side.clear()

        for signal in self.history:
            signal_id = self._signal_id(signal)
            if signal_id:
                self.signal_by_id[signal_id] = signal

            if signal.is_active:
                self.active_signals[self._signal_key(signal)] = signal

            self.last_signal_by_symbol[signal.symbol] = signal
            self.last_signal_by_strategy[signal.strategy_name] = signal
            self.last_signal_by_symbol_side[(signal.symbol, signal.side)] = signal

    def _signal_key(self, signal: StrategySignal) -> str:
        return self._make_signal_key(
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            side=signal.side,
        )

    @staticmethod
    def _signal_id(signal: StrategySignal) -> str | None:
        value = getattr(signal, "signal_id", None)
        if isinstance(value, str) and value.strip():
            return value.strip()

        metadata = getattr(signal, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("signal_id")
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    @staticmethod
    def _make_signal_key(
        *,
        symbol: str,
        strategy_name: str,
        side: SignalSide,
    ) -> str:
        return f"{symbol}:{strategy_name}:{side.value}"

    @staticmethod
    def _parse_side(value: Any) -> SignalSide | None:
        if isinstance(value, SignalSide):
            return value

        if isinstance(value, str):
            try:
                return SignalSide(value)
            except ValueError:
                return None

        return None

    @staticmethod
    def _add_reason(signal: StrategySignal, reason: str) -> None:
        add_reason = getattr(signal, "add_reason", None)
        if callable(add_reason):
            add_reason(reason)
            return

        if reason not in signal.reasons:
            signal.reasons.append(reason)


@dataclass(slots=True)
class StrategyContextStore:
    """
    In-memory context/feature store для strategy layer.

    Відповідає за:
    - останній StrategyContext по symbol;
    - FeatureSnapshot по symbol/feature;
    - regime snapshots;
    - portfolio snapshot;
    - побудову StrategyContext з feature store.

    Це локальний runtime store, не persistence/storage.
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
            for feature in feature_map.values():
                feature.validate()

        for regime in self.regimes.values():
            regime.validate()

        self.portfolio.validate()

    def touch(self, symbol: str | None = None) -> None:
        now = utcnow()
        self.updated_at = now
        if symbol is not None:
            self.updated_at_by_symbol[symbol] = now

    def get_context(self, symbol: str) -> StrategyContext | None:
        return self.contexts.get(symbol)

    def set_context(self, context: StrategyContext) -> None:
        context.validate()
        self.contexts[context.symbol] = context

        self.features.setdefault(context.symbol, {})
        self.features[context.symbol].update(context.feature_map)

        if context.regime is not None:
            self.regimes[context.symbol] = context.regime

        if context.portfolio is not None:
            self.portfolio = context.portfolio

        self.touch(context.symbol)

    def put_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()

        self.features.setdefault(snapshot.symbol, {})[snapshot.name] = snapshot

        context = self.contexts.get(snapshot.symbol)
        if context is not None:
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds
            context.timestamp = snapshot.timestamp

        self.touch(snapshot.symbol)

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()
        self.regimes[snapshot.symbol] = snapshot

        context = self.contexts.get(snapshot.symbol)
        if context is not None:
            context.regime = snapshot
            context.timestamp = snapshot.timestamp

        self.touch(snapshot.symbol)

    def set_portfolio(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()
        self.portfolio = snapshot

        for context in self.contexts.values():
            context.portfolio = snapshot

        self.touch()

    def build_context(
        self,
        symbol: str,
        *,
        timestamp: datetime | None = None,
        timeframe: Timeframe = Timeframe.M1,
        include_regime: bool = True,
        include_portfolio: bool = True,
    ) -> StrategyContext:
        if not symbol.strip():
            raise StrategyStateError("symbol cannot be empty")

        ts = ensure_aware_utc(timestamp or utcnow())

        context = StrategyContext(
            symbol=symbol.strip(),
            timestamp=ts,
            timeframe=timeframe,
            regime=self.regimes.get(symbol) if include_regime else None,
            portfolio=self.portfolio if include_portfolio else None,
        )

        for snapshot in self.features.get(symbol, {}).values():
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        context.validate()
        self.set_context(context)
        return context

    def get_feature(
        self,
        symbol: str,
        name: str,
    ) -> FeatureSnapshot | None:
        return self.features.get(symbol, {}).get(name)

    def remove_symbol(self, symbol: str) -> None:
        self.contexts.pop(symbol, None)
        self.features.pop(symbol, None)
        self.regimes.pop(symbol, None)
        self.updated_at_by_symbol.pop(symbol, None)
        self.touch()

    def prune_stale_features(self) -> int:
        removed = 0

        for symbol, feature_map in list(self.features.items()):
            for name, snapshot in list(feature_map.items()):
                if snapshot.is_expired():
                    feature_map.pop(name, None)
                    removed += 1

            context = self.contexts.get(symbol)
            if context is not None:
                context.feature_map = {
                    name: snapshot
                    for name, snapshot in context.feature_map.items()
                    if not snapshot.is_expired()
                }

        if removed:
            self.touch()

        return removed

    def clear(self) -> None:
        self.contexts.clear()
        self.features.clear()
        self.regimes.clear()
        self.updated_at_by_symbol.clear()
        self.portfolio = PortfolioSnapshot()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "contexts": len(self.contexts),
            "features_symbols": len(self.features),
            "features_total": sum(len(items) for items in self.features.values()),
            "regimes": len(self.regimes),
            "portfolio_updated_at": self.updated_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class StrategyCooldownState:
    """
    Runtime cooldowns for strategy layer.

    Використовується до risk layer тільки як pre-risk throttling.
    """

    strategy_cooldowns: dict[tuple[str, str], CooldownState] = field(default_factory=dict)
    side_cooldowns: dict[tuple[str, SignalSide], CooldownState] = field(default_factory=dict)

    updated_at: datetime | None = None

    def validate(self) -> None:
        for cooldown in self.strategy_cooldowns.values():
            cooldown.validate()

        for cooldown in self.side_cooldowns.values():
            cooldown.validate()

    def touch(self) -> None:
        self.updated_at = utcnow()

    def add_strategy_cooldown(
        self,
        *,
        symbol: str,
        strategy_name: str,
        seconds: int | float,
        reason: str | None = None,
    ) -> None:
        if seconds <= 0:
            return

        until = utcnow() + timedelta(seconds=float(seconds))
        cooldown = CooldownState(
            symbol=symbol,
            strategy_name=strategy_name,
            until=until,
            reason=reason,
        )
        cooldown.validate()

        self.strategy_cooldowns[(symbol, strategy_name)] = cooldown
        self.touch()

    def add_side_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
        seconds: int | float,
        reason: str | None = None,
    ) -> None:
        if seconds <= 0:
            return

        until = utcnow() + timedelta(seconds=float(seconds))
        cooldown = CooldownState(
            symbol=symbol,
            strategy_name=f"side:{side.value}",
            until=until,
            reason=reason,
        )
        cooldown.validate()

        self.side_cooldowns[(symbol, side)] = cooldown
        self.touch()

    def is_strategy_blocked(
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

    def is_side_blocked(
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

    def prune_expired(self) -> int:
        now = utcnow()
        removed = 0

        for key, cooldown in list(self.strategy_cooldowns.items()):
            if not cooldown.is_active(now):
                self.strategy_cooldowns.pop(key, None)
                removed += 1

        for key, cooldown in list(self.side_cooldowns.items()):
            if not cooldown.is_active(now):
                self.side_cooldowns.pop(key, None)
                removed += 1

        if removed:
            self.touch()

        return removed

    def clear(self) -> None:
        self.strategy_cooldowns.clear()
        self.side_cooldowns.clear()
        self.touch()

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_cooldowns": len(self.strategy_cooldowns),
            "side_cooldowns": len(self.side_cooldowns),
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class StrategyMetricsState:
    """
    Runtime metrics for strategy layer.

    Не є storage/persistence. Для dashboard/storage можна робити snapshots.
    """

    evaluations_total: int = 0
    evaluations_passed: int = 0
    evaluations_failed: int = 0

    signals_generated: int = 0
    signals_rejected: int = 0
    signals_expired: int = 0
    signals_confirmed: int = 0
    signals_executed: int = 0
    signals_failed: int = 0
    signals_cancelled: int = 0

    applicability_skipped: int = 0
    errors_total: int = 0

    evaluations_by_strategy: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signals_by_strategy: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signals_by_symbol: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signals_by_category: dict[StrategyCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_strategy: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    last_evaluation_at: dict[str, datetime] = field(default_factory=dict)
    last_signal_at: dict[str, datetime] = field(default_factory=dict)
    last_error_at: dict[str, datetime] = field(default_factory=dict)

    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime | None = None

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

        self.record_status(signal.status)
        self.touch()

    def record_status(self, status: SignalStatus) -> None:
        if status == SignalStatus.REJECTED:
            self.signals_rejected += 1
        elif status == SignalStatus.EXPIRED:
            self.signals_expired += 1
        elif status == SignalStatus.CONFIRMED:
            self.signals_confirmed += 1
        elif status == SignalStatus.EXECUTED:
            self.signals_executed += 1
        elif status == SignalStatus.FAILED:
            self.signals_failed += 1
        elif status == SignalStatus.CANCELLED:
            self.signals_cancelled += 1

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
        self.signals_confirmed = 0
        self.signals_executed = 0
        self.signals_failed = 0
        self.signals_cancelled = 0

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
            "signals_confirmed": self.signals_confirmed,
            "signals_executed": self.signals_executed,
            "signals_failed": self.signals_failed,
            "signals_cancelled": self.signals_cancelled,
            "applicability_skipped": self.applicability_skipped,
            "errors_total": self.errors_total,
            "error_rate": self.error_rate,
            "evaluations_by_strategy": dict(self.evaluations_by_strategy),
            "signals_by_strategy": dict(self.signals_by_strategy),
            "signals_by_symbol": dict(self.signals_by_symbol),
            "signals_by_category": {
                category.value: count
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
    та StrategyEventHandler як shared in-memory runtime state.
    """

    signals: SignalState = field(default_factory=SignalState)
    contexts: StrategyContextStore = field(default_factory=StrategyContextStore)
    cooldowns: StrategyCooldownState = field(default_factory=StrategyCooldownState)
    metrics: StrategyMetricsState = field(default_factory=StrategyMetricsState)

    blocked_symbols: set[str] = field(default_factory=set)
    active_symbols: set[str] = field(default_factory=set)

    risk_halt_active: bool = False
    risk_halt_reason: str | None = None

    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started_at = ensure_aware_utc(self.started_at)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def set_risk_halt(
        self,
        *,
        active: bool,
        reason: str | None = None,
    ) -> None:
        """
        Mirror global risk halt / kill-switch state for strategy runtime,
        dashboard and backtesting.
        """
        self.risk_halt_active = bool(active)
        self.risk_halt_reason = reason if active else None
        self.metadata["risk_halt_active"] = self.risk_halt_active
        self.metadata["risk_halt_reason"] = self.risk_halt_reason
        self.touch()

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
        timeframe: Timeframe = Timeframe.M1,
        include_regime: bool = True,
        include_portfolio: bool = True,
    ) -> StrategyContext:
        context = self.contexts.build_context(
            symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            include_regime=include_regime,
            include_portfolio=include_portfolio,
        )
        self.active_symbols.add(context.symbol)
        self.touch()
        return context

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

    def mark_signal_status(
        self,
        *,
        signal_id: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        side: SignalSide | None = None,
        status: SignalStatus,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategySignal | None:
        signal = self.signals.find_signal(
            signal_id=signal_id,
            symbol=symbol,
            strategy_name=strategy_name,
            side=side,
        )

        if signal is None:
            return None

        updated = self.signals.mark_status(
            signal,
            status=status,
            reason=reason,
            active=status in {SignalStatus.NEW, SignalStatus.PENDING, SignalStatus.CONFIRMED},
            metadata=metadata,
        )

        self.metrics.record_status(status)
        self.touch()
        return updated

    def mark_signal_status_from_payload(
        self,
        *,
        payload: dict[str, Any],
        status: SignalStatus,
        default_reason: str,
    ) -> StrategySignal | None:
        updated = self.signals.mark_status_from_payload(
            payload=payload,
            status=status,
            default_reason=default_reason,
        )

        if updated is not None:
            self.metrics.record_status(status)
            self.touch()

        return updated

    def get_signal_by_id(self, signal_id: str | None) -> StrategySignal | None:
        return self.signals.get_by_signal_id(signal_id)

    def set_regime(self, snapshot: RegimeSnapshot) -> None:
        snapshot.validate()

        self.contexts.set_regime(snapshot)
        self.active_symbols.add(snapshot.symbol)
        self.touch()

    def set_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        snapshot.validate()

        self.contexts.set_portfolio(snapshot)
        self.touch()

    def upsert_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()

        self.contexts.put_feature(snapshot)
        self.active_symbols.add(snapshot.symbol)
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
        return symbol in self.blocked_symbols

    def clear_symbol(self, symbol: str) -> None:
        self.signals.clear_symbol(symbol)
        self.contexts.remove_symbol(symbol)
        self.blocked_symbols.discard(symbol)
        self.active_symbols.discard(symbol)
        self.touch()

    def prune(
        self,
        *,
        max_signal_age_seconds: int,
    ) -> dict[str, int]:
        if max_signal_age_seconds <= 0:
            raise StrategyStateError("max_signal_age_seconds must be > 0")

        cutoff = utcnow() - timedelta(seconds=max_signal_age_seconds)

        removed_signals = self.signals.prune_older_than(cutoff)
        removed_cooldowns = self.cooldowns.prune_expired()
        removed_features = self.contexts.prune_stale_features()

        self.touch()

        return {
            "signals_history": removed_signals.get("history", 0),
            "signals_rejected": removed_signals.get("rejected", 0),
            "signals_expired": removed_signals.get("expired", 0),
            "signals_confirmed": removed_signals.get("confirmed", 0),
            "signals_executed": removed_signals.get("executed", 0),
            "signals_failed": removed_signals.get("failed", 0),
            "signals_cancelled": removed_signals.get("cancelled", 0),
            "cooldowns": removed_cooldowns,
            "features": removed_features,
        }

    def reset(self) -> None:
        self.signals.clear()
        self.contexts.clear()
        self.cooldowns.clear()
        self.metrics.reset()
        self.blocked_symbols.clear()
        self.active_symbols.clear()
        self.started_at = utcnow()
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
            "metadata": dict(self.metadata),
        }


__all__ = [
    "SignalState",
    "StrategyContextStore",
    "StrategyCooldownState",
    "StrategyMetricsState",
    "StrategyRuntimeState",
]