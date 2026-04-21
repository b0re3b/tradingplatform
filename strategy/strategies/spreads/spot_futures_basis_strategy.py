from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import SpreadRegime, SpreadSignalType, SpreadType
from analytics.spreads.models import SpreadSignal, SpreadSnapshot
from .base_spread_strategy import (
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
)


@dataclass(slots=True)
class SpotFuturesBasisStrategyConfig(BaseSpreadStrategyConfig):
    """
    Конфігурація стратегії spot-futures basis.

    Це strategy-layer config поверх готових SpreadSnapshot / SpreadSignal.
    """

    entry_zscore: Decimal = Decimal("2.0")
    exit_zscore: Decimal = Decimal("0.75")
    reduce_zscore: Decimal = Decimal("1.25")
    stop_zscore: Decimal = Decimal("4.5")

    min_funding_adjusted_edge: Decimal = Decimal("0")
    min_basis_abs: Decimal = Decimal("0")

    require_mean_reversion_signal: bool = False
    require_regime_shift_confirmation: bool = False
    allow_regime_shift_entry: bool = True

    allowed_regimes: set[str] = field(
        default_factory=lambda: {
            SpreadRegime.ELEVATED.value,
            SpreadRegime.EXTREME.value,
            SpreadRegime.DISLOCATED.value,
        }
    )

    allowed_spot_exchanges: set[str] = field(default_factory=set)
    allowed_futures_exchanges: set[str] = field(default_factory=set)


class SpotFuturesBasisStrategy(BaseSpreadStrategy):
    """
    Strategy layer для spot-futures basis / mean reversion setups.

    Роль:
    - слухає spot/futures snapshots
    - опціонально слухає spread-signal confirmations
    - визначає basis bias:
        - SHORT_BASIS
        - LONG_BASIS
    - веде lifecycle setup-а:
        idle -> pending/open -> reduce -> close/stop
    - публікує strategy-level intents

    Не відповідає за:
    - розрахунок basis / zscore / funding-adjusted spread
    - побудову snapshot-ів
    - signal generation на analytics-рівні
    - execution / position management
    """

    STRATEGY_NAME = "spot_futures_basis"

    SNAPSHOT_EVENT = "spread.spot_futures.updated"
    SPREAD_SIGNAL_EVENT = "spread.signal.generated"

    ACTION_OPEN = "OPEN_BASIS"
    ACTION_REDUCE = "REDUCE_BASIS"
    ACTION_CLOSE = "CLOSE_BASIS"
    ACTION_STOP = "STOP_BASIS"
    ACTION_REJECT = "REJECT_BASIS"
    ACTION_UPDATE = "UPDATE_BASIS"

    BIAS_LONG = "LONG_BASIS"
    BIAS_SHORT = "SHORT_BASIS"

    def __init__(
        self,
        *,
        event_bus: Any,
        config: SpotFuturesBasisStrategyConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        resolved_config = config or SpotFuturesBasisStrategyConfig()
        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=self.STRATEGY_NAME,
        )
        self._config = resolved_config

        self._latest_snapshots: dict[str, SpreadSnapshot] = {}
        self._latest_signals: dict[str, list[SpreadSignal]] = {}

        self._stats.update(
            {
                "snapshots_received": 0,
                "spread_signals_received": 0,
                "opened_setups": 0,
                "reduced_setups": 0,
                "updated_setups": 0,
                "closed_setups": 0,
                "stopped_setups": 0,
                "rejected_setups": 0,
                "mean_reversion_confirmations": 0,
                "regime_shift_confirmations": 0,
                "bias_flips": 0,
            }
        )

    async def _subscribe_events(self) -> None:
        await self._subscribe(self.SNAPSHOT_EVENT, self.on_spot_futures_snapshot)
        await self._subscribe(self.SPREAD_SIGNAL_EVENT, self.on_spread_signal)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.get_base_stats(),
            "snapshots_received": self._stats["snapshots_received"],
            "spread_signals_received": self._stats["spread_signals_received"],
            "opened_setups": self._stats["opened_setups"],
            "reduced_setups": self._stats["reduced_setups"],
            "updated_setups": self._stats["updated_setups"],
            "closed_setups": self._stats["closed_setups"],
            "stopped_setups": self._stats["stopped_setups"],
            "rejected_setups": self._stats["rejected_setups"],
            "mean_reversion_confirmations": self._stats["mean_reversion_confirmations"],
            "regime_shift_confirmations": self._stats["regime_shift_confirmations"],
            "bias_flips": self._stats["bias_flips"],
            "tracked_snapshots": len(self._latest_snapshots),
            "tracked_signal_keys": len(self._latest_signals),
        }

    async def on_spread_signal(self, signal: SpreadSignal) -> None:
        if not self.is_running:
            return

        if signal.spread_type != SpreadType.SPOT_FUTURES:
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["spread_signals_received"] += 1

                if self._reject_disabled():
                    return

                if self._reject_stale_signal(signal.timestamp):
                    return

                key = self._build_key_from_signal(signal)

                bucket = self._latest_signals.setdefault(key, [])
                bucket.append(signal)

                if len(bucket) > 10:
                    del bucket[:-10]

                if signal.signal_type == SpreadSignalType.MEAN_REVERSION:
                    self._stats["mean_reversion_confirmations"] += 1

                if signal.signal_type == SpreadSignalType.REGIME_SHIFT:
                    self._stats["regime_shift_confirmations"] += 1

            except Exception as exc:
                self._mark_exception(
                    "Failed to process spread confirmation signal",
                    exc,
                    symbol=getattr(signal, "symbol", None),
                    exchange_a=getattr(signal, "exchange_a", None),
                    exchange_b=getattr(signal, "exchange_b", None),
                )

    async def on_spot_futures_snapshot(self, snapshot: SpreadSnapshot) -> None:
        if not self.is_running:
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["snapshots_received"] += 1

                if self._reject_disabled():
                    return

                if snapshot.spread_type != SpreadType.SPOT_FUTURES:
                    return

                key = self._build_key_from_snapshot(snapshot)

                if self._mark_event_seen(key=f"snapshot|{key}", timestamp=snapshot.timestamp):
                    return

                self._latest_snapshots[key] = snapshot

                if self._reject_symbol(snapshot.symbol):
                    await self._reject_snapshot(snapshot, key, "symbol_not_allowed")
                    return

                if self._reject_snapshot_exchanges(snapshot):
                    await self._reject_snapshot(snapshot, key, "exchange_not_allowed")
                    return

                if self._reject_stale_snapshot(snapshot.timestamp):
                    await self._reject_snapshot(snapshot, key, "snapshot_stale")
                    return

                state = self._get_or_create_state(
                    key=key,
                    symbol=snapshot.symbol,
                    exchange_a=snapshot.leg_a_exchange,
                    exchange_b=snapshot.leg_b_exchange,
                    bias=None,
                    metadata={
                        "spread_type": SpreadType.SPOT_FUTURES.value,
                        "spot_exchange": snapshot.leg_a_exchange,
                        "futures_exchange": snapshot.leg_b_exchange,
                    },
                )

                bias = self._resolve_bias(snapshot)
                confirmation = self._resolve_confirmation(snapshot)

                if not self._is_tradeable(snapshot, confirmation):
                    if state.is_active and self._should_close(snapshot, state):
                        await self._close_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="tradeability_lost",
                        )
                    else:
                        await self._reject_snapshot(
                            snapshot,
                            key,
                            reason="snapshot_not_tradeable",
                        )
                    return

                if state.is_active and state.bias is not None and bias is not None and state.bias != bias:
                    self._stats["bias_flips"] += 1
                    await self._stop_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="basis_bias_flipped",
                    )
                    return

                if self._should_open(snapshot, state, confirmation):
                    if self._should_skip_by_cooldown(key, now=snapshot.timestamp):
                        return

                    self._set_state_open(
                        state,
                        bias=bias,
                        reason="open_basis_setup",
                        entry_value=snapshot.spread_bps,
                        entry_zscore=self._extract_zscore(snapshot),
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        metadata=self._build_snapshot_metadata(snapshot, confirmation, bias),
                        now=snapshot.timestamp,
                    )
                    self._stats["opened_setups"] += 1

                    await self._emit_generated(
                        action=self.ACTION_OPEN,
                        symbol=snapshot.symbol,
                        state_key=key,
                        exchange_a=snapshot.leg_a_exchange,
                        exchange_b=snapshot.leg_b_exchange,
                        reason="mean_reversion_basis_setup",
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_open_payload(snapshot, state, confirmation, bias),
                    )
                    return

                if self._should_reduce(snapshot, state):
                    self._set_state_open(
                        state,
                        bias=state.bias,
                        reason="reduce_basis_setup",
                        entry_value=state.entry_value,
                        entry_zscore=state.entry_zscore,
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        metadata=self._build_snapshot_metadata(snapshot, confirmation, state.bias),
                        now=snapshot.timestamp,
                    )
                    self._stats["reduced_setups"] += 1

                    await self._emit_updated(
                        action=self.ACTION_REDUCE,
                        symbol=snapshot.symbol,
                        state_key=key,
                        exchange_a=snapshot.leg_a_exchange,
                        exchange_b=snapshot.leg_b_exchange,
                        reason="basis_reversion_progressed",
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_reduce_payload(snapshot, state, confirmation),
                    )
                    return

                if self._should_close(snapshot, state):
                    await self._close_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="basis_mean_reverted",
                    )
                    return

                if self._should_stop(snapshot, state):
                    await self._stop_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="basis_dislocation_worsened",
                    )
                    return

                if self._should_update(snapshot, state, confirmation):
                    self._set_state_open(
                        state,
                        bias=state.bias,
                        reason="update_basis_setup",
                        entry_value=state.entry_value,
                        entry_zscore=state.entry_zscore,
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        metadata=self._build_snapshot_metadata(snapshot, confirmation, state.bias),
                        now=snapshot.timestamp,
                    )
                    self._stats["updated_setups"] += 1

                    await self._emit_updated(
                        action=self.ACTION_UPDATE,
                        symbol=snapshot.symbol,
                        state_key=key,
                        exchange_a=snapshot.leg_a_exchange,
                        exchange_b=snapshot.leg_b_exchange,
                        reason="basis_state_updated",
                        confidence=self._resolve_confidence(snapshot, confirmation),
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_update_payload(snapshot, state, confirmation),
                    )
                    return

            except Exception as exc:
                self._mark_exception(
                    "Failed to process spot/futures snapshot",
                    exc,
                    symbol=getattr(snapshot, "symbol", None),
                    exchange_a=getattr(snapshot, "leg_a_exchange", None),
                    exchange_b=getattr(snapshot, "leg_b_exchange", None),
                )

    def _build_key_from_snapshot(self, snapshot: SpreadSnapshot) -> str:
        return self._build_state_key(
            self._normalize_symbol(snapshot.symbol),
            self._normalize_exchange(snapshot.leg_a_exchange),
            self._normalize_exchange(snapshot.leg_b_exchange),
        )

    def _build_key_from_signal(self, signal: SpreadSignal) -> str:
        return self._build_state_key(
            self._normalize_symbol(signal.symbol),
            self._normalize_exchange(signal.exchange_a),
            self._normalize_exchange(signal.exchange_b),
        )

    def _reject_snapshot_exchanges(self, snapshot: SpreadSnapshot) -> bool:
        if self._config.allowed_spot_exchanges:
            normalized_allowed_spot = {
                self._normalize_exchange(item)
                for item in self._config.allowed_spot_exchanges
            }
            if self._normalize_exchange(snapshot.leg_a_exchange) not in normalized_allowed_spot:
                self._stats["exchange_skips"] += 1
                return True

        if self._config.allowed_futures_exchanges:
            normalized_allowed_fut = {
                self._normalize_exchange(item)
                for item in self._config.allowed_futures_exchanges
            }
            if self._normalize_exchange(snapshot.leg_b_exchange) not in normalized_allowed_fut:
                self._stats["exchange_skips"] += 1
                return True

        return False

    def _extract_zscore(self, snapshot: SpreadSnapshot) -> Decimal | None:
        if snapshot.stats is None:
            return None
        return snapshot.stats.zscore

    def _resolve_bias(self, snapshot: SpreadSnapshot) -> str | None:
        zscore = self._extract_zscore(snapshot)
        if zscore is None:
            return None

        if zscore >= self._config.entry_zscore:
            return self.BIAS_SHORT

        if zscore <= -self._config.entry_zscore:
            return self.BIAS_LONG

        return None

    def _resolve_confirmation(self, snapshot: SpreadSnapshot) -> dict[str, Any]:
        key = self._build_key_from_snapshot(snapshot)
        signals = self._latest_signals.get(key, [])

        mean_reversion_signal: SpreadSignal | None = None
        regime_shift_signal: SpreadSignal | None = None

        for signal in reversed(signals):
            if signal.signal_type == SpreadSignalType.MEAN_REVERSION and mean_reversion_signal is None:
                mean_reversion_signal = signal
            elif signal.signal_type == SpreadSignalType.REGIME_SHIFT and regime_shift_signal is None:
                regime_shift_signal = signal

            if mean_reversion_signal is not None and regime_shift_signal is not None:
                break

        return {
            "has_mean_reversion_signal": mean_reversion_signal is not None,
            "has_regime_shift_signal": regime_shift_signal is not None,
            "mean_reversion_signal": mean_reversion_signal,
            "regime_shift_signal": regime_shift_signal,
        }

    def _resolve_confidence(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> Decimal:
        base = Decimal("0.50")

        zscore = self._extract_zscore(snapshot)
        if zscore is not None:
            abs_z = abs(zscore)

            if abs_z >= self._config.entry_zscore:
                base += Decimal("0.10")
            if abs_z >= self._config.reduce_zscore:
                base += Decimal("0.05")
            if abs_z >= self._config.stop_zscore:
                base += Decimal("0.05")

        if snapshot.regime.value in self._config.allowed_regimes:
            base += Decimal("0.10")

        if confirmation.get("has_mean_reversion_signal"):
            base += Decimal("0.10")

        if confirmation.get("has_regime_shift_signal"):
            base += Decimal("0.05")

        if snapshot.funding_adjusted_spread is not None:
            base += Decimal("0.05")

        return min(base, Decimal("0.99"))

    def _is_tradeable(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> bool:
        zscore = self._extract_zscore(snapshot)
        if zscore is None:
            return False

        if abs(zscore) < self._config.entry_zscore:
            return False

        if snapshot.regime.value not in self._config.allowed_regimes:
            return False

        basis = snapshot.basis
        if basis is None or abs(basis) < self._config.min_basis_abs:
            return False

        funding_adjusted = snapshot.funding_adjusted_spread
        if funding_adjusted is None:
            return False

        if abs(funding_adjusted) < self._config.min_funding_adjusted_edge:
            return False

        if self._config.require_mean_reversion_signal and not confirmation.get("has_mean_reversion_signal"):
            return False

        if (
            self._config.require_regime_shift_confirmation
            and not confirmation.get("has_regime_shift_signal")
        ):
            return False

        confidence = self._resolve_confidence(snapshot, confirmation)
        if confidence < self._config.min_confidence:
            return False

        bias = self._resolve_bias(snapshot)
        if bias is None:
            return False

        return True

    def _should_open(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> bool:
        if state.status in {"open", "pending", "closing"}:
            return False

        return self._is_tradeable(snapshot, confirmation)

    def _should_reduce(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status != "open":
            return False

        zscore = self._extract_zscore(snapshot)
        if zscore is None:
            return False

        abs_zscore = abs(zscore)

        if abs_zscore <= self._config.reduce_zscore and abs_zscore > self._config.exit_zscore:
            return True

        return False

    def _should_close(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status not in {"open", "pending"}:
            return False

        zscore = self._extract_zscore(snapshot)
        if zscore is None:
            return True

        if abs(zscore) <= self._config.exit_zscore:
            return True

        if snapshot.regime in {SpreadRegime.NORMAL, SpreadRegime.COMPRESSED}:
            return True

        bias_now = self._resolve_bias(snapshot)
        if state.bias is not None and bias_now is not None and bias_now != state.bias:
            return True

        return False

    def _should_stop(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status not in {"open", "pending"}:
            return False

        zscore = self._extract_zscore(snapshot)
        if zscore is None:
            return False

        if abs(zscore) >= self._config.stop_zscore:
            return True

        if snapshot.regime == SpreadRegime.DISLOCATED:
            entry_z = state.entry_zscore
            if entry_z is not None and abs(zscore) > abs(entry_z):
                return True

        return False

    def _should_update(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> bool:
        if state.status != "open":
            return False

        new_confidence = self._resolve_confidence(snapshot, confirmation)
        old_confidence = state.confidence

        if old_confidence is None:
            return True

        if abs(new_confidence - old_confidence) >= Decimal("0.05"):
            return True

        return False

    async def _reject_snapshot(
        self,
        snapshot: SpreadSnapshot,
        key: str,
        reason: str,
    ) -> None:
        state = self._get_or_create_state(
            key=key,
            symbol=snapshot.symbol,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            bias=None,
        )
        self._set_state_closed(
            state,
            status="rejected",
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot, {}, None),
            now=snapshot.timestamp,
        )
        self._stats["rejected_setups"] += 1

        await self._emit_rejected(
            action=self.ACTION_REJECT,
            symbol=snapshot.symbol,
            state_key=key,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            reason=reason,
            confidence=self._resolve_confidence(snapshot, {}),
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_snapshot_metadata(snapshot, {}, None),
        )

    async def _close_from_snapshot(
        self,
        *,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
    ) -> None:
        self._set_state_closed(
            state,
            status="closed",
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot, {}, state.bias),
            now=snapshot.timestamp,
        )
        self._stats["closed_setups"] += 1

        await self._emit_closed(
            action=self.ACTION_CLOSE,
            symbol=snapshot.symbol,
            state_key=state.key,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            reason=reason,
            confidence=state.confidence,
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_close_payload(snapshot, state, reason),
        )

    async def _stop_from_snapshot(
        self,
        *,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
    ) -> None:
        self._set_state_closed(
            state,
            status="closed",
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot, {}, state.bias),
            now=snapshot.timestamp,
        )
        self._stats["stopped_setups"] += 1

        await self._emit_closed(
            action=self.ACTION_STOP,
            symbol=snapshot.symbol,
            state_key=state.key,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            reason=reason,
            confidence=state.confidence,
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_close_payload(snapshot, state, reason),
        )

    def _build_snapshot_metadata(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
        bias: str | None,
    ) -> dict[str, Any]:
        zscore = self._extract_zscore(snapshot)

        return {
            "spot_exchange": snapshot.leg_a_exchange,
            "futures_exchange": snapshot.leg_b_exchange,
            "bias": bias,
            "basis": self._to_decimal_str(snapshot.basis),
            "spread_bps": self._to_decimal_str(snapshot.spread_bps),
            "spread_pct": self._to_decimal_str(snapshot.spread_pct),
            "raw_spread": self._to_decimal_str(snapshot.raw_spread),
            "net_spread": self._to_decimal_str(snapshot.net_spread),
            "funding_adjusted_spread": self._to_decimal_str(snapshot.funding_adjusted_spread),
            "zscore": self._to_decimal_str(zscore),
            "regime": snapshot.regime.value,
            "leg_a_type": snapshot.leg_a_type.value,
            "leg_b_type": snapshot.leg_b_type.value,
            "snapshot_timestamp": self._safe_isoformat(snapshot.timestamp),
            "snapshot_metadata": dict(snapshot.metadata),
            "has_mean_reversion_signal": confirmation.get("has_mean_reversion_signal", False),
            "has_regime_shift_signal": confirmation.get("has_regime_shift_signal", False),
        }

    def _build_open_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
        bias: str | None,
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot, confirmation, bias),
            "state_status": state.status,
            "state_bias": state.bias,
            "state_opened_at": self._safe_isoformat(state.opened_at),
            "entry_zscore": self._to_decimal_str(state.entry_zscore),
            "entry_spread_bps": self._to_decimal_str(state.entry_value),
        }

    def _build_reduce_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot, confirmation, state.bias),
            "state_status": state.status,
            "state_bias": state.bias,
            "entry_zscore": self._to_decimal_str(state.entry_zscore),
            "current_zscore": self._to_decimal_str(self._extract_zscore(snapshot)),
            "reduce_reason": "mean_reversion_progress",
        }

    def _build_update_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot, confirmation, state.bias),
            "state_status": state.status,
            "state_bias": state.bias,
            "state_updated_at": self._safe_isoformat(state.updated_at),
            "state_confidence": self._to_decimal_str(state.confidence),
        }

    def _build_close_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot, {}, state.bias),
            "state_status": state.status,
            "state_bias": state.bias,
            "state_opened_at": self._safe_isoformat(state.opened_at),
            "state_closed_at": self._safe_isoformat(state.closed_at),
            "entry_zscore": self._to_decimal_str(state.entry_zscore),
            "close_reason": reason,
        }