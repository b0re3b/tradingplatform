from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import (
    InstrumentType,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from analytics.spreads.models import SpreadSignal, SpreadSnapshot
from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base_spread_strategy import (
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
    SPOT_FUTURES_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT as ANALYTICS_SPREAD_SIGNAL_EVENT,
    STATE_CLOSED,
)


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")


# ============================================================
# Config helpers
# ============================================================

def _validate_non_negative_decimal(name: str, value: Decimal) -> None:
    if value < DECIMAL_ZERO:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_decimal(name: str, value: Decimal) -> None:
    if value <= DECIMAL_ZERO:
        raise ValueError(f"{name} must be > 0")


def _normalize_regime_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()

    return {
        str(value).strip().lower()
        for value in values
        if value is not None and str(value).strip()
    }


def _normalize_exchange_set(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
    if not values:
        return set()

    return {
        str(value).strip().lower()
        for value in values
        if value is not None and str(value).strip()
    }


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class SpotFuturesBasisStrategyConfig(BaseSpreadStrategyConfig):
    """
    Strategy-layer config для spot/futures basis mean-reversion strategy.

    Працює поверх готових analytics.spreads моделей:
    - SpreadSnapshot з analytics.spreads.spot_futures.updated;
    - SpreadSignal з analytics.spreads.signal.generated.

    Не відповідає за:
    - розрахунок basis;
    - розрахунок z-score;
    - розрахунок funding-adjusted spread;
    - побудову analytics signals;
    - execution/risk approval.
    """

    # Entry / lifecycle thresholds
    entry_zscore: Decimal = Decimal("2.0")
    exit_zscore: Decimal = Decimal("0.75")
    reduce_zscore: Decimal = Decimal("1.25")
    stop_zscore: Decimal = Decimal("4.5")

    # Edge filters
    min_funding_adjusted_edge: Decimal = Decimal("0")
    min_basis_abs: Decimal = Decimal("0")

    # Confirmation policy
    require_mean_reversion_signal: bool = False
    require_regime_shift_confirmation: bool = False
    allow_regime_shift_entry: bool = True

    # Data-quality signal policy
    close_on_data_quality_signal: bool = True
    block_entry_on_data_quality_signal: bool = True

    # Confirmation bucket policy
    max_signals_per_key: int = 20

    # Regime allowlist
    allowed_regimes: set[str] = field(
        default_factory=lambda: {
            SpreadRegime.ELEVATED.value,
            SpreadRegime.EXTREME.value,
            SpreadRegime.DISLOCATED.value,
        }
    )

    # Leg-specific allowlists
    allowed_spot_exchanges: set[str] = field(default_factory=set)
    allowed_futures_exchanges: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        BaseSpreadStrategyConfig.__post_init__(self)

        self.allowed_regimes = _normalize_regime_set(self.allowed_regimes)
        self.allowed_spot_exchanges = _normalize_exchange_set(self.allowed_spot_exchanges)
        self.allowed_futures_exchanges = _normalize_exchange_set(
            self.allowed_futures_exchanges
        )

        self.validate_spot_futures()

    def validate_spot_futures(self) -> None:
        _validate_positive_decimal("entry_zscore", self.entry_zscore)
        _validate_non_negative_decimal("exit_zscore", self.exit_zscore)
        _validate_non_negative_decimal("reduce_zscore", self.reduce_zscore)
        _validate_positive_decimal("stop_zscore", self.stop_zscore)

        _validate_non_negative_decimal(
            "min_funding_adjusted_edge",
            self.min_funding_adjusted_edge,
        )
        _validate_non_negative_decimal("min_basis_abs", self.min_basis_abs)

        if self.exit_zscore > self.reduce_zscore:
            raise ValueError("exit_zscore must be <= reduce_zscore")

        if self.reduce_zscore > self.entry_zscore:
            raise ValueError("reduce_zscore must be <= entry_zscore")

        if self.stop_zscore <= self.entry_zscore:
            raise ValueError("stop_zscore must be > entry_zscore")

        if self.max_signals_per_key <= 0:
            raise ValueError("max_signals_per_key must be > 0")

        unknown_regimes = self.allowed_regimes - set(SpreadRegime.values())
        if unknown_regimes:
            raise ValueError(
                "allowed_regimes contains unsupported values: "
                f"{', '.join(sorted(unknown_regimes))}"
            )


# ============================================================
# Strategy
# ============================================================

class SpotFuturesBasisStrategy(BaseSpreadStrategy):
    """
    Production-grade strategy layer для spot/futures basis mean reversion.

    Вхідні події:
    - analytics.spreads.spot_futures.updated -> SpreadSnapshot;
    - analytics.spreads.signal.generated -> SpreadSignal.

    Роль:
    - слухати готові analytics payload-и;
    - перевіряти contract/freshness/allowlists;
    - визначати LONG_BASIS / SHORT_BASIS;
    - вести lifecycle setup-а;
    - публікувати strategy-level intents через signal.*.

    Не відповідає за:
    - отримання market-data;
    - побудову SpreadSnapshot;
    - побудову SpreadSignal;
    - risk approval;
    - execution;
    - storage напряму.
    """

    STRATEGY_NAME = "spot_futures_basis"

    SNAPSHOT_EVENT = SPOT_FUTURES_SNAPSHOT_EVENT
    SPREAD_SIGNAL_EVENT = ANALYTICS_SPREAD_SIGNAL_EVENT

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
        event_bus: EventBus,
        config: SpotFuturesBasisStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        resolved_config = config or SpotFuturesBasisStrategyConfig()

        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=self.STRATEGY_NAME,
        )

        self._config: SpotFuturesBasisStrategyConfig = resolved_config

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
                "ignored_snapshots": 0,
                "ignored_signals": 0,
                "tradeability_ignores": 0,
                "invalid_payloads": 0,
                "invalid_contracts": 0,
                "mean_reversion_confirmations": 0,
                "regime_shift_confirmations": 0,
                "anomaly_confirmations": 0,
                "widening_confirmations": 0,
                "data_quality_blocks": 0,
                "stale_signals_removed": 0,
                "bias_flips": 0,
            }
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _subscribe_events(self) -> None:
        await self._subscribe_payload(
            self.SNAPSHOT_EVENT,
            self.on_spot_futures_snapshot,
            name="on_spot_futures_snapshot",
        )
        await self._subscribe_payload(
            self.SPREAD_SIGNAL_EVENT,
            self.on_spread_signal,
            name="on_spread_signal",
        )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

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
            "ignored_snapshots": self._stats["ignored_snapshots"],
            "ignored_signals": self._stats["ignored_signals"],
            "tradeability_ignores": self._stats["tradeability_ignores"],
            "invalid_payloads": self._stats["invalid_payloads"],
            "invalid_contracts": self._stats["invalid_contracts"],
            "mean_reversion_confirmations": self._stats["mean_reversion_confirmations"],
            "regime_shift_confirmations": self._stats["regime_shift_confirmations"],
            "anomaly_confirmations": self._stats["anomaly_confirmations"],
            "widening_confirmations": self._stats["widening_confirmations"],
            "data_quality_blocks": self._stats["data_quality_blocks"],
            "stale_signals_removed": self._stats["stale_signals_removed"],
            "bias_flips": self._stats["bias_flips"],
            "tracked_snapshots": len(self._latest_snapshots),
            "tracked_signal_keys": len(self._latest_signals),
            "tracked_signals": sum(len(items) for items in self._latest_signals.values()),
        }

    def get_latest_snapshot(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
    ) -> SpreadSnapshot | None:
        key = self._build_state_key(
            self._normalize_symbol(symbol),
            self._normalize_exchange(spot_exchange),
            self._normalize_exchange(futures_exchange),
        )
        return self._latest_snapshots.get(key)

    def get_latest_signals(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
    ) -> list[SpreadSignal]:
        key = self._build_state_key(
            self._normalize_symbol(symbol),
            self._normalize_exchange(spot_exchange),
            self._normalize_exchange(futures_exchange),
        )
        self._prune_stale_signals(key)
        return list(self._latest_signals.get(key, []))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_spread_signal(self, signal: SpreadSignal) -> None:
        if not self.is_running:
            return

        if not isinstance(signal, SpreadSignal):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid payload for spot/futures spread signal | strategy=%s payload_type=%s",
                self.STRATEGY_NAME,
                type(signal).__name__,
            )
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["spread_signals_received"] += 1

                if self._reject_disabled():
                    return

                if signal.spread_type != SpreadType.SPOT_FUTURES:
                    self._stats["ignored_signals"] += 1
                    return

                if self._reject_stale_signal(signal.timestamp):
                    self._stats["ignored_signals"] += 1
                    return

                key = self._build_key_from_signal(signal)
                if not key:
                    self._stats["invalid_contracts"] += 1
                    self._stats["ignored_signals"] += 1
                    self._logger.warning(
                        "Spread signal cannot be correlated | strategy=%s symbol=%s exchange_a=%s exchange_b=%s signal_type=%s",
                        self.STRATEGY_NAME,
                        signal.symbol,
                        signal.exchange_a,
                        signal.exchange_b,
                        signal.signal_type.value,
                    )
                    return

                self._store_signal(key, signal)
                self._record_signal_confirmation(signal)

                if (
                    signal.signal_type in {
                        SpreadSignalType.STALE_DATA,
                        SpreadSignalType.INVALID_DATA,
                    }
                    and self._config.close_on_data_quality_signal
                ):
                    await self._handle_data_quality_signal(key, signal)

            except Exception as exc:
                self._mark_exception(
                    "Failed to process spot/futures spread signal",
                    exc,
                    symbol=getattr(signal, "symbol", None),
                    exchange_a=getattr(signal, "exchange_a", None),
                    exchange_b=getattr(signal, "exchange_b", None),
                    signal_type=getattr(getattr(signal, "signal_type", None), "value", None),
                )

    async def on_spot_futures_snapshot(self, snapshot: SpreadSnapshot) -> None:
        if not self.is_running:
            return

        if not isinstance(snapshot, SpreadSnapshot):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid payload for spot/futures snapshot | strategy=%s payload_type=%s",
                self.STRATEGY_NAME,
                type(snapshot).__name__,
            )
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["snapshots_received"] += 1

                if self._reject_disabled():
                    return

                key = self._build_key_from_snapshot(snapshot)

                contract_error = self._snapshot_contract_error(snapshot)
                if contract_error is not None:
                    self._stats["invalid_contracts"] += 1
                    await self._reject_snapshot(
                        snapshot,
                        key or "invalid_spot_futures_snapshot",
                        contract_error,
                    )
                    return

                if self._mark_event_seen(
                    key=f"snapshot|{key}",
                    timestamp=snapshot.timestamp,
                ):
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

                self._prune_stale_signals(key)

                state = self._get_state(key)
                confirmation = self._resolve_confirmation(snapshot)
                bias = self._resolve_bias(snapshot)

                if (
                    self._config.block_entry_on_data_quality_signal
                    and confirmation["has_data_quality_block"]
                ):
                    self._stats["data_quality_blocks"] += 1
                    if state is not None and state.is_active:
                        await self._stop_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="data_quality_signal_for_active_setup",
                            confirmation=confirmation,
                        )
                    else:
                        self._ignore_snapshot(
                            snapshot,
                            key,
                            reason="data_quality_signal_blocks_entry",
                        )
                    return

                if (
                    state is not None
                    and state.is_active
                    and state.bias is not None
                    and bias is not None
                    and state.bias != bias
                ):
                    self._stats["bias_flips"] += 1
                    await self._stop_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="basis_bias_flipped",
                        confirmation=confirmation,
                    )
                    return

                if state is not None and state.is_active:
                    if self._should_stop(snapshot, state):
                        await self._stop_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="basis_dislocation_worsened",
                            confirmation=confirmation,
                        )
                        return

                    if self._should_close(snapshot, state):
                        await self._close_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="basis_mean_reverted",
                            confirmation=confirmation,
                        )
                        return

                if not self._is_tradeable(snapshot, confirmation):
                    if state is not None and state.is_active:
                        await self._close_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="tradeability_lost",
                            confirmation=confirmation,
                        )
                    else:
                        self._ignore_snapshot(
                            snapshot,
                            key,
                            reason="snapshot_not_tradeable",
                        )
                    return

                state = self._get_or_create_state(
                    key=key,
                    symbol=snapshot.symbol,
                    exchange_a=snapshot.leg_a_exchange,
                    exchange_b=snapshot.leg_b_exchange,
                    bias=bias,
                    metadata={
                        "spread_type": SpreadType.SPOT_FUTURES.value,
                        "spot_exchange": snapshot.leg_a_exchange,
                        "futures_exchange": snapshot.leg_b_exchange,
                        "spot_instrument_type": snapshot.leg_a_type.value,
                        "futures_instrument_type": snapshot.leg_b_type.value,
                    },
                )

                if self._should_open(snapshot, state, confirmation):
                    if self._should_skip_by_cooldown(key, now=snapshot.timestamp):
                        return

                    confidence = self._resolve_confidence(snapshot, confirmation)

                    self._set_state_open(
                        state,
                        bias=bias,
                        reason="open_basis_setup",
                        entry_value=snapshot.spread_bps,
                        entry_zscore=self._extract_zscore(snapshot),
                        entry_net_edge=snapshot.funding_adjusted_spread,
                        confidence=confidence,
                        metadata=self._build_snapshot_metadata(
                            snapshot,
                            confirmation,
                            bias,
                        ),
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
                        confidence=confidence,
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_open_payload(
                            snapshot,
                            state,
                            confirmation,
                            bias,
                        ),
                    )
                    return

                if self._should_reduce(snapshot, state):
                    confidence = self._resolve_confidence(snapshot, confirmation)

                    self._set_state_open(
                        state,
                        bias=state.bias,
                        reason="reduce_basis_setup",
                        entry_value=state.entry_value,
                        entry_zscore=state.entry_zscore,
                        entry_net_edge=snapshot.funding_adjusted_spread,
                        confidence=confidence,
                        metadata=self._build_snapshot_metadata(
                            snapshot,
                            confirmation,
                            state.bias,
                        ),
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
                        confidence=confidence,
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_reduce_payload(
                            snapshot,
                            state,
                            confirmation,
                        ),
                    )
                    return

                if self._should_update(snapshot, state, confirmation):
                    confidence = self._resolve_confidence(snapshot, confirmation)

                    self._set_state_open(
                        state,
                        bias=state.bias,
                        reason="update_basis_setup",
                        entry_value=state.entry_value,
                        entry_zscore=state.entry_zscore,
                        entry_net_edge=snapshot.funding_adjusted_spread,
                        confidence=confidence,
                        metadata=self._build_snapshot_metadata(
                            snapshot,
                            confirmation,
                            state.bias,
                        ),
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
                        confidence=confidence,
                        spread_type=SpreadType.SPOT_FUTURES.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_update_payload(
                            snapshot,
                            state,
                            confirmation,
                        ),
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

    # ------------------------------------------------------------------
    # Key / contract helpers
    # ------------------------------------------------------------------

    def _build_key_from_snapshot(self, snapshot: SpreadSnapshot) -> str:
        return self._build_state_key(
            self._normalize_symbol(snapshot.symbol),
            self._normalize_exchange(snapshot.leg_a_exchange),
            self._normalize_exchange(snapshot.leg_b_exchange),
        )

    def _build_key_from_signal(self, signal: SpreadSignal) -> str:
        exchange_a = signal.exchange_a or signal.metadata.get("spot_exchange")
        exchange_b = signal.exchange_b or signal.metadata.get("futures_exchange")

        normalized_symbol = self._normalize_symbol(signal.symbol)
        normalized_a = self._normalize_exchange(str(exchange_a) if exchange_a else None)
        normalized_b = self._normalize_exchange(str(exchange_b) if exchange_b else None)

        if not normalized_symbol or not normalized_a or not normalized_b:
            return ""

        return self._build_state_key(
            normalized_symbol,
            normalized_a,
            normalized_b,
        )

    def _snapshot_contract_error(self, snapshot: SpreadSnapshot) -> str | None:
        if snapshot.spread_type != SpreadType.SPOT_FUTURES:
            return "unsupported_spread_type"

        if snapshot.leg_a_type != InstrumentType.SPOT:
            return "leg_a_not_spot"

        if snapshot.leg_b_type not in InstrumentType.derivatives():
            return "leg_b_not_derivative"

        if not snapshot.symbol:
            return "missing_symbol"

        if not snapshot.leg_a_exchange:
            return "missing_spot_exchange"

        if not snapshot.leg_b_exchange:
            return "missing_futures_exchange"

        return None

    def _reject_snapshot_exchanges(self, snapshot: SpreadSnapshot) -> bool:
        spot_exchange = self._normalize_exchange(snapshot.leg_a_exchange)
        futures_exchange = self._normalize_exchange(snapshot.leg_b_exchange)

        if self._config.allowed_spot_exchanges:
            if spot_exchange not in self._config.allowed_spot_exchanges:
                self._stats["exchange_skips"] += 1
                return True

        if self._config.allowed_futures_exchanges:
            if futures_exchange not in self._config.allowed_futures_exchanges:
                self._stats["exchange_skips"] += 1
                return True

        return False

    # ------------------------------------------------------------------
    # Signal storage / confirmation helpers
    # ------------------------------------------------------------------

    def _store_signal(self, key: str, signal: SpreadSignal) -> None:
        bucket = self._latest_signals.setdefault(key, [])
        bucket.append(signal)

        self._prune_stale_signals(key)

        max_size = self._config.max_signals_per_key
        if len(bucket) > max_size:
            del bucket[:-max_size]

    def _prune_stale_signals(self, key: str) -> int:
        bucket = self._latest_signals.get(key)
        if not bucket:
            return 0

        before = len(bucket)
        bucket[:] = [
            signal
            for signal in bucket
            if self._is_signal_fresh(signal.timestamp)
        ]
        removed = before - len(bucket)

        if removed:
            self._stats["stale_signals_removed"] += removed

        if not bucket:
            self._latest_signals.pop(key, None)

        return removed

    def _record_signal_confirmation(self, signal: SpreadSignal) -> None:
        if signal.signal_type == SpreadSignalType.MEAN_REVERSION:
            self._stats["mean_reversion_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.REGIME_SHIFT:
            self._stats["regime_shift_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.ANOMALY:
            self._stats["anomaly_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.WIDENING:
            self._stats["widening_confirmations"] += 1
        elif signal.signal_type in {
            SpreadSignalType.STALE_DATA,
            SpreadSignalType.INVALID_DATA,
        }:
            self._stats["data_quality_blocks"] += 1

    def _resolve_confirmation(self, snapshot: SpreadSnapshot) -> dict[str, Any]:
        key = self._build_key_from_snapshot(snapshot)
        self._prune_stale_signals(key)

        signals = self._latest_signals.get(key, [])

        mean_reversion_signal: SpreadSignal | None = None
        regime_shift_signal: SpreadSignal | None = None
        anomaly_signal: SpreadSignal | None = None
        widening_signal: SpreadSignal | None = None
        stale_data_signal: SpreadSignal | None = None
        invalid_data_signal: SpreadSignal | None = None

        for signal in reversed(signals):
            if (
                signal.signal_type == SpreadSignalType.MEAN_REVERSION
                and mean_reversion_signal is None
            ):
                mean_reversion_signal = signal
            elif (
                signal.signal_type == SpreadSignalType.REGIME_SHIFT
                and regime_shift_signal is None
            ):
                regime_shift_signal = signal
            elif (
                signal.signal_type == SpreadSignalType.ANOMALY
                and anomaly_signal is None
            ):
                anomaly_signal = signal
            elif (
                signal.signal_type == SpreadSignalType.WIDENING
                and widening_signal is None
            ):
                widening_signal = signal
            elif (
                signal.signal_type == SpreadSignalType.STALE_DATA
                and stale_data_signal is None
            ):
                stale_data_signal = signal
            elif (
                signal.signal_type == SpreadSignalType.INVALID_DATA
                and invalid_data_signal is None
            ):
                invalid_data_signal = signal

        has_data_quality_block = (
            stale_data_signal is not None
            or invalid_data_signal is not None
        )

        return {
            "has_mean_reversion_signal": mean_reversion_signal is not None,
            "has_regime_shift_signal": regime_shift_signal is not None,
            "has_anomaly_signal": anomaly_signal is not None,
            "has_widening_signal": widening_signal is not None,
            "has_stale_data_signal": stale_data_signal is not None,
            "has_invalid_data_signal": invalid_data_signal is not None,
            "has_data_quality_block": has_data_quality_block,
            "mean_reversion_signal": mean_reversion_signal,
            "regime_shift_signal": regime_shift_signal,
            "anomaly_signal": anomaly_signal,
            "widening_signal": widening_signal,
            "stale_data_signal": stale_data_signal,
            "invalid_data_signal": invalid_data_signal,
            "signal_count": len(signals),
        }

    async def _handle_data_quality_signal(
        self,
        key: str,
        signal: SpreadSignal,
    ) -> None:
        state = self._get_state(key)
        if state is None or not state.is_active:
            return

        snapshot = self._latest_snapshots.get(key)
        if snapshot is None:
            self._set_state_closed(
                state,
                status=STATE_CLOSED,
                reason="data_quality_signal_without_latest_snapshot",
                metadata={
                    "signal_type": signal.signal_type.value,
                    "signal_message": signal.message,
                    "signal_timestamp": self._safe_isoformat(signal.timestamp),
                },
                now=signal.timestamp,
            )
            self._stats["stopped_setups"] += 1

            await self._emit_closed(
                action=self.ACTION_STOP,
                symbol=signal.symbol,
                state_key=state.key,
                exchange_a=signal.exchange_a,
                exchange_b=signal.exchange_b,
                reason="data_quality_signal_without_latest_snapshot",
                confidence=signal.confidence,
                spread_type=SpreadType.SPOT_FUTURES.value,
                timestamp=signal.timestamp,
                metadata={
                    "source": "spread_signal",
                    "signal_type": signal.signal_type.value,
                    "signal_message": signal.message,
                    "signal_timestamp": self._safe_isoformat(signal.timestamp),
                },
            )
            return

        await self._stop_from_snapshot(
            snapshot=snapshot,
            state=state,
            reason=f"data_quality_signal_{signal.signal_type.value}",
            confirmation=self._resolve_confirmation(snapshot),
        )

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

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

    def _resolve_confidence(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> Decimal:
        base = Decimal("0.50")

        zscore = self._extract_zscore(snapshot)
        if zscore is not None:
            abs_zscore = abs(zscore)

            if abs_zscore >= self._config.entry_zscore:
                base += Decimal("0.10")
            if abs_zscore >= self._config.reduce_zscore:
                base += Decimal("0.05")
            if abs_zscore >= self._config.stop_zscore:
                base -= Decimal("0.10")

        if snapshot.regime.value in self._config.allowed_regimes:
            base += Decimal("0.10")

        if confirmation.get("has_mean_reversion_signal"):
            base += Decimal("0.10")

        if confirmation.get("has_regime_shift_signal"):
            base += Decimal("0.05")

        if confirmation.get("has_anomaly_signal"):
            base += Decimal("0.05")

        if confirmation.get("has_widening_signal"):
            base += Decimal("0.03")

        if confirmation.get("has_data_quality_block"):
            base -= Decimal("0.25")

        if snapshot.funding_adjusted_spread is not None:
            base += Decimal("0.05")

        if snapshot.basis is not None:
            base += Decimal("0.02")

        return min(max(base, DECIMAL_ZERO), Decimal("0.99"))

    def _is_tradeable(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> bool:
        if confirmation.get("has_data_quality_block"):
            return False

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

        has_mean_reversion = bool(confirmation.get("has_mean_reversion_signal"))
        has_regime_shift = bool(confirmation.get("has_regime_shift_signal"))

        if self._config.require_mean_reversion_signal and not has_mean_reversion:
            return False

        if self._config.require_regime_shift_confirmation and not has_regime_shift:
            return False

        if has_regime_shift and not self._config.allow_regime_shift_entry:
            return False

        if self._resolve_bias(snapshot) is None:
            return False

        confidence = self._resolve_confidence(snapshot, confirmation)
        if confidence < self._config.min_confidence:
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
        return self._config.exit_zscore < abs_zscore <= self._config.reduce_zscore

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
            entry_zscore = state.entry_zscore
            if entry_zscore is not None and abs(zscore) > abs(entry_zscore):
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

        confidence_changed = abs(new_confidence - old_confidence) >= Decimal("0.05")

        current_edge = snapshot.funding_adjusted_spread
        previous_edge = state.entry_net_edge

        edge_changed = (
            current_edge is not None
            and previous_edge is not None
            and abs(current_edge - previous_edge) >= self._config.min_funding_adjusted_edge
        )

        return confidence_changed or edge_changed

    # ------------------------------------------------------------------
    # State transitions / emissions
    # ------------------------------------------------------------------

    def _ignore_snapshot(
        self,
        snapshot: SpreadSnapshot,
        key: str,
        *,
        reason: str,
    ) -> None:
        self._stats["ignored_snapshots"] += 1
        if reason == "snapshot_not_tradeable":
            self._stats["tradeability_ignores"] += 1

        self._logger.debug(
            "Spot/futures snapshot ignored | strategy=%s symbol=%s key=%s reason=%s",
            self.STRATEGY_NAME,
            snapshot.symbol,
            key,
            reason,
            extra={
                "strategy": self.STRATEGY_NAME,
                "symbol": snapshot.symbol,
                "state_key": key,
                "reason": reason,
                "zscore": self._to_decimal_str(self._extract_zscore(snapshot)),
                "basis": self._to_decimal_str(snapshot.basis),
                "funding_adjusted_spread": self._to_decimal_str(
                    snapshot.funding_adjusted_spread
                ),
                "regime": snapshot.regime.value,
            },
        )

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
            metadata={
                "spread_type": snapshot.spread_type.value,
                "rejection_reason": reason,
            },
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
        confirmation: dict[str, Any] | None = None,
    ) -> None:
        confirmation = confirmation or self._resolve_confirmation(snapshot)

        self._set_state_closed(
            state,
            status="closed",
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot, confirmation, state.bias),
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
            metadata=self._build_close_payload(snapshot, state, reason, confirmation),
        )

    async def _stop_from_snapshot(
        self,
        *,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
        confirmation: dict[str, Any] | None = None,
    ) -> None:
        confirmation = confirmation or self._resolve_confirmation(snapshot)

        self._set_state_closed(
            state,
            status="closed",
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot, confirmation, state.bias),
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
            metadata=self._build_close_payload(snapshot, state, reason, confirmation),
        )

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

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
            "funding_adjusted_spread": self._to_decimal_str(
                snapshot.funding_adjusted_spread
            ),
            "zscore": self._to_decimal_str(zscore),
            "regime": snapshot.regime.value,
            "direction": snapshot.direction.value,
            "pricing_source": snapshot.pricing_source.value,
            "quote_validity": snapshot.quote_validity.value,
            "leg_a_type": snapshot.leg_a_type.value,
            "leg_b_type": snapshot.leg_b_type.value,
            "leg_a_mid": self._to_decimal_str(snapshot.leg_a_mid),
            "leg_b_mid": self._to_decimal_str(snapshot.leg_b_mid),
            "leg_a_bid": self._to_decimal_str(snapshot.leg_a_bid),
            "leg_a_ask": self._to_decimal_str(snapshot.leg_a_ask),
            "leg_b_bid": self._to_decimal_str(snapshot.leg_b_bid),
            "leg_b_ask": self._to_decimal_str(snapshot.leg_b_ask),
            "snapshot_timestamp": self._safe_isoformat(snapshot.timestamp),
            "snapshot_metadata": dict(snapshot.metadata),
            "has_mean_reversion_signal": bool(
                confirmation.get("has_mean_reversion_signal", False)
            ),
            "has_regime_shift_signal": bool(
                confirmation.get("has_regime_shift_signal", False)
            ),
            "has_anomaly_signal": bool(
                confirmation.get("has_anomaly_signal", False)
            ),
            "has_widening_signal": bool(
                confirmation.get("has_widening_signal", False)
            ),
            "has_data_quality_block": bool(
                confirmation.get("has_data_quality_block", False)
            ),
            "confirmation_signal_count": confirmation.get("signal_count", 0),
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
            "entry_net_edge": self._to_decimal_str(state.entry_net_edge),
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
            "state_entry_net_edge": self._to_decimal_str(state.entry_net_edge),
            "current_net_edge": self._to_decimal_str(snapshot.funding_adjusted_spread),
        }

    def _build_close_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
        confirmation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        confirmation = confirmation or {}

        return {
            **self._build_snapshot_metadata(snapshot, confirmation, state.bias),
            "state_status_before_close": state.status,
            "state_bias": state.bias,
            "state_opened_at": self._safe_isoformat(state.opened_at),
            "state_updated_at": self._safe_isoformat(state.updated_at),
            "state_closed_at": self._safe_isoformat(state.closed_at),
            "close_reason": reason,
            "entry_zscore": self._to_decimal_str(state.entry_zscore),
            "current_zscore": self._to_decimal_str(self._extract_zscore(snapshot)),
            "entry_spread_bps": self._to_decimal_str(state.entry_value),
            "current_spread_bps": self._to_decimal_str(snapshot.spread_bps),
            "entry_net_edge": self._to_decimal_str(state.entry_net_edge),
            "current_net_edge": self._to_decimal_str(snapshot.funding_adjusted_spread),
        }