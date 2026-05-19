from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from analytics.spreads.enums import (
    InstrumentType,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from analytics.spreads.models import DEFAULT_TIMEFRAME, SpreadSignal, SpreadSnapshot
from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base_spread_strategy import (
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
    SPOT_FUTURES_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT as ANALYTICS_SPREAD_SIGNAL_EVENT,
    STATE_CANCELLED,
    STATE_CLOSED,
)


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")


# ============================================================
# Config helpers
# ============================================================

def _to_decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


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


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


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

    # Update thresholds
    min_update_confidence_delta: Decimal = Decimal("0.05")
    min_update_edge_delta: Decimal = Decimal("0")
    min_update_zscore_delta: Decimal = Decimal("0.25")

    # Confirmation policy
    require_mean_reversion_signal: bool = False
    require_regime_shift_confirmation: bool = False
    allow_regime_shift_entry: bool = True

    # Signal confluence policy
    allow_anomaly_entry: bool = True
    allow_widening_entry: bool = False
    widening_requires_wait: bool = True

    # Data-quality signal policy
    close_on_data_quality_signal: bool = True
    block_entry_on_data_quality_signal: bool = True

    # Quote/snapshot quality policy
    require_valid_quote: bool = True
    require_snapshot_edge: bool = True

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

        self.entry_zscore = _to_decimal(self.entry_zscore, Decimal("2.0")) or Decimal("2.0")
        self.exit_zscore = _to_decimal(self.exit_zscore, Decimal("0.75")) or Decimal("0.75")
        self.reduce_zscore = _to_decimal(self.reduce_zscore, Decimal("1.25")) or Decimal("1.25")
        self.stop_zscore = _to_decimal(self.stop_zscore, Decimal("4.5")) or Decimal("4.5")

        self.min_funding_adjusted_edge = (
            _to_decimal(self.min_funding_adjusted_edge, DECIMAL_ZERO) or DECIMAL_ZERO
        )
        self.min_basis_abs = _to_decimal(self.min_basis_abs, DECIMAL_ZERO) or DECIMAL_ZERO

        self.min_update_confidence_delta = (
            _to_decimal(self.min_update_confidence_delta, Decimal("0.05"))
            or Decimal("0.05")
        )
        self.min_update_edge_delta = (
            _to_decimal(self.min_update_edge_delta, DECIMAL_ZERO) or DECIMAL_ZERO
        )
        self.min_update_zscore_delta = (
            _to_decimal(self.min_update_zscore_delta, Decimal("0.25"))
            or Decimal("0.25")
        )

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
        _validate_non_negative_decimal(
            "min_update_confidence_delta",
            self.min_update_confidence_delta,
        )
        _validate_non_negative_decimal(
            "min_update_edge_delta",
            self.min_update_edge_delta,
        )
        _validate_non_negative_decimal(
            "min_update_zscore_delta",
            self.min_update_zscore_delta,
        )

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
    - перевіряти contract/freshness/allowlists/quote_validity;
    - використовувати basis / funding_adjusted_spread / zscore / regime;
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
                "quote_quality_blocks": 0,
                "edge_blocks": 0,
                "scoped_key_hits": 0,
                "legacy_key_fallbacks": 0,
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
            "quote_quality_blocks": self._stats["quote_quality_blocks"],
            "edge_blocks": self._stats["edge_blocks"],
            "scoped_key_hits": self._stats["scoped_key_hits"],
            "legacy_key_fallbacks": self._stats["legacy_key_fallbacks"],
            "tracked_snapshots": len(self._latest_snapshots),
            "tracked_signal_keys": len(self._latest_signals),
            "tracked_signals": sum(len(items) for items in self._latest_signals.values()),
        }

    def get_latest_snapshot(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
        *,
        spot_market_type: str | None = "spot",
        futures_market_type: str | None = None,
        timeframe: str | None = None,
    ) -> SpreadSnapshot | None:
        key = self._build_scoped_state_key(
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=symbol,
            exchange_a=spot_exchange,
            exchange_b=futures_exchange,
            market_type_a=spot_market_type,
            market_type_b=futures_market_type,
            timeframe=timeframe or DEFAULT_TIMEFRAME,
        )
        return self._latest_snapshots.get(key)

    def get_latest_signals(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
        *,
        spot_market_type: str | None = "spot",
        futures_market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[SpreadSignal]:
        key = self._build_scoped_state_key(
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=symbol,
            exchange_a=spot_exchange,
            exchange_b=futures_exchange,
            market_type_a=spot_market_type,
            market_type_b=futures_market_type,
            timeframe=timeframe or DEFAULT_TIMEFRAME,
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
                        "Spot/futures spread signal cannot be correlated | strategy=%s symbol=%s exchange_a=%s exchange_b=%s signal_type=%s",
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
                    signal.signal_type
                    in {
                        SpreadSignalType.STALE_DATA,
                        SpreadSignalType.INVALID_DATA,
                    }
                    and self._config.close_on_data_quality_signal
                ):
                    await self._handle_data_quality_signal(key, signal)
                    return

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
                    metadata=self._build_snapshot_metadata(
                        snapshot,
                        confirmation,
                        bias,
                    ),
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

                self._ignore_snapshot(
                    snapshot,
                    key,
                    reason="no_state_transition",
                )

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
        self._stats["scoped_key_hits"] += 1
        return self._build_scoped_state_key(
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            market_type_a=snapshot.leg_a_market_type,
            market_type_b=snapshot.leg_b_market_type,
            timeframe=snapshot.timeframe,
        )

    def _build_key_from_signal(self, signal: SpreadSignal) -> str:
        metadata_key = self._metadata_str(
            signal.metadata,
            "state_key",
            "spot_futures_key",
            "basis_key",
        )
        if metadata_key:
            return metadata_key

        exchange_a = signal.exchange_a or self._metadata_str(
            signal.metadata,
            "spot_exchange",
            "exchange_a",
            "leg_a_exchange",
        )
        exchange_b = signal.exchange_b or self._metadata_str(
            signal.metadata,
            "futures_exchange",
            "exchange_b",
            "leg_b_exchange",
        )

        if not signal.symbol or not exchange_a or not exchange_b:
            return ""

        return self._build_scoped_state_key(
            spread_type=signal.spread_type,
            symbol=signal.symbol,
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            market_type_a=signal.market_type_a
            or self._metadata_str(signal.metadata, "spot_market_type", "market_type_a"),
            market_type_b=signal.market_type_b
            or self._metadata_str(signal.metadata, "futures_market_type", "market_type_b"),
            timeframe=signal.timeframe
            or self._metadata_str(signal.metadata, "timeframe")
            or DEFAULT_TIMEFRAME,
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

        if self._config.require_valid_quote and snapshot.quote_validity != QuoteValidity.VALID:
            return "invalid_quote_validity"

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
    # Signal confirmation helpers
    # ------------------------------------------------------------------

    def _store_signal(self, key: str, signal: SpreadSignal) -> None:
        bucket = self._latest_signals.setdefault(key, [])
        bucket.append(signal)

        if len(bucket) > self._config.max_signals_per_key:
            del bucket[: len(bucket) - self._config.max_signals_per_key]

    def _record_signal_confirmation(self, signal: SpreadSignal) -> None:
        if signal.signal_type == SpreadSignalType.MEAN_REVERSION:
            self._stats["mean_reversion_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.REGIME_SHIFT:
            self._stats["regime_shift_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.ANOMALY:
            self._stats["anomaly_confirmations"] += 1
        elif signal.signal_type == SpreadSignalType.WIDENING:
            self._stats["widening_confirmations"] += 1

    def _resolve_confirmation(self, snapshot: SpreadSnapshot) -> dict[str, Any]:
        key = self._build_key_from_snapshot(snapshot)
        self._prune_stale_signals(key)

        signals = self._latest_signals.get(key, [])

        has_mean_reversion = any(
            signal.signal_type == SpreadSignalType.MEAN_REVERSION
            for signal in signals
        )
        has_regime_shift = any(
            signal.signal_type == SpreadSignalType.REGIME_SHIFT
            for signal in signals
        )
        has_anomaly = any(
            signal.signal_type == SpreadSignalType.ANOMALY
            for signal in signals
        )
        has_widening = any(
            signal.signal_type == SpreadSignalType.WIDENING
            for signal in signals
        )
        has_data_quality_block = any(
            signal.signal_type
            in {
                SpreadSignalType.STALE_DATA,
                SpreadSignalType.INVALID_DATA,
            }
            for signal in signals
        )

        return {
            "key": key,
            "has_mean_reversion_signal": has_mean_reversion,
            "has_regime_shift_signal": has_regime_shift,
            "has_anomaly_signal": has_anomaly,
            "has_widening_signal": has_widening,
            "has_data_quality_block": has_data_quality_block,
            "signals": [
                self._signal_metadata(signal)
                for signal in signals
            ],
        }

    def _prune_stale_signals(self, key: str) -> None:
        signals = self._latest_signals.get(key)
        if not signals:
            return

        fresh = [
            signal
            for signal in signals
            if self._is_signal_fresh(signal.timestamp)
        ]

        removed = len(signals) - len(fresh)
        if removed:
            self._stats["stale_signals_removed"] += removed

        if fresh:
            self._latest_signals[key] = fresh
        else:
            self._latest_signals.pop(key, None)

    async def _handle_data_quality_signal(
        self,
        key: str,
        signal: SpreadSignal,
    ) -> None:
        state = self._get_state(key)
        if state is None or not state.is_active:
            return

        self._set_state_closed(
            state,
            status=STATE_CANCELLED,
            reason=f"data_quality_signal:{signal.signal_type.value}",
            metadata={
                "signal": self._signal_metadata(signal),
            },
            now=signal.timestamp,
        )
        self._stats["stopped_setups"] += 1

        await self._emit_cancelled(
            action=self.ACTION_STOP,
            symbol=signal.symbol,
            state_key=state.key,
            exchange_a=state.exchange_a,
            exchange_b=state.exchange_b,
            reason=f"data_quality_signal:{signal.signal_type.value}",
            confidence=signal.confidence or state.confidence,
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=signal.timestamp,
            metadata={
                "state": state.to_payload(),
                "signal": self._signal_metadata(signal),
            },
        )

    # ------------------------------------------------------------------
    # Tradeability / lifecycle decisions
    # ------------------------------------------------------------------

    def _is_tradeable(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> bool:
        if self._config.require_valid_quote and snapshot.quote_validity != QuoteValidity.VALID:
            self._stats["quote_quality_blocks"] += 1
            return False

        if self._config.require_snapshot_edge:
            has_edge = getattr(snapshot, "has_edge", None)
            if has_edge is False:
                self._stats["edge_blocks"] += 1
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
        has_anomaly = bool(confirmation.get("has_anomaly_signal"))
        has_widening = bool(confirmation.get("has_widening_signal"))

        if self._config.require_mean_reversion_signal and not has_mean_reversion:
            return False

        if self._config.require_regime_shift_confirmation and not has_regime_shift:
            return False

        if has_regime_shift and not self._config.allow_regime_shift_entry:
            return False

        if has_widening and self._config.widening_requires_wait:
            return False

        if has_anomaly and not self._config.allow_anomaly_entry:
            return False

        if has_widening and not self._config.allow_widening_entry and not has_mean_reversion:
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

        if self._config.require_valid_quote and snapshot.quote_validity != QuoteValidity.VALID:
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

        if self._config.require_valid_quote and snapshot.quote_validity != QuoteValidity.VALID:
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

        confidence_changed = (
            abs(new_confidence - old_confidence)
            >= self._config.min_update_confidence_delta
        )

        current_edge = snapshot.funding_adjusted_spread
        previous_edge = state.entry_net_edge

        edge_changed = (
            current_edge is not None
            and previous_edge is not None
            and abs(current_edge - previous_edge) >= self._config.min_update_edge_delta
        )

        current_zscore = self._extract_zscore(snapshot)
        previous_zscore = state.entry_zscore

        zscore_changed = (
            current_zscore is not None
            and previous_zscore is not None
            and abs(current_zscore - previous_zscore)
            >= self._config.min_update_zscore_delta
        )

        return confidence_changed or edge_changed or zscore_changed

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
            status=STATE_CLOSED,
            reason=reason,
            metadata=self._build_snapshot_metadata(
                snapshot,
                confirmation={},
                bias=None,
            ),
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
            confidence=self._extract_snapshot_confidence(snapshot),
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata={
                "state": state.to_payload(),
                "snapshot": self._build_snapshot_metadata(
                    snapshot,
                    confirmation={},
                    bias=None,
                ),
            },
        )

    async def _stop_from_snapshot(
        self,
        *,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
        confirmation: dict[str, Any],
    ) -> None:
        self._set_state_closed(
            state,
            status=STATE_CANCELLED,
            reason=reason,
            metadata=self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
            now=snapshot.timestamp,
        )
        self._stats["stopped_setups"] += 1

        await self._emit_cancelled(
            action=self.ACTION_STOP,
            symbol=snapshot.symbol,
            state_key=state.key,
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            reason=reason,
            confidence=self._resolve_confidence(snapshot, confirmation),
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_stop_payload(snapshot, state, confirmation),
        )

    async def _close_from_snapshot(
        self,
        *,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
        confirmation: dict[str, Any],
    ) -> None:
        self._set_state_closed(
            state,
            status=STATE_CLOSED,
            reason=reason,
            metadata=self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
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
            confidence=self._resolve_confidence(snapshot, confirmation),
            spread_type=SpreadType.SPOT_FUTURES.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_close_payload(snapshot, state, confirmation),
        )

    # ------------------------------------------------------------------
    # Metadata / payload builders
    # ------------------------------------------------------------------

    def _build_snapshot_metadata(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
        bias: str | None,
    ) -> dict[str, Any]:
        zscore = self._extract_zscore(snapshot)

        return {
            "source": "analytics.spreads.spot_futures.updated",
            "analytics_topic": self.SNAPSHOT_EVENT,
            "spread_type": snapshot.spread_type.value,
            "symbol": snapshot.symbol,
            "timeframe": snapshot.timeframe,
            "spot_exchange": snapshot.leg_a_exchange,
            "futures_exchange": snapshot.leg_b_exchange,
            "spot_market_type": snapshot.leg_a_market_type,
            "futures_market_type": snapshot.leg_b_market_type,
            "spot_exchange_symbol": snapshot.leg_a_exchange_symbol,
            "futures_exchange_symbol": snapshot.leg_b_exchange_symbol,
            "spot_instrument_type": snapshot.leg_a_type.value,
            "futures_instrument_type": snapshot.leg_b_type.value,
            "pricing_source": snapshot.pricing_source.value,
            "raw_spread": self._to_decimal_str(snapshot.raw_spread),
            "spread_pct": self._to_decimal_str(snapshot.spread_pct),
            "spread_bps": self._to_decimal_str(snapshot.spread_bps),
            "net_spread": self._to_decimal_str(snapshot.net_spread),
            "basis": self._to_decimal_str(snapshot.basis),
            "funding_adjusted_spread": self._to_decimal_str(
                snapshot.funding_adjusted_spread
            ),
            "direction": snapshot.direction.value,
            "regime": snapshot.regime.value,
            "zscore": self._to_decimal_str(zscore),
            "quote_validity": snapshot.quote_validity.value,
            "has_edge": getattr(snapshot, "has_edge", None),
            "bias": bias,
            "leg_a_bid": self._to_decimal_str(snapshot.leg_a_bid),
            "leg_a_ask": self._to_decimal_str(snapshot.leg_a_ask),
            "leg_a_mid": self._to_decimal_str(snapshot.leg_a_mid),
            "leg_b_bid": self._to_decimal_str(snapshot.leg_b_bid),
            "leg_b_ask": self._to_decimal_str(snapshot.leg_b_ask),
            "leg_b_mid": self._to_decimal_str(snapshot.leg_b_mid),
            "estimated_fees": self._to_decimal_str(snapshot.estimated_fees),
            "estimated_slippage": self._to_decimal_str(snapshot.estimated_slippage),
            "timestamp": snapshot.timestamp.isoformat(),
            "confirmation": self._confirmation_payload(confirmation),
            "analytics_metadata": dict(snapshot.metadata),
        }

    def _build_open_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
        bias: str | None,
    ) -> dict[str, Any]:
        return {
            "state": state.to_payload(),
            "snapshot": self._build_snapshot_metadata(snapshot, confirmation, bias),
            "entry_policy": {
                "entry_zscore": str(self._config.entry_zscore),
                "min_basis_abs": str(self._config.min_basis_abs),
                "min_funding_adjusted_edge": str(
                    self._config.min_funding_adjusted_edge
                ),
                "require_mean_reversion_signal": (
                    self._config.require_mean_reversion_signal
                ),
                "require_regime_shift_confirmation": (
                    self._config.require_regime_shift_confirmation
                ),
                "allowed_regimes": sorted(self._config.allowed_regimes),
            },
        }

    def _build_reduce_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": state.to_payload(),
            "snapshot": self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
            "reduce_policy": {
                "exit_zscore": str(self._config.exit_zscore),
                "reduce_zscore": str(self._config.reduce_zscore),
            },
        }

    def _build_update_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": state.to_payload(),
            "snapshot": self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
            "update_policy": {
                "min_update_confidence_delta": str(
                    self._config.min_update_confidence_delta
                ),
                "min_update_edge_delta": str(self._config.min_update_edge_delta),
                "min_update_zscore_delta": str(self._config.min_update_zscore_delta),
            },
        }

    def _build_stop_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": state.to_payload(),
            "snapshot": self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
            "stop_policy": {
                "stop_zscore": str(self._config.stop_zscore),
                "close_on_data_quality_signal": self._config.close_on_data_quality_signal,
                "require_valid_quote": self._config.require_valid_quote,
            },
        }

    def _build_close_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": state.to_payload(),
            "snapshot": self._build_snapshot_metadata(
                snapshot,
                confirmation,
                state.bias,
            ),
            "exit_policy": {
                "exit_zscore": str(self._config.exit_zscore),
                "normal_or_compressed_regime_closes": True,
            },
        }

    def _signal_metadata(self, signal: SpreadSignal) -> dict[str, Any]:
        return {
            "signal_type": signal.signal_type.value,
            "spread_type": signal.spread_type.value,
            "symbol": signal.symbol,
            "exchange_a": signal.exchange_a,
            "exchange_b": signal.exchange_b,
            "market_type_a": signal.market_type_a,
            "market_type_b": signal.market_type_b,
            "timeframe": signal.timeframe,
            "exchange_symbol_a": signal.exchange_symbol_a,
            "exchange_symbol_b": signal.exchange_symbol_b,
            "value": self._to_decimal_str(signal.value),
            "threshold": self._to_decimal_str(signal.threshold),
            "confidence": self._to_decimal_str(signal.confidence),
            "message": signal.message,
            "timestamp": signal.timestamp.isoformat(),
            "metadata": dict(signal.metadata),
        }

    def _confirmation_payload(self, confirmation: dict[str, Any]) -> dict[str, Any]:
        return {
            "has_mean_reversion_signal": bool(
                confirmation.get("has_mean_reversion_signal")
            ),
            "has_regime_shift_signal": bool(
                confirmation.get("has_regime_shift_signal")
            ),
            "has_anomaly_signal": bool(confirmation.get("has_anomaly_signal")),
            "has_widening_signal": bool(confirmation.get("has_widening_signal")),
            "has_data_quality_block": bool(
                confirmation.get("has_data_quality_block")
            ),
            "signals": list(confirmation.get("signals", [])),
        }

    # ------------------------------------------------------------------
    # Extractors / resolvers
    # ------------------------------------------------------------------

    def _extract_zscore(self, snapshot: SpreadSnapshot) -> Decimal | None:
        if snapshot.stats is not None and snapshot.stats.zscore is not None:
            return snapshot.stats.zscore

        metadata_value = self._metadata_str(
            snapshot.metadata,
            "zscore",
            "current_zscore",
        )
        return _to_decimal(metadata_value)

    def _extract_snapshot_confidence(self, snapshot: SpreadSnapshot) -> Decimal | None:
        metadata_value = self._metadata_str(
            snapshot.metadata,
            "confidence",
            "snapshot_confidence",
        )
        return _to_decimal(metadata_value)

    def _resolve_confidence(
        self,
        snapshot: SpreadSnapshot,
        confirmation: dict[str, Any],
    ) -> Decimal:
        metadata_confidence = self._extract_snapshot_confidence(snapshot)
        if metadata_confidence is not None:
            confidence = metadata_confidence
        else:
            confidence = Decimal("0.50")

        zscore = self._extract_zscore(snapshot)
        if zscore is not None:
            # 0.50 base + up to 0.35 from zscore strength.
            zscore_component = min(abs(zscore) / self._config.stop_zscore, DECIMAL_ONE)
            confidence += zscore_component * Decimal("0.35")

        if snapshot.regime == SpreadRegime.EXTREME:
            confidence += Decimal("0.05")
        elif snapshot.regime == SpreadRegime.DISLOCATED:
            confidence += Decimal("0.03")
        elif snapshot.regime == SpreadRegime.ELEVATED:
            confidence += Decimal("0.02")

        if confirmation.get("has_mean_reversion_signal"):
            confidence += Decimal("0.05")

        if confirmation.get("has_regime_shift_signal"):
            confidence += Decimal("0.03")

        if confirmation.get("has_anomaly_signal"):
            confidence += Decimal("0.03")

        if confirmation.get("has_widening_signal") and self._config.widening_requires_wait:
            confidence -= Decimal("0.05")

        if confirmation.get("has_data_quality_block"):
            confidence -= Decimal("0.25")

        if snapshot.quote_validity != QuoteValidity.VALID:
            confidence -= Decimal("0.25")

        if confidence < DECIMAL_ZERO:
            return DECIMAL_ZERO

        if confidence > DECIMAL_ONE:
            return DECIMAL_ONE

        return confidence

    def _resolve_bias(self, snapshot: SpreadSnapshot) -> str | None:
        edge = snapshot.funding_adjusted_spread
        if edge is None:
            edge = snapshot.basis

        if edge is None:
            return None

        if edge > DECIMAL_ZERO:
            return self.BIAS_SHORT

        if edge < DECIMAL_ZERO:
            return self.BIAS_LONG

        if snapshot.direction == SpreadDirection.WIDENING:
            return self.BIAS_SHORT

        if snapshot.direction == SpreadDirection.COMPRESSING:
            return self.BIAS_LONG

        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _metadata_str(
        self,
        metadata: dict[str, Any] | None,
        *keys: str,
    ) -> str | None:
        if not metadata:
            return None

        for key in keys:
            if key in metadata and metadata.get(key) is not None:
                value = str(metadata.get(key)).strip()
                if value:
                    return value

        return None