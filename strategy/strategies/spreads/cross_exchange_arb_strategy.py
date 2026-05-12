from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from analytics.spreads.enums import (
    InstrumentType,
    OpportunityStatus,
    SpreadSignalType,
    SpreadType,
)
from analytics.spreads.models import ArbitrageOpportunity, SpreadSignal, SpreadSnapshot
from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base_spread_strategy import (
    ARBITRAGE_OPPORTUNITY_EVENT,
    CROSS_EXCHANGE_SNAPSHOT_EVENT,
    SPREAD_SIGNAL_EVENT as ANALYTICS_SPREAD_SIGNAL_EVENT,
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
    STATE_CANCELLED,
    STATE_CLOSED,
    STATE_REJECTED,
)


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_10_000 = Decimal("10000")


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


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _normalize_instrument_type_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    normalized: set[str] = set()
    allowed_values = set(InstrumentType.values())

    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        if raw not in allowed_values:
            raise ValueError(
                f"Unsupported instrument type in allowed_instrument_types: {value!r}"
            )
        normalized.add(raw)

    return normalized


# ============================================================
# Config
# ============================================================

@dataclass(slots=True)
class CrossExchangeArbStrategyConfig(BaseSpreadStrategyConfig):
    """
    Strategy-layer config для cross-exchange arbitrage.

    Працює поверх готових analytics.spreads payload-ів:
    - ArbitrageOpportunity з analytics.spreads.arbitrage.opportunity;
    - SpreadSnapshot з analytics.spreads.cross_exchange.updated;
    - SpreadSignal з analytics.spreads.signal.generated.

    Не відповідає за:
    - пошук arbitrage opportunity;
    - розрахунок fees/slippage/costs;
    - побудову SpreadSnapshot;
    - execution;
    - risk approval.
    """

    # Entry/exit thresholds
    entry_min_net_edge: Decimal = Decimal("0")
    entry_min_bps: Decimal = Decimal("5")
    exit_min_net_edge: Decimal = Decimal("0")
    exit_min_bps: Decimal = Decimal("0")

    # Update thresholds
    min_update_net_edge_delta: Decimal = Decimal("0.5")
    min_update_net_edge_bps_delta: Decimal = Decimal("1")
    min_update_confidence_delta: Decimal = Decimal("0.05")

    # Confirmation/persistence
    require_persistence: bool = True
    min_persistence_ms: int = 500

    require_arbitrage_signal_confirmation: bool = False
    max_signals_per_key: int = 20

    # Snapshot-driven lifecycle
    close_on_snapshot_edge_loss: bool = True
    close_on_stale_snapshot: bool = True
    close_on_snapshot_status_not_active: bool = True
    update_from_snapshot_metadata: bool = True

    # Data quality signal policy
    close_on_data_quality_signal: bool = True
    block_entry_on_data_quality_signal: bool = True

    # Instrument filters
    allowed_instrument_types: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.allowed_instrument_types = _normalize_instrument_type_set(
            self.allowed_instrument_types
        )
        self.validate_cross_exchange()

    def validate_cross_exchange(self) -> None:
        _validate_non_negative_decimal("entry_min_net_edge", self.entry_min_net_edge)
        _validate_non_negative_decimal("entry_min_bps", self.entry_min_bps)
        _validate_non_negative_decimal("exit_min_net_edge", self.exit_min_net_edge)
        _validate_non_negative_decimal("exit_min_bps", self.exit_min_bps)
        _validate_non_negative_decimal(
            "min_update_net_edge_delta",
            self.min_update_net_edge_delta,
        )
        _validate_non_negative_decimal(
            "min_update_net_edge_bps_delta",
            self.min_update_net_edge_bps_delta,
        )
        _validate_non_negative_decimal(
            "min_update_confidence_delta",
            self.min_update_confidence_delta,
        )
        _validate_non_negative_int("min_persistence_ms", self.min_persistence_ms)

        if self.max_signals_per_key <= 0:
            raise ValueError("max_signals_per_key must be > 0")


# ============================================================
# Strategy
# ============================================================

class CrossExchangeArbStrategy(BaseSpreadStrategy):
    """
    Production-grade strategy layer для cross-exchange arbitrage.

    Вхідні події:
    - analytics.spreads.arbitrage.opportunity -> ArbitrageOpportunity;
    - analytics.spreads.cross_exchange.updated -> SpreadSnapshot;
    - analytics.spreads.signal.generated -> SpreadSignal.

    Роль:
    - приймати готові analytics opportunities/snapshots/signals;
    - перевіряти allowlists, freshness, status, profitability, thresholds;
    - вести strategy-state lifecycle;
    - публікувати strategy-level intents через signal.*.

    Не відповідає за:
    - market-data;
    - arbitrage detection;
    - cost calculation;
    - execution;
    - risk approval;
    - storage напряму.
    """

    STRATEGY_NAME = "cross_exchange_arb"

    OPPORTUNITY_EVENT = ARBITRAGE_OPPORTUNITY_EVENT
    SNAPSHOT_EVENT = CROSS_EXCHANGE_SNAPSHOT_EVENT
    SPREAD_SIGNAL_EVENT = ANALYTICS_SPREAD_SIGNAL_EVENT

    ACTION_OPEN = "OPEN_ARB"
    ACTION_UPDATE = "UPDATE_ARB"
    ACTION_CANCEL = "CANCEL_ARB"
    ACTION_CLOSE = "CLOSE_ARB"
    ACTION_REJECT = "REJECT_ARB"

    BIAS_ARB = "arb"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: CrossExchangeArbStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        resolved_config = config or CrossExchangeArbStrategyConfig()

        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=self.STRATEGY_NAME,
        )

        self._config: CrossExchangeArbStrategyConfig = resolved_config

        self._latest_opportunities: dict[str, ArbitrageOpportunity] = {}
        self._latest_snapshots: dict[str, SpreadSnapshot] = {}
        self._latest_signals: dict[str, list[SpreadSignal]] = {}
        self._pending_since: dict[str, datetime] = {}

        self._stats.update(
            {
                "opportunities_received": 0,
                "snapshots_received": 0,
                "spread_signals_received": 0,
                "opened_setups": 0,
                "updated_setups": 0,
                "cancelled_setups": 0,
                "closed_setups": 0,
                "rejected_setups": 0,
                "ignored_snapshots": 0,
                "ignored_signals": 0,
                "ignored_opportunities": 0,
                "persistence_waits": 0,
                "expired_opportunities": 0,
                "inactive_opportunities": 0,
                "invalid_payloads": 0,
                "invalid_contracts": 0,
                "arbitrage_signal_confirmations": 0,
                "data_quality_blocks": 0,
                "stale_signals_removed": 0,
                "snapshot_edge_closes": 0,
                "snapshot_status_closes": 0,
                "snapshot_updates": 0,
            }
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _subscribe_events(self) -> None:
        await self._subscribe_payload(
            self.OPPORTUNITY_EVENT,
            self.on_arbitrage_opportunity,
            name="on_arbitrage_opportunity",
        )
        await self._subscribe_payload(
            self.SNAPSHOT_EVENT,
            self.on_cross_exchange_snapshot,
            name="on_cross_exchange_snapshot",
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
            "opportunities_received": self._stats["opportunities_received"],
            "snapshots_received": self._stats["snapshots_received"],
            "spread_signals_received": self._stats["spread_signals_received"],
            "opened_setups": self._stats["opened_setups"],
            "updated_setups": self._stats["updated_setups"],
            "cancelled_setups": self._stats["cancelled_setups"],
            "closed_setups": self._stats["closed_setups"],
            "rejected_setups": self._stats["rejected_setups"],
            "ignored_snapshots": self._stats["ignored_snapshots"],
            "ignored_signals": self._stats["ignored_signals"],
            "ignored_opportunities": self._stats["ignored_opportunities"],
            "persistence_waits": self._stats["persistence_waits"],
            "expired_opportunities": self._stats["expired_opportunities"],
            "inactive_opportunities": self._stats["inactive_opportunities"],
            "invalid_payloads": self._stats["invalid_payloads"],
            "invalid_contracts": self._stats["invalid_contracts"],
            "arbitrage_signal_confirmations": (
                self._stats["arbitrage_signal_confirmations"]
            ),
            "data_quality_blocks": self._stats["data_quality_blocks"],
            "stale_signals_removed": self._stats["stale_signals_removed"],
            "snapshot_edge_closes": self._stats["snapshot_edge_closes"],
            "snapshot_status_closes": self._stats["snapshot_status_closes"],
            "snapshot_updates": self._stats["snapshot_updates"],
            "tracked_opportunities": len(self._latest_opportunities),
            "tracked_snapshots": len(self._latest_snapshots),
            "tracked_signal_keys": len(self._latest_signals),
            "tracked_signals": sum(len(items) for items in self._latest_signals.values()),
            "pending_candidates": len(self._pending_since),
        }

    def get_latest_opportunity(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        instrument_type: InstrumentType | str,
    ) -> ArbitrageOpportunity | None:
        key = self._build_state_key(
            self._normalize_symbol(symbol),
            self._normalize_exchange(buy_exchange),
            self._normalize_exchange(sell_exchange),
            self._instrument_type_value(instrument_type),
        )
        return self._latest_opportunities.get(key)

    def get_latest_snapshot(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        instrument_type: InstrumentType | str,
    ) -> SpreadSnapshot | None:
        key = self._build_state_key(
            self._normalize_symbol(symbol),
            self._normalize_exchange(buy_exchange),
            self._normalize_exchange(sell_exchange),
            self._instrument_type_value(instrument_type),
        )
        return self._latest_snapshots.get(key)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_arbitrage_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        if not self.is_running:
            return

        if not isinstance(opportunity, ArbitrageOpportunity):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid payload for arbitrage opportunity | strategy=%s payload_type=%s",
                self.STRATEGY_NAME,
                type(opportunity).__name__,
            )
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["opportunities_received"] += 1

                if self._reject_disabled():
                    return

                key = self._build_key_from_opportunity(opportunity)

                if self._mark_event_seen(
                    key=f"opportunity|{key}",
                    timestamp=opportunity.timestamp,
                ):
                    return

                self._latest_opportunities[key] = opportunity
                self._prune_stale_signals(key)

                if self._reject_symbol(opportunity.symbol):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="symbol_not_allowed",
                    )
                    return

                if self._reject_exchanges(
                    opportunity.buy_exchange,
                    opportunity.sell_exchange,
                ):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="exchange_not_allowed",
                    )
                    return

                if self._reject_instrument_types(opportunity):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="instrument_type_not_allowed",
                    )
                    return

                if self._reject_confidence(opportunity.confidence):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="confidence_below_threshold",
                    )
                    return

                if not self._is_opportunity_active(opportunity):
                    self._stats["inactive_opportunities"] += 1
                    await self._cancel_if_active(
                        opportunity=opportunity,
                        key=key,
                        reason="opportunity_not_active",
                    )
                    return

                if self._is_opportunity_expired(opportunity):
                    self._stats["expired_opportunities"] += 1
                    await self._cancel_if_active(
                        opportunity=opportunity,
                        key=key,
                        reason="opportunity_expired",
                    )
                    return

                if (
                    self._config.block_entry_on_data_quality_signal
                    and self._has_data_quality_block(key)
                ):
                    self._stats["data_quality_blocks"] += 1
                    state = self._get_state(key)
                    if state is not None and state.is_active:
                        await self._cancel_if_active(
                            opportunity=opportunity,
                            key=key,
                            reason="data_quality_signal_for_active_setup",
                        )
                    else:
                        self._ignore_opportunity(
                            opportunity,
                            key,
                            reason="data_quality_signal_blocks_entry",
                        )
                    return

                if (
                    self._config.require_arbitrage_signal_confirmation
                    and not self._has_arbitrage_signal_confirmation(key)
                ):
                    self._stats["persistence_waits"] += 1
                    state = self._get_or_create_state(
                        key=key,
                        symbol=opportunity.symbol,
                        exchange_a=opportunity.buy_exchange,
                        exchange_b=opportunity.sell_exchange,
                        bias=self.BIAS_ARB,
                        metadata=self._build_common_metadata(opportunity),
                    )
                    self._set_state_pending(
                        state,
                        bias=self.BIAS_ARB,
                        reason="waiting_arbitrage_signal_confirmation",
                        confidence=opportunity.confidence,
                        metadata=self._build_common_metadata(opportunity),
                        now=opportunity.timestamp,
                    )
                    return

                if not self._is_tradeable(opportunity):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="opportunity_not_tradeable",
                    )
                    return

                state = self._get_or_create_state(
                    key=key,
                    symbol=opportunity.symbol,
                    exchange_a=opportunity.buy_exchange,
                    exchange_b=opportunity.sell_exchange,
                    bias=self.BIAS_ARB,
                    metadata={
                        "spread_type": SpreadType.CROSS_EXCHANGE.value,
                        "buy_instrument_type": opportunity.buy_instrument_type.value,
                        "sell_instrument_type": opportunity.sell_instrument_type.value,
                    },
                )

                if (
                    self._config.require_persistence
                    and not self._has_persistence_confirmation(
                        key=key,
                        now=opportunity.timestamp,
                    )
                ):
                    self._stats["persistence_waits"] += 1
                    self._set_state_pending(
                        state,
                        bias=self.BIAS_ARB,
                        reason="waiting_persistence_confirmation",
                        confidence=opportunity.confidence,
                        metadata=self._build_common_metadata(opportunity),
                        now=opportunity.timestamp,
                    )
                    return

                if self._should_close(opportunity, state):
                    await self._close_active_setup(
                        opportunity=opportunity,
                        state=state,
                        reason="arbitrage_edge_deteriorated",
                    )
                    return

                if self._should_open(opportunity, state):
                    if self._should_skip_by_cooldown(
                        key,
                        now=opportunity.timestamp,
                    ):
                        return

                    self._set_state_open(
                        state,
                        bias=self.BIAS_ARB,
                        reason="open_arbitrage_setup",
                        entry_value=self._extract_edge_bps(opportunity),
                        entry_net_edge=opportunity.net_edge,
                        confidence=opportunity.confidence,
                        metadata=self._build_common_metadata(opportunity),
                        now=opportunity.timestamp,
                    )
                    self._stats["opened_setups"] += 1

                    await self._emit_generated(
                        action=self.ACTION_OPEN,
                        symbol=opportunity.symbol,
                        state_key=key,
                        exchange_a=opportunity.buy_exchange,
                        exchange_b=opportunity.sell_exchange,
                        reason="profitable_active_arbitrage_opportunity",
                        confidence=opportunity.confidence,
                        spread_type=SpreadType.CROSS_EXCHANGE.value,
                        timestamp=opportunity.timestamp,
                        metadata=self._build_open_payload(opportunity, state),
                    )
                    return

                if self._should_update(opportunity, state):
                    self._set_state_open(
                        state,
                        bias=self.BIAS_ARB,
                        reason="update_arbitrage_setup",
                        entry_value=self._extract_edge_bps(opportunity),
                        entry_net_edge=opportunity.net_edge,
                        confidence=opportunity.confidence,
                        metadata=self._build_common_metadata(opportunity),
                        now=opportunity.timestamp,
                    )
                    self._stats["updated_setups"] += 1

                    await self._emit_updated(
                        action=self.ACTION_UPDATE,
                        symbol=opportunity.symbol,
                        state_key=key,
                        exchange_a=opportunity.buy_exchange,
                        exchange_b=opportunity.sell_exchange,
                        reason="arbitrage_setup_updated",
                        confidence=opportunity.confidence,
                        spread_type=SpreadType.CROSS_EXCHANGE.value,
                        timestamp=opportunity.timestamp,
                        metadata=self._build_update_payload(opportunity, state),
                    )
                    return

            except Exception as exc:
                self._mark_exception(
                    "Failed to process arbitrage opportunity",
                    exc,
                    symbol=getattr(opportunity, "symbol", None),
                    buy_exchange=getattr(opportunity, "buy_exchange", None),
                    sell_exchange=getattr(opportunity, "sell_exchange", None),
                )

    async def on_cross_exchange_snapshot(self, snapshot: SpreadSnapshot) -> None:
        if not self.is_running:
            return

        if not isinstance(snapshot, SpreadSnapshot):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid payload for cross-exchange snapshot | strategy=%s payload_type=%s",
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

                if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
                    self._stats["ignored_snapshots"] += 1
                    return

                key = self._build_key_from_snapshot(snapshot)
                if not key:
                    self._stats["invalid_contracts"] += 1
                    self._ignore_snapshot(
                        snapshot,
                        key="",
                        reason="snapshot_missing_arbitrage_metadata",
                    )
                    return

                if self._mark_event_seen(
                    key=f"snapshot|{key}",
                    timestamp=snapshot.timestamp,
                ):
                    return

                self._latest_snapshots[key] = snapshot

                if (
                    self._config.close_on_stale_snapshot
                    and self._reject_stale_snapshot(snapshot.timestamp)
                ):
                    state = self._get_state(key)
                    if state is not None and state.is_active:
                        await self._close_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="snapshot_stale_for_active_setup",
                        )
                    else:
                        self._ignore_snapshot(snapshot, key, reason="snapshot_stale")
                    return

                state = self._get_state(key)
                if state is None or not state.is_active:
                    self._ignore_snapshot(
                        snapshot,
                        key,
                        reason="no_active_state_for_snapshot",
                    )
                    return

                if (
                    self._config.close_on_snapshot_status_not_active
                    and self._snapshot_status_not_active(snapshot)
                ):
                    self._stats["snapshot_status_closes"] += 1
                    await self._close_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="snapshot_opportunity_status_not_active",
                    )
                    return

                if (
                    self._config.close_on_snapshot_edge_loss
                    and self._snapshot_edge_below_exit(snapshot)
                ):
                    self._stats["snapshot_edge_closes"] += 1
                    await self._close_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="snapshot_edge_below_exit_threshold",
                    )
                    return

                if (
                    self._config.update_from_snapshot_metadata
                    and self._should_update_from_snapshot(snapshot, state)
                ):
                    self._set_state_open(
                        state,
                        bias=self.BIAS_ARB,
                        reason="update_arbitrage_setup_from_snapshot",
                        entry_value=self._extract_snapshot_edge_bps(snapshot),
                        entry_net_edge=self._extract_snapshot_net_edge(snapshot),
                        confidence=self._extract_snapshot_confidence(snapshot) or state.confidence,
                        metadata=self._build_snapshot_metadata(snapshot),
                        now=snapshot.timestamp,
                    )
                    self._stats["updated_setups"] += 1
                    self._stats["snapshot_updates"] += 1

                    await self._emit_updated(
                        action=self.ACTION_UPDATE,
                        symbol=snapshot.symbol,
                        state_key=state.key,
                        exchange_a=state.exchange_a,
                        exchange_b=state.exchange_b,
                        reason="arbitrage_setup_updated_from_snapshot",
                        confidence=state.confidence,
                        spread_type=SpreadType.CROSS_EXCHANGE.value,
                        timestamp=snapshot.timestamp,
                        metadata=self._build_snapshot_update_payload(snapshot, state),
                    )

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange snapshot",
                    exc,
                    symbol=getattr(snapshot, "symbol", None),
                    exchange_a=getattr(snapshot, "leg_a_exchange", None),
                    exchange_b=getattr(snapshot, "leg_b_exchange", None),
                )

    async def on_spread_signal(self, signal: SpreadSignal) -> None:
        if not self.is_running:
            return

        if not isinstance(signal, SpreadSignal):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid payload for cross-exchange spread signal | strategy=%s payload_type=%s",
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

                if signal.spread_type != SpreadType.CROSS_EXCHANGE:
                    self._stats["ignored_signals"] += 1
                    return

                if signal.signal_type not in {
                    SpreadSignalType.ARBITRAGE,
                    SpreadSignalType.STALE_DATA,
                    SpreadSignalType.INVALID_DATA,
                }:
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
                        "Cross-exchange spread signal cannot be correlated | strategy=%s symbol=%s exchange_a=%s exchange_b=%s signal_type=%s",
                        self.STRATEGY_NAME,
                        signal.symbol,
                        signal.exchange_a,
                        signal.exchange_b,
                        signal.signal_type.value,
                    )
                    return

                self._store_signal(key, signal)

                if signal.signal_type == SpreadSignalType.ARBITRAGE:
                    self._stats["arbitrage_signal_confirmations"] += 1
                    return

                if (
                    signal.signal_type in {
                        SpreadSignalType.STALE_DATA,
                        SpreadSignalType.INVALID_DATA,
                    }
                    and self._config.close_on_data_quality_signal
                ):
                    self._stats["data_quality_blocks"] += 1
                    await self._handle_data_quality_signal(key, signal)

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange spread signal",
                    exc,
                    symbol=getattr(signal, "symbol", None),
                    exchange_a=getattr(signal, "exchange_a", None),
                    exchange_b=getattr(signal, "exchange_b", None),
                    signal_type=getattr(getattr(signal, "signal_type", None), "value", None),
                )

    # ------------------------------------------------------------------
    # Key / contract helpers
    # ------------------------------------------------------------------

    def _build_key_from_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> str:
        return self._build_state_key(
            self._normalize_symbol(opportunity.symbol),
            self._normalize_exchange(opportunity.buy_exchange),
            self._normalize_exchange(opportunity.sell_exchange),
            opportunity.buy_instrument_type.value,
        )

    def _build_key_from_snapshot(self, snapshot: SpreadSnapshot) -> str:
        """
        Snapshot для cross-exchange arb strategy має корелюватися через
        buy/sell direction із metadata, а не через leg_a/leg_b fallback.

        Analytics CrossExchangeSpreadAnalyzer уже додає:
        - buy_exchange;
        - sell_exchange;
        - instrument_type.
        """
        buy_exchange = snapshot.metadata.get("buy_exchange")
        sell_exchange = snapshot.metadata.get("sell_exchange")
        instrument_type = snapshot.metadata.get("instrument_type")

        if not buy_exchange or not sell_exchange or not instrument_type:
            return ""

        return self._build_state_key(
            self._normalize_symbol(snapshot.symbol),
            self._normalize_exchange(str(buy_exchange)),
            self._normalize_exchange(str(sell_exchange)),
            str(instrument_type).strip().lower(),
        )

    def _build_key_from_signal(self, signal: SpreadSignal) -> str:
        buy_exchange = (
            signal.metadata.get("buy_exchange")
            or signal.exchange_a
        )
        sell_exchange = (
            signal.metadata.get("sell_exchange")
            or signal.exchange_b
        )
        instrument_type = (
            signal.metadata.get("instrument_type")
            or signal.metadata.get("buy_instrument_type")
        )

        if not buy_exchange or not sell_exchange or not instrument_type:
            return ""

        return self._build_state_key(
            self._normalize_symbol(signal.symbol),
            self._normalize_exchange(str(buy_exchange)),
            self._normalize_exchange(str(sell_exchange)),
            str(instrument_type).strip().lower(),
        )

    def _instrument_type_value(self, value: InstrumentType | str) -> str:
        if isinstance(value, InstrumentType):
            return value.value
        return str(value).strip().lower()

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

    def _has_arbitrage_signal_confirmation(self, key: str) -> bool:
        self._prune_stale_signals(key)
        return any(
            signal.signal_type == SpreadSignalType.ARBITRAGE
            for signal in self._latest_signals.get(key, [])
        )

    def _has_data_quality_block(self, key: str) -> bool:
        self._prune_stale_signals(key)
        return any(
            signal.signal_type in {
                SpreadSignalType.STALE_DATA,
                SpreadSignalType.INVALID_DATA,
            }
            for signal in self._latest_signals.get(key, [])
        )

    async def _handle_data_quality_signal(
        self,
        key: str,
        signal: SpreadSignal,
    ) -> None:
        state = self._get_state(key)
        if state is None or not state.is_active:
            return

        opportunity = self._latest_opportunities.get(key)
        if opportunity is not None:
            await self._cancel_if_active(
                opportunity=opportunity,
                key=key,
                reason=f"data_quality_signal_{signal.signal_type.value}",
            )
            return

        self._set_state_closed(
            state,
            status=STATE_CANCELLED,
            reason=f"data_quality_signal_{signal.signal_type.value}",
            metadata={
                "source": "spread_signal",
                "signal_type": signal.signal_type.value,
                "signal_message": signal.message,
                "signal_timestamp": self._safe_isoformat(signal.timestamp),
            },
            now=signal.timestamp,
        )
        self._pending_since.pop(key, None)
        self._stats["cancelled_setups"] += 1

        await self._emit_cancelled(
            action=self.ACTION_CANCEL,
            symbol=signal.symbol,
            state_key=state.key,
            exchange_a=state.exchange_a,
            exchange_b=state.exchange_b,
            reason=f"data_quality_signal_{signal.signal_type.value}",
            confidence=signal.confidence,
            spread_type=SpreadType.CROSS_EXCHANGE.value,
            timestamp=signal.timestamp,
            metadata={
                "source": "spread_signal",
                "signal_type": signal.signal_type.value,
                "signal_message": signal.message,
                "signal_timestamp": self._safe_isoformat(signal.timestamp),
            },
        )

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def _reject_instrument_types(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> bool:
        allowed = self._config.allowed_instrument_types
        if not allowed:
            return False

        buy_type = opportunity.buy_instrument_type.value
        sell_type = opportunity.sell_instrument_type.value

        return buy_type not in allowed or sell_type not in allowed

    def _is_opportunity_active(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> bool:
        return opportunity.status == OpportunityStatus.ACTIVE

    def _is_opportunity_expired(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> bool:
        if opportunity.expires_at is None:
            return False
        return self._utcnow() >= opportunity.expires_at

    def _is_tradeable(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> bool:
        if not opportunity.is_profitable:
            return False

        if opportunity.net_edge <= self._config.entry_min_net_edge:
            return False

        edge_bps = self._extract_edge_bps(opportunity)
        if edge_bps is not None and edge_bps < self._config.entry_min_bps:
            return False

        return True

    def _has_persistence_confirmation(
        self,
        *,
        key: str,
        now: datetime,
    ) -> bool:
        first_seen_at = self._pending_since.get(key)
        if first_seen_at is None:
            self._pending_since[key] = now
            return False

        required_delta = timedelta(milliseconds=self._config.min_persistence_ms)
        if (now - first_seen_at) < required_delta:
            return False

        self._pending_since.pop(key, None)
        return True

    def _should_open(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status in {"open", "pending", "closing"}:
            return False

        return self._is_tradeable(opportunity)

    def _should_update(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status not in {"open", "pending"}:
            return False

        previous_net_edge = state.entry_net_edge
        previous_edge_bps = state.entry_value
        previous_confidence = state.confidence

        net_edge_changed = (
            previous_net_edge is not None
            and abs(opportunity.net_edge - previous_net_edge)
            >= self._config.min_update_net_edge_delta
        )

        current_edge_bps = self._extract_edge_bps(opportunity)
        edge_bps_changed = (
            previous_edge_bps is not None
            and current_edge_bps is not None
            and abs(current_edge_bps - previous_edge_bps)
            >= self._config.min_update_net_edge_bps_delta
        )

        confidence_changed = (
            previous_confidence is not None
            and opportunity.confidence is not None
            and abs(opportunity.confidence - previous_confidence)
            >= self._config.min_update_confidence_delta
        )

        return net_edge_changed or edge_bps_changed or confidence_changed

    def _should_close(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status not in {"open", "pending"}:
            return False

        if not self._is_opportunity_active(opportunity):
            return True

        if self._is_opportunity_expired(opportunity):
            return True

        if opportunity.net_edge <= self._config.exit_min_net_edge:
            return True

        edge_bps = self._extract_edge_bps(opportunity)
        if edge_bps is not None and edge_bps <= self._config.exit_min_bps:
            return True

        if (
            opportunity.confidence is None
            or opportunity.confidence < self._config.min_confidence
        ):
            return True

        return False

    def _snapshot_status_not_active(self, snapshot: SpreadSnapshot) -> bool:
        status = snapshot.metadata.get("opportunity_status")
        if status is None:
            return False
        return str(status).strip().lower() != OpportunityStatus.ACTIVE.value

    def _snapshot_edge_below_exit(self, snapshot: SpreadSnapshot) -> bool:
        net_edge = self._extract_snapshot_net_edge(snapshot)
        if net_edge is None:
            return True

        if net_edge <= self._config.exit_min_net_edge:
            return True

        edge_bps = self._extract_snapshot_edge_bps(snapshot)
        if edge_bps is not None and edge_bps <= self._config.exit_min_bps:
            return True

        return False

    def _should_update_from_snapshot(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status != "open":
            return False

        snapshot_net_edge = self._extract_snapshot_net_edge(snapshot)
        snapshot_edge_bps = self._extract_snapshot_edge_bps(snapshot)
        snapshot_confidence = self._extract_snapshot_confidence(snapshot)

        net_edge_changed = (
            snapshot_net_edge is not None
            and state.entry_net_edge is not None
            and abs(snapshot_net_edge - state.entry_net_edge)
            >= self._config.min_update_net_edge_delta
        )

        edge_bps_changed = (
            snapshot_edge_bps is not None
            and state.entry_value is not None
            and abs(snapshot_edge_bps - state.entry_value)
            >= self._config.min_update_net_edge_bps_delta
        )

        confidence_changed = (
            snapshot_confidence is not None
            and state.confidence is not None
            and abs(snapshot_confidence - state.confidence)
            >= self._config.min_update_confidence_delta
        )

        return net_edge_changed or edge_bps_changed or confidence_changed

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_edge_bps(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> Decimal | None:
        metadata_value = opportunity.metadata.get("net_edge_bps")
        metadata_bps = _to_decimal(metadata_value)
        if metadata_bps is not None:
            return metadata_bps

        reference_notional = _to_decimal(opportunity.metadata.get("reference_buy_notional"))
        if reference_notional is not None and reference_notional > DECIMAL_ZERO:
            return (opportunity.net_edge / reference_notional) * DECIMAL_10_000

        return opportunity.spread_bps

    def _extract_snapshot_net_edge(self, snapshot: SpreadSnapshot) -> Decimal | None:
        metadata_net_edge = _to_decimal(snapshot.metadata.get("opportunity_net_edge"))
        if metadata_net_edge is not None:
            return metadata_net_edge

        return snapshot.net_spread

    def _extract_snapshot_edge_bps(self, snapshot: SpreadSnapshot) -> Decimal | None:
        metadata_edge_bps = _to_decimal(snapshot.metadata.get("opportunity_net_edge_bps"))
        if metadata_edge_bps is not None:
            return metadata_edge_bps

        return snapshot.spread_bps

    def _extract_snapshot_confidence(self, snapshot: SpreadSnapshot) -> Decimal | None:
        return _to_decimal(snapshot.metadata.get("opportunity_confidence"))

    # ------------------------------------------------------------------
    # State transitions / emissions
    # ------------------------------------------------------------------

    def _ignore_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        key: str,
        *,
        reason: str,
    ) -> None:
        self._stats["ignored_opportunities"] += 1
        self._logger.debug(
            "Cross-exchange opportunity ignored | strategy=%s symbol=%s key=%s reason=%s",
            self.STRATEGY_NAME,
            opportunity.symbol,
            key,
            reason,
            extra={
                "strategy": self.STRATEGY_NAME,
                "symbol": opportunity.symbol,
                "state_key": key,
                "reason": reason,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "net_edge": self._to_decimal_str(opportunity.net_edge),
                "edge_bps": self._to_decimal_str(self._extract_edge_bps(opportunity)),
                "status": opportunity.status.value,
            },
        )

    def _ignore_snapshot(
        self,
        snapshot: SpreadSnapshot,
        key: str,
        *,
        reason: str,
    ) -> None:
        self._stats["ignored_snapshots"] += 1
        self._logger.debug(
            "Cross-exchange snapshot ignored | strategy=%s symbol=%s key=%s reason=%s",
            self.STRATEGY_NAME,
            snapshot.symbol,
            key,
            reason,
            extra={
                "strategy": self.STRATEGY_NAME,
                "symbol": snapshot.symbol,
                "state_key": key,
                "reason": reason,
                "leg_a_exchange": snapshot.leg_a_exchange,
                "leg_b_exchange": snapshot.leg_b_exchange,
                "net_edge": self._to_decimal_str(
                    self._extract_snapshot_net_edge(snapshot)
                ),
                "edge_bps": self._to_decimal_str(
                    self._extract_snapshot_edge_bps(snapshot)
                ),
                "snapshot_metadata": dict(snapshot.metadata),
            },
        )

    async def _reject_setup(
        self,
        *,
        opportunity: ArbitrageOpportunity,
        key: str,
        reason: str,
    ) -> None:
        state = self._get_or_create_state(
            key=key,
            symbol=opportunity.symbol,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            bias=self.BIAS_ARB,
            metadata=self._build_common_metadata(opportunity),
        )

        self._set_state_closed(
            state,
            status=STATE_REJECTED,
            reason=reason,
            metadata=self._build_common_metadata(opportunity),
            now=opportunity.timestamp,
        )
        self._pending_since.pop(key, None)
        self._stats["rejected_setups"] += 1

        await self._emit_rejected(
            action=self.ACTION_REJECT,
            symbol=opportunity.symbol,
            state_key=key,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            reason=reason,
            confidence=opportunity.confidence,
            spread_type=SpreadType.CROSS_EXCHANGE.value,
            timestamp=opportunity.timestamp,
            metadata=self._build_common_metadata(opportunity),
        )

    async def _cancel_if_active(
        self,
        *,
        opportunity: ArbitrageOpportunity,
        key: str,
        reason: str,
    ) -> None:
        state = self._get_state(key)
        if state is None or not state.is_active:
            self._ignore_opportunity(opportunity, key, reason=reason)
            return

        self._set_state_closed(
            state,
            status=STATE_CANCELLED,
            reason=reason,
            metadata=self._build_common_metadata(opportunity),
            now=opportunity.timestamp,
        )
        self._pending_since.pop(key, None)
        self._stats["cancelled_setups"] += 1

        await self._emit_cancelled(
            action=self.ACTION_CANCEL,
            symbol=opportunity.symbol,
            state_key=key,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            reason=reason,
            confidence=opportunity.confidence,
            spread_type=SpreadType.CROSS_EXCHANGE.value,
            timestamp=opportunity.timestamp,
            metadata=self._build_close_payload(opportunity, state, reason),
        )

    async def _close_active_setup(
        self,
        *,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
        reason: str,
    ) -> None:
        self._set_state_closed(
            state,
            status=STATE_CLOSED,
            reason=reason,
            metadata=self._build_common_metadata(opportunity),
            now=opportunity.timestamp,
        )
        self._pending_since.pop(state.key, None)
        self._stats["closed_setups"] += 1

        await self._emit_closed(
            action=self.ACTION_CLOSE,
            symbol=opportunity.symbol,
            state_key=state.key,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            reason=reason,
            confidence=opportunity.confidence,
            spread_type=SpreadType.CROSS_EXCHANGE.value,
            timestamp=opportunity.timestamp,
            metadata=self._build_close_payload(opportunity, state, reason),
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
            status=STATE_CLOSED,
            reason=reason,
            metadata=self._build_snapshot_metadata(snapshot),
            now=snapshot.timestamp,
        )
        self._pending_since.pop(state.key, None)
        self._stats["closed_setups"] += 1

        await self._emit_closed(
            action=self.ACTION_CLOSE,
            symbol=snapshot.symbol,
            state_key=state.key,
            exchange_a=state.exchange_a,
            exchange_b=state.exchange_b,
            reason=reason,
            confidence=state.confidence,
            spread_type=SpreadType.CROSS_EXCHANGE.value,
            timestamp=snapshot.timestamp,
            metadata=self._build_snapshot_close_payload(snapshot, state, reason),
        )

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _build_common_metadata(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> dict[str, Any]:
        return {
            "source": "arbitrage_opportunity",
            "buy_exchange": opportunity.buy_exchange,
            "sell_exchange": opportunity.sell_exchange,
            "buy_instrument_type": opportunity.buy_instrument_type.value,
            "sell_instrument_type": opportunity.sell_instrument_type.value,
            "buy_price": self._to_decimal_str(opportunity.buy_price),
            "sell_price": self._to_decimal_str(opportunity.sell_price),
            "gross_edge": self._to_decimal_str(opportunity.gross_edge),
            "estimated_fees": self._to_decimal_str(opportunity.estimated_fees),
            "estimated_slippage": self._to_decimal_str(opportunity.estimated_slippage),
            "net_edge": self._to_decimal_str(opportunity.net_edge),
            "net_edge_bps": self._to_decimal_str(self._extract_edge_bps(opportunity)),
            "spread_pct": self._to_decimal_str(opportunity.spread_pct),
            "spread_bps": self._to_decimal_str(opportunity.spread_bps),
            "confidence": self._to_decimal_str(opportunity.confidence),
            "opportunity_status": opportunity.status.value,
            "opportunity_timestamp": self._safe_isoformat(opportunity.timestamp),
            "opportunity_expires_at": self._safe_isoformat(opportunity.expires_at),
            "opportunity_metadata": dict(opportunity.metadata),
        }

    def _build_snapshot_metadata(self, snapshot: SpreadSnapshot) -> dict[str, Any]:
        return {
            "source": "cross_exchange_snapshot",
            "symbol": snapshot.symbol,
            "leg_a_exchange": snapshot.leg_a_exchange,
            "leg_b_exchange": snapshot.leg_b_exchange,
            "leg_a_type": snapshot.leg_a_type.value,
            "leg_b_type": snapshot.leg_b_type.value,
            "buy_exchange": snapshot.metadata.get("buy_exchange"),
            "sell_exchange": snapshot.metadata.get("sell_exchange"),
            "instrument_type": snapshot.metadata.get("instrument_type"),
            "raw_spread": self._to_decimal_str(snapshot.raw_spread),
            "spread_pct": self._to_decimal_str(snapshot.spread_pct),
            "spread_bps": self._to_decimal_str(snapshot.spread_bps),
            "net_spread": self._to_decimal_str(snapshot.net_spread),
            "opportunity_net_edge": self._to_decimal_str(
                self._extract_snapshot_net_edge(snapshot)
            ),
            "opportunity_net_edge_bps": self._to_decimal_str(
                self._extract_snapshot_edge_bps(snapshot)
            ),
            "estimated_fees": self._to_decimal_str(snapshot.estimated_fees),
            "estimated_slippage": self._to_decimal_str(snapshot.estimated_slippage),
            "quote_validity": snapshot.quote_validity.value,
            "direction": snapshot.direction.value,
            "regime": snapshot.regime.value,
            "snapshot_timestamp": self._safe_isoformat(snapshot.timestamp),
            "snapshot_metadata": dict(snapshot.metadata),
        }

    def _build_open_payload(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> dict[str, Any]:
        return {
            **self._build_common_metadata(opportunity),
            "state_status": state.status,
            "state_bias": state.bias,
            "entry_net_edge": self._to_decimal_str(state.entry_net_edge),
            "entry_edge_bps": self._to_decimal_str(state.entry_value),
            "state_opened_at": self._safe_isoformat(state.opened_at),
        }

    def _build_update_payload(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> dict[str, Any]:
        return {
            **self._build_common_metadata(opportunity),
            "state_status": state.status,
            "state_bias": state.bias,
            "previous_entry_net_edge": self._to_decimal_str(state.entry_net_edge),
            "current_net_edge": self._to_decimal_str(opportunity.net_edge),
            "previous_entry_edge_bps": self._to_decimal_str(state.entry_value),
            "current_edge_bps": self._to_decimal_str(
                self._extract_edge_bps(opportunity)
            ),
            "state_updated_at": self._safe_isoformat(state.updated_at),
        }

    def _build_close_payload(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
        reason: str,
    ) -> dict[str, Any]:
        return {
            **self._build_common_metadata(opportunity),
            "state_status": state.status,
            "state_bias": state.bias,
            "state_opened_at": self._safe_isoformat(state.opened_at),
            "state_closed_at": self._safe_isoformat(state.closed_at),
            "close_reason": reason,
        }

    def _build_snapshot_update_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot),
            "state_status": state.status,
            "state_bias": state.bias,
            "state_updated_at": self._safe_isoformat(state.updated_at),
            "state_confidence": self._to_decimal_str(state.confidence),
            "state_entry_net_edge": self._to_decimal_str(state.entry_net_edge),
            "state_entry_edge_bps": self._to_decimal_str(state.entry_value),
        }

    def _build_snapshot_close_payload(
        self,
        snapshot: SpreadSnapshot,
        state: SpreadStrategyState,
        reason: str,
    ) -> dict[str, Any]:
        return {
            **self._build_snapshot_metadata(snapshot),
            "state_status_before_close": state.status,
            "state_bias": state.bias,
            "state_opened_at": self._safe_isoformat(state.opened_at),
            "state_updated_at": self._safe_isoformat(state.updated_at),
            "state_closed_at": self._safe_isoformat(state.closed_at),
            "close_reason": reason,
        }