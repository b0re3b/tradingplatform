from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from .base_spread_strategy import (
    BaseSpreadStrategy,
    BaseSpreadStrategyConfig,
    SpreadStrategyState,
)
from analytics.spreads.enums import OpportunityStatus, SpreadType
from analytics.spreads.models import ArbitrageOpportunity, SpreadSnapshot


@dataclass(slots=True)
class CrossExchangeArbStrategyConfig(BaseSpreadStrategyConfig):
    """
    Конфігурація стратегії cross-exchange arbitrage.

    Це не analytics config.
    Це rules-layer поверх готових arbitrage opportunities.
    """

    entry_min_net_edge: Decimal = Decimal("0")
    entry_min_bps: Decimal = Decimal("5")
    exit_min_net_edge: Decimal = Decimal("0")
    min_update_net_edge_delta: Decimal = Decimal("0.5")
    min_update_confidence_delta: Decimal = Decimal("0.05")

    require_persistence: bool = True
    min_persistence_ms: int = 500

    close_on_snapshot_edge_loss: bool = True
    close_on_stale_snapshot: bool = True

    allowed_instrument_types: set[str] = field(default_factory=set)


class CrossExchangeArbStrategy(BaseSpreadStrategy):
    """
    Strategy layer для cross-exchange arbitrage.

    Роль:
    - слухає arbitrage opportunities і cross-exchange snapshots
    - фільтрує executable setups
    - веде lifecycle strategy-state:
        idle -> pending -> open -> updated -> closed/cancelled/rejected
    - публікує strategy-level intents:
        signal.generated / updated / cancelled / closed / rejected

    Не відповідає за:
    - пошук arbitrage можливостей
    - розрахунок costs
    - побудову spread snapshot
    - execution ордерів
    """

    STRATEGY_NAME = "cross_exchange_arb"

    OPPORTUNITY_EVENT = "spread.arbitrage.opportunity"
    SNAPSHOT_EVENT = "spread.cross_exchange.updated"

    ACTION_OPEN = "OPEN_ARB"
    ACTION_UPDATE = "UPDATE_ARB"
    ACTION_CANCEL = "CANCEL_ARB"
    ACTION_CLOSE = "CLOSE_ARB"
    ACTION_REJECT = "REJECT_ARB"

    def __init__(
        self,
        *,
        event_bus: Any,
        config: CrossExchangeArbStrategyConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        resolved_config = config or CrossExchangeArbStrategyConfig()
        super().__init__(
            event_bus=event_bus,
            config=resolved_config,
            scheduler=scheduler,
            service_name=self.STRATEGY_NAME,
        )
        self._config = resolved_config

        self._latest_opportunities: dict[str, ArbitrageOpportunity] = {}
        self._latest_snapshots: dict[str, SpreadSnapshot] = {}
        self._pending_since: dict[str, datetime] = {}

        self._stats.update(
            {
                "opportunities_received": 0,
                "snapshots_received": 0,
                "opened_setups": 0,
                "updated_setups": 0,
                "cancelled_setups": 0,
                "closed_setups": 0,
                "rejected_setups": 0,
                "persistence_waits": 0,
                "expired_opportunities": 0,
                "inactive_opportunities": 0,
            }
        )

    async def _subscribe_events(self) -> None:
        await self._subscribe(self.OPPORTUNITY_EVENT, self.on_arbitrage_opportunity)
        await self._subscribe(self.SNAPSHOT_EVENT, self.on_cross_exchange_snapshot)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self.get_base_stats(),
            "opportunities_received": self._stats["opportunities_received"],
            "snapshots_received": self._stats["snapshots_received"],
            "opened_setups": self._stats["opened_setups"],
            "updated_setups": self._stats["updated_setups"],
            "cancelled_setups": self._stats["cancelled_setups"],
            "closed_setups": self._stats["closed_setups"],
            "rejected_setups": self._stats["rejected_setups"],
            "persistence_waits": self._stats["persistence_waits"],
            "expired_opportunities": self._stats["expired_opportunities"],
            "inactive_opportunities": self._stats["inactive_opportunities"],
            "tracked_opportunities": len(self._latest_opportunities),
            "tracked_snapshots": len(self._latest_snapshots),
            "pending_candidates": len(self._pending_since),
        }

    async def on_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        if not self.is_running:
            return

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["opportunities_received"] += 1

                if self._reject_disabled():
                    return

                key = self._build_key_from_opportunity(opportunity)

                if self._mark_event_seen(key=f"opportunity|{key}", timestamp=opportunity.timestamp):
                    return

                self._latest_opportunities[key] = opportunity

                if self._reject_symbol(opportunity.symbol):
                    await self._reject_setup(
                        opportunity=opportunity,
                        key=key,
                        reason="symbol_not_allowed",
                    )
                    return

                if self._reject_exchanges(opportunity.buy_exchange, opportunity.sell_exchange):
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
                    bias="arb",
                    metadata={
                        "buy_instrument_type": opportunity.buy_instrument_type.value,
                        "sell_instrument_type": opportunity.sell_instrument_type.value,
                    },
                )

                if self._config.require_persistence and not self._has_persistence_confirmation(
                    key=key,
                    now=opportunity.timestamp,
                ):
                    self._stats["persistence_waits"] += 1
                    self._set_state_pending(
                        state,
                        bias="arb",
                        reason="waiting_persistence_confirmation",
                        confidence=opportunity.confidence,
                        metadata=self._build_common_metadata(opportunity),
                        now=opportunity.timestamp,
                    )
                    return

                if self._should_open(opportunity, state):
                    if self._should_skip_by_cooldown(key, now=opportunity.timestamp):
                        return

                    self._set_state_open(
                        state,
                        bias="arb",
                        reason="open_arbitrage_setup",
                        entry_value=opportunity.spread_bps,
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
                        bias="arb",
                        reason="update_arbitrage_setup",
                        entry_value=state.entry_value,
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

                if self._should_close(opportunity, state):
                    await self._close_active_setup(
                        opportunity=opportunity,
                        state=state,
                        reason="arbitrage_edge_deteriorated",
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

        async with self._lock:
            try:
                self._record_event_received()
                self._stats["snapshots_received"] += 1

                if self._reject_disabled():
                    return

                if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
                    return

                key = self._build_key_from_snapshot(snapshot)

                if self._mark_event_seen(key=f"snapshot|{key}", timestamp=snapshot.timestamp):
                    return

                self._latest_snapshots[key] = snapshot

                state = self._get_state(key)
                if state is None or not state.is_active:
                    return

                if self._config.close_on_stale_snapshot and self._reject_stale_snapshot(snapshot.timestamp):
                    await self._close_from_snapshot(
                        snapshot=snapshot,
                        state=state,
                        reason="snapshot_stale_for_active_setup",
                    )
                    return

                if self._config.close_on_snapshot_edge_loss:
                    net_spread = snapshot.net_spread
                    if net_spread is None or net_spread <= self._config.exit_min_net_edge:
                        await self._close_from_snapshot(
                            snapshot=snapshot,
                            state=state,
                            reason="snapshot_net_edge_below_exit_threshold",
                        )
                        return

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange snapshot",
                    exc,
                    symbol=getattr(snapshot, "symbol", None),
                    exchange_a=getattr(snapshot, "leg_a_exchange", None),
                    exchange_b=getattr(snapshot, "leg_b_exchange", None),
                )

    def _build_key_from_opportunity(self, opportunity: ArbitrageOpportunity) -> str:
        return self._build_state_key(
            self._normalize_symbol(opportunity.symbol),
            self._normalize_exchange(opportunity.buy_exchange),
            self._normalize_exchange(opportunity.sell_exchange),
            opportunity.buy_instrument_type.value,
        )

    def _build_key_from_snapshot(self, snapshot: SpreadSnapshot) -> str:
        buy_exchange = snapshot.metadata.get("buy_exchange") or snapshot.leg_a_exchange
        sell_exchange = snapshot.metadata.get("sell_exchange") or snapshot.leg_b_exchange

        instrument_type = snapshot.metadata.get("instrument_type")
        if instrument_type is None:
            instrument_type = snapshot.leg_a_type.value

        return self._build_state_key(
            self._normalize_symbol(snapshot.symbol),
            self._normalize_exchange(str(buy_exchange)),
            self._normalize_exchange(str(sell_exchange)),
            str(instrument_type),
        )

    def _reject_instrument_types(self, opportunity: ArbitrageOpportunity) -> bool:
        allowed = self._config.allowed_instrument_types
        if not allowed:
            return False

        buy_type = opportunity.buy_instrument_type.value
        sell_type = opportunity.sell_instrument_type.value
        return buy_type not in allowed or sell_type not in allowed

    def _is_opportunity_active(self, opportunity: ArbitrageOpportunity) -> bool:
        return opportunity.status == OpportunityStatus.ACTIVE

    def _is_opportunity_expired(self, opportunity: ArbitrageOpportunity) -> bool:
        if opportunity.expires_at is None:
            return False
        return self._utcnow() >= opportunity.expires_at

    def _is_tradeable(self, opportunity: ArbitrageOpportunity) -> bool:
        if not opportunity.is_profitable:
            return False

        if opportunity.net_edge <= self._config.entry_min_net_edge:
            return False

        if opportunity.spread_bps is not None and opportunity.spread_bps < self._config.entry_min_bps:
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

        if not self._is_tradeable(opportunity):
            return False

        return True

    def _should_update(
        self,
        opportunity: ArbitrageOpportunity,
        state: SpreadStrategyState,
    ) -> bool:
        if state.status not in {"open", "pending"}:
            return False

        previous_net_edge = state.entry_net_edge
        previous_confidence = state.confidence

        net_edge_changed = False
        confidence_changed = False

        if previous_net_edge is not None:
            net_edge_changed = (
                abs(opportunity.net_edge - previous_net_edge)
                >= self._config.min_update_net_edge_delta
            )

        if (
            previous_confidence is not None
            and opportunity.confidence is not None
        ):
            confidence_changed = (
                abs(opportunity.confidence - previous_confidence)
                >= self._config.min_update_confidence_delta
            )

        return net_edge_changed or confidence_changed

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

        if opportunity.confidence is None or opportunity.confidence < self._config.min_confidence:
            return True

        return False

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
            bias="arb",
        )
        self._set_state_closed(
            state,
            status="rejected",
            reason=reason,
            metadata=self._build_common_metadata(opportunity),
            now=opportunity.timestamp,
        )
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
            return

        self._set_state_closed(
            state,
            status="cancelled",
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
            status="closed",
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
            status="closed",
            reason=reason,
            metadata={
                "snapshot_spread_bps": self._to_decimal_str(snapshot.spread_bps),
                "snapshot_net_spread": self._to_decimal_str(snapshot.net_spread),
                "snapshot_regime": snapshot.regime.value,
            },
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
            metadata={
                "source": "snapshot",
                "snapshot_spread_bps": self._to_decimal_str(snapshot.spread_bps),
                "snapshot_net_spread": self._to_decimal_str(snapshot.net_spread),
                "snapshot_regime": snapshot.regime.value,
                "state_status_before_close": "open",
            },
        )

    def _build_common_metadata(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> dict[str, Any]:
        return {
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
            "spread_pct": self._to_decimal_str(opportunity.spread_pct),
            "spread_bps": self._to_decimal_str(opportunity.spread_bps),
            "opportunity_status": opportunity.status.value,
            "opportunity_timestamp": self._safe_isoformat(opportunity.timestamp),
            "opportunity_expires_at": self._safe_isoformat(opportunity.expires_at),
            "opportunity_metadata": dict(opportunity.metadata),
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
            "entry_spread_bps": self._to_decimal_str(state.entry_value),
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