from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseSpreadAnalyzer
from .config import CrossExchangeSpreadConfig
from .enums import InstrumentType, PricingSource, QuoteValidity, SpreadType
from .models import ArbitrageOpportunity, QuoteSnapshot, SpreadSnapshot
from .spread_opportunity_detector import (
    OpportunityDetectionResult,
    SpreadOpportunityDetector,
)
from .spread_utils import (
    DECIMAL_ZERO,
    RollingDecimalWindow,
    aligned_quotes,
    infer_direction,
    normalize_exchange,
    normalize_symbol,
    quote_age_ms,
    spread_abs,
    spread_bps,
    spread_pct,
    validate_quote_snapshot,
)


class CrossExchangeSpreadAnalyzer(BaseSpreadAnalyzer):
    """
    Production-grade analyzer для cross-exchange spread analytics.

    Відповідальність:
    - слухати market.quote.updated;
    - кешувати quotes з різних бірж;
    - порівнювати один symbol/instrument_type між біржами;
    - будувати SpreadSnapshot;
    - оновлювати rolling stats;
    - визначати spread regime;
    - шукати ArbitrageOpportunity через SpreadOpportunityDetector;
    - генерувати SpreadSignal через SpreadSignalEngine;
    - публікувати analytics.spreads.cross_exchange.updated;
    - публікувати analytics.spreads.signal.generated;
    - публікувати analytics.spreads.arbitrage.opportunity;
    - запускати cleanup/heartbeat через core.Scheduler.

    Не відповідає за:
    - отримання raw market-data з бірж;
    - execution;
    - risk approval;
    - strategy decisions;
    - storage напряму;
    - прямі виклики strategy/risk/execution.
    """

    DEFAULT_QUOTE_TOPIC = "market.quote.updated"
    DEFAULT_SNAPSHOT_TOPIC = "analytics.spreads.cross_exchange.updated"
    DEFAULT_SIGNAL_TOPIC = "analytics.spreads.signal.generated"
    DEFAULT_OPPORTUNITY_TOPIC = "analytics.spreads.arbitrage.opportunity"

    def __init__(
        self,
        *,
        config: CrossExchangeSpreadConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            service_name=config.service_name,
        )
        self._config: CrossExchangeSpreadConfig = config
        self._opportunity_detector = SpreadOpportunityDetector(config)

        self._quotes: dict[tuple[str, str, InstrumentType], QuoteSnapshot] = {}
        self._spread_windows: dict[
            tuple[str, str, str, InstrumentType],
            RollingDecimalWindow,
        ] = {}
        self._latest_snapshots: dict[
            tuple[str, str, str, InstrumentType],
            SpreadSnapshot,
        ] = {}
        self._latest_opportunities: dict[
            tuple[str, str, str, InstrumentType],
            ArbitrageOpportunity,
        ] = {}

        self._stats.update(
            {
                "quote_events_received": 0,
                "quotes_received": 0,
                "invalid_payloads": 0,
                "invalid_quotes": 0,
                "incomplete_quotes": 0,
                "stale_quotes": 0,
                "unaligned_quotes": 0,
                "instrument_type_skips": 0,
                "preferred_exchange_skips": 0,
                "quotes_stored": 0,
                "snapshots_built": 0,
                "snapshots_skipped": 0,
                "signals_built": 0,
                "opportunities_detected": 0,
                "opportunities_published": 0,
                "opportunities_expired": 0,
                "opportunity_detection_misses": 0,
                "cleanup_runs": 0,
                "cleanup_removed_quotes": 0,
                "cleanup_removed_snapshots": 0,
                "cleanup_removed_windows": 0,
                "cleanup_removed_opportunities": 0,
            }
        )

    # ------------------------------------------------------------------
    # Registration / lifecycle hooks
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscriptions.

        core.EventBus.subscribe() синхронний, тому register() теж синхронний.
        Handler-и приймають core.event_bus.Event.
        """
        if self._registered:
            return

        self._subscribe(
            self._topic("quote_event_topic", self.DEFAULT_QUOTE_TOPIC),
            self.on_quote_update,
            name=f"{self._service_name}.on_quote_update",
        )

        self._registered = True

    def _register_scheduler_jobs(self) -> None:
        """
        Реєструє periodic maintenance jobs через core.Scheduler.

        Жодних власних нескоординованих while-loop.
        """
        self._add_interval_job(
            name=f"{self._service_name}.cleanup",
            func=self.cleanup_stale_state,
            interval=self._config.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=10.0,
            allow_overlap=False,
            enabled=self._config.enabled,
        )

        self._add_interval_job(
            name=f"{self._service_name}.heartbeat",
            func=self.emit_heartbeat,
            interval=self._config.heartbeat_interval_seconds,
            run_immediately=False,
            max_retries=0,
            timeout=5.0,
            allow_overlap=False,
            enabled=self._config.enabled,
        )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_latest_snapshot(
        self,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: InstrumentType,
    ) -> SpreadSnapshot | None:
        normalized_a = normalize_exchange(exchange_a)
        normalized_b = normalize_exchange(exchange_b)

        key = self._snapshot_key_from_values(
            symbol=normalize_symbol(symbol),
            exchange_a=min(normalized_a, normalized_b),
            exchange_b=max(normalized_a, normalized_b),
            instrument_type=instrument_type,
        )
        return self._latest_snapshots.get(key)

    def get_best_opportunities(
        self,
        symbol: str | None = None,
        instrument_type: InstrumentType | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[ArbitrageOpportunity]:
        opportunities = list(self._latest_opportunities.values())

        if symbol is not None:
            normalized_symbol = normalize_symbol(symbol)
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.symbol == normalized_symbol
            ]

        if instrument_type is not None:
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.buy_instrument_type == instrument_type
                and opportunity.sell_instrument_type == instrument_type
            ]

        if profitable_only:
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.is_profitable
            ]

        if active_only:
            active_opportunities: list[ArbitrageOpportunity] = []
            for opportunity in opportunities:
                self._opportunity_detector.expire_opportunity(opportunity)
                if self._opportunity_detector.is_opportunity_active(opportunity):
                    active_opportunities.append(opportunity)
            opportunities = active_opportunities

        opportunities.sort(key=lambda item: item.net_edge, reverse=True)

        if limit is not None and limit > 0:
            return opportunities[:limit]

        return opportunities

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "running": self.is_running,
            "registered": self.is_registered,
            "enabled": self._config.enabled,
            "quotes_cached": len(self._quotes),
            "active_windows": len(self._spread_windows),
            "latest_snapshots": len(self._latest_snapshots),
            "latest_opportunities": len(self._latest_opportunities),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_quote_update(self, event: Event) -> None:
        """
        Handler для market.quote.updated.

        core.EventBus передає Event, тому payload дістаємо через event.payload.
        """
        if not self.is_running or not self._config.enabled:
            return

        self._stats["quote_events_received"] += 1

        payload = event.payload
        if not isinstance(payload, QuoteSnapshot):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid quote event payload | expected=%s actual=%s topic=%s",
                "QuoteSnapshot",
                payload.__class__.__name__ if payload is not None else "None",
                event.topic,
            )
            return

        async with self._lock:
            try:
                self._stats["quotes_received"] += 1

                normalized_quote = self._normalize_quote(payload)

                if not self._is_instrument_type_allowed(normalized_quote.instrument_type):
                    self._stats["instrument_type_skips"] += 1
                    return

                validity = validate_quote_snapshot(
                    normalized_quote,
                    max_age_ms=self._config.max_quote_age_ms,
                )

                if validity == QuoteValidity.INVALID:
                    self._stats["invalid_quotes"] += 1
                    return

                if validity == QuoteValidity.STALE:
                    self._stats["stale_quotes"] += 1
                    return

                if validity == QuoteValidity.INCOMPLETE:
                    self._stats["incomplete_quotes"] += 1
                    return

                self._store_quote(normalized_quote)

                await self._recalculate_for_quote(
                    normalized_quote,
                    correlation_id=event.correlation_id,
                    source_event_id=event.event_id,
                )

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange quote update",
                    exc,
                    exchange=getattr(payload, "exchange", None),
                    symbol=getattr(payload, "symbol", None),
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                )

    # ------------------------------------------------------------------
    # Scheduler jobs
    # ------------------------------------------------------------------
    def _cleanup_orphan_windows(self) -> int:
        """
        Видаляє rolling windows, для яких більше немає актуального latest snapshot.

        RollingDecimalWindow не має власного timestamp, тому TTL-cleanup
        застосовується опосередковано через _latest_snapshots.
        Якщо snapshot видалений як stale, відповідне rolling window теж
        не повинно залишатися активним.
        """
        if not self._spread_windows:
            return 0

        active_snapshot_keys = set(self._latest_snapshots.keys())

        removed = 0
        for key in list(self._spread_windows.keys()):
            if key not in active_snapshot_keys:
                self._spread_windows.pop(key, None)
                removed += 1

        return removed

    async def cleanup_stale_state(self) -> None:
        """
        Periodic cleanup job.

        Запускається тільки через core.Scheduler.
        """
        if not self._config.enabled:
            return

        async with self._lock:
            now = datetime.utcnow()
            ttl = timedelta(seconds=self._config.stale_state_ttl_seconds)

            removed_quotes = self._cleanup_quote_cache(now=now, ttl=ttl)
            removed_snapshots = self._cleanup_snapshot_cache(now=now, ttl=ttl)
            removed_opportunities = self._cleanup_opportunity_cache(now=now)

            removed_orphan_windows = self._cleanup_orphan_windows()
            removed_limit_windows = self._enforce_window_cache_limit()
            removed_windows = removed_orphan_windows + removed_limit_windows

            self._stats["cleanup_runs"] += 1
            self._stats["cleanup_removed_quotes"] += removed_quotes
            self._stats["cleanup_removed_snapshots"] += removed_snapshots
            self._stats["cleanup_removed_opportunities"] += removed_opportunities
            self._stats["cleanup_removed_windows"] += removed_windows

            if removed_quotes or removed_snapshots or removed_opportunities or removed_windows:
                self._logger.debug(
                    "Cross-exchange cleanup completed | quotes=%s snapshots=%s "
                    "opportunities=%s windows=%s orphan_windows=%s limit_windows=%s",
                    removed_quotes,
                    removed_snapshots,
                    removed_opportunities,
                    removed_windows,
                    removed_orphan_windows,
                    removed_limit_windows,
                )

    async def emit_heartbeat(self) -> None:
        """
        Periodic heartbeat для monitoring/dashboard.
        """
        if not self.is_running or not self._config.enabled:
            return

        await self._emit(
            self._topic(
                "analyzer_heartbeat_event_topic",
                "analytics.spreads.analyzer.heartbeat",
            ),
            {
                "analyzer": self.__class__.__name__,
                "service_name": self._service_name,
                "stats": self.get_stats(),
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Normalization / state
    # ------------------------------------------------------------------

    def _normalize_quote(self, quote: QuoteSnapshot) -> QuoteSnapshot:
        return QuoteSnapshot(
            exchange=normalize_exchange(quote.exchange),
            symbol=normalize_symbol(quote.symbol),
            instrument_type=quote.instrument_type,
            bid=quote.bid,
            ask=quote.ask,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            last_price=quote.last_price,
            mark_price=quote.mark_price,
            index_price=quote.index_price,
            timestamp=quote.timestamp,
            received_at=quote.received_at,
            sequence_id=quote.sequence_id,
            metadata=dict(quote.metadata),
        )

    def _store_quote(self, quote: QuoteSnapshot) -> None:
        key = self._quote_key(quote)
        self._quotes[key] = quote
        self._stats["quotes_stored"] += 1
        self._enforce_quote_cache_limit()

    # ------------------------------------------------------------------
    # Recalculation
    # ------------------------------------------------------------------

    async def _recalculate_for_quote(
        self,
        quote: QuoteSnapshot,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> None:
        candidates = [
            other
            for (exchange, symbol, instrument_type), other in self._quotes.items()
            if symbol == quote.symbol
            and instrument_type == quote.instrument_type
            and exchange != quote.exchange
        ]

        for other_quote in candidates:
            if not self._is_exchange_pair_allowed(quote.exchange, other_quote.exchange):
                self._stats["preferred_exchange_skips"] += 1
                continue

            await self._try_build_and_publish(
                quote,
                other_quote,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
            )

    async def _try_build_and_publish(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> None:
        if quote_a.symbol != quote_b.symbol:
            return

        if quote_a.instrument_type != quote_b.instrument_type:
            return

        validity_a = validate_quote_snapshot(
            quote_a,
            max_age_ms=self._config.max_quote_age_ms,
        )
        validity_b = validate_quote_snapshot(
            quote_b,
            max_age_ms=self._config.max_quote_age_ms,
        )

        if validity_a == QuoteValidity.STALE or validity_b == QuoteValidity.STALE:
            self._stats["stale_quotes"] += 1
            return

        if validity_a != QuoteValidity.VALID or validity_b != QuoteValidity.VALID:
            self._stats["invalid_quotes"] += 1
            return

        if not aligned_quotes(
            quote_a,
            quote_b,
            max_age_diff_ms=self._config.max_quote_skew_ms,
        ):
            self._stats["unaligned_quotes"] += 1
            return

        ordered_a, ordered_b = self._order_quotes(quote_a, quote_b)

        snapshot = self._build_snapshot(ordered_a, ordered_b)
        if snapshot is None:
            self._stats["snapshots_skipped"] += 1
            return

        key = self._snapshot_key(snapshot)
        previous_snapshot = self._latest_snapshots.get(key)

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

        opportunity_result = self._detect_opportunity(
            ordered_a,
            ordered_b,
            snapshot,
        )
        opportunity = opportunity_result.opportunity

        self._latest_snapshots[key] = snapshot
        self._stats["calculations_total"] += 1
        self._stats["snapshots_built"] += 1

        await self._publish_snapshot(
            self._topic("snapshot_event_topic", self.DEFAULT_SNAPSHOT_TOPIC),
            snapshot,
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
            headers={
                "source_event_id": source_event_id,
                "spread_type": SpreadType.CROSS_EXCHANGE.value,
            },
        )

        signals = self._evaluate_snapshot_signals(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            opportunity=opportunity,
        )
        self._stats["signals_built"] += len(signals)

        await self._publish_signals(
            self._topic("signal_event_topic", self.DEFAULT_SIGNAL_TOPIC),
            signals,
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers={
                "source_event_id": source_event_id,
                "spread_type": SpreadType.CROSS_EXCHANGE.value,
            },
        )

        if opportunity is not None:
            self._latest_opportunities[key] = opportunity
            await self._publish_opportunity(
                opportunity,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
            )

    # ------------------------------------------------------------------
    # Snapshot building
    # ------------------------------------------------------------------

    def _build_snapshot(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> SpreadSnapshot | None:
        symbol = quote_a.symbol
        instrument_type = quote_a.instrument_type

        mid_a = quote_a.mid_price
        mid_b = quote_b.mid_price

        raw_spread = spread_abs(mid_b, mid_a)
        if raw_spread is None:
            return None

        spread_percent = spread_pct(raw_spread, mid_a)
        spread_bps_value = spread_bps(raw_spread, mid_a)

        window_key = (
            symbol,
            quote_a.exchange,
            quote_b.exchange,
            instrument_type,
        )
        window = self._get_or_create_window(window_key)
        window.append(raw_spread)

        stats = window.stats()
        regime_result = self._regime_detector.detect_from_stats(stats)

        (
            buy_exchange,
            sell_exchange,
            buy_price,
            sell_price,
            gross_edge_per_unit,
        ) = self._best_arbitrage_legs(quote_a, quote_b)

        return SpreadSnapshot(
            spread_type=SpreadType.CROSS_EXCHANGE,
            symbol=symbol,
            leg_a_exchange=quote_a.exchange,
            leg_b_exchange=quote_b.exchange,
            leg_a_type=instrument_type,
            leg_b_type=instrument_type,
            pricing_source=PricingSource.BID_ASK,
            raw_spread=raw_spread,
            spread_pct=spread_percent,
            spread_bps=spread_bps_value,
            net_spread=None,
            basis=None,
            funding_adjusted_spread=None,
            direction=infer_direction(raw_spread),
            regime=regime_result.regime,
            stats=stats,
            leg_a_bid=quote_a.bid,
            leg_a_ask=quote_a.ask,
            leg_b_bid=quote_b.bid,
            leg_b_ask=quote_b.ask,
            leg_a_mid=mid_a,
            leg_b_mid=mid_b,
            estimated_fees=None,
            estimated_slippage=None,
            quote_validity=QuoteValidity.VALID,
            timestamp=max(quote_a.timestamp, quote_b.timestamp),
            metadata={
                "instrument_type": instrument_type.value,
                "quote_a_age_ms": quote_age_ms(quote_a),
                "quote_b_age_ms": quote_age_ms(quote_b),
                "quote_a_sequence_id": quote_a.sequence_id,
                "quote_b_sequence_id": quote_b.sequence_id,
                "regime": regime_result.regime.value,
                "regime_reason": regime_result.reason,
                "is_compressed": regime_result.is_compressed,
                "is_elevated": regime_result.is_elevated,
                "is_extreme": regime_result.is_extreme,
                "is_dislocated": regime_result.is_dislocated,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": str(buy_price) if buy_price is not None else None,
                "sell_price": str(sell_price) if sell_price is not None else None,
                "gross_edge": str(gross_edge_per_unit)
                if gross_edge_per_unit is not None
                else None,
            },
        )

    # ------------------------------------------------------------------
    # Opportunity detection / publishing
    # ------------------------------------------------------------------

    def _detect_opportunity(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        snapshot: SpreadSnapshot,
    ) -> OpportunityDetectionResult:
        result = self._opportunity_detector.detect_from_quotes(
            quote_a=quote_a,
            quote_b=quote_b,
        )

        if not result.found or result.opportunity is None:
            self._stats["opportunity_detection_misses"] += 1
            snapshot.metadata["opportunity_reason"] = result.reason_value
            return result

        opportunity = result.opportunity
        self._stats["opportunities_detected"] += 1

        if result.costs is not None:
            snapshot.estimated_fees = result.costs.estimated_fees
            snapshot.estimated_slippage = result.costs.estimated_slippage
            snapshot.net_spread = result.costs.net_edge
            snapshot.metadata["opportunity_total_costs"] = str(result.costs.total_costs)

        snapshot.metadata["opportunity_reason"] = result.reason_value
        snapshot.metadata["opportunity_net_edge"] = str(opportunity.net_edge)
        snapshot.metadata["opportunity_status"] = opportunity.status.value
        snapshot.metadata["opportunity_confidence"] = (
            str(opportunity.confidence)
            if opportunity.confidence is not None
            else None
        )

        if result.net_edge_bps is not None:
            snapshot.metadata["opportunity_net_edge_bps"] = str(result.net_edge_bps)

        if result.quantity is not None:
            snapshot.metadata["opportunity_quantity"] = str(result.quantity)

        return result

    async def _publish_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> bool:
        accepted = await self._emit(
            self._topic("opportunity_event_topic", self.DEFAULT_OPPORTUNITY_TOPIC),
            opportunity,
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers={
                "source_event_id": source_event_id,
                "spread_type": SpreadType.CROSS_EXCHANGE.value,
                "symbol": opportunity.symbol,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
            },
        )

        if not accepted:
            return False

        self._stats["opportunities_published"] += 1

        self._logger.debug(
            "Arbitrage opportunity published | symbol=%s buy=%s sell=%s net_edge=%s",
            opportunity.symbol,
            opportunity.buy_exchange,
            opportunity.sell_exchange,
            opportunity.net_edge,
            extra={
                "symbol": opportunity.symbol,
                "event_type": self._topic(
                    "opportunity_event_topic",
                    self.DEFAULT_OPPORTUNITY_TOPIC,
                ),
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "gross_edge": str(opportunity.gross_edge),
                "net_edge": str(opportunity.net_edge),
                "spread_bps": str(opportunity.spread_bps)
                if opportunity.spread_bps is not None
                else None,
                "status": opportunity.status.value,
            },
        )
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _order_quotes(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[QuoteSnapshot, QuoteSnapshot]:
        if quote_a.exchange <= quote_b.exchange:
            return quote_a, quote_b
        return quote_b, quote_a

    def _best_arbitrage_legs(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[str | None, str | None, Decimal | None, Decimal | None, Decimal | None]:
        if quote_a.ask is None or quote_a.bid is None:
            return None, None, None, None, None

        if quote_b.ask is None or quote_b.bid is None:
            return None, None, None, None, None

        option_1_edge = quote_b.bid - quote_a.ask
        option_2_edge = quote_a.bid - quote_b.ask

        if option_1_edge >= option_2_edge and option_1_edge > DECIMAL_ZERO:
            return (
                quote_a.exchange,
                quote_b.exchange,
                quote_a.ask,
                quote_b.bid,
                option_1_edge,
            )

        if option_2_edge > option_1_edge and option_2_edge > DECIMAL_ZERO:
            return (
                quote_b.exchange,
                quote_a.exchange,
                quote_b.ask,
                quote_a.bid,
                option_2_edge,
            )

        return None, None, None, None, None

    def _get_or_create_window(
        self,
        key: tuple[str, str, str, InstrumentType],
    ) -> RollingDecimalWindow:
        window = self._spread_windows.get(key)
        if window is not None:
            return window

        if len(self._spread_windows) >= self._config.max_cached_windows:
            self._drop_oldest_window()

        window = RollingDecimalWindow(
            maxlen=self._config.rolling_window_size,
            ema_alpha=self._config.ema_alpha,
        )
        self._spread_windows[key] = window
        return window

    def _is_instrument_type_allowed(self, instrument_type: InstrumentType) -> bool:
        checker = getattr(self._config, "is_instrument_type_allowed", None)
        if callable(checker):
            return bool(checker(instrument_type))

        return instrument_type in self._config.allowed_instrument_types

    def _is_exchange_pair_allowed(self, exchange_a: str, exchange_b: str) -> bool:
        checker = getattr(self._config, "is_exchange_preferred", None)
        if callable(checker):
            return bool(checker(exchange_a)) and bool(checker(exchange_b))

        if not self._config.preferred_exchanges:
            return True

        return (
            normalize_exchange(exchange_a) in self._config.preferred_exchanges
            and normalize_exchange(exchange_b) in self._config.preferred_exchanges
        )

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _cleanup_quote_cache(
        self,
        *,
        now: datetime,
        ttl: timedelta,
    ) -> int:
        stale_keys = [
            key
            for key, quote in self._quotes.items()
            if now - quote.timestamp >= ttl
        ]

        for key in stale_keys:
            self._quotes.pop(key, None)

        return len(stale_keys)

    def _cleanup_snapshot_cache(
        self,
        *,
        now: datetime,
        ttl: timedelta,
    ) -> int:
        stale_keys = [
            key
            for key, snapshot in self._latest_snapshots.items()
            if now - snapshot.timestamp >= ttl
        ]

        for key in stale_keys:
            self._latest_snapshots.pop(key, None)

        return len(stale_keys)

    def _cleanup_opportunities(self, *, now: datetime) -> int:
        removed = 0

        for key, opportunity in list(self._latest_opportunities.items()):
            self._opportunity_detector.expire_opportunity(opportunity, now=now)

            if not self._opportunity_detector.is_opportunity_active(opportunity, now=now):
                self._latest_opportunities.pop(key, None)
                removed += 1

                if opportunity.status.value == "expired":
                    self._stats["opportunities_expired"] += 1

        self._enforce_opportunity_cache_limit()
        return removed

    def _enforce_quote_cache_limit(self) -> None:
        max_items = self._config.max_cached_quotes
        if len(self._quotes) <= max_items:
            return

        sorted_items = sorted(
            self._quotes.items(),
            key=lambda item: item[1].timestamp,
        )

        overflow = len(self._quotes) - max_items
        for key, _ in sorted_items[:overflow]:
            self._quotes.pop(key, None)

    def _enforce_snapshot_cache_limit(self) -> int:
        max_items = self._config.max_cached_snapshots
        if len(self._latest_snapshots) <= max_items:
            return 0

        sorted_items = sorted(
            self._latest_snapshots.items(),
            key=lambda item: item[1].timestamp,
        )

        overflow = len(self._latest_snapshots) - max_items
        for key, _ in sorted_items[:overflow]:
            self._latest_snapshots.pop(key, None)

        return overflow

    def _enforce_window_cache_limit(self) -> int:
        max_items = self._config.max_cached_windows
        if len(self._spread_windows) <= max_items:
            return 0

        overflow = len(self._spread_windows) - max_items
        keys_to_remove = list(self._spread_windows.keys())[:overflow]

        for key in keys_to_remove:
            self._spread_windows.pop(key, None)

        return len(keys_to_remove)

    def _enforce_opportunity_cache_limit(self) -> int:
        max_items = self._config.max_cached_opportunities
        if len(self._latest_opportunities) <= max_items:
            return 0

        sorted_items = sorted(
            self._latest_opportunities.items(),
            key=lambda item: item[1].timestamp,
        )

        overflow = len(self._latest_opportunities) - max_items
        for key, _ in sorted_items[:overflow]:
            self._latest_opportunities.pop(key, None)

        return overflow

    def _drop_oldest_window(self) -> None:
        if not self._spread_windows:
            return

        oldest_key = next(iter(self._spread_windows))
        self._spread_windows.pop(oldest_key, None)

    # ------------------------------------------------------------------
    # Key / topic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quote_key(quote: QuoteSnapshot) -> tuple[str, str, InstrumentType]:
        return (
            quote.exchange,
            quote.symbol,
            quote.instrument_type,
        )

    @staticmethod
    def _snapshot_key(snapshot: SpreadSnapshot) -> tuple[str, str, str, InstrumentType]:
        return (
            snapshot.symbol,
            snapshot.leg_a_exchange,
            snapshot.leg_b_exchange,
            snapshot.leg_a_type,
        )

    @staticmethod
    def _snapshot_key_from_values(
        *,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: InstrumentType,
    ) -> tuple[str, str, str, InstrumentType]:
        return (
            symbol,
            exchange_a,
            exchange_b,
            instrument_type,
        )

    def _topic(self, config_attr: str, fallback: str) -> str:
        value = getattr(self._config, config_attr, None)
        if isinstance(value, str) and value.strip():
            return value
        return fallback