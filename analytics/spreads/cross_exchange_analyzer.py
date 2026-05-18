from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from .base import BaseSpreadAnalyzer
from .config import CrossExchangeSpreadConfig
from .enums import InstrumentType, PricingSource, QuoteValidity, SpreadType
from .models import (
    ArbitrageOpportunity,
    QuoteSnapshot,
    SpreadKey,
    SpreadSnapshot,
    spread_key_to_dict,
)
from .spread_opportunity_detector import (
    OpportunityDetectionResult,
    SpreadOpportunityDetector,
)
from .spread_utils import (
    DECIMAL_ZERO,
    RollingDecimalWindow,
    aligned_quotes,
    infer_direction,
    quote_age_ms,
    spread_abs,
    spread_bps,
    spread_pct,
    validate_quote_snapshot,
)


SnapshotKey = tuple[SpreadKey, SpreadKey]


class CrossExchangeSpreadAnalyzer(BaseSpreadAnalyzer):
    """
    Production-grade analyzer для cross-exchange spread analytics.

    Відповідальність:
    - слухати data-layer market.quote.updated;
    - приймати QuoteSnapshot або dict payload від QuoteCache;
    - кешувати quotes з різних бірж;
    - порівнювати один symbol/instrument_type/market_type/timeframe між біржами;
    - будувати SpreadSnapshot;
    - оновлювати rolling stats;
    - визначати spread regime;
    - шукати ArbitrageOpportunity через SpreadOpportunityDetector;
    - генерувати SpreadSignal через SpreadSignalEngine;
    - публікувати analytics.spreads.cross_exchange.updated;
    - публікувати analytics.spreads.signal.generated;
    - публікувати analytics.spreads.arbitrage.opportunity;
    - запускати cleanup/heartbeat через core.Scheduler.

    Correct production input flow:
        exchange adapters
            -> market.quote / market.orderbook
            -> QuoteCache / OrderBookCache
            -> market.quote.updated
            -> CrossExchangeSpreadAnalyzer
            -> analytics.spreads.*

    Важливо:
    - не отримує raw market-data з бірж напряму;
    - не слухає raw market.quote у production;
    - не викликає strategy/risk/execution напряму;
    - state ізольований через SpreadKey:
      exchange + market_type + symbol + timeframe.
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

        self._quotes: dict[SpreadKey, QuoteSnapshot] = {}
        self._spread_windows: dict[SnapshotKey, RollingDecimalWindow] = {}
        self._latest_snapshots: dict[SnapshotKey, SpreadSnapshot] = {}
        self._latest_opportunities: dict[SnapshotKey, ArbitrageOpportunity] = {}

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
                "exchange_pair_skips": 0,
                "scope_skips": 0,
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
                "events_skipped_not_running": 0,
            }
        )

    # ------------------------------------------------------------------
    # Registration / lifecycle hooks
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscriptions.

        Production topic:
            market.quote.updated

        Raw topic:
            market.quote

        не використовується, якщо config.allow_legacy_raw_topics=False.
        """
        if self._registered:
            return

        self._subscribe_quote_updates(
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
        *,
        market_type: str,
        timeframe: str | None = None,
    ) -> SpreadSnapshot | None:
        normalized_timeframe = timeframe or self._config.default_timeframe

        key_a = self.make_key(
            exchange=exchange_a,
            market_type=market_type,
            symbol=symbol,
            timeframe=normalized_timeframe,
        )
        key_b = self.make_key(
            exchange=exchange_b,
            market_type=market_type,
            symbol=symbol,
            timeframe=normalized_timeframe,
        )

        normalized_key = self._canonical_pair_key(key_a, key_b)
        snapshot = self._latest_snapshots.get(normalized_key)
        if snapshot is None:
            return None

        if snapshot.leg_a_type != instrument_type or snapshot.leg_b_type != instrument_type:
            return None

        return snapshot

    def get_latest_snapshot_by_keys(
        self,
        *,
        quote_a_key: SpreadKey,
        quote_b_key: SpreadKey,
    ) -> SpreadSnapshot | None:
        return self._latest_snapshots.get(
            self._canonical_pair_key(quote_a_key, quote_b_key)
        )

    def get_best_opportunities(
        self,
        symbol: str | None = None,
        instrument_type: InstrumentType | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
        limit: int | None = None,
    ) -> list[ArbitrageOpportunity]:
        opportunities = list(self._latest_opportunities.values())

        if symbol is not None:
            normalized_symbol = self.normalize_symbol(symbol)
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

        if market_type is not None:
            normalized_market_type = self.normalize_market_type(market_type)
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.buy_market_type == normalized_market_type
                and opportunity.sell_market_type == normalized_market_type
            ]

        if timeframe is not None:
            normalized_timeframe = self.normalize_timeframe(timeframe)
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.timeframe == normalized_timeframe
            ]

        if profitable_only:
            opportunities = [
                opportunity
                for opportunity in opportunities
                if opportunity.is_profitable
            ]

        if active_only:
            active_opportunities: list[ArbitrageOpportunity] = []
            now = datetime.utcnow()

            for opportunity in opportunities:
                self._opportunity_detector.expire_opportunity(opportunity, now=now)
                if self._opportunity_detector.is_opportunity_active(opportunity, now=now):
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
            "scope": "exchange:market_type:symbol:timeframe",
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_quote_update(self, event: Event) -> None:
        """
        Handler для market.quote.updated.

        Payload може бути:
        - QuoteSnapshot;
        - dict payload від QuoteCache/data-layer.
        """
        if not self.is_running or not self._config.enabled:
            self._stats["events_skipped_not_running"] += 1
            return

        self._stats["quote_events_received"] += 1

        quote = self.normalize_quote_event(event)
        if quote is None:
            self._stats["invalid_payloads"] += 1
            self._mark_invalid_payload(
                "expected QuoteSnapshot or quote dict payload",
                payload_type=type(getattr(event, "payload", None)).__name__,
                topic=getattr(event, "topic", None),
            )
            return

        async with self._lock:
            try:
                self._stats["quotes_received"] += 1

                normalized_quote = self._normalize_quote(quote)

                if not self._is_quote_allowed(normalized_quote):
                    self._stats["scope_skips"] += 1
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
                    correlation_id=getattr(event, "correlation_id", None),
                    source_event_id=getattr(event, "event_id", None),
                )

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange quote update",
                    exc,
                    exchange=getattr(quote, "exchange", None),
                    market_type=getattr(quote, "market_type", None),
                    symbol=getattr(quote, "symbol", None),
                    timeframe=getattr(quote, "timeframe", None),
                    event_id=getattr(event, "event_id", None),
                    correlation_id=getattr(event, "correlation_id", None),
                )

    # ------------------------------------------------------------------
    # Scheduler jobs
    # ------------------------------------------------------------------

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
            removed_opportunities = self._cleanup_opportunities(now=now)

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
                "scope": "exchange:market_type:symbol:timeframe",
            },
            priority=EventPriority.LOW,
        )

    # ------------------------------------------------------------------
    # Normalization / state
    # ------------------------------------------------------------------

    def _normalize_quote(self, quote: QuoteSnapshot) -> QuoteSnapshot:
        """
        Rebuild через QuoteSnapshot, щоб гарантувати __post_init__
        normalization і зберегти market_type/timeframe/exchange_symbol.
        """
        return QuoteSnapshot(
            exchange=quote.exchange,
            symbol=quote.symbol,
            instrument_type=quote.instrument_type,
            market_type=quote.market_type,
            timeframe=quote.timeframe,
            exchange_symbol=quote.exchange_symbol,
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
        self._quotes[quote.key] = quote
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
            for other in self._quotes.values()
            if self._quotes_can_pair(quote, other)
        ]

        for other_quote in candidates:
            if not self._is_exchange_pair_allowed(quote.exchange, other_quote.exchange):
                self._stats["exchange_pair_skips"] += 1
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
        if not self._quotes_can_pair(quote_a, quote_b):
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

        if self._should_skip_snapshot_emit(snapshot):
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
                "leg_a_key": str(spread_key_to_dict(snapshot.leg_a_key)),
                "leg_b_key": str(spread_key_to_dict(snapshot.leg_b_key)),
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
                "leg_a_key": str(spread_key_to_dict(snapshot.leg_a_key)),
                "leg_b_key": str(spread_key_to_dict(snapshot.leg_b_key)),
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

        window_key = self._pair_key(quote_a, quote_b)
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
            timeframe=quote_a.timeframe,
            leg_a_exchange=quote_a.exchange,
            leg_b_exchange=quote_b.exchange,
            leg_a_type=instrument_type,
            leg_b_type=instrument_type,
            leg_a_market_type=quote_a.market_type,
            leg_b_market_type=quote_b.market_type,
            leg_a_exchange_symbol=quote_a.exchange_symbol,
            leg_b_exchange_symbol=quote_b.exchange_symbol,
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
                "quote_a_scope": spread_key_to_dict(quote_a.key),
                "quote_b_scope": spread_key_to_dict(quote_b.key),
                "quote_a_exchange_symbol": quote_a.exchange_symbol,
                "quote_b_exchange_symbol": quote_b.exchange_symbol,
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
                "gross_edge": (
                    str(gross_edge_per_unit)
                    if gross_edge_per_unit is not None
                    else None
                ),
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

        self._enrich_opportunity_scope(
            opportunity=opportunity,
            quote_a=quote_a,
            quote_b=quote_b,
        )

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
                "buy_market_type": opportunity.buy_market_type,
                "sell_market_type": opportunity.sell_market_type,
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
                "buy_market_type": opportunity.buy_market_type,
                "sell_market_type": opportunity.sell_market_type,
                "timeframe": opportunity.timeframe,
                "gross_edge": str(opportunity.gross_edge),
                "net_edge": str(opportunity.net_edge),
                "spread_bps": (
                    str(opportunity.spread_bps)
                    if opportunity.spread_bps is not None
                    else None
                ),
                "status": opportunity.status.value,
            },
        )
        return True

    def _enrich_opportunity_scope(
        self,
        *,
        opportunity: ArbitrageOpportunity,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> None:
        """
        Якщо OpportunityDetector створив opportunity без нових scoped fields,
        дозаповнюємо їх на основі quote legs.

        Це дає backward compatibility з detector-ом, який ще може не знати
        про market_type/timeframe/exchange_symbol.
        """
        quotes_by_exchange = {
            quote_a.exchange: quote_a,
            quote_b.exchange: quote_b,
        }

        buy_quote = quotes_by_exchange.get(opportunity.buy_exchange)
        sell_quote = quotes_by_exchange.get(opportunity.sell_exchange)

        if buy_quote is not None:
            opportunity.buy_market_type = buy_quote.market_type
            opportunity.timeframe = buy_quote.timeframe
            opportunity.buy_exchange_symbol = buy_quote.exchange_symbol

        if sell_quote is not None:
            opportunity.sell_market_type = sell_quote.market_type
            opportunity.timeframe = sell_quote.timeframe
            opportunity.sell_exchange_symbol = sell_quote.exchange_symbol

        opportunity.metadata.setdefault(
            "buy_scope",
            spread_key_to_dict(opportunity.buy_key),
        )
        opportunity.metadata.setdefault(
            "sell_scope",
            spread_key_to_dict(opportunity.sell_key),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _order_quotes(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[QuoteSnapshot, QuoteSnapshot]:
        key_a = (
            quote_a.exchange,
            quote_a.market_type,
            quote_a.symbol,
            quote_a.timeframe,
        )
        key_b = (
            quote_b.exchange,
            quote_b.market_type,
            quote_b.symbol,
            quote_b.timeframe,
        )

        if key_a <= key_b:
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
        key: SnapshotKey,
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

    def _is_quote_allowed(self, quote: QuoteSnapshot) -> bool:
        if not self._is_instrument_type_allowed(quote.instrument_type):
            self._stats["instrument_type_skips"] += 1
            return False

        checker = getattr(self._config, "is_quote_allowed", None)
        if callable(checker):
            return bool(
                checker(
                    exchange=quote.exchange,
                    market_type=quote.market_type,
                    symbol=quote.symbol,
                    instrument_type=quote.instrument_type,
                    timeframe=quote.timeframe,
                )
            )

        return (
            self._is_exchange_allowed(quote.exchange)
            and self.should_process_key(quote.key)
        )

    def _is_instrument_type_allowed(self, instrument_type: InstrumentType) -> bool:
        checker = getattr(self._config, "is_instrument_type_allowed", None)
        if callable(checker):
            return bool(checker(instrument_type))

        return instrument_type in self._config.allowed_instrument_types

    def _is_exchange_allowed(self, exchange: str) -> bool:
        checker = getattr(self._config, "is_exchange_allowed", None)
        if callable(checker):
            return bool(checker(exchange))

        if not self._config.allowed_exchanges:
            return True

        return self.normalize_exchange(exchange) in self._config.allowed_exchanges

    def _is_exchange_pair_allowed(self, exchange_a: str, exchange_b: str) -> bool:
        if exchange_a == exchange_b:
            return False

        return self._is_exchange_allowed(exchange_a) and self._is_exchange_allowed(exchange_b)

    @staticmethod
    def _quotes_can_pair(
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> bool:
        if quote_a.key == quote_b.key:
            return False

        if quote_a.exchange == quote_b.exchange:
            return False

        if quote_a.symbol != quote_b.symbol:
            return False

        if quote_a.instrument_type != quote_b.instrument_type:
            return False

        if quote_a.market_type != quote_b.market_type:
            return False

        if quote_a.timeframe != quote_b.timeframe:
            return False

        return True

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

        removed_by_limit = self._enforce_snapshot_cache_limit()
        return len(stale_keys) + removed_by_limit

    def _cleanup_orphan_windows(self) -> int:
        """
        Видаляє rolling windows, для яких більше немає актуального latest snapshot.
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

    def _cleanup_opportunities(self, *, now: datetime) -> int:
        removed = 0

        for key, opportunity in list(self._latest_opportunities.items()):
            self._opportunity_detector.expire_opportunity(opportunity, now=now)

            if not self._opportunity_detector.is_opportunity_active(opportunity, now=now):
                self._latest_opportunities.pop(key, None)
                removed += 1

                if opportunity.status.value == "expired":
                    self._stats["opportunities_expired"] += 1

        removed_by_limit = self._enforce_opportunity_cache_limit()
        return removed + removed_by_limit

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
    def _canonical_pair_key(
        key_a: SpreadKey,
        key_b: SpreadKey,
    ) -> SnapshotKey:
        if key_a <= key_b:
            return key_a, key_b
        return key_b, key_a

    @staticmethod
    def _pair_key(
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> SnapshotKey:
        return CrossExchangeSpreadAnalyzer._canonical_pair_key(
            quote_a.key,
            quote_b.key,
        )

    @staticmethod
    def _snapshot_key(snapshot: SpreadSnapshot) -> SnapshotKey:
        return CrossExchangeSpreadAnalyzer._canonical_pair_key(
            snapshot.leg_a_key,
            snapshot.leg_b_key,
        )

    def _topic(self, config_attr: str, fallback: str) -> str:
        value = getattr(self._config, config_attr, None)
        if isinstance(value, str) and value.strip():
            return value
        return fallback