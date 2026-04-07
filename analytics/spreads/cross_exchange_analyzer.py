from __future__ import annotations

from typing import Any

from .base import BaseSpreadAnalyzer
from .config import CrossExchangeSpreadConfig
from .enums import PricingSource, QuoteValidity, SpreadType
from .models import ArbitrageOpportunity, QuoteSnapshot, SpreadSnapshot
from .spread_opportunity_detector import SpreadOpportunityDetector
from .spread_utils import (
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
    Аналізатор cross-exchange спредів.

    Відповідальність:
    - приймати quotes з кількох бірж
    - порівнювати той самий інструмент між біржами
    - будувати SpreadSnapshot
    - визначати regime через rolling stats
    - шукати arbitrage opportunities
    - генерувати сигнали
    - публікувати snapshots / signals / opportunities
    """

    SNAPSHOT_EVENT = "spread.cross_exchange.updated"
    SIGNAL_EVENT = "spread.signal.generated"
    OPPORTUNITY_EVENT = "spread.arbitrage.opportunity"

    def __init__(
        self,
        config: CrossExchangeSpreadConfig,
        event_bus: Any,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            service_name="cross_exchange_spread",
        )
        self._config = config

        self._opportunity_detector = SpreadOpportunityDetector(config)

        self._quotes: dict[tuple[str, str, Any], QuoteSnapshot] = {}
        self._spread_windows: dict[tuple[str, str, str, Any], RollingDecimalWindow] = {}
        self._latest_snapshots: dict[tuple[str, str, str, Any], SpreadSnapshot] = {}
        self._latest_opportunities: dict[tuple[str, str, str, Any], ArbitrageOpportunity] = {}

        self._stats.update(
            {
                "quotes_received": 0,
                "invalid_quotes": 0,
                "stale_quotes": 0,
                "unaligned_quotes": 0,
                "opportunities_published": 0,
            }
        )

    async def _subscribe_events(self) -> None:
        await self._subscribe("quote.updated", self.on_quote_update)

    def get_latest_snapshot(
        self,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: Any,
    ) -> SpreadSnapshot | None:
        key = (
            normalize_symbol(symbol),
            normalize_exchange(exchange_a),
            normalize_exchange(exchange_b),
            instrument_type,
        )
        return self._latest_snapshots.get(key)

    def get_best_opportunities(
        self,
        symbol: str | None = None,
        instrument_type: Any | None = None,
        profitable_only: bool = True,
        active_only: bool = True,
    ) -> list[ArbitrageOpportunity]:
        opportunities = list(self._latest_opportunities.values())

        if symbol is not None:
            normalized_symbol = normalize_symbol(symbol)
            opportunities = [op for op in opportunities if op.symbol == normalized_symbol]

        if instrument_type is not None:
            opportunities = [
                op
                for op in opportunities
                if op.buy_instrument_type == instrument_type and op.sell_instrument_type == instrument_type
            ]

        if profitable_only:
            opportunities = [op for op in opportunities if op.is_profitable]

        if active_only:
            opportunities = [
                self._opportunity_detector.expire_opportunity(op)
                for op in opportunities
            ]
            opportunities = [
                op for op in opportunities
                if self._opportunity_detector.is_opportunity_active(op)
            ]

        opportunities.sort(key=lambda item: item.net_edge, reverse=True)
        return opportunities

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "quotes_cached": len(self._quotes),
            "active_windows": len(self._spread_windows),
            "latest_snapshots": len(self._latest_snapshots),
            "latest_opportunities": len(self._latest_opportunities),
        }

    async def on_quote_update(self, quote: QuoteSnapshot) -> None:
        if not self.is_running or not self._config.enabled:
            return

        async with self._lock:
            try:
                self._stats["quotes_received"] += 1

                normalized_quote = self._normalize_quote(quote)

                if normalized_quote.instrument_type not in self._config.allowed_instrument_types:
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
                    return

                self._store_quote(normalized_quote)
                await self._recalculate_for_quote(normalized_quote)

            except Exception as exc:
                self._mark_exception(
                    "Failed to process cross-exchange quote update",
                    exc,
                    exchange=getattr(quote, "exchange", None),
                    symbol=getattr(quote, "symbol", None),
                )

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
        key = (quote.exchange, quote.symbol, quote.instrument_type)
        self._quotes[key] = quote

    async def _recalculate_for_quote(self, quote: QuoteSnapshot) -> None:
        candidates = [
            other
            for (exchange, symbol, instrument_type), other in self._quotes.items()
            if symbol == quote.symbol
            and instrument_type == quote.instrument_type
            and exchange != quote.exchange
        ]

        for other_quote in candidates:
            if self._config.preferred_exchanges:
                if (
                    quote.exchange not in self._config.preferred_exchanges
                    or other_quote.exchange not in self._config.preferred_exchanges
                ):
                    continue

            await self._try_build_and_publish(quote, other_quote)

    async def _try_build_and_publish(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
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
            return

        key = (
            snapshot.symbol,
            snapshot.leg_a_exchange,
            snapshot.leg_b_exchange,
            snapshot.leg_a_type,
        )

        previous_snapshot = self._latest_snapshots.get(key)

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

        opportunity = await self._detect_opportunity(ordered_a, ordered_b, snapshot)
        self._latest_snapshots[key] = snapshot
        self._stats["calculations_total"] += 1

        await self._publish_snapshot(self.SNAPSHOT_EVENT, snapshot)

        signals = self._evaluate_snapshot_signals(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            opportunity=opportunity,
        )
        await self._publish_signals(self.SIGNAL_EVENT, signals)

        if opportunity is not None:
            self._latest_opportunities[key] = opportunity
            await self._publish_opportunity(opportunity)

    def _order_quotes(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[QuoteSnapshot, QuoteSnapshot]:
        if quote_a.exchange <= quote_b.exchange:
            return quote_a, quote_b
        return quote_b, quote_a

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

        window_key = (symbol, quote_a.exchange, quote_b.exchange, instrument_type)
        window = self._spread_windows.get(window_key)
        if window is None:
            window = RollingDecimalWindow(
                maxlen=self._config.rolling_window_size,
                ema_alpha=self._config.ema_alpha,
            )
            self._spread_windows[window_key] = window

        window.append(raw_spread)
        stats = window.stats()

        regime_result = self._regime_detector.detect_from_stats(stats)

        buy_exchange, sell_exchange, buy_price, sell_price, gross_edge_per_unit = self._best_arbitrage_legs(
            quote_a,
            quote_b,
        )

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
                "regime_reason": regime_result.reason,
                "is_compressed": regime_result.is_compressed,
                "is_elevated": regime_result.is_elevated,
                "is_extreme": regime_result.is_extreme,
                "is_dislocated": regime_result.is_dislocated,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": str(buy_price) if buy_price is not None else None,
                "sell_price": str(sell_price) if sell_price is not None else None,
                "gross_edge": str(gross_edge_per_unit) if gross_edge_per_unit is not None else None,
            },
        )

    async def _detect_opportunity(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        snapshot: SpreadSnapshot,
    ) -> ArbitrageOpportunity | None:
        result = self._opportunity_detector.detect_from_quotes(
            quote_a=quote_a,
            quote_b=quote_b,
        )

        if not result.found or result.opportunity is None:
            return None

        opportunity = result.opportunity

        if result.costs is not None:
            snapshot.estimated_fees = result.costs.estimated_fees
            snapshot.estimated_slippage = result.costs.estimated_slippage
            snapshot.net_spread = result.costs.net_edge

        snapshot.metadata["opportunity_reason"] = result.reason
        snapshot.metadata["opportunity_net_edge"] = str(opportunity.net_edge)
        snapshot.metadata["opportunity_status"] = opportunity.status.value

        return opportunity

    def _best_arbitrage_legs(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[str | None, str | None, Any | None, Any | None, Any | None]:
        if quote_a.ask is None or quote_a.bid is None:
            return None, None, None, None, None

        if quote_b.ask is None or quote_b.bid is None:
            return None, None, None, None, None

        option_1_edge = quote_b.bid - quote_a.ask
        option_2_edge = quote_a.bid - quote_b.ask

        if option_1_edge >= option_2_edge and option_1_edge > 0:
            return (
                quote_a.exchange,
                quote_b.exchange,
                quote_a.ask,
                quote_b.bid,
                option_1_edge,
            )

        if option_2_edge > option_1_edge and option_2_edge > 0:
            return (
                quote_b.exchange,
                quote_a.exchange,
                quote_b.ask,
                quote_a.bid,
                option_2_edge,
            )

        return None, None, None, None, None

    async def _publish_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        self._stats["opportunities_published"] += 1

        self._logger.debug(
            "Arbitrage opportunity published",
            extra={
                "symbol": opportunity.symbol,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "gross_edge": str(opportunity.gross_edge),
                "net_edge": str(opportunity.net_edge),
                "spread_bps": str(opportunity.spread_bps) if opportunity.spread_bps is not None else None,
                "status": opportunity.status.value,
            },
        )

        await self._publish(self.OPPORTUNITY_EVENT, opportunity)