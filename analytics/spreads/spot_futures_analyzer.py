from __future__ import annotations

from typing import Any

from .base import BaseSpreadAnalyzer
from .config import SpotFuturesSpreadConfig
from .enums import InstrumentType, PricingSource, QuoteValidity, SpreadType
from .models import FundingSnapshot, QuoteSnapshot, SpreadSnapshot
from .spread_utils import (
    RollingDecimalWindow,
    aligned_quotes,
    basis_from_prices,
    funding_adjusted_spread,
    infer_direction,
    normalize_exchange,
    normalize_symbol,
    quote_age_ms,
    spread_abs,
    spread_bps,
    spread_pct,
    validate_quote_snapshot,
)


class SpotFuturesSpreadAnalyzer(BaseSpreadAnalyzer):
    """
    Аналізатор спреду між spot і futures/perpetual інструментом.

    Відповідальність:
    - приймати spot/futures quotes
    - приймати funding updates
    - зберігати актуальний state
    - будувати SpreadSnapshot
    - рахувати basis / spread_pct / spread_bps / funding-adjusted spread
    - оновлювати rolling stats
    - визначати regime
    - генерувати сигнали
    - публікувати snapshots і signals у EventBus
    """

    SNAPSHOT_EVENT = "spread.spot_futures.updated"
    SIGNAL_EVENT = "spread.signal.generated"

    def __init__(
        self,
        config: SpotFuturesSpreadConfig,
        event_bus: Any,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            service_name="spot_futures_spread",
        )
        self._config = config

        self._spot_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._futures_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._funding: dict[tuple[str, str], FundingSnapshot] = {}

        self._spread_windows: dict[tuple[str, str, str], RollingDecimalWindow] = {}
        self._latest_snapshots: dict[tuple[str, str, str], SpreadSnapshot] = {}

        self._stats.update(
            {
                "quotes_received": 0,
                "funding_updates": 0,
                "invalid_quotes": 0,
                "stale_quotes": 0,
                "unaligned_quotes": 0,
            }
        )

    async def _subscribe_events(self) -> None:
        await self._subscribe("quote.updated", self.on_quote_update)
        await self._subscribe("funding.updated", self.on_funding_update)

    def get_latest_snapshot(
        self,
        symbol: str,
        spot_exchange: str,
        futures_exchange: str,
    ) -> SpreadSnapshot | None:
        key = (
            normalize_symbol(symbol),
            normalize_exchange(spot_exchange),
            normalize_exchange(futures_exchange),
        )
        return self._latest_snapshots.get(key)

    def get_stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "spot_quotes_cached": len(self._spot_quotes),
            "futures_quotes_cached": len(self._futures_quotes),
            "funding_cached": len(self._funding),
            "active_windows": len(self._spread_windows),
            "latest_snapshots": len(self._latest_snapshots),
        }

    async def on_quote_update(self, quote: QuoteSnapshot) -> None:
        if not self.is_running or not self._config.enabled:
            return

        async with self._lock:
            try:
                self._stats["quotes_received"] += 1

                normalized_quote = self._normalize_quote(quote)
                validity = validate_quote_snapshot(
                    normalized_quote,
                    max_age_ms=self._config.max_quote_age_ms,
                )

                if validity == QuoteValidity.INVALID:
                    self._stats["invalid_quotes"] += 1
                    self._logger.debug(
                        "Rejected invalid quote",
                        extra={
                            "exchange": normalized_quote.exchange,
                            "symbol": normalized_quote.symbol,
                            "instrument_type": normalized_quote.instrument_type.value,
                        },
                    )
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
                    "Failed to process spot/futures quote update",
                    exc,
                    exchange=getattr(quote, "exchange", None),
                    symbol=getattr(quote, "symbol", None),
                )

    async def on_funding_update(self, funding: FundingSnapshot) -> None:
        if not self.is_running or not self._config.enabled:
            return

        async with self._lock:
            try:
                self._stats["funding_updates"] += 1

                normalized_funding = FundingSnapshot(
                    exchange=normalize_exchange(funding.exchange),
                    symbol=normalize_symbol(funding.symbol),
                    funding_rate=funding.funding_rate,
                    timestamp=funding.timestamp,
                    next_funding_time=funding.next_funding_time,
                    predicted_rate=funding.predicted_rate,
                    interval_hours=funding.interval_hours,
                    metadata=dict(funding.metadata),
                )

                key = (normalized_funding.exchange, normalized_funding.symbol)
                self._funding[key] = normalized_funding

                await self._recalculate_for_funding(normalized_funding)

            except Exception as exc:
                self._mark_exception(
                    "Failed to process funding update",
                    exc,
                    exchange=getattr(funding, "exchange", None),
                    symbol=getattr(funding, "symbol", None),
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
        key = (quote.exchange, quote.symbol)

        if quote.instrument_type == InstrumentType.SPOT:
            self._spot_quotes[key] = quote
            return

        if quote.instrument_type in {InstrumentType.PERPETUAL, InstrumentType.FUTURES}:
            self._futures_quotes[key] = quote

    async def _recalculate_for_quote(self, quote: QuoteSnapshot) -> None:
        symbol = quote.symbol

        if quote.instrument_type == InstrumentType.SPOT:
            futures_candidates = [
                fut
                for (exchange, sym), fut in self._futures_quotes.items()
                if sym == symbol and self._is_allowed_futures_exchange(exchange)
            ]

            for futures_quote in futures_candidates:
                await self._try_build_and_publish(quote, futures_quote)
            return

        if quote.instrument_type in {InstrumentType.PERPETUAL, InstrumentType.FUTURES}:
            if not self._is_allowed_futures_exchange(quote.exchange):
                return

            spot_candidates = [
                spot
                for (exchange, sym), spot in self._spot_quotes.items()
                if sym == symbol and self._is_allowed_spot_exchange(exchange)
            ]

            for spot_quote in spot_candidates:
                await self._try_build_and_publish(spot_quote, quote)

    async def _recalculate_for_funding(self, funding: FundingSnapshot) -> None:
        symbol = funding.symbol
        futures_exchange = funding.exchange

        if not self._is_allowed_futures_exchange(futures_exchange):
            return

        futures_quote = self._futures_quotes.get((futures_exchange, symbol))
        if futures_quote is None:
            return

        spot_candidates = [
            spot
            for (exchange, sym), spot in self._spot_quotes.items()
            if sym == symbol and self._is_allowed_spot_exchange(exchange)
        ]

        for spot_quote in spot_candidates:
            await self._try_build_and_publish(spot_quote, futures_quote)

    async def _try_build_and_publish(
        self,
        spot_quote: QuoteSnapshot,
        futures_quote: QuoteSnapshot,
    ) -> None:
        if spot_quote.instrument_type != InstrumentType.SPOT:
            return

        if futures_quote.instrument_type not in {
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        }:
            return

        if spot_quote.symbol != futures_quote.symbol:
            return

        spot_validity = validate_quote_snapshot(
            spot_quote,
            max_age_ms=self._config.max_quote_age_ms,
        )
        futures_validity = validate_quote_snapshot(
            futures_quote,
            max_age_ms=self._config.max_quote_age_ms,
        )

        if spot_validity == QuoteValidity.STALE or futures_validity == QuoteValidity.STALE:
            self._stats["stale_quotes"] += 1
            return

        if spot_validity != QuoteValidity.VALID or futures_validity != QuoteValidity.VALID:
            self._stats["invalid_quotes"] += 1
            return

        if not aligned_quotes(
            spot_quote,
            futures_quote,
            max_age_diff_ms=self._config.max_quote_skew_ms,
        ):
            self._stats["unaligned_quotes"] += 1
            return

        snapshot = self._build_snapshot(spot_quote, futures_quote)
        if snapshot is None:
            return

        key = (snapshot.symbol, snapshot.leg_a_exchange, snapshot.leg_b_exchange)

        previous_snapshot = self._latest_snapshots.get(key)

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

        self._latest_snapshots[key] = snapshot
        self._stats["calculations_total"] += 1

        await self._publish_snapshot(self.SNAPSHOT_EVENT, snapshot)

        signals = self._evaluate_snapshot_signals(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
        )
        await self._publish_signals(self.SIGNAL_EVENT, signals)

    def _build_snapshot(
        self,
        spot_quote: QuoteSnapshot,
        futures_quote: QuoteSnapshot,
    ) -> SpreadSnapshot | None:
        symbol = spot_quote.symbol
        spot_exchange = spot_quote.exchange
        futures_exchange = futures_quote.exchange

        spot_mid = spot_quote.mid_price
        futures_mid = futures_quote.mid_price

        raw_spread = spread_abs(futures_mid, spot_mid)
        if raw_spread is None:
            return None

        spread_percent = spread_pct(raw_spread, spot_mid)
        spread_bps_value = spread_bps(raw_spread, spot_mid)
        basis = basis_from_prices(futures_mid, spot_mid)

        funding_snapshot = self._funding.get((futures_exchange, symbol))
        funding_rate = funding_snapshot.funding_rate if funding_snapshot is not None else None

        funding_adjusted = funding_adjusted_spread(
            raw_spread=raw_spread,
            funding_rate=funding_rate,
            notional=self._config.notional_for_funding_adjustment,
        )

        window_key = (symbol, spot_exchange, futures_exchange)
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

        return SpreadSnapshot(
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=symbol,
            leg_a_exchange=spot_exchange,
            leg_b_exchange=futures_exchange,
            leg_a_type=InstrumentType.SPOT,
            leg_b_type=futures_quote.instrument_type,
            pricing_source=PricingSource.BID_ASK,
            raw_spread=raw_spread,
            spread_pct=spread_percent,
            spread_bps=spread_bps_value,
            net_spread=funding_adjusted,
            basis=basis,
            funding_adjusted_spread=funding_adjusted,
            direction=infer_direction(raw_spread),
            regime=regime_result.regime,
            stats=stats,
            leg_a_bid=spot_quote.bid,
            leg_a_ask=spot_quote.ask,
            leg_b_bid=futures_quote.bid,
            leg_b_ask=futures_quote.ask,
            leg_a_mid=spot_mid,
            leg_b_mid=futures_mid,
            estimated_fees=None,
            estimated_slippage=None,
            quote_validity=QuoteValidity.VALID,
            timestamp=max(spot_quote.timestamp, futures_quote.timestamp),
            metadata={
                "spot_age_ms": quote_age_ms(spot_quote),
                "futures_age_ms": quote_age_ms(futures_quote),
                "funding_rate": str(funding_rate) if funding_rate is not None else None,
                "spot_exchange": spot_exchange,
                "futures_exchange": futures_exchange,
                "regime_reason": regime_result.reason,
                "is_compressed": regime_result.is_compressed,
                "is_elevated": regime_result.is_elevated,
                "is_extreme": regime_result.is_extreme,
                "is_dislocated": regime_result.is_dislocated,
            },
        )

    def _is_allowed_spot_exchange(self, exchange: str) -> bool:
        if self._config.default_spot_exchange is None:
            return True
        return exchange == normalize_exchange(self._config.default_spot_exchange)

    def _is_allowed_futures_exchange(self, exchange: str) -> bool:
        if self._config.default_futures_exchange is None:
            return True
        return exchange == normalize_exchange(self._config.default_futures_exchange)