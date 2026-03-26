from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.logger import get_logger

from .enums import (
    InstrumentType,
    OpportunityStatus,
    PricingSource,
    QuoteValidity,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from .models import ArbitrageOpportunity, QuoteSnapshot, SpreadSignal, SpreadSnapshot
from .utils import (
    RollingDecimalWindow,
    aligned_quotes,
    estimate_fee_cost,
    estimate_simple_slippage,
    infer_direction,
    infer_regime,
    net_edge_after_costs,
    normalize_exchange,
    normalize_symbol,
    quote_age_ms,
    safe_div,
    spread_abs,
    spread_bps,
    spread_pct,
    validate_quote_snapshot,
)

DEFAULT_ZERO = Decimal("0")


@dataclass(slots=True)
class CrossExchangeSpreadConfig:
    enabled: bool = True

    max_quote_age_ms: int = 2_000
    max_quote_skew_ms: int = 1_000

    rolling_window_size: int = 500
    ema_alpha: Decimal = Decimal("0.2")

    min_emit_interval_ms: int = 250
    cooldown_seconds: int = 10

    arbitrage_min_bps: Decimal = Decimal("5")
    anomaly_zscore_threshold: Decimal = Decimal("2.5")
    widening_bps_threshold: Decimal = Decimal("8")

    default_trade_size: Decimal = Decimal("1")
    slippage_max_bps: Decimal = Decimal("5")
    safety_buffer_bps: Decimal = Decimal("1")

    default_taker_fee_rate: Decimal = Decimal("0.001")
    default_maker_fee_rate: Decimal = Decimal("0.0005")

    allowed_instrument_types: set[InstrumentType] = field(
        default_factory=lambda: {InstrumentType.SPOT, InstrumentType.PERPETUAL, InstrumentType.FUTURES}
    )
    preferred_exchanges: set[str] = field(default_factory=set)

    metadata: dict[str, Any] = field(default_factory=dict)


class CrossExchangeSpreadAnalyzer:
    """
    Аналізатор cross-exchange спредів.

    Основні задачі:
    - приймати quotes з кількох бірж
    - порівнювати той самий інструмент між біржами
    - рахувати gross spread / spread bps / net edge
    - враховувати fees/slippage
    - знаходити arbitrage opportunities
    - публікувати snapshots і signals
    """

    def __init__(
        self,
        config: CrossExchangeSpreadConfig,
        event_bus: Any,
        scheduler: Any | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._logger = get_logger(__name__, service_name="cross_exchange_spread")

        self._running = False
        self._lock = asyncio.Lock()

        self._quotes: dict[tuple[str, str, InstrumentType], QuoteSnapshot] = {}
        self._spread_windows: dict[tuple[str, str, str, InstrumentType], RollingDecimalWindow] = {}
        self._latest_snapshots: dict[tuple[str, str, str, InstrumentType], SpreadSnapshot] = {}
        self._latest_opportunities: dict[tuple[str, str, str, InstrumentType], ArbitrageOpportunity] = {}

        self._last_signal_times: dict[str, datetime] = {}
        self._last_emit_times: dict[tuple[str, str, str, InstrumentType], datetime] = {}

        self._stats: dict[str, int] = {
            "quotes_received": 0,
            "calculations_total": 0,
            "snapshots_published": 0,
            "signals_published": 0,
            "opportunities_published": 0,
            "invalid_quotes": 0,
            "stale_quotes": 0,
            "unaligned_quotes": 0,
            "cooldown_skips": 0,
            "emit_skips": 0,
            "exceptions": 0,
        }

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        await self._subscribe_events()

        self._logger.info(
            "CrossExchangeSpreadAnalyzer started",
            extra={
                "max_quote_age_ms": self._config.max_quote_age_ms,
                "max_quote_skew_ms": self._config.max_quote_skew_ms,
                "rolling_window_size": self._config.rolling_window_size,
                "arbitrage_min_bps": str(self._config.arbitrage_min_bps),
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._logger.info("CrossExchangeSpreadAnalyzer stopped")

    async def on_quote_update(self, quote: QuoteSnapshot) -> None:
        if not self._running or not self._config.enabled:
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
                self._stats["exceptions"] += 1
                self._logger.exception(
                    "Failed to process cross-exchange quote update",
                    extra={"error": str(exc)},
                )

    def get_latest_snapshot(
        self,
        symbol: str,
        exchange_a: str,
        exchange_b: str,
        instrument_type: InstrumentType,
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
        instrument_type: InstrumentType | None = None,
        profitable_only: bool = True,
    ) -> list[ArbitrageOpportunity]:
        opportunities = list(self._latest_opportunities.values())

        if symbol is not None:
            symbol = normalize_symbol(symbol)
            opportunities = [op for op in opportunities if op.symbol == symbol]

        if instrument_type is not None:
            opportunities = [
                op
                for op in opportunities
                if op.buy_instrument_type == instrument_type and op.sell_instrument_type == instrument_type
            ]

        if profitable_only:
            opportunities = [op for op in opportunities if op.is_profitable]

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

    async def _subscribe_events(self) -> None:
        subscribe = getattr(self._event_bus, "subscribe", None)
        if subscribe is None:
            self._logger.warning("EventBus does not expose subscribe()")
            return

        await self._maybe_await(
            subscribe("quote.updated", self.on_quote_update)
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

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

        self._latest_snapshots[key] = snapshot
        self._stats["calculations_total"] += 1

        await self._publish_snapshot(snapshot)
        await self._evaluate_and_publish_signals(snapshot)
        await self._maybe_publish_opportunity(snapshot)

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

        buy_exchange, sell_exchange, buy_price, sell_price = self._best_arbitrage_legs(
            quote_a,
            quote_b,
        )

        gross_edge = None
        estimated_fees = None
        estimated_slippage = None
        net_edge = None

        if buy_exchange is not None and sell_exchange is not None:
            gross_edge = sell_price - buy_price

            estimated_fees = self._estimate_total_fees(
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=self._config.default_trade_size,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
            )
            estimated_slippage = self._estimate_total_slippage(
                quote_a=quote_a,
                quote_b=quote_b,
                quantity=self._config.default_trade_size,
            )
            net_edge = net_edge_after_costs(
                gross_edge=gross_edge,
                fees=estimated_fees,
                slippage=estimated_slippage,
            )

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

        regime = infer_regime(
            stats.zscore,
            elevated_threshold=Decimal("1.5"),
            extreme_threshold=self._config.anomaly_zscore_threshold,
            compressed_threshold=Decimal("0.5"),
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
            net_spread=net_edge,
            basis=None,
            funding_adjusted_spread=None,
            direction=infer_direction(raw_spread),
            regime=regime,
            stats=stats,
            leg_a_bid=quote_a.bid,
            leg_a_ask=quote_a.ask,
            leg_b_bid=quote_b.bid,
            leg_b_ask=quote_b.ask,
            leg_a_mid=mid_a,
            leg_b_mid=mid_b,
            estimated_fees=estimated_fees,
            estimated_slippage=estimated_slippage,
            quote_validity=QuoteValidity.VALID,
            timestamp=max(quote_a.timestamp, quote_b.timestamp),
            metadata={
                "instrument_type": instrument_type.value,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": str(buy_price) if buy_price is not None else None,
                "sell_price": str(sell_price) if sell_price is not None else None,
                "gross_edge": str(gross_edge) if gross_edge is not None else None,
                "quote_a_age_ms": quote_age_ms(quote_a),
                "quote_b_age_ms": quote_age_ms(quote_b),
            },
        )

    def _best_arbitrage_legs(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[str | None, str | None, Decimal | None, Decimal | None]:
        if quote_a.ask is None or quote_a.bid is None or quote_b.ask is None or quote_b.bid is None:
            return None, None, None, None

        option_1_buy = quote_a.exchange
        option_1_sell = quote_b.exchange
        option_1_buy_price = quote_a.ask
        option_1_sell_price = quote_b.bid
        option_1_edge = option_1_sell_price - option_1_buy_price

        option_2_buy = quote_b.exchange
        option_2_sell = quote_a.exchange
        option_2_buy_price = quote_b.ask
        option_2_sell_price = quote_a.bid
        option_2_edge = option_2_sell_price - option_2_buy_price

        if option_1_edge >= option_2_edge and option_1_edge > DEFAULT_ZERO:
            return option_1_buy, option_1_sell, option_1_buy_price, option_1_sell_price

        if option_2_edge > option_1_edge and option_2_edge > DEFAULT_ZERO:
            return option_2_buy, option_2_sell, option_2_buy_price, option_2_sell_price

        return None, None, None, None

    def _estimate_total_fees(
        self,
        buy_price: Decimal,
        sell_price: Decimal,
        quantity: Decimal,
        buy_exchange: str,
        sell_exchange: str,
    ) -> Decimal:
        buy_fee_rate = self._get_fee_rate(buy_exchange, side="buy")
        sell_fee_rate = self._get_fee_rate(sell_exchange, side="sell")

        buy_fee = estimate_fee_cost(
            price=buy_price,
            quantity=quantity,
            fee_rate=buy_fee_rate,
        )
        sell_fee = estimate_fee_cost(
            price=sell_price,
            quantity=quantity,
            fee_rate=sell_fee_rate,
        )
        return buy_fee + sell_fee

    def _estimate_total_slippage(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        quantity: Decimal,
    ) -> Decimal:
        slip_a_ratio = estimate_simple_slippage(
            quantity=quantity,
            top_book_size=quote_a.ask_size or quote_a.bid_size,
            max_slippage_bps=self._config.slippage_max_bps,
        )
        slip_b_ratio = estimate_simple_slippage(
            quantity=quantity,
            top_book_size=quote_b.ask_size or quote_b.bid_size,
            max_slippage_bps=self._config.slippage_max_bps,
        )

        cost_a = (quote_a.mid_price or DEFAULT_ZERO) * slip_a_ratio
        cost_b = (quote_b.mid_price or DEFAULT_ZERO) * slip_b_ratio

        return cost_a + cost_b

    def _get_fee_rate(self, exchange: str, side: str) -> Decimal:
        fee_overrides = self._config.metadata.get("fee_rates", {})
        exchange_rates = fee_overrides.get(exchange, {})

        if side == "buy":
            return Decimal(str(exchange_rates.get("buy", self._config.default_taker_fee_rate)))

        if side == "sell":
            return Decimal(str(exchange_rates.get("sell", self._config.default_taker_fee_rate)))

        return self._config.default_taker_fee_rate

    async def _publish_snapshot(self, snapshot: SpreadSnapshot) -> None:
        self._stats["snapshots_published"] += 1

        self._logger.debug(
            "Cross-exchange spread snapshot published",
            extra={
                "symbol": snapshot.symbol,
                "exchange_a": snapshot.leg_a_exchange,
                "exchange_b": snapshot.leg_b_exchange,
                "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps is not None else None,
                "net_spread": str(snapshot.net_spread) if snapshot.net_spread is not None else None,
            },
        )

        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            return

        await self._maybe_await(
            publish("spread.cross_exchange.updated", snapshot)
        )

    async def _evaluate_and_publish_signals(self, snapshot: SpreadSnapshot) -> None:
        await self._maybe_publish_widening_signal(snapshot)
        await self._maybe_publish_anomaly_signal(snapshot)

    async def _maybe_publish_widening_signal(self, snapshot: SpreadSnapshot) -> None:
        if snapshot.spread_bps is None:
            return

        if abs(snapshot.spread_bps) < self._config.widening_bps_threshold:
            return

        signal = SpreadSignal(
            signal_type=SpreadSignalType.WIDENING,
            spread_type=SpreadType.CROSS_EXCHANGE,
            symbol=snapshot.symbol,
            message=(
                f"Cross-exchange spread widened to {snapshot.spread_bps} bps "
                f"between {snapshot.leg_a_exchange} and {snapshot.leg_b_exchange}"
            ),
            value=snapshot.spread_bps,
            threshold=self._config.widening_bps_threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "instrument_type": snapshot.leg_a_type.value,
                "net_spread": str(snapshot.net_spread) if snapshot.net_spread is not None else None,
            },
        )
        await self._publish_signal(signal)

    async def _maybe_publish_anomaly_signal(self, snapshot: SpreadSnapshot) -> None:
        zscore = snapshot.stats.zscore if snapshot.stats else None
        if zscore is None:
            return

        if abs(zscore) < self._config.anomaly_zscore_threshold:
            return

        signal = SpreadSignal(
            signal_type=SpreadSignalType.ANOMALY,
            spread_type=SpreadType.CROSS_EXCHANGE,
            symbol=snapshot.symbol,
            message=(
                f"Cross-exchange spread anomaly detected: z-score={zscore} "
                f"for {snapshot.symbol} on {snapshot.leg_a_exchange}/{snapshot.leg_b_exchange}"
            ),
            value=zscore,
            threshold=self._config.anomaly_zscore_threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps is not None else None,
                "regime": snapshot.regime.value,
            },
        )
        await self._publish_signal(signal)

    async def _maybe_publish_opportunity(self, snapshot: SpreadSnapshot) -> None:
        buy_exchange = snapshot.metadata.get("buy_exchange")
        sell_exchange = snapshot.metadata.get("sell_exchange")

        buy_price_raw = snapshot.metadata.get("buy_price")
        sell_price_raw = snapshot.metadata.get("sell_price")
        gross_edge_raw = snapshot.metadata.get("gross_edge")

        if not buy_exchange or not sell_exchange:
            return
        if buy_price_raw is None or sell_price_raw is None or gross_edge_raw is None:
            return
        if snapshot.net_spread is None:
            return

        reference_price = Decimal(str(buy_price_raw))
        net_bps = spread_bps(snapshot.net_spread, reference_price)
        if net_bps is None:
            return

        threshold_bps = self._config.arbitrage_min_bps + self._config.safety_buffer_bps
        if net_bps < threshold_bps:
            return

        opportunity = ArbitrageOpportunity(
            symbol=snapshot.symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_instrument_type=snapshot.leg_a_type,
            sell_instrument_type=snapshot.leg_b_type,
            buy_price=Decimal(str(buy_price_raw)),
            sell_price=Decimal(str(sell_price_raw)),
            gross_edge=Decimal(str(gross_edge_raw)),
            estimated_fees=snapshot.estimated_fees or DEFAULT_ZERO,
            estimated_slippage=snapshot.estimated_slippage or DEFAULT_ZERO,
            net_edge=snapshot.net_spread,
            spread_pct=spread_pct(snapshot.net_spread, reference_price),
            spread_bps=net_bps,
            confidence=self._confidence_from_snapshot(snapshot),
            status=OpportunityStatus.ACTIVE,
            timestamp=snapshot.timestamp,
            expires_at=snapshot.timestamp + timedelta(milliseconds=self._config.max_quote_age_ms),
            metadata={
                "leg_a_exchange": snapshot.leg_a_exchange,
                "leg_b_exchange": snapshot.leg_b_exchange,
                "regime": snapshot.regime.value,
            },
        )

        key = (
            opportunity.symbol,
            opportunity.buy_exchange,
            opportunity.sell_exchange,
            opportunity.buy_instrument_type,
        )
        self._latest_opportunities[key] = opportunity

        self._stats["opportunities_published"] += 1

        signal = SpreadSignal(
            signal_type=SpreadSignalType.ARBITRAGE,
            spread_type=SpreadType.CROSS_EXCHANGE,
            symbol=opportunity.symbol,
            message=(
                f"Arbitrage candidate detected: buy on {opportunity.buy_exchange}, "
                f"sell on {opportunity.sell_exchange}, net edge={opportunity.net_edge}"
            ),
            value=opportunity.spread_bps,
            threshold=threshold_bps,
            confidence=opportunity.confidence,
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            metadata={
                "buy_price": str(opportunity.buy_price),
                "sell_price": str(opportunity.sell_price),
                "gross_edge": str(opportunity.gross_edge),
                "net_edge": str(opportunity.net_edge),
            },
        )
        await self._publish_signal(signal)

        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            return

        await self._maybe_await(
            publish("spread.cross_exchange.opportunity", opportunity)
        )

    async def _publish_signal(self, signal: SpreadSignal) -> None:
        signal_key = (
            f"{signal.signal_type.value}:{signal.symbol}:"
            f"{signal.exchange_a or 'na'}:{signal.exchange_b or 'na'}"
        )

        if self._is_in_cooldown(signal_key):
            self._stats["cooldown_skips"] += 1
            return

        self._last_signal_times[signal_key] = datetime.utcnow()
        self._stats["signals_published"] += 1

        self._logger.info(
            "Cross-exchange spread signal published",
            extra={
                "signal_type": signal.signal_type.value,
                "symbol": signal.symbol,
                "value": str(signal.value) if signal.value is not None else None,
                "threshold": str(signal.threshold) if signal.threshold is not None else None,
            },
        )

        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            return

        await self._maybe_await(
            publish("spread.signal", signal)
        )

    def _confidence_from_snapshot(self, snapshot: SpreadSnapshot) -> Decimal:
        base = Decimal("0.50")
        stats = snapshot.stats

        if stats and stats.zscore is not None:
            abs_z = abs(stats.zscore)

            if abs_z >= Decimal("3.0"):
                base += Decimal("0.20")
            elif abs_z >= Decimal("2.0"):
                base += Decimal("0.10")

        if snapshot.net_spread is not None and snapshot.net_spread > DEFAULT_ZERO:
            base += Decimal("0.15")

        if snapshot.quote_validity == QuoteValidity.VALID:
            base += Decimal("0.10")

        if snapshot.estimated_slippage is not None and snapshot.estimated_slippage <= DEFAULT_ZERO:
            base += Decimal("0.05")

        if base > Decimal("0.99"):
            return Decimal("0.99")
        return base

    def _is_in_cooldown(self, signal_key: str) -> bool:
        last_time = self._last_signal_times.get(signal_key)
        if last_time is None:
            return False

        elapsed = (datetime.utcnow() - last_time).total_seconds()
        return elapsed < self._config.cooldown_seconds

    def _should_skip_emit(
        self,
        key: tuple[str, str, str, InstrumentType],
        now: datetime,
    ) -> bool:
        last_emit = self._last_emit_times.get(key)
        if last_emit is None:
            self._last_emit_times[key] = now
            return False

        elapsed_ms = (now - last_emit).total_seconds() * 1000
        if elapsed_ms < self._config.min_emit_interval_ms:
            return True

        self._last_emit_times[key] = now
        return False

    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value