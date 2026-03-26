from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from core.logger import get_logger

from .enums import (
    InstrumentType,
    PricingSource,
    QuoteValidity,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from .models import FundingSnapshot, QuoteSnapshot, SpreadSignal, SpreadSnapshot
from .utils import (
    RollingDecimalWindow,
    aligned_quotes,
    basis_from_prices,
    funding_adjusted_spread,
    infer_direction,
    infer_regime,
    normalize_exchange,
    normalize_symbol,
    quote_age_ms,
    spread_abs,
    spread_bps,
    spread_pct,
    validate_quote_snapshot,
)

DEFAULT_ZERO = Decimal("0")


@dataclass(slots=True)
class SpotFuturesSpreadConfig:
    enabled: bool = True

    max_quote_age_ms: int = 2_000
    max_quote_skew_ms: int = 1_000

    rolling_window_size: int = 500
    ema_alpha: Decimal = Decimal("0.2")

    anomaly_zscore_threshold: Decimal = Decimal("2.5")
    mean_reversion_zscore_threshold: Decimal = Decimal("2.0")
    widening_bps_threshold: Decimal = Decimal("8")
    regime_shift_zscore_threshold: Decimal = Decimal("3.0")

    cooldown_seconds: int = 10
    min_emit_interval_ms: int = 250

    notional_for_funding_adjustment: Decimal | None = None

    default_spot_exchange: str | None = None
    default_futures_exchange: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


class SpotFuturesSpreadAnalyzer:
    """
    Аналізатор спреду між spot і futures/perpetual інструментом.

    Основні задачі:
    - приймати spot/futures quotes
    - зберігати актуальний state
    - рахувати raw spread / basis / pct / bps
    - враховувати funding
    - рахувати rolling statistics та z-score
    - генерувати сигнали
    - публікувати snapshots у EventBus
    """

    def __init__(
        self,
        config: SpotFuturesSpreadConfig,
        event_bus: Any,
        scheduler: Any | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._logger = get_logger(__name__, service_name="spot_futures_spread")

        self._running = False
        self._lock = asyncio.Lock()

        self._spot_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._futures_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._funding: dict[tuple[str, str], FundingSnapshot] = {}

        self._spread_windows: dict[tuple[str, str, str], RollingDecimalWindow] = {}
        self._latest_snapshots: dict[tuple[str, str, str], SpreadSnapshot] = {}

        self._last_signal_times: dict[str, datetime] = {}
        self._last_emit_times: dict[tuple[str, str, str], datetime] = {}

        self._stats: dict[str, int] = {
            "quotes_received": 0,
            "funding_updates": 0,
            "calculations_total": 0,
            "snapshots_published": 0,
            "signals_published": 0,
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
            "SpotFuturesSpreadAnalyzer started",
            extra={
                "max_quote_age_ms": self._config.max_quote_age_ms,
                "max_quote_skew_ms": self._config.max_quote_skew_ms,
                "rolling_window_size": self._config.rolling_window_size,
            },
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        self._logger.info("SpotFuturesSpreadAnalyzer stopped")

    async def on_quote_update(self, quote: QuoteSnapshot) -> None:
        if not self._running or not self._config.enabled:
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
                self._stats["exceptions"] += 1
                self._logger.exception(
                    "Failed to process quote update",
                    extra={"error": str(exc)},
                )

    async def on_funding_update(self, funding: FundingSnapshot) -> None:
        if not self._running or not self._config.enabled:
            return

        async with self._lock:
            try:
                self._stats["funding_updates"] += 1

                normalized = FundingSnapshot(
                    exchange=normalize_exchange(funding.exchange),
                    symbol=normalize_symbol(funding.symbol),
                    funding_rate=funding.funding_rate,
                    timestamp=funding.timestamp,
                    next_funding_time=funding.next_funding_time,
                    predicted_rate=funding.predicted_rate,
                    interval_hours=funding.interval_hours,
                    metadata=dict(funding.metadata),
                )

                key = (normalized.exchange, normalized.symbol)
                self._funding[key] = normalized

                await self._recalculate_for_funding(normalized)

            except Exception as exc:
                self._stats["exceptions"] += 1
                self._logger.exception(
                    "Failed to process funding update",
                    extra={"error": str(exc)},
                )

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

    async def _subscribe_events(self) -> None:
        """
        Адаптуй під реальний EventBus.
        Тут я закладаю максимально нейтральний стиль.
        """
        subscribe = getattr(self._event_bus, "subscribe", None)
        if subscribe is None:
            self._logger.warning("EventBus does not expose subscribe()")
            return

        await self._maybe_await(
            subscribe("quote.updated", self.on_quote_update)
        )
        await self._maybe_await(
            subscribe("funding.updated", self.on_funding_update)
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
                fut for (exchange, sym), fut in self._futures_quotes.items()
                if sym == symbol
            ]
            for futures_quote in futures_candidates:
                await self._try_build_and_publish(quote, futures_quote)
            return

        if quote.instrument_type in {InstrumentType.PERPETUAL, InstrumentType.FUTURES}:
            spot_candidates = [
                spot for (exchange, sym), spot in self._spot_quotes.items()
                if sym == symbol
            ]
            for spot_quote in spot_candidates:
                await self._try_build_and_publish(spot_quote, quote)

    async def _recalculate_for_funding(self, funding: FundingSnapshot) -> None:
        symbol = funding.symbol
        futures_exchange = funding.exchange

        futures_quote = self._futures_quotes.get((futures_exchange, symbol))
        if futures_quote is None:
            return

        spot_candidates = [
            spot for (exchange, sym), spot in self._spot_quotes.items()
            if sym == symbol
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

        if futures_quote.instrument_type not in {InstrumentType.PERPETUAL, InstrumentType.FUTURES}:
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

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

        self._latest_snapshots[key] = snapshot
        self._stats["calculations_total"] += 1

        await self._publish_snapshot(snapshot)
        await self._evaluate_and_publish_signals(snapshot)

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
        funding_rate = funding_snapshot.funding_rate if funding_snapshot else None

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

        regime = infer_regime(
            stats.zscore,
            elevated_threshold=Decimal("1.5"),
            extreme_threshold=self._config.anomaly_zscore_threshold,
            compressed_threshold=Decimal("0.5"),
        )

        snapshot = SpreadSnapshot(
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
            regime=regime,
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
            },
        )
        return snapshot

    async def _publish_snapshot(self, snapshot: SpreadSnapshot) -> None:
        self._stats["snapshots_published"] += 1

        self._logger.debug(
            "Spread snapshot published",
            extra={
                "symbol": snapshot.symbol,
                "spot_exchange": snapshot.leg_a_exchange,
                "futures_exchange": snapshot.leg_b_exchange,
                "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps is not None else None,
                "zscore": str(snapshot.stats.zscore) if snapshot.stats and snapshot.stats.zscore is not None else None,
            },
        )

        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            return

        await self._maybe_await(
            publish(
                "spread.spot_futures.updated",
                snapshot,
            )
        )

    async def _evaluate_and_publish_signals(self, snapshot: SpreadSnapshot) -> None:
        await self._maybe_publish_widening_signal(snapshot)
        await self._maybe_publish_anomaly_signal(snapshot)
        await self._maybe_publish_mean_reversion_signal(snapshot)
        await self._maybe_publish_regime_shift_signal(snapshot)

    async def _maybe_publish_widening_signal(self, snapshot: SpreadSnapshot) -> None:
        if snapshot.spread_bps is None:
            return

        if abs(snapshot.spread_bps) < self._config.widening_bps_threshold:
            return

        signal = SpreadSignal(
            signal_type=SpreadSignalType.WIDENING,
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=snapshot.symbol,
            message=(
                f"Spot/futures spread widened to {snapshot.spread_bps} bps "
                f"between {snapshot.leg_a_exchange} and {snapshot.leg_b_exchange}"
            ),
            value=snapshot.spread_bps,
            threshold=self._config.widening_bps_threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "regime": snapshot.regime.value,
                "raw_spread": str(snapshot.raw_spread) if snapshot.raw_spread is not None else None,
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
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=snapshot.symbol,
            message=(
                f"Spot/futures spread anomaly detected: z-score={zscore} "
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

    async def _maybe_publish_mean_reversion_signal(self, snapshot: SpreadSnapshot) -> None:
        stats = snapshot.stats
        if stats is None or stats.zscore is None:
            return

        if abs(stats.zscore) < self._config.mean_reversion_zscore_threshold:
            return

        if snapshot.funding_adjusted_spread is None:
            return

        signal = SpreadSignal(
            signal_type=SpreadSignalType.MEAN_REVERSION,
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=snapshot.symbol,
            message=(
                f"Mean reversion candidate detected for {snapshot.symbol}: "
                f"z-score={stats.zscore}, funding-adjusted spread={snapshot.funding_adjusted_spread}"
            ),
            value=stats.zscore,
            threshold=self._config.mean_reversion_zscore_threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "funding_adjusted_spread": str(snapshot.funding_adjusted_spread),
                "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps is not None else None,
            },
        )
        await self._publish_signal(signal)

    async def _maybe_publish_regime_shift_signal(self, snapshot: SpreadSnapshot) -> None:
        stats = snapshot.stats
        if stats is None or stats.zscore is None:
            return

        if abs(stats.zscore) < self._config.regime_shift_zscore_threshold:
            return

        regime = self._infer_regime_shift(snapshot)
        if regime != SpreadRegime.DISLOCATED:
            return

        signal = SpreadSignal(
            signal_type=SpreadSignalType.REGIME_SHIFT,
            spread_type=SpreadType.SPOT_FUTURES,
            symbol=snapshot.symbol,
            message=(
                f"Regime shift detected for {snapshot.symbol}: "
                f"spread entered dislocated state on {snapshot.leg_a_exchange}/{snapshot.leg_b_exchange}"
            ),
            value=stats.zscore,
            threshold=self._config.regime_shift_zscore_threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "current_regime": regime.value,
                "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps is not None else None,
            },
        )
        await self._publish_signal(signal)

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
            "Spread signal published",
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
                base += Decimal("0.30")
            elif abs_z >= Decimal("2.0"):
                base += Decimal("0.20")
            elif abs_z >= Decimal("1.0"):
                base += Decimal("0.10")

        if snapshot.quote_validity == QuoteValidity.VALID:
            base += Decimal("0.10")

        if snapshot.funding_adjusted_spread is not None:
            base += Decimal("0.05")

        if base > Decimal("0.99"):
            return Decimal("0.99")
        return base

    def _infer_regime_shift(self, snapshot: SpreadSnapshot) -> SpreadRegime:
        stats = snapshot.stats
        if stats is None or stats.zscore is None:
            return snapshot.regime

        if abs(stats.zscore) >= self._config.regime_shift_zscore_threshold:
            return SpreadRegime.DISLOCATED

        return snapshot.regime

    def _is_in_cooldown(self, signal_key: str) -> bool:
        last_time = self._last_signal_times.get(signal_key)
        if last_time is None:
            return False

        elapsed = (datetime.utcnow() - last_time).total_seconds()
        return elapsed < self._config.cooldown_seconds

    def _should_skip_emit(
        self,
        key: tuple[str, str, str],
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