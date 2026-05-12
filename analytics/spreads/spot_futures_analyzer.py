from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

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
    Production-grade analyzer для spot/futures spread analytics.

    Відповідальність:
    - слухати market.quote.updated;
    - слухати market.funding.updated;
    - кешувати spot/futures quotes і funding;
    - будувати SpreadSnapshot;
    - рахувати basis / spread_pct / spread_bps / funding-adjusted spread;
    - оновлювати rolling stats;
    - визначати spread regime;
    - генерувати SpreadSignal через SpreadSignalEngine;
    - публікувати analytics.spreads.spot_futures.updated;
    - публікувати analytics.spreads.signal.generated;
    - запускати cleanup/heartbeat через core.Scheduler.

    Не відповідає за:
    - отримання raw market-data з бірж;
    - execution;
    - risk approval;
    - strategy decisions;
    - storage напряму.
    """

    DEFAULT_QUOTE_TOPIC = "market.quote.updated"
    DEFAULT_FUNDING_TOPIC = "market.funding.updated"
    DEFAULT_SNAPSHOT_TOPIC = "analytics.spreads.spot_futures.updated"
    DEFAULT_SIGNAL_TOPIC = "analytics.spreads.signal.generated"

    def __init__(
        self,
        *,
        config: SpotFuturesSpreadConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            service_name=config.service_name,
        )
        self._config: SpotFuturesSpreadConfig = config

        self._spot_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._futures_quotes: dict[tuple[str, str], QuoteSnapshot] = {}
        self._funding: dict[tuple[str, str], FundingSnapshot] = {}

        self._spread_windows: dict[tuple[str, str, str], RollingDecimalWindow] = {}
        self._latest_snapshots: dict[tuple[str, str, str], SpreadSnapshot] = {}

        self._stats.update(
            {
                "quote_events_received": 0,
                "funding_events_received": 0,
                "quotes_received": 0,
                "funding_updates": 0,
                "invalid_payloads": 0,
                "invalid_quotes": 0,
                "incomplete_quotes": 0,
                "stale_quotes": 0,
                "unaligned_quotes": 0,
                "quotes_stored": 0,
                "funding_stored": 0,
                "snapshots_built": 0,
                "snapshots_skipped": 0,
                "signals_built": 0,
                "cleanup_runs": 0,
                "cleanup_removed_quotes": 0,
                "cleanup_removed_funding": 0,
                "cleanup_removed_snapshots": 0,
                "cleanup_removed_windows": 0,
            }
        )

    # ------------------------------------------------------------------
    # Registration / lifecycle hooks
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscriptions.

        Важливо:
        - register() синхронний, бо core.EventBus.subscribe() синхронний;
        - handlers приймають core.event_bus.Event;
        - повторний register() не дублює subscriptions.
        """
        if self._registered:
            return

        self._subscribe(
            self._topic("quote_event_topic", self.DEFAULT_QUOTE_TOPIC),
            self.on_quote_update,
            name=f"{self._service_name}.on_quote_update",
        )
        self._subscribe(
            self._topic("funding_event_topic", self.DEFAULT_FUNDING_TOPIC),
            self.on_funding_update,
            name=f"{self._service_name}.on_funding_update",
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
            "running": self.is_running,
            "registered": self.is_registered,
            "enabled": self._config.enabled,
            "spot_quotes_cached": len(self._spot_quotes),
            "futures_quotes_cached": len(self._futures_quotes),
            "funding_cached": len(self._funding),
            "active_windows": len(self._spread_windows),
            "latest_snapshots": len(self._latest_snapshots),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_quote_update(self, event: Event) -> None:
        """
        Handler для market.quote.updated.

        core.EventBus завжди передає Event, тому payload дістаємо з event.payload.
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
                validity = validate_quote_snapshot(
                    normalized_quote,
                    max_age_ms=self._config.max_quote_age_ms,
                )

                if validity == QuoteValidity.INVALID:
                    self._stats["invalid_quotes"] += 1
                    self._logger.debug(
                        "Rejected invalid quote | exchange=%s symbol=%s instrument_type=%s",
                        normalized_quote.exchange,
                        normalized_quote.symbol,
                        normalized_quote.instrument_type.value,
                        extra={
                            "exchange": normalized_quote.exchange,
                            "symbol": normalized_quote.symbol,
                            "event_type": "analytics.spreads.invalid_quote",
                        },
                    )
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
                    "Failed to process spot/futures quote update",
                    exc,
                    exchange=getattr(payload, "exchange", None),
                    symbol=getattr(payload, "symbol", None),
                    event_id=event.event_id,
                    correlation_id=event.correlation_id,
                )

    async def on_funding_update(self, event: Event) -> None:
        """
        Handler для market.funding.updated.
        """
        if not self.is_running or not self._config.enabled:
            return

        self._stats["funding_events_received"] += 1

        payload = event.payload
        if not isinstance(payload, FundingSnapshot):
            self._stats["invalid_payloads"] += 1
            self._logger.warning(
                "Invalid funding event payload | expected=%s actual=%s topic=%s",
                "FundingSnapshot",
                payload.__class__.__name__ if payload is not None else "None",
                event.topic,
            )
            return

        async with self._lock:
            try:
                self._stats["funding_updates"] += 1

                normalized_funding = FundingSnapshot(
                    exchange=normalize_exchange(payload.exchange),
                    symbol=normalize_symbol(payload.symbol),
                    funding_rate=payload.funding_rate,
                    timestamp=payload.timestamp,
                    next_funding_time=payload.next_funding_time,
                    predicted_rate=payload.predicted_rate,
                    interval_hours=payload.interval_hours,
                    metadata=dict(payload.metadata),
                )

                if not self._is_allowed_futures_exchange(normalized_funding.exchange):
                    return

                key = (normalized_funding.exchange, normalized_funding.symbol)
                self._funding[key] = normalized_funding
                self._stats["funding_stored"] += 1

                await self._recalculate_for_funding(
                    normalized_funding,
                    correlation_id=event.correlation_id,
                    source_event_id=event.event_id,
                )

            except Exception as exc:
                self._mark_exception(
                    "Failed to process funding update",
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

        Важливо:
        - _spread_windows не мають власного timestamp;
        - тому TTL для них застосовується опосередковано через _latest_snapshots;
        - якщо snapshot був видалений як stale, відповідне rolling window
          теж не повинно залишатися активним.
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

            removed_spot = self._cleanup_quote_cache(self._spot_quotes, now=now, ttl=ttl)
            removed_futures = self._cleanup_quote_cache(self._futures_quotes, now=now, ttl=ttl)
            removed_funding = self._cleanup_funding_cache(now=now, ttl=ttl)
            removed_snapshots = self._cleanup_snapshot_cache(now=now, ttl=ttl)

            removed_orphan_windows = self._cleanup_orphan_windows()
            removed_limit_windows = self._enforce_window_cache_limit()
            removed_windows = removed_orphan_windows + removed_limit_windows

            self._stats["cleanup_runs"] += 1
            self._stats["cleanup_removed_quotes"] += removed_spot + removed_futures
            self._stats["cleanup_removed_funding"] += removed_funding
            self._stats["cleanup_removed_snapshots"] += removed_snapshots
            self._stats["cleanup_removed_windows"] += removed_windows

            if removed_spot or removed_futures or removed_funding or removed_snapshots or removed_windows:
                self._logger.debug(
                    "Spot/futures cleanup completed | spot=%s futures=%s funding=%s "
                    "snapshots=%s windows=%s orphan_windows=%s limit_windows=%s",
                    removed_spot,
                    removed_futures,
                    removed_funding,
                    removed_snapshots,
                    removed_windows,
                    removed_orphan_windows,
                    removed_limit_windows,
                )

    async def emit_heartbeat(self) -> None:
        """
        Periodic heartbeat для dashboard/monitoring.
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
        key = (quote.exchange, quote.symbol)

        if quote.instrument_type == InstrumentType.SPOT:
            if not self._is_allowed_spot_exchange(quote.exchange):
                return
            self._spot_quotes[key] = quote
            self._stats["quotes_stored"] += 1
            self._enforce_quote_cache_limit(self._spot_quotes)
            return

        if quote.instrument_type in InstrumentType.derivatives():
            if not self._is_allowed_futures_exchange(quote.exchange):
                return
            self._futures_quotes[key] = quote
            self._stats["quotes_stored"] += 1
            self._enforce_quote_cache_limit(self._futures_quotes)

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
        symbol = quote.symbol

        if quote.instrument_type == InstrumentType.SPOT:
            futures_candidates = [
                fut
                for (exchange, sym), fut in self._futures_quotes.items()
                if sym == symbol and self._is_allowed_futures_exchange(exchange)
            ]

            for futures_quote in futures_candidates:
                await self._try_build_and_publish(
                    quote,
                    futures_quote,
                    correlation_id=correlation_id,
                    source_event_id=source_event_id,
                )
            return

        if quote.instrument_type in InstrumentType.derivatives():
            if not self._is_allowed_futures_exchange(quote.exchange):
                return

            spot_candidates = [
                spot
                for (exchange, sym), spot in self._spot_quotes.items()
                if sym == symbol and self._is_allowed_spot_exchange(exchange)
            ]

            for spot_quote in spot_candidates:
                await self._try_build_and_publish(
                    spot_quote,
                    quote,
                    correlation_id=correlation_id,
                    source_event_id=source_event_id,
                )

    async def _recalculate_for_funding(
        self,
        funding: FundingSnapshot,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> None:
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
            await self._try_build_and_publish(
                spot_quote,
                futures_quote,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
            )

    async def _try_build_and_publish(
        self,
        spot_quote: QuoteSnapshot,
        futures_quote: QuoteSnapshot,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
    ) -> None:
        if spot_quote.instrument_type != InstrumentType.SPOT:
            return

        if futures_quote.instrument_type not in InstrumentType.derivatives():
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
            self._stats["snapshots_skipped"] += 1
            return

        key = self._snapshot_key(snapshot)
        previous_snapshot = self._latest_snapshots.get(key)

        if self._should_skip_emit(key, snapshot.timestamp):
            self._stats["emit_skips"] += 1
            return

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
                "spread_type": SpreadType.SPOT_FUTURES.value,
            },
        )

        signals = self._evaluate_snapshot_signals(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
        )
        self._stats["signals_built"] += len(signals)

        await self._publish_signals(
            self._topic("signal_event_topic", self.DEFAULT_SIGNAL_TOPIC),
            signals,
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers={
                "source_event_id": source_event_id,
                "spread_type": SpreadType.SPOT_FUTURES.value,
            },
        )

    # ------------------------------------------------------------------
    # Snapshot building
    # ------------------------------------------------------------------

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
        window = self._get_or_create_window(window_key)
        window.append(raw_spread)

        stats = window.stats()
        regime_result = self._regime_detector.detect_from_stats(stats)

        snapshot_timestamp = max(spot_quote.timestamp, futures_quote.timestamp)

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
            timestamp=snapshot_timestamp,
            metadata={
                "spot_age_ms": quote_age_ms(spot_quote),
                "futures_age_ms": quote_age_ms(futures_quote),
                "funding_rate": str(funding_rate) if funding_rate is not None else None,
                "funding_timestamp": funding_snapshot.timestamp.isoformat()
                if funding_snapshot is not None
                else None,
                "spot_exchange": spot_exchange,
                "futures_exchange": futures_exchange,
                "spot_sequence_id": spot_quote.sequence_id,
                "futures_sequence_id": futures_quote.sequence_id,
                "regime": regime_result.regime.value,
                "regime_reason": regime_result.reason,
                "is_compressed": regime_result.is_compressed,
                "is_elevated": regime_result.is_elevated,
                "is_extreme": regime_result.is_extreme,
                "is_dislocated": regime_result.is_dislocated,
            },
        )

    def _get_or_create_window(
        self,
        key: tuple[str, str, str],
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

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _is_allowed_spot_exchange(self, exchange: str) -> bool:
        checker = getattr(self._config, "is_spot_exchange_allowed", None)
        if callable(checker):
            return bool(checker(exchange))

        if self._config.default_spot_exchange is None:
            return True

        return normalize_exchange(exchange) == normalize_exchange(self._config.default_spot_exchange)

    def _is_allowed_futures_exchange(self, exchange: str) -> bool:
        checker = getattr(self._config, "is_futures_exchange_allowed", None)
        if callable(checker):
            return bool(checker(exchange))

        if self._config.default_futures_exchange is None:
            return True

        return normalize_exchange(exchange) == normalize_exchange(self._config.default_futures_exchange)

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _cleanup_quote_cache(
        self,
        cache: dict[tuple[str, str], QuoteSnapshot],
        *,
        now: datetime,
        ttl: timedelta,
    ) -> int:
        stale_keys = [
            key
            for key, quote in cache.items()
            if now - quote.timestamp >= ttl
        ]

        for key in stale_keys:
            cache.pop(key, None)

        return len(stale_keys)

    def _cleanup_funding_cache(
        self,
        *,
        now: datetime,
        ttl: timedelta,
    ) -> int:
        stale_keys = [
            key
            for key, funding in self._funding.items()
            if now - funding.timestamp >= ttl
        ]

        for key in stale_keys:
            self._funding.pop(key, None)

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

    def _enforce_quote_cache_limit(
        self,
        cache: dict[tuple[str, str], QuoteSnapshot],
    ) -> None:
        max_items = self._config.max_cached_quotes
        if len(cache) <= max_items:
            return

        sorted_items = sorted(
            cache.items(),
            key=lambda item: item[1].timestamp,
        )

        overflow = len(cache) - max_items
        for key, _ in sorted_items[:overflow]:
            cache.pop(key, None)

    def _enforce_window_cache_limit(self) -> int:
        max_items = self._config.max_cached_windows
        if len(self._spread_windows) <= max_items:
            return 0

        overflow = len(self._spread_windows) - max_items
        keys_to_remove = list(self._spread_windows.keys())[:overflow]

        for key in keys_to_remove:
            self._spread_windows.pop(key, None)

        return len(keys_to_remove)

    def _drop_oldest_window(self) -> None:
        if not self._spread_windows:
            return

        oldest_key = next(iter(self._spread_windows))
        self._spread_windows.pop(oldest_key, None)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_key(snapshot: SpreadSnapshot) -> tuple[str, str, str]:
        return (
            snapshot.symbol,
            snapshot.leg_a_exchange,
            snapshot.leg_b_exchange,
        )

    @staticmethod
    def _topic(config_attr: str, fallback: str) -> str:
        """
        Placeholder for instance override safety.

        This method is intentionally static in signature style compatibility,
        but actual config access is implemented below through instance binding.
        """
        raise NotImplementedError("Use instance _topic method")

    def _topic(self, config_attr: str, fallback: str) -> str:  # type: ignore[no-redef]
        value = getattr(self._config, config_attr, None)
        if isinstance(value, str) and value.strip():
            return value
        return fallback