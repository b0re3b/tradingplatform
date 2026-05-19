from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import LargeTradeDetectorConfig
from analytics.whales.enums import WhaleComponentName, WhaleTradeSide
from analytics.whales.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LargeTradeSignal,
    SymbolStats,
    TradeRecord,
    WhaleKey,
    make_symbol_stats,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    whale_key_to_dict,
)


class LargeTradeDetector(BaseWhaleComponent):
    """
    Low-level detector для аномально великих трейдів.

    Production EventBus flow:
        exchange adapters
            -> market.trade
            -> TradesCache
            -> market.trades.updated
            -> LargeTradeDetector
            -> analytics.whales.large_trade

    Legacy/manual raw flow:
        market.trade -> LargeTradeDetector

    Raw flow дозволений тільки якщо:
        config.allow_legacy_raw_topics=True

    Direct режим для тестів/backtesting/replay:
        await process_trade_payload(payload)

    Scope:
        exchange + market_type + symbol + timeframe

    Важливо:
    - EventBus/Scheduler передаються через constructor dependency injection;
    - підписки виконуються через register() / EventBus.subscribe();
    - production subscriptions мають слухати data-layer updated topics;
    - cleanup запускається тільки через Scheduler.add_interval_job();
    - власних uncontrolled asyncio cleanup loops немає;
    - rolling state не змішує різні біржі / market_type / timeframe.
    """

    def __init__(
        self,
        *,
        config: LargeTradeDetectorConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
    ) -> None:
        super().__init__(
            component_name=WhaleComponentName.LARGE_TRADE_DETECTOR.value,
            event_bus=event_bus,
            scheduler=scheduler,
            default_exchange=config.default_exchange,
            default_market_type=config.default_market_type,
            default_timeframe=config.default_timeframe,
        )

        self.config = config
        self.config.validate()

        self._stats: dict[WhaleKey, SymbolStats] = {}
        self._state_locks: dict[WhaleKey, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Idempotent: повторний виклик не створює дублікати підписок.

        Production:
            config.production_input_topics -> market.trades.updated

        Legacy:
            config.raw_input_event_name -> market.trade
            тільки якщо config.allow_legacy_raw_topics=True
        """
        if self._registered:
            return

        if not self.config.enabled:
            self.logger.info(
                "LargeTradeDetector registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        self._subscribe_production_many(
            self.config.production_input_topics,
            self.handle_trade_event,
            name="analytics.whales.large_trade_detector.handle_trade_event",
        )

        if self.config.allow_legacy_raw_topics:
            self._subscribe_legacy_raw(
                self.config.raw_input_event_name,
                self.handle_raw_trade_event,
                name="analytics.whales.large_trade_detector.handle_raw_trade_event",
                allow_legacy_raw_topics=self.config.allow_legacy_raw_topics,
            )

        self._registered = True

    async def start(self) -> None:
        if self._started:
            self.logger.warning("LargeTradeDetector already started")
            return

        if not self.config.enabled:
            self.logger.info("LargeTradeDetector is disabled by config")
            return

        await self.register()

        self._add_interval_job(
            name="analytics.whales.large_trade_detector.cleanup",
            func=self.cleanup,
            interval=self.config.cleanup_interval_sec,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=min(30.0, max(1.0, self.config.cleanup_interval_sec)),
            allow_overlap=False,
            enabled=True,
        )

        self._started = True

        self.logger.info(
            "LargeTradeDetector started",
            extra={
                "component": self.component_name,
                "production_input_topics": list(self.config.production_input_topics),
                "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
                "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
                "output_event_name": self.config.output_event_name,
                "rolling_window_size": self.config.rolling_window_size,
                "zscore_threshold": self.config.zscore_threshold,
                "default_abs_notional_threshold": self.config.default_abs_notional_threshold,
                "recalibration_interval": self.config.recalibration_interval,
                "cleanup_interval_sec": self.config.cleanup_interval_sec,
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def stop(self) -> None:
        if not self._started and not self._registered:
            return

        await super().stop()

        self.logger.info(
            "LargeTradeDetector stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # EventBus handlers
    # =========================================================================

    async def handle_trade_event(self, event: Event) -> None:
        """
        Production EventBus handler для market.trades.updated.

        Core EventBus передає core.event_bus.Event, а бізнес-логіка нижче
        працює з dict payload.
        """
        await self._handle_trade_event(
            event,
            allow_raw_payload=False,
        )

    async def handle_raw_trade_event(self, event: Event) -> None:
        """
        Legacy/raw EventBus handler для market.trade.

        Використовувати тільки для migration/test/manual режиму.
        """
        if not self.config.allow_legacy_raw_topics:
            self.logger.warning(
                "Raw trade event skipped: legacy raw topics are disabled",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                },
            )
            return

        await self._handle_trade_event(
            event,
            allow_raw_payload=True,
        )

    async def _handle_trade_event(
        self,
        event: Event,
        *,
        allow_raw_payload: bool,
    ) -> None:
        try:
            payload = self._payload_from_event(event)

            await self.process_trades_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
                allow_raw_payload=allow_raw_payload,
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing trade event",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                    "allow_raw_payload": allow_raw_payload,
                },
            )

    # =========================================================================
    # Public processing API
    # =========================================================================

    async def process_trades_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
        allow_raw_payload: bool = False,
    ) -> list[LargeTradeSignal]:
        """
        Основний production-safe метод обробки trade update payload.

        Production payload очікується з TradesCache / data-layer:
            market.trades.updated

        На відміну від старого single-trade API, цей метод коректно обробляє
        batch payload-и:
            {"trades": [{...}, {...}]}
            {"data": {"trades": [{...}, {...}]}}
            {"data": [{...}, {...}]}
            {"trade": {...}}
            plain trade dict

        Повертає всі LargeTradeSignal, згенеровані з одного EventBus update.
        """
        if not self.config.enabled:
            return []

        trades = self._normalize_trade_payloads(
            payload,
            allow_raw_payload=allow_raw_payload,
            source_topic=source_topic,
        )
        if not trades:
            return []

        signals: list[LargeTradeSignal] = []

        for trade in trades:
            signal = await self._process_trade_record(
                trade,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
                source_topic=source_topic,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    async def process_trade_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
        allow_raw_payload: bool = False,
    ) -> LargeTradeSignal | None:
        """
        Backward-compatible single-result API.

        Новий production EventBus шлях має використовувати process_trades_payload(),
        бо market.trades.updated може містити batch trades. Цей метод залишений
        для tests/backtesting/replay і старого коду: якщо payload містить batch,
        він обробляє весь batch, але повертає перший згенерований сигнал або None.
        """
        signals = await self.process_trades_payload(
            payload,
            correlation_id=correlation_id,
            source_event_id=source_event_id,
            source_topic=source_topic,
            allow_raw_payload=allow_raw_payload,
        )
        return signals[0] if signals else None

    async def process_trade(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> LargeTradeSignal | None:
        """
        Backward-compatible alias для старого direct API.

        Новий код має використовувати process_trade_payload() для single-result
        сумісності або process_trades_payload() для batch-aware обробки.
        """
        return await self.process_trade_payload(event)

    # =========================================================================
    # Core detection logic
    # =========================================================================

    async def _process_trade_record(
        self,
        trade: TradeRecord,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> LargeTradeSignal | None:
        """
        Обробляє вже нормалізований TradeRecord.

        Винесено окремо, щоб batch path і backward-compatible single path
        використовували одну й ту саму detection / cooldown / emit логіку.
        """
        if not self.config.should_process_key(trade.key):
            return None

        if not self._passes_basic_filters(trade):
            return None

        stats, state_lock = await self._get_or_create_key_state(trade)

        signal: LargeTradeSignal | None = None

        async with state_lock:
            mean_before = stats.mean()
            std_before = stats.std()

            abs_threshold = self._get_abs_threshold(trade.key)
            zscore = self._calculate_zscore(
                value=trade.notional,
                mean=mean_before,
                std=std_before,
            )

            absolute_triggered = trade.notional >= abs_threshold
            relative_triggered = self._is_relative_trigger(
                zscore=zscore,
                sample_size=stats.sample_size,
            )

            if absolute_triggered or relative_triggered:
                if self._passes_key_cooldown_for_stats(stats, trade.key):
                    signal = LargeTradeSignal.from_trade(
                        trade=trade,
                        abs_threshold=abs_threshold,
                        mean_notional=mean_before,
                        std_notional=std_before,
                        zscore=zscore,
                        absolute_triggered=absolute_triggered,
                        relative_triggered=relative_triggered,
                    )
                    stats.signals_emitted += 1
                    stats.last_signal_ts_monotonic = time.monotonic()

            stats.add(
                trade.notional,
                recalibration_interval=self.config.recalibration_interval,
            )
            stats.trades_processed += 1

        if signal is not None:
            if self.config.log_signals:
                self.logger.info(
                    "Large trade detected",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "exchange_symbol": signal.exchange_symbol,
                        "side": signal.side,
                        "notional": signal.notional,
                        "zscore": signal.zscore,
                        "trigger_type": signal.trigger_type,
                        "trade_id": signal.trade_id,
                        "source_topic": source_topic,
                        "source_event_id": source_event_id,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            await self._emit_signal(
                signal,
                correlation_id=correlation_id,
                source_event_id=source_event_id,
                source_topic=source_topic,
            )

        return signal

    def _normalize_trade_payloads(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
        *,
        allow_raw_payload: bool,
        source_topic: str | None = None,
    ) -> list[TradeRecord]:
        """
        Нормалізація event/update payload у список TradeRecord.

        Production підтримує data-layer payload-и:
        - {"trade": {...}}
        - {"trades": [{...}, ...]}
        - {"data": {"trade": {...}}}
        - {"data": {"trades": [{...}, ...]}}
        - {"data": [{...}, ...]}
        - plain trade dict

        Event-level поля exchange/market_type/symbol/timeframe використовуються
        як fallback для кожного child trade з batch payload.
        """
        try:
            event = dict(event_payload)

            if self._is_raw_topic(source_topic) and not allow_raw_payload:
                self.logger.warning(
                    "Trade payload dropped: raw topic is not allowed in production path",
                    extra={
                        "component": self.component_name,
                        "source_topic": source_topic,
                    },
                )
                return []

            raw_payloads = self._extract_trade_payloads(event)
            if not raw_payloads:
                self.logger.debug(
                    "Trade event dropped: cannot extract trade payloads",
                    extra={
                        "component": self.component_name,
                        "payload_keys": list(event.keys()),
                    },
                )
                return []

            trades: list[TradeRecord] = []
            for raw_payload in raw_payloads:
                trade = self._normalize_single_trade_payload(
                    event,
                    raw_payload,
                    source_topic=source_topic,
                )
                if trade is not None:
                    trades.append(trade)

            return trades

        except Exception:
            self.logger.exception(
                "Failed to normalize trade payloads",
                extra={
                    "component": self.component_name,
                    "source_topic": source_topic,
                },
            )
            return []

    def _normalize_trade_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
        *,
        allow_raw_payload: bool,
        source_topic: str | None = None,
    ) -> TradeRecord | None:
        """
        Backward-compatible single-trade normalizer.

        Якщо payload містить batch, повертає останній валідний TradeRecord,
        щоб не ламати старий код, який очікував single object.
        Production detection path використовує _normalize_trade_payloads().
        """
        trades = self._normalize_trade_payloads(
            event_payload,
            allow_raw_payload=allow_raw_payload,
            source_topic=source_topic,
        )
        return trades[-1] if trades else None

    def _normalize_single_trade_payload(
        self,
        event: Mapping[str, Any],
        raw_payload: Mapping[str, Any],
        *,
        source_topic: str | None = None,
    ) -> TradeRecord | None:
        """
        Нормалізація одного raw trade payload у TradeRecord.

        Підтримує поля:
        - exchange;
        - market_type;
        - symbol / s / instrument;
        - exchange_symbol;
        - timeframe;
        - price / p;
        - quantity / qty / q / size;
        - side / S / maker_side / direction / m;
        - timestamp_ms / timestamp / ts / T / E.
        """
        try:
            payload = dict(raw_payload)

            symbol = self._extract_symbol_from_trade_payload(payload, event)
            if not symbol:
                self.logger.debug(
                    "Trade event dropped: missing symbol",
                    extra={"component": self.component_name},
                )
                return None

            price = self._safe_float(payload.get("price") or payload.get("p"))
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
                or payload.get("maker_side")
                or payload.get("direction"),
                maker_flag=payload.get("m"),
            )
            timestamp_ms = self._extract_timestamp_ms(payload)

            if price is None or price <= 0:
                self.logger.debug(
                    "Trade event dropped: invalid price",
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                        "price": price,
                    },
                )
                return None

            if quantity is None or quantity <= 0:
                self.logger.debug(
                    "Trade event dropped: invalid quantity",
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                        "quantity": quantity,
                    },
                )
                return None

            if side == WhaleTradeSide.UNKNOWN.value:
                self.logger.debug(
                    "Trade event dropped: invalid side",
                    extra={
                        "component": self.component_name,
                        "symbol": symbol,
                    },
                )
                return None

            trade_id = self._safe_str(
                payload.get("trade_id")
                or payload.get("id")
                or payload.get("t")
            )

            exchange = self._safe_str(
                payload.get("exchange")
                or event.get("exchange")
                or self.default_exchange
            )
            market_type = self._safe_str(
                payload.get("market_type")
                or event.get("market_type")
                or self.default_market_type
            )
            timeframe = self._safe_str(
                payload.get("timeframe")
                or event.get("timeframe")
                or self.default_timeframe
            )
            exchange_symbol = self._safe_str(
                payload.get("exchange_symbol")
                or event.get("exchange_symbol")
                or payload.get("raw_symbol")
                or payload.get("s")
            )

            normalized_exchange = normalize_exchange(exchange)
            normalized_market_type = normalize_market_type(market_type)
            normalized_timeframe = normalize_timeframe(timeframe)
            normalized_symbol = normalize_symbol(symbol)
            normalized_exchange_symbol = normalize_exchange_symbol(
                exchange_symbol,
                fallback_symbol=normalized_symbol,
            )

            metadata = dict(event.get("metadata") or {})
            metadata.update(dict(payload.get("metadata") or {}))
            metadata.update(
                {
                    "source_topic": source_topic,
                    "payload_source": event.get("source"),
                }
            )

            return TradeRecord(
                symbol=normalized_symbol,
                price=price,
                quantity=quantity,
                side=side,
                timestamp_ms=timestamp_ms,
                trade_id=trade_id,
                exchange=normalized_exchange,
                market_type=normalized_market_type,
                timeframe=normalized_timeframe,
                exchange_symbol=normalized_exchange_symbol,
                raw_event=dict(payload),
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize single trade payload",
                extra={
                    "component": self.component_name,
                    "source_topic": source_topic,
                },
            )
            return None

    @staticmethod
    def _extract_trade_payloads(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """
        Витягує всі trade payload-и з data-layer event.

        Підтримує:
        - {"trade": {...}}
        - {"trades": [{...}, ...]}
        - {"data": {"trade": {...}}}
        - {"data": {"trades": [{...}, ...]}}
        - {"data": [{...}, ...]}
        - plain trade dict
        """
        trade = event.get("trade")
        if isinstance(trade, Mapping):
            return [trade]

        trades = event.get("trades")
        if isinstance(trades, list):
            return [item for item in trades if isinstance(item, Mapping)]

        data = event.get("data")
        if isinstance(data, Mapping):
            nested_trade = data.get("trade")
            if isinstance(nested_trade, Mapping):
                return [nested_trade]

            nested_trades = data.get("trades")
            if isinstance(nested_trades, list):
                return [item for item in nested_trades if isinstance(item, Mapping)]

            if "price" in data or "p" in data:
                return [data]

        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]

        if "price" in event or "p" in event:
            return [event]

        return []

    @staticmethod
    def _extract_trade_payload(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Backward-compatible single payload extractor.

        Якщо event містить batch trades, повертає останній trade, як і стара
        реалізація. Новий production path використовує _extract_trade_payloads().
        """
        payloads = LargeTradeDetector._extract_trade_payloads(event)
        return payloads[-1] if payloads else None

    def _passes_basic_filters(self, trade: TradeRecord) -> bool:
        if trade.notional < self.config.min_notional_filter:
            return False

        if self.config.side_filter is not None and trade.side != self.config.side_filter:
            return False

        return True

    @staticmethod
    def _calculate_zscore(
        *,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def _is_relative_trigger(
        self,
        *,
        zscore: float,
        sample_size: int,
    ) -> bool:
        if not self.config.use_relative_detection:
            return False

        if sample_size < self.config.min_samples_for_relative_detection:
            return False

        return zscore >= self.config.zscore_threshold

    def _get_abs_threshold(self, key: WhaleKey) -> float:
        getter = getattr(self.config, "get_key_abs_threshold", None)
        if callable(getter):
            return float(getter(key))

        scope = whale_key_to_dict(key)
        return float(self.config.get_symbol_abs_threshold(scope["symbol"]))

    def _passes_key_cooldown_for_stats(self, stats: SymbolStats, key: WhaleKey) -> bool:
        getter = getattr(self.config, "get_key_cooldown", None)
        cooldown = float(getter(key)) if callable(getter) else self.config.signal_cooldown_sec

        return self._passes_cooldown(
            stats.last_signal_ts_monotonic,
            cooldown,
        )

    async def _emit_signal(
        self,
        signal: LargeTradeSignal,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> None:
        if not self.config.emit_on_bus:
            return

        headers: dict[str, Any] = {
            "scope": str(whale_key_to_dict(signal.key)),
        }
        if source_event_id is not None:
            headers["source_event_id"] = source_event_id
        if source_topic is not None:
            headers["source_topic"] = source_topic

        await self._emit(
            self.config.output_event_name,
            signal,
            priority=EventPriority.NORMAL,
            source=self.component_name,
            correlation_id=correlation_id,
            headers=headers,
        )

    # =========================================================================
    # Scoped state management
    # =========================================================================

    async def _get_or_create_key_state(
        self,
        trade: TradeRecord,
    ) -> tuple[SymbolStats, asyncio.Lock]:
        key = trade.key

        stats = self._stats.get(key)
        lock = self._state_locks.get(key)

        if stats is not None and lock is not None:
            return stats, lock

        async with self._registry_lock:
            stats = self._stats.get(key)
            if stats is None:
                stats = make_symbol_stats(
                    self.config.rolling_window_size,
                    exchange=trade.exchange,
                    market_type=trade.market_type,
                    symbol=trade.symbol,
                    timeframe=trade.timeframe,
                    exchange_symbol=trade.exchange_symbol,
                )
                self._stats[key] = stats

            lock = self._state_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._state_locks[key] = lock

            return stats, lock

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """
        Видаляє неактивні scoped states.

        Запускається через core Scheduler.add_interval_job().
        """
        now_mono = time.monotonic()
        ttl = self.config.stats_ttl_sec

        if ttl <= 0:
            return

        async with self._registry_lock:
            stale_keys = [
                key
                for key, stats in self._stats.items()
                if (now_mono - stats.last_update_ts_monotonic) >= ttl
            ]

            for key in stale_keys:
                self._stats.pop(key, None)
                self._state_locks.pop(key, None)

        if stale_keys:
            self.logger.info(
                "Cleaned stale LargeTradeDetector scoped states",
                extra={
                    "component": self.component_name,
                    "removed_states_count": len(stale_keys),
                    "removed_scopes": [
                        whale_key_to_dict(key)
                        for key in stale_keys[:20]
                    ],
                },
            )

    # =========================================================================
    # Public state / stats API
    # =========================================================================

    def get_key_stats(self, key: WhaleKey) -> dict[str, Any]:
        stats = self._stats.get(key)
        scope = whale_key_to_dict(key)

        if stats is None:
            return {
                **scope,
                "scope": scope,
                "exists": False,
            }

        return {
            **scope,
            "scope": scope,
            "exists": True,
            **stats.to_dict(),
        }

    def get_symbol_stats(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """
        Backward-compatible read API.

        Якщо exchange/market_type/timeframe передані — повертає scoped state.
        Якщо ні — повертає агрегований список state-ів для symbol.
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ValueError:
            return {
                "symbol": symbol,
                "exists": False,
                "error": "invalid_symbol",
            }

        if exchange is not None or market_type is not None or timeframe is not None:
            key = self.make_key(
                exchange=exchange or self.default_exchange,
                market_type=market_type or self.default_market_type,
                symbol=normalized_symbol,
                timeframe=timeframe or self.default_timeframe,
            )
            return self.get_key_stats(key)

        matching = [
            stats.to_dict()
            for key, stats in self._stats.items()
            if whale_key_to_dict(key)["symbol"] == normalized_symbol
        ]

        return {
            "symbol": normalized_symbol,
            "exists": bool(matching),
            "scopes": matching,
        }

    def get_all_stats(self) -> dict[str, Any]:
        return {
            self.scoped_mapping_key(key): stats.to_dict()
            for key, stats in self._stats.items()
        }

    async def reset_key(self, key: WhaleKey) -> None:
        async with self._registry_lock:
            self._stats.pop(key, None)
            self._state_locks.pop(key, None)

        self.logger.info(
            "Reset LargeTradeDetector scoped state",
            extra={
                "component": self.component_name,
                "scope": whale_key_to_dict(key),
            },
        )

    async def reset_symbol(
        self,
        symbol: str,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        """
        Backward-compatible reset API.

        Якщо exchange/market_type/timeframe передані — reset одного key.
        Якщо ні — reset усіх state-ів для symbol.
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ValueError:
            return

        async with self._registry_lock:
            if exchange is not None or market_type is not None or timeframe is not None:
                key = self.make_key(
                    exchange=exchange or self.default_exchange,
                    market_type=market_type or self.default_market_type,
                    symbol=normalized_symbol,
                    timeframe=timeframe or self.default_timeframe,
                )
                removed_keys = [key]
            else:
                removed_keys = [
                    key
                    for key in self._stats.keys()
                    if whale_key_to_dict(key)["symbol"] == normalized_symbol
                ]

            for key in removed_keys:
                self._stats.pop(key, None)
                self._state_locks.pop(key, None)

        self.logger.info(
            "Reset LargeTradeDetector symbol state",
            extra={
                "component": self.component_name,
                "symbol": normalized_symbol,
                "removed_states_count": len(removed_keys),
            },
        )

    async def reset_all(self) -> None:
        async with self._registry_lock:
            self._stats.clear()
            self._state_locks.clear()

        self.logger.info(
            "Reset all LargeTradeDetector states",
            extra={"component": self.component_name},
        )

    def get_healthcheck(self) -> dict[str, Any]:
        health = super().get_healthcheck()
        health.update(
            {
                "enabled": self.config.enabled,
                "tracked_scopes": len(self._stats),
                "state_locks": len(self._state_locks),
                "locking": "per_whale_key",
                "production_input_topics": list(self.config.production_input_topics),
                "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
                "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
                "output_event_name": self.config.output_event_name,
                "scope": "exchange:market_type:symbol:timeframe",
            }
        )
        return health

    # =========================================================================
    # Parsing / normalization helpers
    # =========================================================================

    def _extract_symbol_from_trade_payload(
        self,
        payload: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> str | None:
        """
        Витягує symbol для одного child trade payload.

        Важливе правило для batch payload-ів:
        - якщо child trade явно містить symbol/s/instrument, але значення пусте
          або невалідне — trade відкидається;
        - fallback на event-level symbol дозволений тільки якщо child trade
          взагалі не містить жодного symbol-поля.

        Це не дозволяє невалідному child trade на кшталт {"symbol": ""}
        випадково стати валідним через parent batch symbol.
        """
        child_symbol_keys = ("symbol", "s", "instrument")

        for key in child_symbol_keys:
            if key in payload:
                return self._normalize_symbol(payload.get(key))

        return self._normalize_symbol(event.get("symbol"))

    @staticmethod
    def _normalize_symbol(value: Any) -> str | None:
        try:
            return normalize_symbol(value)
        except ValueError:
            return None

    @staticmethod
    def _normalize_side(
        value: Any,
        maker_flag: Any = None,
    ) -> str:
        side = WhaleTradeSide.normalize(value)

        if side is not WhaleTradeSide.UNKNOWN:
            return side.value

        return WhaleTradeSide.from_maker_flag(maker_flag).value

    @staticmethod
    def _extract_timestamp_ms(payload: Mapping[str, Any]) -> int:
        raw_ts = (
            payload.get("timestamp_ms")
            or payload.get("timestamp")
            or payload.get("ts")
            or payload.get("T")
            or payload.get("E")
        )

        if raw_ts is None:
            return int(time.time() * 1000)

        if isinstance(raw_ts, datetime):
            if raw_ts.tzinfo is None:
                raw_ts = raw_ts.replace(tzinfo=timezone.utc)
            return int(raw_ts.timestamp() * 1000)

        if isinstance(raw_ts, (int, float)):
            # seconds, not milliseconds
            if raw_ts < 10_000_000_000:
                return int(raw_ts * 1000)
            return int(raw_ts)

        if isinstance(raw_ts, str):
            raw_ts = raw_ts.strip()

            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass

            try:
                numeric = float(raw_ts)
                if numeric < 10_000_000_000:
                    return int(numeric * 1000)
                return int(numeric)
            except Exception:
                pass

        return int(time.time() * 1000)

    @staticmethod
    def _is_raw_topic(source_topic: str | None) -> bool:
        return source_topic in {"market.trade", "market.liquidation"}

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        if result != result:  # NaN
            return None

        return result

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        if value is None:
            return None

        text = str(value).strip()
        return text or None


__all__ = [
    "LargeTradeDetector",
]