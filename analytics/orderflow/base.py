from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Mapping

from core.event_bus import Event, EventBus, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import BaseOrderFlowSubConfig
from .enums import (
    OrderFlowMetricType,
    OrderFlowSide,
    OrderFlowSignalType,
    OrderFlowSourceType,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    BaseOrderFlowStats,
    NormalizedTrade,
    OrderFlowKey,
    OrderFlowSignal,
    OrderFlowUpdate,
    OrderbookSnapshot,
    make_orderflow_key,
    orderflow_key_to_dict,
    signal_to_dict,
    update_to_dict,
)


class BaseOrderFlowAnalyzer(ABC):
    """
    Base class for all analytics.orderflow analyzers.

    Responsibilities:
    - EventBus subscription lifecycle via register()/stop();
    - Scheduler-based health and cleanup jobs;
    - shared metrics;
    - signal throttling;
    - update/signal EventBus publishing;
    - common extraction and normalization helpers;
    - futures scope handling.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct input flow:
        exchange adapters
            -> data caches
            -> market.trades.updated / market.orderbook.updated
            -> analytics.orderflow analyzer
            -> analytics.orderflow.*

    Concrete analyzers should implement:
    - process_key()
    - get_latest_stats_by_key()
    - _handle_event()

    Backward-compatible methods process_symbol() / get_latest_stats() are kept
    as wrappers for gradual migration, but new code should use scoped methods.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseOrderFlowSubConfig,
        metric_type: OrderFlowMetricType,
        source_type: OrderFlowSourceType,
        scheduler: Scheduler | None = None,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
        component_module: str = "orderflow",
        default_exchange: str | None = None,
        default_market_type: str = DEFAULT_MARKET_TYPE,
        default_timeframe: str = DEFAULT_TIMEFRAME,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._metric_type = metric_type
        self._source_type = source_type
        self._source_topic_patterns = list(source_topic_patterns or ())

        self._default_exchange = (
            str(default_exchange).strip().lower()
            if default_exchange
            else None
        )
        self._default_market_type = self._normalize_market_type(default_market_type)
        self._default_timeframe = self._normalize_timeframe(default_timeframe)

        self._logger = get_logger(
            __name__,
            service_name=self._config.source_name,
            component="analytics",
            component_module=component_module,
            metric=self._metric_type.value,
            source_type=self._source_type.value,
        )

        self._subscriptions: list[Subscription] = []
        self._running = False
        self._lock = asyncio.Lock()

        self._health_job_id: str | None = None
        self._cleanup_job_id: str | None = None

        self._last_signal_ts_by_key: dict[OrderFlowKey, float] = {}
        self._metrics: dict[str, Any] = self._build_initial_metrics()

        self._validate_config()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register analyzer subscriptions and scheduler jobs.

        This is the standard lifecycle entrypoint used by the project.
        start() is kept as a compatibility alias.
        """
        if self._running:
            self._logger.warning("%s already registered", self.__class__.__name__)
            return

        if not self._config.enabled:
            self._logger.warning("%s is disabled by config", self.__class__.__name__)
            return

        if not self._source_topic_patterns:
            self._logger.warning(
                "%s has no source topic patterns; no EventBus subscriptions created",
                self.__class__.__name__,
            )

        for pattern in self._source_topic_patterns:
            subscription = self._event_bus.subscribe(
                pattern=pattern,
                handler=self._handle_event,
                name=f"{self.__class__.__name__}:{pattern}",
            )
            self._subscriptions.append(subscription)

        self._register_scheduler_jobs()
        self._running = True

        self._logger.info(
            "%s registered | metric=%s source_type=%s patterns=%s scope=%s",
            self.__class__.__name__,
            self._metric_type.value,
            self._source_type.value,
            self._source_topic_patterns,
            "exchange:market_type:symbol:timeframe",
        )

    def start(self) -> None:
        """
        Backward-compatible alias.

        New modules should call register().
        """
        self.register()

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("%s already stopped", self.__class__.__name__)
            return

        for subscription in list(self._subscriptions):
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception:
                self._logger.exception(
                    "Failed to unsubscribe handler | analyzer=%s pattern=%s",
                    self.__class__.__name__,
                    getattr(subscription, "pattern", None),
                )

        self._subscriptions.clear()
        self._disable_scheduler_jobs()
        self._running = False

        self._logger.info("%s stopped", self.__class__.__name__)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    async def process_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        """
        Process one scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_stats_by_key(self, key: OrderFlowKey) -> BaseOrderFlowStats | None:
        """
        Return latest stats for one scoped futures market.
        """
        raise NotImplementedError

    @abstractmethod
    async def _handle_event(self, event: Event) -> None:
        raise NotImplementedError

    async def process_symbol(self, symbol: str) -> BaseOrderFlowStats | None:
        """
        Backward-compatible wrapper.

        This requires default_exchange to be configured because symbol-only
        processing is unsafe in multi-exchange futures mode.
        """
        if not self._default_exchange:
            raise ValueError(
                "process_symbol(symbol) requires default_exchange. "
                "Use process_key(exchange, market_type, symbol, timeframe) instead."
            )

        key = self.make_key(
            exchange=self._default_exchange,
            market_type=self._default_market_type,
            symbol=symbol,
            timeframe=self._default_timeframe,
        )
        return await self.process_key(key)

    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        """
        Backward-compatible wrapper.

        New code should call get_latest_stats_by_key().
        """
        if not self._default_exchange:
            raise ValueError(
                "get_latest_stats(symbol) requires default_exchange. "
                "Use get_latest_stats_by_key(key) instead."
            )

        key = self.make_key(
            exchange=self._default_exchange,
            market_type=self._default_market_type,
            symbol=symbol,
            timeframe=self._default_timeframe,
        )
        return self.get_latest_stats_by_key(key)

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "metric": self._metric_type.value,
            "source_type": self._source_type.value,
            "source_topic_patterns": list(self._source_topic_patterns),
            "scope": "exchange:market_type:symbol:timeframe",
            "defaults": {
                "exchange": self._default_exchange,
                "market_type": self._default_market_type,
                "timeframe": self._default_timeframe,
            },
            "config": {
                "enabled": self._config.enabled,
                "emit_updates": self._config.emit_updates,
                "emit_signals": self._config.emit_signals,
                "min_signal_interval_sec": self._config.min_signal_interval_sec,
                "health_log_interval_sec": self._config.health_log_interval_sec,
                "cleanup_interval_sec": self._config.cleanup_interval_sec,
                "scheduler_job_timeout_sec": self._config.scheduler_job_timeout_sec,
                "scheduler_job_retry_delay_sec": self._config.scheduler_job_retry_delay_sec,
                "scheduler_job_max_retries": self._config.scheduler_job_max_retries,
                "source_name": self._config.source_name,
                "update_topic": self._config.update_topic,
                "signal_topic": self._config.signal_topic,
                "publish_priority": self._config.publish_priority.name,
                "symbol_allowlist": (
                    sorted(self._config.symbol_allowlist)
                    if self._config.symbol_allowlist
                    else None
                ),
            },
            "subscriptions": len(self._subscriptions),
            "health_job_id": self._health_job_id,
            "cleanup_job_id": self._cleanup_job_id,
            "metrics": {
                "processed": self._metrics["processed"],
                "signals_emitted": self._metrics["signals_emitted"],
                "updates_emitted": self._metrics["updates_emitted"],
                "skipped": self._metrics["skipped"],
                "errors": self._metrics["errors"],
                "emit_errors": self._metrics["emit_errors"],
                "keys": dict(self._metrics["keys"]),
            },
        }

    def log_health(self) -> None:
        snapshot = self.stats()
        self._logger.info(
            "%s health | running=%s subscriptions=%s metrics=%s",
            self.__class__.__name__,
            snapshot["running"],
            snapshot["subscriptions"],
            snapshot["metrics"],
        )

    async def cleanup(self) -> None:
        """
        Hook for stale state cleanup.

        Concrete analyzers can override this method. It is scheduled through
        core.scheduler.Scheduler.add_interval_job().
        """
        return None

    # ------------------------------------------------------------------
    # Shared emitters
    # ------------------------------------------------------------------

    async def emit_update(self, stats: BaseOrderFlowStats) -> None:
        if not self._config.emit_updates:
            return

        update = OrderFlowUpdate.from_stats(stats)

        emitted = await self._safe_emit(
            topic=self._config.update_topic,
            payload=update_to_dict(update),
            source=self._config.source_name,
        )

        if emitted:
            self._inc_metric("updates_emitted", stats.key)

    async def emit_signal(self, signal: OrderFlowSignal) -> None:
        if not self._config.emit_signals:
            return

        if not self._can_emit_signal(signal.key):
            self._inc_metric("skipped", signal.key)
            return

        emitted = await self._safe_emit(
            topic=self._config.signal_topic,
            payload=signal_to_dict(signal),
            source=self._config.source_name,
        )

        if emitted:
            self._last_signal_ts_by_key[signal.key] = time.time()
            self._inc_metric("signals_emitted", signal.key)

    def build_signal(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        signal_type: OrderFlowSignalType,
        side: OrderFlowSide,
        strength: float,
        reason: str,
        exchange_symbol: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> OrderFlowSignal:
        return OrderFlowSignal(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            metric=self._metric_type,
            source_type=self._source_type,
            signal_type=signal_type,
            side=side,
            strength=max(0.0, min(1.0, float(strength))),
            reason=reason,
            context=context or {},
        )

    def build_signal_from_stats(
        self,
        *,
        stats: BaseOrderFlowStats,
        signal_type: OrderFlowSignalType,
        side: OrderFlowSide,
        strength: float,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> OrderFlowSignal:
        merged_context = {
            "stats": stats.to_dict(),
            **dict(context or {}),
        }

        return self.build_signal(
            exchange=stats.exchange,
            market_type=stats.market_type,
            symbol=stats.symbol,
            exchange_symbol=stats.exchange_symbol,
            timeframe=stats.timeframe,
            signal_type=signal_type,
            side=side,
            strength=strength,
            reason=reason,
            context=merged_context,
        )

    # ------------------------------------------------------------------
    # Shared event helpers
    # ------------------------------------------------------------------

    def should_process_key(self, key: OrderFlowKey | None) -> bool:
        if key is None:
            return False

        exchange, market_type, symbol, timeframe = key
        if not exchange or not market_type or not symbol or not timeframe:
            return False

        return self._config.should_process_symbol(symbol)

    def should_process_symbol(self, symbol: str | None) -> bool:
        """
        Backward-compatible symbol allowlist check.
        """
        if not symbol:
            return False

        return self._config.should_process_symbol(symbol)

    def extract_key_from_event(self, event: Event) -> OrderFlowKey | None:
        payload = getattr(event, "payload", None)
        key = self._extract_key_from_payload(payload)
        if key is not None:
            return key

        symbol = getattr(event, "symbol", None)
        exchange = getattr(event, "exchange", None) or self._default_exchange
        market_type = getattr(event, "market_type", None) or self._default_market_type
        timeframe = getattr(event, "timeframe", None) or self._default_timeframe

        if symbol and exchange:
            return self.make_key(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                timeframe=str(timeframe),
            )

        return None

    def extract_symbol_from_event(self, event: Event) -> str | None:
        """
        Backward-compatible helper.

        Prefer extract_key_from_event().
        """
        key = self.extract_key_from_event(event)
        return key[2] if key is not None else None

    def extract_exchange_from_event(self, event: Event) -> str | None:
        key = self.extract_key_from_event(event)
        return key[0] if key is not None else None

    def extract_market_type_from_event(self, event: Event) -> str | None:
        key = self.extract_key_from_event(event)
        return key[1] if key is not None else None

    def extract_timeframe_from_event(self, event: Event) -> str | None:
        key = self.extract_key_from_event(event)
        return key[3] if key is not None else None

    def extract_payload_data(self, event: Event) -> Any:
        payload = getattr(event, "payload", None)

        if isinstance(payload, Mapping) and "data" in payload:
            return payload["data"]

        return payload

    def normalize_trade(
        self,
        raw_trade: Any,
        *,
        default_symbol: str | None = None,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
        default_exchange_symbol: str | None = None,
    ) -> NormalizedTrade | None:
        if raw_trade is None:
            return None

        if isinstance(raw_trade, NormalizedTrade):
            return raw_trade if raw_trade.is_valid else None

        if not isinstance(raw_trade, Mapping):
            return None

        symbol = (
            raw_trade.get("symbol")
            or raw_trade.get("s")
            or raw_trade.get("instrument")
            or default_symbol
        )
        exchange = (
            raw_trade.get("exchange")
            or raw_trade.get("venue")
            or raw_trade.get("source_exchange")
            or default_exchange
            or self._default_exchange
        )
        market_type = (
            raw_trade.get("market_type")
            or raw_trade.get("category")
            or raw_trade.get("inst_type")
            or raw_trade.get("instrument_type")
            or default_market_type
            or self._default_market_type
        )
        timeframe = (
            raw_trade.get("timeframe")
            or raw_trade.get("tf")
            or raw_trade.get("interval")
            or default_timeframe
            or self._default_timeframe
        )
        exchange_symbol = (
            raw_trade.get("exchange_symbol")
            or raw_trade.get("raw_symbol")
            or raw_trade.get("exchangeSymbol")
            or default_exchange_symbol
        )

        if not symbol or not exchange:
            return None

        raw_price = raw_trade.get("price", raw_trade.get("p"))
        raw_quantity = raw_trade.get(
            "quantity",
            raw_trade.get("qty", raw_trade.get("q", raw_trade.get("size"))),
        )
        raw_timestamp = raw_trade.get(
            "timestamp",
            raw_trade.get(
                "timestamp_ms",
                raw_trade.get("ts", raw_trade.get("T", time.time())),
            ),
        )

        if raw_price is None or raw_quantity is None or raw_timestamp is None:
            return None

        side = self._extract_trade_side(dict(raw_trade))
        trade_id = raw_trade.get("trade_id", raw_trade.get("id"))
        is_aggressive = bool(
            raw_trade.get("is_aggressive", raw_trade.get("aggressive", False))
        )

        raw_notional = (
            raw_trade.get("notional")
            or raw_trade.get("quote_qty")
            or raw_trade.get("quote_quantity")
            or raw_trade.get("quote_volume")
        )

        try:
            trade = NormalizedTrade.create(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                exchange_symbol=(
                    str(exchange_symbol) if exchange_symbol is not None else None
                ),
                timeframe=str(timeframe),
                side=side,
                price=float(raw_price),
                quantity=float(raw_quantity),
                notional=float(raw_notional) if raw_notional is not None else None,
                timestamp=self._normalize_timestamp(float(raw_timestamp)),
                trade_id=str(trade_id) if trade_id is not None else None,
                is_aggressive=is_aggressive,
                raw=dict(raw_trade),
            )
        except (TypeError, ValueError):
            return None

        return trade if trade.is_valid else None

    def normalize_orderbook_snapshot(
        self,
        raw_snapshot: Any,
        *,
        default_symbol: str | None = None,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
        default_exchange_symbol: str | None = None,
    ) -> OrderbookSnapshot | None:
        if raw_snapshot is None:
            return None

        if isinstance(raw_snapshot, OrderbookSnapshot):
            return raw_snapshot if raw_snapshot.is_valid else None

        if not isinstance(raw_snapshot, Mapping):
            return None

        symbol = (
            raw_snapshot.get("symbol")
            or raw_snapshot.get("s")
            or raw_snapshot.get("instrument")
            or default_symbol
        )
        exchange = (
            raw_snapshot.get("exchange")
            or raw_snapshot.get("venue")
            or raw_snapshot.get("source_exchange")
            or default_exchange
            or self._default_exchange
        )
        market_type = (
            raw_snapshot.get("market_type")
            or raw_snapshot.get("category")
            or raw_snapshot.get("inst_type")
            or raw_snapshot.get("instrument_type")
            or default_market_type
            or self._default_market_type
        )
        timeframe = (
            raw_snapshot.get("timeframe")
            or raw_snapshot.get("tf")
            or raw_snapshot.get("interval")
            or default_timeframe
            or self._default_timeframe
        )
        exchange_symbol = (
            raw_snapshot.get("exchange_symbol")
            or raw_snapshot.get("raw_symbol")
            or raw_snapshot.get("exchangeSymbol")
            or default_exchange_symbol
        )

        if not symbol or not exchange:
            return None

        bids_raw = raw_snapshot.get("bids", raw_snapshot.get("b", []))
        asks_raw = raw_snapshot.get("asks", raw_snapshot.get("a", []))

        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return None

        raw_timestamp = raw_snapshot.get("timestamp")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("timestamp_ms")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("last_update_ts_ms")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("ts")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("T")
        if raw_timestamp is None:
            raw_timestamp = time.time()

        sequence_id = (
            raw_snapshot.get("sequence_id")
            or raw_snapshot.get("sequence")
            or raw_snapshot.get("u")
        )

        try:
            snapshot = OrderbookSnapshot.create(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                exchange_symbol=(
                    str(exchange_symbol) if exchange_symbol is not None else None
                ),
                timeframe=str(timeframe),
                bids=bids_raw,
                asks=asks_raw,
                timestamp=self._normalize_timestamp(float(raw_timestamp)),
                sequence_id=str(sequence_id) if sequence_id is not None else None,
                raw=dict(raw_snapshot),
            )
        except (TypeError, ValueError):
            return None

        return snapshot if snapshot.is_valid else None

    def make_trade_key(self, trade: NormalizedTrade) -> str:
        if trade.trade_id:
            return (
                f"{trade.exchange}:{trade.market_type}:{trade.symbol}:"
                f"{trade.timeframe}:{trade.trade_id}"
            )

        return (
            f"{trade.exchange}:{trade.market_type}:{trade.symbol}:{trade.timeframe}:"
            f"{trade.timestamp:.9f}:{trade.price:.12f}:"
            f"{trade.quantity:.12f}:{trade.side.value}"
        )

    @staticmethod
    def make_key(
        *,
        exchange: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OrderFlowKey:
        return make_orderflow_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        try:
            self._config.validate()
        except Exception:
            self._logger.exception(
                "Invalid analyzer config | analyzer=%s",
                self.__class__.__name__,
            )
            raise

    def _extract_key_from_payload(self, payload: Any) -> OrderFlowKey | None:
        if not isinstance(payload, Mapping):
            return None

        key = self._extract_key_from_mapping(payload)
        if key is not None:
            return key

        data = payload.get("data")
        if isinstance(data, Mapping):
            return self._extract_key_from_mapping(data)

        return None

    def _extract_key_from_mapping(self, data: Mapping[str, Any]) -> OrderFlowKey | None:
        exchange = (
            data.get("exchange")
            or data.get("venue")
            or data.get("source_exchange")
            or self._default_exchange
        )
        market_type = (
            data.get("market_type")
            or data.get("category")
            or data.get("inst_type")
            or data.get("instrument_type")
            or self._default_market_type
        )
        symbol = data.get("symbol") or data.get("s") or data.get("instrument")
        timeframe = (
            data.get("timeframe")
            or data.get("tf")
            or data.get("interval")
            or self._default_timeframe
        )

        if not exchange or not symbol:
            return None

        try:
            return self.make_key(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                timeframe=str(timeframe),
            )
        except ValueError:
            return None

    def _extract_symbol_from_payload(self, payload: Any) -> str | None:
        key = self._extract_key_from_payload(payload)
        return key[2] if key is not None else None

    def _extract_trade_side(self, raw_trade: dict[str, Any]) -> OrderFlowSide:
        side = raw_trade.get("side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        side = raw_trade.get("aggressor_side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        side = raw_trade.get("taker_side")
        if side is not None:
            side_enum = OrderFlowSide.from_value(side)
            if side_enum.is_known:
                return side_enum

        # Binance-style maker flag:
        # m=False => buyer aggressive => buy
        # m=True  => seller aggressive => sell
        maker_flag = raw_trade.get("m")
        if maker_flag is not None:
            return OrderFlowSide.SELL if bool(maker_flag) else OrderFlowSide.BUY

        is_buyer_maker = raw_trade.get("is_buyer_maker")
        if is_buyer_maker is not None:
            return OrderFlowSide.SELL if bool(is_buyer_maker) else OrderFlowSide.BUY

        return OrderFlowSide.UNKNOWN

    def _can_emit_signal(self, key: OrderFlowKey) -> bool:
        now = time.time()
        last_ts = self._last_signal_ts_by_key.get(key, 0.0)
        return (now - last_ts) >= float(self._config.min_signal_interval_sec)

    async def _safe_emit(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        source: str,
    ) -> bool:
        if not topic:
            self._logger.warning(
                "Emit skipped because topic is empty | analyzer=%s",
                self.__class__.__name__,
            )
            self._inc_metric("emit_errors")
            return False

        try:
            return await self._event_bus.emit(
                topic,
                payload,
                priority=self._config.publish_priority,
                source=source,
            )
        except Exception:
            self._inc_metric("emit_errors")
            self._logger.exception(
                "Failed to emit EventBus event | analyzer=%s topic=%s",
                self.__class__.__name__,
                topic,
            )
            return False

    def _inc_metric(
        self,
        name: str,
        key: OrderFlowKey | None = None,
        amount: int = 1,
    ) -> None:
        if name not in self._metrics:
            self._metrics[name] = 0

        self._metrics[name] += amount

        if key is None:
            return

        key_payload = orderflow_key_to_dict(key)
        key_label = ":".join(key)

        key_metrics = self._metrics["keys"].setdefault(
            key_label,
            {
                **key_payload,
                "processed": 0,
                "signals_emitted": 0,
                "updates_emitted": 0,
                "skipped": 0,
                "errors": 0,
                "emit_errors": 0,
            },
        )

        if name in key_metrics:
            key_metrics[name] += amount

    def _build_initial_metrics(self) -> dict[str, Any]:
        return {
            "processed": 0,
            "signals_emitted": 0,
            "updates_emitted": 0,
            "skipped": 0,
            "errors": 0,
            "emit_errors": 0,
            "keys": {},
        }

    @staticmethod
    def _normalize_timestamp(value: float) -> float:
        """
        Normalize timestamp to seconds.

        Data caches may use timestamp_ms. Analytics models use seconds.
        """
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            return timestamp / 1000.0
        return timestamp

    @staticmethod
    def _normalize_market_type(value: Any) -> str:
        normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
        return normalized if normalized else DEFAULT_MARKET_TYPE

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        normalized = str(value or DEFAULT_TIMEFRAME).strip()
        return normalized if normalized else DEFAULT_TIMEFRAME

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        self._health_job_id = self._scheduler.add_interval_job(
            name=f"analytics.orderflow.{self._config.source_name}.health",
            func=self._safe_health_job,
            interval=float(self._config.health_log_interval_sec),
            max_retries=int(self._config.scheduler_job_max_retries),
            retry_delay=float(self._config.scheduler_job_retry_delay_sec),
            timeout=float(self._config.scheduler_job_timeout_sec),
            allow_overlap=False,
            enabled=True,
        )

        self._cleanup_job_id = self._scheduler.add_interval_job(
            name=f"analytics.orderflow.{self._config.source_name}.cleanup",
            func=self._safe_cleanup_job,
            interval=float(self._config.cleanup_interval_sec),
            max_retries=int(self._config.scheduler_job_max_retries),
            retry_delay=float(self._config.scheduler_job_retry_delay_sec),
            timeout=float(self._config.scheduler_job_timeout_sec),
            allow_overlap=False,
            enabled=True,
        )

        self._logger.info(
            "Scheduler jobs registered | analyzer=%s health_job_id=%s cleanup_job_id=%s",
            self.__class__.__name__,
            self._health_job_id,
            self._cleanup_job_id,
        )

    def _disable_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            self._health_job_id = None
            self._cleanup_job_id = None
            return

        for job_id in (self._health_job_id, self._cleanup_job_id):
            if job_id is None:
                continue

            try:
                remove_job = getattr(self._scheduler, "remove_job", None)
                if callable(remove_job):
                    remove_job(job_id)
                else:
                    self._scheduler.disable_job(job_id)
            except Exception:
                self._logger.exception(
                    "Failed to cleanup scheduler job | analyzer=%s job_id=%s",
                    self.__class__.__name__,
                    job_id,
                )

        self._health_job_id = None
        self._cleanup_job_id = None

    async def _safe_cleanup_job(self) -> None:
        try:
            result = self.cleanup()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._inc_metric("errors")
            self._logger.exception(
                "Cleanup job failed | analyzer=%s",
                self.__class__.__name__,
            )

    async def _safe_health_job(self) -> None:
        try:
            result = self.log_health()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._inc_metric("errors")
            self._logger.exception(
                "Health job failed | analyzer=%s",
                self.__class__.__name__,
            )