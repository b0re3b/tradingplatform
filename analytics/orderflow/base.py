from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any

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
    BaseOrderFlowStats,
    NormalizedTrade,
    OrderFlowSignal,
    OrderFlowUpdate,
    OrderbookSnapshot,
    signal_to_dict,
    update_to_dict,
)


class BaseOrderFlowAnalyzer(ABC):
    """
    Base class for all analytics.orderflow analyzers.

    Responsibilities:
    - EventBus subscription lifecycle via register()/stop()
    - Scheduler-based health and cleanup jobs
    - shared metrics
    - signal throttling
    - update/signal EventBus publishing
    - common extraction and normalization helpers

    Concrete analyzers should implement:
    - process_symbol()
    - get_latest_stats()
    - _handle_event()
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseOrderFlowSubConfig,
        metric_type: OrderFlowMetricType,
        source_type: OrderFlowSourceType,
        scheduler: Scheduler ,
        source_topic_patterns: list[str] | tuple[str, ...] | None = None,
        component_module: str = "orderflow",
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._metric_type = metric_type
        self._source_type = source_type
        self._source_topic_patterns = list(source_topic_patterns or ())

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

        self._last_signal_ts_by_symbol: dict[str, float] = {}
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
            "%s registered | metric=%s source_type=%s patterns=%s",
            self.__class__.__name__,
            self._metric_type.value,
            self._source_type.value,
            self._source_topic_patterns,
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
    async def process_symbol(self, symbol: str) -> BaseOrderFlowStats | None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_stats(self, symbol: str) -> BaseOrderFlowStats | None:
        raise NotImplementedError

    @abstractmethod
    async def _handle_event(self, event: Event) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "metric": self._metric_type.value,
            "source_type": self._source_type.value,
            "source_topic_patterns": list(self._source_topic_patterns),
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
                "symbols": dict(self._metrics["symbols"]),
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

        update = OrderFlowUpdate(
            symbol=stats.symbol,
            metric=self._metric_type,
            source_type=self._source_type,
            stats=stats.to_dict(),
        )

        emitted = await self._safe_emit(
            topic=self._config.update_topic,
            payload=update_to_dict(update),
            source=self._config.source_name,
        )

        if emitted:
            self._inc_metric("updates_emitted", stats.symbol)

    async def emit_signal(self, signal: OrderFlowSignal) -> None:
        if not self._config.emit_signals:
            return

        if not self._can_emit_signal(signal.symbol):
            self._inc_metric("skipped", signal.symbol)
            return

        emitted = await self._safe_emit(
            topic=self._config.signal_topic,
            payload=signal_to_dict(signal),
            source=self._config.source_name,
        )

        if emitted:
            self._last_signal_ts_by_symbol[signal.symbol] = time.time()
            self._inc_metric("signals_emitted", signal.symbol)

    def build_signal(
            self,
            *,
            symbol: str,
            signal_type: OrderFlowSignalType,
            side: OrderFlowSide,
            strength: float,
            reason: str,
            context: dict[str, Any] | None = None,
    ) -> OrderFlowSignal:
        return OrderFlowSignal(
            symbol=str(symbol).strip().upper(),
            metric=self._metric_type,
            source_type=self._source_type,
            signal_type=signal_type,
            side=side,
            strength=max(0.0, min(1.0, float(strength))),
            reason=reason,
            context=context or {},
        )
    # ------------------------------------------------------------------
    # Shared event helpers
    # ------------------------------------------------------------------

    def should_process_symbol(self, symbol: str | None) -> bool:
        if not symbol:
            return False

        return self._config.should_process_symbol(symbol)

    def extract_symbol_from_event(self, event: Event) -> str | None:
        payload = getattr(event, "payload", None)

        symbol = self._extract_symbol_from_payload(payload)
        if symbol:
            return symbol

        symbol = getattr(event, "symbol", None)
        if symbol:
            return str(symbol).strip().upper()

        return None

    def extract_exchange_from_event(self, event: Event) -> str | None:
        payload = getattr(event, "payload", None)

        if isinstance(payload, dict):
            exchange = payload.get("exchange")
            if exchange:
                return str(exchange)

            data = payload.get("data")
            if isinstance(data, dict) and data.get("exchange"):
                return str(data["exchange"])

        return None

    def extract_payload_data(self, event: Event) -> Any:
        payload = getattr(event, "payload", None)

        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]

        return payload

    def normalize_trade(
        self,
        raw_trade: Any,
        *,
        default_symbol: str | None = None,
        default_exchange: str | None = None,
    ) -> NormalizedTrade | None:
        if raw_trade is None:
            return None

        if isinstance(raw_trade, NormalizedTrade):
            return raw_trade if raw_trade.is_valid else None

        if not isinstance(raw_trade, dict):
            return None

        symbol = raw_trade.get("symbol") or raw_trade.get("s") or default_symbol
        if not symbol:
            return None

        raw_price = raw_trade.get("price", raw_trade.get("p"))
        raw_quantity = raw_trade.get("quantity", raw_trade.get("qty", raw_trade.get("q")))
        raw_timestamp = raw_trade.get(
            "timestamp",
            raw_trade.get("ts", raw_trade.get("T", time.time())),
        )

        if raw_price is None or raw_quantity is None or raw_timestamp is None:
            return None

        side = self._extract_trade_side(raw_trade)
        trade_id = raw_trade.get("trade_id", raw_trade.get("id"))
        exchange = raw_trade.get("exchange", default_exchange)
        is_aggressive = bool(
            raw_trade.get("is_aggressive", raw_trade.get("aggressive", False))
        )

        try:
            trade = NormalizedTrade.create(
                symbol=str(symbol).upper(),
                side=side,
                price=float(raw_price),
                quantity=float(raw_quantity),
                timestamp=float(raw_timestamp),
                trade_id=str(trade_id) if trade_id is not None else None,
                exchange=str(exchange) if exchange is not None else None,
                is_aggressive=is_aggressive,
                raw=raw_trade,
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
    ) -> OrderbookSnapshot | None:
        if raw_snapshot is None:
            return None

        if isinstance(raw_snapshot, OrderbookSnapshot):
            return raw_snapshot if raw_snapshot.is_valid else None

        if not isinstance(raw_snapshot, dict):
            return None

        symbol = raw_snapshot.get("symbol") or raw_snapshot.get("s") or default_symbol
        if not symbol:
            return None

        bids_raw = raw_snapshot.get("bids", raw_snapshot.get("b", []))
        asks_raw = raw_snapshot.get("asks", raw_snapshot.get("a", []))

        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return None

        raw_timestamp = raw_snapshot.get("timestamp")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("ts")
        if raw_timestamp is None:
            raw_timestamp = raw_snapshot.get("T")
        if raw_timestamp is None:
            raw_timestamp = time.time()

        exchange = raw_snapshot.get("exchange", default_exchange)
        sequence_id = raw_snapshot.get("sequence_id", raw_snapshot.get("u"))

        try:
            snapshot = OrderbookSnapshot.create(
                symbol=str(symbol).upper(),
                bids=bids_raw,
                asks=asks_raw,
                timestamp=float(raw_timestamp),
                exchange=str(exchange) if exchange is not None else None,
                sequence_id=str(sequence_id) if sequence_id is not None else None,
                raw=raw_snapshot,
            )
        except (TypeError, ValueError):
            return None

        return snapshot if snapshot.is_valid else None

    def make_trade_key(self, trade: NormalizedTrade) -> str:
        if trade.trade_id:
            return f"{trade.symbol}:{trade.exchange or ''}:{trade.trade_id}"

        return (
            f"{trade.symbol}:{trade.exchange or ''}:{trade.timestamp:.9f}:"
            f"{trade.price:.12f}:{trade.quantity:.12f}:{trade.side.value}"
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

    def _extract_symbol_from_payload(self, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None

        symbol = payload.get("symbol") or payload.get("s")
        if symbol:
            return str(symbol).strip().upper()

        data = payload.get("data")
        if isinstance(data, dict):
            symbol = data.get("symbol") or data.get("s")
            if symbol:
                return str(symbol).strip().upper()

        return None

    def _extract_trade_side(self, raw_trade: dict[str, Any]) -> OrderFlowSide:
        side = raw_trade.get("side")
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

    def _can_emit_signal(self, symbol: str) -> bool:
        now = time.time()
        normalized_symbol = str(symbol).upper()
        last_ts = self._last_signal_ts_by_symbol.get(normalized_symbol, 0.0)
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
        key: str,
        symbol: str | None = None,
        amount: int = 1,
    ) -> None:
        if key not in self._metrics:
            self._metrics[key] = 0

        self._metrics[key] += amount

        if symbol:
            normalized = str(symbol).upper()
            symbol_metrics = self._metrics["symbols"].setdefault(
                normalized,
                {
                    "processed": 0,
                    "signals_emitted": 0,
                    "updates_emitted": 0,
                    "skipped": 0,
                    "errors": 0,
                    "emit_errors": 0,
                },
            )

            if key in symbol_metrics:
                symbol_metrics[key] += amount

    def _build_initial_metrics(self) -> dict[str, Any]:
        return {
            "processed": 0,
            "signals_emitted": 0,
            "updates_emitted": 0,
            "skipped": 0,
            "errors": 0,
            "emit_errors": 0,
            "symbols": {},
        }

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