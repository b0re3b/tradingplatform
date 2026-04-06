from __future__ import annotations

import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.event_bus import Event, EventBus
from core.logger import get_logger

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
    OrderbookLevel,
    OrderbookSnapshot,
    signal_to_dict,
    stats_to_dict,
)


class BaseOrderFlowAnalyzer(ABC):
    """
    Базовий клас для всіх analyzers пакета analytics.orderflow.

    Відповідальність:
    - lifecycle start/stop
    - subscribe/unsubscribe до EventBus
    - загальні метрики
    - throttling сигналів
    - emit update/signal подій
    - базові helper-и для extraction/normalization
    - інтеграція з scheduler
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: BaseOrderFlowSubConfig,
        metric_type: OrderFlowMetricType,
        source_type: OrderFlowSourceType,
        scheduler: Optional[Any] = None,
        source_topic_patterns: Optional[list[str]] = None,
        component_module: str = "orderflow",
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config
        self._metric_type = metric_type
        self._source_type = source_type
        self._source_topic_patterns = source_topic_patterns or []

        self._logger = get_logger(
            __name__,
            service_name=self._config.source_name,
            component="analytics",
            module=component_module,
        )

        self._subscriptions: list[Any] = []
        self._running = False
        self._lock = asyncio.Lock()

        self._health_job_id: Optional[str] = None
        self._cleanup_job_id: Optional[str] = None

        self._last_signal_ts_by_symbol: dict[str, float] = {}
        self._metrics: dict[str, Any] = self._build_initial_metrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            self._logger.warning("%s already started", self.__class__.__name__)
            return

        if not self._config.enabled:
            self._logger.warning("%s is disabled by config", self.__class__.__name__)
            return

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
            "%s started | metric=%s source_type=%s patterns=%s",
            self.__class__.__name__,
            self._metric_type.value,
            self._source_type.value,
            self._source_topic_patterns,
        )

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("%s already stopped", self.__class__.__name__)
            return

        for sub in self._subscriptions:
            try:
                self._event_bus.unsubscribe(sub)
            except Exception:
                self._logger.exception(
                    "Failed to unsubscribe handler | analyzer=%s",
                    self.__class__.__name__,
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
    async def process_symbol(self, symbol: str) -> Optional[BaseOrderFlowStats]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_stats(self, symbol: str) -> Optional[BaseOrderFlowStats]:
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
            "config": {
                "enabled": self._config.enabled,
                "emit_updates": self._config.emit_updates,
                "emit_signals": self._config.emit_signals,
                "min_signal_interval_sec": self._config.min_signal_interval_sec,
                "health_log_interval_sec": self._config.health_log_interval_sec,
                "cleanup_interval_sec": self._config.cleanup_interval_sec,
                "source_name": self._config.source_name,
                "update_topic": self._config.update_topic,
                "signal_topic": self._config.signal_topic,
                "symbol_allowlist": sorted(self._config.symbol_allowlist)
                if self._config.symbol_allowlist
                else None,
            },
            "health_job_id": self._health_job_id,
            "cleanup_job_id": self._cleanup_job_id,
            "metrics": {
                "processed": self._metrics["processed"],
                "signals_emitted": self._metrics["signals_emitted"],
                "updates_emitted": self._metrics["updates_emitted"],
                "skipped": self._metrics["skipped"],
                "errors": self._metrics["errors"],
                "symbols": dict(self._metrics["symbols"]),
            },
        }

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
            stats=stats_to_dict(stats),
        )

        await self._safe_emit(
            topic=self._config.update_topic,
            payload=signal_or_update_payload(update),
            source=self._config.source_name,
        )
        self._inc_metric("updates_emitted", stats.symbol)

    async def emit_signal(self, signal: OrderFlowSignal) -> None:
        if not self._config.emit_signals:
            return

        if not self._can_emit_signal(signal.symbol):
            self._inc_metric("skipped", signal.symbol)
            return

        await self._safe_emit(
            topic=self._config.signal_topic,
            payload=signal_or_update_payload(signal),
            source=self._config.source_name,
        )

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
        context: Optional[dict[str, Any]] = None,
    ) -> OrderFlowSignal:
        return OrderFlowSignal(
            symbol=symbol,
            metric=self._metric_type,
            signal_type=signal_type,
            side=side,
            strength=max(0.0, min(float(strength), 1.0)),
            reason=reason,
            context=context or {},
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def should_process_symbol(self, symbol: Optional[str]) -> bool:
        if not symbol:
            return False

        normalized = str(symbol).upper()
        allowlist = self._config.symbol_allowlist

        if allowlist and normalized not in allowlist:
            return False

        return True

    def extract_symbol_from_event(self, event: Event) -> Optional[str]:
        payload = getattr(event, "payload", None)

        if isinstance(payload, dict):
            symbol = payload.get("symbol") or payload.get("s")
            if symbol:
                return str(symbol).upper()

            data = payload.get("data")
            if isinstance(data, dict):
                symbol = data.get("symbol") or data.get("s")
                if symbol:
                    return str(symbol).upper()

        symbol = getattr(event, "symbol", None)
        if symbol:
            return str(symbol).upper()

        return None

    def extract_exchange_from_event(self, event: Event) -> Optional[str]:
        payload = getattr(event, "payload", None)
        if isinstance(payload, dict):
            exchange = payload.get("exchange")
            if exchange:
                return str(exchange)
            data = payload.get("data")
            if isinstance(data, dict) and data.get("exchange"):
                return str(data["exchange"])
        return None

    def normalize_trade(
        self,
        raw_trade: Any,
        *,
        default_symbol: Optional[str] = None,
        default_exchange: Optional[str] = None,
    ) -> Optional[NormalizedTrade]:
        if raw_trade is None:
            return None

        if isinstance(raw_trade, NormalizedTrade):
            return raw_trade

        if not isinstance(raw_trade, dict):
            return None

        symbol = raw_trade.get("symbol") or raw_trade.get("s") or default_symbol
        if not symbol:
            return None

        price = raw_trade.get("price", raw_trade.get("p"))
        quantity = raw_trade.get("quantity", raw_trade.get("qty", raw_trade.get("q")))
        timestamp = raw_trade.get("timestamp", raw_trade.get("ts", raw_trade.get("T", time.time())))
        trade_id = raw_trade.get("trade_id", raw_trade.get("id"))
        exchange = raw_trade.get("exchange", default_exchange)

        side = self._extract_trade_side(raw_trade)
        is_aggressive = bool(
            raw_trade.get("is_aggressive", raw_trade.get("aggressive", False))
        )

        try:
            return NormalizedTrade.create(
                symbol=str(symbol).upper(),
                side=side,
                price=float(price),
                quantity=float(quantity),
                timestamp=float(timestamp),
                trade_id=str(trade_id) if trade_id is not None else None,
                exchange=str(exchange) if exchange is not None else None,
                is_aggressive=is_aggressive,
                raw=raw_trade,
            )
        except (TypeError, ValueError):
            return None

    def normalize_orderbook_snapshot(
        self,
        raw_snapshot: Any,
        *,
        default_symbol: Optional[str] = None,
        default_exchange: Optional[str] = None,
    ) -> Optional[OrderbookSnapshot]:
        if raw_snapshot is None:
            return None

        if isinstance(raw_snapshot, OrderbookSnapshot):
            return raw_snapshot

        if not isinstance(raw_snapshot, dict):
            return None

        symbol = raw_snapshot.get("symbol") or raw_snapshot.get("s") or default_symbol
        if not symbol:
            return None

        bids_raw = raw_snapshot.get("bids", [])
        asks_raw = raw_snapshot.get("asks", [])
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return None

        bids = [level for item in bids_raw if (level := OrderbookLevel.from_raw(item)) is not None]
        asks = [level for item in asks_raw if (level := OrderbookLevel.from_raw(item)) is not None]

        if not bids or not asks:
            return None

        return OrderbookSnapshot(
            symbol=str(symbol).upper(),
            bids=bids,
            asks=asks,
            timestamp=float(raw_snapshot.get("timestamp", raw_snapshot.get("ts", time.time()))),
            exchange=raw_snapshot.get("exchange", default_exchange),
            sequence_id=raw_snapshot.get("sequence_id", raw_snapshot.get("u")),
            raw=raw_snapshot,
        )

    def make_trade_key(self, trade: NormalizedTrade) -> str:
        if trade.trade_id:
            return f"{trade.symbol}:{trade.exchange or ''}:{trade.trade_id}"
        return (
            f"{trade.symbol}:{trade.exchange or ''}:{trade.timestamp:.9f}:"
            f"{trade.price:.12f}:{trade.quantity:.12f}:{trade.side.value}"
        )

    def log_health(self) -> None:
        snapshot = self.stats()
        self._logger.info(
            "%s health | metrics=%s",
            self.__class__.__name__,
            snapshot["metrics"],
        )

    async def cleanup(self) -> None:
        """
        Hook для cleanup старого стану.
        За замовчуванням нічого не робить.
        """
        return

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _extract_trade_side(self, raw_trade: dict[str, Any]) -> OrderFlowSide:
        side = raw_trade.get("side")
        if isinstance(side, str):
            side_value = side.lower()
            if side_value == OrderFlowSide.BUY.value:
                return OrderFlowSide.BUY
            if side_value == OrderFlowSide.SELL.value:
                return OrderFlowSide.SELL

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
        last_ts = self._last_signal_ts_by_symbol.get(symbol, 0.0)
        return (now - last_ts) >= float(self._config.min_signal_interval_sec)

    async def _safe_emit(self, *, topic: str, payload: dict[str, Any], source: str) -> None:
        await self._event_bus.emit(
            topic,
            payload,
            priority=self._config.publish_priority,
            source=source,
        )

    def _inc_metric(self, key: str, symbol: Optional[str] = None, amount: int = 1) -> None:
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
            "symbols": {},
        }

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        add_interval_job = getattr(self._scheduler, "add_interval_job", None)
        if add_interval_job is None:
            self._logger.warning(
                "Scheduler does not support add_interval_job | analyzer=%s",
                self.__class__.__name__,
            )
            return

        try:
            self._health_job_id = add_interval_job(
                func=self._safe_health_job,
                seconds=float(self._config.health_log_interval_sec),
                name=f"{self.__class__.__name__}.health",
                timeout_seconds=float(self._config.scheduler_job_timeout_sec),
                retry_delay_seconds=float(self._config.scheduler_job_retry_delay_sec),
                max_retries=int(self._config.scheduler_job_max_retries),
            )
        except TypeError:
            try:
                self._health_job_id = add_interval_job(
                    self._safe_health_job,
                    self._config.health_log_interval_sec,
                    name=f"{self.__class__.__name__}.health",
                )
            except Exception:
                self._logger.exception("Failed to register health job")

        try:
            self._cleanup_job_id = add_interval_job(
                func=self._safe_cleanup_job,
                seconds=float(self._config.cleanup_interval_sec),
                name=f"{self.__class__.__name__}.cleanup",
                timeout_seconds=float(self._config.scheduler_job_timeout_sec),
                retry_delay_seconds=float(self._config.scheduler_job_retry_delay_sec),
                max_retries=int(self._config.scheduler_job_max_retries),
            )
        except TypeError:
            try:
                self._cleanup_job_id = add_interval_job(
                    self._safe_cleanup_job,
                    self._config.cleanup_interval_sec,
                    name=f"{self.__class__.__name__}.cleanup",
                )
            except Exception:
                self._logger.exception("Failed to register cleanup job")

    def _disable_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        disable_job = getattr(self._scheduler, "disable_job", None)
        remove_job = getattr(self._scheduler, "remove_job", None)

        for job_id in (self._health_job_id, self._cleanup_job_id):
            if not job_id:
                continue

            try:
                if disable_job is not None:
                    disable_job(job_id)
                elif remove_job is not None:
                    remove_job(job_id)
            except Exception:
                self._logger.exception(
                    "Failed to disable/remove scheduler job | analyzer=%s job_id=%s",
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
            self._logger.exception(
                "Cleanup job failed | analyzer=%s",
                self.__class__.__name__,
            )

    async def _safe_health_job(self) -> None:
        try:
            self.log_health()
        except Exception:
            self._logger.exception(
                "Health job failed | analyzer=%s",
                self.__class__.__name__,
            )


def signal_or_update_payload(obj: OrderFlowSignal | OrderFlowUpdate) -> dict[str, Any]:
    if isinstance(obj, OrderFlowSignal):
        payload = signal_to_dict(obj)
    else:
        payload = {
            "symbol": obj.symbol,
            "metric": obj.metric.value,
            "source_type": obj.source_type.value,
            "stats": obj.stats,
            "timestamp": obj.timestamp,
        }

    # enum -> str для чистого JSON payload
    for key in ("metric", "signal_type", "side", "source_type"):
        value = payload.get(key)
        if hasattr(value, "value"):
            payload[key] = value.value

    return payload