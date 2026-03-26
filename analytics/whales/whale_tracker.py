from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Deque, Dict, List, Optional

from core.logger import get_logger


@dataclass(slots=True)
class WhaleTradeEvent:
    """
    Подія про виявлену велику активність (whale activity).
    """
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    timestamp: float

    event_type: str = "whale_trade"
    score: float = 0.0
    zscore: float = 0.0
    threshold_used: float = 0.0

    trades_in_cluster: int = 1
    cluster_notional: float = 0.0
    cluster_duration_sec: float = 0.0

    aggressor: Optional[str] = None
    exchange: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WhaleTrackerStats:
    """
    Внутрішня статистика по символу.
    """
    symbol: str
    processed_trades: int = 0
    detected_whale_trades: int = 0
    last_trade_ts: float = 0.0
    last_whale_ts: float = 0.0
    rolling_mean_notional: float = 0.0
    rolling_std_notional: float = 0.0
    last_threshold: float = 0.0


@dataclass(slots=True)
class TradeRecord:
    """
    Нормалізований трейд для внутрішнього використання.
    """
    symbol: str
    price: float
    quantity: float
    side: str
    timestamp: float
    notional: float
    aggressor: Optional[str] = None
    exchange: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class WhaleTracker:
    """
    WhaleTracker виявляє аномально великі трейди та кластери великих трейдів.

    Основні можливості:
    - детекція окремих великих трейдів за абсолютним порогом
    - адаптивна детекція через z-score від rolling window
    - виявлення серій великих трейдів за короткий час
    - публікація сигналів у EventBus

    Очікується, що вхідні події приходять із EventBus у вигляді dict:
    {
        "symbol": "BTCUSDT",
        "price": 65000.0,
        "quantity": 4.2,
        "side": "buy",
        "timestamp": 1710000000.123,
        "aggressor": "buyer",
        "exchange": "binance"
    }

    Або з альтернативними назвами полів:
    - qty / size / amount
    - ts / event_time / trade_time
    """

    def __init__(
        self,
        event_bus: Any,
        *,
        service_name: str = "whale_tracker",
        enabled: bool = True,
        default_abs_notional_threshold: float = 100_000.0,
        symbol_abs_thresholds: Optional[Dict[str, float]] = None,
        rolling_window_size: int = 300,
        zscore_threshold: float = 3.5,
        min_samples_for_zscore: int = 30,
        cluster_window_sec: float = 3.0,
        cluster_min_trades: int = 3,
        cluster_min_notional: float = 250_000.0,
        publish_event_name: str = "analytics.whale.detected",
        subscribe_trade_event: str = "market.trade",
        subscribe_liquidation_event: Optional[str] = "market.liquidation",
        cleanup_interval_sec: float = 30.0,
        history_retention_sec: float = 120.0,
    ) -> None:
        self._event_bus = event_bus
        self._enabled = enabled

        self._default_abs_notional_threshold = float(default_abs_notional_threshold)
        self._symbol_abs_thresholds = symbol_abs_thresholds or {}

        self._rolling_window_size = int(rolling_window_size)
        self._zscore_threshold = float(zscore_threshold)
        self._min_samples_for_zscore = int(min_samples_for_zscore)

        self._cluster_window_sec = float(cluster_window_sec)
        self._cluster_min_trades = int(cluster_min_trades)
        self._cluster_min_notional = float(cluster_min_notional)

        self._publish_event_name = publish_event_name
        self._subscribe_trade_event = subscribe_trade_event
        self._subscribe_liquidation_event = subscribe_liquidation_event

        self._cleanup_interval_sec = float(cleanup_interval_sec)
        self._history_retention_sec = float(history_retention_sec)

        self._logger = get_logger(__name__, service_name=service_name)

        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        self._notional_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._rolling_window_size)
        )
        self._recent_large_trades: Dict[str, Deque[TradeRecord]] = defaultdict(deque)
        self._stats: Dict[str, WhaleTrackerStats] = {}

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Запуск трекера і підписка на події.
        """
        if self._running:
            self._logger.warning("WhaleTracker already running")
            return

        self._running = True

        if hasattr(self._event_bus, "subscribe"):
            await self._event_bus.subscribe(self._subscribe_trade_event, self.handle_trade_event)

            if self._subscribe_liquidation_event:
                await self._event_bus.subscribe(
                    self._subscribe_liquidation_event,
                    self.handle_liquidation_event
                )

        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="whale_tracker_cleanup")

        self._logger.info(
            "WhaleTracker started",
            extra={
                "trade_event": self._subscribe_trade_event,
                "liquidation_event": self._subscribe_liquidation_event,
                "publish_event": self._publish_event_name,
                "rolling_window_size": self._rolling_window_size,
                "zscore_threshold": self._zscore_threshold,
                "cluster_window_sec": self._cluster_window_sec,
                "cluster_min_trades": self._cluster_min_trades,
                "cluster_min_notional": self._cluster_min_notional,
            },
        )

    async def stop(self) -> None:
        """
        Акуратна зупинка.
        """
        if not self._running:
            return

        self._running = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._logger.info("WhaleTracker stopped")

    # -------------------------------------------------------------------------
    # public handlers
    # -------------------------------------------------------------------------

    async def handle_trade_event(self, event: Dict[str, Any]) -> None:
        """
        Основний обробник трейдів із EventBus.
        """
        if not self._enabled:
            return

        try:
            trade = self._normalize_trade(event)
            if trade is None:
                return

            detected = await self.process_trade(trade)
            if detected:
                await self._publish_whale_event(detected)

        except Exception as exc:
            self._logger.exception(
                "Failed to handle trade event",
                extra={"error": str(exc), "event": event},
            )

    async def handle_liquidation_event(self, event: Dict[str, Any]) -> None:
        """
        Опціональна обробка liquidation подій.
        Можна трактувати великі ліквідації як whale-like pressure.
        """
        if not self._enabled:
            return

        try:
            trade = self._normalize_liquidation_as_trade(event)
            if trade is None:
                return

            detected = await self.process_trade(trade, source="liquidation")
            if detected:
                await self._publish_whale_event(detected)

        except Exception as exc:
            self._logger.exception(
                "Failed to handle liquidation event",
                extra={"error": str(exc), "event": event},
            )

    # -------------------------------------------------------------------------
    # core logic
    # -------------------------------------------------------------------------

    async def process_trade(
        self,
        trade: TradeRecord,
        *,
        source: str = "trade",
    ) -> Optional[WhaleTradeEvent]:
        """
        Обробляє один трейд і за потреби повертає WhaleTradeEvent.
        """
        async with self._lock:
            stats = self._stats.setdefault(trade.symbol, WhaleTrackerStats(symbol=trade.symbol))
            stats.processed_trades += 1
            stats.last_trade_ts = trade.timestamp

            history = self._notional_history[trade.symbol]

            abs_threshold = self._get_abs_threshold(trade.symbol)
            rolling_mean, rolling_std = self._compute_rolling_stats(history)
            zscore = self._compute_zscore(
                value=trade.notional,
                mean_value=rolling_mean,
                std_value=rolling_std,
            )

            dynamic_trigger = (
                len(history) >= self._min_samples_for_zscore
                and rolling_std > 0
                and zscore >= self._zscore_threshold
            )
            absolute_trigger = trade.notional >= abs_threshold

            stats.rolling_mean_notional = rolling_mean
            stats.rolling_std_notional = rolling_std
            stats.last_threshold = abs_threshold

            # Оновлюємо history після обчислення статистики,
            # щоб поточний трейд не впливав на свою ж оцінку.
            history.append(trade.notional)

            if not (absolute_trigger or dynamic_trigger):
                return None

            self._append_large_trade(trade)
            cluster = self._build_cluster_info(trade.symbol, trade.timestamp)

            score = self._compute_score(
                notional=trade.notional,
                abs_threshold=abs_threshold,
                zscore=zscore,
                cluster_notional=cluster["cluster_notional"],
                trades_in_cluster=cluster["trades_in_cluster"],
            )

            stats.detected_whale_trades += 1
            stats.last_whale_ts = trade.timestamp

            whale_event = WhaleTradeEvent(
                symbol=trade.symbol,
                side=trade.side,
                price=trade.price,
                quantity=trade.quantity,
                notional=trade.notional,
                timestamp=trade.timestamp,
                event_type="whale_trade" if source == "trade" else "whale_liquidation",
                score=score,
                zscore=zscore,
                threshold_used=abs_threshold,
                trades_in_cluster=cluster["trades_in_cluster"],
                cluster_notional=cluster["cluster_notional"],
                cluster_duration_sec=cluster["cluster_duration_sec"],
                aggressor=trade.aggressor,
                exchange=trade.exchange,
                metadata={
                    "source": source,
                    "absolute_trigger": absolute_trigger,
                    "dynamic_trigger": dynamic_trigger,
                    "rolling_mean_notional": rolling_mean,
                    "rolling_std_notional": rolling_std,
                },
            )

            self._logger.info(
                "Whale activity detected",
                extra={
                    "symbol": whale_event.symbol,
                    "side": whale_event.side,
                    "price": whale_event.price,
                    "quantity": whale_event.quantity,
                    "notional": whale_event.notional,
                    "score": whale_event.score,
                    "zscore": whale_event.zscore,
                    "cluster_notional": whale_event.cluster_notional,
                    "trades_in_cluster": whale_event.trades_in_cluster,
                    "source": source,
                },
            )

            return whale_event

    # -------------------------------------------------------------------------
    # normalization
    # -------------------------------------------------------------------------

    def _normalize_trade(self, event: Dict[str, Any]) -> Optional[TradeRecord]:
        """
        Приводить довільний market trade event до уніфікованого вигляду.
        """
        symbol = event.get("symbol")
        if not symbol:
            self._logger.debug("Trade skipped: missing symbol", extra={"event": event})
            return None

        price = self._safe_float(event.get("price"))
        quantity = self._safe_float(
            event.get("quantity", event.get("qty", event.get("size", event.get("amount"))))
        )

        if price <= 0 or quantity <= 0:
            self._logger.debug(
                "Trade skipped: invalid price or quantity",
                extra={"symbol": symbol, "price": price, "quantity": quantity},
            )
            return None

        side = str(event.get("side", "unknown")).lower()
        timestamp = self._extract_timestamp(event)

        return TradeRecord(
            symbol=str(symbol).upper(),
            price=price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            notional=price * quantity,
            aggressor=event.get("aggressor"),
            exchange=event.get("exchange"),
            raw=event,
        )

    def _normalize_liquidation_as_trade(self, event: Dict[str, Any]) -> Optional[TradeRecord]:
        """
        Перетворює liquidation event у формат TradeRecord.
        """
        symbol = event.get("symbol")
        if not symbol:
            return None

        price = self._safe_float(event.get("price"))
        quantity = self._safe_float(
            event.get("quantity", event.get("qty", event.get("size", event.get("amount"))))
        )
        if price <= 0 or quantity <= 0:
            return None

        side = str(event.get("side", event.get("liquidation_side", "unknown"))).lower()
        timestamp = self._extract_timestamp(event)

        return TradeRecord(
            symbol=str(symbol).upper(),
            price=price,
            quantity=quantity,
            side=side,
            timestamp=timestamp,
            notional=price * quantity,
            aggressor="liquidation",
            exchange=event.get("exchange"),
            raw=event,
        )

    # -------------------------------------------------------------------------
    # stats / thresholds / cluster logic
    # -------------------------------------------------------------------------

    def _get_abs_threshold(self, symbol: str) -> float:
        return float(self._symbol_abs_thresholds.get(symbol.upper(), self._default_abs_notional_threshold))

    def _compute_rolling_stats(self, history: Deque[float]) -> tuple[float, float]:
        if not history:
            return 0.0, 0.0
        if len(history) == 1:
            return history[0], 0.0
        values = list(history)
        return mean(values), pstdev(values)

    def _compute_zscore(self, value: float, mean_value: float, std_value: float) -> float:
        if std_value <= 0:
            return 0.0
        return (value - mean_value) / std_value

    def _append_large_trade(self, trade: TradeRecord) -> None:
        bucket = self._recent_large_trades[trade.symbol]
        bucket.append(trade)
        self._prune_old_large_trades(bucket, now_ts=trade.timestamp)

    def _prune_old_large_trades(self, bucket: Deque[TradeRecord], *, now_ts: float) -> None:
        while bucket and (now_ts - bucket[0].timestamp) > self._history_retention_sec:
            bucket.popleft()

    def _build_cluster_info(self, symbol: str, now_ts: float) -> Dict[str, Any]:
        bucket = self._recent_large_trades[symbol]

        cluster_trades: List[TradeRecord] = [
            trade for trade in bucket
            if (now_ts - trade.timestamp) <= self._cluster_window_sec
        ]

        if not cluster_trades:
            return {
                "trades_in_cluster": 1,
                "cluster_notional": 0.0,
                "cluster_duration_sec": 0.0,
            }

        cluster_notional = sum(t.notional for t in cluster_trades)
        first_ts = min(t.timestamp for t in cluster_trades)
        last_ts = max(t.timestamp for t in cluster_trades)
        duration = max(0.0, last_ts - first_ts)

        qualifies = (
            len(cluster_trades) >= self._cluster_min_trades
            or cluster_notional >= self._cluster_min_notional
        )

        if not qualifies:
            return {
                "trades_in_cluster": len(cluster_trades),
                "cluster_notional": cluster_notional,
                "cluster_duration_sec": duration,
            }

        return {
            "trades_in_cluster": len(cluster_trades),
            "cluster_notional": cluster_notional,
            "cluster_duration_sec": duration,
        }

    def _compute_score(
        self,
        *,
        notional: float,
        abs_threshold: float,
        zscore: float,
        cluster_notional: float,
        trades_in_cluster: int,
    ) -> float:
        """
        Композитний score для ранжування whale-сигналів.
        """
        threshold_ratio = notional / abs_threshold if abs_threshold > 0 else 0.0
        cluster_bonus = math.log1p(max(cluster_notional, 0.0) / max(abs_threshold, 1.0))
        trade_count_bonus = min(trades_in_cluster / 10.0, 1.5)
        zscore_bonus = max(zscore, 0.0) / 5.0

        score = (
            threshold_ratio * 0.55
            + zscore_bonus * 0.20
            + cluster_bonus * 0.15
            + trade_count_bonus * 0.10
        )

        return round(score, 6)

    # -------------------------------------------------------------------------
    # publishing
    # -------------------------------------------------------------------------

    async def _publish_whale_event(self, whale_event: WhaleTradeEvent) -> None:
        payload = asdict(whale_event)

        if hasattr(self._event_bus, "publish"):
            await self._event_bus.publish(self._publish_event_name, payload)
        elif hasattr(self._event_bus, "emit"):
            await self._event_bus.emit(self._publish_event_name, payload)
        else:
            self._logger.warning(
                "EventBus has no publish/emit method; whale event not dispatched",
                extra={"event_name": self._publish_event_name, "payload": payload},
            )

    # -------------------------------------------------------------------------
    # monitoring / introspection
    # -------------------------------------------------------------------------

    def get_symbol_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        stats = self._stats.get(symbol.upper())
        if not stats:
            return None
        return asdict(stats)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {symbol: asdict(stats) for symbol, stats in self._stats.items()}

    def reset_symbol(self, symbol: str) -> None:
        symbol = symbol.upper()
        self._notional_history.pop(symbol, None)
        self._recent_large_trades.pop(symbol, None)
        self._stats.pop(symbol, None)

        self._logger.info("WhaleTracker symbol state reset", extra={"symbol": symbol})

    def reset_all(self) -> None:
        self._notional_history.clear()
        self._recent_large_trades.clear()
        self._stats.clear()

        self._logger.info("WhaleTracker state reset")

    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval_sec)

                async with self._lock:
                    now_ts = time.time()

                    for symbol, bucket in list(self._recent_large_trades.items()):
                        self._prune_old_large_trades(bucket, now_ts=now_ts)
                        if not bucket:
                            self._recent_large_trades.pop(symbol, None)

                self._logger.debug("WhaleTracker cleanup completed")

        except asyncio.CancelledError:
            self._logger.debug("WhaleTracker cleanup task cancelled")
            raise
        except Exception as exc:
            self._logger.exception(
                "WhaleTracker cleanup loop failed",
                extra={"error": str(exc)},
            )

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def _extract_timestamp(self, event: Dict[str, Any]) -> float:
        raw_ts = (
            event.get("timestamp")
            or event.get("ts")
            or event.get("event_time")
            or event.get("trade_time")
            or time.time()
        )

        ts = self._safe_float(raw_ts)
        if ts <= 0:
            return time.time()

        # Якщо timestamp у мілісекундах
        if ts > 10_000_000_000:
            ts /= 1000.0

        return ts

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default