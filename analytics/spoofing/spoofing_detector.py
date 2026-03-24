from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.logger import get_logger


class SpoofingSide(str, Enum):
    BID = "bid"
    ASK = "ask"


class SpoofingType(str, Enum):
    BID_SPOOF = "bid_spoof"
    ASK_SPOOF = "ask_spoof"


class CandidateStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@dataclass(slots=True)
class SpoofingDetectorConfig:
    """
    Конфіг для детектора спуфінгу.

    Пояснення логіки:
    - min_abs_size: мінімальний абсолютний розмір заявки, щоб вважати її підозрілою
    - min_size_multiple_vs_avg: заявка має бути у X разів більшою за середній рівень стакану
    - max_distance_bps: рівень має бути близько до best bid/ask
    - min_lifetime_ms / max_lifetime_ms: типовий spoof живе недовго
    - cancel_ratio_threshold: яка частина обсягу має зникнути
    - price_move_confirmation_bps: на скільки має зрушити mid price після зникнення
    - trade_pressure_window_ms: вікно угод для оцінки агресії
    - min_opposite_pressure_ratio: підтвердження, що рух іде у "маніпулятивний" бік
    - cooldown_ms_same_level: щоб не спамити повторними сигналами по тому ж рівню
    - cleanup_interval_sec: періодична очистка state
    """
    enabled: bool = True
    symbols: Optional[List[str]] = None

    min_abs_size: float = 50_000.0
    min_size_multiple_vs_avg: float = 4.0
    max_distance_bps: float = 15.0

    min_lifetime_ms: int = 100
    max_lifetime_ms: int = 8_000

    cancel_ratio_threshold: float = 0.75
    price_move_confirmation_bps: float = 4.0

    trade_pressure_window_ms: int = 3_000
    min_opposite_pressure_ratio: float = 1.35

    candidate_ttl_ms: int = 12_000
    cooldown_ms_same_level: int = 15_000

    max_candidates_per_symbol: int = 200
    max_trade_events_per_symbol: int = 2_000

    cleanup_interval_sec: int = 5
    emit_raw_candidate_events: bool = False


@dataclass(slots=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(slots=True)
class TradeTick:
    symbol: str
    price: float
    qty: float
    side: str  # "buy" | "sell"
    ts_ms: int


@dataclass(slots=True)
class SpoofingCandidate:
    id: str
    symbol: str
    side: SpoofingSide
    price: float
    initial_size: float
    peak_size: float

    detected_ts_ms: int
    last_seen_ts_ms: int

    best_bid_at_detection: float
    best_ask_at_detection: float
    mid_at_detection: float

    avg_same_side_size_at_detection: float
    distance_bps_at_detection: float
    size_multiple_at_detection: float

    status: CandidateStatus = CandidateStatus.ACTIVE
    removed_ts_ms: Optional[int] = None
    removal_size: Optional[float] = None
    cancel_ratio: Optional[float] = None

    confirmation_ts_ms: Optional[int] = None
    confirmation_price_move_bps: Optional[float] = None
    opposite_pressure_ratio: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpoofingSignal:
    id: str
    symbol: str
    spoofing_type: SpoofingType
    side: SpoofingSide

    level_price: float
    initial_size: float
    peak_size: float
    cancel_ratio: float

    detected_ts_ms: int
    removed_ts_ms: int
    confirmed_ts_ms: int

    mid_at_detection: float
    mid_at_confirmation: float
    price_move_bps: float
    opposite_pressure_ratio: float

    distance_bps_at_detection: float
    size_multiple_at_detection: float

    score: float
    confidence: float

    details: Dict[str, Any] = field(default_factory=dict)


class SpoofingDetector:
    """
    Production-style detector спуфінгу для модульної трейдинг-системи.

    Очікувані події:
    - market.orderbook.updated
    - market.trade
    - market.trades.aggressive   (опціонально, якщо у тебе вже є AggressiveTrades)

    Публікує:
    - analytics.spoofing.candidate_detected
    - analytics.spoofing.candidate_removed
    - analytics.spoofing.detected

    Формат orderbook event (мінімально потрібне):
    {
        "symbol": "BTCUSDT",
        "ts_ms": 1712345678901,
        "best_bid": 68000.0,
        "best_ask": 68000.5,
        "bids": [{"price": 67999.9, "size": 1200.0}, ...],
        "asks": [{"price": 68000.5, "size": 900.0}, ...]
    }

    Формат trade event:
    {
        "symbol": "BTCUSDT",
        "price": 68001.0,
        "qty": 12.5,
        "side": "buy",
        "ts_ms": 1712345678950
    }
    """

    def __init__(
        self,
        config: SpoofingDetectorConfig,
        event_bus: Any,
        scheduler: Optional[Any] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.logger = logger or get_logger(__name__, service_name="spoofing_detector")

        self._is_running = False
        self._started_at_ms: Optional[int] = None

        self._latest_mid_by_symbol: Dict[str, float] = {}
        self._latest_best_bid_by_symbol: Dict[str, float] = {}
        self._latest_best_ask_by_symbol: Dict[str, float] = {}

        self._candidates_by_symbol: Dict[str, Dict[str, SpoofingCandidate]] = defaultdict(dict)
        self._trades_by_symbol: Dict[str, Deque[TradeTick]] = defaultdict(deque)
        self._cooldowns: Dict[Tuple[str, str, float], int] = {}

        self._stats: Dict[str, Any] = {
            "orderbook_events": 0,
            "trade_events": 0,
            "candidates_created": 0,
            "candidates_removed": 0,
            "signals_confirmed": 0,
            "expired_candidates": 0,
            "invalidated_candidates": 0,
            "errors": 0,
        }

        self.logger.info(
            "SpoofingDetector initialized",
            extra={
                "enabled": self.config.enabled,
                "symbols": self.config.symbols,
                "min_abs_size": self.config.min_abs_size,
                "min_size_multiple_vs_avg": self.config.min_size_multiple_vs_avg,
                "max_distance_bps": self.config.max_distance_bps,
            },
        )

    async def start(self) -> None:
        if self._is_running:
            self.logger.warning("SpoofingDetector already running")
            return

        if not self.config.enabled:
            self.logger.warning("SpoofingDetector disabled by config")
            return

        self._is_running = True
        self._started_at_ms = self._now_ms()

        await self._subscribe_events()
        await self._setup_scheduler_jobs()

        self.logger.info(
            "SpoofingDetector started",
            extra={"started_at_ms": self._started_at_ms},
        )

    async def stop(self) -> None:
        if not self._is_running:
            return

        self._is_running = False
        self.logger.info("SpoofingDetector stopped")

    async def _subscribe_events(self) -> None:
        """
        Передбачається, що EventBus вміє subscribe(event_name, handler).
        Якщо у тебе інший контракт EventBus — адаптуєш лише цей шар.
        """
        subscribe = getattr(self.event_bus, "subscribe", None)
        if subscribe is None:
            raise AttributeError("EventBus does not support 'subscribe'")

        maybe_awaitable = subscribe("market.orderbook.updated", self.on_orderbook_update)
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

        maybe_awaitable = subscribe("market.trade", self.on_trade)
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

        maybe_awaitable = subscribe("market.trades.aggressive", self.on_trade)
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

        self.logger.info(
            "SpoofingDetector subscribed to EventBus events",
            extra={
                "events": [
                    "market.orderbook.updated",
                    "market.trade",
                    "market.trades.aggressive",
                ]
            },
        )

    async def _setup_scheduler_jobs(self) -> None:
        if self.scheduler is None:
            return

        add_interval_job = getattr(self.scheduler, "add_interval_job", None)
        if add_interval_job is None:
            self.logger.warning("Scheduler does not support add_interval_job")
            return

        result = add_interval_job(
            name="spoofing_detector_cleanup",
            interval_seconds=self.config.cleanup_interval_sec,
            coro=self.cleanup_expired_candidates,
            overlap=False,
        )
        if asyncio.iscoroutine(result):
            await result

        self.logger.info(
            "SpoofingDetector cleanup job registered",
            extra={"cleanup_interval_sec": self.config.cleanup_interval_sec},
        )

    async def on_orderbook_update(self, event: Dict[str, Any]) -> None:
        if not self._is_running:
            return

        try:
            symbol = str(event["symbol"])
            if not self._symbol_allowed(symbol):
                return

            ts_ms = int(event.get("ts_ms") or self._now_ms())
            best_bid = float(event["best_bid"])
            best_ask = float(event["best_ask"])
            bids = self._normalize_levels(event.get("bids", []))
            asks = self._normalize_levels(event.get("asks", []))

            if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
                return

            mid = (best_bid + best_ask) / 2.0

            self._latest_best_bid_by_symbol[symbol] = best_bid
            self._latest_best_ask_by_symbol[symbol] = best_ask
            self._latest_mid_by_symbol[symbol] = mid
            self._stats["orderbook_events"] += 1

            await self._update_existing_candidates(
                symbol=symbol,
                ts_ms=ts_ms,
                best_bid=best_bid,
                best_ask=best_ask,
                mid=mid,
                bids=bids,
                asks=asks,
            )

            await self._scan_for_new_candidates(
                symbol=symbol,
                ts_ms=ts_ms,
                best_bid=best_bid,
                best_ask=best_ask,
                mid=mid,
                bids=bids,
                asks=asks,
            )

        except Exception as exc:
            self._stats["errors"] += 1
            self.logger.exception(
                "Failed to process orderbook update",
                extra={"event": self._safe_event(event), "error": str(exc)},
            )

    async def on_trade(self, event: Dict[str, Any]) -> None:
        if not self._is_running:
            return

        try:
            symbol = str(event["symbol"])
            if not self._symbol_allowed(symbol):
                return

            trade = TradeTick(
                symbol=symbol,
                price=float(event["price"]),
                qty=float(event.get("qty", event.get("size", 0.0))),
                side=str(event["side"]).lower(),
                ts_ms=int(event.get("ts_ms") or self._now_ms()),
            )

            if trade.qty <= 0:
                return

            dq = self._trades_by_symbol[symbol]
            dq.append(trade)
            self._stats["trade_events"] += 1

            while len(dq) > self.config.max_trade_events_per_symbol:
                dq.popleft()

            self._prune_old_trades(symbol=symbol, now_ms=trade.ts_ms)

            await self._try_confirm_removed_candidates(symbol=symbol, now_ms=trade.ts_ms)

        except Exception as exc:
            self._stats["errors"] += 1
            self.logger.exception(
                "Failed to process trade event",
                extra={"event": self._safe_event(event), "error": str(exc)},
            )

    async def _scan_for_new_candidates(
        self,
        symbol: str,
        ts_ms: int,
        best_bid: float,
        best_ask: float,
        mid: float,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
    ) -> None:
        avg_bid_size = self._avg_size(bids)
        avg_ask_size = self._avg_size(asks)

        for level in bids:
            if not self._is_suspicious_level(
                side=SpoofingSide.BID,
                level=level,
                avg_same_side_size=avg_bid_size,
                best_bid=best_bid,
                best_ask=best_ask,
            ):
                continue

            if self._is_on_cooldown(symbol, SpoofingSide.BID, level.price, ts_ms):
                continue

            if self._has_similar_active_candidate(symbol, SpoofingSide.BID, level.price):
                continue

            candidate = SpoofingCandidate(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=SpoofingSide.BID,
                price=level.price,
                initial_size=level.size,
                peak_size=level.size,
                detected_ts_ms=ts_ms,
                last_seen_ts_ms=ts_ms,
                best_bid_at_detection=best_bid,
                best_ask_at_detection=best_ask,
                mid_at_detection=mid,
                avg_same_side_size_at_detection=avg_bid_size,
                distance_bps_at_detection=self._distance_bps_from_touch(
                    side=SpoofingSide.BID,
                    price=level.price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                ),
                size_multiple_at_detection=(level.size / avg_bid_size) if avg_bid_size > 0 else 999.0,
            )
            await self._register_candidate(candidate)

        for level in asks:
            if not self._is_suspicious_level(
                side=SpoofingSide.ASK,
                level=level,
                avg_same_side_size=avg_ask_size,
                best_bid=best_bid,
                best_ask=best_ask,
            ):
                continue

            if self._is_on_cooldown(symbol, SpoofingSide.ASK, level.price, ts_ms):
                continue

            if self._has_similar_active_candidate(symbol, SpoofingSide.ASK, level.price):
                continue

            candidate = SpoofingCandidate(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=SpoofingSide.ASK,
                price=level.price,
                initial_size=level.size,
                peak_size=level.size,
                detected_ts_ms=ts_ms,
                last_seen_ts_ms=ts_ms,
                best_bid_at_detection=best_bid,
                best_ask_at_detection=best_ask,
                mid_at_detection=mid,
                avg_same_side_size_at_detection=avg_ask_size,
                distance_bps_at_detection=self._distance_bps_from_touch(
                    side=SpoofingSide.ASK,
                    price=level.price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                ),
                size_multiple_at_detection=(level.size / avg_ask_size) if avg_ask_size > 0 else 999.0,
            )
            await self._register_candidate(candidate)

    async def _update_existing_candidates(
        self,
        symbol: str,
        ts_ms: int,
        best_bid: float,
        best_ask: float,
        mid: float,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
    ) -> None:
        active_candidates = self._candidates_by_symbol.get(symbol, {})
        if not active_candidates:
            return

        bid_map = {round(x.price, 8): x.size for x in bids}
        ask_map = {round(x.price, 8): x.size for x in asks}

        to_remove: List[str] = []

        for candidate_id, candidate in list(active_candidates.items()):
            if candidate.status != CandidateStatus.ACTIVE:
                continue

            candidate.last_seen_ts_ms = ts_ms

            current_size = (
                bid_map.get(round(candidate.price, 8), 0.0)
                if candidate.side == SpoofingSide.BID
                else ask_map.get(round(candidate.price, 8), 0.0)
            )

            if current_size > candidate.peak_size:
                candidate.peak_size = current_size

            lifetime_ms = ts_ms - candidate.detected_ts_ms
            if lifetime_ms > self.config.candidate_ttl_ms:
                candidate.status = CandidateStatus.EXPIRED
                to_remove.append(candidate_id)
                self._stats["expired_candidates"] += 1
                continue

            if current_size <= 0:
                removed = await self._mark_candidate_removed(
                    candidate=candidate,
                    removed_ts_ms=ts_ms,
                    removal_size=0.0,
                    mid=mid,
                )
                if removed:
                    continue

            cancel_ratio = 1.0 - (current_size / candidate.peak_size if candidate.peak_size > 0 else 1.0)
            if cancel_ratio >= self.config.cancel_ratio_threshold:
                removed = await self._mark_candidate_removed(
                    candidate=candidate,
                    removed_ts_ms=ts_ms,
                    removal_size=current_size,
                    mid=mid,
                )
                if removed:
                    continue

            if not self._still_logically_valid(candidate, best_bid=best_bid, best_ask=best_ask):
                candidate.status = CandidateStatus.INVALIDATED
                to_remove.append(candidate_id)
                self._stats["invalidated_candidates"] += 1

        for candidate_id in to_remove:
            active_candidates.pop(candidate_id, None)

    async def _mark_candidate_removed(
        self,
        candidate: SpoofingCandidate,
        removed_ts_ms: int,
        removal_size: float,
        mid: float,
    ) -> bool:
        lifetime_ms = removed_ts_ms - candidate.detected_ts_ms
        if lifetime_ms < self.config.min_lifetime_ms:
            candidate.status = CandidateStatus.INVALIDATED
            self._candidates_by_symbol[candidate.symbol].pop(candidate.id, None)
            self._stats["invalidated_candidates"] += 1
            return False

        if lifetime_ms > self.config.max_lifetime_ms:
            candidate.status = CandidateStatus.EXPIRED
            self._candidates_by_symbol[candidate.symbol].pop(candidate.id, None)
            self._stats["expired_candidates"] += 1
            return False

        candidate.removed_ts_ms = removed_ts_ms
        candidate.removal_size = removal_size
        candidate.cancel_ratio = 1.0 - (
            removal_size / candidate.peak_size if candidate.peak_size > 0 else 1.0
        )
        candidate.status = CandidateStatus.CANCELLED

        self._stats["candidates_removed"] += 1

        await self._emit_event(
            "analytics.spoofing.candidate_removed",
            {
                "candidate": asdict(candidate),
                "mid_at_removal": mid,
                "lifetime_ms": lifetime_ms,
            },
        )

        return True

    async def _try_confirm_removed_candidates(self, symbol: str, now_ms: int) -> None:
        candidates = self._candidates_by_symbol.get(symbol, {})
        if not candidates:
            return

        latest_mid = self._latest_mid_by_symbol.get(symbol)
        if latest_mid is None:
            return

        to_delete: List[str] = []

        for candidate_id, candidate in list(candidates.items()):
            if candidate.status != CandidateStatus.CANCELLED:
                continue

            if candidate.removed_ts_ms is None:
                continue

            time_since_removed_ms = now_ms - candidate.removed_ts_ms
            if time_since_removed_ms > self.config.trade_pressure_window_ms:
                candidate.status = CandidateStatus.EXPIRED
                to_delete.append(candidate_id)
                self._stats["expired_candidates"] += 1
                continue

            price_move_bps = self._calc_price_move_after_removal_bps(candidate, latest_mid)
            if price_move_bps is None:
                continue

            opposite_pressure_ratio = self._calc_opposite_pressure_ratio(
                symbol=symbol,
                candidate=candidate,
                now_ms=now_ms,
            )

            if price_move_bps < self.config.price_move_confirmation_bps:
                continue

            if opposite_pressure_ratio < self.config.min_opposite_pressure_ratio:
                continue

            signal = self._build_signal(
                candidate=candidate,
                confirmed_ts_ms=now_ms,
                latest_mid=latest_mid,
                price_move_bps=price_move_bps,
                opposite_pressure_ratio=opposite_pressure_ratio,
            )

            candidate.status = CandidateStatus.CONFIRMED
            candidate.confirmation_ts_ms = now_ms
            candidate.confirmation_price_move_bps = price_move_bps
            candidate.opposite_pressure_ratio = opposite_pressure_ratio

            self._set_cooldown(
                symbol=candidate.symbol,
                side=candidate.side,
                price=candidate.price,
                now_ms=now_ms,
            )

            await self._emit_event("analytics.spoofing.detected", asdict(signal))
            self._stats["signals_confirmed"] += 1
            to_delete.append(candidate_id)

            self.logger.info(
                "Spoofing confirmed",
                extra={
                    "symbol": signal.symbol,
                    "spoofing_type": signal.spoofing_type.value,
                    "level_price": signal.level_price,
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "price_move_bps": signal.price_move_bps,
                    "opposite_pressure_ratio": signal.opposite_pressure_ratio,
                },
            )

        for candidate_id in to_delete:
            candidates.pop(candidate_id, None)

    def _build_signal(
        self,
        candidate: SpoofingCandidate,
        confirmed_ts_ms: int,
        latest_mid: float,
        price_move_bps: float,
        opposite_pressure_ratio: float,
    ) -> SpoofingSignal:
        spoofing_type = (
            SpoofingType.BID_SPOOF
            if candidate.side == SpoofingSide.BID
            else SpoofingType.ASK_SPOOF
        )

        score = self._calculate_score(
            size_multiple=candidate.size_multiple_at_detection,
            cancel_ratio=candidate.cancel_ratio or 0.0,
            price_move_bps=price_move_bps,
            opposite_pressure_ratio=opposite_pressure_ratio,
            distance_bps=candidate.distance_bps_at_detection,
        )
        confidence = min(score / 100.0, 1.0)

        return SpoofingSignal(
            id=str(uuid.uuid4()),
            symbol=candidate.symbol,
            spoofing_type=spoofing_type,
            side=candidate.side,
            level_price=candidate.price,
            initial_size=candidate.initial_size,
            peak_size=candidate.peak_size,
            cancel_ratio=float(candidate.cancel_ratio or 0.0),
            detected_ts_ms=candidate.detected_ts_ms,
            removed_ts_ms=int(candidate.removed_ts_ms or confirmed_ts_ms),
            confirmed_ts_ms=confirmed_ts_ms,
            mid_at_detection=candidate.mid_at_detection,
            mid_at_confirmation=latest_mid,
            price_move_bps=price_move_bps,
            opposite_pressure_ratio=opposite_pressure_ratio,
            distance_bps_at_detection=candidate.distance_bps_at_detection,
            size_multiple_at_detection=candidate.size_multiple_at_detection,
            score=score,
            confidence=confidence,
            details={
                "lifetime_ms": (candidate.removed_ts_ms or confirmed_ts_ms) - candidate.detected_ts_ms,
                "avg_same_side_size_at_detection": candidate.avg_same_side_size_at_detection,
                "best_bid_at_detection": candidate.best_bid_at_detection,
                "best_ask_at_detection": candidate.best_ask_at_detection,
            },
        )

    async def _register_candidate(self, candidate: SpoofingCandidate) -> None:
        symbol_candidates = self._candidates_by_symbol[candidate.symbol]
        if len(symbol_candidates) >= self.config.max_candidates_per_symbol:
            oldest_id = min(symbol_candidates, key=lambda x: symbol_candidates[x].detected_ts_ms)
            symbol_candidates.pop(oldest_id, None)

        symbol_candidates[candidate.id] = candidate
        self._stats["candidates_created"] += 1

        self.logger.debug(
            "Spoofing candidate registered",
            extra={
                "candidate_id": candidate.id,
                "symbol": candidate.symbol,
                "side": candidate.side.value,
                "price": candidate.price,
                "size": candidate.initial_size,
                "distance_bps": candidate.distance_bps_at_detection,
                "size_multiple": candidate.size_multiple_at_detection,
            },
        )

        if self.config.emit_raw_candidate_events:
            await self._emit_event(
                "analytics.spoofing.candidate_detected",
                {"candidate": asdict(candidate)},
            )

    async def cleanup_expired_candidates(self) -> None:
        now_ms = self._now_ms()

        for symbol, candidates in list(self._candidates_by_symbol.items()):
            to_delete: List[str] = []
            for candidate_id, candidate in candidates.items():
                age_ms = now_ms - candidate.detected_ts_ms
                if age_ms > self.config.candidate_ttl_ms:
                    candidate.status = CandidateStatus.EXPIRED
                    to_delete.append(candidate_id)

            for candidate_id in to_delete:
                candidates.pop(candidate_id, None)
                self._stats["expired_candidates"] += 1

        for symbol in list(self._trades_by_symbol.keys()):
            self._prune_old_trades(symbol=symbol, now_ms=now_ms)

        expired_cooldowns = [
            key for key, expire_ts in self._cooldowns.items()
            if expire_ts <= now_ms
        ]
        for key in expired_cooldowns:
            self._cooldowns.pop(key, None)

        self.logger.debug(
            "SpoofingDetector cleanup completed",
            extra={
                "active_symbols": len(self._candidates_by_symbol),
                "stats": self._stats.copy(),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        active_candidates = sum(
            len(symbol_candidates)
            for symbol_candidates in self._candidates_by_symbol.values()
        )

        return {
            **self._stats,
            "is_running": self._is_running,
            "started_at_ms": self._started_at_ms,
            "active_candidates": active_candidates,
            "tracked_symbols": len(self._latest_mid_by_symbol),
        }

    def reset_stats(self) -> None:
        self._stats = {
            "orderbook_events": 0,
            "trade_events": 0,
            "candidates_created": 0,
            "candidates_removed": 0,
            "signals_confirmed": 0,
            "expired_candidates": 0,
            "invalidated_candidates": 0,
            "errors": 0,
        }

    def _symbol_allowed(self, symbol: str) -> bool:
        if not self.config.symbols:
            return True
        return symbol in self.config.symbols

    def _normalize_levels(self, levels: List[Any]) -> List[OrderBookLevel]:
        normalized: List[OrderBookLevel] = []

        for item in levels:
            if isinstance(item, OrderBookLevel):
                if item.price > 0 and item.size >= 0:
                    normalized.append(item)
                continue

            if isinstance(item, dict):
                price = float(item.get("price", 0.0))
                size = float(item.get("size", item.get("qty", 0.0)))
                if price > 0 and size >= 0:
                    normalized.append(OrderBookLevel(price=price, size=size))
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                price = float(item[0])
                size = float(item[1])
                if price > 0 and size >= 0:
                    normalized.append(OrderBookLevel(price=price, size=size))

        return normalized

    def _avg_size(self, levels: List[OrderBookLevel]) -> float:
        positive = [x.size for x in levels if x.size > 0]
        if not positive:
            return 0.0
        return sum(positive) / len(positive)

    def _is_suspicious_level(
        self,
        side: SpoofingSide,
        level: OrderBookLevel,
        avg_same_side_size: float,
        best_bid: float,
        best_ask: float,
    ) -> bool:
        if level.size < self.config.min_abs_size:
            return False

        size_multiple = (level.size / avg_same_side_size) if avg_same_side_size > 0 else 999.0
        if size_multiple < self.config.min_size_multiple_vs_avg:
            return False

        distance_bps = self._distance_bps_from_touch(
            side=side,
            price=level.price,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        if distance_bps > self.config.max_distance_bps:
            return False

        return True

    def _distance_bps_from_touch(
        self,
        side: SpoofingSide,
        price: float,
        best_bid: float,
        best_ask: float,
    ) -> float:
        if side == SpoofingSide.BID:
            if best_bid <= 0:
                return 999999.0
            return abs(best_bid - price) / best_bid * 10_000.0

        if best_ask <= 0:
            return 999999.0
        return abs(price - best_ask) / best_ask * 10_000.0

    def _has_similar_active_candidate(
        self,
        symbol: str,
        side: SpoofingSide,
        price: float,
        tolerance_bps: float = 1.0,
    ) -> bool:
        candidates = self._candidates_by_symbol.get(symbol, {})
        for candidate in candidates.values():
            if candidate.status != CandidateStatus.ACTIVE:
                continue
            if candidate.side != side:
                continue

            diff_bps = abs(candidate.price - price) / price * 10_000.0
            if diff_bps <= tolerance_bps:
                return True

        return False

    def _still_logically_valid(
        self,
        candidate: SpoofingCandidate,
        best_bid: float,
        best_ask: float,
    ) -> bool:
        """
        Якщо ринок уже сильно відійшов або рівень втратив сенс,
        candidate можна інвалідовувати.
        """
        distance_bps = self._distance_bps_from_touch(
            side=candidate.side,
            price=candidate.price,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        return distance_bps <= self.config.max_distance_bps * 3.0

    def _calc_price_move_after_removal_bps(
        self,
        candidate: SpoofingCandidate,
        latest_mid: float,
    ) -> Optional[float]:
        if candidate.mid_at_detection <= 0 or latest_mid <= 0:
            return None

        if candidate.side == SpoofingSide.BID:
            # bid spoof -> підтримка "фейкова" -> після зникнення ціна має піти вниз
            move = (candidate.mid_at_detection - latest_mid) / candidate.mid_at_detection * 10_000.0
            return max(move, 0.0)

        # ask spoof -> опір "фейковий" -> після зникнення ціна має піти вгору
        move = (latest_mid - candidate.mid_at_detection) / candidate.mid_at_detection * 10_000.0
        return max(move, 0.0)

    def _calc_opposite_pressure_ratio(
        self,
        symbol: str,
        candidate: SpoofingCandidate,
        now_ms: int,
    ) -> float:
        trades = self._trades_by_symbol.get(symbol, deque())
        if not trades:
            return 0.0

        start_ms = now_ms - self.config.trade_pressure_window_ms
        buy_qty = 0.0
        sell_qty = 0.0

        for trade in trades:
            if trade.ts_ms < start_ms:
                continue
            if trade.side == "buy":
                buy_qty += trade.qty
            elif trade.side == "sell":
                sell_qty += trade.qty

        if candidate.side == SpoofingSide.BID:
            # після bid spoof очікуємо sell pressure
            return sell_qty / max(buy_qty, 1e-9)

        # після ask spoof очікуємо buy pressure
        return buy_qty / max(sell_qty, 1e-9)

    def _calculate_score(
        self,
        size_multiple: float,
        cancel_ratio: float,
        price_move_bps: float,
        opposite_pressure_ratio: float,
        distance_bps: float,
    ) -> float:
        size_score = min(size_multiple / 8.0, 1.0) * 30.0
        cancel_score = min(cancel_ratio, 1.0) * 25.0
        move_score = min(price_move_bps / 12.0, 1.0) * 25.0
        pressure_score = min(opposite_pressure_ratio / 3.0, 1.0) * 15.0

        # чим ближче до touch, тим підозріліше
        proximity_score = max(0.0, 1.0 - (distance_bps / max(self.config.max_distance_bps, 1e-9))) * 5.0

        score = size_score + cancel_score + move_score + pressure_score + proximity_score
        return round(min(score, 100.0), 2)

    def _is_on_cooldown(
        self,
        symbol: str,
        side: SpoofingSide,
        price: float,
        now_ms: int,
    ) -> bool:
        key = self._cooldown_key(symbol, side, price)
        expire_ts = self._cooldowns.get(key)
        return expire_ts is not None and expire_ts > now_ms

    def _set_cooldown(
        self,
        symbol: str,
        side: SpoofingSide,
        price: float,
        now_ms: int,
    ) -> None:
        key = self._cooldown_key(symbol, side, price)
        self._cooldowns[key] = now_ms + self.config.cooldown_ms_same_level

    def _cooldown_key(self, symbol: str, side: SpoofingSide, price: float) -> Tuple[str, str, float]:
        return symbol, side.value, round(price, 4)

    def _prune_old_trades(self, symbol: str, now_ms: int) -> None:
        dq = self._trades_by_symbol.get(symbol)
        if not dq:
            return

        cutoff = now_ms - max(self.config.trade_pressure_window_ms * 3, 15_000)
        while dq and dq[0].ts_ms < cutoff:
            dq.popleft()

    async def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        emit = getattr(self.event_bus, "emit", None)
        if emit is None:
            self.logger.warning(
                "EventBus has no emit method",
                extra={"event_name": event_name, "payload_keys": list(payload.keys())},
            )
            return

        result = emit(event_name, payload)
        if asyncio.iscoroutine(result):
            await result

    def _safe_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(event, dict):
            return {"raw_type": str(type(event))}
        return {k: event.get(k) for k in list(event.keys())[:20]}

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)