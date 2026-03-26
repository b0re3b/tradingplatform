from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from core.logger import get_logger


class SpreadSide(str, Enum):
    BUY_A_SELL_B = "buy_a_sell_b"
    BUY_B_SELL_A = "buy_b_sell_a"


class OpportunityStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass(slots=True)
class ExchangeFee:
    maker: Decimal = Decimal("0")
    taker: Decimal = Decimal("0")
    withdrawal: Decimal = Decimal("0")

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "ExchangeFee":
        if not data:
            return cls()
        return cls(
            maker=_d(data.get("maker", 0)),
            taker=_d(data.get("taker", 0)),
            withdrawal=_d(data.get("withdrawal", 0)),
        )


@dataclass(slots=True)
class BestQuote:
    exchange: str
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    ts_exchange: float
    ts_received: float
    sequence: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def mid(self) -> Decimal:
        if self.bid <= 0 or self.ask <= 0:
            return Decimal("0")
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_abs(self) -> Decimal:
        return max(Decimal("0"), self.ask - self.bid)

    @property
    def is_valid(self) -> bool:
        return (
            self.bid > 0
            and self.ask > 0
            and self.ask >= self.bid
            and self.bid_size >= 0
            and self.ask_size >= 0
        )


@dataclass(slots=True)
class SpreadOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    executable_size: Decimal
    gross_spread_abs: Decimal
    gross_spread_bps: Decimal
    estimated_fees_abs: Decimal
    estimated_slippage_abs: Decimal
    net_spread_abs: Decimal
    net_spread_bps: Decimal
    confidence: Decimal
    side: SpreadSide
    detected_at: float
    latency_ms_buy: float
    latency_ms_sell: float
    quote_age_diff_ms: float
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, Enum):
                data[key] = value.value
        return data


@dataclass(slots=True)
class SpreadStats:
    quotes_ingested: int = 0
    opportunities_emitted: int = 0
    opportunities_rejected: int = 0
    stale_quotes_dropped: int = 0
    invalid_quotes_dropped: int = 0
    symbols_evaluated: int = 0
    pairs_evaluated: int = 0
    cooldown_skips: int = 0
    same_exchange_skips: int = 0
    insufficient_size_skips: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class CrossExchangeSpreadsConfig:
    enabled: bool = True
    symbols: Tuple[str, ...] = ()
    exchanges: Tuple[str, ...] = ()
    min_net_spread_bps: Decimal = Decimal("4")
    min_gross_spread_bps: Decimal = Decimal("6")
    min_executable_notional_usdt: Decimal = Decimal("100")
    min_confidence: Decimal = Decimal("0.55")
    max_quote_age_ms: int = 1800
    max_cross_exchange_time_diff_ms: int = 800
    opportunity_ttl_ms: int = 1500
    evaluation_cooldown_ms: int = 300
    default_slippage_bps: Decimal = Decimal("1.5")
    slippage_multiplier: Decimal = Decimal("1.0")
    emit_rejections: bool = False
    publish_topic: str = "analytics.cross_exchange_spreads.opportunity"
    rejection_topic: str = "analytics.cross_exchange_spreads.rejection"
    metrics_topic: str = "analytics.cross_exchange_spreads.metrics"
    health_topic: str = "analytics.cross_exchange_spreads.health"
    log_snapshots: bool = False
    snapshot_depth: int = 200

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "CrossExchangeSpreadsConfig":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            symbols=tuple(data.get("symbols", ()) or ()),
            exchanges=tuple(data.get("exchanges", ()) or ()),
            min_net_spread_bps=_d(data.get("min_net_spread_bps", 4)),
            min_gross_spread_bps=_d(data.get("min_gross_spread_bps", 6)),
            min_executable_notional_usdt=_d(data.get("min_executable_notional_usdt", 100)),
            min_confidence=_d(data.get("min_confidence", "0.55")),
            max_quote_age_ms=int(data.get("max_quote_age_ms", 1800)),
            max_cross_exchange_time_diff_ms=int(data.get("max_cross_exchange_time_diff_ms", 800)),
            opportunity_ttl_ms=int(data.get("opportunity_ttl_ms", 1500)),
            evaluation_cooldown_ms=int(data.get("evaluation_cooldown_ms", 300)),
            default_slippage_bps=_d(data.get("default_slippage_bps", "1.5")),
            slippage_multiplier=_d(data.get("slippage_multiplier", "1.0")),
            emit_rejections=bool(data.get("emit_rejections", False)),
            publish_topic=str(
                data.get("publish_topic", "analytics.cross_exchange_spreads.opportunity")
            ),
            rejection_topic=str(
                data.get("rejection_topic", "analytics.cross_exchange_spreads.rejection")
            ),
            metrics_topic=str(data.get("metrics_topic", "analytics.cross_exchange_spreads.metrics")),
            health_topic=str(data.get("health_topic", "analytics.cross_exchange_spreads.health")),
            log_snapshots=bool(data.get("log_snapshots", False)),
            snapshot_depth=int(data.get("snapshot_depth", 200)),
        )


class CrossExchangeSpreads:
    """
    Analytics-модуль для пошуку крос-біржових спредів.

    Концепція:
    - слухає нормалізовані top-of-book / best bid-ask події
    - зберігає останні котирування по symbol/exchange
    - порівнює біржі між собою
    - оцінює gross/net spread з урахуванням fees/slippage
    - публікує opportunity events у EventBus

    Очікуваний формат події:
    {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "bid": "65000.10",
        "ask": "65000.30",
        "bid_size": "0.8",
        "ask_size": "1.2",
        "ts_exchange": 1710000000.123,
        "ts_received": 1710000000.130,
        "sequence": 12345,
        ...
    }
    """

    def __init__(
        self,
        *,
        event_bus: Any,
        config: Optional[CrossExchangeSpreadsConfig | Mapping[str, Any]] = None,
        scheduler: Optional[Any] = None,
        module_name: str = "CrossExchangeSpreads",
        exchange_fees: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = (
            config
            if isinstance(config, CrossExchangeSpreadsConfig)
            else CrossExchangeSpreadsConfig.from_mapping(config)
        )
        self.module_name = module_name

        self.logger = get_logger(
            __name__,
            service_name="analytics.cross_exchange_spreads",
            component=self.module_name,
        )

        self._quotes: Dict[str, Dict[str, BestQuote]] = defaultdict(dict)
        self._fees: Dict[str, ExchangeFee] = {
            exchange.lower(): ExchangeFee.from_mapping(data)
            for exchange, data in (exchange_fees or {}).items()
        }
        self._last_eval_ts: Dict[str, float] = {}
        self._active_opportunities: Dict[str, SpreadOpportunity] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.snapshot_depth)
        self._stats = SpreadStats()

        self._running = False
        self._lock = asyncio.Lock()
        self._subscriptions: List[Any] = []
        self._scheduler_jobs: List[str] = []

        self.logger.info(
            "CrossExchangeSpreads initialized",
            extra={
                "enabled": self.config.enabled,
                "symbols": list(self.config.symbols),
                "exchanges": list(self.config.exchanges),
                "min_net_spread_bps": str(self.config.min_net_spread_bps),
                "max_quote_age_ms": self.config.max_quote_age_ms,
            },
        )

    # =========================
    # Lifecycle
    # =========================

    async def start(self) -> None:
        if self._running:
            self.logger.warning("CrossExchangeSpreads already running")
            return

        self._running = True

        await self._subscribe_events()
        await self._register_jobs()

        self.logger.info(
            "CrossExchangeSpreads started",
            extra={"module": self.module_name},
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        await self._unsubscribe_events()
        await self._remove_jobs()

        self.logger.info(
            "CrossExchangeSpreads stopped",
            extra={
                "stats": self._stats.to_dict(),
                "active_opportunities": len(self._active_opportunities),
            },
        )

    @property
    def is_running(self) -> bool:
        return self._running

    # =========================
    # Event wiring
    # =========================

    async def _subscribe_events(self) -> None:
        """
        Підписки підлаштуй під свій EventBus API.
        Нижче — концептуально нейтральна форма.
        """
        if hasattr(self.event_bus, "subscribe"):
            handlers = [
                ("market.best_quote", self.on_best_quote),
                ("market.ticker", self.on_best_quote),
                ("market.orderbook.top", self.on_best_quote),
            ]

            for topic, handler in handlers:
                try:
                    sub = await _maybe_await(self.event_bus.subscribe(topic, handler))
                    self._subscriptions.append((topic, handler, sub))
                    self.logger.info(
                        "Subscribed to topic",
                        extra={"topic": topic},
                    )
                except Exception as exc:
                    self.logger.exception(
                        "Failed to subscribe",
                        extra={"topic": topic, "error": str(exc)},
                    )

    async def _unsubscribe_events(self) -> None:
        if not hasattr(self.event_bus, "unsubscribe"):
            self._subscriptions.clear()
            return

        for topic, handler, sub in self._subscriptions:
            try:
                if sub is not None:
                    await _maybe_await(self.event_bus.unsubscribe(topic, sub))
                else:
                    await _maybe_await(self.event_bus.unsubscribe(topic, handler))
            except Exception as exc:
                self.logger.exception(
                    "Failed to unsubscribe",
                    extra={"topic": topic, "error": str(exc)},
                )
        self._subscriptions.clear()

    async def _register_jobs(self) -> None:
        if self.scheduler is None:
            return

        if hasattr(self.scheduler, "add_interval_job"):
            try:
                job = await _maybe_await(
                    self.scheduler.add_interval_job(
                        name="cross_exchange_spreads_prune",
                        interval_seconds=max(1, self.config.opportunity_ttl_ms // 1000),
                        coro=self.prune_expired_opportunities,
                        enabled=True,
                        tags=["analytics", "cross_exchange_spreads"],
                    )
                )
                self._scheduler_jobs.append(getattr(job, "id", "cross_exchange_spreads_prune"))
            except Exception as exc:
                self.logger.exception(
                    "Failed to register prune job",
                    extra={"error": str(exc)},
                )

            try:
                job = await _maybe_await(
                    self.scheduler.add_interval_job(
                        name="cross_exchange_spreads_emit_health",
                        interval_seconds=5,
                        coro=self.emit_health,
                        enabled=True,
                        tags=["analytics", "cross_exchange_spreads"],
                    )
                )
                self._scheduler_jobs.append(
                    getattr(job, "id", "cross_exchange_spreads_emit_health")
                )
            except Exception as exc:
                self.logger.exception(
                    "Failed to register health job",
                    extra={"error": str(exc)},
                )

    async def _remove_jobs(self) -> None:
        if self.scheduler is None or not hasattr(self.scheduler, "remove_job"):
            self._scheduler_jobs.clear()
            return

        for job_id in self._scheduler_jobs:
            try:
                await _maybe_await(self.scheduler.remove_job(job_id))
            except Exception as exc:
                self.logger.exception(
                    "Failed to remove scheduler job",
                    extra={"job_id": job_id, "error": str(exc)},
                )
        self._scheduler_jobs.clear()

    # =========================
    # Public ingestion API
    # =========================

    async def on_best_quote(self, event: Mapping[str, Any]) -> None:
        if not self._running or not self.config.enabled:
            return

        quote = self._normalize_quote(event)
        if quote is None:
            return

        async with self._lock:
            self._stats.quotes_ingested += 1
            self._quotes[quote.symbol][quote.exchange] = quote

            if self.config.log_snapshots:
                self._history.append(
                    {
                        "type": "quote",
                        "ts": time.time(),
                        "symbol": quote.symbol,
                        "exchange": quote.exchange,
                        "bid": str(quote.bid),
                        "ask": str(quote.ask),
                        "bid_size": str(quote.bid_size),
                        "ask_size": str(quote.ask_size),
                    }
                )

            if self._should_skip_eval_due_to_cooldown(quote.symbol):
                self._stats.cooldown_skips += 1
                return

            self._last_eval_ts[quote.symbol] = time.time()

            await self._evaluate_symbol(quote.symbol)

    async def update_fee(self, exchange: str, *, maker: Any = None, taker: Any = None, withdrawal: Any = None) -> None:
        exchange = exchange.lower().strip()
        current = self._fees.get(exchange, ExchangeFee())

        self._fees[exchange] = ExchangeFee(
            maker=_d(maker) if maker is not None else current.maker,
            taker=_d(taker) if taker is not None else current.taker,
            withdrawal=_d(withdrawal) if withdrawal is not None else current.withdrawal,
        )

        self.logger.info(
            "Exchange fee updated",
            extra={
                "exchange": exchange,
                "maker": str(self._fees[exchange].maker),
                "taker": str(self._fees[exchange].taker),
                "withdrawal": str(self._fees[exchange].withdrawal),
            },
        )

    async def force_evaluate(self, symbol: str) -> None:
        async with self._lock:
            await self._evaluate_symbol(symbol.upper().strip())

    # =========================
    # Core logic
    # =========================

    async def _evaluate_symbol(self, symbol: str) -> None:
        quotes_map = self._quotes.get(symbol, {})
        if len(quotes_map) < 2:
            return

        self._stats.symbols_evaluated += 1

        quotes = [q for q in quotes_map.values() if self._is_quote_usable(q)]
        if len(quotes) < 2:
            return

        best_opportunity: Optional[SpreadOpportunity] = None
        best_rejection: Optional[Dict[str, Any]] = None

        for i in range(len(quotes)):
            for j in range(i + 1, len(quotes)):
                qa = quotes[i]
                qb = quotes[j]

                self._stats.pairs_evaluated += 1

                candidate_ab = self._build_opportunity(buy_quote=qa, sell_quote=qb)
                candidate_ba = self._build_opportunity(buy_quote=qb, sell_quote=qa)

                for candidate in (candidate_ab, candidate_ba):
                    if candidate is None:
                        continue

                    accepted, reason = self._accept_opportunity(candidate)

                    if accepted:
                        if (
                            best_opportunity is None
                            or candidate.net_spread_bps > best_opportunity.net_spread_bps
                        ):
                            best_opportunity = candidate
                    else:
                        if best_rejection is None:
                            best_rejection = {
                                "symbol": symbol,
                                "reason": reason,
                                "candidate": candidate.to_payload(),
                            }

        if best_opportunity:
            await self._emit_opportunity(best_opportunity)
        elif best_rejection and self.config.emit_rejections:
            await self._emit_rejection(best_rejection)

    def _build_opportunity(
        self,
        *,
        buy_quote: BestQuote,
        sell_quote: BestQuote,
    ) -> Optional[SpreadOpportunity]:
        if buy_quote.exchange == sell_quote.exchange:
            self._stats.same_exchange_skips += 1
            return None

        if buy_quote.ask <= 0 or sell_quote.bid <= 0:
            return None

        gross_spread_abs = sell_quote.bid - buy_quote.ask
        if gross_spread_abs <= 0:
            return None

        executable_size = min(buy_quote.ask_size, sell_quote.bid_size)
        if executable_size <= 0:
            self._stats.insufficient_size_skips += 1
            return None

        reference_price = max(buy_quote.ask, Decimal("0.00000001"))
        gross_spread_bps = (gross_spread_abs / reference_price) * Decimal("10000")

        fees_abs = self._estimate_total_fees_abs(
            buy_exchange=buy_quote.exchange,
            sell_exchange=sell_quote.exchange,
            buy_price=buy_quote.ask,
            sell_price=sell_quote.bid,
            size=executable_size,
        )

        slippage_abs = self._estimate_slippage_abs(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            size=executable_size,
        )

        net_spread_abs = gross_spread_abs - fees_abs - slippage_abs
        net_spread_bps = (net_spread_abs / reference_price) * Decimal("10000")

        now = time.time()
        latency_ms_buy = max(0.0, (now - buy_quote.ts_received) * 1000.0)
        latency_ms_sell = max(0.0, (now - sell_quote.ts_received) * 1000.0)
        quote_age_diff_ms = abs(buy_quote.ts_received - sell_quote.ts_received) * 1000.0

        confidence = self._estimate_confidence(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            gross_spread_bps=gross_spread_bps,
            net_spread_bps=net_spread_bps,
            executable_size=executable_size,
            quote_age_diff_ms=quote_age_diff_ms,
        )

        return SpreadOpportunity(
            symbol=buy_quote.symbol,
            buy_exchange=buy_quote.exchange,
            sell_exchange=sell_quote.exchange,
            buy_price=buy_quote.ask,
            sell_price=sell_quote.bid,
            executable_size=executable_size,
            gross_spread_abs=gross_spread_abs,
            gross_spread_bps=gross_spread_bps,
            estimated_fees_abs=fees_abs,
            estimated_slippage_abs=slippage_abs,
            net_spread_abs=net_spread_abs,
            net_spread_bps=net_spread_bps,
            confidence=confidence,
            side=SpreadSide.BUY_A_SELL_B,
            detected_at=now,
            latency_ms_buy=latency_ms_buy,
            latency_ms_sell=latency_ms_sell,
            quote_age_diff_ms=quote_age_diff_ms,
            meta={
                "buy_mid": str(buy_quote.mid),
                "sell_mid": str(sell_quote.mid),
                "buy_bid": str(buy_quote.bid),
                "buy_ask": str(buy_quote.ask),
                "sell_bid": str(sell_quote.bid),
                "sell_ask": str(sell_quote.ask),
                "buy_sequence": buy_quote.sequence,
                "sell_sequence": sell_quote.sequence,
            },
        )

    def _accept_opportunity(self, opportunity: SpreadOpportunity) -> Tuple[bool, str]:
        if opportunity.gross_spread_bps < self.config.min_gross_spread_bps:
            self._stats.opportunities_rejected += 1
            return False, "gross_spread_below_threshold"

        if opportunity.net_spread_bps < self.config.min_net_spread_bps:
            self._stats.opportunities_rejected += 1
            return False, "net_spread_below_threshold"

        notional = opportunity.buy_price * opportunity.executable_size
        if notional < self.config.min_executable_notional_usdt:
            self._stats.opportunities_rejected += 1
            return False, "notional_below_threshold"

        if opportunity.quote_age_diff_ms > self.config.max_cross_exchange_time_diff_ms:
            self._stats.opportunities_rejected += 1
            return False, "quote_time_diff_too_large"

        if opportunity.confidence < self.config.min_confidence:
            self._stats.opportunities_rejected += 1
            return False, "confidence_below_threshold"

        return True, "accepted"

    def _estimate_total_fees_abs(
        self,
        *,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: Decimal,
        sell_price: Decimal,
        size: Decimal,
    ) -> Decimal:
        buy_fee = self._fees.get(buy_exchange.lower(), ExchangeFee())
        sell_fee = self._fees.get(sell_exchange.lower(), ExchangeFee())

        buy_notional = buy_price * size
        sell_notional = sell_price * size

        buy_fee_abs = buy_notional * buy_fee.taker
        sell_fee_abs = sell_notional * sell_fee.taker

        return buy_fee_abs + sell_fee_abs

    def _estimate_slippage_abs(
        self,
        *,
        buy_quote: BestQuote,
        sell_quote: BestQuote,
        size: Decimal,
    ) -> Decimal:
        """
        Базова модель:
        - беремо default_slippage_bps
        - штрафуємо за низьку ліквідність top-of-book
        - масштабуємо через slippage_multiplier
        """
        base_bps = self.config.default_slippage_bps

        buy_liquidity_penalty = self._liquidity_penalty(size=size, visible_size=buy_quote.ask_size)
        sell_liquidity_penalty = self._liquidity_penalty(size=size, visible_size=sell_quote.bid_size)

        total_bps = (base_bps + buy_liquidity_penalty + sell_liquidity_penalty) * self.config.slippage_multiplier

        ref_price = max(buy_quote.ask, sell_quote.bid)
        return (ref_price * size) * (total_bps / Decimal("10000"))

    def _liquidity_penalty(self, *, size: Decimal, visible_size: Decimal) -> Decimal:
        if visible_size <= 0:
            return Decimal("50")

        ratio = size / visible_size
        if ratio <= Decimal("0.25"):
            return Decimal("0.20")
        if ratio <= Decimal("0.50"):
            return Decimal("0.50")
        if ratio <= Decimal("0.80"):
            return Decimal("1.20")
        if ratio <= Decimal("1.00"):
            return Decimal("2.50")
        return Decimal("5.00")

    def _estimate_confidence(
        self,
        *,
        buy_quote: BestQuote,
        sell_quote: BestQuote,
        gross_spread_bps: Decimal,
        net_spread_bps: Decimal,
        executable_size: Decimal,
        quote_age_diff_ms: float,
    ) -> Decimal:
        """
        Confidence score [0..1]:
        - вищий net spread => краще
        - менша розсинхронізація котирувань => краще
        - більший executable size => краще
        - надто широкий own spread на біржі => гірше
        """
        score = Decimal("0.50")

        if net_spread_bps >= self.config.min_net_spread_bps * Decimal("2"):
            score += Decimal("0.20")
        elif net_spread_bps >= self.config.min_net_spread_bps:
            score += Decimal("0.10")

        if gross_spread_bps >= self.config.min_gross_spread_bps * Decimal("2"):
            score += Decimal("0.05")

        if quote_age_diff_ms <= self.config.max_cross_exchange_time_diff_ms * 0.25:
            score += Decimal("0.10")
        elif quote_age_diff_ms <= self.config.max_cross_exchange_time_diff_ms * 0.50:
            score += Decimal("0.05")
        else:
            score -= Decimal("0.10")

        if executable_size > 0:
            score += Decimal("0.05")

        buy_own_spread_bps = _safe_bps(buy_quote.ask - buy_quote.bid, buy_quote.ask)
        sell_own_spread_bps = _safe_bps(sell_quote.ask - sell_quote.bid, sell_quote.ask)

        if buy_own_spread_bps > Decimal("5"):
            score -= Decimal("0.05")
        if sell_own_spread_bps > Decimal("5"):
            score -= Decimal("0.05")

        return min(Decimal("1.0"), max(Decimal("0.0"), score))

    # =========================
    # Emit / health / metrics
    # =========================

    async def _emit_opportunity(self, opportunity: SpreadOpportunity) -> None:
        key = self._make_opportunity_key(
            opportunity.symbol,
            opportunity.buy_exchange,
            opportunity.sell_exchange,
        )
        self._active_opportunities[key] = opportunity
        self._stats.opportunities_emitted += 1

        payload = {
            "event_type": "cross_exchange_spread_opportunity",
            "module": self.module_name,
            "symbol": opportunity.symbol,
            "opportunity": opportunity.to_payload(),
            "ts": time.time(),
        }

        if self.config.log_snapshots:
            self._history.append(
                {
                    "type": "opportunity",
                    "ts": time.time(),
                    "payload": payload,
                }
            )

        try:
            if hasattr(self.event_bus, "emit"):
                await _maybe_await(self.event_bus.emit(self.config.publish_topic, payload))
        except Exception as exc:
            self.logger.exception(
                "Failed to emit opportunity",
                extra={"error": str(exc), "payload": payload},
            )
            return

        self.logger.info(
            "Cross-exchange opportunity detected",
            extra={
                "symbol": opportunity.symbol,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "buy_price": str(opportunity.buy_price),
                "sell_price": str(opportunity.sell_price),
                "size": str(opportunity.executable_size),
                "net_spread_bps": str(opportunity.net_spread_bps),
                "confidence": str(opportunity.confidence),
            },
        )

    async def _emit_rejection(self, rejection: Dict[str, Any]) -> None:
        try:
            if hasattr(self.event_bus, "emit"):
                await _maybe_await(self.event_bus.emit(self.config.rejection_topic, rejection))
        except Exception as exc:
            self.logger.exception(
                "Failed to emit rejection",
                extra={"error": str(exc), "rejection": rejection},
            )

    async def emit_health(self) -> None:
        payload = self.get_health_snapshot()

        try:
            if hasattr(self.event_bus, "emit"):
                await _maybe_await(self.event_bus.emit(self.config.health_topic, payload))
        except Exception as exc:
            self.logger.exception(
                "Failed to emit health snapshot",
                extra={"error": str(exc)},
            )

    async def emit_metrics(self) -> None:
        payload = {
            "event_type": "cross_exchange_spreads_metrics",
            "module": self.module_name,
            "stats": self._stats.to_dict(),
            "symbols_tracked": len(self._quotes),
            "active_opportunities": len(self._active_opportunities),
            "ts": time.time(),
        }

        try:
            if hasattr(self.event_bus, "emit"):
                await _maybe_await(self.event_bus.emit(self.config.metrics_topic, payload))
        except Exception as exc:
            self.logger.exception(
                "Failed to emit metrics",
                extra={"error": str(exc)},
            )

    async def prune_expired_opportunities(self) -> None:
        now = time.time()
        ttl_sec = self.config.opportunity_ttl_ms / 1000.0

        expired_keys = [
            key
            for key, opp in self._active_opportunities.items()
            if (now - opp.detected_at) > ttl_sec
        ]

        for key in expired_keys:
            opp = self._active_opportunities.pop(key)
            opp.status = OpportunityStatus.EXPIRED

            self.logger.debug(
                "Expired opportunity removed",
                extra={
                    "symbol": opp.symbol,
                    "buy_exchange": opp.buy_exchange,
                    "sell_exchange": opp.sell_exchange,
                },
            )

    # =========================
    # Query / snapshot API
    # =========================

    def get_symbol_snapshot(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper().strip()
        quotes = self._quotes.get(symbol, {})

        return {
            "symbol": symbol,
            "quotes": {
                exchange: {
                    "bid": str(quote.bid),
                    "ask": str(quote.ask),
                    "bid_size": str(quote.bid_size),
                    "ask_size": str(quote.ask_size),
                    "ts_exchange": quote.ts_exchange,
                    "ts_received": quote.ts_received,
                    "is_valid": quote.is_valid,
                }
                for exchange, quote in quotes.items()
            },
            "last_eval_ts": self._last_eval_ts.get(symbol),
        }

    def get_health_snapshot(self) -> Dict[str, Any]:
        now = time.time()

        stale_quotes = 0
        total_quotes = 0
        for _, qmap in self._quotes.items():
            for q in qmap.values():
                total_quotes += 1
                if ((now - q.ts_received) * 1000.0) > self.config.max_quote_age_ms:
                    stale_quotes += 1

        return {
            "event_type": "cross_exchange_spreads_health",
            "module": self.module_name,
            "running": self._running,
            "enabled": self.config.enabled,
            "symbols_tracked": len(self._quotes),
            "total_quotes": total_quotes,
            "stale_quotes": stale_quotes,
            "active_opportunities": len(self._active_opportunities),
            "stats": self._stats.to_dict(),
            "ts": now,
        }

    def get_recent_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._history)[-limit:]

    # =========================
    # Helpers
    # =========================

    def _normalize_quote(self, event: Mapping[str, Any]) -> Optional[BestQuote]:
        try:
            exchange = str(event.get("exchange", "")).lower().strip()
            symbol = str(event.get("symbol", "")).upper().strip()

            if not exchange or not symbol:
                self._stats.invalid_quotes_dropped += 1
                return None

            if self.config.symbols and symbol not in self.config.symbols:
                return None

            if self.config.exchanges and exchange not in self.config.exchanges:
                return None

            quote = BestQuote(
                exchange=exchange,
                symbol=symbol,
                bid=_d(event.get("bid", 0)),
                ask=_d(event.get("ask", 0)),
                bid_size=_d(event.get("bid_size", 0)),
                ask_size=_d(event.get("ask_size", 0)),
                ts_exchange=float(event.get("ts_exchange", event.get("timestamp", time.time()))),
                ts_received=float(event.get("ts_received", time.time())),
                sequence=event.get("sequence"),
                raw=dict(event),
            )

            if not quote.is_valid:
                self._stats.invalid_quotes_dropped += 1
                self.logger.debug(
                    "Invalid quote dropped",
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "bid": str(quote.bid),
                        "ask": str(quote.ask),
                    },
                )
                return None

            age_ms = (time.time() - quote.ts_received) * 1000.0
            if age_ms > self.config.max_quote_age_ms:
                self._stats.stale_quotes_dropped += 1
                self.logger.debug(
                    "Stale quote dropped",
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "age_ms": age_ms,
                    },
                )
                return None

            return quote

        except Exception as exc:
            self._stats.invalid_quotes_dropped += 1
            self.logger.exception(
                "Failed to normalize quote",
                extra={"error": str(exc), "event": dict(event)},
            )
            return None

    def _is_quote_usable(self, quote: BestQuote) -> bool:
        if not quote.is_valid:
            return False
        age_ms = (time.time() - quote.ts_received) * 1000.0
        return age_ms <= self.config.max_quote_age_ms

    def _should_skip_eval_due_to_cooldown(self, symbol: str) -> bool:
        last_ts = self._last_eval_ts.get(symbol)
        if last_ts is None:
            return False
        elapsed_ms = (time.time() - last_ts) * 1000.0
        return elapsed_ms < self.config.evaluation_cooldown_ms

    def _make_opportunity_key(self, symbol: str, buy_exchange: str, sell_exchange: str) -> str:
        return f"{symbol}:{buy_exchange}->{sell_exchange}"


def _d(value: Any) -> Decimal:
    try:
        if isinstance(value, Decimal):
            return value
        if value is None:
            return Decimal("0")
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def _safe_bps(abs_value: Decimal, price: Decimal) -> Decimal:
    if price <= 0:
        return Decimal("0")
    return (abs_value / price) * Decimal("10000")


async def _maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result