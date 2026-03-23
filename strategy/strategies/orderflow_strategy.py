from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.event_bus import Event, EventBus, EventPriority
from core.logger import get_logger


@dataclass(slots=True)
class StrategySignalSnapshot:
    source: str
    side: str
    strength: float
    signal_type: str
    reason: str
    payload: dict[str, Any]
    timestamp: float


@dataclass(slots=True)
class OrderflowStrategyState:
    symbol: str
    imbalance_signal: Optional[StrategySignalSnapshot] = None
    aggressive_trades_signal: Optional[StrategySignalSnapshot] = None
    volume_delta_signal: Optional[StrategySignalSnapshot] = None
    cvd_signal: Optional[StrategySignalSnapshot] = None
    last_price: Optional[float] = None
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderflowStrategyDecision:
    symbol: str
    side: str
    score: float
    confirmations: int
    reason: str
    last_price: Optional[float]
    components: dict[str, dict[str, Any]]
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class OrderflowStrategyConfig:
    enabled: bool = True

    signal_ttl_sec: float = 8.0
    signal_cooldown_sec: float = 2.0
    state_cleanup_interval_sec: float = 20.0
    health_log_interval_sec: float = 30.0

    min_confirmations: int = 2
    min_score_to_emit: float = 2.2

    imbalance_weight: float = 1.0
    aggressive_trades_weight: float = 1.2
    volume_delta_weight: float = 1.0
    cvd_weight: float = 1.1

    prefer_stronger_side_only: bool = True
    allow_mixed_signals: bool = False

    max_state_age_sec: float = 120.0

    scheduler_job_timeout_sec: float = 10.0
    scheduler_job_retry_delay_sec: float = 1.0
    scheduler_job_max_retries: int = 1

    publish_priority: EventPriority = EventPriority.HIGH
    symbol_allowlist: Optional[set[str]] = None

    imbalance_topic: str = "analytics.orderbook.imbalance.signal"
    aggressive_trades_topic: str = "analytics.trades.aggressive.signal"
    volume_delta_topic: str = "analytics.trades.volume_delta.signal"
    cvd_topic: str = "analytics.trades.cvd.signal"

    strategy_signal_topic: str = "strategy.orderflow.signal"
    strategy_state_topic: str = "strategy.orderflow.state.updated"

    source_name: str = "orderflow_strategy"

    @classmethod
    def from_app_config(cls, app_config: Any) -> "OrderflowStrategyConfig":
        strategy_cfg = getattr(app_config, "strategy", None)
        orderflow_cfg = getattr(strategy_cfg, "orderflow_strategy", None) if strategy_cfg else None

        if orderflow_cfg is None:
            return cls()

        return cls(
            enabled=getattr(orderflow_cfg, "enabled", True),
            signal_ttl_sec=getattr(orderflow_cfg, "signal_ttl_sec", 8.0),
            signal_cooldown_sec=getattr(orderflow_cfg, "signal_cooldown_sec", 2.0),
            state_cleanup_interval_sec=getattr(orderflow_cfg, "state_cleanup_interval_sec", 20.0),
            health_log_interval_sec=getattr(orderflow_cfg, "health_log_interval_sec", 30.0),
            min_confirmations=getattr(orderflow_cfg, "min_confirmations", 2),
            min_score_to_emit=getattr(orderflow_cfg, "min_score_to_emit", 2.2),
            imbalance_weight=getattr(orderflow_cfg, "imbalance_weight", 1.0),
            aggressive_trades_weight=getattr(orderflow_cfg, "aggressive_trades_weight", 1.2),
            volume_delta_weight=getattr(orderflow_cfg, "volume_delta_weight", 1.0),
            cvd_weight=getattr(orderflow_cfg, "cvd_weight", 1.1),
            prefer_stronger_side_only=getattr(orderflow_cfg, "prefer_stronger_side_only", True),
            allow_mixed_signals=getattr(orderflow_cfg, "allow_mixed_signals", False),
            max_state_age_sec=getattr(orderflow_cfg, "max_state_age_sec", 120.0),
            scheduler_job_timeout_sec=getattr(
                orderflow_cfg,
                "scheduler_job_timeout_sec",
                10.0,
            ),
            scheduler_job_retry_delay_sec=getattr(
                orderflow_cfg,
                "scheduler_job_retry_delay_sec",
                1.0,
            ),
            scheduler_job_max_retries=getattr(
                orderflow_cfg,
                "scheduler_job_max_retries",
                1,
            ),
            publish_priority=getattr(
                orderflow_cfg,
                "publish_priority",
                EventPriority.HIGH,
            ),
            symbol_allowlist=set(getattr(orderflow_cfg, "symbol_allowlist", []) or []),
            imbalance_topic=getattr(
                orderflow_cfg,
                "imbalance_topic",
                "analytics.orderbook.imbalance.signal",
            ),
            aggressive_trades_topic=getattr(
                orderflow_cfg,
                "aggressive_trades_topic",
                "analytics.trades.aggressive.signal",
            ),
            volume_delta_topic=getattr(
                orderflow_cfg,
                "volume_delta_topic",
                "analytics.trades.volume_delta.signal",
            ),
            cvd_topic=getattr(
                orderflow_cfg,
                "cvd_topic",
                "analytics.trades.cvd.signal",
            ),
            strategy_signal_topic=getattr(
                orderflow_cfg,
                "strategy_signal_topic",
                "strategy.orderflow.signal",
            ),
            strategy_state_topic=getattr(
                orderflow_cfg,
                "strategy_state_topic",
                "strategy.orderflow.state.updated",
            ),
            source_name=getattr(
                orderflow_cfg,
                "source_name",
                "orderflow_strategy",
            ),
        )


class OrderflowStrategy:
    """
    Strategy layer поверх analytics/orderflow.

    Слухає signal events від:
    - OrderbookImbalance
    - AggressiveTrades
    - VolumeDelta
    - CVD

    І будує strategy-level decision по confluence.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: Optional[OrderflowStrategyConfig] = None,
        app_config: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._config = config or (
            OrderflowStrategyConfig.from_app_config(app_config)
            if app_config is not None
            else OrderflowStrategyConfig()
        )

        self._logger = get_logger(
            __name__,
            service_name=self._config.source_name,
            component="strategy",
            module="orderflow",
        )

        self._subscriptions: list[Any] = []
        self._running = False
        self._lock = asyncio.Lock()

        self._state_by_symbol: dict[str, OrderflowStrategyState] = {}
        self._last_strategy_signal_ts_by_symbol: dict[str, float] = {}

        self._health_job_id: Optional[str] = None
        self._cleanup_job_id: Optional[str] = None

        self._metrics: dict[str, Any] = {
            "processed_events": 0,
            "signals_emitted": 0,
            "states_emitted": 0,
            "skipped": 0,
            "errors": 0,
            "symbols": {},
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            self._logger.warning("OrderflowStrategy already started")
            return

        if not self._config.enabled:
            self._logger.warning("OrderflowStrategy is disabled by config")
            return

        subscriptions = [
            (self._config.imbalance_topic, self._handle_imbalance_signal, "imbalance"),
            (self._config.aggressive_trades_topic, self._handle_aggressive_trades_signal, "aggressive"),
            (self._config.volume_delta_topic, self._handle_volume_delta_signal, "volume_delta"),
            (self._config.cvd_topic, self._handle_cvd_signal, "cvd"),
        ]

        for topic, handler, name_suffix in subscriptions:
            subscription = self._event_bus.subscribe(
                pattern=topic,
                handler=handler,
                name=f"{self.__class__.__name__}:{name_suffix}",
            )
            self._subscriptions.append(subscription)

        self._register_scheduler_jobs()

        self._running = True
        self._logger.info(
            "OrderflowStrategy started | min_confirmations=%s min_score_to_emit=%.4f signal_ttl_sec=%.2f",
            self._config.min_confirmations,
            self._config.min_score_to_emit,
            self._config.signal_ttl_sec,
        )

    def stop(self) -> None:
        if not self._running:
            self._logger.warning("OrderflowStrategy already stopped")
            return

        for sub in self._subscriptions:
            try:
                self._event_bus.unsubscribe(sub)
            except Exception:
                self._logger.exception("Failed to unsubscribe OrderflowStrategy handler")

        self._subscriptions.clear()
        self._disable_scheduler_jobs()

        self._running = False
        self._logger.info("OrderflowStrategy stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, symbol: str) -> Optional[OrderflowStrategyState]:
        return self._state_by_symbol.get(symbol)

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "tracked_symbols": len(self._state_by_symbol),
            "processed_events": self._metrics["processed_events"],
            "signals_emitted": self._metrics["signals_emitted"],
            "states_emitted": self._metrics["states_emitted"],
            "skipped": self._metrics["skipped"],
            "errors": self._metrics["errors"],
            "health_job_id": self._health_job_id,
            "cleanup_job_id": self._cleanup_job_id,
            "config": {
                "signal_ttl_sec": self._config.signal_ttl_sec,
                "signal_cooldown_sec": self._config.signal_cooldown_sec,
                "min_confirmations": self._config.min_confirmations,
                "min_score_to_emit": self._config.min_score_to_emit,
                "allow_mixed_signals": self._config.allow_mixed_signals,
                "prefer_stronger_side_only": self._config.prefer_stronger_side_only,
            },
            "symbols": dict(self._metrics["symbols"]),
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_imbalance_signal(self, event: Event) -> None:
        await self._ingest_signal_event(event, source="imbalance_signal")

    async def _handle_aggressive_trades_signal(self, event: Event) -> None:
        await self._ingest_signal_event(event, source="aggressive_trades_signal")

    async def _handle_volume_delta_signal(self, event: Event) -> None:
        await self._ingest_signal_event(event, source="volume_delta_signal")

    async def _handle_cvd_signal(self, event: Event) -> None:
        await self._ingest_signal_event(event, source="cvd_signal")

    async def _ingest_signal_event(self, event: Event, *, source: str) -> None:
        symbol = self._extract_symbol_from_event(event)
        if not symbol:
            self._logger.debug(
                "OrderflowStrategy received signal without symbol | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            self._inc_metric("skipped")
            return

        if not self._should_process_symbol(symbol):
            self._inc_metric("skipped", symbol)
            return

        snapshot = self._build_snapshot(source=source, event=event)
        if snapshot is None:
            self._inc_metric("skipped", symbol)
            return

        async with self._lock:
            try:
                state = self._state_by_symbol.setdefault(
                    symbol,
                    OrderflowStrategyState(symbol=symbol),
                )

                self._apply_snapshot_to_state(state, snapshot)
                state.updated_at = time.time()
                state.last_price = self._extract_last_price_from_payload(event.payload)

                self._purge_expired_state_signals(state)
                self._inc_metric("processed_events", symbol)

                await self._emit_state_update(state)

                decision = self._evaluate_state(state)
                if decision is not None:
                    await self._emit_strategy_signal(decision)

            except Exception:
                self._inc_metric("errors", symbol)
                self._logger.exception(
                    "Failed to ingest orderflow analytics signal | symbol=%s source=%s",
                    symbol,
                    source,
                )

    # ------------------------------------------------------------------
    # State / evaluation
    # ------------------------------------------------------------------

    def _build_snapshot(self, *, source: str, event: Event) -> Optional[StrategySignalSnapshot]:
        payload = event.payload
        if not isinstance(payload, dict):
            return None

        side = payload.get("side")
        signal_type = payload.get("signal_type", "unknown")
        reason = payload.get("reason", "unknown")
        strength = payload.get("strength", 0.0)
        timestamp = payload.get("timestamp", time.time())

        if not isinstance(side, str) or side.lower() not in {"bullish", "bearish"}:
            return None

        try:
            strength_value = float(strength)
        except Exception:
            strength_value = 0.0

        try:
            timestamp_value = float(timestamp)
        except Exception:
            timestamp_value = time.time()

        return StrategySignalSnapshot(
            source=source,
            side=side.lower(),
            strength=max(0.0, strength_value),
            signal_type=str(signal_type),
            reason=str(reason),
            payload=dict(payload),
            timestamp=timestamp_value,
        )

    def _apply_snapshot_to_state(
        self,
        state: OrderflowStrategyState,
        snapshot: StrategySignalSnapshot,
    ) -> None:
        if snapshot.source == "imbalance_signal":
            state.imbalance_signal = snapshot
        elif snapshot.source == "aggressive_trades_signal":
            state.aggressive_trades_signal = snapshot
        elif snapshot.source == "volume_delta_signal":
            state.volume_delta_signal = snapshot
        elif snapshot.source == "cvd_signal":
            state.cvd_signal = snapshot

    def _purge_expired_state_signals(self, state: OrderflowStrategyState) -> None:
        now = time.time()
        ttl = self._config.signal_ttl_sec

        for attr_name in (
            "imbalance_signal",
            "aggressive_trades_signal",
            "volume_delta_signal",
            "cvd_signal",
        ):
            snapshot = getattr(state, attr_name)
            if snapshot is None:
                continue

            if now - snapshot.timestamp > ttl:
                setattr(state, attr_name, None)

    def _evaluate_state(self, state: OrderflowStrategyState) -> Optional[OrderflowStrategyDecision]:
        now = time.time()
        last_signal_ts = self._last_strategy_signal_ts_by_symbol.get(state.symbol, 0.0)

        if now - last_signal_ts < self._config.signal_cooldown_sec:
            return None

        components = self._collect_active_components(state)
        if not components:
            return None

        bullish_score, bullish_confirmations, bullish_components = self._score_side(
            state=state,
            side="bullish",
            components=components,
        )
        bearish_score, bearish_confirmations, bearish_components = self._score_side(
            state=state,
            side="bearish",
            components=components,
        )

        if not self._config.allow_mixed_signals:
            if bullish_confirmations > 0 and bearish_confirmations > 0:
                if self._config.prefer_stronger_side_only:
                    if bullish_score == bearish_score:
                        return None
                else:
                    return None

        candidate_side: Optional[str] = None
        candidate_score = 0.0
        candidate_confirmations = 0
        candidate_components: dict[str, dict[str, Any]] = {}

        if bullish_score >= bearish_score:
            candidate_side = "bullish"
            candidate_score = bullish_score
            candidate_confirmations = bullish_confirmations
            candidate_components = bullish_components
        else:
            candidate_side = "bearish"
            candidate_score = bearish_score
            candidate_confirmations = bearish_confirmations
            candidate_components = bearish_components

        if candidate_confirmations < self._config.min_confirmations:
            return None

        if candidate_score < self._config.min_score_to_emit:
            return None

        self._last_strategy_signal_ts_by_symbol[state.symbol] = now

        return OrderflowStrategyDecision(
            symbol=state.symbol,
            side=candidate_side,
            score=candidate_score,
            confirmations=candidate_confirmations,
            reason=self._build_decision_reason(candidate_side, candidate_components),
            last_price=state.last_price,
            components=candidate_components,
        )

    def _collect_active_components(
        self,
        state: OrderflowStrategyState,
    ) -> dict[str, StrategySignalSnapshot]:
        now = time.time()
        ttl = self._config.signal_ttl_sec
        result: dict[str, StrategySignalSnapshot] = {}

        mapping = {
            "imbalance": state.imbalance_signal,
            "aggressive_trades": state.aggressive_trades_signal,
            "volume_delta": state.volume_delta_signal,
            "cvd": state.cvd_signal,
        }

        for key, snapshot in mapping.items():
            if snapshot is None:
                continue
            if now - snapshot.timestamp > ttl:
                continue
            result[key] = snapshot

        return result

    def _score_side(
        self,
        *,
        state: OrderflowStrategyState,
        side: str,
        components: dict[str, StrategySignalSnapshot],
    ) -> tuple[float, int, dict[str, dict[str, Any]]]:
        score = 0.0
        confirmations = 0
        component_details: dict[str, dict[str, Any]] = {}

        weights = {
            "imbalance": self._config.imbalance_weight,
            "aggressive_trades": self._config.aggressive_trades_weight,
            "volume_delta": self._config.volume_delta_weight,
            "cvd": self._config.cvd_weight,
        }

        for component_name, snapshot in components.items():
            if snapshot.side != side:
                continue

            weight = weights[component_name]
            normalized_strength = self._normalize_strength(snapshot.strength)
            contribution = weight * normalized_strength

            score += contribution
            confirmations += 1
            component_details[component_name] = {
                "side": snapshot.side,
                "strength": snapshot.strength,
                "normalized_strength": normalized_strength,
                "weight": weight,
                "contribution": contribution,
                "signal_type": snapshot.signal_type,
                "reason": snapshot.reason,
                "timestamp": snapshot.timestamp,
            }

        return score, confirmations, component_details

    def _normalize_strength(self, strength: float) -> float:
        if strength <= 0:
            return 0.0

        if strength < 1.0:
            return strength

        if strength < 10.0:
            return 1.0 + (strength / 10.0)

        if strength < 100.0:
            return 1.5 + (strength / 100.0)

        return 2.5

    def _build_decision_reason(
        self,
        side: str,
        components: dict[str, dict[str, Any]],
    ) -> str:
        ordered = sorted(
            components.items(),
            key=lambda item: item[1]["contribution"],
            reverse=True,
        )

        parts = [f"side={side}"]
        for component_name, info in ordered:
            parts.append(
                f"{component_name}:{info['signal_type']}:{info['reason']}"
            )
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    async def _emit_state_update(self, state: OrderflowStrategyState) -> None:
        payload = {
            "symbol": state.symbol,
            "last_price": state.last_price,
            "updated_at": state.updated_at,
            "active_signals": {
                "imbalance": self._snapshot_to_payload(state.imbalance_signal),
                "aggressive_trades": self._snapshot_to_payload(state.aggressive_trades_signal),
                "volume_delta": self._snapshot_to_payload(state.volume_delta_signal),
                "cvd": self._snapshot_to_payload(state.cvd_signal),
            },
        }

        accepted = await self._event_bus.emit(
            topic=self._config.strategy_state_topic,
            payload=payload,
            priority=EventPriority.NORMAL,
            source=self._config.source_name,
            headers={
                "symbol": state.symbol,
                "strategy_type": "orderflow",
            },
        )

        if accepted:
            self._inc_metric("states_emitted", state.symbol)

    async def _emit_strategy_signal(self, decision: OrderflowStrategyDecision) -> None:
        payload = {
            "symbol": decision.symbol,
            "side": decision.side,
            "score": decision.score,
            "strength": decision.score,
            "confirmations": decision.confirmations,
            "reason": decision.reason,
            "signal_type": "orderflow_confluence",
            "last_price": decision.last_price,
            "components": decision.components,
            "timestamp": decision.timestamp,
        }

        accepted = await self._event_bus.emit(
            topic=self._config.strategy_signal_topic,
            payload=payload,
            priority=self._config.publish_priority,
            source=self._config.source_name,
            headers={
                "symbol": decision.symbol,
                "strategy_type": "orderflow",
                "side": decision.side,
            },
        )

        if accepted:
            self._inc_metric("signals_emitted", decision.symbol)
            self._logger.info(
                "Orderflow strategy signal emitted | symbol=%s side=%s score=%.4f confirmations=%s",
                decision.symbol,
                decision.side,
                decision.score,
                decision.confirmations,
            )

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        try:
            if hasattr(self._scheduler, "get_job_by_name"):
                existing_health = self._scheduler.get_job_by_name("orderflow_strategy_health")
                if existing_health is not None:
                    self._health_job_id = existing_health.job_id

                existing_cleanup = self._scheduler.get_job_by_name("orderflow_strategy_cleanup")
                if existing_cleanup is not None:
                    self._cleanup_job_id = existing_cleanup.job_id

            if self._health_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._health_job_id = self._scheduler.add_interval_job(
                    name="orderflow_strategy_health",
                    func=self._log_health_snapshot,
                    interval=self._config.health_log_interval_sec,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_sec,
                    timeout=self._config.scheduler_job_timeout_sec,
                    allow_overlap=False,
                    enabled=True,
                )

                self._logger.info(
                    "OrderflowStrategy health scheduler job registered | job_id=%s",
                    self._health_job_id,
                )

            if self._cleanup_job_id is None and hasattr(self._scheduler, "add_interval_job"):
                self._cleanup_job_id = self._scheduler.add_interval_job(
                    name="orderflow_strategy_cleanup",
                    func=self._cleanup_stale_state,
                    interval=self._config.state_cleanup_interval_sec,
                    run_immediately=False,
                    max_retries=self._config.scheduler_job_max_retries,
                    retry_delay=self._config.scheduler_job_retry_delay_sec,
                    timeout=self._config.scheduler_job_timeout_sec,
                    allow_overlap=False,
                    enabled=True,
                )

                self._logger.info(
                    "OrderflowStrategy cleanup scheduler job registered | job_id=%s",
                    self._cleanup_job_id,
                )

        except Exception:
            self._logger.exception("Failed to register scheduler jobs for OrderflowStrategy")

    def _disable_scheduler_jobs(self) -> None:
        if self._scheduler is None:
            return

        for job_id, job_name in (
            (self._health_job_id, "orderflow_strategy_health"),
            (self._cleanup_job_id, "orderflow_strategy_cleanup"),
        ):
            if job_id is None:
                continue

            try:
                if hasattr(self._scheduler, "disable_job"):
                    self._scheduler.disable_job(job_id)
                    self._logger.info(
                        "OrderflowStrategy scheduler job disabled | name=%s job_id=%s",
                        job_name,
                        job_id,
                    )
            except Exception:
                self._logger.exception(
                    "Failed to disable OrderflowStrategy scheduler job | name=%s job_id=%s",
                    job_name,
                    job_id,
                )

    async def _log_health_snapshot(self) -> None:
        self._logger.info(
            "OrderflowStrategy health | running=%s tracked_symbols=%s processed_events=%s signals=%s errors=%s",
            self._running,
            len(self._state_by_symbol),
            self._metrics["processed_events"],
            self._metrics["signals_emitted"],
            self._metrics["errors"],
        )

    async def _cleanup_stale_state(self) -> None:
        async with self._lock:
            now = time.time()
            removed = 0

            for symbol in list(self._state_by_symbol.keys()):
                state = self._state_by_symbol[symbol]
                if now - state.updated_at <= self._config.max_state_age_sec:
                    continue

                self._state_by_symbol.pop(symbol, None)
                self._last_strategy_signal_ts_by_symbol.pop(symbol, None)
                removed += 1

            self._logger.debug(
                "OrderflowStrategy cleanup finished | removed=%s tracked_symbols=%s",
                removed,
                len(self._state_by_symbol),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_to_payload(
        self,
        snapshot: Optional[StrategySignalSnapshot],
    ) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None

        return {
            "source": snapshot.source,
            "side": snapshot.side,
            "strength": snapshot.strength,
            "signal_type": snapshot.signal_type,
            "reason": snapshot.reason,
            "timestamp": snapshot.timestamp,
            "payload": snapshot.payload,
        }

    def _extract_symbol_from_event(self, event: Event) -> Optional[str]:
        payload = event.payload

        if isinstance(payload, dict):
            symbol = payload.get("symbol") or payload.get("instrument") or payload.get("pair")
            if symbol:
                return str(symbol)

        if event.headers:
            header_symbol = event.headers.get("symbol")
            if header_symbol:
                return str(header_symbol)

        return None

    def _extract_last_price_from_payload(self, payload: Any) -> Optional[float]:
        if not isinstance(payload, dict):
            return None

        for key in ("last_price", "mid_price", "price", "best_bid", "best_ask"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue

        return None

    def _should_process_symbol(self, symbol: str) -> bool:
        allowlist = self._config.symbol_allowlist
        if not allowlist:
            return True
        return symbol in allowlist

    def _inc_metric(self, key: str, symbol: Optional[str] = None, amount: int = 1) -> None:
        self._metrics[key] = self._metrics.get(key, 0) + amount

        if symbol:
            symbols = self._metrics.setdefault("symbols", {})
            symbol_stats = symbols.setdefault(
                symbol,
                {
                    "processed_events": 0,
                    "signals_emitted": 0,
                    "states_emitted": 0,
                    "skipped": 0,
                    "errors": 0,
                },
            )
            if key in symbol_stats:
                symbol_stats[key] += amount