from __future__ import annotations

from analytics.strategy_contract import ensure_strategy_payload_contract
from datetime import datetime, timedelta
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.liquidations.config import CascadeDetectorConfig
from analytics.liquidations.enums import LiquidationEventType, LiquidationStatus
from analytics.liquidations.metrics import LiquidationMetrics
from analytics.liquidations.models import (
    CascadeDetectionResult,
    LiquidationEvent,
    LiquidationKey,
    LiquidationWindowStats,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)
from analytics.liquidations.state import LiquidationState, SymbolLiquidationState, get_shared_liquidation_state
from analytics.liquidations.utils import (
    build_cluster_from_events_for_key,
    clamp_float,
    compute_acceleration_ratio,
    compute_window_stats_for_key,
    ensure_utc,
    filter_events_by_key,
    filter_events_by_side,
    infer_severity,
    normalize_score,
    prune_events_older_than,
    scoped_key_to_string,
    utc_now,
)


class CascadeDetector:
    """
    Analytics detector для liquidation cascades.

    Correct production flow:
        exchange adapters
            -> market.liquidation
            -> LiquidationStream
            -> market.liquidation.normalized
            -> CascadeDetector
            -> analytics.liquidations.cascade_detected
            -> analytics.liquidations.exhaustion_detected

    Відповідальність:
    - слухає normalized/data-layer topic, а не raw market.liquidation;
    - бере sliding window із LiquidationState;
    - рахує side dominance / notional burst / acceleration / price compaction;
    - формує CascadeDetectionResult з повним liquidation scope;
    - публікує analytics.liquidations.* через EventBus;
    - реєструє healthcheck/snapshot/cleanup jobs через core.Scheduler.

    Scope:
        exchange + market_type + symbol + timeframe

    Цей клас НЕ:
    - не читає WebSocket;
    - не нормалізує raw exchange payload;
    - не викликає strategy/risk/execution напряму;
    - не приймає торгових рішень.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        config: CascadeDetectorConfig,
        state: LiquidationState | None = None,
        scheduler: Scheduler | None = None,
        metrics: LiquidationMetrics | None = None,
        market_state_store: Any | None = None,
        service_name: str = "cascade_detector",
    ) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config
        self.config.validate()
        self.config.assert_input_topic_allowed()

        self.state = state or get_shared_liquidation_state()
        self.metrics = metrics or LiquidationMetrics()
        self.market_state_store = market_state_store
        self.service_name = service_name

        self.logger = get_logger(
            __name__,
            service_name=self.service_name,
            event_type="analytics.liquidations.cascade_detector",
        )

        self._registered = False
        self._running = False

        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None

        self._subscription: Subscription | None = None

        self._processed_events = 0
        self._processed_snapshot_scopes = 0
        self._processed_snapshot_detections = 0
        self._invalid_payload_skips = 0
        self._cascade_signals_emitted = 0
        self._exhaustion_signals_emitted = 0
        self._cooldown_skips = 0
        self._empty_window_skips = 0
        self._threshold_skips = 0

        self._last_event_at: datetime | None = None
        self._last_signal_at: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

        self._latest_signals: list[CascadeDetectionResult] = []
        self._latest_signals_limit = self.config.recent_signals_limit

        self._healthcheck_job_id: str | None = None
        self._snapshot_job_id: str | None = None
        self._cleanup_job_id: str | None = None

    # ---------------------------------------------------------------------
    # Lifecycle / registration
    # ---------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscription і Scheduler jobs.

        core.EventBus.subscribe() є sync-методом, тому тут немає await.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "register", _analytics_args)
        except Exception:
            pass
        if self._registered:
            self.logger.warning("CascadeDetector already registered")
            return

        if not self.config.enabled:
            self.logger.info("CascadeDetector registration skipped: disabled by config")
            return

        self.config.assert_input_topic_allowed()

        if self.market_state_store is None:
            self._subscription = self.event_bus.subscribe(
                self.config.input_topic,
                self.on_liquidation_event,
                name=f"{self.service_name}.on_liquidation_event",
            )
        else:
            self.logger.info(
                "CascadeDetector registered in state-driven mode; normalized EventBus input subscription is disabled"
            )

        self._register_scheduler_jobs()
        self._registered = True

        self.logger.info(
            "CascadeDetector registered",
            extra={
                "input_topic": self.config.input_topic,
                "output_topics": list(self.config.output_topics),
                "scheduler_enabled": self.scheduler is not None,
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def start(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "start", _analytics_args)
        except Exception:
            pass
        if self._running:
            self.logger.warning("CascadeDetector already running")
            return

        if not self.config.enabled:
            self.logger.warning("CascadeDetector is disabled by config")
            return

        if not self._registered:
            self.register()

        self._running = True
        self._started_at = utc_now()
        self._stopped_at = None
        self._last_error = None
        self._last_error_at = None

        self.logger.info(
            "CascadeDetector started",
            extra={
                "input_topic": self.config.input_topic,
                "publish_topic_detected": self.config.publish_topic_detected,
                "publish_topic_exhaustion": self.config.publish_topic_exhaustion,
                "window_seconds": self.config.window_seconds,
                "min_events": self.config.min_events,
                "min_total_notional_usd": str(self.config.min_total_notional_usd),
                "min_side_imbalance_ratio": self.config.min_side_imbalance_ratio,
                "scope": "exchange:market_type:symbol:timeframe",
                "input_mode": "market_state" if self.market_state_store is not None else "event_bus_legacy",
            },
        )

    async def stop(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stop", _analytics_args)
        except Exception:
            pass
        if not self._running:
            return

        self._running = False
        self._stopped_at = utc_now()

        self.logger.info(
            "CascadeDetector stopped",
            extra=self.get_stats(),
        )

    async def restart(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "restart", _analytics_args)
        except Exception:
            pass
        await self.stop()
        await self.start()

    # ---------------------------------------------------------------------
    # State-driven input API
    # ---------------------------------------------------------------------

    async def process_market_state_snapshot(
        self,
        snapshot: Any | None = None,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        depth: int | None = None,
    ) -> CascadeDetectionResult | None:
        """Evaluate cascade state for a MarketStateStore snapshot.

        LiquidationStream should process the same snapshot first and update the
        shared LiquidationState.  This detector then reads that shared state and
        emits analytics.liquidations.* if thresholds pass.
        """
        if snapshot is None:
            if self.market_state_store is None:
                return None
            if not symbol:
                raise ValueError("symbol is required when snapshot is not provided")
            from data.market_models import MarketScope

            scope = MarketScope(
                exchange=exchange or "binance",
                market_type=market_type or "usdm_futures",
                symbol=symbol,
                timeframe=timeframe or "1m",
            )
            snapshot = await self.market_state_store.snapshot(scope, depth=depth)

        scope = getattr(snapshot, "scope", None)
        if scope is None:
            self._invalid_payload_skips += 1
            return None

        key = make_liquidation_key(
            exchange=getattr(scope, "exchange", None),
            market_type=getattr(scope, "market_type", None),
            symbol=getattr(scope, "symbol", None),
            timeframe=getattr(scope, "timeframe", None) or "1m",
        )
        symbol_state = self.state.get_key(key)
        self._processed_snapshot_scopes += 1

        if symbol_state is None or symbol_state.is_empty:
            self._empty_window_skips += 1
            return None

        # Trigger off the newest event in the shared state for this scope.
        trigger_event = list(symbol_state.events)[-1]
        if symbol_state.is_in_cooldown(trigger_event.timestamp):
            self._cooldown_skips += 1
            return None

        result = await self._detect_for_symbol_state(
            symbol_state,
            trigger_event=trigger_event,
            correlation_id=getattr(trigger_event, "correlation_id", None),
        )
        if result is not None:
            self._processed_snapshot_detections += 1
            await self._emit_detection_result(result)
        return result

    # ---------------------------------------------------------------------
    # Main EventBus handler
    # ---------------------------------------------------------------------

    async def on_liquidation_event(self, event: Event) -> None:
        """
        Основний EventBus handler.

        Очікує, що event.payload уже є LiquidationEvent із data-layer topic:
            market.liquidation.normalized
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_liquidation_event", _analytics_args)
        except Exception:
            pass
        if not self._running:
            return

        payload = event.payload

        if not isinstance(payload, LiquidationEvent):
            self._invalid_payload_skips += 1
            self.logger.debug(
                "CascadeDetector received non-LiquidationEvent payload",
                extra={
                    "topic": event.topic,
                    "event_id": event.event_id,
                    "payload_type": type(payload).__name__,
                },
            )
            return

        liquidation_event = payload

        try:
            self._processed_events += 1
            self._last_event_at = ensure_utc(liquidation_event.timestamp)

            symbol_state = self.state.get_key(liquidation_event.key)

            if symbol_state is None or symbol_state.is_empty:
                self._empty_window_skips += 1
                return

            if symbol_state.is_in_cooldown(liquidation_event.timestamp):
                self._cooldown_skips += 1
                return

            result = await self._detect_for_symbol_state(
                symbol_state,
                trigger_event=liquidation_event,
                correlation_id=event.correlation_id,
            )

            if result is None:
                return

            await self._emit_detection_result(result)

        except Exception as exc:
            self._last_error_at = utc_now()
            self._last_error = repr(exc)

            self.logger.exception(
                "Unhandled error in CascadeDetector.on_liquidation_event",
                extra={
                    "topic": event.topic,
                    "event_id": event.event_id,
                    "exchange": liquidation_event.exchange,
                    "market_type": liquidation_event.market_type,
                    "symbol": liquidation_event.symbol,
                    "timeframe": liquidation_event.timeframe,
                    "exchange_symbol": liquidation_event.exchange_symbol,
                    "scope": liquidation_key_to_dict(liquidation_event.key),
                    "error": repr(exc),
                },
            )

    # ---------------------------------------------------------------------
    # Detection
    # ---------------------------------------------------------------------

    async def _detect_for_symbol_state(
        self,
        symbol_state: SymbolLiquidationState,
        *,
        trigger_event: LiquidationEvent,
        correlation_id: str | None = None,
    ) -> CascadeDetectionResult | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_for_symbol_state", _analytics_args)
        except Exception:
            pass
        if trigger_event.key != symbol_state.key:
            raise ValueError(
                "Trigger event scope does not match SymbolLiquidationState: "
                f"event={liquidation_key_to_dict(trigger_event.key)} "
                f"state={liquidation_key_to_dict(symbol_state.key)}"
            )

        window_events = self._get_window_events(
            symbol_state,
            now=trigger_event.timestamp,
        )
        window_events = filter_events_by_key(window_events, trigger_event.key)

        if not window_events:
            self._empty_window_skips += 1
            return None

        stats = compute_window_stats_for_key(
            key=trigger_event.key,
            events=window_events,
        )

        if not self._passes_base_thresholds(stats):
            self._threshold_skips += 1
            return None

        dominant_side_events = filter_events_by_side(
            window_events,
            stats.dominant_side,
        )

        if len(dominant_side_events) < self.config.min_events:
            self._threshold_skips += 1
            return None

        acceleration_ratio = (
            compute_acceleration_ratio(dominant_side_events)
            if self.config.acceleration_enabled
            else 0.0
        )

        intensity_score = self._compute_intensity_score(
            stats,
            acceleration_ratio=acceleration_ratio,
        )

        severity = infer_severity(
            intensity_score=intensity_score,
            low_threshold=self.config.low_severity_threshold,
            medium_threshold=self.config.medium_severity_threshold,
            high_threshold=self.config.high_severity_threshold,
            extreme_threshold=self.config.extreme_severity_threshold,
        )

        cluster = build_cluster_from_events_for_key(
            key=trigger_event.key,
            side=stats.dominant_side,
            events=dominant_side_events,
            severity=severity,
            status=LiquidationStatus.CONFIRMED,
            source=self.service_name,
        )

        if cluster is None:
            return None

        continuation_bias = self._compute_continuation_bias(
            stats=stats,
            acceleration_ratio=acceleration_ratio,
        )
        exhaustion_bias = self._compute_exhaustion_bias(
            stats=stats,
            acceleration_ratio=acceleration_ratio,
        )
        confidence = self._compute_confidence(
            stats=stats,
            intensity_score=intensity_score,
            acceleration_ratio=acceleration_ratio,
        )

        result = CascadeDetectionResult(
            exchange=trigger_event.exchange,
            market_type=trigger_event.market_type,
            symbol=trigger_event.symbol,
            timeframe=trigger_event.timeframe,
            exchange_symbol=trigger_event.exchange_symbol,
            side=stats.dominant_side,
            direction=cluster.direction,
            detected_at=utc_now(),
            cluster=cluster,
            intensity_score=intensity_score,
            confidence=confidence,
            continuation_bias=continuation_bias,
            exhaustion_bias=exhaustion_bias,
            event_count=stats.total_events,
            total_notional_usd=stats.total_notional_usd,
            window_seconds=self.config.window_seconds,
            price_range_pct=stats.price_range_pct,
            severity=severity,
            status=LiquidationStatus.CONFIRMED,
            correlation_id=correlation_id or trigger_event.correlation_id,
            source=self.service_name,
            metadata={
                "scope": liquidation_key_to_dict(trigger_event.key),
                "scope_key": scoped_key_to_string(trigger_event.key),
                "exchange_symbol": trigger_event.exchange_symbol,
                "trigger_event_id": trigger_event.event_id,
                "trigger_event_timestamp": trigger_event.timestamp.isoformat(),
                "side_imbalance_ratio": stats.side_imbalance_ratio,
                "event_imbalance_ratio": stats.event_imbalance_ratio,
                "dominant_side": stats.dominant_side.value,
                "acceleration_ratio": acceleration_ratio,
                "long_events": stats.long_events,
                "short_events": stats.short_events,
                "long_notional_usd": str(stats.long_notional_usd),
                "short_notional_usd": str(stats.short_notional_usd),
            },
        )

        cooldown_until = result.detected_at + timedelta(
            seconds=self.config.cooldown_seconds,
        )
        symbol_state.set_cascade_detected(result.detected_at, cooldown_until)

        self._remember_signal(result)
        return result

    # ---------------------------------------------------------------------
    # Window / thresholds
    # ---------------------------------------------------------------------

    def _get_window_events(
        self,
        symbol_state: SymbolLiquidationState,
        *,
        now: datetime,
    ) -> list[LiquidationEvent]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_window_events", _analytics_args)
        except Exception:
            pass
        min_ts = ensure_utc(now) - timedelta(seconds=self.config.window_seconds)
        return prune_events_older_than(
            list(symbol_state.events),
            min_timestamp=min_ts,
        )

    def _passes_base_thresholds(self, stats: LiquidationWindowStats) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_passes_base_thresholds", _analytics_args)
        except Exception:
            pass
        if stats.total_events < self.config.min_events:
            return False

        if stats.total_notional_usd < self.config.min_total_notional_usd:
            return False

        if not stats.dominant_side.is_known:
            return False

        if stats.side_imbalance_ratio < self.config.min_side_imbalance_ratio:
            return False

        if self.config.price_compaction_enabled:
            if stats.price_range_pct > self.config.max_price_range_pct:
                return False

        return True

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    def _compute_intensity_score(
        self,
        stats: LiquidationWindowStats,
        *,
        acceleration_ratio: float,
    ) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_intensity_score", _analytics_args)
        except Exception:
            pass
        notional_score = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 3.0,
        )

        imbalance_score = clamp_float(stats.side_imbalance_ratio)

        if self.config.price_compaction_enabled:
            continuation_signal = clamp_float(
                1.0
                - (
                    stats.price_range_pct
                    / max(self.config.max_price_range_pct, 1e-9)
                )
            )
        else:
            continuation_signal = 0.5

        acceleration_score = 0.0
        if self.config.acceleration_enabled:
            acceleration_score = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        total_weight = self.config.score_weights_sum
        if total_weight <= 0:
            return 0.0

        weighted = (
            continuation_signal * self.config.continuation_score_weight
            + imbalance_score * self.config.imbalance_score_weight
            + notional_score * self.config.notional_score_weight
            + acceleration_score * self.config.acceleration_score_weight
        ) / total_weight

        return clamp_float(weighted)

    def _compute_continuation_bias(
        self,
        *,
        stats: LiquidationWindowStats,
        acceleration_ratio: float,
    ) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_continuation_bias", _analytics_args)
        except Exception:
            pass
        imbalance_component = clamp_float(stats.side_imbalance_ratio)

        acceleration_component = 0.0
        if self.config.acceleration_enabled:
            acceleration_component = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        compaction_component = 0.5
        if self.config.price_compaction_enabled and self.config.max_price_range_pct > 0:
            compaction_component = clamp_float(
                1.0 - (stats.price_range_pct / self.config.max_price_range_pct)
            )

        result = (
            imbalance_component * 0.45
            + acceleration_component * 0.35
            + compaction_component * 0.20
        )
        return clamp_float(result)

    def _compute_exhaustion_bias(
        self,
        *,
        stats: LiquidationWindowStats,
        acceleration_ratio: float,
    ) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_exhaustion_bias", _analytics_args)
        except Exception:
            pass
        burst_component = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 4.0,
        )

        weak_acceleration_component = 0.0
        if self.config.acceleration_enabled:
            weak_acceleration_component = clamp_float(
                1.0
                - normalize_score(
                    acceleration_ratio,
                    max(self.config.min_acceleration_ratio, 1.0),
                )
            )

        range_expansion_component = 0.0
        if self.config.price_compaction_enabled and self.config.max_price_range_pct > 0:
            range_expansion_component = clamp_float(
                stats.price_range_pct / (self.config.max_price_range_pct * 2.0)
            )

        result = (
            burst_component * 0.40
            + weak_acceleration_component * 0.35
            + range_expansion_component * 0.25
        )
        return clamp_float(result)

    def _compute_confidence(
        self,
        *,
        stats: LiquidationWindowStats,
        intensity_score: float,
        acceleration_ratio: float,
    ) -> float:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_confidence", _analytics_args)
        except Exception:
            pass
        notional_component = normalize_score(
            float(stats.total_notional_usd),
            float(self.config.min_total_notional_usd) * 4.0,
        )
        imbalance_component = clamp_float(stats.side_imbalance_ratio)

        acceleration_component = 0.5
        if self.config.acceleration_enabled:
            acceleration_component = normalize_score(
                acceleration_ratio,
                max(self.config.min_acceleration_ratio * 1.5, 1.0),
            )

        result = (
            intensity_score * 0.40
            + notional_component * 0.25
            + imbalance_component * 0.25
            + acceleration_component * 0.10
        )
        return clamp_float(result)

    # ---------------------------------------------------------------------
    # Event publishing
    # ---------------------------------------------------------------------

    async def _emit_detection_result(self, result: CascadeDetectionResult) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_detection_result", _analytics_args)
        except Exception:
            pass
        headers = self._headers_for_result(
            result,
            event_type=LiquidationEventType.CASCADE,
        )

        strategy_payload = ensure_strategy_payload_contract(
            result,
            topic=self.config.publish_topic_detected,
            source=self.service_name,
            domain="liquidations",
        )
        accepted = await self.event_bus.emit(
            self.config.publish_topic_detected,
            strategy_payload,
            priority=EventPriority.HIGH,
            source=self.service_name,
            correlation_id=result.correlation_id,
            headers=headers,
        )

        if accepted:
            self.metrics.observe_cascade(result)
            self._cascade_signals_emitted += 1
            self._last_signal_at = result.detected_at

        self.logger.info(
            "Liquidation cascade detected",
            extra={
                "exchange": result.exchange,
                "market_type": result.market_type,
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "exchange_symbol": result.exchange_symbol,
                "scope": liquidation_key_to_dict(result.key),
                "side": result.side.value,
                "severity": result.severity.value,
                "intensity_score": result.intensity_score,
                "confidence": result.confidence,
                "continuation_bias": result.continuation_bias,
                "exhaustion_bias": result.exhaustion_bias,
                "event_count": result.event_count,
                "total_notional_usd": str(result.total_notional_usd),
                "price_range_pct": result.price_range_pct,
            },
        )

        if result.favors_exhaustion:
            exhaustion_headers = self._headers_for_result(
                result,
                event_type=LiquidationEventType.EXHAUSTION,
            )

            exhaustion_payload = ensure_strategy_payload_contract(
                result,
                topic=self.config.publish_topic_exhaustion,
                source=self.service_name,
                domain="liquidations",
            )
            exhaustion_accepted = await self.event_bus.emit(
                self.config.publish_topic_exhaustion,
                exhaustion_payload,
                priority=EventPriority.HIGH,
                source=self.service_name,
                correlation_id=result.correlation_id,
                headers=exhaustion_headers,
            )

            if exhaustion_accepted:
                self.metrics.observe_exhaustion(result)
                self._exhaustion_signals_emitted += 1

    async def emit_runtime_snapshot(
        self,
        topic: str | None = None,
    ) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_runtime_snapshot", _analytics_args)
        except Exception:
            pass
        snapshot = {
            "service": self.service_name,
            "running": self._running,
            "stats": self.get_stats(),
            "health": self.get_health(),
            "latest_signals": [
                item.to_dict(serialize=True)
                for item in self.get_recent_signals(limit=25)
            ],
            "scope": "exchange:market_type:symbol:timeframe",
            "emitted_at": utc_now().isoformat(),
        }

        return await self.event_bus.emit(
            topic or self.config.publish_topic_snapshot,
            snapshot,
            priority=EventPriority.LOW,
            source=self.service_name,
            headers={
                "event_type": LiquidationEventType.SNAPSHOT.value,
            },
        )

    async def emit_health(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_health", _analytics_args)
        except Exception:
            pass
        health = self.get_health()

        return await self.event_bus.emit(
            self.config.publish_topic_health,
            health,
            priority=EventPriority.LOW,
            source=self.service_name,
            headers={
                "event_type": LiquidationEventType.HEALTH.value,
            },
        )

    # ---------------------------------------------------------------------
    # Signal memory / query API
    # ---------------------------------------------------------------------

    def _remember_signal(self, result: CascadeDetectionResult) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_remember_signal", _analytics_args)
        except Exception:
            pass
        self._latest_signals.append(result)

        if len(self._latest_signals) > self._latest_signals_limit:
            self._latest_signals = self._latest_signals[-self._latest_signals_limit :]

    async def detect_now(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> CascadeDetectionResult | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_now", _analytics_args)
        except Exception:
            pass
        key = self._make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        symbol_state = self.state.get_key(key)

        if symbol_state is None or symbol_state.is_empty:
            return None

        trigger_event = symbol_state.events[-1]
        result = await self._detect_for_symbol_state(
            symbol_state,
            trigger_event=trigger_event,
        )

        if result is not None:
            await self._emit_detection_result(result)

        return result

    def detect_now_key(
        self,
        key: LiquidationKey,
    ) -> CascadeDetectionResult | None:
        """
        Sync convenience method не робимо, бо detection публікує async EventBus.
        Залишено явну помилку, щоб не створювати прихованих background tasks.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_now_key", _analytics_args)
        except Exception:
            pass
        raise RuntimeError("Use async detect_now(..., market_type=..., timeframe=...)")

    def get_recent_signals(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        limit: int = 50,
    ) -> list[CascadeDetectionResult]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_recent_signals", _analytics_args)
        except Exception:
            pass
        if limit <= 0:
            return []

        target_exchange = normalize_exchange(exchange) if exchange is not None else None
        target_symbol = normalize_symbol(symbol) if symbol is not None else None
        target_market_type = (
            normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        target_timeframe = (
            normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        result: list[CascadeDetectionResult] = []

        for signal in reversed(self._latest_signals):
            if target_exchange and signal.exchange != target_exchange:
                continue
            if target_market_type and signal.market_type != target_market_type:
                continue
            if target_symbol and signal.symbol != target_symbol:
                continue
            if target_timeframe and signal.timeframe != target_timeframe:
                continue

            result.append(signal)

            if len(result) >= limit:
                break

        return result

    def get_key_last_signal(
        self,
        key: LiquidationKey,
    ) -> CascadeDetectionResult | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_last_signal", _analytics_args)
        except Exception:
            pass
        for signal in reversed(self._latest_signals):
            if signal.key == key:
                return signal
        return None

    def get_symbol_last_signal(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> CascadeDetectionResult | None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_last_signal", _analytics_args)
        except Exception:
            pass
        signals = self.get_recent_signals(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            limit=1,
        )
        return signals[0] if signals else None

    def get_hot_symbols(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_hot_symbols", _analytics_args)
        except Exception:
            pass
        latest_by_key: dict[LiquidationKey, CascadeDetectionResult] = {}

        for signal in self._latest_signals:
            previous = latest_by_key.get(signal.key)

            if previous is None or signal.detected_at > previous.detected_at:
                latest_by_key[signal.key] = signal

        rows = [
            {
                "exchange": signal.exchange,
                "market_type": signal.market_type,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "exchange_symbol": signal.exchange_symbol,
                "scope": liquidation_key_to_dict(signal.key),
                "severity": signal.severity.value,
                "intensity_score": signal.intensity_score,
                "confidence": signal.confidence,
                "continuation_bias": signal.continuation_bias,
                "exhaustion_bias": signal.exhaustion_bias,
                "detected_at": signal.detected_at.isoformat(),
                "total_notional_usd": str(signal.total_notional_usd),
            }
            for signal in latest_by_key.values()
        ]

        rows.sort(
            key=lambda row: (
                float(row["intensity_score"]),
                float(row["confidence"]),
            ),
            reverse=True,
        )

        return rows[: max(0, limit)]

    def get_key_diagnostic(
        self,
        key: LiquidationKey,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_diagnostic", _analytics_args)
        except Exception:
            pass
        scope = liquidation_key_to_dict(key)
        symbol_state = self.state.get_key(key)

        if symbol_state is None:
            return {
                **scope,
                "scope": scope,
                "exists": False,
            }

        now = utc_now()
        window_events = self._get_window_events(symbol_state, now=now)
        window_events = filter_events_by_key(window_events, key)

        stats = compute_window_stats_for_key(
            key=key,
            events=window_events,
        )

        last_signal = self.get_key_last_signal(key)

        return {
            **scope,
            "scope": scope,
            "exchange_symbol": symbol_state.exchange_symbol,
            "exists": True,
            "buffer_snapshot": symbol_state.snapshot().to_dict(serialize=True),
            "window_stats": stats.to_dict(serialize=True),
            "cooldown_active": symbol_state.is_in_cooldown(now),
            "last_signal": (
                last_signal.to_dict(serialize=True)
                if last_signal is not None
                else None
            ),
        }

    def get_symbol_diagnostic(
        self,
        exchange: str,
        symbol: str,
        *,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_diagnostic", _analytics_args)
        except Exception:
            pass
        key = self._make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.get_key_diagnostic(key)

    # ---------------------------------------------------------------------
    # Scheduler jobs
    # ---------------------------------------------------------------------

    def _register_scheduler_jobs(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None:
            return

        if self._healthcheck_job_id is None:
            self._healthcheck_job_id = self.scheduler.add_interval_job(
                name=self.config.healthcheck_job_name,
                func=self._scheduled_healthcheck,
                interval=self.config.healthcheck_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

        if self._snapshot_job_id is None:
            self._snapshot_job_id = self.scheduler.add_interval_job(
                name=self.config.snapshot_job_name,
                func=self._scheduled_snapshot,
                interval=self.config.snapshot_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

        if self._cleanup_job_id is None:
            self._cleanup_job_id = self.scheduler.add_interval_job(
                name=self.config.cleanup_job_name,
                func=self._scheduled_cleanup,
                interval=self.config.cleanup_interval_seconds,
                run_immediately=False,
                max_retries=self.config.scheduler_job_max_retries,
                retry_delay=self.config.scheduler_job_retry_delay_seconds,
                timeout=self.config.scheduler_job_timeout_seconds,
                allow_overlap=False,
                enabled=True,
            )

    async def _scheduled_healthcheck(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_scheduled_healthcheck", _analytics_args)
        except Exception:
            pass
        health = self.get_health()
        await self.emit_health()

        if health["status"] != "healthy":
            self.logger.warning(
                "CascadeDetector health degraded",
                extra=health,
            )

    async def _scheduled_snapshot(self) -> None:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_scheduled_snapshot", _analytics_args)
        except Exception:
            pass
        await self.emit_runtime_snapshot()

    async def _scheduled_cleanup(self) -> None:
        """
        Cleanup тут не чистить LiquidationState — це відповідальність stream/state cleanup.
        Тут чистимо тільки локальний buffer latest signals.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_scheduled_cleanup", _analytics_args)
        except Exception:
            pass
        if len(self._latest_signals) > self._latest_signals_limit:
            self._latest_signals = self._latest_signals[-self._latest_signals_limit :]

    # ---------------------------------------------------------------------
    # Health / stats
    # ---------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_health", _analytics_args)
        except Exception:
            pass
        status = "healthy"

        if not self._running:
            status = "stopped"
        elif self._last_error is not None:
            status = "degraded"

        seconds_since_last_signal = (
            (utc_now() - ensure_utc(self._last_signal_at)).total_seconds()
            if self._last_signal_at
            else None
        )

        seconds_since_last_event = (
            (utc_now() - ensure_utc(self._last_event_at)).total_seconds()
            if self._last_event_at
            else None
        )

        return {
            "status": status,
            "running": self._running,
            "registered": self._registered,
            "scope": "exchange:market_type:symbol:timeframe",
            "input_topic": self.config.input_topic,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "seconds_since_last_event": seconds_since_last_event,
            "seconds_since_last_signal": seconds_since_last_signal,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
        }

    def get_stats(self) -> dict[str, Any]:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_stats", _analytics_args)
        except Exception:
            pass
        uptime_seconds = (
            max(0.0, (utc_now() - ensure_utc(self._started_at)).total_seconds())
            if self._started_at
            else 0.0
        )

        return {
            "service_name": self.service_name,
            "running": self._running,
            "registered": self._registered,
            "scope": "exchange:market_type:symbol:timeframe",
            "input_topic": self.config.input_topic,
            "output_topics": list(self.config.output_topics),
            "uptime_seconds": uptime_seconds,
            "processed_events": self._processed_events,
            "input_mode": "market_state" if self.market_state_store is not None else "event_bus_legacy",
            "processed_snapshot_scopes": self._processed_snapshot_scopes,
            "processed_snapshot_detections": self._processed_snapshot_detections,
            "invalid_payload_skips": self._invalid_payload_skips,
            "cascade_signals_emitted": self._cascade_signals_emitted,
            "exhaustion_signals_emitted": self._exhaustion_signals_emitted,
            "cooldown_skips": self._cooldown_skips,
            "empty_window_skips": self._empty_window_skips,
            "threshold_skips": self._threshold_skips,
            "tracked_scopes": self.state.scopes_count,
            "tracked_symbols": self.state.symbols_count,
            "latest_signals_buffered": len(self._latest_signals),
            "latest_signals_limit": self._latest_signals_limit,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
            "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
            "last_error": self._last_error,
            "last_error_at": self._last_error_at.isoformat() if self._last_error_at else None,
            "healthcheck_job_id": self._healthcheck_job_id,
            "snapshot_job_id": self._snapshot_job_id,
            "cleanup_job_id": self._cleanup_job_id,
        }

    @property
    def is_running(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_running", _analytics_args)
        except Exception:
            pass
        return self._running

    @property
    def is_registered(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_registered", _analytics_args)
        except Exception:
            pass
        return self._registered

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _headers_for_result(
        result: CascadeDetectionResult,
        *,
        event_type: LiquidationEventType,
    ) -> dict[str, str]:
        try:
            _analytics_class_name = "CascadeDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_headers_for_result", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": result.exchange,
            "market_type": result.market_type,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "exchange_symbol": result.exchange_symbol or result.symbol,
            "scope": scoped_key_to_string(result.key),
            "event_type": event_type.value,
            "severity": result.severity.value,
        }

    @staticmethod
    def _make_key(
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> LiquidationKey:
        try:
            _analytics_class_name = "CascadeDetector"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_make_key", _analytics_args)
        except Exception:
            pass
        return make_liquidation_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )


__all__ = [
    "CascadeDetector",
]
