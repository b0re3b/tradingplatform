from __future__ import annotations
from core.logger import get_logger

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import WhaleClusterAnalyzerConfig
from analytics.whales.enums import WhaleComponentName, WhaleTradeSide
from analytics.whales.models import (
    SymbolClusterState,
    WhaleActivityRecord,
    WhaleClusterAnalysisResult,
    WhaleClusterExhaustionSignal,
    WhaleClusterSignal,
    WhaleClusterUpdateSignal,
    WhaleKey,
    WhaleLiquidationContextRecord,
    WhalePressureRecord,
    make_symbol_cluster_state,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    whale_key_to_dict,
)


class WhaleClusterAnalyzer(BaseWhaleComponent):
    """
    Третій шар whale-аналітики.

    Input:
        analytics.whales.whale_activity
        analytics.whales.whale_pressure
        analytics.whales.whale_liquidation_context

    Output:
        analytics.whales.whale_cluster
        analytics.whales.whale_cluster_update
        analytics.whales.whale_cluster_exhaustion

    Correct pipeline:
        LargeTradeDetector
            -> analytics.whales.large_trade
            -> WhaleTracker
            -> analytics.whales.whale_activity / whale_pressure / whale_liquidation_context
            -> WhaleClusterAnalyzer
            -> analytics.whales.whale_cluster.*

    Scope:
        exchange + market_type + symbol + timeframe

    Важливо:
    - не читає біржові adapters напряму;
    - не слухає raw market-data topics;
    - не змішує state різних бірж / market_type / timeframe;
    - cleanup запускається тільки через Scheduler.add_interval_job();
    - власних uncontrolled asyncio cleanup loops немає.
    """

    def __init__(
        self,
        *,
        config: WhaleClusterAnalyzerConfig,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
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
        super().__init__(
            component_name=WhaleComponentName.WHALE_CLUSTER_ANALYZER.value,
            event_bus=event_bus,
            scheduler=scheduler,
            default_exchange=config.default_exchange,
            default_market_type=config.default_market_type,
            default_timeframe=config.default_timeframe,
        )

        self.config = config
        self.config.validate()

        self._states: dict[WhaleKey, SymbolClusterState] = {}

        # Registry lock захищає створення per-key lock-ів і короткі snapshot/reset операції.
        # Бізнес-обробка блокує тільки конкретний WhaleKey, а не весь analyzer.
        self._state_locks: dict[WhaleKey, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Idempotent: повторний виклик не створює дублікати підписок.
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
            return

        if not self.config.enabled:
            self.logger.info(
                "WhaleClusterAnalyzer registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        self._subscribe_production(
            self.config.whale_activity_event_name,
            self.handle_whale_activity_event,
            name="analytics.whales.whale_cluster_analyzer.handle_whale_activity_event",
        )
        self._subscribe_production(
            self.config.whale_pressure_event_name,
            self.handle_whale_pressure_event,
            name="analytics.whales.whale_cluster_analyzer.handle_whale_pressure_event",
        )
        self._subscribe_production(
            self.config.whale_liquidation_context_event_name,
            self.handle_whale_liquidation_context_event,
            name=(
                "analytics.whales.whale_cluster_analyzer."
                "handle_whale_liquidation_context_event"
            ),
        )

        self._registered = True

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
        if self._started:
            self.logger.warning("WhaleClusterAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleClusterAnalyzer is disabled by config")
            return

        await self.register()

        self._add_interval_job(
            name="analytics.whales.whale_cluster_analyzer.cleanup",
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
            "WhaleClusterAnalyzer started",
            extra={
                "component": self.component_name,
                "production_input_topics": list(self.config.production_input_topics),
                "whale_activity_event_name": self.config.whale_activity_event_name,
                "whale_pressure_event_name": self.config.whale_pressure_event_name,
                "whale_liquidation_context_event_name": (
                    self.config.whale_liquidation_context_event_name
                ),
                "whale_cluster_event_name": self.config.whale_cluster_event_name,
                "whale_cluster_update_event_name": (
                    self.config.whale_cluster_update_event_name
                ),
                "whale_cluster_exhaustion_event_name": (
                    self.config.whale_cluster_exhaustion_event_name
                ),
                "analysis_window_sec": self.config.analysis_window_sec,
                "cluster_ttl_sec": self.config.cluster_ttl_sec,
                "min_cluster_score_to_emit": self.config.min_cluster_score_to_emit,
                "cleanup_interval_sec": self.config.cleanup_interval_sec,
                "scope": "exchange:market_type:symbol:timeframe",
                "locking": "per_whale_key",
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
        if not self._started and not self._registered:
            return

        await super().stop()

        self.logger.info(
            "WhaleClusterAnalyzer stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # EventBus handlers
    # =========================================================================

    async def handle_whale_activity_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_whale_activity_event", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._payload_from_event(event)

            await self.process_whale_activity_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing whale activity event",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

    async def handle_whale_pressure_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_whale_pressure_event", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._payload_from_event(event)

            await self.process_whale_pressure_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing whale pressure event",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

    async def handle_whale_liquidation_context_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_whale_liquidation_context_event", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._payload_from_event(event)

            await self.process_whale_liquidation_context_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing whale liquidation context event",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

    # =========================================================================
    # Public processing API
    # =========================================================================

    async def process_whale_activity_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> WhaleClusterAnalysisResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_activity_payload", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_activity_payload(payload)
        if record is None:
            return WhaleClusterAnalysisResult()

        if not self.config.should_process_key(record.key):
            return WhaleClusterAnalysisResult()

        state_lock = await self._get_state_lock(record.key)

        async with state_lock:
            state = self._get_or_create_state(record)
            state.activity_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_state(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(
            result,
            correlation_id=correlation_id,
            source_event_id=source_event_id,
            source_topic=source_topic,
        )
        return result

    async def process_whale_pressure_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> WhaleClusterAnalysisResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_pressure_payload", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_pressure_payload(payload)
        if record is None:
            return WhaleClusterAnalysisResult()

        if not self.config.should_process_key(record.key):
            return WhaleClusterAnalysisResult()

        state_lock = await self._get_state_lock(record.key)

        async with state_lock:
            state = self._get_or_create_state(record)
            state.pressure_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_state(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(
            result,
            correlation_id=correlation_id,
            source_event_id=source_event_id,
            source_topic=source_topic,
        )
        return result

    async def process_whale_liquidation_context_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> WhaleClusterAnalysisResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_liquidation_context_payload", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return WhaleClusterAnalysisResult()

        record = self._normalize_whale_liquidation_context_payload(payload)
        if record is None:
            return WhaleClusterAnalysisResult()

        if not self.config.should_process_key(record.key):
            return WhaleClusterAnalysisResult()

        state_lock = await self._get_state_lock(record.key)

        async with state_lock:
            state = self._get_or_create_state(record)
            state.liquidation_context_records.append(record)
            state.total_events_seen += 1
            state.touch()

            self._update_cluster_seen_range(state, record.timestamp_ms)
            self._prune_symbol_state(state, record.timestamp_ms)

            result = self._analyze_state(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        await self._emit_analysis_result(
            result,
            correlation_id=correlation_id,
            source_event_id=source_event_id,
            source_topic=source_topic,
        )
        return result

    async def process_whale_activity_event(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        """
        Backward-compatible alias для старого direct API.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_activity_event", _analytics_args)
        except Exception:
            pass
        return await self.process_whale_activity_payload(event)

    async def process_whale_pressure_event(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        """
        Backward-compatible alias для старого direct API.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_pressure_event", _analytics_args)
        except Exception:
            pass
        return await self.process_whale_pressure_payload(event)

    async def process_whale_liquidation_context_event(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleClusterAnalysisResult:
        """
        Backward-compatible alias для старого direct API.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_whale_liquidation_context_event", _analytics_args)
        except Exception:
            pass
        return await self.process_whale_liquidation_context_payload(event)

    # =========================================================================
    # Core analysis
    # =========================================================================

    def _analyze_state(
        self,
        *,
        state: SymbolClusterState,
        current_ts_ms: int,
    ) -> WhaleClusterAnalysisResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_analyze_state", _analytics_args)
        except Exception:
            pass
        activity_count = len(state.activity_records)
        total_activity_notional = sum(
            record.total_notional for record in state.activity_records
        )

        if activity_count < self.config.min_activity_signals:
            return WhaleClusterAnalysisResult()

        if total_activity_notional < self.config.min_total_activity_notional:
            return WhaleClusterAnalysisResult()

        cluster_side = self._determine_cluster_side(state)
        directional_bias = self._calculate_directional_bias(state, cluster_side)
        persistence_score = self._calculate_persistence_score(state, current_ts_ms)
        activity_score = self._calculate_activity_score(state)
        pressure_score = self._calculate_pressure_score(state, cluster_side)
        liquidation_context_score = self._calculate_liquidation_context_score(
            state,
            cluster_side,
        )

        cluster_score = self._clamp_0_1(
            activity_score * self.config.activity_weight
            + pressure_score * self.config.pressure_weight
            + liquidation_context_score * self.config.liquidation_context_weight
            + persistence_score * self.config.persistence_weight
        )

        continuation_probability = self._clamp_0_1(
            0.45 * cluster_score
            + 0.35 * directional_bias
            + 0.20 * persistence_score
        )

        exhaustion_probability = self._clamp_0_1(
            0.50 * (1.0 - directional_bias)
            + 0.30 * liquidation_context_score
            + 0.20 * (1.0 - pressure_score)
        )

        cluster_signal: WhaleClusterSignal | None = None
        cluster_update_signal: WhaleClusterUpdateSignal | None = None
        cluster_exhaustion_signal: WhaleClusterExhaustionSignal | None = None

        if (
            cluster_score >= self.config.min_cluster_score_to_emit
            and continuation_probability >= self.config.min_continuation_probability_to_emit
            and self._passes_cooldown(
                state.last_cluster_emit_ts_monotonic,
                self.config.cluster_emit_cooldown_sec,
            )
        ):
            cluster_signal = WhaleClusterSignal(
                exchange=state.exchange,
                market_type=state.market_type,
                symbol=state.symbol,
                timeframe=state.timeframe,
                exchange_symbol=state.exchange_symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                persistence_score=persistence_score,
                directional_bias=directional_bias,
                continuation_probability=continuation_probability,
                exhaustion_probability=exhaustion_probability,
                activity_signal_count=len(state.activity_records),
                pressure_signal_count=len(state.pressure_records),
                liquidation_context_count=len(state.liquidation_context_records),
                total_activity_notional=total_activity_notional,
                total_pressure_notional=sum(
                    record.total_notional for record in state.pressure_records
                ),
                total_liquidation_context_notional=sum(
                    record.liquidation_total_notional
                    for record in state.liquidation_context_records
                ),
                first_seen_ts_ms=state.cluster_first_seen_ts_ms or current_ts_ms,
                last_seen_ts_ms=state.cluster_last_seen_ts_ms or current_ts_ms,
                timestamp_ms=current_ts_ms,
                metadata={
                    "scope": whale_key_to_dict(state.key),
                },
            )
            state.total_clusters_emitted += 1
            state.last_cluster_emit_ts_monotonic = time.monotonic()

        if (
            cluster_score >= self.config.min_cluster_score_to_emit
            and self._passes_cooldown(
                state.last_cluster_update_emit_ts_monotonic,
                self.config.cluster_update_cooldown_sec,
            )
        ):
            cluster_update_signal = WhaleClusterUpdateSignal(
                exchange=state.exchange,
                market_type=state.market_type,
                symbol=state.symbol,
                timeframe=state.timeframe,
                exchange_symbol=state.exchange_symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                persistence_score=persistence_score,
                continuation_probability=continuation_probability,
                exhaustion_probability=exhaustion_probability,
                activity_signal_count=len(state.activity_records),
                pressure_signal_count=len(state.pressure_records),
                liquidation_context_count=len(state.liquidation_context_records),
                timestamp_ms=current_ts_ms,
                metadata={
                    "scope": whale_key_to_dict(state.key),
                },
            )
            state.total_cluster_updates_emitted += 1
            state.last_cluster_update_emit_ts_monotonic = time.monotonic()

        if (
            exhaustion_probability >= self.config.min_exhaustion_probability_to_emit
            and self._passes_cooldown(
                state.last_cluster_exhaustion_emit_ts_monotonic,
                self.config.cluster_exhaustion_cooldown_sec,
            )
        ):
            cluster_exhaustion_signal = WhaleClusterExhaustionSignal(
                exchange=state.exchange,
                market_type=state.market_type,
                symbol=state.symbol,
                timeframe=state.timeframe,
                exchange_symbol=state.exchange_symbol,
                cluster_side=cluster_side,
                cluster_score=cluster_score,
                exhaustion_probability=exhaustion_probability,
                reversal_risk=exhaustion_probability,
                timestamp_ms=current_ts_ms,
                metadata={
                    "scope": whale_key_to_dict(state.key),
                },
            )
            state.total_cluster_exhaustions_emitted += 1
            state.last_cluster_exhaustion_emit_ts_monotonic = time.monotonic()

        return WhaleClusterAnalysisResult(
            whale_cluster_signal=cluster_signal,
            whale_cluster_update_signal=cluster_update_signal,
            whale_cluster_exhaustion_signal=cluster_exhaustion_signal,
        )

    # =========================================================================
    # Scoring
    # =========================================================================

    def _determine_cluster_side(self, state: SymbolClusterState) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_determine_cluster_side", _analytics_args)
        except Exception:
            pass
        buy_activity = sum(
            record.total_notional
            for record in state.activity_records
            if record.side == WhaleTradeSide.BUY.value
        )
        sell_activity = sum(
            record.total_notional
            for record in state.activity_records
            if record.side == WhaleTradeSide.SELL.value
        )

        buy_pressure = sum(
            record.buy_notional
            for record in state.pressure_records
            if record.dominant_side == WhaleTradeSide.BUY.value
        )
        sell_pressure = sum(
            record.sell_notional
            for record in state.pressure_records
            if record.dominant_side == WhaleTradeSide.SELL.value
        )

        buy_liq_context = sum(
            record.whale_total_notional
            for record in state.liquidation_context_records
            if record.whale_side == WhaleTradeSide.BUY.value
        )
        sell_liq_context = sum(
            record.whale_total_notional
            for record in state.liquidation_context_records
            if record.whale_side == WhaleTradeSide.SELL.value
        )

        return (
            WhaleTradeSide.BUY.value
            if buy_activity + buy_pressure + buy_liq_context
            >= sell_activity + sell_pressure + sell_liq_context
            else WhaleTradeSide.SELL.value
        )

    def _calculate_directional_bias(
        self,
        state: SymbolClusterState,
        cluster_side: str,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_directional_bias", _analytics_args)
        except Exception:
            pass
        total_side_notional = 0.0
        total_other_notional = 0.0

        for record in state.activity_records:
            if record.side == cluster_side:
                total_side_notional += record.total_notional
            else:
                total_other_notional += record.total_notional

        for record in state.pressure_records:
            dominant_notional = max(record.buy_notional, record.sell_notional)
            non_dominant_notional = min(record.buy_notional, record.sell_notional)

            if record.dominant_side == cluster_side:
                total_side_notional += dominant_notional
                total_other_notional += non_dominant_notional
            else:
                total_side_notional += non_dominant_notional
                total_other_notional += dominant_notional

        for record in state.liquidation_context_records:
            if record.whale_side == cluster_side:
                total_side_notional += record.whale_total_notional
            else:
                total_other_notional += record.whale_total_notional

        total = total_side_notional + total_other_notional
        if total <= 0:
            return 0.0

        return self._clamp_0_1(total_side_notional / total)

    def _calculate_persistence_score(
        self,
        state: SymbolClusterState,
        current_ts_ms: int,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_persistence_score", _analytics_args)
        except Exception:
            pass
        first_seen = state.cluster_first_seen_ts_ms
        last_seen = state.cluster_last_seen_ts_ms

        if first_seen is None or last_seen is None:
            return 0.0

        duration_ms = max(0, last_seen - first_seen)
        duration_sec = duration_ms / 1000.0

        if duration_sec <= 0:
            return 0.0

        normalized = duration_sec / max(1.0, float(self.config.analysis_window_sec))

        lag_sec = max(0.0, (current_ts_ms - last_seen) / 1000.0)
        freshness_penalty = 1.0
        if lag_sec > 0:
            freshness_penalty = max(
                0.25,
                1.0 - lag_sec / max(1.0, float(self.config.analysis_window_sec)),
            )

        return self._clamp_0_1(normalized * freshness_penalty)

    def _calculate_activity_score(self, state: SymbolClusterState) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_activity_score", _analytics_args)
        except Exception:
            pass
        if not state.activity_records:
            return 0.0

        signal_factor = min(
            1.0,
            len(state.activity_records) / max(1, self.config.min_activity_signals),
        )

        total_notional = sum(record.total_notional for record in state.activity_records)
        notional_factor = min(
            1.0,
            total_notional / max(1.0, self.config.min_total_activity_notional),
        )

        return self._clamp_0_1(0.45 * signal_factor + 0.55 * notional_factor)

    def _calculate_pressure_score(
        self,
        state: SymbolClusterState,
        cluster_side: str,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_pressure_score", _analytics_args)
        except Exception:
            pass
        if not state.pressure_records:
            return 0.0

        aligned_records = [
            record
            for record in state.pressure_records
            if record.dominant_side == cluster_side
        ]
        if not aligned_records:
            return 0.0

        avg_imbalance = (
            sum(record.imbalance_ratio for record in aligned_records)
            / len(aligned_records)
        )
        alignment_ratio = len(aligned_records) / len(state.pressure_records)

        return self._clamp_0_1(0.60 * avg_imbalance + 0.40 * alignment_ratio)

    def _calculate_liquidation_context_score(
        self,
        state: SymbolClusterState,
        cluster_side: str,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_liquidation_context_score", _analytics_args)
        except Exception:
            pass
        if not state.liquidation_context_records:
            return 0.0

        aligned_records = [
            record
            for record in state.liquidation_context_records
            if record.whale_side == cluster_side
        ]
        if not aligned_records:
            return 0.0

        avg_context_strength = (
            sum(record.context_strength for record in aligned_records)
            / len(aligned_records)
        )
        alignment_ratio = len(aligned_records) / len(state.liquidation_context_records)

        return self._clamp_0_1(0.70 * avg_context_strength + 0.30 * alignment_ratio)

    # =========================================================================
    # Emission
    # =========================================================================

    async def _emit_analysis_result(
        self,
        result: WhaleClusterAnalysisResult,
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_analysis_result", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_on_bus or not result.has_signals:
            return

        base_headers: dict[str, Any] = {}
        if source_event_id is not None:
            base_headers["source_event_id"] = source_event_id
        if source_topic is not None:
            base_headers["source_topic"] = source_topic

        if result.whale_cluster_signal is not None:
            signal = result.whale_cluster_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster detected",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "cluster_side": signal.cluster_side,
                        "cluster_score": signal.cluster_score,
                        "continuation_probability": signal.continuation_probability,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_cluster_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

        if result.whale_cluster_update_signal is not None:
            signal = result.whale_cluster_update_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster update emitted",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "cluster_side": signal.cluster_side,
                        "cluster_score": signal.cluster_score,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_cluster_update_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

        if result.whale_cluster_exhaustion_signal is not None:
            signal = result.whale_cluster_exhaustion_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale cluster exhaustion emitted",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "cluster_side": signal.cluster_side,
                        "exhaustion_probability": signal.exhaustion_probability,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_cluster_exhaustion_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

    # =========================================================================
    # State management
    # =========================================================================

    async def _get_state_lock(self, key: WhaleKey) -> asyncio.Lock:
        """
        Повертає lock для конкретного scoped state.

        Registry lock використовується тільки для безпечного створення lock-а.
        Processing не блокує весь WhaleClusterAnalyzer, а лише один WhaleKey.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_state_lock", _analytics_args)
        except Exception:
            pass
        lock = self._state_locks.get(key)
        if lock is not None:
            return lock

        async with self._registry_lock:
            lock = self._state_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._state_locks[key] = lock
            return lock

    async def _snapshot_state_keys(self) -> list[WhaleKey]:
        """
        Повертає snapshot ключів без довгого утримання registry lock.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_snapshot_state_keys", _analytics_args)
        except Exception:
            pass
        async with self._registry_lock:
            return list(self._states.keys())

    def _get_or_create_state(
        self,
        record: WhaleActivityRecord | WhalePressureRecord | WhaleLiquidationContextRecord,
    ) -> SymbolClusterState:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_or_create_state", _analytics_args)
        except Exception:
            pass
        key = record.key
        state = self._states.get(key)
        if state is not None:
            return state

        state = make_symbol_cluster_state(
            activity_window_size=self.config.activity_buffer_size,
            pressure_window_size=self.config.pressure_buffer_size,
            liquidation_context_window_size=self.config.liquidation_context_buffer_size,
            exchange=record.exchange,
            market_type=record.market_type,
            symbol=record.symbol,
            timeframe=record.timeframe,
            exchange_symbol=record.exchange_symbol,
        )
        self._states[key] = state
        return state

    @staticmethod
    def _update_cluster_seen_range(
        state: SymbolClusterState,
        timestamp_ms: int,
    ) -> None:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_cluster_seen_range", _analytics_args)
        except Exception:
            pass
        if state.cluster_first_seen_ts_ms is None:
            state.cluster_first_seen_ts_ms = timestamp_ms
        else:
            state.cluster_first_seen_ts_ms = min(
                state.cluster_first_seen_ts_ms,
                timestamp_ms,
            )

        if state.cluster_last_seen_ts_ms is None:
            state.cluster_last_seen_ts_ms = timestamp_ms
        else:
            state.cluster_last_seen_ts_ms = max(
                state.cluster_last_seen_ts_ms,
                timestamp_ms,
            )

    def _prune_symbol_state(
        self,
        state: SymbolClusterState,
        current_ts_ms: int,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_prune_symbol_state", _analytics_args)
        except Exception:
            pass
        cutoff_ms = current_ts_ms - self.config.analysis_window_sec * 1000

        while (
            state.activity_records
            and state.activity_records[0].timestamp_ms < cutoff_ms
        ):
            state.activity_records.popleft()

        while (
            state.pressure_records
            and state.pressure_records[0].timestamp_ms < cutoff_ms
        ):
            state.pressure_records.popleft()

        while (
            state.liquidation_context_records
            and state.liquidation_context_records[0].timestamp_ms < cutoff_ms
        ):
            state.liquidation_context_records.popleft()

        timestamps = [
            *(record.timestamp_ms for record in state.activity_records),
            *(record.timestamp_ms for record in state.pressure_records),
            *(record.timestamp_ms for record in state.liquidation_context_records),
        ]

        if timestamps:
            state.cluster_first_seen_ts_ms = min(timestamps)
            state.cluster_last_seen_ts_ms = max(timestamps)
        else:
            state.cluster_first_seen_ts_ms = None
            state.cluster_last_seen_ts_ms = None

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup(self) -> None:
        """
        Видаляє неактивні scoped states.

        Запускається через core Scheduler.add_interval_job().

        Важливо:
        - видаляємо тільки state;
        - per-key lock залишаємо, щоб не створювати race condition, коли coroutine
          вже чекає старий lock, а інший coroutine створює новий lock для того ж key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup", _analytics_args)
        except Exception:
            pass
        ttl = self.config.stats_ttl_sec
        if ttl <= 0:
            return

        now_mono = time.monotonic()
        stale_keys: list[WhaleKey] = []

        for key in await self._snapshot_state_keys():
            state_lock = await self._get_state_lock(key)

            async with state_lock:
                state = self._states.get(key)
                if state is None:
                    continue

                if (now_mono - state.last_update_ts_monotonic) < ttl:
                    continue

                self._states.pop(key, None)
                stale_keys.append(key)

        if stale_keys:
            self.logger.info(
                "Cleaned stale WhaleClusterAnalyzer scoped states",
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

    def get_key_state(self, key: WhaleKey) -> dict[str, Any]:
        """
        Read-only snapshot API.

        Це sync read API, тому він не бере async lock. Для dashboard/stats це ок;
        mutation path захищений per-key lock-ами.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_state", _analytics_args)
        except Exception:
            pass
        state = self._states.get(key)
        scope = whale_key_to_dict(key)

        if state is None:
            return {
                **scope,
                "scope": scope,
                "exists": False,
            }

        return {
            **scope,
            "scope": scope,
            "exists": True,
            **state.to_dict(),
        }

    def get_symbol_state(
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
        Якщо ні — повертає всі scope-и для symbol.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_symbol_state", _analytics_args)
        except Exception:
            pass
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
            return self.get_key_state(key)

        matching = [
            state.to_dict()
            for key, state in self._states.items()
            if whale_key_to_dict(key)["symbol"] == normalized_symbol
        ]

        return {
            "symbol": normalized_symbol,
            "exists": bool(matching),
            "scopes": matching,
        }

    def get_all_states(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_all_states", _analytics_args)
        except Exception:
            pass
        return {
            self.scoped_mapping_key(key): state.to_dict()
            for key, state in self._states.items()
        }

    async def reset_key(self, key: WhaleKey) -> None:
        """
        Reset одного scoped state.

        Lock не видаляється спеціально, щоб не створювати race condition з coroutine,
        який уже очікує цей per-key lock.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_key", _analytics_args)
        except Exception:
            pass
        state_lock = await self._get_state_lock(key)

        async with state_lock:
            self._states.pop(key, None)

        self.logger.info(
            "Reset WhaleClusterAnalyzer scoped state",
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_symbol", _analytics_args)
        except Exception:
            pass
        try:
            normalized_symbol = normalize_symbol(symbol)
        except ValueError:
            return

        if exchange is not None or market_type is not None or timeframe is not None:
            removed_keys = [
                self.make_key(
                    exchange=exchange or self.default_exchange,
                    market_type=market_type or self.default_market_type,
                    symbol=normalized_symbol,
                    timeframe=timeframe or self.default_timeframe,
                )
            ]
        else:
            removed_keys = [
                key
                for key in await self._snapshot_state_keys()
                if whale_key_to_dict(key)["symbol"] == normalized_symbol
            ]

        for key in removed_keys:
            state_lock = await self._get_state_lock(key)

            async with state_lock:
                self._states.pop(key, None)

        self.logger.info(
            "Reset WhaleClusterAnalyzer symbol state",
            extra={
                "component": self.component_name,
                "symbol": normalized_symbol,
                "removed_states_count": len(removed_keys),
            },
        )

    async def reset_all(self) -> None:
        """
        Reset усіх scoped states.

        Використовувати для manual/test/replay reset.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_all", _analytics_args)
        except Exception:
            pass
        for key in await self._snapshot_state_keys():
            state_lock = await self._get_state_lock(key)

            async with state_lock:
                self._states.pop(key, None)

        self.logger.info(
            "Reset all WhaleClusterAnalyzer states",
            extra={"component": self.component_name},
        )

    def get_healthcheck(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_healthcheck", _analytics_args)
        except Exception:
            pass
        health = super().get_healthcheck()
        health.update(
            {
                "enabled": self.config.enabled,
                "tracked_scopes": len(self._states),
                "state_locks": len(self._state_locks),
                "locking": "per_whale_key",
                "production_input_topics": list(self.config.production_input_topics),
                "whale_activity_event_name": self.config.whale_activity_event_name,
                "whale_pressure_event_name": self.config.whale_pressure_event_name,
                "whale_liquidation_context_event_name": (
                    self.config.whale_liquidation_context_event_name
                ),
                "whale_cluster_event_name": self.config.whale_cluster_event_name,
                "whale_cluster_update_event_name": (
                    self.config.whale_cluster_update_event_name
                ),
                "whale_cluster_exhaustion_event_name": (
                    self.config.whale_cluster_exhaustion_event_name
                ),
                "scope": "exchange:market_type:symbol:timeframe",
            }
        )
        return health

    # =========================================================================
    # Normalization helpers
    # =========================================================================

    def _normalize_whale_activity_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleActivityRecord | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_whale_activity_payload", _analytics_args)
        except Exception:
            pass
        try:
            event = dict(event_payload)
            raw_payload = event.get("data", event)

            if not isinstance(raw_payload, Mapping):
                self.logger.debug(
                    "Whale activity event dropped: payload data is not mapping",
                    extra={
                        "component": self.component_name,
                        "payload_type": type(raw_payload).__name__,
                    },
                )
                return None

            payload = dict(raw_payload)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
                or event.get("symbol")
            )
            side = self._normalize_side(payload.get("side") or payload.get("S"))
            trade_count = self._safe_int(payload.get("trade_count"))
            total_notional = self._safe_float(payload.get("total_notional"))
            avg_notional = self._safe_float(payload.get("avg_notional"))
            max_notional = self._safe_float(payload.get("max_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None
            if trade_count is None or trade_count <= 0:
                return None
            if total_notional is None or total_notional <= 0:
                return None
            if avg_notional is None or avg_notional <= 0:
                return None
            if max_notional is None or max_notional <= 0:
                return None
            if window_sec is None or window_sec <= 0:
                return None

            normalized_symbol = normalize_symbol(symbol)
            exchange = normalize_exchange(
                payload.get("exchange")
                or event.get("exchange")
                or self.default_exchange
            )
            market_type = normalize_market_type(
                payload.get("market_type")
                or event.get("market_type")
                or self.default_market_type
            )
            timeframe = normalize_timeframe(
                payload.get("timeframe")
                or event.get("timeframe")
                or self.default_timeframe
            )
            exchange_symbol = normalize_exchange_symbol(
                payload.get("exchange_symbol")
                or event.get("exchange_symbol")
                or payload.get("raw_symbol")
                or payload.get("s"),
                fallback_symbol=normalized_symbol,
            )

            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "source": event.get("source"),
                    "scope": {
                        "exchange": exchange,
                        "market_type": market_type,
                        "symbol": normalized_symbol,
                        "timeframe": timeframe,
                    },
                }
            )

            return WhaleActivityRecord(
                symbol=normalized_symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                side=side,
                trade_count=trade_count,
                total_notional=total_notional,
                avg_notional=avg_notional,
                max_notional=max_notional,
                window_sec=window_sec,
                timestamp_ms=timestamp_ms,
                raw_event=event,
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize whale activity payload",
                extra={"component": self.component_name},
            )
            return None

    def _normalize_whale_pressure_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
    ) -> WhalePressureRecord | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_whale_pressure_payload", _analytics_args)
        except Exception:
            pass
        try:
            event = dict(event_payload)
            raw_payload = event.get("data", event)

            if not isinstance(raw_payload, Mapping):
                self.logger.debug(
                    "Whale pressure event dropped: payload data is not mapping",
                    extra={
                        "component": self.component_name,
                        "payload_type": type(raw_payload).__name__,
                    },
                )
                return None

            payload = dict(raw_payload)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
                or event.get("symbol")
            )
            dominant_side = self._normalize_side(
                payload.get("dominant_side")
                or payload.get("side")
                or payload.get("S")
            )
            buy_trade_count = self._safe_int(payload.get("buy_trade_count"))
            sell_trade_count = self._safe_int(payload.get("sell_trade_count"))
            buy_notional = self._safe_float(payload.get("buy_notional"))
            sell_notional = self._safe_float(payload.get("sell_notional"))
            total_notional = self._safe_float(payload.get("total_notional"))
            imbalance_ratio = self._safe_float(payload.get("imbalance_ratio"))
            net_flow_notional = self._safe_float(payload.get("net_flow_notional"))
            window_sec = self._safe_int(payload.get("window_sec"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or dominant_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if buy_trade_count is None or buy_trade_count < 0:
                return None
            if sell_trade_count is None or sell_trade_count < 0:
                return None
            if buy_notional is None or buy_notional < 0:
                return None
            if sell_notional is None or sell_notional < 0:
                return None
            if total_notional is None or total_notional <= 0:
                return None
            if imbalance_ratio is None or not 0.0 <= imbalance_ratio <= 1.0:
                return None
            if net_flow_notional is None:
                return None
            if window_sec is None or window_sec <= 0:
                return None

            normalized_symbol = normalize_symbol(symbol)
            exchange = normalize_exchange(
                payload.get("exchange")
                or event.get("exchange")
                or self.default_exchange
            )
            market_type = normalize_market_type(
                payload.get("market_type")
                or event.get("market_type")
                or self.default_market_type
            )
            timeframe = normalize_timeframe(
                payload.get("timeframe")
                or event.get("timeframe")
                or self.default_timeframe
            )
            exchange_symbol = normalize_exchange_symbol(
                payload.get("exchange_symbol")
                or event.get("exchange_symbol")
                or payload.get("raw_symbol")
                or payload.get("s"),
                fallback_symbol=normalized_symbol,
            )

            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "source": event.get("source"),
                    "scope": {
                        "exchange": exchange,
                        "market_type": market_type,
                        "symbol": normalized_symbol,
                        "timeframe": timeframe,
                    },
                }
            )

            return WhalePressureRecord(
                symbol=normalized_symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                dominant_side=dominant_side,
                buy_trade_count=buy_trade_count,
                sell_trade_count=sell_trade_count,
                buy_notional=buy_notional,
                sell_notional=sell_notional,
                total_notional=total_notional,
                imbalance_ratio=imbalance_ratio,
                net_flow_notional=net_flow_notional,
                window_sec=window_sec,
                timestamp_ms=timestamp_ms,
                raw_event=event,
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize whale pressure payload",
                extra={"component": self.component_name},
            )
            return None

    def _normalize_whale_liquidation_context_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleLiquidationContextRecord | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_whale_liquidation_context_payload", _analytics_args)
        except Exception:
            pass
        try:
            event = dict(event_payload)
            raw_payload = event.get("data", event)

            if not isinstance(raw_payload, Mapping):
                self.logger.debug(
                    "Whale liquidation context event dropped: payload data is not mapping",
                    extra={
                        "component": self.component_name,
                        "payload_type": type(raw_payload).__name__,
                    },
                )
                return None

            payload = dict(raw_payload)

            symbol = self._normalize_symbol(
                payload.get("symbol")
                or payload.get("s")
                or payload.get("instrument")
                or event.get("symbol")
            )
            whale_side = self._normalize_side(payload.get("whale_side"))
            liquidation_side = self._normalize_side(payload.get("liquidation_side"))
            whale_total_notional = self._safe_float(payload.get("whale_total_notional"))
            whale_trade_count = self._safe_int(payload.get("whale_trade_count"))
            liquidation_total_notional = self._safe_float(
                payload.get("liquidation_total_notional")
            )
            liquidation_count = self._safe_int(payload.get("liquidation_count"))
            context_strength = self._safe_float(payload.get("context_strength"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None:
                return None
            if whale_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if liquidation_side == WhaleTradeSide.UNKNOWN.value:
                return None
            if whale_total_notional is None or whale_total_notional <= 0:
                return None
            if whale_trade_count is None or whale_trade_count <= 0:
                return None
            if liquidation_total_notional is None or liquidation_total_notional <= 0:
                return None
            if liquidation_count is None or liquidation_count <= 0:
                return None
            if context_strength is None or not 0.0 <= context_strength <= 1.0:
                return None

            normalized_symbol = normalize_symbol(symbol)
            exchange = normalize_exchange(
                payload.get("exchange")
                or event.get("exchange")
                or self.default_exchange
            )
            market_type = normalize_market_type(
                payload.get("market_type")
                or event.get("market_type")
                or self.default_market_type
            )
            timeframe = normalize_timeframe(
                payload.get("timeframe")
                or event.get("timeframe")
                or self.default_timeframe
            )
            exchange_symbol = normalize_exchange_symbol(
                payload.get("exchange_symbol")
                or event.get("exchange_symbol")
                or payload.get("raw_symbol")
                or payload.get("s"),
                fallback_symbol=normalized_symbol,
            )

            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "source": event.get("source"),
                    "scope": {
                        "exchange": exchange,
                        "market_type": market_type,
                        "symbol": normalized_symbol,
                        "timeframe": timeframe,
                    },
                }
            )

            return WhaleLiquidationContextRecord(
                symbol=normalized_symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                whale_side=whale_side,
                whale_total_notional=whale_total_notional,
                whale_trade_count=whale_trade_count,
                liquidation_side=liquidation_side,
                liquidation_total_notional=liquidation_total_notional,
                liquidation_count=liquidation_count,
                context_strength=self._clamp_0_1(context_strength),
                timestamp_ms=timestamp_ms,
                raw_event=event,
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize whale liquidation context payload",
                extra={"component": self.component_name},
            )
            return None

    @staticmethod
    def _normalize_symbol(value: Any) -> str | None:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_symbol", _analytics_args)
        except Exception:
            pass
        try:
            return normalize_symbol(value)
        except ValueError:
            return None

    @staticmethod
    def _normalize_side(value: Any) -> str:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_side", _analytics_args)
        except Exception:
            pass
        return WhaleTradeSide.normalize(value).value

    @staticmethod
    def _extract_timestamp_ms(payload: Mapping[str, Any]) -> int:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_timestamp_ms", _analytics_args)
        except Exception:
            pass
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
    def _safe_float(value: Any) -> float | None:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_float", _analytics_args)
        except Exception:
            pass
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
    def _safe_int(value: Any) -> int | None:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_int", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        try:
            _analytics_class_name = "WhaleClusterAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_safe_str", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None

        text = str(value).strip()
        return text or None


__all__ = [
    "WhaleClusterAnalyzer",
]