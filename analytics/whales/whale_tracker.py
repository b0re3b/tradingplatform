from __future__ import annotations
from core.logger import get_logger

import asyncio
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from core.event_bus import Event, EventBus, EventPriority
from core.scheduler import Scheduler

from analytics.whales.base import BaseWhaleComponent
from analytics.whales.config import WhaleTrackerConfig
from analytics.whales.enums import WhaleComponentName, WhaleTradeSide
from analytics.whales.models import (
    LiquidationRecord,
    SymbolTrackerState,
    WhaleActivitySignal,
    WhaleKey,
    WhaleLiquidationContextSignal,
    WhalePressureSignal,
    WhaleTradeRecord,
    WhaleTrackerResult,
    make_symbol_tracker_state,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    whale_key_to_dict,
)


class WhaleTracker(BaseWhaleComponent):
    """
    High-level tracker для whale activity / pressure / liquidation context.

    Production input:
        analytics.whales.large_trade
        market.liquidations.updated або analytics.liquidations.*

    Legacy raw input:
        market.liquidation

    Output:
        analytics.whales.whale_activity
        analytics.whales.whale_pressure
        analytics.whales.whale_liquidation_context

    Correct production flow:
        exchange adapters
            -> market.trade
            -> TradesCache
            -> market.trades.updated
            -> LargeTradeDetector
            -> analytics.whales.large_trade
            -> WhaleTracker

        liquidation stream/cache
            -> market.liquidation
            -> LiquidationCache / liquidation analytics layer
            -> market.liquidations.updated або analytics.liquidations.*
            -> WhaleTracker

    Scope:
        exchange + market_type + symbol + timeframe

    Важливо:
    - не читає біржові adapters напряму;
    - не слухає raw market.liquidation у production, якщо legacy mode не дозволений;
    - state не змішує різні біржі / market_type / timeframe;
    - cleanup запускається тільки через Scheduler.add_interval_job();
    - власних uncontrolled asyncio cleanup loops немає;
    - state mutation блокується per WhaleKey, а не глобально на весь tracker.
    """

    def __init__(
        self,
        *,
        config: WhaleTrackerConfig,
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
            component_name=WhaleComponentName.WHALE_TRACKER.value,
            event_bus=event_bus,
            scheduler=scheduler,
            default_exchange=config.default_exchange,
            default_market_type=config.default_market_type,
            default_timeframe=config.default_timeframe,
        )

        self.config = config
        self.config.validate()

        self._states: dict[WhaleKey, SymbolTrackerState] = {}

        # Registry lock захищає створення lock-ів і короткі registry/snapshot операції.
        # Бізнес-обробка блокує тільки конкретний WhaleKey.
        self._state_locks: dict[WhaleKey, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def register(self) -> None:
        """
        Зареєструвати EventBus subscriptions.

        Production:
            analytics.whales.large_trade
            market.liquidations.updated / analytics.liquidations.*

        Legacy:
            market.liquidation тільки якщо config.allow_legacy_raw_topics=True.
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
                "WhaleTracker registration skipped: disabled by config",
                extra={"component": self.component_name},
            )
            return

        self._subscribe_production(
            self.config.large_trade_event_name,
            self.handle_large_trade_event,
            name="analytics.whales.whale_tracker.handle_large_trade_event",
        )

        if self.config.subscribe_liquidations:
            self._subscribe_production_many(
                self.config.liquidation_event_patterns,
                self.handle_liquidation_event,
                name="analytics.whales.whale_tracker.handle_liquidation_event",
            )

            if self.config.allow_legacy_raw_topics:
                self._subscribe_legacy_raw(
                    self.config.raw_liquidation_event_name,
                    self.handle_raw_liquidation_event,
                    name="analytics.whales.whale_tracker.handle_raw_liquidation_event",
                    allow_legacy_raw_topics=self.config.allow_legacy_raw_topics,
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
            self.logger.warning("WhaleTracker already started")
            return

        if not self.config.enabled:
            self.logger.info("WhaleTracker is disabled by config")
            return

        await self.register()

        self._add_interval_job(
            name="analytics.whales.whale_tracker.cleanup",
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
            "WhaleTracker started",
            extra={
                "component": self.component_name,
                "production_input_topics": list(self.config.production_input_topics),
                "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
                "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
                "large_trade_event_name": self.config.large_trade_event_name,
                "liquidation_event_patterns": list(self.config.liquidation_event_patterns),
                "whale_activity_event_name": self.config.whale_activity_event_name,
                "whale_pressure_event_name": self.config.whale_pressure_event_name,
                "whale_liquidation_context_event_name": (
                    self.config.whale_liquidation_context_event_name
                ),
                "cluster_window_sec": self.config.cluster_window_sec,
                "pressure_window_sec": self.config.pressure_window_sec,
                "liquidation_window_sec": self.config.liquidation_window_sec,
                "subscribe_liquidations": self.config.subscribe_liquidations,
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
            "WhaleTracker stopped",
            extra={"component": self.component_name},
        )

    # =========================================================================
    # EventBus handlers
    # =========================================================================

    async def handle_large_trade_event(self, event: Event) -> None:
        """
        EventBus handler для analytics.whales.large_trade.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_large_trade_event", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._payload_from_event(event)

            await self.process_large_trade_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing large trade event",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

    async def handle_liquidation_event(self, event: Event) -> None:
        """
        Production EventBus handler для market.liquidations.updated /
        analytics.liquidations.*.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_liquidation_event", _analytics_args)
        except Exception:
            pass
        await self._handle_liquidation_event(
            event,
            allow_raw_payload=False,
        )

    async def handle_raw_liquidation_event(self, event: Event) -> None:
        """
        Legacy raw handler для market.liquidation.

        Використовувати тільки для migration/test/manual режиму.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "handle_raw_liquidation_event", _analytics_args)
        except Exception:
            pass
        if not self.config.allow_legacy_raw_topics:
            self.logger.warning(
                "Raw liquidation event skipped: legacy raw topics are disabled",
                extra={
                    "component": self.component_name,
                    "topic": getattr(event, "topic", None),
                },
            )
            return

        await self._handle_liquidation_event(
            event,
            allow_raw_payload=True,
        )

    async def _handle_liquidation_event(
        self,
        event: Event,
        *,
        allow_raw_payload: bool,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_liquidation_event", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._payload_from_event(event)

            await self.process_liquidation_payload(
                payload,
                correlation_id=self._event_correlation_id(event),
                source_event_id=getattr(event, "event_id", None),
                source_topic=getattr(event, "topic", None),
                allow_raw_payload=allow_raw_payload,
            )

        except Exception:
            self.logger.exception(
                "Unhandled error while processing liquidation event",
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

    async def process_large_trade_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
    ) -> WhaleTrackerResult:
        """
        Обробити payload large_trade signal.

        Використовується:
        - EventBus handler-ом;
        - тестами;
        - backtesting/replay.

        Locking:
        - normalization виконується без lock;
        - mutation/detection блокується тільки для record.key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_large_trade_payload", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return WhaleTrackerResult()

        record = self._normalize_large_trade_payload(payload)
        if record is None:
            return WhaleTrackerResult()

        if not self.config.should_process_key(record.key):
            return WhaleTrackerResult()

        state_lock = await self._get_state_lock(record.key)

        async with state_lock:
            state = self._get_or_create_state(record)
            state.large_trades.append(record)
            state.total_large_trades_seen += 1
            state.touch()

            self._prune_symbol_state(state, record.timestamp_ms)

            whale_activity_signal = self._detect_whale_activity(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )
            whale_pressure_signal = self._detect_whale_pressure(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )
            whale_liquidation_context_signal = self._detect_whale_liquidation_context(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        result = WhaleTrackerResult(
            whale_activity_signal=whale_activity_signal,
            whale_pressure_signal=whale_pressure_signal,
            whale_liquidation_context_signal=whale_liquidation_context_signal,
        )

        await self._emit_detected_signals(
            result,
            correlation_id=correlation_id,
            source_event_id=source_event_id,
            source_topic=source_topic,
        )
        return result

    async def process_liquidation_payload(
        self,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        correlation_id: str | None = None,
        source_event_id: str | None = None,
        source_topic: str | None = None,
        allow_raw_payload: bool = False,
    ) -> WhaleLiquidationContextSignal | None:
        """
        Обробити payload liquidation event.

        Production payload має приходити з data-layer / liquidation analytics topic.
        Raw market.liquidation дозволений тільки якщо allow_raw_payload=True.

        Locking:
        - normalization виконується без lock;
        - mutation/detection блокується тільки для record.key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_liquidation_payload", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.subscribe_liquidations:
            return None

        record = self._normalize_liquidation_payload(
            payload,
            allow_raw_payload=allow_raw_payload,
            source_topic=source_topic,
        )
        if record is None:
            return None

        if not self.config.should_process_key(record.key):
            return None

        state_lock = await self._get_state_lock(record.key)

        async with state_lock:
            state = self._get_or_create_state(record)
            state.liquidations.append(record)
            state.total_liquidations_seen += 1
            state.touch()

            self._prune_symbol_state(state, record.timestamp_ms)

            whale_liquidation_context_signal = self._detect_whale_liquidation_context(
                state=state,
                current_ts_ms=record.timestamp_ms,
            )

        if whale_liquidation_context_signal is not None:
            await self._emit_detected_signals(
                WhaleTrackerResult(
                    whale_liquidation_context_signal=whale_liquidation_context_signal,
                ),
                correlation_id=correlation_id,
                source_event_id=source_event_id,
                source_topic=source_topic,
            )

        return whale_liquidation_context_signal

    async def process_large_trade_event(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleTrackerResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_large_trade_event", _analytics_args)
        except Exception:
            pass
        return await self.process_large_trade_payload(event)

    async def process_liquidation_event(
        self,
        event: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleLiquidationContextSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "process_liquidation_event", _analytics_args)
        except Exception:
            pass
        return await self.process_liquidation_payload(event)

    # =========================================================================
    # Detection logic
    # =========================================================================

    def _detect_whale_activity(
        self,
        *,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> WhaleActivitySignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_whale_activity", _analytics_args)
        except Exception:
            pass
        cluster_start_ms = current_ts_ms - self.config.cluster_window_sec * 1000

        buys = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= cluster_start_ms
            and trade.side == WhaleTradeSide.BUY.value
        ]
        sells = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= cluster_start_ms
            and trade.side == WhaleTradeSide.SELL.value
        ]

        buy_signal = self._build_whale_activity_signal_if_triggered(
            side=WhaleTradeSide.BUY.value,
            trades=buys,
            state=state,
            current_ts_ms=current_ts_ms,
        )
        if buy_signal is not None:
            return buy_signal

        return self._build_whale_activity_signal_if_triggered(
            side=WhaleTradeSide.SELL.value,
            trades=sells,
            state=state,
            current_ts_ms=current_ts_ms,
        )

    def _build_whale_activity_signal_if_triggered(
        self,
        *,
        side: str,
        trades: Sequence[WhaleTradeRecord],
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> WhaleActivitySignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_whale_activity_signal_if_triggered", _analytics_args)
        except Exception:
            pass
        if len(trades) < self.config.cluster_min_trades:
            return None

        total_notional = sum(trade.notional for trade in trades)
        if total_notional < self.config.cluster_min_total_notional:
            return None

        if not self._passes_cooldown(
            state.last_whale_activity_signal_ts_monotonic,
            self.config.whale_activity_cooldown_sec,
        ):
            return None

        max_notional = max(trade.notional for trade in trades)
        avg_notional = total_notional / len(trades)

        signal = WhaleActivitySignal(
            exchange=state.exchange,
            market_type=state.market_type,
            symbol=state.symbol,
            timeframe=state.timeframe,
            exchange_symbol=state.exchange_symbol,
            side=side,
            trade_count=len(trades),
            total_notional=total_notional,
            avg_notional=avg_notional,
            max_notional=max_notional,
            window_sec=self.config.cluster_window_sec,
            timestamp_ms=current_ts_ms,
            metadata={
                "scope": whale_key_to_dict(state.key),
            },
        )

        state.whale_activity_signals_emitted += 1
        state.last_whale_activity_signal_ts_monotonic = time.monotonic()
        return signal

    def _detect_whale_pressure(
        self,
        *,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> WhalePressureSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_whale_pressure", _analytics_args)
        except Exception:
            pass
        pressure_start_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        trades = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= pressure_start_ms
        ]

        if len(trades) < self.config.pressure_min_trades:
            return None

        buy_trades = [
            trade for trade in trades
            if trade.side == WhaleTradeSide.BUY.value
        ]
        sell_trades = [
            trade for trade in trades
            if trade.side == WhaleTradeSide.SELL.value
        ]

        buy_notional = sum(trade.notional for trade in buy_trades)
        sell_notional = sum(trade.notional for trade in sell_trades)
        total_notional = buy_notional + sell_notional

        if total_notional < self.config.pressure_min_total_notional:
            return None

        dominant_side = (
            WhaleTradeSide.BUY.value
            if buy_notional >= sell_notional
            else WhaleTradeSide.SELL.value
        )
        dominant_notional = max(buy_notional, sell_notional)
        imbalance_ratio = dominant_notional / total_notional if total_notional > 0 else 0.0

        if imbalance_ratio < self.config.pressure_imbalance_ratio_threshold:
            return None

        if not self._passes_cooldown(
            state.last_whale_pressure_signal_ts_monotonic,
            self.config.whale_pressure_cooldown_sec,
        ):
            return None

        signal = WhalePressureSignal(
            exchange=state.exchange,
            market_type=state.market_type,
            symbol=state.symbol,
            timeframe=state.timeframe,
            exchange_symbol=state.exchange_symbol,
            dominant_side=dominant_side,
            buy_trade_count=len(buy_trades),
            sell_trade_count=len(sell_trades),
            buy_notional=buy_notional,
            sell_notional=sell_notional,
            total_notional=total_notional,
            imbalance_ratio=imbalance_ratio,
            net_flow_notional=buy_notional - sell_notional,
            window_sec=self.config.pressure_window_sec,
            timestamp_ms=current_ts_ms,
            metadata={
                "scope": whale_key_to_dict(state.key),
            },
        )

        state.whale_pressure_signals_emitted += 1
        state.last_whale_pressure_signal_ts_monotonic = time.monotonic()
        return signal

    def _detect_whale_liquidation_context(
        self,
        *,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> WhaleLiquidationContextSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_whale_liquidation_context", _analytics_args)
        except Exception:
            pass
        whale_window_start_ms = current_ts_ms - self.config.cluster_window_sec * 1000
        liquidation_window_start_ms = current_ts_ms - self.config.liquidation_window_sec * 1000

        recent_whale_trades = [
            trade
            for trade in state.large_trades
            if trade.timestamp_ms >= whale_window_start_ms
        ]
        recent_liquidations = [
            liquidation
            for liquidation in state.liquidations
            if liquidation.timestamp_ms >= liquidation_window_start_ms
        ]

        if not recent_whale_trades or not recent_liquidations:
            return None

        buy_whale_trades = [
            trade for trade in recent_whale_trades
            if trade.side == WhaleTradeSide.BUY.value
        ]
        sell_whale_trades = [
            trade for trade in recent_whale_trades
            if trade.side == WhaleTradeSide.SELL.value
        ]

        buy_whale_notional = sum(trade.notional for trade in buy_whale_trades)
        sell_whale_notional = sum(trade.notional for trade in sell_whale_trades)

        if (
            max(buy_whale_notional, sell_whale_notional)
            < self.config.liquidation_context_min_notional
        ):
            return None

        whale_side = (
            WhaleTradeSide.BUY.value
            if buy_whale_notional >= sell_whale_notional
            else WhaleTradeSide.SELL.value
        )

        whale_trades = (
            buy_whale_trades
            if whale_side == WhaleTradeSide.BUY.value
            else sell_whale_trades
        )

        opposite_liquidation_side = (
            WhaleTradeSide.SELL.value
            if whale_side == WhaleTradeSide.BUY.value
            else WhaleTradeSide.BUY.value
        )

        related_liquidations = [
            liquidation
            for liquidation in recent_liquidations
            if liquidation.side == opposite_liquidation_side
        ]

        if not related_liquidations:
            return None

        whale_total_notional = sum(trade.notional for trade in whale_trades)
        liquidation_total_notional = sum(
            liquidation.notional for liquidation in related_liquidations
        )

        if whale_total_notional < self.config.liquidation_context_min_notional:
            return None

        if not self._passes_cooldown(
            state.last_whale_liquidation_context_signal_ts_monotonic,
            self.config.whale_liquidation_context_cooldown_sec,
        ):
            return None

        context_strength = self._calculate_context_strength(
            whale_total_notional=whale_total_notional,
            liquidation_total_notional=liquidation_total_notional,
            whale_trade_count=len(whale_trades),
            liquidation_count=len(related_liquidations),
        )

        signal = WhaleLiquidationContextSignal(
            exchange=state.exchange,
            market_type=state.market_type,
            symbol=state.symbol,
            timeframe=state.timeframe,
            exchange_symbol=state.exchange_symbol,
            whale_side=whale_side,
            whale_total_notional=whale_total_notional,
            whale_trade_count=len(whale_trades),
            liquidation_side=opposite_liquidation_side,
            liquidation_total_notional=liquidation_total_notional,
            liquidation_count=len(related_liquidations),
            context_strength=context_strength,
            timestamp_ms=current_ts_ms,
            metadata={
                "scope": whale_key_to_dict(state.key),
            },
        )

        state.whale_liquidation_context_signals_emitted += 1
        state.last_whale_liquidation_context_signal_ts_monotonic = time.monotonic()
        return signal

    def _calculate_context_strength(
        self,
        *,
        whale_total_notional: float,
        liquidation_total_notional: float,
        whale_trade_count: int,
        liquidation_count: int,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_context_strength", _analytics_args)
        except Exception:
            pass
        if whale_total_notional <= 0:
            return 0.0

        liquidation_ratio = liquidation_total_notional / whale_total_notional
        trade_factor = min(
            1.0,
            whale_trade_count / max(1, self.config.cluster_min_trades),
        )
        liquidation_factor = min(1.0, liquidation_count / 3.0)

        raw_score = (
            0.6 * min(1.0, liquidation_ratio)
            + 0.2 * trade_factor
            + 0.2 * liquidation_factor
        )
        return self._clamp_0_1(raw_score)

    # =========================================================================
    # Emission
    # =========================================================================

    async def _emit_detected_signals(
        self,
        result: WhaleTrackerResult,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_detected_signals", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_on_bus or not result.has_signals:
            return

        base_headers: dict[str, Any] = {}
        if source_event_id is not None:
            base_headers["source_event_id"] = source_event_id
        if source_topic is not None:
            base_headers["source_topic"] = source_topic

        if result.whale_activity_signal is not None:
            signal = result.whale_activity_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale activity detected",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "side": signal.side,
                        "trade_count": signal.trade_count,
                        "total_notional": signal.total_notional,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_activity_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

        if result.whale_pressure_signal is not None:
            signal = result.whale_pressure_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale pressure detected",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "dominant_side": signal.dominant_side,
                        "imbalance_ratio": signal.imbalance_ratio,
                        "total_notional": signal.total_notional,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_pressure_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

        if result.whale_liquidation_context_signal is not None:
            signal = result.whale_liquidation_context_signal

            if self.config.log_signals:
                self.logger.info(
                    "Whale liquidation context detected",
                    extra={
                        "component": self.component_name,
                        "exchange": signal.exchange,
                        "market_type": signal.market_type,
                        "symbol": signal.symbol,
                        "timeframe": signal.timeframe,
                        "whale_side": signal.whale_side,
                        "liquidation_side": signal.liquidation_side,
                        "context_strength": signal.context_strength,
                        "scope": whale_key_to_dict(signal.key),
                    },
                )

            headers = {
                **base_headers,
                "scope": str(whale_key_to_dict(signal.key)),
            }

            await self._emit(
                self.config.whale_liquidation_context_event_name,
                signal,
                priority=EventPriority.NORMAL,
                source=self.component_name,
                correlation_id=correlation_id,
                headers=headers,
            )

    # =========================================================================
    # State locking / state management
    # =========================================================================

    async def _get_state_lock(self, key: WhaleKey) -> asyncio.Lock:
        """
        Повертає lock для конкретного scoped state.

        Registry lock використовується тільки для безпечного створення lock-а.
        Processing не блокує весь WhaleTracker, а лише один WhaleKey.
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
        Повертає snapshot ключів для cleanup/reset без довгого утримання registry lock.
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
        record: WhaleTradeRecord | LiquidationRecord,
    ) -> SymbolTrackerState:
        """
        Має викликатися під lock-ом конкретного record.key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_or_create_state", _analytics_args)
        except Exception:
            pass
        key = record.key
        state = self._states.get(key)
        if state is not None:
            return state

        state = make_symbol_tracker_state(
            large_trade_window_size=self.config.large_trade_buffer_size,
            liquidation_window_size=self.config.liquidation_buffer_size,
            exchange=record.exchange,
            market_type=record.market_type,
            symbol=record.symbol,
            timeframe=record.timeframe,
            exchange_symbol=record.exchange_symbol,
        )
        self._states[key] = state
        return state

    def _prune_symbol_state(
        self,
        state: SymbolTrackerState,
        current_ts_ms: int,
    ) -> None:
        """
        Має викликатися під lock-ом конкретного state.key.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_prune_symbol_state", _analytics_args)
        except Exception:
            pass
        cluster_cutoff_ms = current_ts_ms - self.config.cluster_window_sec * 1000
        pressure_cutoff_ms = current_ts_ms - self.config.pressure_window_sec * 1000
        liquidation_cutoff_ms = current_ts_ms - self.config.liquidation_window_sec * 1000

        trade_cutoff_ms = min(cluster_cutoff_ms, pressure_cutoff_ms)

        while (
            state.large_trades
            and state.large_trades[0].timestamp_ms < trade_cutoff_ms
        ):
            state.large_trades.popleft()

        while (
            state.liquidations
            and state.liquidations[0].timestamp_ms < liquidation_cutoff_ms
        ):
            state.liquidations.popleft()

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
                "Cleaned stale WhaleTracker scoped states",
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

        Це sync read API, тому він не бере async lock.
        Для dashboard/stats це прийнятно; mutation path захищений per-key lock-ами.
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
            "Reset WhaleTracker scoped state",
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
            "Reset WhaleTracker symbol state",
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
            "Reset all WhaleTracker states",
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
                "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
                "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
                "large_trade_event_name": self.config.large_trade_event_name,
                "liquidation_event_patterns": list(self.config.liquidation_event_patterns),
                "subscribe_liquidations": self.config.subscribe_liquidations,
                "whale_activity_event_name": self.config.whale_activity_event_name,
                "whale_pressure_event_name": self.config.whale_pressure_event_name,
                "whale_liquidation_context_event_name": (
                    self.config.whale_liquidation_context_event_name
                ),
                "scope": "exchange:market_type:symbol:timeframe",
            }
        )
        return health

    # =========================================================================
    # Normalization helpers
    # =========================================================================

    def _normalize_large_trade_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
    ) -> WhaleTradeRecord | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_large_trade_payload", _analytics_args)
        except Exception:
            pass
        try:
            event = dict(event_payload)
            raw_payload = event.get("data", event)

            if not isinstance(raw_payload, Mapping):
                self.logger.debug(
                    "Large trade event dropped: payload data is not mapping",
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
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
                or payload.get("dominant_side")
            )
            price = self._safe_float(payload.get("price") or payload.get("p"))
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            notional = self._safe_float(payload.get("notional"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None

            if price is None or price <= 0:
                return None

            if quantity is None or quantity <= 0:
                return None

            if notional is None:
                notional = price * quantity

            if notional <= 0:
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

            return WhaleTradeRecord(
                symbol=normalized_symbol,
                side=side,
                notional=notional,
                price=price,
                quantity=quantity,
                timestamp_ms=timestamp_ms,
                zscore=self._safe_float(payload.get("zscore")) or 0.0,
                trigger_type=str(payload.get("trigger_type") or "unknown"),
                trade_id=self._safe_str(
                    payload.get("trade_id")
                    or payload.get("id")
                    or payload.get("t")
                ),
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                raw_event=event,
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize large trade payload",
                extra={"component": self.component_name},
            )
            return None

    def _normalize_liquidation_payload(
        self,
        event_payload: Mapping[str, Any] | dict[str, Any],
        *,
        allow_raw_payload: bool,
        source_topic: str | None = None,
    ) -> LiquidationRecord | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_liquidation_payload", _analytics_args)
        except Exception:
            pass
        try:
            event = dict(event_payload)

            if self._is_raw_liquidation_topic(source_topic) and not allow_raw_payload:
                self.logger.warning(
                    "Liquidation payload dropped: raw topic is not allowed in production path",
                    extra={
                        "component": self.component_name,
                        "source_topic": source_topic,
                    },
                )
                return None

            raw_payload = self._extract_liquidation_payload(event)
            if raw_payload is None:
                self.logger.debug(
                    "Liquidation event dropped: cannot extract liquidation payload",
                    extra={
                        "component": self.component_name,
                        "payload_keys": list(event.keys()),
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
            side = self._normalize_side(
                payload.get("side")
                or payload.get("S")
                or payload.get("direction")
            )
            price = self._safe_float(payload.get("price") or payload.get("p"))
            quantity = self._safe_float(
                payload.get("quantity")
                or payload.get("qty")
                or payload.get("q")
                or payload.get("size")
            )
            notional = self._safe_float(payload.get("notional"))
            timestamp_ms = self._extract_timestamp_ms(payload)

            if symbol is None or side == WhaleTradeSide.UNKNOWN.value:
                return None

            if price is None or price <= 0:
                return None

            if quantity is None or quantity <= 0:
                return None

            if notional is None:
                notional = price * quantity

            if notional <= 0:
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
                    "source_topic": source_topic,
                    "payload_source": event.get("source"),
                    "scope": {
                        "exchange": exchange,
                        "market_type": market_type,
                        "symbol": normalized_symbol,
                        "timeframe": timeframe,
                    },
                }
            )

            return LiquidationRecord(
                symbol=normalized_symbol,
                side=side,
                notional=notional,
                price=price,
                quantity=quantity,
                timestamp_ms=timestamp_ms,
                liquidation_id=self._safe_str(
                    payload.get("liquidation_id")
                    or payload.get("id")
                    or payload.get("t")
                ),
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                raw_event=event,
                metadata=metadata,
            )

        except Exception:
            self.logger.exception(
                "Failed to normalize liquidation payload",
                extra={
                    "component": self.component_name,
                    "source_topic": source_topic,
                },
            )
            return None

    @staticmethod
    def _extract_liquidation_payload(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Витягує liquidation payload із data-layer event.

        Підтримує:
        - {"liquidation": {...}}
        - {"data": {...}}
        - {"liquidations": [{...}, ...]} — бере останню liquidation
        - plain liquidation dict
        """
        try:
            _analytics_class_name = "WhaleTracker"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_liquidation_payload", _analytics_args)
        except Exception:
            pass
        liquidation = event.get("liquidation")
        if isinstance(liquidation, Mapping):
            return liquidation

        data = event.get("data")
        if isinstance(data, Mapping):
            return data

        liquidations = event.get("liquidations")
        if isinstance(liquidations, list) and liquidations:
            last_item = liquidations[-1]
            if isinstance(last_item, Mapping):
                return last_item

        if "price" in event or "p" in event:
            return event

        return None

    @staticmethod
    def _is_raw_liquidation_topic(source_topic: str | None) -> bool:
        try:
            _analytics_class_name = "WhaleTracker"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_raw_liquidation_topic", _analytics_args)
        except Exception:
            pass
        return source_topic == "market.liquidation"

    @staticmethod
    def _normalize_symbol(value: Any) -> str | None:
        try:
            _analytics_class_name = "WhaleTracker"
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
            _analytics_class_name = "WhaleTracker"
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
            _analytics_class_name = "WhaleTracker"
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
            _analytics_class_name = "WhaleTracker"
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
    def _safe_str(value: Any) -> str | None:
        try:
            _analytics_class_name = "WhaleTracker"
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
    "WhaleTracker",
]