from __future__ import annotations

from analytics.strategy_contract import ensure_strategy_payload_contract
import inspect
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler
from analytics.market_state_contract import MarketStateSnapshotSource, _plain

from .config import OIAnalyzerConfig
from .enums import OIAnomalyType, OIRegime
from .models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    OIAnalysisResult,
    OIFeatures,
    OIKey,
    OIMarketContext,
    OISnapshot,
    OIState,
    make_oi_key,
    oi_key_to_dict,
    oi_key_to_string,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)
from .oi_anomaly_detector import OIAnomalyDetector
from .oi_divergence import OIDivergenceDetector
from .oi_features import OIFeatureBuilder
from .oi_regime_detector import OIRegimeDetector


@dataclass(slots=True)
class OIInstrumentBuffers:
    """
    Rolling history for one futures scope:

        exchange + market_type + symbol + timeframe

    All series are stored in chronological order:
        oldest -> newest.

    Important:
    WebSocket/cache events may arrive out of order. Buffers therefore insert
    values by timestamp instead of blindly appending to the tail.
    """

    oi_values: deque[float]
    oi_timestamps: deque[float]

    price_values: deque[float]
    price_timestamps: deque[float]

    volume_values: deque[float]
    volume_timestamps: deque[float]

    feature_history: deque[OIFeatures]

    def append_oi(self, oi: float, timestamp: float) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "append_oi", _analytics_args)
        except Exception:
            pass
        self._insert_chronological(
            values=self.oi_values,
            timestamps=self.oi_timestamps,
            value=oi,
            timestamp=timestamp,
        )

    def append_price(self, price: float, timestamp: float) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "append_price", _analytics_args)
        except Exception:
            pass
        self._insert_chronological(
            values=self.price_values,
            timestamps=self.price_timestamps,
            value=price,
            timestamp=timestamp,
        )

    def append_volume(self, volume: float, timestamp: float) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "append_volume", _analytics_args)
        except Exception:
            pass
        self._insert_chronological(
            values=self.volume_values,
            timestamps=self.volume_timestamps,
            value=volume,
            timestamp=timestamp,
        )

    @staticmethod
    def _insert_chronological(
        *,
        values: deque[float],
        timestamps: deque[float],
        value: float,
        timestamp: float,
    ) -> None:
        try:
            _analytics_class_name = "OIInstrumentBuffers"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_insert_chronological", _analytics_args)
        except Exception:
            pass
        value_f = float(value)
        timestamp_f = float(timestamp)

        if not math.isfinite(value_f) or not math.isfinite(timestamp_f):
            return

        value_items = list(values)
        timestamp_items = list(timestamps)

        insert_at = len(timestamp_items)
        for idx, existing_ts in enumerate(timestamp_items):
            if timestamp_f < existing_ts:
                insert_at = idx
                break

        value_items.insert(insert_at, value_f)
        timestamp_items.insert(insert_at, timestamp_f)

        maxlen = values.maxlen
        if maxlen is not None:
            while len(value_items) > maxlen:
                value_items.pop(0)
                timestamp_items.pop(0)

        values.clear()
        timestamps.clear()
        values.extend(value_items)
        timestamps.extend(timestamp_items)


@dataclass(slots=True)
class OIAnalyzerRuntimeStats:
    """
    Runtime diagnostics для OIAnalyzer.

    Це lightweight state без EventBus/Scheduler/logger.
    """

    open_interest_events_processed: int = 0
    candle_events_processed: int = 0
    candles_updated_events_processed: int = 0
    trades_events_processed: int = 0
    funding_events_processed: int = 0
    liquidations_events_processed: int = 0
    orderflow_events_processed: int = 0

    analyses_built: int = 0

    emitted_updates: int = 0
    emitted_regime_changes: int = 0
    emitted_divergences: int = 0
    emitted_anomalies: int = 0
    emitted_squeeze_setups: int = 0
    emitted_capitulations: int = 0
    emitted_metrics: int = 0
    emitted_state_cleaned: int = 0

    skipped_by_scope_filter: int = 0
    skipped_invalid_payload: int = 0
    skipped_missing_oi: int = 0
    skipped_missing_context: int = 0

    cleanup_runs: int = 0
    cleanup_removed_states: int = 0

    errors_count: int = 0
    last_error: str | None = None
    last_error_at: float | None = None

    processed_by_topic: dict[str, int] = field(default_factory=dict)

    def record_topic(self, topic: str | None) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "record_topic", _analytics_args)
        except Exception:
            pass
        key = topic or "unknown"
        self.processed_by_topic[key] = self.processed_by_topic.get(key, 0) + 1

    def record_error(self, error: Exception, timestamp: float) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "record_error", _analytics_args)
        except Exception:
            pass
        self.errors_count += 1
        self.last_error = repr(error)
        self.last_error_at = float(timestamp)

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "open_interest_events_processed": self.open_interest_events_processed,
            "candle_events_processed": self.candle_events_processed,
            "candles_updated_events_processed": self.candles_updated_events_processed,
            "trades_events_processed": self.trades_events_processed,
            "funding_events_processed": self.funding_events_processed,
            "liquidations_events_processed": self.liquidations_events_processed,
            "orderflow_events_processed": self.orderflow_events_processed,
            "analyses_built": self.analyses_built,
            "emitted_updates": self.emitted_updates,
            "emitted_regime_changes": self.emitted_regime_changes,
            "emitted_divergences": self.emitted_divergences,
            "emitted_anomalies": self.emitted_anomalies,
            "emitted_squeeze_setups": self.emitted_squeeze_setups,
            "emitted_capitulations": self.emitted_capitulations,
            "emitted_metrics": self.emitted_metrics,
            "emitted_state_cleaned": self.emitted_state_cleaned,
            "skipped_by_scope_filter": self.skipped_by_scope_filter,
            "skipped_invalid_payload": self.skipped_invalid_payload,
            "skipped_missing_oi": self.skipped_missing_oi,
            "skipped_missing_context": self.skipped_missing_context,
            "cleanup_runs": self.cleanup_runs,
            "cleanup_removed_states": self.cleanup_removed_states,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "processed_by_topic": dict(self.processed_by_topic),
        }


class OIAnalyzer:
    """
    Event-driven orchestration layer for futures Open Interest analytics.

    Responsibilities:
    - subscribe to normalized data-layer / analytics-layer events;
    - maintain per-scope context/state/history;
    - build OI features;
    - detect regimes, divergences, and anomalies;
    - emit analytics.oi.* events through EventBus;
    - schedule cleanup/metrics jobs through Scheduler.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: OIAnalyzerConfig | None = None,
        feature_builder: OIFeatureBuilder | None = None,
        regime_detector: OIRegimeDetector | None = None,
        divergence_detector: OIDivergenceDetector | None = None,
        anomaly_detector: OIAnomalyDetector | None = None,
        market_state_store: Any | None = None,
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
        self.config = config or OIAnalyzerConfig()
        self.config.validate()
        self.config.assert_production_topics_allowed()
        self._market_state_store = market_state_store
        self._state_snapshot_source = (
            MarketStateSnapshotSource(market_state_store) if market_state_store is not None else None
        )

        self.logger = get_logger(
            __name__,
            service_name=self.config.source_name,
            event_type="analytics_open_interest",
        )

        self.feature_builder = feature_builder or OIFeatureBuilder(self.config)
        self.regime_detector = regime_detector or OIRegimeDetector(self.config)
        self.divergence_detector = divergence_detector or OIDivergenceDetector(self.config)
        self.anomaly_detector = anomaly_detector or OIAnomalyDetector(self.config)

        self._history_size = self.config.windows.history_size

        self._buffers: dict[OIKey, OIInstrumentBuffers] = {}
        self._states: dict[OIKey, OIState] = {}
        self._cooldowns: dict[tuple[str, str, str, str, str], float] = {}
        self._last_context_ts: dict[OIKey, float] = {}

        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._metrics_job_id: str | None = None

        self._registered = False
        self._stats = OIAnalyzerRuntimeStats()

    async def process_market_state_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> Any | None:
        """Evaluate open-interest analytics from MarketStateStore snapshot."""
        source = getattr(self, "_state_snapshot_source", None)
        if source is None:
            return None
        snapshot = await source.snapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or getattr(self.config, "default_timeframe", None),
        )
        oi = getattr(snapshot, "open_interest", None) if snapshot is not None else None
        if oi is None:
            return None
        payload = _plain(oi)
        if not isinstance(payload, dict):
            payload = {"open_interest": payload}
        payload = {
            **dict(payload),
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe or getattr(self.config, "default_timeframe", None),
            "source_topic": "market_state.snapshot",
        }
        for name in ("_on_open_interest_updated", "_handle_open_interest_updated", "on_open_interest_updated"):
            method = getattr(self, name, None)
            if callable(method):
                result = method(payload)
                if hasattr(result, "__await__"):
                    result = await result
                return result
        return payload

    async def process_market_snapshot(self, snapshot: Any) -> Any | None:
        """MarketScheduler-compatible evaluator callback."""
        scope = getattr(snapshot, "scope", None)
        return await self.process_market_state_snapshot(
            exchange=getattr(scope, "exchange", None) or getattr(snapshot, "exchange", None) or getattr(self.config, "default_exchange", "binance"),
            market_type=getattr(scope, "market_type", None) or getattr(snapshot, "market_type", None) or getattr(self.config, "default_market_type", "usdm_futures"),
            symbol=getattr(scope, "symbol", None) or getattr(snapshot, "symbol", None),
            timeframe=getattr(scope, "timeframe", None) or getattr(snapshot, "timeframe", None) or getattr(self.config, "default_timeframe", None),
        )

    # ------------------------------------------------------------------
    # Lifecycle / registration
    # ------------------------------------------------------------------

    def register(self) -> None:
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
            self.logger.warning("OIAnalyzer already registered")
            return

        if not self.config.enabled:
            self.logger.info("OIAnalyzer registration skipped: disabled by config")
            return

        self.config.assert_production_topics_allowed()

        self._register_event_subscriptions()
        self._register_scheduler_jobs()

        self._registered = True
        self.logger.info(
            "OIAnalyzer registered",
            extra={
                "subscriptions": len(self._subscriptions),
                "scheduler_enabled": self.scheduler is not None,
                "input_topics": list(self.config.production_input_topics),
                "output_topics": list(self.config.output_topics),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    def unregister(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "unregister", _analytics_args)
        except Exception:
            pass
        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except Exception as exc:
                self.logger.warning(
                    "Failed to unsubscribe OIAnalyzer subscription",
                    extra={
                        "subscription": repr(subscription),
                        "error": repr(exc),
                    },
                )

        self._subscriptions.clear()
        self._unregister_scheduler_jobs()

        self._registered = False
        self.logger.info("OIAnalyzer unregistered")

    def _register_event_subscriptions(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_event_subscriptions", _analytics_args)
        except Exception:
            pass
        self._subscribe_topics(
            self.config.open_interest_topics,
            self.on_open_interest,
            name_prefix="oi_analyzer.on_open_interest_updated",
        )
        self._subscribe_topics(
            self.config.candle_topics,
            self.on_candle,
            name_prefix="oi_analyzer.on_candle_closed",
        )
        self._subscribe_topics(
            self.config.candles_updated_topics,
            self.on_candles_updated,
            name_prefix="oi_analyzer.on_candles_updated",
        )
        self._subscribe_topics(
            self.config.trades_topics,
            self.on_trades_updated,
            name_prefix="oi_analyzer.on_trades_updated",
        )
        self._subscribe_topics(
            self.config.funding_topics,
            self.on_funding,
            name_prefix="oi_analyzer.on_funding_updated",
        )
        self._subscribe_topics(
            self.config.liquidations_topics,
            self.on_liquidation,
            name_prefix="oi_analyzer.on_liquidations_updated",
        )
        self._subscribe_topics(
            self.config.orderflow_topics,
            self.on_orderflow_update,
            name_prefix="oi_analyzer.on_orderflow_updated",
        )

    def _subscribe_topics(
        self,
        topics: tuple[str, ...],
        handler: Any,
        *,
        name_prefix: str,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_topics", _analytics_args)
        except Exception:
            pass
        for topic in topics:
            self.config.assert_input_topic_allowed(topic)
            self._subscriptions.append(
                self.event_bus.subscribe(
                    topic,
                    handler,
                    name=f"{name_prefix}:{topic}",
                )
            )

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
            if (
                self.config.maintenance.enable_periodic_cleanup
                or self.config.maintenance.enable_metrics_emit
            ):
                self.logger.warning(
                    "OIAnalyzer maintenance jobs are enabled but scheduler is not provided"
                )
            return

        maintenance = self.config.maintenance

        if maintenance.enable_periodic_cleanup:
            self._cleanup_job_id = self._add_interval_job_once(
                name=maintenance.cleanup_job_name,
                func=self.cleanup_stale_state,
                interval=maintenance.cleanup_interval_sec,
                timeout=maintenance.cleanup_job_timeout_sec,
            )

        if maintenance.enable_metrics_emit:
            self._metrics_job_id = self._add_interval_job_once(
                name=maintenance.metrics_job_name,
                func=self.emit_metrics,
                interval=maintenance.metrics_interval_sec,
                timeout=maintenance.metrics_job_timeout_sec,
            )

    def _add_interval_job_once(
        self,
        *,
        name: str,
        func: Any,
        interval: float,
        timeout: float | None,
    ) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_add_interval_job_once", _analytics_args)
        except Exception:
            pass
        assert self.scheduler is not None

        existing = self.scheduler.get_job_by_name(name)
        if existing is not None:
            return existing.job_id

        return self.scheduler.add_interval_job(
            name=name,
            func=func,
            interval=interval,
            run_immediately=False,
            timeout=timeout,
            max_retries=self.config.maintenance.scheduler_job_max_retries,
            retry_delay=self.config.maintenance.scheduler_job_retry_delay_sec,
            allow_overlap=False,
        )

    def _unregister_scheduler_jobs(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_unregister_scheduler_jobs", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None:
            self._cleanup_job_id = None
            self._metrics_job_id = None
            return

        for job_id in (self._cleanup_job_id, self._metrics_job_id):
            if job_id is None:
                continue

            try:
                result = self.scheduler.remove_job(job_id)
                if inspect.isawaitable(result):
                    self.logger.warning(
                        "Scheduler.remove_job returned awaitable; "
                        "job may require explicit async cleanup by caller",
                        extra={"job_id": job_id},
                    )
            except KeyError:
                self.logger.debug(
                    "OIAnalyzer scheduler job already removed",
                    extra={"job_id": job_id},
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to remove OIAnalyzer scheduler job",
                    extra={"job_id": job_id, "error": repr(exc)},
                )

        self._cleanup_job_id = None
        self._metrics_job_id = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_open_interest(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_open_interest", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            snapshot = self._parse_open_interest_payload(payload)
            if snapshot is None:
                return

            key = snapshot.key
            if not self._should_process_key(key):
                return

            self._stats.open_interest_events_processed += 1

            buffers = self._get_or_create_buffers(key)
            state = self._get_or_create_state(key)

            buffers.append_oi(snapshot.oi, snapshot.timestamp)
            state.apply_snapshot(snapshot)

            context = self._get_context_for_key(key)
            if self.config.require_price_context and context is None:
                self._stats.skipped_missing_context += 1
                self.logger.debug(
                    "Skipping OI analysis: price context is required but missing",
                    extra=self._key_payload(key),
                )
                return

            analysis_context = context or self._build_empty_context(snapshot)

            features = self.feature_builder.build_from_raw_inputs(
                snapshot=snapshot,
                context=analysis_context,
                oi_values=list(buffers.oi_values),
                oi_timestamps=list(buffers.oi_timestamps),
                price_values=list(buffers.price_values),
                price_timestamps=list(buffers.price_timestamps),
                volume_values=list(buffers.volume_values),
                volume_timestamps=list(buffers.volume_timestamps),
            )

            buffers.feature_history.append(features)
            state.apply_features(features)

            regime_result = self.regime_detector.detect(features)
            divergence_result = self._detect_divergence_if_possible(key)
            anomaly_result = self.anomaly_detector.detect(features)

            analysis_result = OIAnalysisResult(
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                exchange_symbol=snapshot.exchange_symbol,
                timestamp=snapshot.timestamp,
                snapshot=snapshot,
                context=analysis_context,
                features=features,
                regime=regime_result,
                divergence=divergence_result,
                anomaly=anomaly_result,
                metadata={
                    "scope": oi_key_to_dict(key),
                    "scope_key": oi_key_to_string(key),
                    "feature_history_size": len(buffers.feature_history),
                    "oi_history_size": len(buffers.oi_values),
                    "price_history_size": len(buffers.price_values),
                    "volume_history_size": len(buffers.volume_values),
                    "source_event_id": getattr(event, "event_id", None),
                    "source_topic": getattr(event, "topic", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                },
            )

            previous_regime = state.last_regime
            state.apply_analysis(analysis_result)
            self._stats.analyses_built += 1

            await self._emit_analysis_events(
                key=key,
                previous_regime=previous_regime,
                analysis=analysis_result,
                correlation_id=getattr(event, "correlation_id", None),
            )

        except Exception as exc:
            self._record_exception(
                "Failed to process open interest event",
                exc,
                event=event,
            )

    async def on_candle(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_candle", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            applied = await self._apply_candle_payload(payload)
            if applied:
                self._stats.candle_events_processed += 1

        except Exception as exc:
            self._record_exception(
                "Failed to process candle event",
                exc,
                event=event,
            )

    async def on_candles_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_candles_updated", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            candles = payload.get("candles")

            applied_count = 0

            if isinstance(candles, list):
                for candle in candles:
                    if isinstance(candle, Mapping):
                        merged = {
                            **payload,
                            **candle,
                            "exchange": candle.get("exchange", payload.get("exchange")),
                            "market_type": candle.get(
                                "market_type",
                                payload.get("market_type"),
                            ),
                            "symbol": candle.get("symbol", payload.get("symbol")),
                            "timeframe": candle.get(
                                "timeframe",
                                payload.get("timeframe"),
                            ),
                            "exchange_symbol": candle.get(
                                "exchange_symbol",
                                payload.get("exchange_symbol"),
                            ),
                        }
                        if await self._apply_candle_payload(merged):
                            applied_count += 1
            else:
                if await self._apply_candle_payload(payload):
                    applied_count += 1

            if applied_count:
                self._stats.candles_updated_events_processed += applied_count

        except Exception as exc:
            self._record_exception(
                "Failed to process candles updated event",
                exc,
                event=event,
            )

    async def on_trades_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_trades_updated", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            trades = payload.get("trades")

            applied_count = 0

            if isinstance(trades, list):
                for trade in trades:
                    if isinstance(trade, Mapping):
                        merged = {
                            **payload,
                            **trade,
                            "exchange": trade.get("exchange", payload.get("exchange")),
                            "market_type": trade.get(
                                "market_type",
                                payload.get("market_type"),
                            ),
                            "symbol": trade.get("symbol", payload.get("symbol")),
                            "timeframe": trade.get(
                                "timeframe",
                                payload.get("timeframe"),
                            ),
                            "exchange_symbol": trade.get(
                                "exchange_symbol",
                                payload.get("exchange_symbol"),
                            ),
                        }
                        if self._apply_trade_payload(merged):
                            applied_count += 1
            else:
                if self._apply_trade_payload(payload):
                    applied_count += 1

            if applied_count:
                self._stats.trades_events_processed += applied_count

        except Exception as exc:
            self._record_exception(
                "Failed to process trades updated event",
                exc,
                event=event,
            )

    async def on_funding(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_funding", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                self._stats.skipped_invalid_payload += 1
                return

            if not self._should_process_key(key):
                return

            timestamp = self._extract_timestamp(payload)
            funding_rate = self._extract_float(
                payload,
                "funding_rate",
                "funding",
                "rate",
            )
            predicted_rate = self._extract_float(
                payload,
                "predicted_rate",
                "predicted_funding_rate",
                "next_funding_rate",
            )
            next_funding_time_ms = self._extract_float(
                payload,
                "next_funding_time_ms",
                "nextFundingTime",
                "funding_time",
            )

            if funding_rate is None and predicted_rate is None:
                return

            context = self._get_or_create_context(key, timestamp)

            if funding_rate is not None:
                context.funding_rate = funding_rate

            if predicted_rate is not None:
                context.predicted_funding_rate = predicted_rate

            if next_funding_time_ms is not None:
                context.next_funding_time_ms = next_funding_time_ms

            self._advance_context_timestamp(
                key=key,
                context=context,
                timestamp=timestamp,
            )
            context.source = str(payload.get("source") or "funding_cache")

            state = self._get_or_create_state(key)
            state.apply_context(context)

            self._stats.funding_events_processed += 1

        except Exception as exc:
            self._record_exception(
                "Failed to process funding updated event",
                exc,
                event=event,
            )

    async def on_liquidation(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_liquidation", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                self._stats.skipped_invalid_payload += 1
                return

            if not self._should_process_key(key):
                return

            timestamp = self._extract_timestamp(payload)

            long_liq = self._extract_float(
                payload,
                "long_liquidations",
                "long_liq",
                "liquidated_longs",
            )
            short_liq = self._extract_float(
                payload,
                "short_liquidations",
                "short_liq",
                "liquidated_shorts",
            )

            side = self._extract_str(payload, "side", "liquidation_side")
            qty = self._extract_float(payload, "qty", "quantity", "size", "volume")

            context = self._get_or_create_context(key, timestamp)

            if long_liq is not None:
                context.long_liquidations = long_liq

            if short_liq is not None:
                context.short_liquidations = short_liq

            if qty is not None and qty >= 0 and side:
                normalized_side = side.lower().strip()
                if normalized_side in {"long", "buy"}:
                    context.long_liquidations = qty
                elif normalized_side in {"short", "sell"}:
                    context.short_liquidations = qty

            self._advance_context_timestamp(
                key=key,
                context=context,
                timestamp=timestamp,
            )
            context.source = str(payload.get("source") or "liquidations_analytics")

            state = self._get_or_create_state(key)
            state.apply_context(context)

            self._stats.liquidations_events_processed += 1

        except Exception as exc:
            self._record_exception(
                "Failed to process liquidations updated event",
                exc,
                event=event,
            )

    async def on_orderflow_update(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_orderflow_update", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled:
            return

        self._stats.record_topic(getattr(event, "topic", None))

        try:
            payload = self._extract_payload(event)
            key = self._extract_key_from_payload(payload)
            if key is None:
                self._stats.skipped_invalid_payload += 1
                return

            if not self._should_process_key(key):
                return

            timestamp = self._extract_timestamp(payload)
            context = self._get_or_create_context(key, timestamp)

            cvd_delta = self._extract_float(payload, "cvd_delta", "delta", "cvd")
            aggressive_buy_volume = self._extract_float(
                payload,
                "aggressive_buy_volume",
                "buy_volume",
                "aggressive_buys",
            )
            aggressive_sell_volume = self._extract_float(
                payload,
                "aggressive_sell_volume",
                "sell_volume",
                "aggressive_sells",
            )

            if cvd_delta is not None:
                context.cvd_delta = cvd_delta

            if aggressive_buy_volume is not None:
                context.aggressive_buy_volume = aggressive_buy_volume

            if aggressive_sell_volume is not None:
                context.aggressive_sell_volume = aggressive_sell_volume

            self._advance_context_timestamp(
                key=key,
                context=context,
                timestamp=timestamp,
            )
            context.source = str(payload.get("source") or "orderflow_analytics")

            state = self._get_or_create_state(key)
            state.apply_context(context)

            self._stats.orderflow_events_processed += 1

        except Exception as exc:
            self._record_exception(
                "Failed to process orderflow updated event",
                exc,
                event=event,
            )

    # ------------------------------------------------------------------
    # Scheduled maintenance jobs
    # ------------------------------------------------------------------

    async def cleanup_stale_state(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup_stale_state", _analytics_args)
        except Exception:
            pass
        self._stats.cleanup_runs += 1

        now_ts = self._now()
        stale_after = self.config.stale_state_cleanup_after_sec
        keys_to_delete: list[OIKey] = []

        for key, state in list(self._states.items()):
            if state.is_stale(now_ts, stale_after):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            self._states.pop(key, None)
            self._buffers.pop(key, None)
            self._last_context_ts.pop(key, None)

            cooldown_keys = [cd_key for cd_key in self._cooldowns if cd_key[:4] == key]
            for cooldown_key in cooldown_keys:
                self._cooldowns.pop(cooldown_key, None)

        if keys_to_delete:
            self._stats.cleanup_removed_states += len(keys_to_delete)

            self.logger.info(
                "Cleaned stale OI state",
                extra={
                    "removed_count": len(keys_to_delete),
                    "removed_keys": [
                        self._key_payload(key)
                        for key in keys_to_delete
                    ],
                },
            )

            accepted = await self._emit(
                "analytics.oi.state_cleaned",
                {
                    "timestamp": now_ts,
                    "removed_count": len(keys_to_delete),
                    "removed_keys": [
                        self._key_payload(key)
                        for key in keys_to_delete
                    ],
                },
                priority=EventPriority.LOW,
                headers={
                    "event_type": "state_cleaned",
                    "scope": "exchange:market_type:symbol:timeframe",
                },
            )
            if accepted:
                self._stats.emitted_state_cleaned += 1

    async def emit_metrics(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_metrics", _analytics_args)
        except Exception:
            pass
        accepted = await self._emit(
            self.config.metrics_topic,
            self.stats(),
            priority=EventPriority.LOW,
            headers={
                "event_type": "metrics",
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )
        if accepted:
            self._stats.emitted_metrics += 1

    async def emit_health(self) -> None:
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
        await self._emit(
            "analytics.oi.health",
            {
                "timestamp": self._now(),
                "registered": self._registered,
                "enabled": self.config.enabled,
                "states": len(self._states),
                "buffers": len(self._buffers),
                "subscriptions": len(self._subscriptions),
                "scheduler_available": self.scheduler is not None,
                "cleanup_job_id": self._cleanup_job_id,
                "metrics_job_id": self._metrics_job_id,
                "input_topics": list(self.config.production_input_topics),
                "output_topics": list(self.config.output_topics),
                "scope": "exchange:market_type:symbol:timeframe",
                "runtime_stats": self._stats.to_dict(),
            },
            priority=EventPriority.LOW,
            headers={
                "event_type": "health",
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_state(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_state", _analytics_args)
        except Exception:
            pass
        return self._states.get(
            self._normalize_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def get_state_key(self, key: OIKey) -> OIState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_state_key", _analytics_args)
        except Exception:
            pass
        return self._states.get(key)

    def get_last_analysis(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIAnalysisResult | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_last_analysis", _analytics_args)
        except Exception:
            pass
        state = self.get_state(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        if state is None:
            return None
        return state.last_analysis

    def get_last_analysis_key(self, key: OIKey) -> OIAnalysisResult | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_last_analysis_key", _analytics_args)
        except Exception:
            pass
        state = self._states.get(key)
        if state is None:
            return None
        return state.last_analysis

    def get_feature_history(
        self,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> list[OIFeatures]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_feature_history", _analytics_args)
        except Exception:
            pass
        key = self._normalize_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.get_feature_history_key(key)

    def get_feature_history_key(self, key: OIKey) -> list[OIFeatures]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_feature_history_key", _analytics_args)
        except Exception:
            pass
        buffers = self._buffers.get(key)
        if buffers is None:
            return []
        return list(buffers.feature_history)

    def stats(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stats", _analytics_args)
        except Exception:
            pass
        return {
            "timestamp": self._now(),
            "registered": self._registered,
            "enabled": self.config.enabled,
            "states": len(self._states),
            "buffers": len(self._buffers),
            "cooldowns": len(self._cooldowns),
            "subscriptions": len(self._subscriptions),
            "history_size": self._history_size,
            "cleanup_job_registered": self._cleanup_job_id is not None,
            "metrics_job_registered": self._metrics_job_id is not None,
            "input_topics": list(self.config.production_input_topics),
            "output_topics": list(self.config.output_topics),
            "scope": "exchange:market_type:symbol:timeframe",
            "runtime_stats": self._stats.to_dict(),
            "instruments": [
                {
                    **self._key_payload(key),
                    "oi_history_size": len(buffers.oi_values),
                    "price_history_size": len(buffers.price_values),
                    "volume_history_size": len(buffers.volume_values),
                    "feature_history_size": len(buffers.feature_history),
                    "has_state": key in self._states,
                }
                for key, buffers in self._buffers.items()
            ],
        }

    # ------------------------------------------------------------------
    # Analytics flow helpers
    # ------------------------------------------------------------------

    def _detect_divergence_if_possible(
        self,
        key: OIKey,
    ) -> Any | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_divergence_if_possible", _analytics_args)
        except Exception:
            pass
        buffers = self._buffers.get(key)
        if buffers is None or len(buffers.feature_history) < 3:
            return None

        try:
            return self.divergence_detector.detect(list(buffers.feature_history))
        except Exception as exc:
            self.logger.exception(
                "Failed to detect OI divergence",
                extra={**self._key_payload(key), "error": repr(exc)},
            )
            return None

    async def _emit_analysis_events(
        self,
        *,
        key: OIKey,
        previous_regime: OIRegime,
        analysis: OIAnalysisResult,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_analysis_events", _analytics_args)
        except Exception:
            pass
        if self.config.emit_updates:
            await self._emit_oi_updated(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_regime_changes
            and analysis.regime.regime != previous_regime
            and self._cooldown_passed(
                key,
                "regime_change",
                self.config.cooldowns.regime_change_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_regime_changed(
                previous_regime=previous_regime,
                analysis=analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_divergences
            and analysis.divergence is not None
            and analysis.divergence.detected
            and self._cooldown_passed(
                key,
                "divergence",
                self.config.cooldowns.divergence_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_divergence_detected(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_anomalies
            and analysis.anomaly is not None
            and analysis.anomaly.detected
            and self._cooldown_passed(
                key,
                "anomaly",
                self.config.cooldowns.anomaly_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_anomaly_detected(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_squeeze_events
            and analysis.regime.regime == OIRegime.SQUEEZE_SETUP
            and self._cooldown_passed(
                key,
                "squeeze_setup",
                self.config.cooldowns.squeeze_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_squeeze_setup(
                analysis,
                correlation_id=correlation_id,
            )

        if (
            self.config.emit_capitulation_events
            and self._is_capitulation_event(analysis)
            and self._cooldown_passed(
                key,
                "capitulation",
                self.config.cooldowns.capitulation_event_cooldown_sec,
                analysis.timestamp,
            )
        ):
            await self._emit_capitulation_detected(
                analysis,
                correlation_id=correlation_id,
            )

    @staticmethod
    def _is_capitulation_event(analysis: OIAnalysisResult) -> bool:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_capitulation_event", _analytics_args)
        except Exception:
            pass
        if analysis.regime.regime == OIRegime.CAPITULATION:
            return True

        if analysis.anomaly is None or not analysis.anomaly.detected:
            return False

        return analysis.anomaly.anomaly_type in {
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
            OIAnomalyType.OI_COLLAPSE,
        }

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    async def _emit_oi_updated(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_oi_updated", _analytics_args)
        except Exception:
            pass
        accepted = await self._emit(
            self.config.update_topic,
            self._analysis_payload(analysis),
            priority=EventPriority.NORMAL,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="oi_updated",
            ),
        )
        if accepted:
            self._stats.emitted_updates += 1

    async def _emit_regime_changed(
        self,
        *,
        previous_regime: OIRegime,
        analysis: OIAnalysisResult,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_regime_changed", _analytics_args)
        except Exception:
            pass
        accepted = await self._emit(
            self.config.regime_change_topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "previous_regime": previous_regime.value,
                "new_regime": analysis.regime.regime.value,
                "confidence": analysis.regime.confidence,
                "score": analysis.regime.score,
                "reasons": list(analysis.regime.reasons),
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="regime_changed",
            ),
        )
        if accepted:
            self._stats.emitted_regime_changes += 1

    async def _emit_divergence_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_divergence_detected", _analytics_args)
        except Exception:
            pass
        if analysis.divergence is None:
            return

        accepted = await self._emit(
            self.config.divergence_topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "divergence_type": analysis.divergence.divergence_type.value,
                "confidence": analysis.divergence.confidence,
                "score": analysis.divergence.score,
                "window_size": analysis.divergence.window_size,
                "reasons": list(analysis.divergence.reasons),
                "regime": analysis.regime.regime.value,
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="divergence_detected",
            ),
        )
        if accepted:
            self._stats.emitted_divergences += 1

    async def _emit_anomaly_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_anomaly_detected", _analytics_args)
        except Exception:
            pass
        if analysis.anomaly is None:
            return

        accepted = await self._emit(
            self.config.anomaly_topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "anomaly_type": analysis.anomaly.anomaly_type.value,
                "strength": analysis.anomaly.strength.value,
                "confidence": analysis.anomaly.confidence,
                "score": analysis.anomaly.score,
                "reasons": list(analysis.anomaly.reasons),
                "regime": analysis.regime.regime.value,
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="anomaly_detected",
            ),
        )
        if accepted:
            self._stats.emitted_anomalies += 1

    async def _emit_squeeze_setup(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_squeeze_setup", _analytics_args)
        except Exception:
            pass
        accepted = await self._emit(
            self.config.squeeze_setup_topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "regime": analysis.regime.regime.value,
                "confidence": analysis.regime.confidence,
                "score": analysis.regime.score,
                "reasons": list(analysis.regime.reasons),
                "features": analysis.features.to_dict(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="squeeze_setup",
            ),
        )
        if accepted:
            self._stats.emitted_squeeze_setups += 1

    async def _emit_capitulation_detected(
        self,
        analysis: OIAnalysisResult,
        *,
        correlation_id: str | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_capitulation_detected", _analytics_args)
        except Exception:
            pass
        anomaly_type = (
            analysis.anomaly.anomaly_type.value
            if analysis.anomaly is not None and analysis.anomaly.detected
            else None
        )

        accepted = await self._emit(
            self.config.capitulation_topic,
            {
                **self._analysis_scope_payload(analysis),
                "timestamp": analysis.timestamp,
                "regime": analysis.regime.regime.value,
                "regime_confidence": analysis.regime.confidence,
                "anomaly_type": anomaly_type,
                "features": analysis.features.to_dict(),
                "reasons": self._collect_capitulation_reasons(analysis),
            },
            priority=EventPriority.CRITICAL,
            correlation_id=correlation_id,
            headers=self._headers_for_analysis(
                analysis,
                event_type="capitulation_detected",
            ),
        )
        if accepted:
            self._stats.emitted_capitulations += 1

    async def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        headers: dict[str, str] | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit", _analytics_args)
        except Exception:
            pass
        strategy_payload = ensure_strategy_payload_contract(
            payload,
            topic=topic,
            source=self.config.source_name,
            domain="open_interest",
        )
        return await self.event_bus.emit(
            topic,
            strategy_payload,
            priority=priority,
            source=self.config.source_name,
            correlation_id=correlation_id,
            headers=headers or {},
        )

    def _analysis_payload(self, analysis: OIAnalysisResult) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_analysis_payload", _analytics_args)
        except Exception:
            pass
        return analysis.to_dict()

    @staticmethod
    def _analysis_scope_payload(analysis: OIAnalysisResult) -> dict[str, Any]:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_analysis_scope_payload", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": analysis.exchange,
            "market_type": analysis.market_type,
            "symbol": analysis.symbol,
            "exchange_symbol": analysis.exchange_symbol,
            "timeframe": analysis.timeframe,
            "scope": analysis.scope,
            "scope_key": analysis.scope_key,
            "oi_key": analysis.key,
            "key": list(analysis.key),
        }

    @staticmethod
    def _headers_for_analysis(
        analysis: OIAnalysisResult,
        *,
        event_type: str,
    ) -> dict[str, str]:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_headers_for_analysis", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": analysis.exchange,
            "market_type": analysis.market_type,
            "symbol": analysis.symbol,
            "timeframe": analysis.timeframe,
            "exchange_symbol": analysis.exchange_symbol or analysis.symbol,
            "scope": analysis.scope_key,
            "event_type": event_type,
        }

    @staticmethod
    def _collect_capitulation_reasons(analysis: OIAnalysisResult) -> list[str]:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_collect_capitulation_reasons", _analytics_args)
        except Exception:
            pass
        reasons: list[str] = []
        reasons.extend(analysis.regime.reasons)

        if analysis.anomaly is not None:
            reasons.extend(analysis.anomaly.reasons)

        return list(dict.fromkeys(reasons))

    # ------------------------------------------------------------------
    # Context / state / buffers
    # ------------------------------------------------------------------

    def _get_or_create_buffers(
        self,
        key: OIKey,
    ) -> OIInstrumentBuffers:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_or_create_buffers", _analytics_args)
        except Exception:
            pass
        buffers = self._buffers.get(key)
        if buffers is not None:
            return buffers

        buffers = OIInstrumentBuffers(
            oi_values=deque(maxlen=self._history_size),
            oi_timestamps=deque(maxlen=self._history_size),
            price_values=deque(maxlen=self._history_size),
            price_timestamps=deque(maxlen=self._history_size),
            volume_values=deque(maxlen=self._history_size),
            volume_timestamps=deque(maxlen=self._history_size),
            feature_history=deque(maxlen=self._history_size),
        )
        self._buffers[key] = buffers
        return buffers

    def _get_or_create_state(
        self,
        key: OIKey,
    ) -> OIState:
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
        state = self._states.get(key)
        if state is not None:
            return state

        scope = oi_key_to_dict(key)
        state = OIState(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
        )
        self._states[key] = state
        return state

    def _get_or_create_context(
        self,
        key: OIKey,
        timestamp: float,
    ) -> OIMarketContext:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_or_create_context", _analytics_args)
        except Exception:
            pass
        state = self._get_or_create_state(key)

        if state.last_context is None:
            scope = oi_key_to_dict(key)
            state.last_context = OIMarketContext(
                exchange=scope["exchange"],
                market_type=scope["market_type"],
                symbol=scope["symbol"],
                timeframe=scope["timeframe"],
                timestamp=timestamp,
            )

        return state.last_context

    def _get_context_for_key(
        self,
        key: OIKey,
    ) -> OIMarketContext | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_get_context_for_key", _analytics_args)
        except Exception:
            pass
        state = self._states.get(key)
        if state is None or state.last_context is None:
            return None

        if state.last_context.is_stale(
            self._now(),
            self.config.stale_context_after_sec,
        ):
            return None

        return state.last_context

    @staticmethod
    def _build_empty_context(snapshot: OISnapshot) -> OIMarketContext:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_empty_context", _analytics_args)
        except Exception:
            pass
        return OIMarketContext(
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            symbol=snapshot.symbol,
            exchange_symbol=snapshot.exchange_symbol,
            timeframe=snapshot.timeframe,
            timestamp=snapshot.timestamp,
            mark_price=snapshot.mark_price,
            index_price=snapshot.index_price,
            source="empty_context",
        )

    def _cooldown_passed(
        self,
        key: OIKey,
        event_kind: str,
        cooldown_sec: float,
        now_ts: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_cooldown_passed", _analytics_args)
        except Exception:
            pass
        cooldown_key = (*key, event_kind)
        last_ts = self._cooldowns.get(cooldown_key)

        if last_ts is None or (now_ts - last_ts) >= cooldown_sec:
            self._cooldowns[cooldown_key] = now_ts
            return True

        return False

    # ------------------------------------------------------------------
    # Context update helpers
    # ------------------------------------------------------------------

    async def _apply_candle_payload(self, payload: Mapping[str, Any]) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_apply_candle_payload", _analytics_args)
        except Exception:
            pass
        key = self._extract_key_from_payload(payload)
        if key is None:
            self._stats.skipped_invalid_payload += 1
            return False

        if not self._should_process_key(key):
            return False

        timestamp = self._extract_timestamp(payload)
        close_price = self._extract_float(
            payload,
            "close",
            "c",
            "price",
            "last_price",
        )
        mark_price = self._extract_float(payload, "mark_price")
        index_price = self._extract_float(payload, "index_price")

        volume = self._extract_float(payload, "volume", "v", "base_volume")
        quote_volume = self._extract_float(
            payload,
            "quote_volume",
            "quoteVolume",
            "quote_volume_24h",
        )

        buffers = self._get_or_create_buffers(key)
        context = self._get_or_create_context(key, timestamp)

        if close_price is not None:
            buffers.append_price(close_price, timestamp)
            self._refresh_price_context_from_buffers(
                context=context,
                buffers=buffers,
            )

        if volume is not None and volume >= 0:
            buffers.append_volume(volume, timestamp)
            self._refresh_volume_context_from_buffers(
                context=context,
                buffers=buffers,
            )

        if self._is_current_or_new_context_timestamp(key, timestamp):
            if mark_price is not None:
                context.mark_price = mark_price

            if index_price is not None:
                context.index_price = index_price

            if quote_volume is not None and quote_volume >= 0:
                context.quote_volume = quote_volume

            context.source = str(payload.get("source") or "candles_cache")

        self._advance_context_timestamp(
            key=key,
            context=context,
            timestamp=timestamp,
        )

        state = self._get_or_create_state(key)
        state.apply_context(context)
        return True

    def _apply_trade_payload(self, payload: Mapping[str, Any]) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_apply_trade_payload", _analytics_args)
        except Exception:
            pass
        key = self._extract_key_from_payload(payload)
        if key is None:
            self._stats.skipped_invalid_payload += 1
            return False

        if not self._should_process_key(key):
            return False

        timestamp = self._extract_timestamp(payload)
        price = self._extract_float(payload, "price", "p")
        qty = self._extract_float(payload, "qty", "quantity", "q", "size", "volume")
        side = self._extract_str(payload, "side", "taker_side", "aggressor_side")

        buffers = self._get_or_create_buffers(key)
        context = self._get_or_create_context(key, timestamp)

        if price is not None:
            buffers.append_price(price, timestamp)
            self._refresh_price_context_from_buffers(
                context=context,
                buffers=buffers,
            )

        if qty is not None and qty >= 0:
            buffers.append_volume(qty, timestamp)
            self._refresh_volume_context_from_buffers(
                context=context,
                buffers=buffers,
            )

            if self._is_current_or_new_context_timestamp(key, timestamp):
                self._update_aggressive_flow_context(
                    context=context,
                    side=side,
                    qty=qty,
                )

        if self._is_current_or_new_context_timestamp(key, timestamp):
            context.source = str(payload.get("source") or "trades_cache")

        self._advance_context_timestamp(
            key=key,
            context=context,
            timestamp=timestamp,
        )

        state = self._get_or_create_state(key)
        state.apply_context(context)
        return True

    def _refresh_price_context_from_buffers(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_price_context_from_buffers", _analytics_args)
        except Exception:
            pass
        if not buffers.price_values:
            return

        latest_price = float(buffers.price_values[-1])
        previous_price = (
            float(buffers.price_values[-2])
            if len(buffers.price_values) >= 2
            else None
        )

        context.price = latest_price

        if previous_price is not None:
            context.price_delta = latest_price - previous_price
            if abs(previous_price) > 1e-12:
                context.price_delta_pct = (
                    (latest_price - previous_price) / abs(previous_price)
                ) * 100.0

    def _refresh_volume_context_from_buffers(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_volume_context_from_buffers", _analytics_args)
        except Exception:
            pass
        if not buffers.volume_values:
            return

        latest_volume = float(buffers.volume_values[-1])
        context.volume = latest_volume

        volume_ma = self.feature_builder.compute_moving_average(
            list(buffers.volume_values),
            self.config.windows.volume_window,
        )
        context.volume_ma = volume_ma
        context.volume_ratio = self.feature_builder.compute_volume_ratio(
            latest_volume,
            volume_ma,
        )

    def _advance_context_timestamp(
        self,
        *,
        key: OIKey,
        context: OIMarketContext,
        timestamp: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_advance_context_timestamp", _analytics_args)
        except Exception:
            pass
        current_ts = self._last_context_ts.get(key)

        if current_ts is None or timestamp >= current_ts:
            context.timestamp = timestamp
            self._last_context_ts[key] = timestamp
            return

        context.timestamp = current_ts

    def _is_current_or_new_context_timestamp(
        self,
        key: OIKey,
        timestamp: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_current_or_new_context_timestamp", _analytics_args)
        except Exception:
            pass
        current_ts = self._last_context_ts.get(key)
        return current_ts is None or timestamp >= current_ts

    def _update_price_context(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
        price: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_price_context", _analytics_args)
        except Exception:
            pass
        buffers.append_price(price, context.timestamp)
        self._refresh_price_context_from_buffers(
            context=context,
            buffers=buffers,
        )

    def _update_volume_context(
        self,
        *,
        context: OIMarketContext,
        buffers: OIInstrumentBuffers,
        volume: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_volume_context", _analytics_args)
        except Exception:
            pass
        buffers.append_volume(volume, context.timestamp)
        self._refresh_volume_context_from_buffers(
            context=context,
            buffers=buffers,
        )

    @staticmethod
    def _update_aggressive_flow_context(
        *,
        context: OIMarketContext,
        side: str | None,
        qty: float,
    ) -> None:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_update_aggressive_flow_context", _analytics_args)
        except Exception:
            pass
        if not side:
            return

        normalized_side = side.lower().strip()
        if normalized_side in {"buy", "bid", "long"}:
            context.aggressive_buy_volume = qty
        elif normalized_side in {"sell", "ask", "short"}:
            context.aggressive_sell_volume = qty

    # ------------------------------------------------------------------
    # Parsing / normalization
    # ------------------------------------------------------------------

    def _parse_open_interest_payload(
        self,
        payload: Mapping[str, Any],
    ) -> OISnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parse_open_interest_payload", _analytics_args)
        except Exception:
            pass
        key = self._extract_key_from_payload(payload)
        if key is None:
            self._stats.skipped_invalid_payload += 1
            self.logger.warning("OI payload missing exchange/market_type/symbol")
            return None

        if not self._should_process_key(key):
            return None

        timestamp = self._extract_timestamp(payload)
        oi = self._extract_float(
            payload,
            "oi",
            "open_interest",
            "openInterest",
            "value",
        )

        if oi is None:
            self._stats.skipped_missing_oi += 1
            self.logger.warning(
                "OI payload missing OI value",
                extra=self._key_payload(key),
            )
            return None

        scope = oi_key_to_dict(key)

        return OISnapshot(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
            exchange_symbol=self._extract_str(payload, "exchange_symbol"),
            timestamp=timestamp,
            oi=oi,
            open_interest_value=self._extract_float(
                payload,
                "open_interest_value",
                "oi_value",
                "notional_value",
            ),
            mark_price=self._extract_float(payload, "mark_price", "markPrice"),
            index_price=self._extract_float(payload, "index_price", "indexPrice"),
            source=self._extract_str(payload, "source") or "open_interest_cache",
            metadata={
                "scope": scope,
                "scope_key": oi_key_to_string(key),
                "raw_topic_source": payload.get("source"),
                "received_at_ms": payload.get("received_at_ms"),
                "timestamp_ms": payload.get("timestamp_ms"),
            },
        )

    @staticmethod
    def _extract_payload(event: Event | Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_payload", _analytics_args)
        except Exception:
            pass
        payload = event.payload if isinstance(event, Event) else event

        if isinstance(payload, Mapping):
            return payload

        raise ValueError("Event payload must be a mapping-like object")

    def _extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
    ) -> OIKey | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_key_from_payload", _analytics_args)
        except Exception:
            pass
        exchange = self._extract_str(payload, "exchange", "venue", "source_exchange")
        market_type = self._extract_str(
            payload,
            "market_type",
            "category",
            "inst_type",
            "instrument_type",
        )
        symbol = self._extract_str(payload, "symbol", "instrument", "market")
        timeframe = self._extract_str(payload, "timeframe", "tf", "interval")

        if not symbol:
            return None

        return self._normalize_key(
            exchange=exchange or self.config.default_exchange,
            market_type=market_type or self.config.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.config.default_timeframe,
        )

    def _normalize_key(
        self,
        *,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
    ) -> OIKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_key", _analytics_args)
        except Exception:
            pass
        normalized_symbol = (
            normalize_symbol(symbol)
            if self.config.normalize_symbol
            else str(symbol).strip()
        )

        return make_oi_key(
            exchange=normalize_exchange(exchange or self.config.default_exchange),
            market_type=normalize_market_type(market_type or self.config.default_market_type),
            symbol=normalized_symbol,
            timeframe=normalize_timeframe(timeframe or self.config.default_timeframe),
        )

    def _extract_timestamp(self, payload: Mapping[str, Any]) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_timestamp", _analytics_args)
        except Exception:
            pass
        timestamp = self._extract_float(
            payload,
            "timestamp",
            "timestamp_ms",
            "received_at_ms",
            "ts",
            "time",
            "event_time",
            "T",
            "close_time_ms",
            "open_time_ms",
            "last_update_ts_ms",
        )

        if timestamp is None:
            return self._now()

        if timestamp > 10_000_000_000:
            return timestamp / 1000.0

        return timestamp

    @staticmethod
    def _extract_float(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> float | None:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_float", _analytics_args)
        except Exception:
            pass
        for key in keys:
            if key not in payload or payload[key] is None:
                continue

            try:
                value = float(payload[key])
            except (TypeError, ValueError, OverflowError):
                continue

            if math.isfinite(value):
                return value

        return None

    @staticmethod
    def _extract_str(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> str | None:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_str", _analytics_args)
        except Exception:
            pass
        for key in keys:
            if key not in payload or payload[key] is None:
                continue

            value = str(payload[key]).strip()
            if value:
                return value

        return None

    def _should_process_key(self, key: OIKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_should_process_key", _analytics_args)
        except Exception:
            pass
        if self.config.should_process_key(key):
            return True

        self._stats.skipped_by_scope_filter += 1
        self.logger.debug(
            "OI event skipped by scope filter",
            extra=self._key_payload(key),
        )
        return False

    @staticmethod
    def _key_payload(key: OIKey) -> dict[str, Any]:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_key_payload", _analytics_args)
        except Exception:
            pass
        scope = oi_key_to_dict(key)
        return {
            **scope,
            "scope": scope,
            "scope_key": oi_key_to_string(key),
            "oi_key": key,
            "key": list(key),
        }

    def _record_exception(
        self,
        message: str,
        exc: Exception,
        *,
        event: Event | None = None,
        extra: dict[str, Any] | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_record_exception", _analytics_args)
        except Exception:
            pass
        self._stats.record_error(exc, self._now())

        payload: dict[str, Any] = {
            "error": repr(exc),
        }

        if event is not None:
            payload.update(
                {
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                    "source": getattr(event, "source", None),
                    "correlation_id": getattr(event, "correlation_id", None),
                }
            )

        if extra:
            payload.update(extra)

        self.logger.exception(message, extra=payload)

    @staticmethod
    def _now() -> float:
        try:
            _analytics_class_name = "OIAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_now", _analytics_args)
        except Exception:
            pass
        return time.time()


__all__ = [
    "OIInstrumentBuffers",
    "OIAnalyzerRuntimeStats",
    "OIAnalyzer",
]
