from __future__ import annotations

from analytics.strategy_contract import ensure_strategy_payload_contract
import asyncio
import json
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Deque
from uuid import uuid4

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler
from analytics.market_state_contract import MarketStateSnapshotSource, _plain

from analytics.funding.config import FundingAnalyzerConfig
from analytics.funding.enums import FundingDataSource, FundingEventType, FundingTimeframe
from analytics.funding.funding_divergence import FundingDivergenceConfig, FundingDivergenceDetector
from analytics.funding.funding_extremes import FundingExtremesConfig, FundingExtremesDetector
from analytics.funding.funding_flip_detector import FundingFlipDetector, FundingFlipDetectorConfig
from analytics.funding.funding_pressure import FundingPressureAnalyzer, FundingPressureConfig
from analytics.funding.funding_regime_detector import FundingRegimeDetector, FundingRegimeDetectorConfig
from analytics.funding.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    FundingAnalyticsEvent,
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingKey,
    FundingPressureState,
    FundingRegimeState,
    FundingSignal,
    FundingSignalType,
    FundingSnapshot,
    FundingStatistics,
    ensure_utc,
    funding_key_to_dict,
    make_funding_key,
    model_to_payload,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)


@dataclass(slots=True)
class FundingMarketContext:
    """
    Локальний context cache для funding divergence / pressure analysis.

    Scope цього context визначається ключем FundingKey:
        exchange + market_type + symbol + timeframe
    """

    latest_open_interest: float | None = None
    previous_open_interest: float | None = None

    latest_price: float | None = None
    previous_price: float | None = None

    latest_cvd: float | None = None
    previous_cvd: float | None = None

    long_liquidations: float | None = None
    short_liquidations: float | None = None

    updated_at: datetime | None = None
    liquidation_updated_at: datetime | None = None


class FundingAnalyzer:
    """
    Event-driven orchestration module для analytics.funding.

    Correct production input flow:
        exchange adapters
            -> market.funding
            -> FundingCache
            -> market.funding.updated
            -> FundingAnalyzer
            -> analytics.funding.*

    Context flow:
        OpenInterestCache -> market.open_interest.updated -> FundingAnalyzer
        CandlesCache -> market.candle.closed -> FundingAnalyzer
        TradesCache -> market.trades.updated -> FundingAnalyzer
        Orderflow/CVD analytics -> analytics.orderflow.updated -> FundingAnalyzer
        Liquidation cache/analytics -> market.liquidations.updated / analytics.liquidations.*
            -> FundingAnalyzer

    Core-вимоги:
    - EventBus/Scheduler через constructor dependency injection;
    - register() для EventBus.subscribe();
    - EventBus.emit() для output analytics events;
    - periodic cleanup/parquet flush тільки через Scheduler.add_interval_job();
    - raw market.* topics заборонені в production, якщо allow_legacy_raw_topics=False;
    - state keyed через FundingKey: exchange + market_type + symbol + timeframe;
    - не створює біржові clients і не читає exchange adapters напряму.
    """

    SOURCE = "analytics.funding.funding_analyzer"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: FundingAnalyzerConfig | None = None,
        regime_detector: FundingRegimeDetector | None = None,
        pressure_analyzer: FundingPressureAnalyzer | None = None,
        flip_detector: FundingFlipDetector | None = None,
        extremes_detector: FundingExtremesDetector | None = None,
        divergence_detector: FundingDivergenceDetector | None = None,
        parquet_storage: Any | None = None,
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
        self.config = config or FundingAnalyzerConfig()
        self.config.validate()
        self.parquet_storage = parquet_storage
        self._market_state_store = market_state_store
        self._state_snapshot_source = (
            MarketStateSnapshotSource(market_state_store) if market_state_store is not None else None
        )

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_analyzer",
        )

        self.regime_detector = regime_detector or FundingRegimeDetector(
            FundingRegimeDetectorConfig(default_timeframe=self.config.default_timeframe)
        )
        self.pressure_analyzer = pressure_analyzer or FundingPressureAnalyzer(
            FundingPressureConfig(default_timeframe=self.config.default_timeframe)
        )
        self.flip_detector = flip_detector or FundingFlipDetector(
            FundingFlipDetectorConfig(default_timeframe=self.config.default_timeframe)
        )
        self.extremes_detector = extremes_detector or FundingExtremesDetector(
            FundingExtremesConfig(default_timeframe=self.config.default_timeframe)
        )
        self.divergence_detector = divergence_detector or FundingDivergenceDetector(
            FundingDivergenceConfig(default_timeframe=self.config.default_timeframe)
        )

        self._history: dict[FundingKey, Deque[FundingSnapshot]] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_per_key)
        )
        self._market_context: dict[FundingKey, FundingMarketContext] = defaultdict(
            FundingMarketContext
        )

        self._latest_statistics: dict[FundingKey, FundingStatistics] = {}
        self._latest_regime_state: dict[FundingKey, FundingRegimeState] = {}
        self._latest_pressure_state: dict[FundingKey, FundingPressureState] = {}
        self._latest_flip_event: dict[FundingKey, FundingFlipEvent] = {}
        self._latest_extreme_event: dict[FundingKey, FundingExtremeEvent] = {}
        self._latest_divergence_event: dict[FundingKey, FundingDivergenceEvent] = {}

        self._locks: dict[FundingKey, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._heartbeat_job_id: str | None = None
        self._parquet_flush_job_id: str | None = None

        self._history_write_buffer: list[dict[str, Any]] = []
        self._history_buffer_lock = asyncio.Lock()
        self._parquet_unavailable_logged = False

        self._registered = False
        self._started = False
        self._last_emit_at: dict[tuple[str, FundingKey], datetime] = {}

    async def process_market_state_snapshot(
        self,
        *,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> Any | None:
        """Evaluate funding analytics from MarketStateStore funding snapshot.

        This is the state-driven input contract replacing market.funding.updated
        as the primary source. Existing EventBus handlers remain as fallback.
        """
        source = getattr(self, "_state_snapshot_source", None)
        if source is None:
            return None
        snapshot = await source.snapshot(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe or getattr(self.config, "default_timeframe", None),
        )
        funding = getattr(snapshot, "funding", None) if snapshot is not None else None
        if funding is None:
            return None
        payload = _plain(funding)
        if not isinstance(payload, dict):
            payload = {"funding": payload}
        payload = {
            **dict(payload),
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe or getattr(self.config, "default_timeframe", None),
            "source_topic": "market_state.snapshot",
        }
        # Reuse private EventBus handler when available to keep domain logic intact.
        for name in ("_on_funding_updated", "_handle_funding_updated", "on_funding_updated"):
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
    # Lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register EventBus subscriptions and Scheduler jobs.

        Sync method: core.EventBus.subscribe() і core.Scheduler.add_interval_job()
        є sync API.
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
            self.logger.warning("FundingAnalyzer already registered")
            return

        if not self.config.enabled:
            self.logger.info("FundingAnalyzer registration skipped: disabled")
            return

        self._subscribe_many(
            self.config.funding_input_topics,
            self.on_funding,
            name="funding_analyzer.on_funding",
        )
        self._subscribe_many(
            self.config.open_interest_event_patterns,
            self.on_open_interest,
            name="funding_analyzer.on_open_interest",
            enabled=self.config.use_open_interest_context,
        )
        self._subscribe_many(
            self.config.candle_event_patterns,
            self.on_candle,
            name="funding_analyzer.on_candle",
            enabled=self.config.use_price_context,
        )
        self._subscribe_many(
            self.config.trade_event_patterns,
            self.on_trade,
            name="funding_analyzer.on_trade",
            enabled=self.config.use_trades_context,
        )
        self._subscribe_many(
            self.config.cvd_event_patterns,
            self.on_cvd_update,
            name="funding_analyzer.on_cvd_update",
            enabled=self.config.use_cvd_context,
        )
        self._subscribe_many(
            self.config.liquidation_event_patterns,
            self.on_liquidation,
            name="funding_analyzer.on_liquidation",
            enabled=self.config.use_liquidation_context,
        )

        if self.config.allow_legacy_raw_topics:
            self._subscribe_legacy_raw_topics()

        self._register_cleanup_job()
        self._register_heartbeat_job()
        self._register_parquet_flush_job()

        self._registered = True

        self.logger.info(
            "FundingAnalyzer registered | subscriptions=%s cleanup_job=%s "
            "heartbeat_job=%s parquet_flush_job=%s",
            len(self._subscriptions),
            self._cleanup_job_id,
            self._heartbeat_job_id,
            self._parquet_flush_job_id,
            extra={
                "production_input_topics": list(self.config.production_input_topics),
                "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
                "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    def unregister(self) -> None:
        """
        Remove EventBus subscriptions and disable Scheduler jobs.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "unregister", _analytics_args)
        except Exception:
            pass
        if not self._registered:
            self.logger.warning("FundingAnalyzer is not registered")
            return

        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe FundingAnalyzer subscription",
                    extra={
                        "pattern": getattr(subscription, "pattern", None),
                        "name": getattr(subscription, "name", None),
                    },
                )
        self._subscriptions.clear()

        self._disable_scheduler_job(self._cleanup_job_id)
        self._disable_scheduler_job(self._heartbeat_job_id)
        self._disable_scheduler_job(self._parquet_flush_job_id)

        self._cleanup_job_id = None
        self._heartbeat_job_id = None
        self._parquet_flush_job_id = None

        self._registered = False

        self.logger.info("FundingAnalyzer unregistered")

    async def start(self) -> None:
        """
        Load historical parquet state, then register EventBus/Scheduler integration.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "start", _analytics_args)
        except Exception:
            pass
        if self._started:
            self.logger.warning("FundingAnalyzer already started")
            return

        if not self.config.enabled:
            self.logger.info("FundingAnalyzer start skipped: disabled")
            return

        if self.config.enable_parquet_history and self.config.load_history_from_parquet_on_start:
            await self.load_history_from_parquet()

        self.register()
        self._started = True

        await self._emit_lifecycle_event(
            self.config.analyzer_started_event_name,
            {
                "service_name": self.config.service_name,
                "production_input_topics": list(self.config.production_input_topics),
                "scope": "exchange:market_type:symbol:timeframe",
            },
        )

    async def stop(self) -> None:
        """
        Flush buffered history and unregister EventBus/Scheduler integration.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "stop", _analytics_args)
        except Exception:
            pass
        if not self._started and not self._registered:
            return

        await self.flush_history_to_parquet()

        if self._registered:
            self.unregister()

        self._started = False

        await self._emit_lifecycle_event(
            self.config.analyzer_stopped_event_name,
            {
                "service_name": self.config.service_name,
                "stats": self.stats(),
            },
        )

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_many(
        self,
        topics: tuple[str, ...],
        handler: Any,
        *,
        name: str,
        enabled: bool = True,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_many", _analytics_args)
        except Exception:
            pass
        if not enabled:
            return

        for topic in topics:
            self.config.assert_topic_allowed(topic, allow_raw=False)
            self._subscriptions.append(
                self.event_bus.subscribe(
                    topic,
                    handler,
                    name=name,
                )
            )

    def _subscribe_legacy_raw_topics(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_subscribe_legacy_raw_topics", _analytics_args)
        except Exception:
            pass
        raw_subscriptions = (
            (
                self.config.raw_funding_event_name,
                self.on_funding,
                "funding_analyzer.on_raw_funding",
            ),
            (
                self.config.raw_open_interest_event_name,
                self.on_open_interest,
                "funding_analyzer.on_raw_open_interest",
            ),
            (
                self.config.raw_candle_event_name,
                self.on_candle,
                "funding_analyzer.on_raw_candle",
            ),
            (
                self.config.raw_trade_event_name,
                self.on_trade,
                "funding_analyzer.on_raw_trade",
            ),
            (
                self.config.raw_liquidation_event_name,
                self.on_liquidation,
                "funding_analyzer.on_raw_liquidation",
            ),
        )

        for topic, handler, name in raw_subscriptions:
            self.config.assert_topic_allowed(topic, allow_raw=True)
            self._subscriptions.append(
                self.event_bus.subscribe(topic, handler, name=name)
            )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_latest_snapshot(
        self,
        symbol: str,
        exchange: str = "unknown",
        *,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_latest_snapshot", _analytics_args)
        except Exception:
            pass
        history = self._history.get(
            self._make_key(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
            )
        )
        if not history:
            return None
        return history[-1]

    def get_statistics(
        self,
        symbol: str,
        exchange: str = "unknown",
        *,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingStatistics | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_statistics", _analytics_args)
        except Exception:
            pass
        return self._latest_statistics.get(
            self._make_key(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
            )
        )

    def get_regime_state(
        self,
        symbol: str,
        exchange: str = "unknown",
        *,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingRegimeState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_regime_state", _analytics_args)
        except Exception:
            pass
        return self._latest_regime_state.get(
            self._make_key(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
            )
        )

    def get_pressure_state(
        self,
        symbol: str,
        exchange: str = "unknown",
        *,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingPressureState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_pressure_state", _analytics_args)
        except Exception:
            pass
        return self._latest_pressure_state.get(
            self._make_key(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
            )
        )

    def get_market_context(
        self,
        symbol: str,
        exchange: str = "unknown",
        *,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingMarketContext | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_market_context", _analytics_args)
        except Exception:
            pass
        return self._market_context.get(
            self._make_key(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
            )
        )

    def get_key_snapshot(self, key: FundingKey) -> FundingSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_key_snapshot", _analytics_args)
        except Exception:
            pass
        history = self._history.get(key)
        if not history:
            return None
        return history[-1]

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
            "started": self._started,
            "registered": self._registered,
            "subscriptions": len(self._subscriptions),
            "cleanup_job_id": self._cleanup_job_id,
            "heartbeat_job_id": self._heartbeat_job_id,
            "parquet_flush_job_id": self._parquet_flush_job_id,
            "parquet_history_enabled": self.config.enable_parquet_history,
            "parquet_buffer_size": len(self._history_write_buffer),
            "parquet_root": str(self._parquet_root()),
            "keys_tracked": len(self._history),
            "contexts_tracked": len(self._market_context),
            "latest_statistics": len(self._latest_statistics),
            "latest_regime_states": len(self._latest_regime_state),
            "latest_pressure_states": len(self._latest_pressure_state),
            "latest_flip_events": len(self._latest_flip_event),
            "latest_extreme_events": len(self._latest_extreme_event),
            "latest_divergence_events": len(self._latest_divergence_event),
            "production_input_topics": list(self.config.production_input_topics),
            "legacy_raw_input_topics": list(self.config.legacy_raw_input_topics),
            "allow_legacy_raw_topics": self.config.allow_legacy_raw_topics,
            "scope": "exchange:market_type:symbol:timeframe",
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

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
        if self._is_legacy_raw_event_blocked(event):
            self.logger.warning(
                "Funding raw topic ignored because legacy raw topics are disabled | topic=%s",
                getattr(event, "topic", None),
            )
            return

        try:
            payload = self._extract_payload(event)
            snapshot = self._parse_funding_snapshot(payload)
        except Exception:
            self.logger.exception("Failed to parse funding event")
            return

        key = snapshot.key

        if not self.config.should_process_key(key):
            return

        if len(self._history) >= self.config.max_tracked_keys and key not in self._history:
            self.logger.warning(
                "FundingAnalyzer max_tracked_keys reached; funding event skipped | key=%s",
                key,
                extra={"scope": funding_key_to_dict(key)},
            )
            return

        lock = self._locks[key]
        lock_acquired = False

        try:
            await asyncio.wait_for(lock.acquire(), timeout=3.0)
            lock_acquired = True
        except asyncio.TimeoutError:
            self.logger.warning(
                "FundingAnalyzer lock timeout | key=%s",
                key,
                extra={"scope": funding_key_to_dict(key)},
            )
            return

        try:
            context = self._market_context[key]
            previous_snapshot = self._history[key][-1] if self._history[key] else None
            previous_regime_state = self._latest_regime_state.get(key)

            self._enrich_snapshot(snapshot, context)
            self._history[key].append(snapshot)

            statistics = self._build_statistics(
                snapshot=snapshot,
                history=self._history[key],
            )

            regime_state = self.regime_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                previous_state=previous_regime_state,
                timeframe=snapshot.timeframe,
            )
            regime_state = self._copy_scope(regime_state, snapshot)

            pressure_state = self.pressure_analyzer.analyze(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                previous_snapshot=previous_snapshot,
                previous_open_interest=context.previous_open_interest,
                current_price=context.latest_price,
                previous_price=context.previous_price,
                timeframe=snapshot.timeframe,
            )
            pressure_state = self._copy_scope(pressure_state, snapshot)

            flip_event = self.flip_detector.detect(
                current_snapshot=snapshot,
                previous_snapshot=previous_snapshot,
                statistics=statistics,
                timeframe=snapshot.timeframe,
            )
            if flip_event is not None:
                flip_event = self._copy_scope(flip_event, snapshot)

            extreme_event = self.extremes_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                timeframe=snapshot.timeframe,
            )
            if extreme_event is not None:
                extreme_event = self._copy_scope(extreme_event, snapshot)

            divergence_event = self.divergence_detector.detect(
                snapshot=snapshot,
                statistics=statistics,
                price_change_pct=self._calc_price_change_pct(
                    context.previous_price,
                    context.latest_price,
                ),
                oi_change_pct=self._calc_change_pct(
                    context.previous_open_interest,
                    context.latest_open_interest,
                ),
                cvd_change=self._calc_delta(
                    context.previous_cvd,
                    context.latest_cvd,
                ),
                long_liquidations=context.long_liquidations,
                short_liquidations=context.short_liquidations,
                timeframe=snapshot.timeframe,
            )
            if divergence_event is not None:
                divergence_event = self._copy_scope(divergence_event, snapshot)

            self._latest_statistics[key] = statistics
            self._latest_regime_state[key] = regime_state
            self._latest_pressure_state[key] = pressure_state

            if flip_event is not None:
                self._latest_flip_event[key] = flip_event
            if extreme_event is not None:
                self._latest_extreme_event[key] = extreme_event
            if divergence_event is not None:
                self._latest_divergence_event[key] = divergence_event

            await self._buffer_history_record(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                context=context,
            )

            correlation_id = getattr(event, "correlation_id", None)

            await self._publish_updated_event(
                snapshot=snapshot,
                statistics=statistics,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                correlation_id=correlation_id,
            )
            await self._publish_snapshot_event(
                snapshot=snapshot,
                correlation_id=correlation_id,
            )
            await self._publish_regime_event(regime_state, correlation_id)
            await self._publish_pressure_event(pressure_state, correlation_id)
            await self._publish_flip_event(flip_event, correlation_id)
            await self._publish_extreme_event(extreme_event, correlation_id)
            await self._publish_divergence_event(divergence_event, correlation_id)
            await self._publish_signal_events(
                snapshot=snapshot,
                regime_state=regime_state,
                pressure_state=pressure_state,
                flip_event=flip_event,
                extreme_event=extreme_event,
                divergence_event=divergence_event,
                correlation_id=correlation_id,
            )

        except Exception:
            self.logger.exception(
                "Failed to process funding event | symbol=%s exchange=%s market_type=%s timeframe=%s",
                snapshot.symbol,
                snapshot.exchange.value,
                snapshot.market_type,
                snapshot.timeframe.value,
                extra={"scope": funding_key_to_dict(snapshot.key)},
            )
        finally:
            if lock_acquired:
                lock.release()

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
        if self._is_legacy_raw_event_blocked(event):
            self.logger.warning(
                "Open interest raw topic ignored because legacy raw topics are disabled | topic=%s",
                getattr(event, "topic", None),
            )
            return

        try:
            payload = self._extract_payload(event)
            key = self._key_from_payload(payload)
            if key is None or not self.config.should_process_key(key):
                return

            new_oi = self._to_optional_float(
                self._first_present(
                    payload,
                    "open_interest",
                    "open_interest_value",
                    "value",
                )
            )
            if new_oi is None:
                return

            context = self._market_context[key]
            context.previous_open_interest = context.latest_open_interest
            context.latest_open_interest = new_oi
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process open interest event")

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
        if self._is_legacy_raw_event_blocked(event):
            self.logger.warning(
                "Candle raw topic ignored because legacy raw topics are disabled | topic=%s",
                getattr(event, "topic", None),
            )
            return

        try:
            payload = self._extract_payload(event)
            key = self._key_from_payload(payload)
            if key is None or not self.config.should_process_key(key):
                return

            price = self._to_optional_float(
                self._first_present(payload, "close", "price")
            )
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process candle event")

    async def on_trade(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_trade", _analytics_args)
        except Exception:
            pass
        if self._is_legacy_raw_event_blocked(event):
            self.logger.warning(
                "Trade raw topic ignored because legacy raw topics are disabled | topic=%s",
                getattr(event, "topic", None),
            )
            return

        try:
            payload = self._extract_payload(event)
            trade_payload = self._extract_nested_payload(payload, "trade", "trades")

            key = self._key_from_payload(trade_payload, fallback_payload=payload)
            if key is None or not self.config.should_process_key(key):
                return

            price = self._to_optional_float(
                self._first_present(trade_payload, "price", "p")
            )
            if price is None:
                return

            context = self._market_context[key]
            context.previous_price = context.latest_price
            context.latest_price = price
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process trade event")

    async def on_cvd_update(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_cvd_update", _analytics_args)
        except Exception:
            pass
        try:
            payload = self._extract_payload(event)
            inner_payload = (
                payload.get("payload")
                if isinstance(payload.get("payload"), dict)
                else payload
            )

            key = self._key_from_payload(inner_payload, fallback_payload=payload)
            if key is None or not self.config.should_process_key(key):
                return

            cvd_value = self._to_optional_float(
                self._first_present(
                    inner_payload,
                    "cvd",
                    "cumulative_volume_delta",
                )
            )
            if cvd_value is None:
                return

            context = self._market_context[key]
            context.previous_cvd = context.latest_cvd
            context.latest_cvd = cvd_value
            context.updated_at = self._utc_now()
        except Exception:
            self.logger.exception("Failed to process CVD update event")

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
        if self._is_legacy_raw_event_blocked(event):
            self.logger.warning(
                "Liquidation raw topic ignored because legacy raw topics are disabled | topic=%s",
                getattr(event, "topic", None),
            )
            return

        try:
            payload = self._extract_payload(event)
            liquidation_payload = self._extract_nested_payload(
                payload,
                "liquidation",
                "liquidations",
            )

            key = self._key_from_payload(liquidation_payload, fallback_payload=payload)
            if key is None or not self.config.should_process_key(key):
                return

            side = str(
                self._first_present(
                    liquidation_payload,
                    "side",
                    "position_side",
                )
                or ""
            ).lower().strip()

            quantity = self._to_optional_float(
                self._first_present(
                    liquidation_payload,
                    "qty",
                    "quantity",
                    "size",
                )
            )
            price = self._to_optional_float(
                self._first_present(liquidation_payload, "price")
            )
            notional = self._to_optional_float(
                self._first_present(liquidation_payload, "notional")
            )

            liquidation_value = notional
            if liquidation_value is None and quantity is not None and price is not None:
                liquidation_value = quantity * price

            context = self._market_context[key]

            if liquidation_value is not None:
                if side in {"long", "buy"}:
                    context.long_liquidations = liquidation_value
                elif side in {"short", "sell"}:
                    context.short_liquidations = liquidation_value

            if side not in {"long", "short", "buy", "sell"}:
                long_liq = self._to_optional_float(
                    self._first_present(
                        liquidation_payload,
                        "long_liquidations",
                        "long_liquidation_notional",
                    )
                )
                short_liq = self._to_optional_float(
                    self._first_present(
                        liquidation_payload,
                        "short_liquidations",
                        "short_liquidation_notional",
                    )
                )

                if long_liq is not None:
                    context.long_liquidations = long_liq
                if short_liq is not None:
                    context.short_liquidations = short_liq

            context.updated_at = self._utc_now()
            context.liquidation_updated_at = context.updated_at
        except Exception:
            self.logger.exception("Failed to process liquidation event")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_statistics(
        self,
        *,
        snapshot: FundingSnapshot,
        history: Deque[FundingSnapshot],
    ) -> FundingStatistics:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_statistics", _analytics_args)
        except Exception:
            pass
        if not history:
            raise ValueError("history must not be empty")

        window_size = max(1, int(self.config.statistics_window_size))
        window = list(history)[-window_size:]

        rates = [item.funding_rate for item in window]
        current_rate = rates[-1]
        mean_rate = sum(rates) / len(rates)
        median_rate = median(rates)

        if len(rates) > 1:
            variance = sum((value - mean_rate) ** 2 for value in rates) / len(rates)
            std_rate = variance**0.5
        else:
            std_rate = 0.0

        zscore = (current_rate - mean_rate) / std_rate if std_rate > 0 else None
        percentile = self._calc_percentile(rates, current_rate)

        return FundingStatistics(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            current_rate=current_rate,
            mean_rate=mean_rate,
            median_rate=median_rate,
            std_rate=std_rate,
            min_rate=min(rates),
            max_rate=max(rates),
            zscore=zscore,
            percentile=percentile,
            sample_size=len(rates),
            window_start=window[0].event_time,
            window_end=window[-1].event_time,
        )

    def _enrich_snapshot(
        self,
        snapshot: FundingSnapshot,
        context: FundingMarketContext,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_enrich_snapshot", _analytics_args)
        except Exception:
            pass
        if snapshot.open_interest is None:
            snapshot.open_interest = context.latest_open_interest
        if snapshot.mark_price is None:
            snapshot.mark_price = context.latest_price

    @staticmethod
    def _copy_scope(model: Any, snapshot: FundingSnapshot) -> Any:
        """
        Force detector result scope to match the canonical FundingSnapshot scope.

        Detector-и можуть бути pure-компонентами й іноді повертати модель зі
        старим або неповним scope. FundingAnalyzer є orchestrator-ом, тому саме
        він нормалізує final analytics model перед cache/publish.
        """
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_copy_scope", _analytics_args)
        except Exception:
            pass
        for attr, value in (
                ("exchange", snapshot.exchange),
                ("market_type", snapshot.market_type),
                ("symbol", snapshot.symbol),
                ("timeframe", snapshot.timeframe),
                ("exchange_symbol", snapshot.exchange_symbol),
        ):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, value)
                except Exception:
                    pass

        metadata = getattr(model, "metadata", None)
        if isinstance(metadata, dict):
            metadata["scope"] = funding_key_to_dict(snapshot.key)
            metadata["exchange_symbol"] = snapshot.exchange_symbol

        return model

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    async def _publish_updated_event(
            self,
            *,
            snapshot: FundingSnapshot,
            statistics: FundingStatistics,
            regime_state: FundingRegimeState,
            pressure_state: FundingPressureState,
            flip_event: FundingFlipEvent | None,
            extreme_event: FundingExtremeEvent | None,
            divergence_event: FundingDivergenceEvent | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_updated_event", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_analytics_events:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.SNAPSHOT,
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            payload={
                "symbol": snapshot.symbol,
                "exchange": snapshot.exchange.value,
                "market_type": snapshot.market_type,
                "timeframe": snapshot.timeframe.value,
                "exchange_symbol": snapshot.exchange_symbol,
                "scope": funding_key_to_dict(snapshot.key),
                "snapshot": snapshot.to_dict(),
                "statistics": statistics.to_dict(),
                "regime_state": regime_state.to_dict(),
                "pressure_state": pressure_state.to_dict(),
                "flip_event": flip_event.to_dict() if flip_event is not None else None,
                "extreme_event": extreme_event.to_dict() if extreme_event is not None else None,
                "divergence_event": divergence_event.to_dict() if divergence_event is not None else None,
            },
            event_time=snapshot.event_time,
            source=self.SOURCE,
        )

        await self._emit_analytics_event(
            topic=self.config.analytics_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
            key=snapshot.key,
        )

    async def _publish_snapshot_event(
        self,
        *,
        snapshot: FundingSnapshot,
        correlation_id: str | None,
    ) -> None:
        """
        Окремий analytics.funding.snapshot event.

        analytics.funding.updated є aggregate event, але config має окремий
        snapshot_event_name + emit_snapshots, тому snapshot має виходити окремо.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_snapshot_event", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_snapshots:
            return

        event = FundingAnalyticsEvent(
            event_type=FundingEventType.SNAPSHOT,
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=snapshot.timeframe,
            exchange_symbol=snapshot.exchange_symbol,
            payload=model_to_payload(snapshot),
            event_time=snapshot.event_time,
            source=self.SOURCE,
        )

        await self._emit_analytics_event(
            topic=self.config.snapshot_event_name,
            payload=event.to_dict(),
            correlation_id=correlation_id,
            key=snapshot.key,
        )

    async def _publish_regime_event(
        self,
        regime_state: FundingRegimeState,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_regime_event", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_regime_events:
            return
        if not regime_state.changed:
            return

        await self._publish_model_event(
            topic=self.config.regime_event_name,
            event_type=FundingEventType.REGIME,
            model=regime_state,
            correlation_id=correlation_id,
        )

    async def _publish_pressure_event(
        self,
        pressure_state: FundingPressureState,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_pressure_event", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_pressure_events:
            return

        should_emit = self.pressure_analyzer.is_high_pressure(pressure_state)
        if not should_emit:
            return

        await self._publish_model_event(
            topic=self.config.pressure_event_name,
            event_type=FundingEventType.PRESSURE,
            model=pressure_state,
            correlation_id=correlation_id,
        )

    async def _publish_flip_event(
        self,
        flip_event: FundingFlipEvent | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_flip_event", _analytics_args)
        except Exception:
            pass
        if flip_event is None or not self.config.emit_flip_events:
            return

        await self._publish_model_event(
            topic=self.config.flip_event_name,
            event_type=FundingEventType.FLIP,
            model=flip_event,
            correlation_id=correlation_id,
        )

    async def _publish_extreme_event(
        self,
        extreme_event: FundingExtremeEvent | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_extreme_event", _analytics_args)
        except Exception:
            pass
        if extreme_event is None or not self.config.emit_extreme_events:
            return

        await self._publish_model_event(
            topic=self.config.extreme_event_name,
            event_type=FundingEventType.EXTREME,
            model=extreme_event,
            correlation_id=correlation_id,
        )

    async def _publish_divergence_event(
        self,
        divergence_event: FundingDivergenceEvent | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_divergence_event", _analytics_args)
        except Exception:
            pass
        if divergence_event is None or not self.config.emit_divergence_events:
            return

        await self._publish_model_event(
            topic=self.config.divergence_event_name,
            event_type=FundingEventType.DIVERGENCE,
            model=divergence_event,
            correlation_id=correlation_id,
        )

    async def _publish_model_event(
        self,
        *,
        topic: str,
        event_type: FundingEventType,
        model: Any,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_model_event", _analytics_args)
        except Exception:
            pass
        event = FundingAnalyticsEvent(
            event_type=event_type,
            symbol=model.symbol,
            exchange=model.exchange,
            market_type=model.market_type,
            timeframe=model.timeframe,
            exchange_symbol=model.exchange_symbol,
            payload=model_to_payload(model),
            event_time=model.event_time,
            source=self.SOURCE,
        )

        await self._emit_analytics_event(
            topic=topic,
            payload=event.to_dict(),
            correlation_id=correlation_id,
            key=model.key,
        )

    async def _publish_signal_events(
        self,
        *,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_signal_events", _analytics_args)
        except Exception:
            pass
        if not self.config.emit_signals:
            return

        signals = self._build_signals(
            snapshot=snapshot,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
        )

        for signal in signals:
            if signal.confidence < self.config.signal_min_confidence:
                continue

            if self._should_skip_emit(
                event_name=self.config.signal_event_name,
                key=signal.key,
            ):
                continue

            event = FundingAnalyticsEvent(
                event_type=FundingEventType.SIGNAL,
                symbol=signal.symbol,
                exchange=signal.exchange,
                market_type=signal.market_type,
                timeframe=signal.timeframe,
                exchange_symbol=signal.exchange_symbol,
                payload=signal.to_dict(),
                event_time=signal.event_time,
                source=self.SOURCE,
            )
            await self._emit_analytics_event(
                topic=self.config.signal_event_name,
                payload=event.to_dict(),
                correlation_id=correlation_id,
                priority=EventPriority.HIGH,
                key=signal.key,
            )

    async def _emit_analytics_event(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        correlation_id: str | None,
        priority: EventPriority = EventPriority.NORMAL,
        key: FundingKey | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_analytics_event", _analytics_args)
        except Exception:
            pass
        strategy_payload = ensure_strategy_payload_contract(
            payload,
            topic=topic,
            source=self.SOURCE,
            domain="funding",
        )
        await self.event_bus.emit(
            topic,
            strategy_payload,
            priority=priority,
            source=self.SOURCE,
            correlation_id=correlation_id,
            headers={
                "scope": str(funding_key_to_dict(key)) if key is not None else None,
            },
        )

    async def _emit_lifecycle_event(
        self,
        topic: str,
        payload: dict[str, Any],
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_emit_lifecycle_event", _analytics_args)
        except Exception:
            pass
        try:
            await self.event_bus.emit(
                topic,
                payload,
                priority=EventPriority.LOW,
                source=self.SOURCE,
            )
        except Exception:
            self.logger.exception("Failed to emit FundingAnalyzer lifecycle event")

    # ------------------------------------------------------------------
    # Signal builders
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        *,
        snapshot: FundingSnapshot,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
    ) -> list[FundingSignal]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signals", _analytics_args)
        except Exception:
            pass
        signals: list[FundingSignal] = []

        if self.config.signal_on_regime_change and regime_state.changed:
            previous_regime = (
                regime_state.previous_regime.value
                if regime_state.previous_regime
                else "unknown"
            )
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    market_type=snapshot.market_type,
                    timeframe=snapshot.timeframe,
                    exchange_symbol=snapshot.exchange_symbol,
                    signal_type=FundingSignalType.REGIME_CHANGE,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._regime_signal_score(regime_state),
                    confidence=regime_state.confidence,
                    description=(
                        f"Funding regime changed from {previous_regime} "
                        f"to {regime_state.regime.value}"
                    ),
                    supporting_factors=[
                        f"funding_rate={snapshot.funding_rate:.8f}",
                        (
                            f"percentile={regime_state.percentile:.2f}"
                            if regime_state.percentile is not None
                            else "percentile=None"
                        ),
                        (
                            f"zscore={regime_state.zscore:.4f}"
                            if regime_state.zscore is not None
                            else "zscore=None"
                        ),
                    ],
                    tags=["funding", "regime"],
                    event_time=snapshot.event_time,
                    metadata=self._merge_signal_metadata(
                        origin="regime",
                        snapshot=snapshot,
                    ),
                )
            )

        if self.config.signal_on_high_pressure and self.pressure_analyzer.is_high_pressure(pressure_state):
            signal_type = (
                FundingSignalType.SQUEEZE_WARNING
                if self.pressure_analyzer.is_squeeze_risk(pressure_state)
                else FundingSignalType.CROWDING_WARNING
            )

            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    market_type=snapshot.market_type,
                    timeframe=snapshot.timeframe,
                    exchange_symbol=snapshot.exchange_symbol,
                    signal_type=signal_type,
                    bias=pressure_state.bias,
                    regime=regime_state.regime,
                    score=self._pressure_signal_score(pressure_state),
                    confidence=max(
                        pressure_state.squeeze_probability or 0.0,
                        pressure_state.mean_reversion_probability or 0.0,
                        pressure_state.pressure_score,
                    ),
                    description=self.pressure_analyzer.build_summary(pressure_state),
                    supporting_factors=[
                        f"pressure_score={pressure_state.pressure_score:.4f}",
                        (
                            f"squeeze_probability={pressure_state.squeeze_probability:.4f}"
                            if pressure_state.squeeze_probability is not None
                            else "squeeze_probability=None"
                        ),
                        (
                            f"mean_reversion_probability={pressure_state.mean_reversion_probability:.4f}"
                            if pressure_state.mean_reversion_probability is not None
                            else "mean_reversion_probability=None"
                        ),
                    ],
                    tags=["funding", "pressure", pressure_state.level.value],
                    event_time=snapshot.event_time,
                    metadata=self._merge_signal_metadata(
                        origin="pressure",
                        snapshot=snapshot,
                        extra={
                            "pressure_level": pressure_state.level.value,
                            "pressure_direction": pressure_state.direction.value,
                        },
                    ),
                )
            )

            if (
                pressure_state.mean_reversion_probability is not None
                and pressure_state.mean_reversion_probability >= self.config.signal_min_confidence
            ):
                signals.append(
                    FundingSignal(
                        symbol=snapshot.symbol,
                        exchange=snapshot.exchange,
                        market_type=snapshot.market_type,
                        timeframe=snapshot.timeframe,
                        exchange_symbol=snapshot.exchange_symbol,
                        signal_type=FundingSignalType.REVERSION_SETUP,
                        bias=pressure_state.bias,
                        regime=regime_state.regime,
                        score=-self._pressure_signal_score(pressure_state),
                        confidence=pressure_state.mean_reversion_probability,
                        description=(
                            "Funding mean-reversion setup detected from pressure state | "
                            f"probability={pressure_state.mean_reversion_probability:.4f}"
                        ),
                        supporting_factors=[
                            f"pressure_score={pressure_state.pressure_score:.4f}",
                            f"mean_reversion_probability={pressure_state.mean_reversion_probability:.4f}",
                            f"price_stall_confirmation={pressure_state.price_stall_confirmation}",
                            f"oi_confirmation={pressure_state.oi_confirmation}",
                        ],
                        tags=["funding", "pressure", "mean_reversion"],
                        event_time=snapshot.event_time,
                        metadata=self._merge_signal_metadata(
                            origin="pressure_reversion",
                            snapshot=snapshot,
                        ),
                    )
                )

        if self.config.signal_on_flip and flip_event is not None:
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    market_type=snapshot.market_type,
                    timeframe=snapshot.timeframe,
                    exchange_symbol=snapshot.exchange_symbol,
                    signal_type=FundingSignalType.FLIP_DETECTED,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._flip_signal_score(flip_event),
                    confidence=flip_event.confidence,
                    description=self.flip_detector.build_summary(flip_event),
                    supporting_factors=[
                        f"previous_rate={flip_event.previous_rate:.8f}",
                        f"current_rate={flip_event.current_rate:.8f}",
                        f"flip_magnitude={flip_event.flip_magnitude:.8f}",
                    ],
                    tags=["funding", "flip", flip_event.flip_type.value],
                    event_time=snapshot.event_time,
                    metadata=self._merge_signal_metadata(
                        origin="flip",
                        snapshot=snapshot,
                        extra={"flip_type": flip_event.flip_type.value},
                    ),
                )
            )

        if self.config.signal_on_extreme and extreme_event is not None:
            base_extreme_factors = [
                f"extreme_type={extreme_event.extreme_type.value}",
                f"severity={extreme_event.severity:.4f}",
                f"reversal_risk={extreme_event.is_reversal_risk}",
                f"squeeze_risk={extreme_event.is_squeeze_risk}",
            ]

            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    market_type=snapshot.market_type,
                    timeframe=snapshot.timeframe,
                    exchange_symbol=snapshot.exchange_symbol,
                    signal_type=FundingSignalType.EXTREME_DETECTED,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._extreme_signal_score(extreme_event),
                    confidence=extreme_event.severity,
                    description=self.extremes_detector.build_summary(extreme_event),
                    supporting_factors=base_extreme_factors,
                    tags=["funding", "extreme", extreme_event.extreme_type.value],
                    event_time=snapshot.event_time,
                    metadata=self._merge_signal_metadata(
                        origin="extreme",
                        snapshot=snapshot,
                        extra={"extreme_type": extreme_event.extreme_type.value},
                    ),
                )
            )

            if extreme_event.is_squeeze_risk:
                signals.append(
                    FundingSignal(
                        symbol=snapshot.symbol,
                        exchange=snapshot.exchange,
                        market_type=snapshot.market_type,
                        timeframe=snapshot.timeframe,
                        exchange_symbol=snapshot.exchange_symbol,
                        signal_type=FundingSignalType.SQUEEZE_WARNING,
                        bias=regime_state.bias,
                        regime=regime_state.regime,
                        score=self._extreme_signal_score(extreme_event),
                        confidence=extreme_event.severity,
                        description=(
                            "Funding squeeze risk from extreme funding state | "
                            f"{self.extremes_detector.build_summary(extreme_event)}"
                        ),
                        supporting_factors=base_extreme_factors,
                        tags=["funding", "extreme", "squeeze"],
                        event_time=snapshot.event_time,
                        metadata=self._merge_signal_metadata(
                            origin="extreme_squeeze",
                            snapshot=snapshot,
                            extra={"extreme_type": extreme_event.extreme_type.value},
                        ),
                    )
                )

            if extreme_event.is_reversal_risk:
                signals.append(
                    FundingSignal(
                        symbol=snapshot.symbol,
                        exchange=snapshot.exchange,
                        market_type=snapshot.market_type,
                        timeframe=snapshot.timeframe,
                        exchange_symbol=snapshot.exchange_symbol,
                        signal_type=FundingSignalType.REVERSION_SETUP,
                        bias=regime_state.bias,
                        regime=regime_state.regime,
                        score=-self._extreme_signal_score(extreme_event),
                        confidence=extreme_event.severity,
                        description=(
                            "Funding reversion setup from extreme funding state | "
                            f"{self.extremes_detector.build_summary(extreme_event)}"
                        ),
                        supporting_factors=base_extreme_factors,
                        tags=["funding", "extreme", "reversion"],
                        event_time=snapshot.event_time,
                        metadata=self._merge_signal_metadata(
                            origin="extreme_reversion",
                            snapshot=snapshot,
                            extra={"extreme_type": extreme_event.extreme_type.value},
                        ),
                    )
                )

        if self.config.signal_on_divergence and divergence_event is not None:
            signals.append(
                FundingSignal(
                    symbol=snapshot.symbol,
                    exchange=snapshot.exchange,
                    market_type=snapshot.market_type,
                    timeframe=snapshot.timeframe,
                    exchange_symbol=snapshot.exchange_symbol,
                    signal_type=FundingSignalType.DIVERGENCE_DETECTED,
                    bias=regime_state.bias,
                    regime=regime_state.regime,
                    score=self._divergence_signal_score(divergence_event),
                    confidence=divergence_event.confidence,
                    description=self.divergence_detector.build_summary(divergence_event),
                    supporting_factors=[
                        f"type={divergence_event.divergence_type.value}",
                        (
                            f"price_change_pct={divergence_event.price_change_pct}"
                            if divergence_event.price_change_pct is not None
                            else "price_change_pct=None"
                        ),
                        (
                            f"oi_change_pct={divergence_event.oi_change_pct}"
                            if divergence_event.oi_change_pct is not None
                            else "oi_change_pct=None"
                        ),
                        (
                            f"cvd_change={divergence_event.cvd_change}"
                            if divergence_event.cvd_change is not None
                            else "cvd_change=None"
                        ),
                    ],
                    tags=["funding", "divergence", divergence_event.divergence_type.value],
                    event_time=snapshot.event_time,
                    metadata=self._merge_signal_metadata(
                        origin="divergence",
                        snapshot=snapshot,
                        extra={
                            "divergence_type": divergence_event.divergence_type.value,
                        },
                    ),
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Signal score helpers
    # ------------------------------------------------------------------

    def _regime_signal_score(self, regime_state: FundingRegimeState) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_regime_signal_score", _analytics_args)
        except Exception:
            pass
        if regime_state.current_rate > 0:
            return -regime_state.confidence
        if regime_state.current_rate < 0:
            return regime_state.confidence
        return 0.0

    def _pressure_signal_score(self, pressure_state: FundingPressureState) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_pressure_signal_score", _analytics_args)
        except Exception:
            pass
        if pressure_state.direction.value == "long":
            return -pressure_state.pressure_score
        if pressure_state.direction.value == "short":
            return pressure_state.pressure_score
        return 0.0

    def _flip_signal_score(self, flip_event: FundingFlipEvent) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_flip_signal_score", _analytics_args)
        except Exception:
            pass
        if flip_event.flip_type.value == "negative_to_positive":
            return -flip_event.confidence
        if flip_event.flip_type.value == "positive_to_negative":
            return flip_event.confidence
        return 0.0

    def _extreme_signal_score(self, extreme_event: FundingExtremeEvent) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extreme_signal_score", _analytics_args)
        except Exception:
            pass
        if extreme_event.funding_rate > 0:
            return -extreme_event.severity
        if extreme_event.funding_rate < 0:
            return extreme_event.severity
        return 0.0

    def _divergence_signal_score(self, divergence_event: FundingDivergenceEvent) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_divergence_signal_score", _analytics_args)
        except Exception:
            pass
        if self.divergence_detector.is_bullish_divergence(divergence_event):
            return divergence_event.confidence
        if self.divergence_detector.is_bearish_divergence(divergence_event):
            return -divergence_event.confidence
        return 0.0

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _register_cleanup_job(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_cleanup_job", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None:
            self.logger.info("FundingAnalyzer cleanup job disabled: scheduler not provided")
            return

        existing_job = self.scheduler.get_job_by_name(self.config.cleanup_job_name)
        if existing_job is not None:
            self._cleanup_job_id = existing_job.job_id
            return

        self._cleanup_job_id = self.scheduler.add_interval_job(
            name=self.config.cleanup_job_name,
            func=self.cleanup_stale_state,
            interval=self.config.cleanup_interval_sec,
            timeout=min(30.0, max(1.0, self.config.cleanup_interval_sec)),
            max_retries=1,
            retry_delay=1.0,
            allow_overlap=False,
            run_immediately=False,
            enabled=True,
        )

    def _register_heartbeat_job(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_heartbeat_job", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None:
            self.logger.info("FundingAnalyzer heartbeat job disabled: scheduler not provided")
            return

        existing_job = self.scheduler.get_job_by_name(self.config.heartbeat_job_name)
        if existing_job is not None:
            self._heartbeat_job_id = existing_job.job_id
            return

        self._heartbeat_job_id = self.scheduler.add_interval_job(
            name=self.config.heartbeat_job_name,
            func=self.emit_heartbeat,
            interval=self.config.heartbeat_interval_sec,
            timeout=5.0,
            max_retries=0,
            retry_delay=0.0,
            allow_overlap=False,
            run_immediately=False,
            enabled=True,
        )

    def _register_parquet_flush_job(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_parquet_flush_job", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_parquet_history:
            return

        if self.scheduler is None:
            self.logger.info("FundingAnalyzer parquet flush job disabled: scheduler not provided")
            return

        existing_job = self.scheduler.get_job_by_name(self.config.parquet_flush_job_name)
        if existing_job is not None:
            self._parquet_flush_job_id = existing_job.job_id
            return

        self._parquet_flush_job_id = self.scheduler.add_interval_job(
            name=self.config.parquet_flush_job_name,
            func=self.flush_history_to_parquet,
            interval=self.config.parquet_flush_interval_sec,
            timeout=self.config.parquet_flush_timeout_sec,
            max_retries=1,
            retry_delay=1.0,
            allow_overlap=False,
            run_immediately=False,
            enabled=True,
        )

    def _disable_scheduler_job(self, job_id: str | None) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_disable_scheduler_job", _analytics_args)
        except Exception:
            pass
        if self.scheduler is None or job_id is None:
            return

        try:
            self.scheduler.disable_job(job_id)
        except KeyError:
            self.logger.warning("Scheduler job not found during unregister | job_id=%s", job_id)

    async def emit_heartbeat(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "emit_heartbeat", _analytics_args)
        except Exception:
            pass
        await self._emit_lifecycle_event(
            self.config.analyzer_heartbeat_event_name,
            {
                "service_name": self.config.service_name,
                "stats": self.stats(),
            },
        )

    async def cleanup_stale_state(self) -> None:
        """
        Scheduler-managed cleanup for stale market context and liquidation context.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup_stale_state", _analytics_args)
        except Exception:
            pass
        now = self._utc_now()
        removed_contexts = 0
        cleared_liquidations = 0

        for key, context in list(self._market_context.items()):
            if context.updated_at is not None:
                age = (now - context.updated_at).total_seconds()
                if age >= self.config.stale_state_ttl_sec and key not in self._history:
                    self._market_context.pop(key, None)
                    self._locks.pop(key, None)
                    removed_contexts += 1
                    continue

            if context.liquidation_updated_at is not None:
                liq_age = (now - context.liquidation_updated_at).total_seconds()
                if liq_age >= self.config.stale_state_ttl_sec:
                    context.long_liquidations = None
                    context.short_liquidations = None
                    context.liquidation_updated_at = None
                    cleared_liquidations += 1

        if removed_contexts or cleared_liquidations:
            self.logger.info(
                "FundingAnalyzer cleanup completed | removed_contexts=%s cleared_liquidations=%s",
                removed_contexts,
                cleared_liquidations,
            )

    # ------------------------------------------------------------------
    # Parquet-backed analytics history
    # ------------------------------------------------------------------

    async def get_history(
        self,
        *,
        symbol: str,
        exchange: str = "unknown",
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
        limit: int = 100,
        include_parquet: bool = True,
    ) -> list[FundingSnapshot]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_history", _analytics_args)
        except Exception:
            pass
        if limit <= 0:
            return []

        key = self._make_key(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
        )
        in_memory = list(self._history.get(key, []))[-limit:]

        if len(in_memory) >= limit or not include_parquet or not self.config.enable_parquet_history:
            return in_memory[-limit:]

        records = await self.get_historical_records(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            limit=limit,
        )
        snapshots = [self._history_row_to_snapshot(row) for row in records]
        snapshots = [snapshot for snapshot in snapshots if snapshot is not None]

        merged: dict[str, FundingSnapshot] = {
            snapshot.event_time.isoformat(): snapshot
            for snapshot in snapshots + in_memory
        }
        return sorted(merged.values(), key=lambda item: item.event_time)[-limit:]

    async def get_historical_records(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_historical_records", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_parquet_history:
            return []

        return await asyncio.to_thread(
            self._read_history_rows_from_parquet,
            symbol,
            exchange,
            market_type,
            timeframe.value if isinstance(timeframe, FundingTimeframe) else timeframe,
            ensure_utc(since) if since is not None else None,
            ensure_utc(until) if until is not None else None,
            limit,
        )

    async def load_history_from_parquet(
        self,
        *,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
    ) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "load_history_from_parquet", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_parquet_history:
            return 0

        records = await self.get_historical_records(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            limit=None,
        )

        loaded = 0
        per_key_loaded: dict[FundingKey, int] = defaultdict(int)

        for record in records:
            snapshot = self._history_row_to_snapshot(record)
            if snapshot is None:
                continue

            key = snapshot.key
            if per_key_loaded[key] >= self.config.parquet_max_load_records_per_key:
                continue

            self._history[key].append(snapshot)
            per_key_loaded[key] += 1
            loaded += 1

        if loaded:
            self.logger.info(
                "FundingAnalyzer history loaded from parquet | records=%s keys=%s",
                loaded,
                len(per_key_loaded),
            )
        return loaded

    async def flush_history_to_parquet(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "flush_history_to_parquet", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_parquet_history:
            return 0

        async with self._history_buffer_lock:
            if not self._history_write_buffer:
                return 0
            rows = list(self._history_write_buffer)
            self._history_write_buffer.clear()

        try:
            written = await asyncio.to_thread(self._write_history_rows_to_parquet, rows)
            if written:
                self.logger.debug("FundingAnalyzer parquet history flushed | records=%s", written)
            return written
        except Exception:
            async with self._history_buffer_lock:
                self._history_write_buffer[0:0] = rows
            self.logger.exception("Failed to flush FundingAnalyzer history to parquet")
            return 0

    async def _buffer_history_record(
        self,
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        context: FundingMarketContext,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_buffer_history_record", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_parquet_history:
            return

        row = self._build_history_row(
            snapshot=snapshot,
            statistics=statistics,
            regime_state=regime_state,
            pressure_state=pressure_state,
            flip_event=flip_event,
            extreme_event=extreme_event,
            divergence_event=divergence_event,
            context=context,
        )

        should_flush = False
        async with self._history_buffer_lock:
            self._history_write_buffer.append(row)
            should_flush = (
                len(self._history_write_buffer) >= self.config.parquet_flush_batch_size
            )

        if should_flush:
            await self.flush_history_to_parquet()

    def _build_history_row(
        self,
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        pressure_state: FundingPressureState,
        flip_event: FundingFlipEvent | None,
        extreme_event: FundingExtremeEvent | None,
        divergence_event: FundingDivergenceEvent | None,
        context: FundingMarketContext,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_history_row", _analytics_args)
        except Exception:
            pass
        snapshot_dict = snapshot.to_dict()
        statistics_dict = statistics.to_dict()
        regime_dict = regime_state.to_dict()
        pressure_dict = pressure_state.to_dict()

        return {
            "event_kind": "funding_analysis",
            "event_time": snapshot_dict.get("event_time"),
            "received_at": snapshot_dict.get("received_at"),
            "exchange": snapshot.exchange.value,
            "market_type": snapshot.market_type,
            "symbol": snapshot.symbol,
            "timeframe": snapshot.timeframe.value,
            "exchange_symbol": snapshot.exchange_symbol,
            "funding_rate": snapshot.funding_rate,
            "predicted_funding_rate": snapshot.predicted_funding_rate,
            "mark_price": snapshot.mark_price,
            "index_price": snapshot.index_price,
            "basis": snapshot.basis,
            "funding_sign": snapshot.funding_sign,
            "open_interest": snapshot.open_interest,
            "volume_24h": snapshot.volume_24h,
            "next_funding_time": snapshot_dict.get("next_funding_time"),
            "latest_open_interest": context.latest_open_interest,
            "previous_open_interest": context.previous_open_interest,
            "latest_price": context.latest_price,
            "previous_price": context.previous_price,
            "latest_cvd": context.latest_cvd,
            "previous_cvd": context.previous_cvd,
            "long_liquidations": context.long_liquidations,
            "short_liquidations": context.short_liquidations,
            "statistics_json": self._json_dumps(statistics_dict),
            "regime_json": self._json_dumps(regime_dict),
            "pressure_json": self._json_dumps(pressure_dict),
            "flip_json": self._json_dumps(flip_event.to_dict() if flip_event is not None else None),
            "extreme_json": self._json_dumps(extreme_event.to_dict() if extreme_event is not None else None),
            "divergence_json": self._json_dumps(divergence_event.to_dict() if divergence_event is not None else None),
            "metadata_json": self._json_dumps(snapshot.metadata),
            "created_at": self._utc_now().isoformat(),
        }

    def _write_history_rows_to_parquet(self, rows: list[dict[str, Any]]) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_write_history_rows_to_parquet", _analytics_args)
        except Exception:
            pass
        if not rows:
            return 0

        external_writer = (
            getattr(self.parquet_storage, "append_records", None)
            or getattr(self.parquet_storage, "write_records", None)
        )
        if external_writer is not None:
            external_writer(dataset=self.config.parquet_dataset_name, records=rows)
            return len(rows)

        pd = self._import_pandas_for_parquet()
        if pd is None:
            return 0

        root = self._parquet_root()
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

        for row in rows:
            event_time = str(row.get("event_time") or self._utc_now().isoformat())
            event_date = event_time[:10]
            grouped[
                (
                    row["exchange"],
                    row["market_type"],
                    row["symbol"],
                    row["timeframe"],
                    event_date,
                )
            ].append(row)

        written = 0
        for (exchange, market_type, symbol, timeframe, event_date), group_rows in grouped.items():
            output_dir = (
                root
                / "snapshots"
                / f"exchange={exchange}"
                / f"market_type={market_type}"
                / f"symbol={symbol}"
                / f"timeframe={timeframe}"
                / f"date={event_date}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"part-{int(self._utc_now().timestamp() * 1000)}-{uuid4().hex}.parquet"
            pd.DataFrame(group_rows).to_parquet(output_file, index=False)
            written += len(group_rows)

        return written

    def _read_history_rows_from_parquet(
        self,
        symbol: str | None,
        exchange: str | None,
        market_type: str | None,
        timeframe: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_read_history_rows_from_parquet", _analytics_args)
        except Exception:
            pass
        external_reader = getattr(self.parquet_storage, "read_records", None)
        if external_reader is not None:
            rows = external_reader(
                dataset=self.config.parquet_dataset_name,
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                since=since,
                until=until,
                limit=limit,
            )
            return list(rows or [])

        pd = self._import_pandas_for_parquet()
        if pd is None:
            return []

        root = self._parquet_root() / "snapshots"
        if not root.exists():
            return []

        files = list(root.rglob("*.parquet"))

        if exchange is not None:
            exchange_part = f"exchange={normalize_exchange(exchange).value}"
            files = [path for path in files if exchange_part in path.parts]
        if market_type is not None:
            market_type_part = f"market_type={normalize_market_type(market_type)}"
            files = [path for path in files if market_type_part in path.parts]
        if symbol is not None:
            symbol_part = f"symbol={normalize_symbol(symbol)}"
            files = [path for path in files if symbol_part in path.parts]
        if timeframe is not None:
            timeframe_part = f"timeframe={normalize_timeframe(timeframe).value}"
            files = [path for path in files if timeframe_part in path.parts]

        frames = []
        for file_path in files:
            try:
                frames.append(pd.read_parquet(file_path))
            except Exception:
                self.logger.exception("Failed to read funding parquet file | path=%s", file_path)

        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)

        if "event_time" in df.columns:
            df["_event_dt"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
            if since is not None:
                df = df[df["_event_dt"] >= since]
            if until is not None:
                df = df[df["_event_dt"] <= until]
            df = df.sort_values("_event_dt")
            df = df.drop(columns=["_event_dt"])

        if limit is not None and limit > 0:
            df = df.tail(limit)

        return df.to_dict(orient="records")

    def _history_row_to_snapshot(self, row: dict[str, Any]) -> FundingSnapshot | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_history_row_to_snapshot", _analytics_args)
        except Exception:
            pass
        try:
            metadata = self._json_loads(row.get("metadata_json")) or {}

            symbol = normalize_symbol(row["symbol"])
            exchange = normalize_exchange(row.get("exchange", "unknown"))
            market_type = normalize_market_type(row.get("market_type") or DEFAULT_MARKET_TYPE)
            timeframe = normalize_timeframe(row.get("timeframe") or DEFAULT_TIMEFRAME)
            exchange_symbol = normalize_exchange_symbol(
                row.get("exchange_symbol"),
                fallback_symbol=symbol,
            )

            return FundingSnapshot(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                exchange_symbol=exchange_symbol,
                funding_rate=float(row.get("funding_rate", 0.0)),
                predicted_funding_rate=self._to_optional_float(row.get("predicted_funding_rate")),
                mark_price=self._to_optional_float(row.get("mark_price")),
                index_price=self._to_optional_float(row.get("index_price")),
                open_interest=self._to_optional_float(row.get("open_interest")),
                volume_24h=self._to_optional_float(row.get("volume_24h")),
                next_funding_time=(
                    self._parse_datetime(row["next_funding_time"])
                    if row.get("next_funding_time")
                    else None
                ),
                event_time=(
                    self._parse_datetime(row.get("event_time"))
                    if row.get("event_time")
                    else self._utc_now()
                ),
                received_at=(
                    self._parse_datetime(row.get("received_at"))
                    if row.get("received_at")
                    else self._utc_now()
                ),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        except Exception:
            self.logger.exception("Failed to restore FundingSnapshot from parquet row")
            return None

    def _parquet_root(self) -> Path:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parquet_root", _analytics_args)
        except Exception:
            pass
        return (
            Path(self.config.parquet_base_path).expanduser()
            / self.config.parquet_dataset_name
        )

    def _import_pandas_for_parquet(self):
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_import_pandas_for_parquet", _analytics_args)
        except Exception:
            pass
        try:
            import pandas as pd  # type: ignore

            return pd
        except Exception:
            if not self._parquet_unavailable_logged:
                self.logger.warning(
                    "Parquet history is enabled but pandas/pyarrow/fastparquet is unavailable; "
                    "install pandas with pyarrow or inject parquet_storage"
                )
                self._parquet_unavailable_logged = True
            return None

    # ------------------------------------------------------------------
    # Parsing / key helpers
    # ------------------------------------------------------------------

    def _parse_funding_snapshot(self, payload: dict[str, Any]) -> FundingSnapshot:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parse_funding_snapshot", _analytics_args)
        except Exception:
            pass
        symbol = normalize_symbol(payload["symbol"])
        exchange = normalize_exchange(payload.get("exchange", "unknown"))
        market_type = normalize_market_type(
            self._first_present(
                payload,
                "market_type",
                "category",
                "market",
            )
            or self.config.default_market_type
        )
        timeframe = normalize_timeframe(
            self._first_present(payload, "timeframe")
            or self.config.default_timeframe
        )
        exchange_symbol = normalize_exchange_symbol(
            self._first_present(
                payload,
                "exchange_symbol",
                "raw_symbol",
                "instrument",
                "s",
            ),
            fallback_symbol=symbol,
        )

        next_funding_time_raw = self._first_present(
            payload,
            "next_funding_time",
            "next_funding_time_ms",
            "next_funding_timestamp",
        )
        next_funding_time = (
            self._parse_datetime(next_funding_time_raw)
            if next_funding_time_raw is not None
            else None
        )

        event_time_raw = self._first_present(
            payload,
            "event_time",
            "timestamp_ms",
            "timestamp",
            "ts",
            "funding_time",
        )
        received_at_raw = self._first_present(
            payload,
            "received_at",
            "received_at_ms",
        )

        event_time = (
            self._parse_datetime(event_time_raw)
            if event_time_raw is not None
            else self._utc_now()
        )
        received_at = (
            self._parse_datetime(received_at_raw)
            if received_at_raw is not None
            else self._utc_now()
        )

        raw_metadata = payload.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        metadata.setdefault(
            "scope",
            funding_key_to_dict(
                make_funding_key(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            ),
        )
        metadata.setdefault("exchange_symbol", exchange_symbol)

        funding_rate_raw = self._first_present(payload, "funding_rate", "rate")
        if funding_rate_raw is None:
            raise ValueError("funding_rate is required")

        return FundingSnapshot(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
            funding_rate=float(funding_rate_raw),
            predicted_funding_rate=self._to_optional_float(
                self._first_present(
                    payload,
                    "predicted_funding_rate",
                    "predicted_rate",
                )
            ),
            mark_price=self._to_optional_float(
                self._first_present(payload, "mark_price")
            ),
            index_price=self._to_optional_float(
                self._first_present(payload, "index_price")
            ),
            open_interest=self._to_optional_float(
                self._first_present(
                    payload,
                    "open_interest",
                    "open_interest_value",
                )
            ),
            volume_24h=self._to_optional_float(
                self._first_present(payload, "volume_24h")
            ),
            next_funding_time=next_funding_time,
            event_time=event_time,
            received_at=received_at,
            metadata=metadata,
        )

    def _key_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_payload: Mapping[str, Any] | None = None,
    ) -> FundingKey | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_key_from_payload", _analytics_args)
        except Exception:
            pass
        fallback_payload = fallback_payload or {}

        symbol = (
            self._first_present(payload, "symbol", "s")
            or self._first_present(fallback_payload, "symbol", "s")
        )
        if not symbol:
            return None

        exchange = (
            self._first_present(payload, "exchange")
            or self._first_present(fallback_payload, "exchange")
            or "unknown"
        )
        market_type = (
            self._first_present(payload, "market_type")
            or self._first_present(fallback_payload, "market_type")
            or self.config.default_market_type
        )
        timeframe = (
            self._first_present(payload, "timeframe")
            or self._first_present(fallback_payload, "timeframe")
            or self.config.default_timeframe
        )

        try:
            return self._make_key(
                symbol=str(symbol),
                exchange=str(exchange),
                market_type=str(market_type),
                timeframe=timeframe,
            )
        except Exception:
            return None

    def _make_key(
        self,
        *,
        symbol: str,
        exchange: str | FundingDataSource,
        market_type: str | None = None,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_make_key", _analytics_args)
        except Exception:
            pass
        return make_funding_key(
            exchange=exchange,
            market_type=market_type or self.config.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.config.default_timeframe,
        )

    @staticmethod
    def _extract_nested_payload(
        payload: dict[str, Any],
        singular_key: str,
        plural_key: str,
    ) -> dict[str, Any]:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_extract_nested_payload", _analytics_args)
        except Exception:
            pass
        singular = payload.get(singular_key)
        if isinstance(singular, dict):
            return singular

        plural = payload.get(plural_key)
        if isinstance(plural, list) and plural:
            last_item = plural[-1]
            if isinstance(last_item, dict):
                return last_item

        data = payload.get("data")
        if isinstance(data, dict):
            return data

        return payload

    @staticmethod
    def _extract_payload(event: Event) -> dict[str, Any]:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
        payload = event.payload
        if not isinstance(payload, dict):
            raise TypeError(f"Event payload must be dict, got: {type(payload)!r}")
        return payload

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _is_legacy_raw_event_blocked(self, event: Event) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_legacy_raw_event_blocked", _analytics_args)
        except Exception:
            pass
        topic = getattr(event, "topic", None)

        if topic is None:
            return False

        return (
            topic in self.config.legacy_raw_input_topics
            and not self.config.allow_legacy_raw_topics
        )

    @staticmethod
    def _first_present(
        payload: Mapping[str, Any],
        *keys: str,
    ) -> Any:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_first_present", _analytics_args)
        except Exception:
            pass
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    @staticmethod
    def _merge_signal_metadata(
        *,
        origin: str,
        snapshot: FundingSnapshot,
        base: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_merge_signal_metadata", _analytics_args)
        except Exception:
            pass
        metadata = dict(base or {})
        metadata.update(
            {
                "signal_origin": origin,
                "scope": funding_key_to_dict(snapshot.key),
                "exchange_symbol": snapshot.exchange_symbol,
            }
        )

        if extra:
            metadata.update(extra)

        return metadata

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_parse_datetime", _analytics_args)
        except Exception:
            pass
        if isinstance(value, datetime):
            return ensure_utc(value)

        if isinstance(value, (int, float)):
            if value > 1_000_000_000_000:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            normalized = value.strip().replace("Z", "+00:00")
            return ensure_utc(datetime.fromisoformat(normalized))

        raise TypeError(f"Unsupported datetime value: {value!r}")

    @staticmethod
    def _utc_now() -> datetime:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_utc_now", _analytics_args)
        except Exception:
            pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_to_optional_float", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _calc_percentile(values: list[float], current_value: float) -> float | None:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_percentile", _analytics_args)
        except Exception:
            pass
        if not values:
            return None

        sorted_values = sorted(values)
        count = len(sorted_values)
        less_or_equal = sum(1 for value in sorted_values if value <= current_value)
        return max(0.0, min(100.0, (less_or_equal / count) * 100.0))

    @staticmethod
    def _calc_change_pct(
        previous: float | None,
        current: float | None,
    ) -> float | None:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_change_pct", _analytics_args)
        except Exception:
            pass
        if previous is None or current is None or previous == 0:
            return None
        return (current - previous) / previous

    def _calc_price_change_pct(
        self,
        previous_price: float | None,
        current_price: float | None,
    ) -> float | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_price_change_pct", _analytics_args)
        except Exception:
            pass
        return self._calc_change_pct(previous_price, current_price)

    @staticmethod
    def _calc_delta(
        previous: float | None,
        current: float | None,
    ) -> float | None:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_delta", _analytics_args)
        except Exception:
            pass
        if previous is None or current is None:
            return None
        return current - previous

    @staticmethod
    def _json_dumps(value: Any) -> str | None:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_json_dumps", _analytics_args)
        except Exception:
            pass
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _json_loads(value: Any) -> Any:
        try:
            _analytics_class_name = "FundingAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_json_loads", _analytics_args)
        except Exception:
            pass
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return None

    def _should_skip_emit(
        self,
        *,
        event_name: str,
        key: FundingKey,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_should_skip_emit", _analytics_args)
        except Exception:
            pass
        now = self._utc_now()
        emit_key = (event_name, key)
        last_at = self._last_emit_at.get(emit_key)

        cooldown_sec = self.config.get_signal_cooldown(key)

        if last_at is not None:
            elapsed = (now - last_at).total_seconds()
            if elapsed < cooldown_sec:
                return True

        self._last_emit_at[emit_key] = now
        return False


__all__ = [
    "FundingAnalyzer",
    "FundingMarketContext",
]
