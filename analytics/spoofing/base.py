from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import SpoofingConfig
from .enums import SpoofingComponent, SpoofingSide
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DetectorResult,
    OrderbookLevelSnapshot,
    SpoofingFeatures,
    SpoofingKey,
    make_spoofing_key,
    scoped_metadata,
    spoofing_key_to_dict,
    utc_now,
)


class BaseSpoofingModule(ABC):
    """
    Базовий клас для всіх analytics.spoofing модулів.

    Відповідає за:
    - constructor dependency injection для EventBus / Scheduler / Config;
    - централізований logger через core.logger.get_logger;
    - єдиний register() контракт;
    - shared scoped multi-exchange futures helpers;
    - безпечну публікацію подій через EventBus.emit();
    - спільні pure helper-и для detector/tracker/scoring логіки.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct input flow:
        exchange adapters
            -> market.orderbook / market.trade
            -> OrderBookCache / TradesCache
            -> market.orderbook.updated / market.trades.updated
            -> analytics.spoofing
            -> analytics.spoofing.*

    Важливо:
    - цей клас не створює EventBus або Scheduler самостійно;
    - не запускає власних asyncio loops;
    - periodic jobs мають реєструватися через Scheduler.add_interval_job()
      у конкретному integration-компоненті, зазвичай SpoofingAnalyzer.
    """

    component: SpoofingComponent = SpoofingComponent.ANALYZER

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config

        self.logger = get_logger(
            __name__,
            service="analytics.spoofing",
            component=self.component.value,
        )

    # -------------------------------------------------------------------------
    # Lifecycle / registration
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Реєструє EventBus subscriptions і/або Scheduler jobs.

        За замовчуванням модуль нічого не реєструє.
        Detector-и зазвичай лишаються чистими evaluator-ами.
        SpoofingAnalyzer / BaseSpoofingAnalyzer перевизначає цей метод.
        """
        return None

    def require_event_bus(self) -> EventBus:
        """
        Повертає EventBus або кидає помилку для компонентів, де він обов'язковий.
        """
        if self.event_bus is None:
            raise RuntimeError(f"{self.__class__.__name__} requires EventBus")
        return self.event_bus

    def require_scheduler(self) -> Scheduler:
        """
        Повертає Scheduler або кидає помилку для компонентів, де він обов'язковий.
        """
        if self.scheduler is None:
            raise RuntimeError(f"{self.__class__.__name__} requires Scheduler")
        return self.scheduler

    # -------------------------------------------------------------------------
    # Time helpers
    # -------------------------------------------------------------------------

    def now(self) -> datetime:
        return utc_now()

    def ensure_utc(self, dt: datetime | None) -> datetime:
        if dt is None:
            return utc_now()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # -------------------------------------------------------------------------
    # Scoped multi-exchange futures helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_exchange(exchange: object) -> str:
        normalized = str(exchange or "").strip().lower()
        if not normalized:
            raise ValueError("exchange must not be empty")
        return normalized

    @staticmethod
    def normalize_market_type(market_type: object = DEFAULT_MARKET_TYPE) -> str:
        normalized = str(market_type or DEFAULT_MARKET_TYPE).strip().lower()
        return normalized if normalized else DEFAULT_MARKET_TYPE

    @staticmethod
    def normalize_symbol(symbol: object) -> str:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        return normalized

    @staticmethod
    def normalize_timeframe(timeframe: object = DEFAULT_TIMEFRAME) -> str:
        normalized = str(timeframe or DEFAULT_TIMEFRAME).strip()
        return normalized if normalized else DEFAULT_TIMEFRAME

    @staticmethod
    def normalize_exchange_symbol(
        exchange_symbol: object,
        *,
        fallback_symbol: str,
    ) -> str:
        normalized = str(exchange_symbol or "").strip()
        return normalized if normalized else fallback_symbol

    def make_key(
        self,
        *,
        exchange: object,
        symbol: object,
        market_type: object = DEFAULT_MARKET_TYPE,
        timeframe: object = DEFAULT_TIMEFRAME,
    ) -> SpoofingKey:
        return make_spoofing_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

    def make_default_key(
        self,
        *,
        symbol: object,
        timeframe: object | None = None,
    ) -> SpoofingKey:
        """
        Backward-compatible helper для symbol-only викликів.

        У multi-exchange futures режимі symbol-only небезпечний, тому
        default_exchange має бути явно заданий у config.
        """
        default_exchange = getattr(self.config, "default_exchange", None) or getattr(
            self.config,
            "exchange",
            None,
        )
        if not default_exchange:
            raise ValueError(
                "make_default_key(symbol) requires config.default_exchange. "
                "Use make_key(exchange=..., market_type=..., symbol=..., timeframe=...) instead."
            )

        return self.make_key(
            exchange=default_exchange,
            market_type=getattr(self.config, "default_market_type", DEFAULT_MARKET_TYPE),
            symbol=symbol,
            timeframe=timeframe or getattr(
                self.config,
                "default_timeframe",
                DEFAULT_TIMEFRAME,
            ),
        )

    def key_to_dict(self, key: SpoofingKey) -> dict[str, str]:
        return spoofing_key_to_dict(key)

    def scope_metadata(
        self,
        *,
        exchange: object,
        symbol: object,
        market_type: object = DEFAULT_MARKET_TYPE,
        timeframe: object = DEFAULT_TIMEFRAME,
        exchange_symbol: object | None = None,
    ) -> dict[str, str]:
        return scoped_metadata(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            exchange_symbol=exchange_symbol,
        )

    def should_process_key(self, key: SpoofingKey) -> bool:
        """
        Перевіряє, чи дозволений scoped futures key поточним config.

        Підтримує новий config.is_key_allowed(), а також legacy
        is_symbol_allowed().
        """
        if hasattr(self.config, "is_key_allowed"):
            return bool(self.config.is_key_allowed(key))

        scope = spoofing_key_to_dict(key)
        if hasattr(self.config, "is_symbol_allowed"):
            return bool(self.config.is_symbol_allowed(scope["symbol"]))

        return True

    def should_process_scope(
        self,
        *,
        exchange: object,
        symbol: object,
        market_type: object = DEFAULT_MARKET_TYPE,
        timeframe: object = DEFAULT_TIMEFRAME,
    ) -> bool:
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        return self.should_process_key(key)

    def extract_key_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        default_exchange: str | None = None,
        default_market_type: str | None = None,
        default_timeframe: str | None = None,
    ) -> SpoofingKey | None:
        """
        Витягує SpoofingKey з EventBus payload.

        Очікуваний production payload походить від data cache:
            market.orderbook.updated / market.trades.updated

        Мінімальні поля:
            exchange, symbol

        Optional:
            market_type, timeframe
        """
        exchange = payload.get("exchange") or default_exchange
        symbol = payload.get("symbol")

        if not exchange or not symbol:
            return None

        market_type = (
            payload.get("market_type")
            or default_market_type
            or getattr(self.config, "default_market_type", DEFAULT_MARKET_TYPE)
        )
        timeframe = (
            payload.get("timeframe")
            or default_timeframe
            or getattr(self.config, "default_timeframe", DEFAULT_TIMEFRAME)
        )

        try:
            return self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        except ValueError:
            return None

    def extract_key_from_event(self, event: Event) -> SpoofingKey | None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            return None

        return self.extract_key_from_payload(payload)

    # -------------------------------------------------------------------------
    # Numeric helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def normalize_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return self.clamp(numerator / denominator, 0.0, 1.0)

    @staticmethod
    def bps_distance(price_a: float, price_b: float) -> float:
        if price_a <= 0 or price_b <= 0:
            return 0.0
        return abs(price_a - price_b) / price_b * 10_000.0

    @staticmethod
    def signed_bps_move(current_price: float, reference_price: float) -> float:
        if current_price <= 0 or reference_price <= 0:
            return 0.0
        return (current_price - reference_price) / reference_price * 10_000.0

    # -------------------------------------------------------------------------
    # Domain builders
    # -------------------------------------------------------------------------

    def parse_spoofing_side(self, side: str | SpoofingSide | None) -> SpoofingSide:
        if isinstance(side, SpoofingSide):
            return side

        if side is None:
            return SpoofingSide.UNKNOWN

        normalized = str(side).strip().lower()

        if normalized in {"bid", "buy", "b", "long"}:
            return SpoofingSide.BID

        if normalized in {"ask", "sell", "s", "short"}:
            return SpoofingSide.ASK

        return SpoofingSide.UNKNOWN

    def build_level_snapshot(
        self,
        *,
        symbol: str,
        exchange: str,
        side: str | SpoofingSide,
        price: float,
        size: float,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange_symbol: str | None = None,
        best_bid: float | None = None,
        best_ask: float | None = None,
        mid_price: float | None = None,
        spread: float | None = None,
        sequence_id: int | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderbookLevelSnapshot:
        """
        Будує normalized OrderbookLevelSnapshot.

        Production analyzer має будувати такі snapshots із OrderBookCache /
        market.orderbook.updated, а не напряму з raw exchange adapter payload.
        """
        normalized_symbol = self.normalize_symbol(symbol)

        return OrderbookLevelSnapshot(
            symbol=normalized_symbol,
            exchange=self.normalize_exchange(exchange),
            market_type=self.normalize_market_type(market_type),
            timeframe=self.normalize_timeframe(timeframe),
            exchange_symbol=self.normalize_exchange_symbol(
                exchange_symbol,
                fallback_symbol=normalized_symbol,
            ),
            side=self.parse_spoofing_side(side),
            price=self.safe_float(price),
            size=self.safe_float(size),
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            sequence_id=sequence_id,
            timestamp=self.ensure_utc(timestamp),
            metadata=metadata or {},
        )

    # -------------------------------------------------------------------------
    # Event publishing
    # -------------------------------------------------------------------------

    async def emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Безпечно публікує подію через core.event_bus.EventBus.emit().

        Якщо EventBus не переданий — подія не публікується, але модуль не падає.
        Це дозволяє тестувати detector-и як чисту доменну логіку.
        """
        if self.event_bus is None:
            self.log_debug(
                "Event emit skipped because event_bus is None",
                topic=topic,
            )
            return False

        try:
            return await self.event_bus.emit(
                topic,
                self._serialize_value(payload),
                priority=priority,
                source=f"analytics.spoofing.{self.component.value}",
                correlation_id=correlation_id,
                headers=headers or {},
            )
        except Exception:
            self.logger.exception(
                "Failed to emit spoofing event | topic=%s",
                topic,
            )
            raise

    async def emit_update(
        self,
        payload: dict[str, Any],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str | None = None,
    ) -> bool:
        return await self.emit_event(
            self.config.analyzer.event_topic_updated,
            payload,
            priority=priority,
            correlation_id=correlation_id,
        )

    async def emit_error(
        self,
        error: Exception,
        *,
        context: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        if not getattr(self.config.analyzer, "publish_errors", True):
            return False

        return await self.emit_event(
            self.config.analyzer.event_topic_error,
            {
                "component": self.component.value,
                "error_type": error.__class__.__name__,
                "error": str(error),
                "context": context or {},
                "payload": payload or {},
                "timestamp": self.now(),
            },
            priority=EventPriority.HIGH,
            correlation_id=correlation_id,
        )

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------

    def serialize_dataclass(self, obj: Any) -> dict[str, Any]:
        if not is_dataclass(obj):
            raise TypeError(f"Object of type {type(obj).__name__} is not a dataclass")

        return self._serialize_value(asdict(obj))

    def feature_payload(self, features: SpoofingFeatures | None) -> dict[str, Any] | None:
        if features is None:
            return None
        return self.serialize_dataclass(features)

    def detector_result_payload(self, result: DetectorResult) -> dict[str, Any]:
        return self.serialize_dataclass(result)

    def _serialize_value(self, value: Any) -> Any:
        """
        Перетворює payload у EventBus/API-friendly структуру.

        datetime -> ISO string
        Enum -> enum.value
        dataclass -> dict
        dict/list/tuple/set -> recursively serialized
        """
        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return {
                key: self._serialize_value(item)
                for key, item in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): self._serialize_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, set):
            return sorted(self._serialize_value(item) for item in value)

        return value

    # -------------------------------------------------------------------------
    # Logging wrappers
    # -------------------------------------------------------------------------

    def log_debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, extra=kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, extra=kwargs)

    def log_exception(self, message: str, **kwargs: Any) -> None:
        self.logger.exception(message, extra=kwargs)


class BaseSpoofingAnalyzer(BaseSpoofingModule, ABC):
    """
    Базовий клас для integration-level spoofing analyzer-ів.

    Відповідає за:
    - EventBus subscription lifecycle;
    - production source topic patterns;
    - scoped SpoofingKey extraction;
    - Scheduler cleanup job lifecycle;
    - key-first processing contract.

    Concrete analyzer має реалізувати:
    - process_key()
    - get_latest_output_by_key()
    - _handle_event()
    - cleanup()
    """

    component = SpoofingComponent.ANALYZER

    def __init__(
        self,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
    ) -> None:
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )
        self._subscriptions: list[Subscription] = []
        self._cleanup_job_id: str | None = None
        self._running = False

    def register(self) -> None:
        """
        Реєструє production subscriptions і Scheduler cleanup job.

        Production topics мають бути data-layer topics:
            market.orderbook.updated
            market.trades.updated

        Raw topics:
            market.orderbook
            market.trade

        не мають використовуватись у production.
        """
        if self._running:
            self.log_warning("%s already registered", analyzer=self.__class__.__name__)
            return

        if not self.config.enabled or not self.config.analyzer.enabled:
            self.log_warning(
                "Spoofing analyzer registration skipped: disabled by config",
                analyzer=self.__class__.__name__,
            )
            return

        event_bus = self.require_event_bus()

        for pattern in self._source_topic_patterns():
            subscription = event_bus.subscribe(
                pattern,
                self._handle_event,
                name=f"{self.__class__.__name__}:{pattern}",
            )
            self._subscriptions.append(subscription)

        self._register_cleanup_job()
        self._running = True

        self.log_info(
            "Spoofing analyzer registered",
            analyzer=self.__class__.__name__,
            source_topic_patterns=list(self._source_topic_patterns()),
            cleanup_job_id=self._cleanup_job_id,
            scope="exchange:market_type:symbol:timeframe",
        )

    def stop(self) -> None:
        if not self._running:
            self.log_warning("%s already stopped", analyzer=self.__class__.__name__)
            return

        if self.event_bus is not None:
            for subscription in list(self._subscriptions):
                try:
                    self.event_bus.unsubscribe(subscription)
                except Exception:
                    self.log_exception(
                        "Failed to unsubscribe spoofing handler",
                        analyzer=self.__class__.__name__,
                        pattern=getattr(subscription, "pattern", None),
                    )

        self._subscriptions.clear()
        self._disable_cleanup_job()
        self._running = False

        self.log_info(
            "Spoofing analyzer stopped",
            analyzer=self.__class__.__name__,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def _source_topic_patterns(self) -> tuple[str, ...]:
        analyzer_config = self.config.analyzer

        patterns = getattr(analyzer_config, "source_topic_patterns", None)
        if patterns:
            return tuple(patterns)

        orderbook_patterns = getattr(
            analyzer_config,
            "source_topic_patterns_orderbook",
            ("market.orderbook.updated",),
        )
        trade_patterns = getattr(
            analyzer_config,
            "source_topic_patterns_trade",
            ("market.trades.updated",),
        )

        return tuple(orderbook_patterns) + tuple(trade_patterns)

    def _register_cleanup_job(self) -> None:
        if self.scheduler is None:
            self.log_warning(
                "Scheduler cleanup registration skipped: scheduler is None",
                analyzer=self.__class__.__name__,
            )
            return

        if not self.config.analyzer.scheduler_cleanup_enabled:
            return

        self._cleanup_job_id = self.scheduler.add_interval_job(
            name=self.config.analyzer.scheduler_cleanup_job_name,
            func=self.cleanup,
            interval=self.config.cleanup_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=0.5,
            timeout=5.0,
            allow_overlap=False,
            enabled=True,
        )

    def _disable_cleanup_job(self) -> None:
        if self.scheduler is None or self._cleanup_job_id is None:
            return

        try:
            self.scheduler.disable_job(self._cleanup_job_id)
        except Exception:
            self.log_exception(
                "Failed to disable spoofing cleanup job",
                cleanup_job_id=self._cleanup_job_id,
            )
        finally:
            self._cleanup_job_id = None

    @abstractmethod
    async def process_key(self, key: SpoofingKey) -> Any:
        """
        Основний production API.

        key:
            exchange + market_type + symbol + timeframe
        """
        raise NotImplementedError

    async def process_market(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> Any:
        key = self.make_key(
            exchange=exchange,
            market_type=market_type or getattr(
                self.config,
                "default_market_type",
                DEFAULT_MARKET_TYPE,
            ),
            symbol=symbol,
            timeframe=timeframe or getattr(
                self.config,
                "default_timeframe",
                DEFAULT_TIMEFRAME,
            ),
        )
        return await self.process_key(key)

    async def process_symbol(self, symbol: str) -> Any:
        """
        Backward-compatible wrapper.

        У multi-exchange futures режимі symbol-only обробка дозволена лише якщо
        config.default_exchange заданий явно.
        """
        key = self.make_default_key(symbol=symbol)
        return await self.process_key(key)

    @abstractmethod
    def get_latest_output_by_key(self, key: SpoofingKey) -> Any:
        """
        Повертає останній analyzer output для одного scoped futures market.
        """
        raise NotImplementedError

    @abstractmethod
    async def _handle_event(self, event: Event) -> None:
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self) -> Any:
        """
        Cleanup job entrypoint для Scheduler.
        """
        raise NotImplementedError


class BaseSpoofingDetector(BaseSpoofingModule, ABC):
    """
    Базовий клас для spoofing detector/scorer компонентів.

    Detector-и мають залишатися чистими evaluator-ами:
    - не підписуються самостійно на EventBus;
    - не запускають Scheduler jobs;
    - приймають доменні моделі;
    - повертають DetectorResult або None.
    """

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> DetectorResult | None:
        """
        Синхронний аналіз доменної моделі.

        Якщо конкретному detector-у колись знадобиться async I/O,
        варто додати окремий async method у конкретному класі, а не ламати
        базовий detector contract.
        """
        raise NotImplementedError


class BaseSpoofingTracker(BaseSpoofingModule, ABC):
    """
    Базовий клас для stateful tracker-компонентів.

    Tracker може мати mutable in-memory state, але не повинен запускати
    власні неконтрольовані loops. Cleanup викликається напряму analyzer-ом
    або через Scheduler.add_interval_job().
    """

    @abstractmethod
    def cleanup(self, now: datetime | None = None) -> int:
        """
        Очищення простроченого стану.

        Повертає кількість видалених/expired елементів.
        """
        raise NotImplementedError


__all__ = [
    "BaseSpoofingModule",
    "BaseSpoofingAnalyzer",
    "BaseSpoofingDetector",
    "BaseSpoofingTracker",
]