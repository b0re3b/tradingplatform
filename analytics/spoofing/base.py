from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.event_bus import EventBus, EventPriority
from core.logger import get_logger
from core.scheduler import Scheduler

from .config import SpoofingConfig
from .enums import SpoofingComponent, SpoofingSide
from .models import (
    DetectorResult,
    OrderbookLevelSnapshot,
    SpoofingFeatures,
    utc_now,
)


class BaseSpoofingModule(ABC):
    """
    Базовий клас для всіх analytics.spoofing модулів.

    Відповідає за:
    - constructor dependency injection для EventBus / Scheduler / Config;
    - централізований logger через core.logger.get_logger;
    - єдиний register() контракт;
    - безпечну публікацію подій через EventBus.emit();
    - спільні pure helper-и для detector/tracker/scoring логіки.

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
        SpoofingAnalyzer перевизначає цей метод і підписується на market.* events.
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
        if isinstance(side, str) and side in SpoofingSide._value2member_map_:
            return SpoofingSide(side)
        return SpoofingSide.UNKNOWN

    def build_level_snapshot(
        self,
        *,
        symbol: str,
        exchange: str,
        side: str | SpoofingSide,
        price: float,
        size: float,
        best_bid: float | None = None,
        best_ask: float | None = None,
        mid_price: float | None = None,
        spread: float | None = None,
        sequence_id: int | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderbookLevelSnapshot:
        """
        Будує нормалізований OrderbookLevelSnapshot із raw payload values.
        """
        return OrderbookLevelSnapshot(
            symbol=symbol,
            exchange=exchange,
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
                payload,
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
        Перетворює dataclass/asdict payload у EventBus/API-friendly структуру.

        datetime -> ISO string
        Enum -> enum.value
        dict/list/tuple -> recursively serialized
        """
        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, dict):
            return {
                str(key): self._serialize_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]

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
    "BaseSpoofingDetector",
    "BaseSpoofingTracker",
]