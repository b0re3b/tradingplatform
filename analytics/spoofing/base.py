from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger

from .config import SpoofingConfig
from .enums import SpoofingComponent
from .models import DetectorResult, OrderbookLevelSnapshot, SpoofingFeatures, utc_now


class BaseSpoofingModule(ABC):
    """
    Базовий клас для всіх spoofing-модулів.

    Дає:
    - доступ до event_bus
    - доступ до config
    - стандартний logger
    - спільні helper-методи
    """

    component: SpoofingComponent = SpoofingComponent.ANALYZER

    def __init__(
        self,
        event_bus: Any | None,
        config: SpoofingConfig,
    ) -> None:
        self.event_bus = event_bus
        self.config = config
        self.logger = get_logger(__name__, service_name=f"spoofing.{self.component.value}")

    def register(self) -> None:
        """
        За замовчуванням модуль може не реєструвати підписки.
        Analyzer зазвичай перевизначає це явно.
        """
        return None

    def now(self) -> datetime:
        return utc_now()

    def ensure_utc(self, dt: datetime | None) -> datetime:
        if dt is None:
            return utc_now()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def normalize_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return self.clamp(numerator / denominator, 0.0, 1.0)

    def bps_distance(self, price_a: float, price_b: float) -> float:
        if price_a <= 0 or price_b <= 0:
            return 0.0
        return abs(price_a - price_b) / price_b * 10_000.0

    def signed_bps_move(self, current_price: float, reference_price: float) -> float:
        if current_price <= 0 or reference_price <= 0:
            return 0.0
        return (current_price - reference_price) / reference_price * 10_000.0

    def build_level_snapshot(
        self,
        *,
        symbol: str,
        exchange: str,
        side: str,
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
        from .enums import SpoofingSide

        side_enum = SpoofingSide(side) if side in SpoofingSide._value2member_map_ else SpoofingSide.UNKNOWN

        return OrderbookLevelSnapshot(
            symbol=symbol,
            exchange=exchange,
            side=side_enum,
            price=price,
            size=size,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=spread,
            sequence_id=sequence_id,
            timestamp=self.ensure_utc(timestamp),
            metadata=metadata or {},
        )

    async def emit_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: Any | None = None,
    ) -> None:
        if self.event_bus is None:
            return

        kwargs: dict[str, Any] = {
            "source": f"spoofing.{self.component.value}",
        }

        if priority is not None:
            kwargs["priority"] = priority

        await self.event_bus.emit(topic, payload, **kwargs)

    def serialize_dataclass(self, obj: Any) -> dict[str, Any]:
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not a dataclass")

    def feature_payload(self, features: SpoofingFeatures | None) -> dict[str, Any] | None:
        if features is None:
            return None
        return self.serialize_dataclass(features)

    def detector_result_payload(self, result: DetectorResult) -> dict[str, Any]:
        payload = self.serialize_dataclass(result)
        return payload

    def log_debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, extra=kwargs or None)

    def log_info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, extra=kwargs or None)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, extra=kwargs or None)

    def log_error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, extra=kwargs or None)


class BaseSpoofingDetector(BaseSpoofingModule, ABC):
    """
    Базовий клас для detector/scorer-компонентів spoofing-пакета.
    """

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> DetectorResult | None:
        """
        Синхронний аналіз. Якщо треба async-логіка — можна зробити окремий метод.
        """
        raise NotImplementedError


class BaseSpoofingTracker(BaseSpoofingModule, ABC):
    """
    Базовий клас для stateful tracker-компонентів.
    """

    @abstractmethod
    def cleanup(self, now: datetime | None = None) -> int:
        """
        Очищення простроченого стану.
        Повертає кількість видалених елементів.
        """
        raise NotImplementedError