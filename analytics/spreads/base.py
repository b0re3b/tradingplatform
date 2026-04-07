from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from core.logger import get_logger

from .config import BaseSpreadConfig
from .models import SpreadSignal, SpreadSnapshot
from .spread_regime_detector import SpreadRegimeDetector
from .spread_signal_engine import SpreadSignalEngine


class BaseSpreadAnalyzer(ABC):
    """
    Базовий клас для spread analyzer-компонентів.

    Відповідальність:
    - lifecycle: start / stop
    - спільна інтеграція з EventBus
    - publish helpers
    - cooldown / emit throttling
    - базові stats
    - logger / lock / signal engine / regime detector

    Не відповідає за:
    - конкретну бізнес-логіку spread-аналізу
    - побудову snapshot-ів
    - специфічну обробку quote/funding/opportunity даних
    """

    def __init__(
        self,
        config: BaseSpreadConfig,
        event_bus: Any,
        scheduler: Any | None = None,
        *,
        service_name: str = "spread_analyzer",
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._scheduler = scheduler

        self._logger = get_logger(__name__, service_name=service_name)

        self._regime_detector = SpreadRegimeDetector(config)
        self._signal_engine = SpreadSignalEngine(
            config=config,
            regime_detector=self._regime_detector,
        )

        self._running = False
        self._lock = asyncio.Lock()

        self._last_signal_times: dict[str, datetime] = {}
        self._last_emit_times: dict[tuple[Any, ...], datetime] = {}

        self._stats: dict[str, int] = self._build_base_stats()

    @abstractmethod
    async def _subscribe_events(self) -> None:
        """
        Конкретний analyzer має сам визначити, на які події EventBus підписуватись.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Конкретний analyzer має повернути розширену статистику.
        """
        raise NotImplementedError

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        await self._subscribe_events()

        self._logger.info(
            "%s started",
            self.__class__.__name__,
            extra=self._build_start_log_extra(),
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        self._logger.info(
            "%s stopped",
            self.__class__.__name__,
            extra=self._build_stop_log_extra(),
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def _build_base_stats(self) -> dict[str, int]:
        return {
            "calculations_total": 0,
            "snapshots_published": 0,
            "signals_published": 0,
            "cooldown_skips": 0,
            "emit_skips": 0,
            "exceptions": 0,
        }

    def _build_start_log_extra(self) -> dict[str, Any]:
        return {
            "max_quote_age_ms": self._config.max_quote_age_ms,
            "max_quote_skew_ms": self._config.max_quote_skew_ms,
            "rolling_window_size": self._config.rolling_window_size,
            "min_emit_interval_ms": self._config.min_emit_interval_ms,
            "cooldown_seconds": self._config.cooldown_seconds,
        }

    def _build_stop_log_extra(self) -> dict[str, Any]:
        return {
            "stats": self._stats.copy(),
        }

    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
            return await value
        return value

    async def _subscribe(
        self,
        event_name: str,
        handler: Any,
    ) -> None:
        subscribe = getattr(self._event_bus, "subscribe", None)
        if subscribe is None:
            self._logger.warning(
                "EventBus does not expose subscribe()",
                extra={"event_name": event_name},
            )
            return

        await self._maybe_await(subscribe(event_name, handler))

    async def _publish(
        self,
        event_name: str,
        payload: Any,
    ) -> None:
        publish = getattr(self._event_bus, "publish", None)
        if publish is None:
            self._logger.warning(
                "EventBus does not expose publish()",
                extra={"event_name": event_name},
            )
            return

        await self._maybe_await(publish(event_name, payload))

    def _should_skip_emit(
        self,
        key: tuple[Any, ...],
        timestamp: datetime,
    ) -> bool:
        last_emit_at = self._last_emit_times.get(key)
        if last_emit_at is None:
            self._last_emit_times[key] = timestamp
            return False

        min_interval = timedelta(milliseconds=self._config.min_emit_interval_ms)
        if (timestamp - last_emit_at) < min_interval:
            return True

        self._last_emit_times[key] = timestamp
        return False

    def _should_skip_signal(
        self,
        signal: SpreadSignal,
    ) -> bool:
        signal_key = self._build_signal_key(signal)
        now = signal.timestamp

        last_signal_at = self._last_signal_times.get(signal_key)
        if last_signal_at is None:
            self._last_signal_times[signal_key] = now
            return False

        cooldown = timedelta(seconds=self._config.cooldown_seconds)
        if (now - last_signal_at) < cooldown:
            return True

        self._last_signal_times[signal_key] = now
        return False

    def _build_signal_key(self, signal: SpreadSignal) -> str:
        exchange_a = signal.exchange_a or "na"
        exchange_b = signal.exchange_b or "na"

        return (
            f"{signal.signal_type.value}|"
            f"{signal.spread_type.value}|"
            f"{signal.symbol}|"
            f"{exchange_a}|"
            f"{exchange_b}"
        )

    async def _publish_snapshot(
        self,
        event_name: str,
        snapshot: SpreadSnapshot,
    ) -> None:
        self._stats["snapshots_published"] += 1

        self._logger.debug(
            "Spread snapshot published",
            extra={
                "event_name": event_name,
                "symbol": snapshot.symbol,
                "spread_type": snapshot.spread_type.value,
                "exchange_a": snapshot.leg_a_exchange,
                "exchange_b": snapshot.leg_b_exchange,
                "spread_bps": self._to_str(snapshot.spread_bps),
                "net_spread": self._to_str(snapshot.net_spread),
                "regime": snapshot.regime.value,
            },
        )

        await self._publish(event_name, snapshot)

    async def _publish_signal(
        self,
        event_name: str,
        signal: SpreadSignal,
    ) -> bool:
        if self._should_skip_signal(signal):
            self._stats["cooldown_skips"] += 1
            return False

        self._stats["signals_published"] += 1

        self._logger.debug(
            "Spread signal published",
            extra={
                "event_name": event_name,
                "signal_type": signal.signal_type.value,
                "spread_type": signal.spread_type.value,
                "symbol": signal.symbol,
                "exchange_a": signal.exchange_a,
                "exchange_b": signal.exchange_b,
                "value": self._to_str(signal.value),
                "threshold": self._to_str(signal.threshold),
                "confidence": self._to_str(signal.confidence),
            },
        )

        await self._publish(event_name, signal)
        return True

    async def _publish_signals(
        self,
        event_name: str,
        signals: list[SpreadSignal],
    ) -> int:
        published_count = 0

        for signal in signals:
            published = await self._publish_signal(event_name, signal)
            if published:
                published_count += 1

        return published_count

    def _evaluate_snapshot_signals(
        self,
        snapshot: SpreadSnapshot,
        previous_snapshot: SpreadSnapshot | None = None,
        opportunity: Any | None = None,
    ) -> list[SpreadSignal]:
        result = self._signal_engine.evaluate_snapshot(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            opportunity=opportunity,
        )
        return result.signals

    def _mark_exception(
        self,
        message: str,
        exc: Exception,
        **extra: Any,
    ) -> None:
        self._stats["exceptions"] += 1
        self._logger.exception(
            message,
            extra={
                "error": str(exc),
                **extra,
            },
        )

    @staticmethod
    def _to_str(value: Decimal | None) -> str | None:
        return str(value) if value is not None else None