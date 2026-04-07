from __future__ import annotations

import abc
import asyncio
import inspect
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

from core.logger import get_logger
from analytics.price_action.models import Candle


StateT = TypeVar("StateT")


@dataclass(slots=True)
class BasePriceActionConfig:
    """
    Shared config part for all price_action modules.
    Individual analyzers can extend this dataclass.
    """
    emit_events: bool = True
    event_namespace: str = "price_action"
    publish_snapshots: bool = False

    def validate(self) -> None:
        if not self.event_namespace:
            raise ValueError("event_namespace must not be empty")


class BasePriceActionModule(Generic[StateT], abc.ABC):
    """
    Shared infrastructure for all price_action analyzers.

    Responsibilities:
    - standard logger initialization
    - common candle parsing
    - sync/async EventBus compatibility
    - safe event publishing
    - snapshot serialization helpers
    - graceful shutdown of pending async tasks
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: Optional[Any] = None,
        config: Optional[BasePriceActionConfig] = None,
        service_name: str = "price_action.base",
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.config = config or BasePriceActionConfig()
        self.config.validate()

        self.logger = get_logger(__name__, service_name=service_name)
        self._pending_tasks: Set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset rolling state, caches and in-memory derived entities.
        """

    @abc.abstractmethod
    def get_state(self) -> StateT:
        """
        Return current strongly-typed state object.
        """

    @abc.abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Return point-in-time serialized snapshot of the module state.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """
        Gracefully await pending async EventBus tasks.
        """
        if not self._pending_tasks:
            return

        pending = list(self._pending_tasks)
        self.logger.info(
            "Shutting down price action module with pending tasks",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "pending_tasks": len(pending),
                "module": self.__class__.__name__,
            },
        )
        await asyncio.gather(*pending, return_exceptions=True)

    # ------------------------------------------------------------------
    # Parsing / normalization helpers
    # ------------------------------------------------------------------

    def _parse_candle(self, raw: Mapping[str, Any], *, index: Optional[int] = None) -> Candle:
        timestamp = self._ensure_utc_datetime(
            raw.get("timestamp")
            or raw.get("ts")
            or raw.get("time")
            or raw.get("datetime")
        )

        candle_index = index if index is not None else int(raw.get("index", 0))

        return Candle(
            timestamp=timestamp,
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume", 0.0)),
            index=candle_index,
        )

    def _parse_candles(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        start_index: Optional[int] = None,
    ) -> list[Candle]:
        parsed: list[Candle] = []

        for offset, raw in enumerate(candles):
            index = None if start_index is None else start_index + offset
            parsed.append(self._parse_candle(raw, index=index))

        return parsed

    def _ensure_utc_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            if value > 1_000_000_000_000:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _build_event_name(self, suffix: str) -> str:
        suffix = suffix.strip(".")
        if not suffix:
            return self.config.event_namespace
        return f"{self.config.event_namespace}.{suffix}"

    async def _await_emit_result(self, result: Awaitable[Any]) -> Any:
        return await result

    def _track_task(self, task: asyncio.Task[Any]) -> None:
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def _emit_event(
        self,
        event_name: str,
        payload: Mapping[str, Any],
        *,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Safe EventBus emit wrapper.

        Supports both:
        - async emit(...)
        - sync emit(...)

        This method is intentionally non-async so analyzer code can stay simple.
        """
        if not self.config.emit_events or self.event_bus is None:
            return

        emit_fn = getattr(self.event_bus, "emit", None)
        if emit_fn is None:
            self.logger.warning(
                "EventBus has no emit method",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "module": self.__class__.__name__,
                    "event_name": event_name,
                },
            )
            return

        source_name = source or self.__class__.__name__.lower()

        try:
            result = emit_fn(event_name, dict(payload), source=source_name, **kwargs)

            if inspect.isawaitable(result):
                task = asyncio.create_task(self._await_emit_result(result))
                self._track_task(task)
        except Exception:
            self.logger.exception(
                "Failed to emit event",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "module": self.__class__.__name__,
                    "event_name": event_name,
                    "payload": self._safe_serialize(payload),
                },
            )

    def _publish_snapshot(self, *, snapshot_name: str = "snapshot") -> None:
        if not self.config.publish_snapshots:
            return

        self._emit_event(
            self._build_event_name(snapshot_name),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "snapshot": self.snapshot(),
                "published_at": datetime.now(timezone.utc).isoformat(),
            },
            source=self.__class__.__name__.lower(),
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_config(self) -> Dict[str, Any]:
        serialized = self._safe_serialize(self.config)
        if isinstance(serialized, dict):
            return serialized
        return {"value": serialized}

    def _safe_serialize(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return {k: self._safe_serialize(v) for k, v in asdict(value).items()}

        if isinstance(value, Mapping):
            return {str(k): self._safe_serialize(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._safe_serialize(v) for v in value]

        return value

    def _snapshot_envelope(
        self,
        *,
        state: Any,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "module": self.__class__.__name__,
            "state": self._safe_serialize(state),
        }
        if metadata:
            payload["metadata"] = self._safe_serialize(dict(metadata))
        return payload