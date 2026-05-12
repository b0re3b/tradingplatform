from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Mapping, MutableMapping, Sequence, TypeVar

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.price_action.models import Candle


StateT = TypeVar("StateT")


@dataclass(slots=True)
class BasePriceActionConfig:
    """
    Shared config contract for all analytics.price_action modules.

    Individual analyzers should extend this dataclass and keep their own
    domain parameters there.

    This config is local to the price_action domain and should be injected
    into analyzers by the application/bootstrap layer.
    """

    emit_events: bool = True
    event_namespace: str = "analytics.price_action"
    publish_snapshots: bool = False
    snapshot_interval_seconds: float | None = None

    subscribe_market_candles: bool = True
    market_candle_topic: str = "market.candle"
    market_candles_topic: str = "market.candles"

    event_priority: EventPriority = EventPriority.NORMAL

    def validate(self) -> None:
        if not self.event_namespace:
            raise ValueError("event_namespace must not be empty")

        if self.snapshot_interval_seconds is not None and self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be > 0 when provided")

        if not self.market_candle_topic:
            raise ValueError("market_candle_topic must not be empty")

        if not self.market_candles_topic:
            raise ValueError("market_candles_topic must not be empty")


class BasePriceActionModule(Generic[StateT], abc.ABC):
    """
    Shared infrastructure for all analytics.price_action analyzers.

    Responsibilities:
    - explicit EventBus dependency
    - optional Scheduler dependency
    - standard logger initialization via core.logger.get_logger
    - EventBus subscription lifecycle through register()
    - async event publishing through EventBus.emit()
    - optional scheduled snapshot publishing through Scheduler.add_interval_job()
    - common candle parsing
    - snapshot serialization helpers

    Domain analyzers should keep computation sync where practical
    and expose async EventBus handlers for integration.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: BasePriceActionConfig | None = None,
        service_name: str = "analytics.price_action.base",
    ) -> None:
        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")

        self.symbol = symbol
        self.timeframe = timeframe
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config or BasePriceActionConfig()
        self.config.validate()

        self.module_name = self.__class__.__name__
        self.service_name = service_name

        self.logger = get_logger(
            __name__,
            service_name=service_name,
            event_type="analytics_price_action",
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        self._subscriptions: list[Subscription] = []
        self._scheduled_job_ids: list[str] = []
        self._registered = False

    # ------------------------------------------------------------------
    # Abstract domain API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset rolling state, caches and in-memory derived entities.
        """

    @abc.abstractmethod
    def get_state(self) -> StateT:
        """
        Return current strongly typed state object.
        """

    @abc.abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """
        Return point-in-time serialized snapshot of the module state.
        """

    # ------------------------------------------------------------------
    # Registration / lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register this analyzer in the event-driven system.

        Subclasses may override this method, but should call super().register()
        before adding custom subscriptions.

        Default behavior:
        - subscribe to market.candle
        - subscribe to market.candles
        - schedule periodic snapshots when enabled
        """
        if self._registered:
            self.logger.warning(
                "Price action module already registered",
                extra={"module": self.module_name},
            )
            return

        if self.config.subscribe_market_candles:
            self._subscribe(
                self.config.market_candle_topic,
                self.on_candle_event,
                name=f"{self.module_name}.on_candle_event",
            )
            self._subscribe(
                self.config.market_candles_topic,
                self.on_candles_event,
                name=f"{self.module_name}.on_candles_event",
            )

        self._register_snapshot_job()

        self._registered = True
        self.logger.info(
            "Price action module registered",
            extra={
                "module": self.module_name,
                "subscriptions": len(self._subscriptions),
                "scheduled_jobs": len(self._scheduled_job_ids),
            },
        )

    async def shutdown(self) -> None:
        """
        Graceful module shutdown hook.

        The core EventBus owns worker tasks, and the core Scheduler owns
        scheduled jobs. This method disables scheduler jobs that were created
        by this module.
        """
        for job_id in list(self._scheduled_job_ids):
            try:
                if self.scheduler is not None and self.scheduler.get_job(job_id) is not None:
                    self.scheduler.disable_job(job_id)
            except Exception:
                self.logger.exception(
                    "Failed to disable scheduled job during shutdown",
                    extra={"module": self.module_name, "job_id": job_id},
                )

        self.logger.info(
            "Price action module shutdown completed",
            extra={
                "module": self.module_name,
                "scheduled_jobs": len(self._scheduled_job_ids),
            },
        )

    def _subscribe(
        self,
        pattern: str,
        handler,
        *,
        name: str | None = None,
    ) -> Subscription:
        subscription = self.event_bus.subscribe(
            pattern,
            handler,
            name=name or getattr(handler, "__name__", self.module_name),
        )
        self._subscriptions.append(subscription)
        return subscription

    def _register_snapshot_job(self) -> None:
        if not self.config.publish_snapshots:
            return

        if self.config.snapshot_interval_seconds is None:
            return

        if self.scheduler is None:
            self.logger.warning(
                "Snapshot publishing requested but Scheduler is not provided",
                extra={"module": self.module_name},
            )
            return

        job_id = self.scheduler.add_interval_job(
            name=f"{self.config.event_namespace}.snapshot.{self.symbol}.{self.timeframe}",
            func=self.publish_snapshot,
            interval=self.config.snapshot_interval_seconds,
            run_immediately=False,
            max_retries=1,
            retry_delay=1.0,
            timeout=5.0,
            allow_overlap=False,
            enabled=True,
        )
        self._scheduled_job_ids.append(job_id)

    # ------------------------------------------------------------------
    # Default EventBus handlers
    # ------------------------------------------------------------------

    async def on_candle_event(self, event: Event) -> None:
        """
        Default single-candle handler.

        Subclasses that support event-driven candle ingestion should override
        this method and call their own add_candle()/add_data() logic.
        """
        self.logger.debug(
            "Candle event ignored by base module",
            extra={
                "module": self.module_name,
                "topic": event.topic,
                "event_id": event.event_id,
            },
        )

    async def on_candles_event(self, event: Event) -> None:
        """
        Default batch-candles handler.

        Subclasses that support event-driven candle ingestion should override
        this method and call their own add_candles()/add_data() logic.
        """
        self.logger.debug(
            "Candles event ignored by base module",
            extra={
                "module": self.module_name,
                "topic": event.topic,
                "event_id": event.event_id,
            },
        )

    # ------------------------------------------------------------------
    # Parsing / normalization helpers
    # ------------------------------------------------------------------

    def _parse_candle(self, raw: Mapping[str, Any], *, index: int | None = None) -> Candle:
        timestamp_value = (
            raw.get("timestamp")
            or raw.get("ts")
            or raw.get("time")
            or raw.get("datetime")
        )

        timestamp = self._ensure_utc_datetime(timestamp_value)
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
        start_index: int | None = None,
    ) -> list[Candle]:
        parsed: list[Candle] = []

        for offset, raw in enumerate(candles):
            index = None if start_index is None else start_index + offset
            parsed.append(self._parse_candle(raw, index=index))

        return parsed

    def _extract_candles_payload(self, event: Event) -> list[Mapping[str, Any]]:
        """
        Normalize common market candle event payload shapes.

        Supported payloads:
        - single candle dict
        - list of candle dicts
        - {"candle": {...}}
        - {"candles": [{...}, ...]}
        """
        payload = event.payload

        if isinstance(payload, Mapping):
            if "candles" in payload and isinstance(payload["candles"], Sequence):
                return [x for x in payload["candles"] if isinstance(x, Mapping)]

            if "candle" in payload and isinstance(payload["candle"], Mapping):
                return [payload["candle"]]

            return [payload]

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [x for x in payload if isinstance(x, Mapping)]

        return []

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

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _build_event_name(self, suffix: str | Enum) -> str:
        raw_suffix = suffix.value if isinstance(suffix, Enum) else str(suffix)
        raw_suffix = raw_suffix.strip(".")

        if not raw_suffix:
            return self.config.event_namespace

        return f"{self.config.event_namespace}.{raw_suffix}"

    async def _emit_event(
        self,
        event_name: str,
        payload: Mapping[str, Any],
        *,
        source: str | None = None,
        priority: EventPriority | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        """
        Emit an analytics event through core.event_bus.EventBus.

        This method is intentionally async and does not create background tasks.
        Subclasses should await it from their async EventBus handlers.
        """
        if not self.config.emit_events:
            return False

        source_name = source or self.module_name

        try:
            return await self.event_bus.emit(
                event_name,
                dict(payload),
                priority=priority or self.config.event_priority,
                source=source_name,
                correlation_id=correlation_id,
                headers=headers,
            )
        except Exception:
            self.logger.exception(
                "Failed to emit price action event",
                extra={
                    "module": self.module_name,
                    "event_name": event_name,
                    "payload": self._safe_serialize(payload),
                },
            )
            return False

    async def _emit_many(
        self,
        events: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        source: str | None = None,
        priority: EventPriority | None = None,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> int:
        emitted = 0

        for event_name, payload in events:
            accepted = await self._emit_event(
                event_name,
                payload,
                source=source,
                priority=priority,
                correlation_id=correlation_id,
                headers=headers,
            )
            if accepted:
                emitted += 1

        return emitted

    async def publish_snapshot(
        self,
        *,
        snapshot_name: str = "snapshot",
        correlation_id: str | None = None,
    ) -> bool:
        if not self.config.publish_snapshots:
            return False

        return await self._emit_event(
            self._build_event_name(snapshot_name),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "snapshot": self.snapshot(),
                "published_at": self._now_utc().isoformat(),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_config(self) -> dict[str, Any]:
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
        metadata: MutableMapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "generated_at": self._now_utc().isoformat(),
            "module": self.module_name,
            "state": self._safe_serialize(state),
        }

        if metadata:
            payload["metadata"] = self._safe_serialize(dict(metadata))

        return payload