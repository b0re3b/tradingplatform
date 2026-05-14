from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Mapping, MutableMapping, Sequence, TypeVar

from core.event_bus import Event, EventBus, EventPriority, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.price_action.models import Candle


StateT = TypeVar("StateT")

EventHandler = Callable[[Event], Awaitable[None]]


@dataclass(slots=True)
class BasePriceActionConfig:
    """
    Shared config contract for all analytics.price_action modules.

    Concrete analyzers should extend this dataclass and define only their
    domain-specific parameters there. Infrastructure behavior stays here:
    EventBus topics, event publishing, snapshots and scheduler integration.
    """

    emit_events: bool = True
    event_namespace: str = "analytics.price_action"

    publish_snapshots: bool = False
    snapshot_interval_seconds: float | None = None
    snapshot_job_timeout_seconds: float = 5.0
    snapshot_job_max_retries: int = 1
    snapshot_job_retry_delay_seconds: float = 1.0

    subscribe_market_candles: bool = True
    market_candle_topic: str = "market.candle"
    market_candles_topic: str = "market.candles"

    event_priority: EventPriority = EventPriority.NORMAL

    def validate(self) -> None:
        self.event_namespace = self._normalize_topic(self.event_namespace)
        self.market_candle_topic = self._normalize_topic(self.market_candle_topic)
        self.market_candles_topic = self._normalize_topic(self.market_candles_topic)

        if not self.event_namespace:
            raise ValueError("event_namespace must not be empty")

        if self.snapshot_interval_seconds is not None and self.snapshot_interval_seconds <= 0:
            raise ValueError("snapshot_interval_seconds must be > 0 when provided")

        if self.snapshot_job_timeout_seconds <= 0:
            raise ValueError("snapshot_job_timeout_seconds must be > 0")

        if self.snapshot_job_max_retries < 0:
            raise ValueError("snapshot_job_max_retries must be >= 0")

        if self.snapshot_job_retry_delay_seconds < 0:
            raise ValueError("snapshot_job_retry_delay_seconds must be >= 0")

        if self.subscribe_market_candles:
            if not self.market_candle_topic:
                raise ValueError("market_candle_topic must not be empty")
            if not self.market_candles_topic:
                raise ValueError("market_candles_topic must not be empty")

        if not isinstance(self.event_priority, EventPriority):
            self.event_priority = EventPriority(self.event_priority)

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        return str(topic or "").strip().strip(".")


class BasePriceActionModule(Generic[StateT], abc.ABC):
    """
    Base infrastructure for all analytics.price_action analyzers.

    Contract:
    - event_bus is mandatory and injected through constructor;
    - scheduler is optional, but all periodic jobs must go through it;
    - logger is created only through core.logger.get_logger;
    - subscriptions are registered via register() / EventBus.subscribe();
    - events are published only via awaited EventBus.emit();
    - sync domain logic should return results, while async handlers publish them.
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
        symbol = str(symbol or "").strip().upper()
        timeframe = str(timeframe or "").strip()

        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an instance of core.event_bus.EventBus")
        if scheduler is not None and not isinstance(scheduler, Scheduler):
            raise TypeError("scheduler must be an instance of core.scheduler.Scheduler or None")

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
        self._shutdown = False

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
        Register module subscriptions and scheduled jobs.

        Subclasses may override this method, but must call super().register()
        before adding domain-specific subscriptions.
        """
        if self._registered:
            self.logger.warning(
                "Price action module already registered",
                extra={"price_action_module": self.module_name},
            )
            return

        if self._shutdown:
            raise RuntimeError(f"{self.module_name} is already shut down and cannot be registered again")

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
                "price_action_module": self.module_name,
                "subscriptions": len(self._subscriptions),
                "scheduled_jobs": len(self._scheduled_job_ids),
            },
        )

    def unregister(self) -> None:
        """
        Unsubscribe module handlers and disable module-owned scheduler jobs.

        EventBus and Scheduler lifecycles remain owned by the application/core layer.
        """
        for subscription in list(self._subscriptions):
            try:
                self.event_bus.unsubscribe(subscription)
            except Exception:
                self.logger.exception(
                    "Failed to unsubscribe price action handler",
                    extra={
                        "price_action_module": self.module_name,
                        "pattern": getattr(subscription, "pattern", None),
                        "handler": getattr(subscription, "name", None),
                    },
                )

        self._subscriptions.clear()

        for job_id in list(self._scheduled_job_ids):
            try:
                if self.scheduler is not None and self.scheduler.get_job(job_id) is not None:
                    self.scheduler.disable_job(job_id)
            except Exception:
                self.logger.exception(
                    "Failed to disable scheduled job",
                    extra={"price_action_module": self.module_name, "job_id": job_id},
                )

        self._scheduled_job_ids.clear()
        self._registered = False

        self.logger.info(
            "Price action module unregistered",
            extra={"price_action_module": self.module_name},
        )

    async def shutdown(self) -> None:
        """
        Graceful module shutdown hook.

        This does not stop EventBus or Scheduler. Those are core/application-owned.
        """
        if self._shutdown:
            return

        self.unregister()
        self._shutdown = True

        self.logger.info(
            "Price action module shutdown completed",
            extra={"price_action_module": self.module_name},
        )

    def _subscribe(
        self,
        pattern: str,
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> Subscription:
        normalized_pattern = self._normalize_topic(pattern)
        if not normalized_pattern:
            raise ValueError("subscription pattern must not be empty")

        subscription = self.event_bus.subscribe(
            normalized_pattern,
            handler,
            name=name or getattr(handler, "__name__", self.module_name),
        )
        self._subscriptions.append(subscription)

        self.logger.debug(
            "Price action handler subscribed",
            extra={
                "price_action_module": self.module_name,
                "pattern": normalized_pattern,
                "handler": subscription.name,
            },
        )
        return subscription

    def _register_snapshot_job(self) -> None:
        if not self.config.publish_snapshots:
            return

        if self.config.snapshot_interval_seconds is None:
            return

        if self.scheduler is None:
            self.logger.warning(
                "Snapshot publishing requested but Scheduler is not provided",
                extra={"price_action_module": self.module_name},
            )
            return

        job_name = (
            f"{self.config.event_namespace}.snapshot."
            f"{self.symbol.lower()}.{self.timeframe}"
        )

        job_id = self.scheduler.add_interval_job(
            name=job_name,
            func=self.publish_snapshot,
            interval=self.config.snapshot_interval_seconds,
            run_immediately=False,
            max_retries=self.config.snapshot_job_max_retries,
            retry_delay=self.config.snapshot_job_retry_delay_seconds,
            timeout=self.config.snapshot_job_timeout_seconds,
            allow_overlap=False,
            enabled=True,
        )
        self._scheduled_job_ids.append(job_id)

        self.logger.info(
            "Price action snapshot job registered",
            extra={
                "price_action_module": self.module_name,
                "job_id": job_id,
                "job_name": job_name,
                "interval": self.config.snapshot_interval_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Default EventBus handlers
    # ------------------------------------------------------------------

    async def on_candle_event(self, event: Event) -> None:
        """
        Default single-candle handler.

        Concrete analyzers should override this method and publish their own
        update results from async context.
        """
        self.logger.debug(
            "Candle event ignored by base module",
            extra={
                "price_action_module": self.module_name,
                "topic": event.topic,
                "event_id": event.event_id,
            },
        )

    async def on_candles_event(self, event: Event) -> None:
        """
        Default batch-candles handler.

        Concrete analyzers should override this method and publish their own
        update results from async context.
        """
        self.logger.debug(
            "Candles event ignored by base module",
            extra={
                "price_action_module": self.module_name,
                "topic": event.topic,
                "event_id": event.event_id,
            },
        )

    # ------------------------------------------------------------------
    # Parsing / normalization helpers
    # ------------------------------------------------------------------

    def _parse_candle(self, raw: Mapping[str, Any], *, index: int | None = None) -> Candle:
        if not isinstance(raw, Mapping):
            raise TypeError("raw candle must be a mapping")

        timestamp_value = (
            raw.get("timestamp")
            or raw.get("ts")
            or raw.get("time")
            or raw.get("datetime")
        )

        candle_index = index if index is not None else int(raw.get("index", 0))

        try:
            return Candle(
                timestamp=self._ensure_utc_datetime(timestamp_value),
                open=float(raw["open"]),
                high=float(raw["high"]),
                low=float(raw["low"]),
                close=float(raw["close"]),
                volume=float(raw.get("volume", 0.0)),
                index=candle_index,
            )
        except KeyError as exc:
            raise ValueError(f"missing required candle field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candle payload: {exc}") from exc

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

        Supported:
        - single candle mapping;
        - list/tuple of candle mappings;
        - {"candle": {...}};
        - {"candles": [{...}, ...]}.
        """
        payload = event.payload

        if isinstance(payload, Mapping):
            candle = payload.get("candle")
            candles = payload.get("candles")

            if isinstance(candle, Mapping):
                return [candle]

            if isinstance(candles, Sequence) and not isinstance(candles, (str, bytes, bytearray)):
                return [item for item in candles if isinstance(item, Mapping)]

            if self._looks_like_candle(payload):
                return [payload]

            return []

        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return [item for item in payload if isinstance(item, Mapping)]

        return []

    def _looks_like_candle(self, payload: Mapping[str, Any]) -> bool:
        required = {"open", "high", "low", "close"}
        return required.issubset(payload.keys())

    def _ensure_utc_datetime(self, value: Any) -> datetime:
        if value is None:
            raise TypeError("timestamp is required")

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            # milliseconds
            if value > 1_000_000_000_000:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            # seconds
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("timestamp string must not be empty")

            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"

            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        raise TypeError(f"unsupported timestamp type: {type(value)!r}")

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_topic(topic: str | Enum) -> str:
        raw = topic.value if isinstance(topic, Enum) else str(topic or "")
        return raw.strip().strip(".")

    # ------------------------------------------------------------------
    # EventBus helpers
    # ------------------------------------------------------------------

    def _build_event_name(self, suffix: str | Enum) -> str:
        raw_suffix = self._normalize_topic(suffix)
        namespace = self._normalize_topic(self.config.event_namespace)

        if not raw_suffix:
            return namespace

        return f"{namespace}.{raw_suffix}"

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
        Emit analytics event through core.event_bus.EventBus.

        No background task is created here. Callers must await this method.
        """
        if not self.config.emit_events:
            return False

        normalized_event_name = self._normalize_topic(event_name)
        if not normalized_event_name:
            raise ValueError("event_name must not be empty")

        safe_payload = self._safe_serialize(payload)
        if not isinstance(safe_payload, Mapping):
            safe_payload = {"value": safe_payload}

        try:
            return await self.event_bus.emit(
                normalized_event_name,
                dict(safe_payload),
                priority=priority if priority is not None else self.config.event_priority,
                source=source or self.module_name,
                correlation_id=correlation_id,
                headers=headers or {},
            )
        except Exception:
            payload_keys = list(payload.keys()) if isinstance(payload, Mapping) else None
            self.logger.exception(
                "Failed to emit price action event",
                extra={
                    "price_action_module": self.module_name,
                    "event_name": normalized_event_name,
                    "payload_keys": payload_keys,
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
                "module": self.module_name,
                "snapshot": self.snapshot(),
                "published_at": self._now_utc().isoformat(),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

    async def publish_reset(
        self,
        *,
        correlation_id: str | None = None,
    ) -> bool:
        return await self._emit_event(
            self._build_event_name("reset"),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "module": self.module_name,
                "reset_at": self._now_utc().isoformat(),
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

        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._safe_serialize(v) for v in value]

        return str(value)

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