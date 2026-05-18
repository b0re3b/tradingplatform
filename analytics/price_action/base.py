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

from analytics.price_action.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    Candle,
    PriceActionKey,
    make_price_action_key,
    normalize_exchange,
    normalize_exchange_symbol,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
    price_action_key_to_dict,
)


StateT = TypeVar("StateT")

EventHandler = Callable[[Event], Awaitable[None]]


@dataclass(slots=True)
class BasePriceActionConfig:
    """
    Shared infrastructure config for all analytics.price_action modules.

    Concrete analyzers should extend this dataclass and define only their
    domain-specific parameters there.

    Correct data flow:
        exchange adapters
            -> market.candle
            -> CandlesCache
            -> market.candle.closed / market.candles.updated
            -> analytics.price_action
            -> analytics.price_action.*

    Price action analyzers should not consume raw exchange candle events by default.
    """

    emit_events: bool = True
    event_namespace: str = "analytics.price_action"

    publish_snapshots: bool = False
    snapshot_interval_seconds: float | None = None
    snapshot_job_timeout_seconds: float = 5.0
    snapshot_job_max_retries: int = 1
    snapshot_job_retry_delay_seconds: float = 1.0

    subscribe_market_candles: bool = True

    # Data-layer topics, not raw exchange topics.
    market_candle_topic: str = "market.candle.closed"
    market_candles_topic: str = "market.candles.updated"

    # Scope safety.
    require_event_scope: bool = True

    event_priority: EventPriority = EventPriority.NORMAL

    def validate(self) -> None:
        self.event_namespace = self._normalize_topic(self.event_namespace)
        self.market_candle_topic = self._normalize_topic(self.market_candle_topic)
        self.market_candles_topic = self._normalize_topic(self.market_candles_topic)

        if not self.event_namespace:
            raise ValueError("event_namespace must not be empty")

        if (
            self.snapshot_interval_seconds is not None
            and self.snapshot_interval_seconds <= 0
        ):
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

        self.require_event_scope = bool(self.require_event_scope)

    @staticmethod
    def _normalize_topic(topic: str) -> str:
        return str(topic or "").strip().strip(".")


class BasePriceActionModule(Generic[StateT], abc.ABC):
    """
    Base infrastructure for all analytics.price_action analyzers.

    Contract:
    - EventBus is mandatory and injected through constructor;
    - Scheduler is optional, but all periodic jobs must go through it;
    - logger is created only through core.logger.get_logger;
    - subscriptions are registered via register() / EventBus.subscribe();
    - events are published only via awaited EventBus.emit();
    - candle input comes from CandlesCache, not raw exchange adapters;
    - every module is scoped by exchange + market_type + symbol + timeframe.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Futures examples:
        ("binance", "usdm_futures", "BTCUSDT", "1m")
        ("bybit", "linear", "BTCUSDT", "1m")
        ("okx", "swap", "BTCUSDT", "1m")
        ("mexc", "usdm_futures", "BTCUSDT", "1m")
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        *,
        event_bus: EventBus,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        exchange_symbol: str | None = None,
        scheduler: Scheduler | None = None,
        config: BasePriceActionConfig | None = None,
        service_name: str = "analytics.price_action.base",
    ) -> None:
        symbol = normalize_symbol(symbol)
        timeframe = normalize_timeframe(timeframe)
        exchange = normalize_exchange(exchange)
        market_type = normalize_market_type(market_type)
        exchange_symbol = normalize_exchange_symbol(
            exchange_symbol,
            fallback_symbol=symbol,
        )

        if not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an instance of core.event_bus.EventBus")

        if scheduler is not None and not isinstance(scheduler, Scheduler):
            raise TypeError(
                "scheduler must be an instance of core.scheduler.Scheduler or None"
            )

        self.exchange = exchange
        self.market_type = market_type
        self.symbol = symbol
        self.exchange_symbol = exchange_symbol
        self.timeframe = timeframe

        self.key: PriceActionKey = make_price_action_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

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
            exchange=self.exchange,
            market_type=self.market_type,
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
                extra=self._log_scope_extra(),
            )
            return

        if self._shutdown:
            raise RuntimeError(
                f"{self.module_name} is already shut down and cannot be registered again"
            )

        if self.config.subscribe_market_candles:
            self._subscribe(
                self.config.market_candle_topic,
                self._on_candle_event_scoped,
                name=f"{self.module_name}.on_candle_event",
            )
            self._subscribe(
                self.config.market_candles_topic,
                self._on_candles_event_scoped,
                name=f"{self.module_name}.on_candles_event",
            )

        self._register_snapshot_job()

        self._registered = True
        self.logger.info(
            "Price action module registered",
            extra={
                **self._log_scope_extra(),
                "subscriptions": len(self._subscriptions),
                "scheduled_jobs": len(self._scheduled_job_ids),
                "market_candle_topic": self.config.market_candle_topic,
                "market_candles_topic": self.config.market_candles_topic,
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
                        **self._log_scope_extra(),
                        "pattern": getattr(subscription, "pattern", None),
                        "handler": getattr(subscription, "name", None),
                    },
                )

        self._subscriptions.clear()

        for job_id in list(self._scheduled_job_ids):
            try:
                if (
                    self.scheduler is not None
                    and self.scheduler.get_job(job_id) is not None
                ):
                    self.scheduler.disable_job(job_id)
            except Exception:
                self.logger.exception(
                    "Failed to disable scheduled job",
                    extra={**self._log_scope_extra(), "job_id": job_id},
                )

        self._scheduled_job_ids.clear()
        self._registered = False

        self.logger.info(
            "Price action module unregistered",
            extra=self._log_scope_extra(),
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
            extra=self._log_scope_extra(),
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
                **self._log_scope_extra(),
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
                extra=self._log_scope_extra(),
            )
            return

        job_name = (
            f"{self.config.event_namespace}.snapshot."
            f"{self.exchange}.{self.market_type}."
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
                **self._log_scope_extra(),
                "job_id": job_id,
                "job_name": job_name,
                "interval": self.config.snapshot_interval_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Scoped EventBus wrappers
    # ------------------------------------------------------------------

    async def _on_candle_event_scoped(self, event: Event) -> None:
        """
        Wrapper around subclass on_candle_event().

        It filters data-layer candle events by:
            exchange + market_type + symbol + timeframe
        """
        if not self._event_matches_module_scope(event):
            self.logger.debug(
                "Candle event skipped because scope does not match module",
                extra={
                    **self._log_scope_extra(),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )
            return

        await self.on_candle_event(event)

    async def _on_candles_event_scoped(self, event: Event) -> None:
        """
        Wrapper around subclass on_candles_event().

        Batch payloads are accepted when the top-level payload matches module
        scope, or at least one candle inside the batch matches module scope.
        """
        if not self._event_matches_module_scope(event, allow_batch=True):
            self.logger.debug(
                "Candles event skipped because scope does not match module",
                extra={
                    **self._log_scope_extra(),
                    "topic": getattr(event, "topic", None),
                    "event_id": getattr(event, "event_id", None),
                },
            )
            return

        await self.on_candles_event(event)

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
                **self._log_scope_extra(),
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
                **self._log_scope_extra(),
                "topic": event.topic,
                "event_id": event.event_id,
            },
        )

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    @property
    def scope_payload(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.exchange_symbol,
            "timeframe": self.timeframe,
            "key": list(self.key),
        }

    def _log_scope_extra(self) -> dict[str, Any]:
        return {
            "price_action_module": self.module_name,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "key": list(self.key),
        }

    def _event_matches_module_scope(
        self,
        event: Event,
        *,
        allow_batch: bool = False,
    ) -> bool:
        payload = getattr(event, "payload", None)

        if not self.config.require_event_scope:
            return True

        payload_key = self._extract_key_from_payload(payload)
        if payload_key is not None:
            return payload_key == self.key

        if allow_batch:
            candle_payloads = self._extract_candles_payload(event)
            if not candle_payloads:
                return False

            return any(
                self._extract_key_from_payload(candle_payload) == self.key
                for candle_payload in candle_payloads
            )

        return False

    def _extract_key_from_payload(self, payload: Any) -> PriceActionKey | None:
        if not isinstance(payload, Mapping):
            return None

        key = self._extract_key_from_mapping(payload)
        if key is not None:
            return key

        candle = payload.get("candle")
        if isinstance(candle, Mapping):
            key = self._extract_key_from_mapping(candle)
            if key is not None:
                return key

        data = payload.get("data")
        if isinstance(data, Mapping):
            key = self._extract_key_from_mapping(data)
            if key is not None:
                return key

        return None

    def _extract_key_from_mapping(
        self,
        payload: Mapping[str, Any],
    ) -> PriceActionKey | None:
        exchange = (
            payload.get("exchange")
            or payload.get("venue")
            or payload.get("source_exchange")
        )
        market_type = (
            payload.get("market_type")
            or payload.get("category")
            or payload.get("inst_type")
            or payload.get("instrument_type")
        )
        symbol = (
            payload.get("symbol")
            or payload.get("s")
            or payload.get("instrument")
            or payload.get("market")
        )
        timeframe = (
            payload.get("timeframe")
            or payload.get("tf")
            or payload.get("interval")
        )

        if not exchange or not market_type or not symbol or not timeframe:
            return None

        try:
            return make_price_action_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        except ValueError:
            return None

    def _ensure_payload_scope(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Add module scope to any emitted payload.

        Caller payload can override neither scope nor key. This prevents a child
        analyzer from accidentally emitting symbol-only or wrong-scope events.
        """
        safe_payload = dict(payload)
        safe_payload.update(self.scope_payload)
        return safe_payload

    # ------------------------------------------------------------------
    # Parsing / normalization helpers
    # ------------------------------------------------------------------

    def _parse_candle(
        self,
        raw: Mapping[str, Any],
        *,
        index: int | None = None,
    ) -> Candle:
        if not isinstance(raw, Mapping):
            raise TypeError("raw candle must be a mapping")

        timestamp_value = (
            raw.get("timestamp")
            or raw.get("timestamp_ms")
            or raw.get("close_time_ms")
            or raw.get("open_time_ms")
            or raw.get("received_at_ms")
            or raw.get("ts")
            or raw.get("time")
            or raw.get("datetime")
        )

        candle_index = index if index is not None else int(raw.get("index", 0))

        exchange = (
            raw.get("exchange")
            or raw.get("venue")
            or raw.get("source_exchange")
            or self.exchange
        )
        market_type = (
            raw.get("market_type")
            or raw.get("category")
            or raw.get("inst_type")
            or raw.get("instrument_type")
            or self.market_type
        )
        symbol = raw.get("symbol") or raw.get("s") or raw.get("instrument") or self.symbol
        timeframe = (
            raw.get("timeframe")
            or raw.get("tf")
            or raw.get("interval")
            or self.timeframe
        )
        exchange_symbol = (
            raw.get("exchange_symbol")
            or raw.get("raw_symbol")
            or raw.get("exchangeSymbol")
            or self.exchange_symbol
        )

        try:
            candle = Candle(
                exchange=str(exchange),
                market_type=str(market_type),
                symbol=str(symbol),
                exchange_symbol=(
                    str(exchange_symbol) if exchange_symbol is not None else None
                ),
                timeframe=str(timeframe),
                timestamp=self._ensure_utc_datetime(timestamp_value),
                open=float(raw["open"]),
                high=float(raw["high"]),
                low=float(raw["low"]),
                close=float(raw["close"]),
                volume=float(raw.get("volume", 0.0)),
                quote_volume=(
                    float(raw["quote_volume"])
                    if raw.get("quote_volume") is not None
                    else None
                ),
                trades_count=(
                    int(raw["trades_count"])
                    if raw.get("trades_count") is not None
                    else None
                ),
                is_closed=bool(raw.get("is_closed", True)),
                index=candle_index,
                metadata=dict(raw.get("metadata") or {}),
            )
        except KeyError as exc:
            raise ValueError(f"missing required candle field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candle payload: {exc}") from exc

        if candle.key != self.key:
            raise ValueError(
                "candle scope does not match price action module scope: "
                f"candle={candle.key}, module={self.key}"
            )

        return candle

    def _parse_candles(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        start_index: int | None = None,
    ) -> list[Candle]:
        parsed: list[Candle] = []

        for offset, raw in enumerate(candles):
            index = None if start_index is None else start_index + offset
            try:
                parsed.append(self._parse_candle(raw, index=index))
            except ValueError:
                self.logger.debug(
                    "Candle skipped while parsing batch because scope or payload is invalid",
                    extra={
                        **self._log_scope_extra(),
                        "batch_offset": offset,
                    },
                )

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
                return [self._merge_parent_scope(payload, candle)]

            if isinstance(candles, Sequence) and not isinstance(
                candles,
                (str, bytes, bytearray),
            ):
                return [
                    self._merge_parent_scope(payload, item)
                    for item in candles
                    if isinstance(item, Mapping)
                ]

            if self._looks_like_candle(payload):
                return [payload]

            return []

        if isinstance(payload, Sequence) and not isinstance(
            payload,
            (str, bytes, bytearray),
        ):
            return [item for item in payload if isinstance(item, Mapping)]

        return []

    def _merge_parent_scope(
        self,
        parent: Mapping[str, Any],
        child: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(child)

        for key in (
            "exchange",
            "market_type",
            "symbol",
            "exchange_symbol",
            "timeframe",
        ):
            if key not in merged and key in parent:
                merged[key] = parent[key]

        return merged

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
        The module scope is always injected into payload.
        """
        if not self.config.emit_events:
            return False

        normalized_event_name = self._normalize_topic(event_name)
        if not normalized_event_name:
            raise ValueError("event_name must not be empty")

        scoped_payload = self._ensure_payload_scope(payload)
        safe_payload = self._safe_serialize(scoped_payload)

        if not isinstance(safe_payload, Mapping):
            safe_payload = {"value": safe_payload, **self.scope_payload}

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
                    **self._log_scope_extra(),
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
            return {
                k: self._safe_serialize(v)
                for k, v in asdict(value).items()
            }

        if isinstance(value, Mapping):
            return {
                str(k): self._safe_serialize(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                self._safe_serialize(v)
                for v in value
            ]

        return str(value)

    def _snapshot_envelope(
        self,
        *,
        state: Any,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self.scope_payload,
            "generated_at": self._now_utc().isoformat(),
            "module": self.module_name,
            "state": self._safe_serialize(state),
        }

        if metadata:
            payload["metadata"] = self._safe_serialize(dict(metadata))

        return payload