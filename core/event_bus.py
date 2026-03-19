from __future__ import annotations

import asyncio
import fnmatch
import inspect
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

from core.logger import get_logger


HandlerType = Callable[["Event"], Union[None, Awaitable[None]]]
MiddlewareType = Callable[["Event"], Union[Optional["Event"], Awaitable[Optional["Event"]]]]


class EventPriority(int, Enum):
    CRITICAL = 0
    HIGH = 10
    NORMAL = 20
    LOW = 30


class QueueFullPolicy(str, Enum):
    WAIT = "wait"
    DROP_NEW = "drop_new"
    DROP_OLDEST = "drop_oldest"


@dataclass(slots=True)
class Event:
    topic: str
    payload: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    headers: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    def copy_with(
        self,
        *,
        topic: Optional[str] = None,
        payload: Any = None,
        priority: Optional[EventPriority] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[dict[str, Any]] = None,
        retry_count: Optional[int] = None,
    ) -> "Event":
        return Event(
            topic=topic if topic is not None else self.topic,
            payload=self.payload if payload is None else payload,
            priority=priority if priority is not None else self.priority,
            timestamp=self.timestamp,
            event_id=self.event_id,
            source=source if source is not None else self.source,
            correlation_id=correlation_id if correlation_id is not None else self.correlation_id,
            headers=dict(self.headers if headers is None else headers),
            retry_count=retry_count if retry_count is not None else self.retry_count,
        )


@dataclass(slots=True)
class Subscription:
    pattern: str
    handler: HandlerType
    name: str
    enabled: bool = True


@dataclass(slots=True)
class _QueuedItem:
    event: Event


class EventBus:
    """
    Async event bus для модульної трейдинг-системи.

    Підтримує:
    - publish / emit
    - wildcard subscriptions
    - async + sync handlers
    - middleware chain
    - bounded queue
    - queue overflow policies
    - retry
    - graceful shutdown
    - metrics
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 20000,
        worker_count: int = 6,
        queue_full_policy: QueueFullPolicy = QueueFullPolicy.DROP_OLDEST,
        max_retries: int = 1,
        retry_delay: float = 0.02,
        enable_metrics: bool = True,
        service_name: str = "event_bus",
    ) -> None:
        self._max_queue_size = max_queue_size
        self._worker_count = worker_count
        self._queue_full_policy = queue_full_policy
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._enable_metrics = enable_metrics
        self._service_name = service_name

        self._logger = get_logger(
            __name__,
            event_type="event_bus",
        )

        self._queue: asyncio.PriorityQueue[
            tuple[int, int, _QueuedItem]
        ] = asyncio.PriorityQueue(maxsize=max_queue_size)

        self._subscriptions: list[Subscription] = []
        self._middlewares: list[MiddlewareType] = []
        self._workers: list[asyncio.Task] = []

        self._publish_seq = 0
        self._publish_lock = asyncio.Lock()

        self._running = False
        self._stopping = False

        self._error_handler: Optional[
            Callable[[Event, Exception, str], Union[None, Awaitable[None]]]
        ] = None

        self._metrics: dict[str, Any] = {
            "published": 0,
            "processed": 0,
            "failed": 0,
            "dropped": 0,
            "retried": 0,
            "queue_size": 0,
            "subscriptions": 0,
            "topic_published": {},
            "topic_processed": {},
            "handler_errors": {},
        }

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            self._logger.warning("EventBus already started")
            return

        self._running = True
        self._stopping = False

        for worker_id in range(self._worker_count):
            task = asyncio.create_task(
                self._worker_loop(worker_id),
                name=f"event-bus-worker-{worker_id}",
            )
            self._workers.append(task)

        self._logger.info(
            "EventBus started | workers=%s max_queue_size=%s policy=%s",
            self._worker_count,
            self._max_queue_size,
            self._queue_full_policy.value,
        )

    async def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        if not self._running:
            self._logger.warning("EventBus already stopped")
            return

        self._stopping = True

        self._logger.info(
            "Stopping EventBus | drain=%s timeout=%s",
            drain,
            timeout,
        )

        if drain:
            start_ts = time.time()
            while not self._queue.empty():
                if time.time() - start_ts >= timeout:
                    self._logger.warning(
                        "EventBus drain timeout reached | remaining_queue=%s",
                        self._queue.qsize(),
                    )
                    break
                await asyncio.sleep(0.05)

        for task in self._workers:
            task.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._running = False
        self._stopping = False

        self._logger.info("EventBus stopped")

    # ---------------------------------------------------------------------
    # Subscription API
    # ---------------------------------------------------------------------

    def subscribe(
        self,
        pattern: str,
        handler: HandlerType,
        *,
        name: Optional[str] = None,
    ) -> Subscription:
        subscription = Subscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
        )
        self._subscriptions.append(subscription)
        self._metrics["subscriptions"] = len(self._subscriptions)

        self._logger.info(
            "Handler subscribed | pattern=%s handler=%s",
            pattern,
            subscription.name,
        )
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        self._subscriptions = [sub for sub in self._subscriptions if sub is not subscription]
        self._metrics["subscriptions"] = len(self._subscriptions)

        self._logger.info(
            "Handler unsubscribed | pattern=%s handler=%s",
            subscription.pattern,
            subscription.name,
        )

    def add_middleware(self, middleware: MiddlewareType) -> None:
        middleware_name = getattr(middleware, "__name__", middleware.__class__.__name__)
        self._middlewares.append(middleware)

        self._logger.info(
            "Middleware added | middleware=%s",
            middleware_name,
        )

    def set_error_handler(
        self,
        handler: Callable[[Event, Exception, str], Union[None, Awaitable[None]]],
    ) -> None:
        self._error_handler = handler
        self._logger.info("Global event bus error handler registered")

    # ---------------------------------------------------------------------
    # Publish API
    # ---------------------------------------------------------------------

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[dict[str, Any]] = None,
    ) -> bool:
        return await self.publish(
            Event(
                topic=topic,
                payload=payload,
                priority=priority,
                source=source,
                correlation_id=correlation_id,
                headers=headers or {},
            )
        )

    async def publish(self, event: Event) -> bool:
        if not self._running:
            raise RuntimeError("EventBus is not started")

        if self._stopping:
            self._logger.warning(
                "Event rejected because EventBus is stopping | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            return False

        event = await self._apply_middlewares(event)
        if event is None:
            self._inc_metric("dropped")
            return False

        accepted = await self._enqueue_event(event)
        if accepted:
            self._inc_metric("published")
            self._inc_nested_metric("topic_published", event.topic)
            self._metrics["queue_size"] = self._queue.qsize()

        return accepted

    async def publish_nowait_best_effort(self, event: Event) -> bool:
        if not self._running:
            raise RuntimeError("EventBus is not started")

        if self._stopping:
            return False

        event = await self._apply_middlewares(event)
        if event is None:
            self._inc_metric("dropped")
            return False

        accepted = self._try_enqueue_nowait(event)
        if accepted:
            self._inc_metric("published")
            self._inc_nested_metric("topic_published", event.topic)
            self._metrics["queue_size"] = self._queue.qsize()

        return accepted

    # ---------------------------------------------------------------------
    # Workers
    # ---------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        worker_logger = get_logger(
            __name__,
            event_type="event_bus_worker",
            worker_id=worker_id,
        )

        worker_logger.info("EventBus worker started")

        try:
            while True:
                _, _, queued_item = await self._queue.get()
                event = queued_item.event

                try:
                    await self._dispatch(event)
                    self._inc_metric("processed")
                    self._inc_nested_metric("topic_processed", event.topic)
                except Exception as exc:
                    self._inc_metric("failed")

                    worker_logger.exception(
                        "Event dispatch failed | topic=%s event_id=%s retry_count=%s",
                        event.topic,
                        event.event_id,
                        event.retry_count,
                    )

                    if event.retry_count < self._max_retries:
                        self._inc_metric("retried")
                        if self._retry_delay > 0:
                            await asyncio.sleep(self._retry_delay)

                        retry_event = event.copy_with(
                            retry_count=event.retry_count + 1
                        )
                        await self._enqueue_event(retry_event)
                    else:
                        await self._handle_dispatch_error(
                            event=event,
                            exc=exc,
                            location="dispatch",
                        )
                finally:
                    self._queue.task_done()
                    self._metrics["queue_size"] = self._queue.qsize()

        except asyncio.CancelledError:
            worker_logger.info("EventBus worker cancelled")
            raise
        except Exception:
            worker_logger.exception("EventBus worker crashed")
            raise
        finally:
            worker_logger.info("EventBus worker stopped")

    async def _dispatch(self, event: Event) -> None:
        matched_subscriptions = self._match_subscriptions(event.topic)

        if not matched_subscriptions:
            self._logger.debug(
                "No subscribers for topic | topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            return

        for subscription in matched_subscriptions:
            try:
                await self._invoke_handler(subscription, event)
            except Exception as exc:
                self._register_handler_error(subscription.name)

                get_logger(
                    __name__,
                    event_type="event_handler",
                    handler_name=subscription.name,
                ).exception(
                    "Handler failed | topic=%s event_id=%s",
                    event.topic,
                    event.event_id,
                )

                await self._handle_dispatch_error(
                    event=event,
                    exc=exc,
                    location=subscription.name,
                )

    async def _invoke_handler(self, subscription: Subscription, event: Event) -> None:
        result = subscription.handler(event)
        if inspect.isawaitable(result):
            await result

    # ---------------------------------------------------------------------
    # Queue / middleware / matching
    # ---------------------------------------------------------------------

    def _match_subscriptions(self, topic: str) -> list[Subscription]:
        return [
            sub
            for sub in self._subscriptions
            if sub.enabled and fnmatch.fnmatch(topic, sub.pattern)
        ]

    async def _apply_middlewares(self, event: Event) -> Optional[Event]:
        current_event: Optional[Event] = event

        for middleware in self._middlewares:
            if current_event is None:
                return None

            result = middleware(current_event)
            current_event = await result if inspect.isawaitable(result) else result

            if current_event is None:
                self._logger.debug(
                    "Event dropped by middleware | topic=%s event_id=%s",
                    event.topic,
                    event.event_id,
                )
                return None

        return current_event

    async def _enqueue_event(self, event: Event) -> bool:
        async with self._publish_lock:
            item = (
                int(event.priority),
                self._publish_seq,
                _QueuedItem(event=event),
            )
            self._publish_seq += 1

            if self._queue_full_policy == QueueFullPolicy.WAIT:
                await self._queue.put(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_NEW:
                if self._queue.full():
                    self._inc_metric("dropped")
                    self._logger.warning(
                        "Queue full, dropping new event | topic=%s event_id=%s",
                        event.topic,
                        event.event_id,
                    )
                    return False

                self._queue.put_nowait(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_OLDEST:
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                        self._inc_metric("dropped")

                        self._logger.warning(
                            "Queue full, dropped oldest event | topic=%s event_id=%s",
                            event.topic,
                            event.event_id,
                        )
                    except asyncio.QueueEmpty:
                        pass

                self._queue.put_nowait(item)
                return True

            raise ValueError(f"Unsupported queue full policy: {self._queue_full_policy}")

    def _try_enqueue_nowait(self, event: Event) -> bool:
        item = (
            int(event.priority),
            self._publish_seq,
            _QueuedItem(event=event),
        )
        self._publish_seq += 1

        if self._queue_full_policy == QueueFullPolicy.WAIT:
            if self._queue.full():
                self._inc_metric("dropped")
                return False
            self._queue.put_nowait(item)
            return True

        if self._queue_full_policy == QueueFullPolicy.DROP_NEW:
            if self._queue.full():
                self._inc_metric("dropped")
                return False
            self._queue.put_nowait(item)
            return True

        if self._queue_full_policy == QueueFullPolicy.DROP_OLDEST:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._inc_metric("dropped")
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(item)
            return True

        return False

    # ---------------------------------------------------------------------
    # Errors / metrics
    # ---------------------------------------------------------------------

    async def _handle_dispatch_error(
        self,
        *,
        event: Event,
        exc: Exception,
        location: str,
    ) -> None:
        if self._error_handler is None:
            return

        try:
            result = self._error_handler(event, exc, location)
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception(
                "Global event bus error handler failed | location=%s event_id=%s",
                location,
                event.event_id,
            )

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "stopping": self._stopping,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._max_queue_size,
            "worker_count": self._worker_count,
            "subscriptions": len(self._subscriptions),
            "published": self._metrics["published"],
            "processed": self._metrics["processed"],
            "failed": self._metrics["failed"],
            "dropped": self._metrics["dropped"],
            "retried": self._metrics["retried"],
            "topic_published": dict(self._metrics["topic_published"]),
            "topic_processed": dict(self._metrics["topic_processed"]),
            "handler_errors": dict(self._metrics["handler_errors"]),
        }

    def _inc_metric(self, key: str, amount: int = 1) -> None:
        if not self._enable_metrics:
            return
        self._metrics[key] = self._metrics.get(key, 0) + amount

    def _inc_nested_metric(self, key: str, nested_key: str, amount: int = 1) -> None:
        if not self._enable_metrics:
            return
        nested = self._metrics.setdefault(key, {})
        nested[nested_key] = nested.get(nested_key, 0) + amount

    def _register_handler_error(self, handler_name: str) -> None:
        if not self._enable_metrics:
            return
        errors = self._metrics.setdefault("handler_errors", {})
        errors[handler_name] = errors.get(handler_name, 0) + 1