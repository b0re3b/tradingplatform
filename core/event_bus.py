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


class HandlerDispatchMode(str, Enum):
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"


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
    dispatch_mode: HandlerDispatchMode | None = None


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
    - topic-aware queue overflow policies
    - retry
    - graceful shutdown
    - metrics
    """

    # High-volume market-data topics are intentionally lossy. They may be
    # dropped under backpressure before they are allowed to evict critical
    # trading/system events from the shared queue.
    LOSSY_TOPIC_PREFIXES: tuple[str, ...] = (
        "market.trade",
        "market.trades.",
        "market.orderbook",
        "market.orderbook.",
    )

    # Protected topics must not be evicted by market-data pressure. If the
    # queue is full and no lossy queued event can be removed, async publishers
    # will backpressure until capacity is available.
    PROTECTED_TOPIC_PREFIXES: tuple[str, ...] = (
        "market.candle",
        "market.candles.",
        "analytics.",
        "strategy.",
        "signal.",
        "risk.",
        "execution.",
        "position.",
        "account.",
        "system.",
    )

    def __init__(
        self,
        *,
        max_queue_size: int = 100000,
        worker_count: int = 12,
        queue_full_policy: QueueFullPolicy = QueueFullPolicy.DROP_OLDEST,
        max_retries: int = 1,
        retry_delay: float = 0.02,
        enable_metrics: bool = True,
        handler_dispatch_mode: HandlerDispatchMode | str = HandlerDispatchMode.CONCURRENT,
        handler_timeout: float | None = None,
        service_name: str = "event_bus",
    ) -> None:
        self._max_queue_size = max_queue_size
        self._worker_count = worker_count
        self._queue_full_policy = queue_full_policy
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._enable_metrics = enable_metrics
        self._handler_dispatch_mode = self._coerce_dispatch_mode(handler_dispatch_mode)
        self._handler_timeout = handler_timeout
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
            "topic_dropped": {},
            "drop_reasons": {},
            "protected_enqueue_waits": 0,
            "handler_errors": {},
            "handler_dispatch_mode": self._handler_dispatch_mode.value,
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
            "EventBus started | workers=%s max_queue_size=%s policy=%s handler_dispatch_mode=%s handler_timeout=%s",
            self._worker_count,
            self._max_queue_size,
            self._queue_full_policy.value,
            self._handler_dispatch_mode.value,
            self._handler_timeout,
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
        dispatch_mode: HandlerDispatchMode | str | None = None,
    ) -> Subscription:
        subscription = Subscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
            dispatch_mode=self._coerce_dispatch_mode(dispatch_mode) if dispatch_mode is not None else None,
        )
        self._subscriptions.append(subscription)
        self._metrics["subscriptions"] = len(self._subscriptions)

        self._logger.info(
            "Handler subscribed | pattern=%s handler=%s dispatch_mode=%s",
            pattern,
            subscription.name,
            (subscription.dispatch_mode or self._handler_dispatch_mode).value,
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

        # Keep compatibility with order-sensitive handlers by allowing per-
        # subscription or global sequential mode. The default is concurrent so
        # one slow analytics/cache/notification handler does not block every
        # other handler for the same market-data event.
        sequential_subscriptions = [
            sub for sub in matched_subscriptions
            if (sub.dispatch_mode or self._handler_dispatch_mode) == HandlerDispatchMode.SEQUENTIAL
        ]
        concurrent_subscriptions = [
            sub for sub in matched_subscriptions
            if (sub.dispatch_mode or self._handler_dispatch_mode) == HandlerDispatchMode.CONCURRENT
        ]

        for subscription in sequential_subscriptions:
            await self._invoke_handler_safely(subscription, event)

        if concurrent_subscriptions:
            await asyncio.gather(
                *(self._invoke_handler_safely(subscription, event) for subscription in concurrent_subscriptions),
                return_exceptions=True,
            )

    async def _invoke_handler_safely(self, subscription: Subscription, event: Event) -> None:
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
            if self._handler_timeout is not None:
                await asyncio.wait_for(result, timeout=self._handler_timeout)
            else:
                await result

    # ---------------------------------------------------------------------
    # Queue / middleware / matching
    # ---------------------------------------------------------------------

    @staticmethod
    def _coerce_dispatch_mode(value: HandlerDispatchMode | str) -> HandlerDispatchMode:
        if isinstance(value, HandlerDispatchMode):
            return value
        try:
            return HandlerDispatchMode(str(value).strip().lower())
        except Exception as exc:
            raise ValueError(f"Invalid handler dispatch mode: {value!r}") from exc

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
            item = self._make_queue_item(event)

            if self._queue_full_policy == QueueFullPolicy.WAIT:
                await self._queue.put(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_NEW:
                if self._queue.full():
                    # DROP_NEW is explicit best-effort behaviour. Keep this
                    # policy simple, but track the actual incoming topic.
                    self._record_drop(
                        event,
                        reason="queue_full_drop_new",
                        log_message="Queue full, dropping new event",
                    )
                    return False

                self._queue.put_nowait(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_OLDEST:
                if not self._queue.full():
                    self._queue.put_nowait(item)
                    return True

                return await self._enqueue_with_topic_aware_drop(item)

            raise ValueError(f"Unsupported queue full policy: {self._queue_full_policy}")

    def _try_enqueue_nowait(self, event: Event) -> bool:
        item = self._make_queue_item(event)

        if self._queue_full_policy == QueueFullPolicy.WAIT:
            if self._queue.full():
                self._record_drop(
                    event,
                    reason="queue_full_wait_nowait_rejected",
                    log_message="Queue full, best-effort event rejected",
                )
                return False

            self._queue.put_nowait(item)
            return True

        if self._queue_full_policy == QueueFullPolicy.DROP_NEW:
            if self._queue.full():
                self._record_drop(
                    event,
                    reason="queue_full_drop_new",
                    log_message="Queue full, dropping new event",
                )
                return False

            self._queue.put_nowait(item)
            return True

        if self._queue_full_policy == QueueFullPolicy.DROP_OLDEST:
            if not self._queue.full():
                self._queue.put_nowait(item)
                return True

            dropped = self._drop_one_queued_lossy_event_nowait(
                incoming_event=event,
                reason="queue_full_drop_queued_lossy_nowait",
            )
            if dropped:
                self._queue.put_nowait(item)
                return True

            if self._is_lossy_topic(event.topic):
                self._record_drop(
                    event,
                    reason="queue_full_drop_incoming_lossy_nowait",
                    log_message="Queue full, dropping incoming lossy event",
                )
                return False

            # Best-effort publish cannot safely block. Refuse the protected
            # incoming event instead of evicting another protected event.
            self._record_drop(
                event,
                reason="queue_full_protected_nowait_rejected",
                log_message="Queue full, protected best-effort event rejected",
            )
            return False

        return False

    def _make_queue_item(self, event: Event) -> tuple[int, int, _QueuedItem]:
        item = (
            int(event.priority),
            self._publish_seq,
            _QueuedItem(event=event),
        )
        self._publish_seq += 1
        return item

    async def _enqueue_with_topic_aware_drop(
        self,
        item: tuple[int, int, _QueuedItem],
    ) -> bool:
        event = item[2].event

        dropped = self._drop_one_queued_lossy_event_nowait(
            incoming_event=event,
            reason="queue_full_drop_queued_lossy",
        )
        if dropped:
            self._queue.put_nowait(item)
            return True

        if self._is_lossy_topic(event.topic):
            self._record_drop(
                event,
                reason="queue_full_drop_incoming_lossy",
                log_message="Queue full, dropping incoming lossy event",
            )
            return False

        # Protected events should not be sacrificed because of market-data
        # pressure. If the queue contains no lossy event to evict, apply
        # backpressure to the protected publisher until a worker frees capacity.
        self._inc_metric("protected_enqueue_waits")
        self._inc_nested_metric("drop_reasons", "protected_backpressure_wait")
        self._logger.warning(
            "Queue full, waiting to enqueue protected event | topic=%s event_id=%s queue_size=%s",
            event.topic,
            event.event_id,
            self._queue.qsize(),
        )
        await self._queue.put(item)
        return True

    def _drop_one_queued_lossy_event_nowait(
        self,
        *,
        incoming_event: Event,
        reason: str,
    ) -> bool:
        if self._queue.empty():
            return False

        drained: list[tuple[int, int, _QueuedItem]] = []

        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # We are temporarily removing this item from the Queue. Mark the
            # original queue task as done; reinserted items will create a fresh
            # unfinished task and will be task_done()'d by workers later.
            self._queue.task_done()
            drained.append(queued)

        if not drained:
            return False

        drop_index: int | None = None

        # Drop the oldest queued lossy item by publish sequence. Do not use
        # PriorityQueue.get_nowait() semantics here because it returns the
        # highest-priority item, which is exactly what we must avoid dropping.
        for index, (_, _, queued_item) in enumerate(drained):
            queued_event = queued_item.event
            if self._is_lossy_topic(queued_event.topic):
                if drop_index is None or drained[index][1] < drained[drop_index][1]:
                    drop_index = index

        dropped_event: Event | None = None
        if drop_index is not None:
            _, _, dropped_item = drained.pop(drop_index)
            dropped_event = dropped_item.event

        for queued in drained:
            self._queue.put_nowait(queued)

        if dropped_event is None:
            return False

        self._record_drop(
            dropped_event,
            reason=reason,
            log_message="Queue full, dropped queued lossy event",
            incoming_event=incoming_event,
        )
        return True

    @classmethod
    def _is_lossy_topic(cls, topic: str) -> bool:
        return topic.startswith(cls.LOSSY_TOPIC_PREFIXES)

    @classmethod
    def _is_protected_topic(cls, topic: str) -> bool:
        return topic.startswith(cls.PROTECTED_TOPIC_PREFIXES)

    def _record_drop(
        self,
        event: Event,
        *,
        reason: str,
        log_message: str,
        incoming_event: Event | None = None,
    ) -> None:
        self._inc_metric("dropped")
        self._inc_nested_metric("topic_dropped", event.topic)
        self._inc_nested_metric("drop_reasons", reason)

        if incoming_event is None:
            self._logger.warning(
                "%s | dropped_topic=%s dropped_event_id=%s reason=%s queue_size=%s",
                log_message,
                event.topic,
                event.event_id,
                reason,
                self._queue.qsize(),
            )
            return

        self._logger.warning(
            "%s | dropped_topic=%s dropped_event_id=%s incoming_topic=%s incoming_event_id=%s reason=%s queue_size=%s",
            log_message,
            event.topic,
            event.event_id,
            incoming_event.topic,
            incoming_event.event_id,
            reason,
            self._queue.qsize(),
        )

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
            "topic_dropped": dict(self._metrics["topic_dropped"]),
            "drop_reasons": dict(self._metrics["drop_reasons"]),
            "protected_enqueue_waits": self._metrics["protected_enqueue_waits"],
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