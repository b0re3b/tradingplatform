from __future__ import annotations

import asyncio
import fnmatch
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


HandlerType = Callable[["Event"], Union[None, Awaitable[None]]]
MiddlewareType = Callable[["Event"], Union["Event", Awaitable["Event"]]]


class EventPriority(int, Enum):
    LOW = 30
    NORMAL = 20
    HIGH = 10
    CRITICAL = 0


class QueueFullPolicy(str, Enum):
    WAIT = "wait"
    DROP_NEW = "drop_new"
    DROP_OLDEST = "drop_oldest"


@dataclass(slots=True)
class Event:
    """
    Базова подія для системи.

    topic:
        Ієрархічна назва події, наприклад:
        - market.trade
        - market.orderbook
        - signal.generated
        - risk.kill_switch
        - execution.order_filled

    payload:
        Дані події.

    priority:
        Чим менше число, тим вищий пріоритет у PriorityQueue.

    timestamp:
        Час створення події (time.time()).

    event_id:
        Унікальний id події для трасування.

    source:
        Джерело, наприклад:
        - binance_ws
        - strategy_engine
        - risk_manager

    correlation_id:
        Дозволяє зв’язувати події одного ланцюжка.
        Наприклад сигнал -> ордер -> fill -> pnl update.

    headers:
        Додаткові службові поля.

    retry_count:
        Скільки разів подія вже ретраїлась.
    """
    topic: str
    payload: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    def copy_with(
        self,
        *,
        topic: Optional[str] = None,
        payload: Any = None,
        priority: Optional[EventPriority] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        retry_count: Optional[int] = None,
    ) -> "Event":
        return Event(
            topic=topic if topic is not None else self.topic,
            payload=self.payload if payload is None else payload,
            priority=priority if priority is not None else self.priority,
            timestamp=self.timestamp,
            event_id=self.event_id,
            source=source if source is not None else self.source,
            correlation_id=(
                correlation_id if correlation_id is not None else self.correlation_id
            ),
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
    """
    Wrapper для PriorityQueue.

    PriorityQueue сортує за tuple у порядку:
    (priority, enqueue_seq, item)
    """
    event: Event


class EventBus:
    """
    Асинхронний event bus для high-throughput системи.

    Основні можливості:
    - publish / subscribe
    - wildcard topics через fnmatch
      приклад:
        market.*
        signal.*
        risk.*
        *.error

    - обробка sync та async handler'ів
    - bounded queue
    - queue full policy
    - worker pool
    - middleware chain
    - retry для publish failures
    - централізований error hook
    - graceful shutdown

    Підходить для архітектури:
        market data -> analytics -> strategy -> execution -> risk -> ai
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 10000,
        worker_count: int = 4,
        queue_full_policy: QueueFullPolicy = QueueFullPolicy.DROP_OLDEST,
        max_retries: int = 2,
        retry_delay: float = 0.05,
        enable_metrics: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(self.__class__.__name__)

        self._subscriptions: List[Subscription] = []
        self._middlewares: List[MiddlewareType] = []

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._worker_count = worker_count
        self._queue_full_policy = queue_full_policy
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stopping = False

        self._publish_seq = 0
        self._publish_lock = asyncio.Lock()

        self._error_handler: Optional[
            Callable[[Event, Exception, str], Union[None, Awaitable[None]]]
        ] = None

        self._enable_metrics = enable_metrics
        self._metrics: Dict[str, Any] = {
            "published": 0,
            "processed": 0,
            "failed": 0,
            "dropped": 0,
            "retried": 0,
            "active_workers": 0,
            "queue_size": 0,
            "handler_errors": {},
            "topic_published": {},
            "topic_processed": {},
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._stopping = False

        for idx in range(self._worker_count):
            task = asyncio.create_task(self._worker_loop(idx), name=f"eventbus-worker-{idx}")
            self._workers.append(task)

        self._logger.info(
            "EventBus started | workers=%s queue_max=%s policy=%s",
            self._worker_count,
            self._queue.maxsize,
            self._queue_full_policy.value,
        )

    async def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        """
        Graceful shutdown.

        drain=True:
            дочекається обробки черги.

        drain=False:
            швидко зупиняє workers.
        """
        if not self._running:
            return

        self._stopping = True

        if drain:
            start = time.time()
            while not self._queue.empty():
                if time.time() - start > timeout:
                    self._logger.warning("EventBus drain timeout reached")
                    break
                await asyncio.sleep(0.05)

        for task in self._workers:
            task.cancel()

        results = await asyncio.gather(*self._workers, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                self._logger.exception("Worker shutdown error: %s", res)

        self._workers.clear()
        self._running = False
        self._stopping = False

        self._logger.info("EventBus stopped")

    # -------------------------------------------------------------------------
    # Subscription API
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        pattern: str,
        handler: HandlerType,
        *,
        name: Optional[str] = None,
    ) -> Subscription:
        """
        Підписка на topic pattern.

        Приклади:
            bus.subscribe("market.trade", on_trade)
            bus.subscribe("market.*", on_any_market_event)
            bus.subscribe("risk.*", on_risk_event)
        """
        sub = Subscription(
            pattern=pattern,
            handler=handler,
            name=name or getattr(handler, "__name__", "anonymous_handler"),
        )
        self._subscriptions.append(sub)

        self._logger.debug("Subscribed handler=%s pattern=%s", sub.name, sub.pattern)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        self._subscriptions = [s for s in self._subscriptions if s is not subscription]
        self._logger.debug(
            "Unsubscribed handler=%s pattern=%s",
            subscription.name,
            subscription.pattern,
        )

    def add_middleware(self, middleware: MiddlewareType) -> None:
        self._middlewares.append(middleware)
        self._logger.debug(
            "Middleware added: %s",
            getattr(middleware, "__name__", middleware.__class__.__name__),
        )

    def set_error_handler(
        self,
        handler: Callable[[Event, Exception, str], Union[None, Awaitable[None]]],
    ) -> None:
        """
        Глобальний error handler для помилок у subscriber handler'ах.
        """
        self._error_handler = handler

    # -------------------------------------------------------------------------
    # Publish API
    # -------------------------------------------------------------------------

    async def publish(self, event: Event) -> bool:
        """
        Публікація події в шину.

        Повертає:
            True  - подія прийнята в чергу
            False - подія відкинута
        """
        if not self._running:
            raise RuntimeError("EventBus is not started")

        if self._stopping:
            self._logger.warning(
                "EventBus is stopping, event rejected | topic=%s event_id=%s",
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
        """
        Корисно для ultra-hot path, де не хочеться блокуватися.
        Якщо черга переповнена — застосовується policy.
        """
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

    async def emit(
        self,
        topic: str,
        payload: Any,
        *,
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Зручний helper замість ручного Event(...).
        """
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

    # -------------------------------------------------------------------------
    # Worker loop
    # -------------------------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        self._logger.info("EventBus worker started | id=%s", worker_id)

        while True:
            try:
                self._metrics["active_workers"] = len(
                    [w for w in self._workers if not w.done()]
                )

                priority, seq, queued_item = await self._queue.get()
                event = queued_item.event

                try:
                    await self._dispatch(event)
                    self._inc_metric("processed")
                    self._inc_nested_metric("topic_processed", event.topic)

                except Exception as exc:
                    self._inc_metric("failed")
                    self._logger.exception(
                        "Dispatch failed | worker=%s topic=%s event_id=%s error=%s",
                        worker_id,
                        event.topic,
                        event.event_id,
                        exc,
                    )

                    if event.retry_count < self._max_retries:
                        self._inc_metric("retried")
                        await asyncio.sleep(self._retry_delay)

                        retry_event = event.copy_with(
                            retry_count=event.retry_count + 1
                        )
                        await self._enqueue_event(retry_event)
                    else:
                        await self._handle_dispatch_error(event, exc, "dispatch")

                finally:
                    self._queue.task_done()
                    self._metrics["queue_size"] = self._queue.qsize()

            except asyncio.CancelledError:
                self._logger.info("EventBus worker cancelled | id=%s", worker_id)
                raise
            except Exception as exc:
                self._logger.exception(
                    "Worker loop crashed | worker=%s error=%s",
                    worker_id,
                    exc,
                )
                await asyncio.sleep(0.1)

    async def _dispatch(self, event: Event) -> None:
        matched = self._match_subscriptions(event.topic)

        if not matched:
            self._logger.debug(
                "No subscribers for topic=%s event_id=%s",
                event.topic,
                event.event_id,
            )
            return

        # Важливо:
        # тут handlers запускаються послідовно, щоб зберегти передбачуваність.
        # Для окремих high-throughput use-case можна зробити fan-out паралельно,
        # але тоді ускладнюється консистентність і контроль помилок.
        for sub in matched:
            try:
                await self._invoke_handler(sub, event)
            except Exception as exc:
                self._register_handler_error(sub.name)
                self._logger.exception(
                    "Handler failed | handler=%s topic=%s event_id=%s error=%s",
                    sub.name,
                    event.topic,
                    event.event_id,
                    exc,
                )
                await self._handle_dispatch_error(event, exc, sub.name)

    async def _invoke_handler(self, subscription: Subscription, event: Event) -> None:
        result = subscription.handler(event)
        if inspect.isawaitable(result):
            await result

    # -------------------------------------------------------------------------
    # Matching / middleware / queue
    # -------------------------------------------------------------------------

    def _match_subscriptions(self, topic: str) -> List[Subscription]:
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
            item = (int(event.priority), self._publish_seq, _QueuedItem(event=event))
            self._publish_seq += 1

            if self._queue_full_policy == QueueFullPolicy.WAIT:
                await self._queue.put(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_NEW:
                if self._queue.full():
                    self._inc_metric("dropped")
                    self._logger.warning(
                        "Queue full, dropping NEW event | topic=%s event_id=%s",
                        event.topic,
                        event.event_id,
                    )
                    return False

                self._queue.put_nowait(item)
                return True

            if self._queue_full_policy == QueueFullPolicy.DROP_OLDEST:
                if self._queue.full():
                    try:
                        _ = self._queue.get_nowait()
                        self._queue.task_done()
                        self._inc_metric("dropped")
                        self._logger.warning(
                            "Queue full, dropped OLDEST event to enqueue new one | "
                            "topic=%s event_id=%s",
                            event.topic,
                            event.event_id,
                        )
                    except asyncio.QueueEmpty:
                        pass

                self._queue.put_nowait(item)
                return True

            raise ValueError(f"Unsupported queue policy: {self._queue_full_policy}")

    def _try_enqueue_nowait(self, event: Event) -> bool:
        item = (int(event.priority), self._publish_seq, _QueuedItem(event=event))
        self._publish_seq += 1

        if self._queue_full_policy == QueueFullPolicy.WAIT:
            # nowait-версія для WAIT не блокує, тому якщо full — drop
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
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                    self._inc_metric("dropped")
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(item)
            return True

        return False

    # -------------------------------------------------------------------------
    # Error handling / metrics
    # -------------------------------------------------------------------------

    async def _handle_dispatch_error(
        self,
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
        except Exception as hook_exc:
            self._logger.exception(
                "Global error handler failed | location=%s error=%s",
                location,
                hook_exc,
            )

    def stats(self) -> Dict[str, Any]:
        data = dict(self._metrics)
        data["queue_size"] = self._queue.qsize()
        data["subscriptions"] = len(self._subscriptions)
        data["running"] = self._running
        data["stopping"] = self._stopping
        return data

    def _inc_metric(self, key: str, amount: int = 1) -> None:
        if not self._enable_metrics:
            return
        self._metrics[key] = self._metrics.get(key, 0) + amount

    def _inc_nested_metric(self, key: str, nested_key: str, amount: int = 1) -> None:
        if not self._enable_metrics:
            return
        if key not in self._metrics:
            self._metrics[key] = {}
        self._metrics[key][nested_key] = self._metrics[key].get(nested_key, 0) + amount

    def _register_handler_error(self, handler_name: str) -> None:
        if not self._enable_metrics:
            return
        errors = self._metrics.setdefault("handler_errors", {})
        errors[handler_name] = errors.get(handler_name, 0) + 1