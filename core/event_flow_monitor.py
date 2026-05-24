from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.event_bus import Event, EventBus, Subscription
from core.logger import get_logger


DEFAULT_MONITORED_TOPICS: tuple[str, ...] = (
    "market.candle.closed",
    "market.candles.updated",
    "market.trade",
    "market.trades.batch",
    "market.trades.updated",
    "market.orderbook.batch",
    "market.orderbook.updated",
    "analytics.price_action.updated",
    "analytics.orderflow.cvd.updated",
    "analytics.funding.updated",
    "analytics.oi.updated",
    "analytics.liquidity.*",
    "analytics.liquidations.*",
    "analytics.spoofing.*",
    "analytics.whales.*",
    "analytics.spreads.*",
    "storage.*",
    "market.state.*",
    "system.*",
    "strategy.*",
    "signal.rejected",
    "signal.generated",
)


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_list(key: str, default: Iterable[str]) -> tuple[str, ...]:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(slots=True)
class EventFlowMonitorConfig:
    enabled: bool = True
    interval_seconds: float = 60.0
    topics: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MONITORED_TOPICS)
    log_zero_topics: bool = True
    emit_system_event: bool = False
    write_jsonl: bool = False
    jsonl_path: str = "logs/event_flow_monitor.jsonl"

    @classmethod
    def from_env(cls) -> "EventFlowMonitorConfig":
        return cls(
            enabled=_env_bool("EVENT_FLOW_MONITOR_ENABLED", True),
            interval_seconds=_env_float("EVENT_FLOW_MONITOR_INTERVAL_SECONDS", 60.0),
            topics=_env_list("EVENT_FLOW_MONITOR_TOPICS", DEFAULT_MONITORED_TOPICS),
            log_zero_topics=_env_bool("EVENT_FLOW_MONITOR_LOG_ZERO_TOPICS", True),
            emit_system_event=_env_bool("EVENT_FLOW_MONITOR_EMIT_SYSTEM_EVENT", False),
            write_jsonl=_env_bool("EVENT_FLOW_MONITOR_WRITE_JSONL", False),
            jsonl_path=os.getenv("EVENT_FLOW_MONITOR_JSONL_PATH", "logs/event_flow_monitor.jsonl"),
        )


class EventFlowMonitor:
    """
    Lightweight EventBus telemetry component.

    It subscribes to selected topics and periodically logs per-topic counters
    plus EventBus queue/drop metrics. Use this to quickly locate where the
    market-data -> analytics -> strategy event flow breaks.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: EventFlowMonitorConfig | None = None,
        *,
        service_name: str = "event_flow_monitor",
    ) -> None:
        self.event_bus = event_bus
        self.config = config or EventFlowMonitorConfig.from_env()
        self.service_name = service_name
        self.logger = get_logger(__name__, event_type="event_flow_monitor")

        self._subscriptions: list[Subscription] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._counts: Counter[str] = Counter()
        self._last_counts: Counter[str] = Counter()
        self._started_at = time.time()
        self._rejection_reasons: Counter[str] = Counter()
        self._last_rejection_reasons: Counter[str] = Counter()

    async def start(self) -> None:
        if not self.config.enabled:
            self.logger.info("EventFlowMonitor disabled")
            return

        if self._running:
            return

        self._running = True
        for topic in self.config.topics:
            self._subscriptions.append(
                self.event_bus.subscribe(
                    topic,
                    self._on_event,
                    name=f"{self.service_name}:{topic}",
                )
            )

        self._task = asyncio.create_task(
            self._report_loop(),
            name="event-flow-monitor",
        )

        self.logger.info(
            "EventFlowMonitor started | interval_seconds=%s topics=%s",
            self.config.interval_seconds,
            ",".join(self.config.topics),
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        for subscription in list(self._subscriptions):
            self.event_bus.unsubscribe(subscription)
        self._subscriptions.clear()

        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        self.logger.info("EventFlowMonitor stopped")

    async def _on_event(self, event: Event) -> None:
        self._counts[event.topic] += 1
        if event.topic == "signal.rejected":
            payload = getattr(event, "payload", None)
            reason = "unknown"
            if isinstance(payload, dict):
                reason = str(payload.get("reason") or payload.get("rejection_reason") or "unknown")
                metadata = payload.get("metadata")
                if reason == "unknown" and isinstance(metadata, dict):
                    reason = str(metadata.get("reason") or metadata.get("failure_stage") or "unknown")
            self._rejection_reasons[reason] += 1

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        pattern = str(pattern or "").strip()
        topic = str(topic or "").strip()
        if not pattern:
            return False
        if pattern == topic:
            return True
        if pattern.endswith("*"):
            return topic.startswith(pattern[:-1])
        return False

    def _count_for_pattern(self, pattern: str, counts: Counter[str]) -> int:
        if "*" not in pattern:
            return counts.get(pattern, 0)
        return sum(value for topic, value in counts.items() if self._topic_matches(pattern, topic))

    async def _report_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.config.interval_seconds)
                await self.report()
        except asyncio.CancelledError:
            raise

    async def report(self) -> dict[str, Any]:
        stats = self.event_bus.stats()
        counts = {topic: self._count_for_pattern(topic, self._counts) for topic in self.config.topics}
        deltas = {
            topic: (
                self._count_for_pattern(topic, self._counts)
                - self._count_for_pattern(topic, self._last_counts)
            )
            for topic in self.config.topics
        }
        rejection_reason_counts = dict(self._rejection_reasons.most_common(20))
        rejection_reason_deltas = {
            reason: count - self._last_rejection_reasons.get(reason, 0)
            for reason, count in self._rejection_reasons.items()
        }
        rejection_reason_deltas = {
            reason: delta
            for reason, delta in sorted(rejection_reason_deltas.items(), key=lambda kv: kv[1], reverse=True)[:20]
            if delta
        }
        self._last_counts = Counter(self._counts)
        self._last_rejection_reasons = Counter(self._rejection_reasons)

        # Top-20 most-published topics from EventBus internal counters.
        # This reveals what is actually flooding the queue, even if the topic
        # is not in the monitored list above.
        all_published: dict[str, int] = stats.get("topic_published", {})
        top_published = dict(
            sorted(all_published.items(), key=lambda kv: kv[1], reverse=True)[:20]
        )
        all_dropped: dict[str, int] = stats.get("topic_dropped", {})
        top_dropped = dict(
            sorted(all_dropped.items(), key=lambda kv: kv[1], reverse=True)[:20]
        )

        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "uptime_seconds": round(time.time() - self._started_at, 3),
            "interval_seconds": self.config.interval_seconds,
            "counts": counts,
            "deltas": deltas,
            "eventbus": {
                "queue_size": stats.get("queue_size", 0),
                "queue_utilization": stats.get("queue_utilization", 0.0),
                "max_queue_size": stats.get("max_queue_size"),
                "published": stats.get("published", 0),
                "processed": stats.get("processed", 0),
                "failed": stats.get("failed", 0),
                "dropped": stats.get("dropped", 0),
                "retried": stats.get("retried", 0),
                "drop_reasons": stats.get("drop_reasons", {}),
                "topic_dropped": stats.get("topic_dropped", {}),
                "handler_errors": stats.get("handler_errors", {}),
                "top_published": top_published,
                "top_dropped": top_dropped,
                "rejection_reasons": rejection_reason_counts,
                "rejection_reason_deltas": rejection_reason_deltas,
            },
        }

        visible_deltas = deltas if self.config.log_zero_topics else {
            topic: value for topic, value in deltas.items() if value
        }

        self.logger.warning(
            "Event flow monitor | queue_size=%s queue_utilization=%.2f dropped=%s failed=%s"
            " interval_counts=%s total_counts=%s drop_reasons=%s top_published=%s top_dropped=%s rejection_reason_deltas=%s handler_errors=%s",
            payload["eventbus"]["queue_size"],
            payload["eventbus"]["queue_utilization"],
            payload["eventbus"]["dropped"],
            payload["eventbus"]["failed"],
            visible_deltas,
            counts,
            payload["eventbus"]["drop_reasons"],
            top_published,
            top_dropped,
            rejection_reason_deltas,
            payload["eventbus"]["handler_errors"],
        )

        if self.config.write_jsonl:
            self._write_jsonl(payload)

        if self.config.emit_system_event:
            await self.event_bus.emit(
                "system.event_flow_monitor.report",
                payload,
                source=self.service_name,
            )

        return payload

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        path = Path(self.config.jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")