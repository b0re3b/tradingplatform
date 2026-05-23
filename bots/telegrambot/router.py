"""
Telegram bot package router.

Routing layer для Telegram notification service.

Цей модуль:
- не викликає Telegram API;
- не форматує HTML-повідомлення;
- не підписується на EventBus;
- не містить торгової бізнес-логіки;
- тільки визначає: EventBus event -> TelegramTopicRoute.

Основний контракт:
TelegramEventPayload -> TelegramTopicRoute
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from .config import TelegramBotConfig
from .enums import (
    TelegramEventCategory,
    TelegramMessageType,
    TelegramNotificationLevel,
    TelegramPriority,
    TelegramRoutePolicy,
    TelegramTopic,
)
from .exceptions import (
    TelegramRoutingError,
    TelegramTopicNotConfiguredError,
)
from .models import TelegramEventPayload, TelegramTopicRoute


@dataclass(slots=True, frozen=True)
class TelegramRoutingRule:
    """
    Одна routing rule.

    pattern:
        fnmatch pattern для EventBus event.name.
        Наприклад: analytics.orderflow.*

    topic:
        Логічна Telegram-гілка.

    message_type:
        Тип повідомлення для formatter.py.

    category:
        Високорівнева категорія події.
    """

    pattern: str
    topic: TelegramTopic
    message_type: TelegramMessageType
    category: TelegramEventCategory
    priority: TelegramPriority = TelegramPriority.NORMAL
    level: TelegramNotificationLevel = TelegramNotificationLevel.INFO
    enabled: bool = True
    reason: str | None = None

    def matches(self, event_name: str) -> bool:
        if not self.enabled:
            return False
        return fnmatch(event_name, self.pattern)


class TelegramRouter:
    """
    Router для EventBus -> Telegram topic.

    Використовується handlers.py:
        route = router.resolve(event_payload)

    Router не знає про Telegram HTTP API і не викликає client.send_message().
    """

    # Analytics events are high-volume and include many lifecycle/cache updates.
    # Telegram should receive only actionable trading-style analytics events.
    _ACTIONABLE_ANALYTICS_EVENTS: frozenset[str] = frozenset(
        {
            # Orderflow
            "analytics.orderflow.cvd.signal",
            "analytics.orderflow.volume_delta.signal",
            "analytics.orderflow.aggressive_trades.signal",
            "analytics.orderflow.orderbook_imbalance.signal",

            # Liquidity
            "analytics.liquidity.signal.updated",
            "analytics.liquidity.level.detected",
            "analytics.liquidity.level.swept",
            "analytics.liquidity.stop_cluster.detected",

            # Liquidations
            "analytics.liquidations.cascade_detected",
            "analytics.liquidations.exhaustion_detected",

            # Whales
            "analytics.whales.whale_activity",
            "analytics.whales.whale_pressure",
            "analytics.whales.whale_liquidation_context",
            "analytics.whales.whale_cluster_exhaustion",

            # Spoofing
            "analytics.spoofing.detected",

            # Spreads
            "analytics.spreads.signal.generated",
            "analytics.spreads.arbitrage.opportunity",

            # Funding
            "analytics.funding.signal",
            "analytics.funding.extreme",
            "analytics.funding.flip",
            "analytics.funding.divergence",
            "analytics.funding.pressure",

            # Open Interest: current package may emit analytics.oi.*
            "analytics.oi.divergence",
            "analytics.oi.divergence.detected",
            "analytics.oi.anomaly",
            "analytics.oi.anomaly.detected",
            "analytics.oi.squeeze_setup",
            "analytics.oi.capitulation",
            "analytics.oi.capitulation.detected",
            "analytics.oi.regime.changed",
            "analytics.open_interest.divergence",
            "analytics.open_interest.divergence.detected",
            "analytics.open_interest.anomaly",
            "analytics.open_interest.anomaly.detected",
            "analytics.open_interest.squeeze_setup",
            "analytics.open_interest.capitulation",
            "analytics.open_interest.capitulation.detected",
            "analytics.open_interest.regime.changed",
        }
    )

    _ACTIONABLE_ANALYTICS_PATTERNS: tuple[str, ...] = (
        "analytics.*.signal",
        "analytics.*.*.signal",
        "analytics.*.signal.generated",
        "analytics.*.setup",
        "analytics.*.*.setup",
        "analytics.*.detected",
        "analytics.*.*.detected",
        "analytics.*.opportunity",
        "analytics.*.*.opportunity",
        "analytics.*.divergence",
        "analytics.*.anomaly",
        "analytics.*.capitulation",
        "analytics.*.squeeze_setup",
        "analytics.*.extreme",
        "analytics.*.flip",
        "analytics.*.pressure",
        "analytics.*.cascade_detected",
        "analytics.*.exhaustion_detected",
    )

    _NON_ACTIONABLE_ANALYTICS_SUFFIXES: tuple[str, ...] = (
        ".started",
        ".stopped",
        ".heartbeat",
        ".health",
        ".healthcheck",
        ".metrics",
        ".snapshot",
        ".updated",
        ".cleanup",
        ".error",
        ".failed",
        ".warning",
        ".lifecycle",
    )

    _NON_ACTIONABLE_ANALYTICS_CONTAINS: tuple[str, ...] = (
        ".analyzer.",
        ".service.",
        ".scheduler.",
        ".persistence.",
        ".state.",
        ".cache.",
    )

    _DIAGNOSTIC_SIGNAL_REJECT_REASONS: frozenset[str] = frozenset(
        {
            "no_strategies_routed",
            "no_strategy_routed",
            "no_matching_strategy",
            "no_route",
            "not_routed",
            "no_raw_signals",
            "no_signals",
            "no_signal",
            "no_signal_generated",
            "no_strategy_signal",
            "no_candidate_signals",
            "empty_signal_batch",
        }
    )

    def __init__(
        self,
        config: TelegramBotConfig,
        *,
        rules: list[TelegramRoutingRule] | None = None,
    ) -> None:
        self.config = config
        self._rules: list[TelegramRoutingRule] = rules or self.default_rules()

    @staticmethod
    def default_rules() -> list[TelegramRoutingRule]:
        """
        Базові правила маршрутизації для нашої trading architecture.
        """

        return [
            # -----------------------------------------------------------------
            # Analytics domains
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="analytics.orderflow.*",
                topic=TelegramTopic.ORDERFLOW,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="orderflow analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.liquidity.*",
                topic=TelegramTopic.LIQUIDITY,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="liquidity analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.price_action.*",
                topic=TelegramTopic.PRICE_ACTION,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="price action analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.liquidations.*",
                topic=TelegramTopic.LIQUIDATIONS,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.WARNING,
                reason="liquidations analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.whales.*",
                topic=TelegramTopic.WHALES,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.WARNING,
                reason="whale analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.spoofing.*",
                topic=TelegramTopic.SPOOFING,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.WARNING,
                reason="spoofing analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.spreads.*",
                topic=TelegramTopic.SPREADS,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="spreads analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.funding.*",
                topic=TelegramTopic.FUNDING,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="funding analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.open_interest.*",
                topic=TelegramTopic.OPEN_INTEREST,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="open interest analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.oi.*",
                topic=TelegramTopic.OPEN_INTEREST,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="open interest analytics event",
            ),
            TelegramRoutingRule(
                pattern="analytics.*",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.ANALYTICS_ALERT,
                category=TelegramEventCategory.ANALYTICS,
                priority=TelegramPriority.LOW,
                level=TelegramNotificationLevel.INFO,
                reason="generic analytics event fallback",
            ),

            # -----------------------------------------------------------------
            # News / AI
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="news.*",
                topic=TelegramTopic.NEWS,
                message_type=TelegramMessageType.NEWS_ALERT,
                category=TelegramEventCategory.NEWS,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.INFO,
                reason="news event",
            ),
            TelegramRoutingRule(
                pattern="ai.news.*",
                topic=TelegramTopic.NEWS,
                message_type=TelegramMessageType.NEWS_ALERT,
                category=TelegramEventCategory.AI,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.INFO,
                reason="ai news event",
            ),
            TelegramRoutingRule(
                pattern="ai.market_context.*",
                topic=TelegramTopic.NEWS,
                message_type=TelegramMessageType.NEWS_ALERT,
                category=TelegramEventCategory.AI,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="ai market context event",
            ),

            # -----------------------------------------------------------------
            # Signals
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="signal.generated",
                topic=TelegramTopic.SIGNALS,
                message_type=TelegramMessageType.SIGNAL_GENERATED,
                category=TelegramEventCategory.SIGNAL,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="strategy generated signal",
            ),
            TelegramRoutingRule(
                pattern="signal.updated",
                topic=TelegramTopic.SIGNALS,
                message_type=TelegramMessageType.SIGNAL_UPDATED,
                category=TelegramEventCategory.SIGNAL,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="strategy updated signal",
            ),
            TelegramRoutingRule(
                pattern="signal.rejected",
                topic=TelegramTopic.SIGNALS,
                message_type=TelegramMessageType.SIGNAL_REJECTED,
                category=TelegramEventCategory.SIGNAL,
                priority=TelegramPriority.LOW,
                level=TelegramNotificationLevel.WARNING,
                reason="strategy/risk rejected signal",
            ),
            TelegramRoutingRule(
                pattern="signal.confirmed",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.SIGNAL_CONFIRMED,
                category=TelegramEventCategory.SIGNAL,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.SUCCESS,
                reason="risk confirmed signal",
            ),

            # -----------------------------------------------------------------
            # Execution / orders
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="execution.order_submitted",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.ORDER_SUBMITTED,
                category=TelegramEventCategory.EXECUTION,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="order submitted",
            ),
            TelegramRoutingRule(
                pattern="execution.order_filled",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.ORDER_FILLED,
                category=TelegramEventCategory.EXECUTION,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.SUCCESS,
                reason="order filled",
            ),
            TelegramRoutingRule(
                pattern="execution.order_rejected",
                topic=TelegramTopic.RISK,
                message_type=TelegramMessageType.ORDER_REJECTED,
                category=TelegramEventCategory.EXECUTION,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.ERROR,
                reason="order rejected",
            ),
            TelegramRoutingRule(
                pattern="execution.order_cancelled",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.ORDER_CANCELLED,
                category=TelegramEventCategory.EXECUTION,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.WARNING,
                reason="order cancelled",
            ),

            # -----------------------------------------------------------------
            # Positions / trades
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="position.opened",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.POSITION_OPENED,
                category=TelegramEventCategory.POSITION,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.SUCCESS,
                reason="position opened",
            ),
            TelegramRoutingRule(
                pattern="position.updated",
                topic=TelegramTopic.OPEN_TRADES,
                message_type=TelegramMessageType.POSITION_UPDATED,
                category=TelegramEventCategory.POSITION,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.INFO,
                reason="position updated",
            ),
            TelegramRoutingRule(
                pattern="position.closed",
                topic=TelegramTopic.CLOSED_TRADES,
                message_type=TelegramMessageType.POSITION_CLOSED,
                category=TelegramEventCategory.POSITION,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.SUCCESS,
                reason="position closed",
            ),

            # -----------------------------------------------------------------
            # Risk
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="risk.limit_warning",
                topic=TelegramTopic.RISK,
                message_type=TelegramMessageType.RISK_WARNING,
                category=TelegramEventCategory.RISK,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.WARNING,
                reason="risk limit warning",
            ),
            TelegramRoutingRule(
                pattern="risk.position_blocked",
                topic=TelegramTopic.RISK,
                message_type=TelegramMessageType.RISK_BLOCKED,
                category=TelegramEventCategory.RISK,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.ERROR,
                reason="risk blocked position",
            ),
            TelegramRoutingRule(
                pattern="risk.kill_switch",
                topic=TelegramTopic.RISK,
                message_type=TelegramMessageType.RISK_KILL_SWITCH,
                category=TelegramEventCategory.RISK,
                priority=TelegramPriority.CRITICAL,
                level=TelegramNotificationLevel.CRITICAL,
                reason="risk kill switch",
            ),
            TelegramRoutingRule(
                pattern="risk.*",
                topic=TelegramTopic.RISK,
                message_type=TelegramMessageType.RISK_WARNING,
                category=TelegramEventCategory.RISK,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.WARNING,
                reason="generic risk event",
            ),

            # -----------------------------------------------------------------
            # System
            # -----------------------------------------------------------------
            TelegramRoutingRule(
                pattern="system.telegram_bot.healthcheck",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.HEALTHCHECK,
                category=TelegramEventCategory.SYSTEM,
                priority=TelegramPriority.LOW,
                level=TelegramNotificationLevel.INFO,
                reason="telegram bot healthcheck",
            ),
            TelegramRoutingRule(
                pattern="system.*.error",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.SYSTEM_ERROR,
                category=TelegramEventCategory.SYSTEM,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.ERROR,
                reason="system error",
            ),
            TelegramRoutingRule(
                pattern="system.*.failed",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.SYSTEM_ERROR,
                category=TelegramEventCategory.SYSTEM,
                priority=TelegramPriority.HIGH,
                level=TelegramNotificationLevel.ERROR,
                reason="system failure",
            ),
            TelegramRoutingRule(
                pattern="system.*.warning",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.SYSTEM_WARNING,
                category=TelegramEventCategory.SYSTEM,
                priority=TelegramPriority.NORMAL,
                level=TelegramNotificationLevel.WARNING,
                reason="system warning",
            ),
            TelegramRoutingRule(
                pattern="system.*",
                topic=TelegramTopic.SYSTEM,
                message_type=TelegramMessageType.SYSTEM_INFO,
                category=TelegramEventCategory.SYSTEM,
                priority=TelegramPriority.LOW,
                level=TelegramNotificationLevel.INFO,
                reason="generic system event",
            ),
        ]

    def resolve(self, event: TelegramEventPayload) -> TelegramTopicRoute:
        """
        Основний метод router-а.

        Повертає TelegramTopicRoute для EventBus-події.
        """

        event_name = event.metadata.event_name

        if not event_name:
            raise TelegramRoutingError(
                "Cannot route Telegram event without event_name.",
                details={"event": event.to_dict()},
            )

        if self._is_feature_disabled(event):
            return self._skipped_route(
                event=event,
                reason="telegram feature disabled by config",
            )

        if self._should_skip_non_actionable_analytics(event_name):
            return self._skipped_route(
                event=event,
                reason=f"non-actionable analytics event skipped: {event_name}",
            )

        if self._should_skip_diagnostic_signal_reject(event):
            return self._skipped_route(
                event=event,
                reason="diagnostic signal.rejected skipped",
            )

        rule = self._find_rule(event_name)

        if rule is None:
            return self._resolve_unknown(event)

        route = self._route_from_rule(event=event, rule=rule)
        return self._apply_payload_overrides(route=route, event=event)

    def add_rule(self, rule: TelegramRoutingRule) -> None:
        """
        Додає custom routing rule у кінець списку.

        Якщо потрібен вищий пріоритет — використовуй prepend_rule().
        """

        self._rules.append(rule)

    def prepend_rule(self, rule: TelegramRoutingRule) -> None:
        """
        Додає custom routing rule на початок списку.
        """

        self._rules.insert(0, rule)

    def replace_rules(self, rules: list[TelegramRoutingRule]) -> None:
        """
        Повністю замінює routing rules.
        """

        self._rules = list(rules)

    def list_rules(self) -> list[dict[str, Any]]:
        """
        Safe representation routing rules для stats/debug.
        """

        return [
            {
                "pattern": rule.pattern,
                "topic": rule.topic.value,
                "message_type": rule.message_type.value,
                "category": rule.category.value,
                "priority": rule.priority.value,
                "level": rule.level.value,
                "enabled": rule.enabled,
                "reason": rule.reason,
            }
            for rule in self._rules
        ]

    def topic_for_analytics_domain(self, domain: str | None) -> TelegramTopic:
        """
        Явний mapping analytics domain -> TelegramTopic.

        Використовується як fallback для analytics.* подій.
        """

        if not domain:
            return TelegramTopic.SYSTEM

        normalized = domain.strip().lower()

        mapping: dict[str, TelegramTopic] = {
            "orderflow": TelegramTopic.ORDERFLOW,
            "liquidity": TelegramTopic.LIQUIDITY,
            "price_action": TelegramTopic.PRICE_ACTION,
            "liquidations": TelegramTopic.LIQUIDATIONS,
            "whales": TelegramTopic.WHALES,
            "spoofing": TelegramTopic.SPOOFING,
            "spreads": TelegramTopic.SPREADS,
            "funding": TelegramTopic.FUNDING,
            "open_interest": TelegramTopic.OPEN_INTEREST,
            "oi": TelegramTopic.OPEN_INTEREST,
        }

        return mapping.get(normalized, TelegramTopic.SYSTEM)

    def category_for_event_name(self, event_name: str) -> TelegramEventCategory:
        """
        Визначає категорію за namespace event_name.
        """

        if not event_name:
            return TelegramEventCategory.UNKNOWN

        namespace = event_name.split(".", maxsplit=1)[0]

        mapping: dict[str, TelegramEventCategory] = {
            "analytics": TelegramEventCategory.ANALYTICS,
            "news": TelegramEventCategory.NEWS,
            "ai": TelegramEventCategory.AI,
            "signal": TelegramEventCategory.SIGNAL,
            "risk": TelegramEventCategory.RISK,
            "execution": TelegramEventCategory.EXECUTION,
            "position": TelegramEventCategory.POSITION,
            "system": TelegramEventCategory.SYSTEM,
        }

        return mapping.get(namespace, TelegramEventCategory.UNKNOWN)

    def _should_skip_non_actionable_analytics(self, event_name: str) -> bool:
        normalized = event_name.strip().lower()

        if not normalized.startswith("analytics."):
            return False

        if normalized in self._ACTIONABLE_ANALYTICS_EVENTS:
            return False

        if any(part in normalized for part in self._NON_ACTIONABLE_ANALYTICS_CONTAINS):
            return True

        if normalized.endswith(self._NON_ACTIONABLE_ANALYTICS_SUFFIXES):
            return True

        if any(fnmatch(normalized, pattern) for pattern in self._ACTIONABLE_ANALYTICS_PATTERNS):
            return False

        return True

    def _should_skip_diagnostic_signal_reject(self, event: TelegramEventPayload) -> bool:
        event_name = event.metadata.event_name.strip().lower()
        if event_name != "signal.rejected":
            return False

        payload = event.payload if isinstance(event.payload, dict) else {}
        reason = str(payload.get("reason") or payload.get("reject_reason") or "").strip().lower()
        if reason in self._DIAGNOSTIC_SIGNAL_REJECT_REASONS:
            return True

        source_event_name = str(
            payload.get("event_name")
            or payload.get("source_event_name")
            or payload.get("source_topic")
            or ""
        ).strip().lower()

        if not source_event_name and isinstance(payload.get("route"), dict):
            source_event_name = str(payload["route"].get("event_name") or "").strip().lower()

        raw_signals = payload.get("raw_signals")
        has_raw_signals = isinstance(raw_signals, list) and len(raw_signals) > 0

        if (
            source_event_name.startswith("analytics.")
            and self._should_skip_non_actionable_analytics(source_event_name)
            and not has_raw_signals
        ):
            return True

        return False

    def _find_rule(self, event_name: str) -> TelegramRoutingRule | None:
        for rule in self._rules:
            if rule.matches(event_name):
                return rule
        return None

    def _route_from_rule(
        self,
        *,
        event: TelegramEventPayload,
        rule: TelegramRoutingRule,
    ) -> TelegramTopicRoute:
        topic = rule.topic
        thread_id = self.config.get_topic_id(topic)

        if thread_id is None:
            return self._handle_missing_topic(
                event=event,
                topic=topic,
                message_type=rule.message_type,
                category=rule.category,
                priority=rule.priority,
                level=rule.level,
                reason=rule.reason,
            )

        return TelegramTopicRoute(
            topic=topic,
            message_type=rule.message_type,
            thread_id=thread_id,
            category=rule.category,
            priority=rule.priority,
            level=rule.level,
            reason=rule.reason,
        )

    def _apply_payload_overrides(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramTopicRoute:
        """
        Дозволяє payload обережно уточнити priority/level/topic.

        Наприклад analytics event може прийти з:
        {
            "telegram_priority": "high",
            "telegram_level": "warning"
        }

        Topic override підтримується тільки якщо topic існує в TelegramTopic.
        """

        payload = event.payload

        topic = route.topic
        thread_id = route.thread_id
        priority = route.priority
        level = route.level

        topic_override = payload.get("telegram_topic")
        if topic_override:
            parsed_topic = self._parse_topic(topic_override)
            if parsed_topic is not None:
                topic = parsed_topic
                thread_id = self.config.get_topic_id(topic)

                if thread_id is None:
                    return self._handle_missing_topic(
                        event=event,
                        topic=topic,
                        message_type=route.message_type,
                        category=route.category,
                        priority=route.priority,
                        level=route.level,
                        reason="payload telegram_topic override has no configured thread_id",
                    )

        priority_override = payload.get("telegram_priority")
        if priority_override:
            parsed_priority = self._parse_priority(priority_override)
            if parsed_priority is not None:
                priority = parsed_priority

        level_override = payload.get("telegram_level")
        if level_override:
            parsed_level = self._parse_level(level_override)
            if parsed_level is not None:
                level = parsed_level

        return TelegramTopicRoute(
            topic=topic,
            message_type=route.message_type,
            thread_id=thread_id,
            category=route.category,
            priority=priority,
            level=level,
            reason=route.reason,
        )

    def _resolve_unknown(self, event: TelegramEventPayload) -> TelegramTopicRoute:
        category = self.category_for_event_name(event.metadata.event_name)

        if self.config.route_policy == TelegramRoutePolicy.SKIP:
            return self._skipped_route(
                event=event,
                reason="unknown event skipped by route policy",
            )

        if self.config.route_policy == TelegramRoutePolicy.RAISE:
            raise TelegramRoutingError(
                "No Telegram routing rule found for event.",
                details={
                    "event_name": event.metadata.event_name,
                    "category": category.value,
                    "route_policy": self.config.route_policy.value,
                },
            )

        if self.config.route_policy == TelegramRoutePolicy.SEND_TO_DEFAULT:
            return self._default_route(
                event=event,
                category=category,
                reason="unknown event routed to default topic",
            )

        return self._system_route(
            event=event,
            category=category,
            reason="unknown event routed to system topic",
        )

    def _default_route(
        self,
        *,
        event: TelegramEventPayload,
        category: TelegramEventCategory,
        reason: str,
    ) -> TelegramTopicRoute:
        thread_id = self.config.default_topic_id

        if thread_id is None or thread_id <= 0:
            return self._system_route(
                event=event,
                category=category,
                reason="default topic is not configured; routed to system topic",
            )

        return TelegramTopicRoute(
            topic=TelegramTopic.SYSTEM,
            message_type=self._generic_message_type_for_category(category),
            thread_id=thread_id,
            category=category,
            priority=TelegramPriority.LOW,
            level=TelegramNotificationLevel.INFO,
            reason=reason,
        )

    def _system_route(
        self,
        *,
        event: TelegramEventPayload,
        category: TelegramEventCategory,
        reason: str,
    ) -> TelegramTopicRoute:
        thread_id = self.config.get_topic_id(TelegramTopic.SYSTEM)

        if thread_id is None:
            if self.config.route_policy == TelegramRoutePolicy.RAISE:
                raise TelegramTopicNotConfiguredError(
                    "Telegram SYSTEM topic is not configured.",
                    topic=TelegramTopic.SYSTEM.value,
                    event_name=event.metadata.event_name,
                )

            return self._skipped_route(
                event=event,
                reason="system topic is not configured",
            )

        return TelegramTopicRoute(
            topic=TelegramTopic.SYSTEM,
            message_type=self._generic_message_type_for_category(category),
            thread_id=thread_id,
            category=category,
            priority=TelegramPriority.LOW,
            level=TelegramNotificationLevel.INFO,
            reason=reason,
        )

    def _skipped_route(
        self,
        *,
        event: TelegramEventPayload,
        reason: str,
    ) -> TelegramTopicRoute:
        """
        Route для події, яку треба пропустити.

        thread_id=None означає: не відправляти.
        handlers.py зможе зафіксувати SKIPPED delivery.
        """

        category = self.category_for_event_name(event.metadata.event_name)

        return TelegramTopicRoute(
            topic=TelegramTopic.SYSTEM,
            message_type=self._generic_message_type_for_category(category),
            thread_id=None,
            category=category,
            priority=TelegramPriority.LOW,
            level=TelegramNotificationLevel.INFO,
            reason=reason,
        )

    def _handle_missing_topic(
        self,
        *,
        event: TelegramEventPayload,
        topic: TelegramTopic,
        message_type: TelegramMessageType,
        category: TelegramEventCategory,
        priority: TelegramPriority,
        level: TelegramNotificationLevel,
        reason: str | None,
    ) -> TelegramTopicRoute:
        if self.config.route_policy == TelegramRoutePolicy.RAISE:
            raise TelegramTopicNotConfiguredError(
                "Telegram topic is not configured.",
                topic=topic.value,
                event_name=event.metadata.event_name,
                details={
                    "message_type": message_type.value,
                    "category": category.value,
                    "route_policy": self.config.route_policy.value,
                },
            )

        if self.config.route_policy == TelegramRoutePolicy.SKIP:
            return TelegramTopicRoute(
                topic=topic,
                message_type=message_type,
                thread_id=None,
                category=category,
                priority=priority,
                level=level,
                reason=reason or "telegram topic is not configured; skipped",
            )

        if self.config.route_policy == TelegramRoutePolicy.SEND_TO_DEFAULT:
            default_topic_id = self.config.default_topic_id
            if default_topic_id is not None and default_topic_id > 0:
                return TelegramTopicRoute(
                    topic=topic,
                    message_type=message_type,
                    thread_id=default_topic_id,
                    category=category,
                    priority=priority,
                    level=level,
                    reason=reason or "telegram topic missing; routed to default topic id",
                )

        system_thread_id = self.config.get_topic_id(TelegramTopic.SYSTEM)
        if system_thread_id is not None:
            return TelegramTopicRoute(
                topic=TelegramTopic.SYSTEM,
                message_type=message_type,
                thread_id=system_thread_id,
                category=category,
                priority=priority,
                level=level,
                reason=reason or f"topic {topic.value} missing; routed to system topic",
            )

        return TelegramTopicRoute(
            topic=topic,
            message_type=message_type,
            thread_id=None,
            category=category,
            priority=priority,
            level=level,
            reason=reason or "telegram topic is not configured and no fallback exists",
        )

    def _generic_message_type_for_category(
        self,
        category: TelegramEventCategory,
    ) -> TelegramMessageType:
        if category == TelegramEventCategory.NEWS:
            return TelegramMessageType.NEWS_ALERT

        if category == TelegramEventCategory.ANALYTICS:
            return TelegramMessageType.ANALYTICS_ALERT

        if category == TelegramEventCategory.RISK:
            return TelegramMessageType.RISK_WARNING

        if category == TelegramEventCategory.SYSTEM:
            return TelegramMessageType.SYSTEM_INFO

        if category == TelegramEventCategory.POSITION:
            return TelegramMessageType.POSITION_UPDATED

        if category == TelegramEventCategory.EXECUTION:
            return TelegramMessageType.ORDER_SUBMITTED

        if category == TelegramEventCategory.SIGNAL:
            return TelegramMessageType.SIGNAL_UPDATED

        return TelegramMessageType.SYSTEM_INFO

    def _is_feature_disabled(self, event: TelegramEventPayload) -> bool:
        category = event.category
        event_name = event.metadata.event_name

        if category == TelegramEventCategory.UNKNOWN:
            category = self.category_for_event_name(event_name)

        if category == TelegramEventCategory.ANALYTICS:
            return not self.config.enable_analytics_alerts

        if category in {TelegramEventCategory.NEWS, TelegramEventCategory.AI}:
            return not self.config.enable_news_alerts

        if category == TelegramEventCategory.SIGNAL:
            return not self.config.enable_signal_alerts

        if category in {TelegramEventCategory.POSITION, TelegramEventCategory.EXECUTION}:
            return not self.config.enable_trade_updates

        if category == TelegramEventCategory.RISK:
            return not self.config.enable_risk_alerts

        if category == TelegramEventCategory.SYSTEM:
            return not self.config.enable_system_alerts

        return False

    def _parse_topic(self, value: Any) -> TelegramTopic | None:
        if isinstance(value, TelegramTopic):
            return value

        if value is None:
            return None

        normalized = str(value).strip().lower()
        if not normalized:
            return None

        for topic in TelegramTopic:
            if normalized in {topic.value.lower(), topic.name.lower()}:
                return topic

        return None

    def _parse_priority(self, value: Any) -> TelegramPriority | None:
        if isinstance(value, TelegramPriority):
            return value

        if value is None:
            return None

        normalized = str(value).strip().lower()
        if not normalized:
            return None

        for priority in TelegramPriority:
            if normalized in {priority.value.lower(), priority.name.lower()}:
                return priority

        return None

    def _parse_level(self, value: Any) -> TelegramNotificationLevel | None:
        if isinstance(value, TelegramNotificationLevel):
            return value

        if value is None:
            return None

        normalized = str(value).strip().lower()
        if not normalized:
            return None

        for level in TelegramNotificationLevel:
            if normalized in {level.value.lower(), level.name.lower()}:
                return level

        return None