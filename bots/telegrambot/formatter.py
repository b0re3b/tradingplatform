"""
Telegram bot package formatter.

Formatter layer для Telegram notification service.

Цей модуль:
- не викликає Telegram API;
- не підписується на EventBus;
- не містить торгової бізнес-логіки;
- не читає market data напряму;
- тільки перетворює TelegramEventPayload + TelegramTopicRoute
  у TelegramFormattedMessage.

Важливо:
- Основний parse_mode: HTML.
- Усі значення payload екрануються через html.escape().
- bot_token/secrets ніколи не форматуються в повідомлення.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Any

from .config import TelegramBotConfig
from .enums import (
    TelegramEventCategory,
    TelegramMessageType,
    TelegramNotificationLevel,
    TelegramParseMode,
    TelegramPriority,
    TelegramTopic,
    TelegramTradeResult,
)
from .exceptions import (
    TelegramFormattingError,
    TelegramMessageTooLongError,
    TelegramPayloadError,
    TelegramTemplateError,
)
from .models import (
    TelegramEventPayload,
    TelegramFormattedMessage,
    TelegramMessageChunk,
    TelegramTopicRoute,
)
from .templates import (
    MESSAGE_CHUNK_PREFIX_TEMPLATE,
    get_template_for_analytics_domain,
    get_template_for_message_type,
)


SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "token",
        "bot_token",
        "api_key",
        "api_secret",
        "secret",
        "password",
        "passphrase",
        "authorization",
        "auth",
        "private_key",
        "signature",
    }
)


@dataclass(slots=True)
class TelegramFormatter:
    """
    Основний formatter для Telegram notification layer.

    Отримує:
    - route: TelegramTopicRoute;
    - event: TelegramEventPayload;

    Повертає:
    - TelegramFormattedMessage.
    """

    config: TelegramBotConfig

    def format(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        """
        Форматує EventBus event у TelegramFormattedMessage.
        """

        try:
            if route.message_type == TelegramMessageType.ANALYTICS_ALERT:
                return self.format_analytics_event(route=route, event=event)

            if route.message_type == TelegramMessageType.NEWS_ALERT:
                return self.format_news_event(route=route, event=event)

            if route.message_type in {
                TelegramMessageType.SIGNAL_GENERATED,
                TelegramMessageType.SIGNAL_CONFIRMED,
                TelegramMessageType.SIGNAL_REJECTED,
                TelegramMessageType.SIGNAL_UPDATED,
            }:
                return self.format_signal_event(route=route, event=event)

            if route.message_type in {
                TelegramMessageType.ORDER_SUBMITTED,
                TelegramMessageType.ORDER_FILLED,
                TelegramMessageType.ORDER_REJECTED,
                TelegramMessageType.ORDER_CANCELLED,
            }:
                return self.format_order_event(route=route, event=event)

            if route.message_type in {
                TelegramMessageType.POSITION_OPENED,
                TelegramMessageType.POSITION_UPDATED,
                TelegramMessageType.POSITION_CLOSED,
            }:
                return self.format_position_event(route=route, event=event)

            if route.message_type in {
                TelegramMessageType.RISK_WARNING,
                TelegramMessageType.RISK_BLOCKED,
                TelegramMessageType.RISK_KILL_SWITCH,
            }:
                return self.format_risk_event(route=route, event=event)

            if route.message_type in {
                TelegramMessageType.SYSTEM_INFO,
                TelegramMessageType.SYSTEM_WARNING,
                TelegramMessageType.SYSTEM_ERROR,
                TelegramMessageType.HEALTHCHECK,
            }:
                return self.format_system_event(route=route, event=event)

            return self.format_generic_event(route=route, event=event)

        except TelegramFormattingError:
            raise
        except KeyError as exc:
            raise TelegramPayloadError(
                "Telegram event payload is missing required field.",
                details={
                    "event_name": event.metadata.event_name,
                    "missing_key": str(exc),
                    "message_type": route.message_type.value,
                },
                cause=exc,
            ) from exc
        except Exception as exc:
            raise TelegramFormattingError(
                "Failed to format Telegram message.",
                details={
                    "event_name": event.metadata.event_name,
                    "message_type": route.message_type.value,
                    "topic": route.topic.value,
                },
                cause=exc,
            ) from exc

    def format_analytics_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        domain = self._analytics_domain(event)
        template = get_template_for_analytics_domain(domain)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "title": self._safe(payload.get("title", self._analytics_title(domain))),
                "market_block": self._market_block(payload),
                "confidence_block": self._confidence_block(payload),
                "exchange": self._safe(payload.get("exchange")),
                "symbol": self._safe(payload.get("symbol")),
                "market_type": self._safe(payload.get("market_type", "futures")),
                "timeframe": self._safe(payload.get("timeframe")),
                "side": self._safe(payload.get("side", payload.get("direction", "n/a"))),
                "signal_name": self._safe(
                    payload.get("signal_name", payload.get("name", "analytics"))
                ),
                "strength": self._safe(payload.get("strength", payload.get("level", "n/a"))),
                "score": self._format_number(payload.get("score")),
                "confidence": self._format_number(payload.get("confidence")),
                "summary": self._multiline(payload.get("summary", payload.get("message", "n/a"))),
                # domain-specific placeholders
                "pattern": self._safe(payload.get("pattern")),
                "delta": self._format_number(payload.get("delta")),
                "cvd": self._format_number(payload.get("cvd")),
                "absorption": self._safe(payload.get("absorption", payload.get("absorption_detected"))),
                "liquidity_event": self._safe(payload.get("liquidity_event", payload.get("event_type"))),
                "level": self._format_price(payload.get("level", payload.get("price_level"))),
                "sweep_detected": self._safe(payload.get("sweep_detected")),
                "structure": self._safe(payload.get("structure")),
                "trend": self._safe(payload.get("trend")),
                "liquidation_volume": self._format_number(payload.get("liquidation_volume")),
                "cluster_price": self._format_price(payload.get("cluster_price")),
                "cascade_risk": self._safe(payload.get("cascade_risk")),
                "whale_action": self._safe(payload.get("whale_action", payload.get("action"))),
                "volume": self._format_number(payload.get("volume")),
                "notional": self._format_number(payload.get("notional")),
                "fake_liquidity": self._format_number(payload.get("fake_liquidity")),
                "base_exchange": self._safe(payload.get("base_exchange")),
                "quote_exchange": self._safe(payload.get("quote_exchange")),
                "spread": self._format_number(payload.get("spread")),
                "spread_pct": self._format_pct(payload.get("spread_pct")),
                "funding_rate": self._format_pct(payload.get("funding_rate")),
                "predicted_rate": self._format_pct(payload.get("predicted_rate")),
                "next_funding_time": self._format_timestamp(payload.get("next_funding_time_ms")),
                "bias": self._safe(payload.get("bias")),
                "open_interest": self._format_number(payload.get("open_interest")),
                "oi_delta": self._format_number(payload.get("oi_delta")),
                "oi_delta_pct": self._format_pct(payload.get("oi_delta_pct")),
                "price_change_pct": self._format_pct(payload.get("price_change_pct")),
                "interpretation": self._safe(payload.get("interpretation")),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=self._analytics_title(domain),
        )

    def format_news_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(TelegramMessageType.NEWS_ALERT)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "source": self._safe(payload.get("source")),
                "impact": self._safe(payload.get("impact", payload.get("importance", "n/a"))),
                "sentiment": self._safe(payload.get("sentiment")),
                "symbols": self._format_list(payload.get("symbols")),
                "headline": self._multiline(payload.get("headline", payload.get("title", "n/a"))),
                "summary": self._multiline(payload.get("summary", payload.get("description", "n/a"))),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title="News Alert",
        )

    def format_signal_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "market_block": self._market_block(payload),
                "price_levels_block": self._price_levels_block(payload),
                "exchange": self._safe(payload.get("exchange")),
                "symbol": self._safe(payload.get("symbol")),
                "market_type": self._safe(payload.get("market_type", "futures")),
                "timeframe": self._safe(payload.get("timeframe")),
                "strategy_name": self._safe(payload.get("strategy_name", payload.get("strategy"))),
                "side": self._safe(payload.get("side")),
                "signal_type": self._safe(payload.get("signal_type")),
                "score": self._format_number(payload.get("score")),
                "entry_price": self._format_price(payload.get("entry_price", payload.get("entry"))),
                "stop_loss": self._format_price(payload.get("stop_loss", payload.get("sl"))),
                "take_profit": self._format_price(payload.get("take_profit", payload.get("tp"))),
                "risk_reward": self._format_number(payload.get("risk_reward", payload.get("rr"))),
                "reason": self._multiline(payload.get("reason", payload.get("summary", "n/a"))),
                "risk_pct": self._format_pct(payload.get("risk_pct")),
                "position_size": self._format_number(payload.get("position_size", payload.get("size"))),
                "risk_decision": self._multiline(payload.get("risk_decision", payload.get("decision", "n/a"))),
                "reject_reason": self._safe(payload.get("reject_reason", payload.get("reason"))),
                "details": self._format_details(payload.get("details", payload)),
                "signal_id": self._safe(payload.get("signal_id")),
                "status": self._safe(payload.get("status")),
                "update_type": self._safe(payload.get("update_type")),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def format_order_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "market_block": self._market_block(payload),
                "exchange": self._safe(payload.get("exchange")),
                "symbol": self._safe(payload.get("symbol")),
                "market_type": self._safe(payload.get("market_type", "futures")),
                "timeframe": self._safe(payload.get("timeframe")),
                "order_id": self._safe(payload.get("order_id")),
                "client_order_id": self._safe(payload.get("client_order_id")),
                "side": self._safe(payload.get("side")),
                "order_type": self._safe(payload.get("order_type", payload.get("type"))),
                "quantity": self._format_number(payload.get("quantity", payload.get("qty"))),
                "price": self._format_price(payload.get("price")),
                "filled_quantity": self._format_number(payload.get("filled_quantity", payload.get("filled_qty"))),
                "avg_price": self._format_price(payload.get("avg_price", payload.get("average_price"))),
                "fee": self._format_number(payload.get("fee")),
                "reject_reason": self._safe(payload.get("reject_reason", payload.get("reason"))),
                "cancel_reason": self._safe(payload.get("cancel_reason", payload.get("reason"))),
                "details": self._format_details(payload.get("details", payload)),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def format_position_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        realized_pnl = payload.get("realized_pnl", payload.get("pnl"))
        realized_pnl_pct = payload.get("realized_pnl_pct", payload.get("pnl_pct"))
        trade_result = self._trade_result(realized_pnl=realized_pnl, realized_pnl_pct=realized_pnl_pct)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "market_block": self._market_block(payload),
                "exchange": self._safe(payload.get("exchange")),
                "symbol": self._safe(payload.get("symbol")),
                "market_type": self._safe(payload.get("market_type", "futures")),
                "timeframe": self._safe(payload.get("timeframe")),
                "position_id": self._safe(payload.get("position_id")),
                "strategy_name": self._safe(payload.get("strategy_name", payload.get("strategy"))),
                "side": self._safe(payload.get("side")),
                "entry_price": self._format_price(payload.get("entry_price", payload.get("entry"))),
                "exit_price": self._format_price(payload.get("exit_price", payload.get("exit"))),
                "position_size": self._format_number(payload.get("position_size", payload.get("size"))),
                "leverage": self._format_leverage(payload.get("leverage")),
                "stop_loss": self._format_price(payload.get("stop_loss", payload.get("sl"))),
                "take_profit": self._format_price(payload.get("take_profit", payload.get("tp"))),
                "risk_pct": self._format_pct(payload.get("risk_pct")),
                "reason": self._multiline(payload.get("reason", payload.get("summary", "n/a"))),
                "status": self._safe(payload.get("status")),
                "mark_price": self._format_price(payload.get("mark_price")),
                "unrealized_pnl": self._format_number(payload.get("unrealized_pnl")),
                "unrealized_pnl_pct": self._format_pct(payload.get("unrealized_pnl_pct")),
                "details": self._format_details(payload.get("details", payload)),
                "realized_pnl": self._format_number(realized_pnl),
                "realized_pnl_pct": self._format_pct(realized_pnl_pct),
                "trade_result": self._safe(trade_result.value),
                "close_reason": self._safe(payload.get("close_reason", payload.get("reason"))),
                "summary": self._multiline(payload.get("summary", payload.get("reason", "n/a"))),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def format_risk_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "market_block": self._market_block(payload),
                "risk_type": self._safe(payload.get("risk_type", payload.get("type"))),
                "severity": self._safe(payload.get("severity", route.level.value)),
                "symbol": self._safe(payload.get("symbol")),
                "strategy_name": self._safe(payload.get("strategy_name", payload.get("strategy"))),
                "message": self._multiline(payload.get("message", payload.get("reason", "n/a"))),
                "details": self._format_details(payload.get("details", payload)),
                "exchange": self._safe(payload.get("exchange")),
                "market_type": self._safe(payload.get("market_type", "futures")),
                "timeframe": self._safe(payload.get("timeframe")),
                "side": self._safe(payload.get("side")),
                "block_reason": self._safe(payload.get("block_reason", payload.get("reason"))),
                "risk_rule": self._safe(payload.get("risk_rule", payload.get("rule"))),
                "status": self._safe(payload.get("status")),
                "reason": self._safe(payload.get("reason")),
                "scope": self._safe(payload.get("scope")),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def format_system_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "service": self._safe(payload.get("service", payload.get("source", event.metadata.source))),
                "status": self._safe(payload.get("status", "n/a")),
                "message": self._multiline(payload.get("message", payload.get("summary", "n/a"))),
                "details": self._format_details(payload.get("details", payload)),
                "error_type": self._safe(payload.get("error_type", payload.get("type"))),
                "latency_ms": self._format_number(payload.get("latency_ms")),
                "sent_messages": self._format_number(payload.get("sent_messages")),
                "failed_messages": self._format_number(payload.get("failed_messages")),
                "success_rate": self._format_pct(payload.get("success_rate")),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def format_generic_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(route.message_type)

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "title": self._safe(route.message_type.value.replace("_", " ").title()),
                "category": self._safe(event.category.value),
                "message_type": self._safe(route.message_type.value),
                "payload": self._format_json(payload),
            }
        )

        return self._build_message(
            template=template,
            values=values,
            route=route,
            title=route.message_type.value,
        )

    def split_message(
        self,
        message: TelegramFormattedMessage,
        *,
        max_length: int | None = None,
    ) -> list[TelegramMessageChunk]:
        """
        Розбиває довге повідомлення на chunks.

        HTML-aware split тут не повний парсер, але ми стараємось різати по рядках.
        Formatter уже екранує значення, тому ризик зламати user content мінімальний.
        """

        limit = max_length or self.config.max_message_length

        if limit <= 0:
            raise TelegramFormattingError(
                "Telegram max message length must be positive.",
                details={"max_length": limit},
            )

        if len(message.text) <= limit:
            return [TelegramMessageChunk(index=1, total=1, text=message.text)]

        if not self.config.split_long_messages:
            raise TelegramMessageTooLongError(
                "Telegram message is too long and split_long_messages is disabled.",
                details={
                    "message_type": message.message_type.value,
                    "topic": message.topic.value,
                    "length": len(message.text),
                    "max_length": limit,
                },
            )

        chunks: list[str] = []
        current = ""

        for line in message.text.splitlines(keepends=True):
            if len(line) > limit:
                if current:
                    chunks.append(current.rstrip())
                    current = ""

                for part in self._split_long_line(line, max_length=limit):
                    chunks.append(part.rstrip())
                continue

            if len(current) + len(line) > limit:
                chunks.append(current.rstrip())
                current = line
            else:
                current += line

        if current.strip():
            chunks.append(current.rstrip())

        total = len(chunks)
        title = message.title or message.message_type.value

        result: list[TelegramMessageChunk] = []
        for index, text in enumerate(chunks, start=1):
            prefix = MESSAGE_CHUNK_PREFIX_TEMPLATE.format(
                title=escape(str(title)),
                index=index,
                total=total,
            )
            available = limit - len(prefix)

            if available <= 0:
                raise TelegramFormattingError(
                    "Telegram chunk prefix is longer than max message length.",
                    details={"max_length": limit, "prefix_length": len(prefix)},
                )

            chunk_text = text
            if len(chunk_text) > available:
                chunk_text = chunk_text[:available].rstrip()

            result.append(
                TelegramMessageChunk(
                    index=index,
                    total=total,
                    text=f"{prefix}{chunk_text}",
                )
            )

        return result

    def _build_message(
        self,
        *,
        template: str,
        values: dict[str, Any],
        route: TelegramTopicRoute,
        title: str | None = None,
    ) -> TelegramFormattedMessage:
        try:
            text = template.format(**values).strip()
        except KeyError as exc:
            raise TelegramTemplateError(
                "Telegram template placeholder is missing.",
                details={
                    "missing_placeholder": str(exc),
                    "message_type": route.message_type.value,
                    "topic": route.topic.value,
                },
                cause=exc,
            ) from exc

        if not text:
            raise TelegramFormattingError(
                "Telegram formatted text is empty.",
                details={
                    "message_type": route.message_type.value,
                    "topic": route.topic.value,
                },
            )

        message = TelegramFormattedMessage(
            text=text,
            topic=route.topic,
            message_type=route.message_type,
            parse_mode=self.config.parse_mode,
            level=route.level,
            priority=route.priority,
            title=title,
            disable_web_page_preview=self.config.disable_web_page_preview,
            disable_notification=self._disable_notification(route),
            protect_content=self.config.protect_content,
            extra={
                "route_reason": route.reason,
                "category": route.category.value,
            },
        )

        if len(message.text) > self.config.max_message_length and not self.config.split_long_messages:
            raise TelegramMessageTooLongError(
                "Telegram formatted message exceeds max length.",
                details={
                    "message_type": route.message_type.value,
                    "topic": route.topic.value,
                    "length": len(message.text),
                    "max_length": self.config.max_message_length,
                },
            )

        return message

    def _base_values(
        self,
        *,
        event: TelegramEventPayload,
        route: TelegramTopicRoute,
    ) -> dict[str, Any]:
        return {
            "event_name": self._safe(event.metadata.event_name),
            "timestamp": self._format_timestamp(event.metadata.timestamp_ms),
            "footer": self._footer(event),
            "category": self._safe(event.category.value),
            "message_type": self._safe(route.message_type.value),
        }

    def _footer(self, event: TelegramEventPayload) -> str:
        event_name = self._safe(event.metadata.event_name)
        timestamp = self._format_timestamp(event.metadata.timestamp_ms)
        return (
            f"<i>event:</i> <code>{event_name}</code>\n"
            f"<i>time:</i> <code>{timestamp}</code>"
        )

    def _market_block(self, payload: dict[str, Any]) -> str:
        exchange = self._safe(payload.get("exchange"))
        symbol = self._safe(payload.get("symbol"))
        market_type = self._safe(payload.get("market_type", "futures"))
        timeframe = self._safe(payload.get("timeframe"))

        return (
            f"<b>Exchange:</b> <code>{exchange}</code>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Market:</b> <code>{market_type}</code>\n"
            f"<b>Timeframe:</b> <code>{timeframe}</code>"
        )

    def _confidence_block(self, payload: dict[str, Any]) -> str:
        score = self._format_number(payload.get("score"))
        confidence = self._format_number(payload.get("confidence"))

        return (
            f"<b>Score:</b> <code>{score}</code>\n"
            f"<b>Confidence:</b> <code>{confidence}</code>"
        )

    def _price_levels_block(self, payload: dict[str, Any]) -> str:
        entry_price = self._format_price(payload.get("entry_price", payload.get("entry")))
        stop_loss = self._format_price(payload.get("stop_loss", payload.get("sl")))
        take_profit = self._format_price(payload.get("take_profit", payload.get("tp")))
        risk_reward = self._format_number(payload.get("risk_reward", payload.get("rr")))

        return (
            f"<b>Entry:</b> <code>{entry_price}</code>\n"
            f"<b>Stop Loss:</b> <code>{stop_loss}</code>\n"
            f"<b>Take Profit:</b> <code>{take_profit}</code>\n"
            f"<b>Risk/Reward:</b> <code>{risk_reward}</code>"
        )

    def _analytics_domain(self, event: TelegramEventPayload) -> str | None:
        payload_domain = event.payload.get("domain") or event.payload.get("analytics_domain")
        if payload_domain:
            return str(payload_domain)

        parts = event.metadata.event_name.split(".")
        if len(parts) >= 2 and parts[0] == TelegramEventCategory.ANALYTICS.value:
            return parts[1]

        if event.category == TelegramEventCategory.ANALYTICS and len(parts) >= 2:
            return parts[1]

        return None

    def _analytics_title(self, domain: str | None) -> str:
        if not domain:
            return "Analytics Alert"

        return f"{domain.replace('_', ' ').title()} Alert"

    def _disable_notification(self, route: TelegramTopicRoute) -> bool:
        if self.config.disable_notification:
            return True

        return route.priority in {TelegramPriority.LOW, TelegramPriority.NORMAL} and route.level not in {
            TelegramNotificationLevel.ERROR,
            TelegramNotificationLevel.CRITICAL,
        }

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Видаляє/маскує sensitive keys перед форматуванням.
        """

        return {
            key: self._sanitize_value(key=key, value=value)
            for key, value in payload.items()
        }

    def _sanitize_value(self, *, key: str, value: Any) -> Any:
        normalized_key = key.lower()

        if any(sensitive in normalized_key for sensitive in SENSITIVE_KEYS):
            return "***"

        if isinstance(value, dict):
            return {
                nested_key: self._sanitize_value(key=str(nested_key), value=nested_value)
                for nested_key, nested_value in value.items()
            }

        if isinstance(value, list):
            return [
                self._sanitize_value(key=key, value=item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._sanitize_value(key=key, value=item)
                for item in value
            )

        return value

    def _safe(self, value: Any, *, default: str = "n/a") -> str:
        if value is None:
            return default

        if isinstance(value, bool):
            return "yes" if value else "no"

        text = str(value).strip()
        if not text:
            return default

        return escape(text, quote=True)

    def _multiline(self, value: Any, *, default: str = "n/a") -> str:
        if value is None:
            return default

        if isinstance(value, (dict, list, tuple)):
            return f"<pre>{self._format_json(value)}</pre>"

        text = str(value).strip()
        if not text:
            return default

        return escape(text, quote=True)

    def _format_number(self, value: Any, *, default: str = "n/a") -> str:
        if value is None or value == "":
            return default

        try:
            number = float(value)
        except (TypeError, ValueError):
            return self._safe(value, default=default)

        if number.is_integer():
            return f"{int(number)}"

        return f"{number:.6f}".rstrip("0").rstrip(".")

    def _format_price(self, value: Any, *, default: str = "n/a") -> str:
        if value is None or value == "":
            return default

        try:
            number = float(value)
        except (TypeError, ValueError):
            return self._safe(value, default=default)

        if abs(number) >= 100:
            return f"{number:.2f}"

        if abs(number) >= 1:
            return f"{number:.4f}".rstrip("0").rstrip(".")

        return f"{number:.8f}".rstrip("0").rstrip(".")

    def _format_pct(self, value: Any, *, default: str = "n/a") -> str:
        if value is None or value == "":
            return default

        try:
            number = float(value)
        except (TypeError, ValueError):
            return self._safe(value, default=default)

        # Якщо прийшло 0.015 — трактуємо як 1.5%.
        # Якщо прийшло 1.5 — залишаємо як 1.5%.
        pct = number * 100 if abs(number) <= 1 else number
        return f"{pct:.2f}%"

    def _format_leverage(self, value: Any, *, default: str = "n/a") -> str:
        if value is None or value == "":
            return default

        try:
            number = float(value)
        except (TypeError, ValueError):
            return self._safe(value, default=default)

        if number.is_integer():
            return f"{int(number)}x"

        return f"{number:.2f}x"

    def _format_list(self, value: Any, *, default: str = "n/a") -> str:
        if value is None:
            return default

        if isinstance(value, str):
            return self._safe(value, default=default)

        if isinstance(value, (list, tuple, set)):
            items = [self._safe(item) for item in value if item is not None]
            return ", ".join(items) if items else default

        return self._safe(value, default=default)

    def _format_details(self, value: Any, *, default: str = "n/a") -> str:
        if value is None:
            return default

        if isinstance(value, str):
            return self._multiline(value, default=default)

        return f"<pre>{self._format_json(value)}</pre>"

    def _format_json(self, value: Any) -> str:
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                default=str,
                sort_keys=True,
            )
        except TypeError:
            text = str(value)

        return escape(text, quote=True)

    def _format_timestamp(self, timestamp_ms: Any) -> str:
        if timestamp_ms is None or timestamp_ms == "":
            return datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            ts = float(timestamp_ms)
        except (TypeError, ValueError):
            return self._safe(timestamp_ms)

        # Якщо timestamp у секундах, а не ms.
        if ts < 10_000_000_000:
            ts *= 1000

        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.isoformat(timespec="seconds")

    def _trade_result(
        self,
        *,
        realized_pnl: Any,
        realized_pnl_pct: Any,
    ) -> TelegramTradeResult:
        value = realized_pnl_pct
        if value is None:
            value = realized_pnl

        if value is None or value == "":
            return TelegramTradeResult.UNKNOWN

        try:
            number = float(value)
        except (TypeError, ValueError):
            return TelegramTradeResult.UNKNOWN

        if number > 0:
            return TelegramTradeResult.WIN

        if number < 0:
            return TelegramTradeResult.LOSS

        return TelegramTradeResult.BREAKEVEN

    def _split_long_line(self, line: str, *, max_length: int) -> list[str]:
        if max_length <= 0:
            raise TelegramFormattingError(
                "max_length must be positive for Telegram message splitting.",
                details={"max_length": max_length},
            )

        return [
            line[index : index + max_length]
            for index in range(0, len(line), max_length)
        ]