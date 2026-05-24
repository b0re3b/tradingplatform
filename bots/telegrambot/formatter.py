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
    TelegramPriority,
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
        """
        Formats analytics events as compact domain-specific Telegram alerts.

        Analytics payloads in this project are intentionally nested and differ
        across domains. This formatter flattens the public payload and then
        renders only fields that are actually present. Missing values are not
        shown as n/a; irrelevant fields are omitted entirely.
        """

        raw_payload = self._sanitize_payload(event.payload)
        payload = self._analytics_view_payload(raw_payload)
        domain = (self._analytics_domain(event) or "analytics").strip().lower()
        event_name = event.metadata.event_name

        title = self._analytics_signal_title(domain=domain, event_name=event_name, payload=payload)
        sections = [f"<b>{title}</b>"]

        market_block = self._market_block_compact(payload)
        if market_block:
            sections.append(market_block)

        metric_block = self._analytics_metric_block(domain=domain, event_name=event_name, payload=payload)
        if metric_block:
            sections.append(metric_block)

        confidence_block = self._analytics_score_block(payload)
        if confidence_block:
            sections.append(confidence_block)

        summary = self._analytics_summary(domain, event_name, payload)
        if summary is not None and str(summary).strip():
            sections.append(f"<b>Summary:</b>\n{self._multiline(summary)}")

        sections.append(self._footer(event))

        return self._build_message(
            template="{body}",
            values={"body": "\n\n".join(sections)},
            route=route,
            title=title,
        )

    def format_news_event(
        self,
        *,
        route: TelegramTopicRoute,
        event: TelegramEventPayload,
    ) -> TelegramFormattedMessage:
        payload = self._sanitize_payload(event.payload)
        template = get_template_for_message_type(TelegramMessageType.NEWS_ALERT)

        # NewsAIService publishes nested payloads:
        # {"item": {...}, "score": {...}, "features": {...}, ...}.
        # Keep backward compatibility with older flat news payloads.
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        score = payload.get("score") if isinstance(payload.get("score"), dict) else {}
        features = payload.get("features") if isinstance(payload.get("features"), dict) else {}

        symbols = (
            payload.get("symbols")
            or item.get("symbols")
            or features.get("symbols")
            or features.get("mentioned_symbols")
        )

        source = (
            payload.get("source")
            or payload.get("source_name")
            or item.get("source_name")
            or item.get("source")
        )

        impact = (
            payload.get("impact")
            or payload.get("importance")
            or score.get("impact_level")
            or score.get("impact_score")
        )

        sentiment = (
            payload.get("sentiment")
            or score.get("sentiment")
            or score.get("market_bias")
        )

        headline = (
            payload.get("headline")
            or payload.get("title")
            or item.get("headline")
            or item.get("title")
            or "n/a"
        )

        summary = (
            payload.get("summary")
            or payload.get("description")
            or score.get("summary")
            or score.get("explanation")
            or item.get("summary")
            or item.get("description")
            or item.get("body")
            or "n/a"
        )

        values = self._base_values(event=event, route=route)
        values.update(
            {
                "source": self._safe(source),
                "impact": self._safe(impact),
                "sentiment": self._safe(sentiment),
                "symbols": self._format_list(symbols),
                "headline": self._multiline(headline),
                "summary": self._multiline(summary),
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
                "details": self._format_signal_details(payload, route.message_type),
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

    def _analytics_view_payload(self, payload: dict[str, Any]) -> dict[str, Any]:


        merged: dict[str, Any] = {}

        def merge_dict(value: Any) -> None:
            if isinstance(value, dict):
                merged.update(value)

        merge_dict(payload.get("payload"))
        merge_dict(payload.get("data"))
        merge_dict(payload.get("result"))
        merge_dict(payload.get("signal"))

        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}

        merge_dict(context.get("stats") if isinstance(context, dict) else None)
        merge_dict(stats)
        merge_dict(context)

        # Keep scope values available both as nested scope and flat fields.
        scope = self._first_present(payload, "scope", "context.scope", "stats.scope")
        if isinstance(scope, dict):
            for key in ("exchange", "market_type", "symbol", "timeframe", "exchange_symbol"):
                if key in scope and key not in merged:
                    merged[key] = scope[key]

        # Top-level fields must win because they are the public event contract.
        merged.update(payload)

        # Common semantic aliases used by formatter templates.
        if "cvd" not in merged:
            cvd = self._first_present(merged, "cvd_value", "cvd_close", "cumulative_volume_delta")
            if cvd is not None:
                merged["cvd"] = cvd

        if "delta" not in merged:
            delta = self._first_present(merged, "volume_delta", "net_volume_delta", "notional_delta", "delta_ratio")
            if delta is not None:
                merged["delta"] = delta

        if "score" not in merged and "strength" in merged:
            merged["score"] = merged["strength"]

        if "confidence" not in merged and "strength" in merged:
            merged["confidence"] = merged["strength"]

        if "pattern" not in merged:
            pattern = self._first_present(merged, "metric", "signal_type", "reason")
            if pattern is not None:
                merged["pattern"] = pattern

        return merged

    def _first_present(self, payload: dict[str, Any], *paths: str, default: Any = None) -> Any:
        for path in paths:
            value = self._get_path(payload, path)
            if value is not None and value != "":
                return value
        return default

    def _get_path(self, payload: dict[str, Any], path: str) -> Any:
        current: Any = payload
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            return None
        return current

    def _confidence_block_from_values(self, score: Any, confidence: Any) -> str:
        return (
            f"<b>Score:</b> <code>{self._format_number(score)}</code>\n"
            f"<b>Confidence:</b> <code>{self._format_number(confidence)}</code>"
        )

    def _analytics_summary(self, domain: str | None, event_name: str, payload: dict[str, Any]) -> Any:
        explicit = self._first_present(payload, "summary", "message", "description")
        if explicit is not None:
            return explicit

        normalized_domain = (domain or "").lower()
        if normalized_domain == "orderflow":
            reason = self._safe_plain(self._first_present(payload, "reason", "signal_type", "metric", default="orderflow signal"))
            parts = [reason.replace("_", " ")]

            last_price = self._first_present(payload, "last_price", "price", "mid_price")
            if last_price is not None:
                parts.append(f"last price {self._format_price(last_price)}")

            delta = self._first_present(payload, "volume_delta", "net_volume_delta", "delta")
            if delta is not None:
                parts.append(f"delta {self._format_number(delta)}")

            cvd = self._first_present(payload, "cvd_value", "cvd_close", "cvd", "cumulative_volume_delta")
            if cvd is not None:
                parts.append(f"CVD {self._format_number(cvd)}")

            delta_ratio = self._first_present(payload, "delta_ratio", "imbalance_ratio")
            if delta_ratio is not None:
                parts.append(f"ratio {self._format_number(delta_ratio)}")

            trades_count = self._first_present(payload, "trades_count")
            if trades_count is not None:
                parts.append(f"trades {self._format_number(trades_count)}")

            return " | ".join(parts)

        if normalized_domain == "liquidity":
            parts: list[str] = []
            bias = self._first_present(payload, "bias", "side", "direction")
            if bias is not None:
                parts.append(f"bias {self._safe_plain(bias)}")

            sweep_up = self._first_present(payload, "sweep_risk_up", "up_sweep_risk", "sweep_risk.up")
            sweep_down = self._first_present(payload, "sweep_risk_down", "down_sweep_risk", "sweep_risk.down")
            if sweep_up is not None:
                parts.append(f"sweep up {self._format_number(sweep_up)}")
            if sweep_down is not None:
                parts.append(f"sweep down {self._format_number(sweep_down)}")

            magnet_up = self._first_present(payload, "magnet_score_up", "up_magnet_score", "magnet_score.up")
            magnet_down = self._first_present(payload, "magnet_score_down", "down_magnet_score", "magnet_score.down")
            if magnet_up is not None:
                parts.append(f"magnet up {self._format_number(magnet_up)}")
            if magnet_down is not None:
                parts.append(f"magnet down {self._format_number(magnet_down)}")

            nearest_buy = self._first_present(payload, "nearest_buy_side_liquidity")
            nearest_sell = self._first_present(payload, "nearest_sell_side_liquidity")
            if nearest_buy is not None:
                parts.append(f"nearest buy-side {self._format_price(nearest_buy)}")
            if nearest_sell is not None:
                parts.append(f"nearest sell-side {self._format_price(nearest_sell)}")

            explanation = self._first_present(payload, "explanation")
            if explanation is not None:
                parts.append(self._safe_plain(explanation))

            if parts:
                return " | ".join(parts)

        reason = self._first_present(payload, "reason", "signal_type", "event_type")
        if reason is not None:
            return str(reason).replace("_", " ")

        # Do not turn a generic lifecycle/update topic into a fake summary.
        if normalized_domain == "liquidity" and event_name.endswith(".updated"):
            return None

        return event_name

    def _analytics_signal_title(self, *, domain: str, event_name: str, payload: dict[str, Any]) -> str:
        domain_title = {
            "orderflow": "📊 Orderflow Signal",
            "liquidity": "💧 Liquidity Signal",
            "price_action": "📈 Price Action Signal",
            "liquidations": "🔥 Liquidation Signal",
            "whales": "🐋 Whale Signal",
            "spoofing": "🎭 Spoofing Signal",
            "spreads": "🔁 Spread Signal",
            "funding": "💸 Funding Signal",
            "open_interest": "📉 Open Interest Signal",
        }.get(domain, "📊 Analytics Signal")

        event_tail = event_name.split("analytics.", 1)[-1] if event_name else ""
        detail = self._first_present(
            payload,
            "signal_name",
            "setup_type",
            "event_type",
            "signal_type",
            "metric",
            "pattern",
            "reason",
        )
        if detail is None and domain == "liquidity" and event_name.endswith(".signal.updated"):
            bias = self._first_present(payload, "bias", "side", "direction")
            if bias is not None:
                bias_text = str(bias).strip().lower()
                if bias_text and bias_text not in {"neutral", "none", "unknown", "flat", "0"}:
                    detail = f"{bias_text} bias"

        if detail is None and event_tail:
            parts = event_tail.split(".")
            if len(parts) > 1:
                fallback_detail = parts[-1] if parts[-1] != "signal" else parts[-2]
                # For liquidity signal.updated, "Updated" is a lifecycle verb,
                # not a useful alert type. Keep the title clean unless payload
                # provides a real signal detail.
                if not (domain == "liquidity" and fallback_detail == "updated"):
                    detail = fallback_detail

        if detail is None:
            return domain_title

        detail_text = str(detail).replace("_", " ").replace(".", " ").strip().title()
        if not detail_text or detail_text.lower() in domain_title.lower():
            return domain_title
        return f"{domain_title} · {self._safe(detail_text)}"

    def _analytics_metric_block(self, *, domain: str, event_name: str, payload: dict[str, Any]) -> str:
        normalized = domain.lower()
        if normalized == "oi":
            normalized = "open_interest"

        if normalized == "orderflow":
            specs = self._orderflow_specs(event_name)
        elif normalized == "liquidity":
            specs = self._liquidity_specs(event_name)
        elif normalized == "price_action":
            specs = self._price_action_specs(event_name)
        elif normalized == "liquidations":
            specs = self._liquidations_specs(event_name)
        elif normalized == "whales":
            specs = self._whales_specs(event_name)
        elif normalized == "spoofing":
            specs = self._spoofing_specs(event_name)
        elif normalized == "spreads":
            specs = self._spreads_specs(event_name)
        elif normalized == "funding":
            specs = self._funding_specs(event_name)
        elif normalized == "open_interest":
            specs = self._open_interest_specs(event_name)
        else:
            specs = self._generic_analytics_specs(event_name)

        lines = []
        seen_labels: set[str] = set()
        for label, paths, formatter in specs:
            if label in seen_labels:
                continue
            value = self._first_present(payload, *paths)
            line = self._analytics_line(label, value, formatter=formatter)
            if line:
                seen_labels.add(label)
                lines.append(line)

        return "\n".join(lines)

    def _analytics_score_block(self, payload: dict[str, Any]) -> str:
        specs = (
            ("Score", ("score", "quality_score", "signal_score", "setup_score", "confluence_score", "weighted_score"), "number"),
            ("Confidence", ("confidence", "signal_confidence", "setup_confidence", "confidence_score"), "number"),
            ("Strength", ("strength", "signal_strength", "level", "intensity"), "number_or_text"),
        )
        lines = []
        for label, paths, formatter in specs:
            value = self._first_present(payload, *paths)
            line = self._analytics_line(label, value, formatter=formatter)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _market_block_compact(self, payload: dict[str, Any]) -> str:
        specs = (
            ("Exchange", ("exchange", "scope.exchange", "contract.exchange"), "text"),
            ("Symbol", ("symbol", "scope.symbol", "contract.symbol"), "text"),
            ("Market", ("market_type", "scope.market_type", "contract.market_type", "category"), "text"),
            ("Timeframe", ("timeframe", "scope.timeframe", "contract.timeframe"), "text"),
        )
        lines = []
        for label, paths, formatter in specs:
            value = self._first_present(payload, *paths)
            line = self._analytics_line(label, value, formatter=formatter)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _analytics_line(self, label: str, value: Any, *, formatter: str = "text") -> str:
        if value is None or value == "":
            return ""

        if isinstance(value, (list, tuple, set)) and not value:
            return ""

        if isinstance(value, dict) and not value:
            return ""

        if formatter == "price":
            formatted = self._format_price(value, default="")
        elif formatter == "number":
            formatted = self._format_number(value, default="")
        elif formatter == "pct":
            formatted = self._format_pct(value, default="")
        elif formatter == "timestamp":
            formatted = self._format_timestamp(value)
            if formatted == "n/a":
                formatted = ""
        elif formatter == "bool":
            formatted = self._format_bool(value, default="")
        elif formatter == "list":
            formatted = self._format_list(value, default="")
        elif formatter == "liquidity_level":
            formatted = self._format_liquidity_level(value, default="")
        elif formatter == "number_or_text":
            formatted = self._format_number(value, default="")
            if not formatted:
                formatted = self._safe(value, default="")
        else:
            formatted = self._safe(value, default="")

        if not formatted:
            return ""
        return f"<b>{escape(label, quote=True)}:</b> <code>{formatted}</code>"

    def _format_bool(self, value: Any, *, default: str = "n/a") -> str:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return "yes" if value else "no"
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "detected"}:
            return "yes"
        if text in {"false", "0", "no", "n", "none", "not_detected"}:
            return "no"
        return self._safe(value, default=default)

    def _orderflow_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        event = event_name.lower()
        common = (
            ("Side", ("side", "direction", "bias"), "text"),
            ("Price", ("last_price", "price", "mid_price", "close"), "price"),
        )
        if "cvd" in event:
            return common + (
                ("CVD", ("cvd", "cvd_value", "cvd_close", "cumulative_volume_delta"), "number"),
                ("Volume Delta", ("volume_delta", "net_volume_delta", "delta"), "number"),
                ("Delta Ratio", ("delta_ratio", "volume_delta_ratio"), "number"),
                ("Buy Volume", ("buy_volume", "aggressive_buy_volume"), "number"),
                ("Sell Volume", ("sell_volume", "aggressive_sell_volume"), "number"),
                ("Trades", ("trades_count", "trade_count", "count"), "number"),
            )
        if "volume_delta" in event:
            return common + (
                ("Volume Delta", ("volume_delta", "net_volume_delta", "delta"), "number"),
                ("Delta Ratio", ("delta_ratio", "volume_delta_ratio"), "number"),
                ("Buy Volume", ("buy_volume", "aggressive_buy_volume"), "number"),
                ("Sell Volume", ("sell_volume", "aggressive_sell_volume"), "number"),
                ("Total Volume", ("total_volume", "volume"), "number"),
                ("Trades", ("trades_count", "trade_count", "count"), "number"),
            )
        if "aggressive" in event:
            return common + (
                ("Aggressive Buy", ("aggressive_buy_volume", "buy_volume"), "number"),
                ("Aggressive Sell", ("aggressive_sell_volume", "sell_volume"), "number"),
                ("Aggressor Imbalance", ("aggressor_imbalance", "imbalance_ratio", "delta_ratio"), "number"),
                ("Large Trades", ("large_trades_count", "large_trade_count"), "number"),
                ("Total Volume", ("total_volume", "volume"), "number"),
            )
        if "orderbook_imbalance" in event or "imbalance" in event:
            return common + (
                ("Best Bid", ("best_bid", "bid_price"), "price"),
                ("Best Ask", ("best_ask", "ask_price"), "price"),
                ("Spread", ("spread",), "price"),
                ("Bid Volume", ("bid_volume",), "number"),
                ("Ask Volume", ("ask_volume",), "number"),
                ("Imbalance Ratio", ("imbalance_ratio", "raw_imbalance_ratio"), "number"),
                ("Imbalance Diff", ("imbalance_diff",), "number"),
                ("Depth Levels", ("depth_levels_used", "levels_used"), "number"),
            )
        return common + (
            ("Pattern", ("pattern", "metric", "signal_type", "reason"), "text"),
            ("Volume Delta", ("volume_delta", "net_volume_delta", "delta"), "number"),
            ("CVD", ("cvd", "cvd_value", "cvd_close", "cumulative_volume_delta"), "number"),
            ("Volume", ("volume", "total_volume"), "number"),
        )

    def _liquidity_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Event", ("liquidity_event", "event_type", "type", "reason"), "text"),
            ("Bias", ("bias",), "text"),
            ("Side", ("side", "direction", "sweep_side"), "text"),
            ("Level", ("level", "price_level", "liquidity_level", "price"), "price"),
            ("Current Price", ("current_price", "last_price", "price", "mid_price"), "price"),
            ("Sweep", ("sweep_detected", "swept", "is_swept", "detected"), "bool"),
            ("Sweep Risk Up", ("sweep_risk_up", "up_sweep_risk", "sweep_risk.up"), "number"),
            ("Sweep Risk Down", ("sweep_risk_down", "down_sweep_risk", "sweep_risk.down"), "number"),
            ("Magnet Up", ("magnet_score_up", "up_magnet_score", "magnet_score.up"), "number"),
            ("Magnet Down", ("magnet_score_down", "down_magnet_score", "magnet_score.down"), "number"),
            ("Nearest Buy-Side", ("nearest_buy_side_liquidity",), "liquidity_level"),
            ("Nearest Sell-Side", ("nearest_sell_side_liquidity",), "liquidity_level"),
            ("Liquidity", ("liquidity", "liquidity_score", "liquidity_volume", "volume"), "number"),
            ("Stop Cluster", ("stop_cluster", "stop_cluster_price", "cluster_price"), "price"),
            ("Distance", ("distance", "distance_to_level", "distance_pct"), "number_or_text"),
        )

    def _price_action_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Structure", ("structure", "market_structure", "structure_type"), "text"),
            ("Pattern", ("pattern", "pattern_type", "event_type", "reason"), "text"),
            ("Trend", ("trend", "trend_direction", "direction", "bias"), "text"),
            ("Side", ("side", "signal_side"), "text"),
            ("Price", ("price", "current_price", "last_price", "close"), "price"),
            ("Level", ("level", "price_level", "support", "resistance"), "price"),
            ("Breakout", ("breakout", "breakout_detected"), "bool"),
            ("Retest", ("retest", "retest_detected"), "bool"),
        )

    def _liquidations_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Side", ("side", "liquidation_side", "direction"), "text"),
            ("Price", ("price", "avg_price", "weighted_avg_price", "cluster_price"), "price"),
            ("Liquidation Volume", ("liquidation_volume", "total_liquidation_volume", "volume", "size"), "number"),
            ("Notional", ("notional", "notional_value", "total_notional", "usd_value"), "number"),
            ("Cluster Price", ("cluster_price", "level"), "price"),
            ("Liquidations", ("liquidation_count", "count", "events_count"), "number"),
            ("Cascade Risk", ("cascade_risk", "risk", "risk_level", "cascade_score"), "number_or_text"),
            ("Exhaustion", ("exhaustion", "exhaustion_detected", "is_exhausted"), "bool"),
        )

    def _whales_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Action", ("whale_action", "action", "event_type", "type"), "text"),
            ("Side", ("side", "direction", "bias"), "text"),
            ("Price", ("price", "avg_price", "last_price"), "price"),
            ("Volume", ("volume", "total_volume", "whale_volume"), "number"),
            ("Notional", ("notional", "notional_value", "usd_value", "total_notional"), "number"),
            ("Whale Count", ("whale_count", "wallet_count", "participants", "count"), "number"),
            ("Pressure", ("pressure", "whale_pressure", "pressure_score"), "number_or_text"),
            ("Net Flow", ("net_flow", "net_volume", "net_notional"), "number"),
            ("Absorption", ("absorption", "absorption_detected"), "bool"),
        )

    def _spoofing_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Pattern", ("pattern", "pattern_type", "spoofing_type", "event_type"), "text"),
            ("Side", ("side", "direction"), "text"),
            ("Price", ("price", "level", "best_price"), "price"),
            ("Fake Liquidity", ("fake_liquidity", "fake_liquidity_score", "spoof_volume", "volume"), "number"),
            ("Layer Count", ("layer_count", "layers", "levels_count"), "number"),
            ("Bid Spoof Volume", ("bid_spoof_volume", "spoof_bid_volume"), "number"),
            ("Ask Spoof Volume", ("ask_spoof_volume", "spoof_ask_volume"), "number"),
            ("Pull Ratio", ("pull_ratio", "cancel_ratio"), "number"),
        )

    def _spreads_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Symbol", ("symbol", "base_symbol"), "text"),
            ("Base Exchange", ("base_exchange", "exchange_a", "long_exchange"), "text"),
            ("Quote Exchange", ("quote_exchange", "exchange_b", "short_exchange"), "text"),
            ("Spread", ("spread", "spread_value", "basis", "basis_value"), "number"),
            ("Spread %", ("spread_pct", "basis_pct", "spread_percent"), "pct"),
            ("Funding Adj. Basis", ("funding_adjusted_basis", "adjusted_basis"), "number"),
            ("Long Leg", ("long_leg", "long_exchange"), "text"),
            ("Short Leg", ("short_leg", "short_exchange"), "text"),
            ("Expected PnL", ("expected_pnl", "expected_profit", "edge"), "number"),
        )

    def _funding_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Funding Rate", ("funding_rate", "current_funding_rate"), "pct"),
            ("Predicted Rate", ("predicted_rate", "predicted_funding_rate", "forecast_funding_rate"), "pct"),
            ("Next Funding", ("next_funding_time", "next_funding_time_ms", "funding_time"), "timestamp"),
            ("Bias", ("bias", "direction", "side", "funding_bias"), "text"),
            ("Mark Price", ("mark_price", "price", "last_price"), "price"),
            ("Index Price", ("index_price",), "price"),
            ("Regime", ("regime", "funding_regime", "regime_name"), "text"),
            ("Pressure", ("pressure", "funding_pressure", "pressure_score"), "number_or_text"),
        )

    def _open_interest_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Open Interest", ("open_interest", "oi", "current_open_interest", "open_interest_value"), "number"),
            ("OI Delta", ("oi_delta", "open_interest_delta", "delta"), "number"),
            ("OI Delta %", ("oi_delta_pct", "open_interest_delta_pct", "delta_pct"), "pct"),
            ("Price", ("price", "last_price", "mark_price", "close"), "price"),
            ("Price Change %", ("price_change_pct", "price_pct_change", "change_pct"), "pct"),
            ("Regime", ("regime", "oi_regime", "regime_name"), "text"),
            ("Divergence", ("divergence", "divergence_type"), "text"),
            ("Anomaly", ("anomaly", "anomaly_type"), "text"),
            ("Interpretation", ("interpretation", "reason", "signal_type"), "text"),
        )

    def _generic_analytics_specs(self, event_name: str) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return (
            ("Signal", ("signal_name", "name", "event_type", "signal_type", "type"), "text"),
            ("Side", ("side", "direction", "bias"), "text"),
            ("Price", ("price", "last_price", "mark_price", "mid_price", "close"), "price"),
            ("Level", ("level", "price_level"), "price"),
            ("Volume", ("volume", "total_volume"), "number"),
            ("Notional", ("notional", "notional_value"), "number"),
        )

    @staticmethod
    def _safe_plain(value: Any, *, default: str = "n/a") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

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

        extracted = self._extract_price_value(value)
        if extracted is None:
            return default if isinstance(value, (dict, list, tuple, set)) else self._safe(value, default=default)

        try:
            number = float(extracted)
        except (TypeError, ValueError):
            return self._safe(extracted, default=default)

        if abs(number) >= 100:
            return f"{number:.2f}"

        if abs(number) >= 1:
            return f"{number:.4f}".rstrip("0").rstrip(".")

        return f"{number:.8f}".rstrip("0").rstrip(".")

    def _extract_price_value(self, value: Any) -> Any:
        """
        Extracts a numeric price from nested analytics objects.

        Liquidity analytics can publish full level objects under fields such as
        ``nearest_sell_side_liquidity``. Telegram messages should show the
        tradable price, not the whole serialized dictionary.
        """
        if value is None or value == "":
            return None

        if isinstance(value, dict):
            for key in (
                "price",
                "level",
                "price_level",
                "liquidity_level",
                "cluster_midpoint",
                "cluster_price",
                "midpoint",
            ):
                nested = value.get(key)
                if nested is not None and nested != "":
                    return self._extract_price_value(nested)
            return None

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            # Exchange levels often arrive as [price, size]. Full liquidity
            # level lists can also arrive; in both cases the first extractable
            # price is the most useful compact representation.
            for item in value:
                extracted = self._extract_price_value(item)
                if extracted is not None:
                    return extracted
            return None

        return value

    def _format_liquidity_level(self, value: Any, *, default: str = "n/a") -> str:
        """Format a liquidity level object without dumping its full payload."""
        if value is None or value == "":
            return default

        if not isinstance(value, dict):
            return self._format_price(value, default=default)

        price = self._format_price(value, default="")
        if not price:
            return default

        level_type = self._safe_plain(value.get("level_type"), default="").replace("_", " ")
        side = self._safe_plain(value.get("side"), default="").replace("_", "-")
        status = self._safe_plain(value.get("sweep_status") or value.get("status"), default="").replace("_", " ")
        confidence = self._format_number(value.get("confidence"), default="")

        prefix_parts = [part for part in (level_type, side) if part]
        text = f"{' '.join(prefix_parts)} @ {price}" if prefix_parts else price

        suffix_parts = []
        if status:
            suffix_parts.append(status)
        if confidence:
            suffix_parts.append(f"conf {confidence}")

        if suffix_parts:
            text = f"{text} ({', '.join(suffix_parts)})"

        return escape(text, quote=True)

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

    def _format_signal_details(self, payload: dict[str, Any], message_type: TelegramMessageType) -> str:
        """
        Формує деталі для signal events без дампу всього payload.

        Для SIGNAL_REJECTED витягує тільки корисні поля:
        strategy_name, source_topic/event_name, raw_signals count, reason.
        Повний payload (route з 135 features тощо) ніколи не дампується.
        """
        # Якщо в payload є явний details — використовуємо його
        explicit_details = payload.get("details")
        if explicit_details is not None and isinstance(explicit_details, (dict, str)):
            return self._format_details(explicit_details)

        if message_type == TelegramMessageType.SIGNAL_REJECTED:
            parts: dict[str, Any] = {}

            strategy = payload.get("strategy_name") or payload.get("strategy")
            if strategy:
                parts["strategy"] = strategy

            source = (
                payload.get("source_topic")
                or payload.get("source_event_name")
                or payload.get("event_name")
            )
            if source:
                parts["source"] = source

            reason = payload.get("reason") or payload.get("reject_reason")
            if reason:
                parts["reason"] = reason

            raw_signals = payload.get("raw_signals")
            if isinstance(raw_signals, list):
                parts["raw_signals_count"] = len(raw_signals)

            route = payload.get("route")
            if isinstance(route, dict):
                route_event = route.get("event_name")
                if route_event:
                    parts["route_event"] = route_event

            if not parts:
                return "n/a"

            return f"<pre>{self._format_json(parts)}</pre>"

        # Для інших signal types — стандартна поведінка, але без fallback на весь payload
        return "n/a"

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
            return "n/a"

        if isinstance(timestamp_ms, str):
            text = timestamp_ms.strip()
            if not text:
                return "n/a"
            # Already ISO-like timestamp. Keep it instead of trying to parse as float.
            if "T" in text or "-" in text:
                return self._safe(text)

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