"""
Telegram bot package templates.

HTML-шаблони повідомлень для Telegram notification layer.

Цей модуль:
- не викликає Telegram API;
- не підписується на EventBus;
- не містить торгової бізнес-логіки;
- містить тільки шаблони та lightweight template registry.

Важливо:
- Основний parse_mode у нас HTML.
- Значення payload перед підстановкою має екранувати formatter.py.
- Не вставляти сирий user/API/token data без escaping.
"""

from __future__ import annotations

from types import MappingProxyType

from .enums import TelegramMessageType


# =============================================================================
# Shared blocks
# =============================================================================

MESSAGE_FOOTER_TEMPLATE = """
<i>event:</i> <code>{event_name}</code>
<i>time:</i> <code>{timestamp}</code>
""".strip()


CONFIDENCE_BLOCK_TEMPLATE = """
<b>Score:</b> <code>{score}</code>
<b>Confidence:</b> <code>{confidence}</code>
""".strip()


MARKET_BLOCK_TEMPLATE = """
<b>Exchange:</b> <code>{exchange}</code>
<b>Symbol:</b> <code>{symbol}</code>
<b>Market:</b> <code>{market_type}</code>
<b>Timeframe:</b> <code>{timeframe}</code>
""".strip()


PRICE_LEVELS_BLOCK_TEMPLATE = """
<b>Entry:</b> <code>{entry_price}</code>
<b>Stop Loss:</b> <code>{stop_loss}</code>
<b>Take Profit:</b> <code>{take_profit}</code>
<b>Risk/Reward:</b> <code>{risk_reward}</code>
""".strip()


# =============================================================================
# Analytics templates
# =============================================================================

ANALYTICS_ALERT_TEMPLATE = """
<b>📊 {title}</b>

{market_block}

<b>Direction:</b> <code>{side}</code>
<b>Signal:</b> <code>{signal_name}</code>
<b>Strength:</b> <code>{strength}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


ORDERFLOW_ALERT_TEMPLATE = """
<b>📊 Orderflow Alert</b>

{market_block}

<b>Pattern:</b> <code>{pattern}</code>
<b>Side:</b> <code>{side}</code>
<b>Delta:</b> <code>{delta}</code>
<b>CVD:</b> <code>{cvd}</code>
<b>Absorption:</b> <code>{absorption}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


LIQUIDITY_ALERT_TEMPLATE = """
<b>💧 Liquidity Alert</b>

{market_block}

<b>Event:</b> <code>{liquidity_event}</code>
<b>Level:</b> <code>{level}</code>
<b>Side:</b> <code>{side}</code>
<b>Sweep:</b> <code>{sweep_detected}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


PRICE_ACTION_ALERT_TEMPLATE = """
<b>📈 Price Action Alert</b>

{market_block}

<b>Structure:</b> <code>{structure}</code>
<b>Pattern:</b> <code>{pattern}</code>
<b>Trend:</b> <code>{trend}</code>
<b>Side:</b> <code>{side}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


LIQUIDATIONS_ALERT_TEMPLATE = """
<b>🔥 Liquidation Alert</b>

{market_block}

<b>Side:</b> <code>{side}</code>
<b>Liquidation Volume:</b> <code>{liquidation_volume}</code>
<b>Cluster Price:</b> <code>{cluster_price}</code>
<b>Cascade Risk:</b> <code>{cascade_risk}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


WHALES_ALERT_TEMPLATE = """
<b>🐋 Whale Alert</b>

{market_block}

<b>Action:</b> <code>{whale_action}</code>
<b>Side:</b> <code>{side}</code>
<b>Volume:</b> <code>{volume}</code>
<b>Notional:</b> <code>{notional}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


SPOOFING_ALERT_TEMPLATE = """
<b>🎭 Spoofing Alert</b>

{market_block}

<b>Pattern:</b> <code>{pattern}</code>
<b>Side:</b> <code>{side}</code>
<b>Fake Liquidity:</b> <code>{fake_liquidity}</code>
<b>Confidence:</b> <code>{confidence}</code>

<b>Summary:</b>
{summary}

{footer}
""".strip()


SPREADS_ALERT_TEMPLATE = """
<b>🔁 Spread / Arbitrage Alert</b>

<b>Symbol:</b> <code>{symbol}</code>
<b>Base Exchange:</b> <code>{base_exchange}</code>
<b>Quote Exchange:</b> <code>{quote_exchange}</code>
<b>Spread:</b> <code>{spread}</code>
<b>Spread %:</b> <code>{spread_pct}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


FUNDING_ALERT_TEMPLATE = """
<b>💸 Funding Alert</b>

{market_block}

<b>Funding Rate:</b> <code>{funding_rate}</code>
<b>Predicted Rate:</b> <code>{predicted_rate}</code>
<b>Next Funding:</b> <code>{next_funding_time}</code>
<b>Bias:</b> <code>{bias}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


OPEN_INTEREST_ALERT_TEMPLATE = """
<b>📉 Open Interest Alert</b>

{market_block}

<b>Open Interest:</b> <code>{open_interest}</code>
<b>OI Delta:</b> <code>{oi_delta}</code>
<b>OI Delta %:</b> <code>{oi_delta_pct}</code>
<b>Price Change %:</b> <code>{price_change_pct}</code>
<b>Interpretation:</b> <code>{interpretation}</code>

{confidence_block}

<b>Summary:</b>
{summary}

{footer}
""".strip()


# =============================================================================
# News / AI templates
# =============================================================================

NEWS_ALERT_TEMPLATE = """
<b>📰 News Alert</b>

<b>Source:</b> <code>{source}</code>
<b>Impact:</b> <code>{impact}</code>
<b>Sentiment:</b> <code>{sentiment}</code>
<b>Symbols:</b> <code>{symbols}</code>

<b>Headline:</b>
{headline}

<b>Summary:</b>
{summary}

{footer}
""".strip()


AI_MARKET_CONTEXT_TEMPLATE = """
<b>🧠 AI Market Context</b>

<b>Scope:</b> <code>{scope}</code>
<b>Bias:</b> <code>{bias}</code>
<b>Confidence:</b> <code>{confidence}</code>
<b>Symbols:</b> <code>{symbols}</code>

<b>Context:</b>
{context}

<b>Risks:</b>
{risks}

{footer}
""".strip()


# =============================================================================
# Signal templates
# =============================================================================

SIGNAL_GENERATED_TEMPLATE = """
<b>🟡 Signal Generated</b>

{market_block}

<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>
<b>Signal Type:</b> <code>{signal_type}</code>
<b>Score:</b> <code>{score}</code>

{price_levels_block}

<b>Reason:</b>
{reason}

{footer}
""".strip()


SIGNAL_CONFIRMED_TEMPLATE = """
<b>🟢 Signal Confirmed</b>

{market_block}

<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>
<b>Risk:</b> <code>{risk_pct}</code>
<b>Position Size:</b> <code>{position_size}</code>

{price_levels_block}

<b>Risk Decision:</b>
{risk_decision}

{footer}
""".strip()


SIGNAL_REJECTED_TEMPLATE = """
<b>🔴 Signal Rejected</b>

{market_block}

<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>
<b>Reason:</b> <code>{reject_reason}</code>
<b>Score:</b> <code>{score}</code>

<b>Details:</b>
{details}

{footer}
""".strip()


SIGNAL_UPDATED_TEMPLATE = """
<b>🔄 Signal Updated</b>

{market_block}

<b>Signal ID:</b> <code>{signal_id}</code>
<b>Status:</b> <code>{status}</code>
<b>Update:</b> <code>{update_type}</code>

<b>Details:</b>
{details}

{footer}
""".strip()


# =============================================================================
# Execution / order templates
# =============================================================================

ORDER_SUBMITTED_TEMPLATE = """
<b>📨 Order Submitted</b>

{market_block}

<b>Order ID:</b> <code>{order_id}</code>
<b>Client Order ID:</b> <code>{client_order_id}</code>
<b>Side:</b> <code>{side}</code>
<b>Type:</b> <code>{order_type}</code>
<b>Quantity:</b> <code>{quantity}</code>
<b>Price:</b> <code>{price}</code>

{footer}
""".strip()


ORDER_FILLED_TEMPLATE = """
<b>✅ Order Filled</b>

{market_block}

<b>Order ID:</b> <code>{order_id}</code>
<b>Side:</b> <code>{side}</code>
<b>Filled Qty:</b> <code>{filled_quantity}</code>
<b>Avg Price:</b> <code>{avg_price}</code>
<b>Fee:</b> <code>{fee}</code>

{footer}
""".strip()


ORDER_REJECTED_TEMPLATE = """
<b>❌ Order Rejected</b>

{market_block}

<b>Order ID:</b> <code>{order_id}</code>
<b>Side:</b> <code>{side}</code>
<b>Reason:</b> <code>{reject_reason}</code>

<b>Details:</b>
{details}

{footer}
""".strip()


ORDER_CANCELLED_TEMPLATE = """
<b>🚫 Order Cancelled</b>

{market_block}

<b>Order ID:</b> <code>{order_id}</code>
<b>Side:</b> <code>{side}</code>
<b>Reason:</b> <code>{cancel_reason}</code>

{footer}
""".strip()


# =============================================================================
# Position / trade templates
# =============================================================================

POSITION_OPENED_TEMPLATE = """
<b>🟢 Open Trade</b>

{market_block}

<b>Position ID:</b> <code>{position_id}</code>
<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>
<b>Entry:</b> <code>{entry_price}</code>
<b>Size:</b> <code>{position_size}</code>
<b>Leverage:</b> <code>{leverage}</code>

<b>Stop Loss:</b> <code>{stop_loss}</code>
<b>Take Profit:</b> <code>{take_profit}</code>
<b>Risk:</b> <code>{risk_pct}</code>

<b>Reason:</b>
{reason}

{footer}
""".strip()


POSITION_UPDATED_TEMPLATE = """
<b>🔄 Trade Updated</b>

{market_block}

<b>Position ID:</b> <code>{position_id}</code>
<b>Side:</b> <code>{side}</code>
<b>Status:</b> <code>{status}</code>
<b>Entry:</b> <code>{entry_price}</code>
<b>Mark Price:</b> <code>{mark_price}</code>
<b>Unrealized PnL:</b> <code>{unrealized_pnl}</code>
<b>Unrealized PnL %:</b> <code>{unrealized_pnl_pct}</code>

<b>Update:</b>
{details}

{footer}
""".strip()


POSITION_CLOSED_TEMPLATE = """
<b>✅ Closed Trade / Result</b>

{market_block}

<b>Position ID:</b> <code>{position_id}</code>
<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>

<b>Entry:</b> <code>{entry_price}</code>
<b>Exit:</b> <code>{exit_price}</code>
<b>Size:</b> <code>{position_size}</code>

<b>Realized PnL:</b> <code>{realized_pnl}</code>
<b>Realized PnL %:</b> <code>{realized_pnl_pct}</code>
<b>Result:</b> <code>{trade_result}</code>
<b>Close Reason:</b> <code>{close_reason}</code>

<b>Summary:</b>
{summary}

{footer}
""".strip()


# =============================================================================
# Risk templates
# =============================================================================

RISK_WARNING_TEMPLATE = """
<b>⚠️ Risk Warning</b>

<b>Type:</b> <code>{risk_type}</code>
<b>Severity:</b> <code>{severity}</code>
<b>Symbol:</b> <code>{symbol}</code>
<b>Strategy:</b> <code>{strategy_name}</code>

<b>Message:</b>
{message}

<b>Details:</b>
{details}

{footer}
""".strip()


RISK_BLOCKED_TEMPLATE = """
<b>⛔ Position Blocked</b>

{market_block}

<b>Strategy:</b> <code>{strategy_name}</code>
<b>Side:</b> <code>{side}</code>
<b>Reason:</b> <code>{block_reason}</code>
<b>Risk Rule:</b> <code>{risk_rule}</code>

<b>Details:</b>
{details}

{footer}
""".strip()


RISK_KILL_SWITCH_TEMPLATE = """
<b>🚨 Risk Kill Switch</b>

<b>Status:</b> <code>{status}</code>
<b>Reason:</b> <code>{reason}</code>
<b>Scope:</b> <code>{scope}</code>

<b>Details:</b>
{details}

{footer}
""".strip()


# =============================================================================
# System templates
# =============================================================================

SYSTEM_INFO_TEMPLATE = """
<b>🛠 System Info</b>

<b>Service:</b> <code>{service}</code>
<b>Status:</b> <code>{status}</code>

<b>Message:</b>
{message}

{footer}
""".strip()


SYSTEM_WARNING_TEMPLATE = """
<b>⚠️ System Warning</b>

<b>Service:</b> <code>{service}</code>
<b>Status:</b> <code>{status}</code>

<b>Warning:</b>
{message}

<b>Details:</b>
{details}

{footer}
""".strip()


SYSTEM_ERROR_TEMPLATE = """
<b>❌ System Error</b>

<b>Service:</b> <code>{service}</code>
<b>Error:</b> <code>{error_type}</code>

<b>Message:</b>
{message}

<b>Details:</b>
{details}

{footer}
""".strip()


HEALTHCHECK_TEMPLATE = """
<b>💓 Telegram Bot Healthcheck</b>

<b>Status:</b> <code>{status}</code>
<b>Latency:</b> <code>{latency_ms}</code>
<b>Sent:</b> <code>{sent_messages}</code>
<b>Failed:</b> <code>{failed_messages}</code>
<b>Success Rate:</b> <code>{success_rate}</code>

{footer}
""".strip()


# =============================================================================
# Fallback / default templates
# =============================================================================

GENERIC_EVENT_TEMPLATE = """
<b>{title}</b>

<b>Category:</b> <code>{category}</code>
<b>Type:</b> <code>{message_type}</code>

<b>Payload:</b>
<pre>{payload}</pre>

{footer}
""".strip()


UNKNOWN_EVENT_TEMPLATE = """
<b>❔ Unknown Event</b>

<b>Event:</b> <code>{event_name}</code>
<b>Category:</b> <code>{category}</code>

<b>Payload:</b>
<pre>{payload}</pre>

{footer}
""".strip()


MESSAGE_CHUNK_PREFIX_TEMPLATE = """
<b>{title}</b>
<i>Part {index}/{total}</i>

""".lstrip()


# =============================================================================
# Template registry
# =============================================================================

MESSAGE_TYPE_TEMPLATES: MappingProxyType[TelegramMessageType, str] = MappingProxyType(
    {
        TelegramMessageType.ANALYTICS_ALERT: ANALYTICS_ALERT_TEMPLATE,
        TelegramMessageType.NEWS_ALERT: NEWS_ALERT_TEMPLATE,
        TelegramMessageType.SIGNAL_GENERATED: SIGNAL_GENERATED_TEMPLATE,
        TelegramMessageType.SIGNAL_REJECTED: SIGNAL_REJECTED_TEMPLATE,
        TelegramMessageType.SIGNAL_CONFIRMED: SIGNAL_CONFIRMED_TEMPLATE,
        TelegramMessageType.SIGNAL_UPDATED: SIGNAL_UPDATED_TEMPLATE,
        TelegramMessageType.ORDER_SUBMITTED: ORDER_SUBMITTED_TEMPLATE,
        TelegramMessageType.ORDER_FILLED: ORDER_FILLED_TEMPLATE,
        TelegramMessageType.ORDER_REJECTED: ORDER_REJECTED_TEMPLATE,
        TelegramMessageType.ORDER_CANCELLED: ORDER_CANCELLED_TEMPLATE,
        TelegramMessageType.POSITION_OPENED: POSITION_OPENED_TEMPLATE,
        TelegramMessageType.POSITION_UPDATED: POSITION_UPDATED_TEMPLATE,
        TelegramMessageType.POSITION_CLOSED: POSITION_CLOSED_TEMPLATE,
        TelegramMessageType.RISK_WARNING: RISK_WARNING_TEMPLATE,
        TelegramMessageType.RISK_BLOCKED: RISK_BLOCKED_TEMPLATE,
        TelegramMessageType.RISK_KILL_SWITCH: RISK_KILL_SWITCH_TEMPLATE,
        TelegramMessageType.SYSTEM_INFO: SYSTEM_INFO_TEMPLATE,
        TelegramMessageType.SYSTEM_WARNING: SYSTEM_WARNING_TEMPLATE,
        TelegramMessageType.SYSTEM_ERROR: SYSTEM_ERROR_TEMPLATE,
        TelegramMessageType.HEALTHCHECK: HEALTHCHECK_TEMPLATE,
    }
)


ANALYTICS_DOMAIN_TEMPLATES: MappingProxyType[str, str] = MappingProxyType(
    {
        "orderflow": ORDERFLOW_ALERT_TEMPLATE,
        "liquidity": LIQUIDITY_ALERT_TEMPLATE,
        "price_action": PRICE_ACTION_ALERT_TEMPLATE,
        "liquidations": LIQUIDATIONS_ALERT_TEMPLATE,
        "whales": WHALES_ALERT_TEMPLATE,
        "spoofing": SPOOFING_ALERT_TEMPLATE,
        "spreads": SPREADS_ALERT_TEMPLATE,
        "funding": FUNDING_ALERT_TEMPLATE,
        "open_interest": OPEN_INTEREST_ALERT_TEMPLATE,
    }
)


DEFAULT_TEMPLATE = GENERIC_EVENT_TEMPLATE


def get_template_for_message_type(message_type: TelegramMessageType) -> str:
    """
    Повертає шаблон для TelegramMessageType.
    """

    return MESSAGE_TYPE_TEMPLATES.get(message_type, DEFAULT_TEMPLATE)


def get_template_for_analytics_domain(domain: str | None) -> str:
    """
    Повертає доменний analytics-шаблон.

    Якщо domain невідомий — повертається generic analytics template.
    """

    if not domain:
        return ANALYTICS_ALERT_TEMPLATE

    return ANALYTICS_DOMAIN_TEMPLATES.get(domain, ANALYTICS_ALERT_TEMPLATE)