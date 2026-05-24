from __future__ import annotations

"""
Shared analytics -> strategy payload contract helpers.

Every actionable analytics event that may be consumed by the strategy layer should
carry a stable top-level market price contract.  Individual analytics modules may
store price as `features.price`, `snapshot.mark_price`, `stats.last_price`,
`mid_price`, etc.; this helper lifts the best available value into fields that
StrategyContext/SignalBuilder can resolve consistently.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

STRATEGY_CONTRACT_VERSION = "analytics-strategy-v1"

_PRICE_PATHS: tuple[tuple[str, str], ...] = (
    ("current_price", "current_price"),
    ("last_price", "last_price"),
    ("price", "price"),
    ("close", "close"),
    ("mark_price", "mark_price"),
    ("index_price", "index_price"),
    ("reference_price", "reference_price"),
    ("entry_price", "entry_price"),
    ("mid_price", "mid_price"),
    ("current_mid_price", "current_mid_price"),
    ("best_bid", "best_bid"),
    ("best_ask", "best_ask"),
    ("features.current_price", "features.current_price"),
    ("features.last_price", "features.last_price"),
    ("features.price", "features.price"),
    ("features.mark_price", "features.mark_price"),
    ("features.index_price", "features.index_price"),
    ("features.close", "features.close"),
    ("snapshot.current_price", "snapshot.current_price"),
    ("snapshot.last_price", "snapshot.last_price"),
    ("snapshot.price", "snapshot.price"),
    ("snapshot.mark_price", "snapshot.mark_price"),
    ("snapshot.index_price", "snapshot.index_price"),
    ("state.current_price", "state.current_price"),
    ("state.last_price", "state.last_price"),
    ("state.price", "state.price"),
    ("state.close", "state.close"),
    ("state.state.current_price", "state.state.current_price"),
    ("state.state.last_price", "state.state.last_price"),
    ("state.state.price", "state.state.price"),
    ("state.state.close", "state.state.close"),
    ("state.state.trend.last_price", "state.state.trend.last_price"),
    ("state.state.market_structure.last_price", "state.state.market_structure.last_price"),
    ("state.state.support_resistance.last_price", "state.state.support_resistance.last_price"),
    ("state.state.liquidity.last_price", "state.state.liquidity.last_price"),
    ("state.state.fair_value_gap.last_price", "state.state.fair_value_gap.last_price"),
    ("context.current_price", "context.current_price"),
    ("context.last_price", "context.last_price"),
    ("context.latest_price", "context.latest_price"),
    ("context.price", "context.price"),
    ("context.mark_price", "context.mark_price"),
    ("context.mid_price", "context.mid_price"),
    ("stats.current_price", "stats.current_price"),
    ("stats.last_price", "stats.last_price"),
    ("stats.price", "stats.price"),
    ("stats.close", "stats.close"),
    ("stats.mid_price", "stats.mid_price"),
    ("orderbook.mid_price", "orderbook.mid_price"),
    ("orderbook.best_bid", "orderbook.best_bid"),
    ("orderbook.best_ask", "orderbook.best_ask"),
    ("trade.price", "trade.price"),
    ("event.price", "event.price"),
    ("liquidation.price", "liquidation.price"),
    ("signal.price", "signal.price"),
    ("signal.current_price", "signal.current_price"),
    ("signal.last_price", "signal.last_price"),
)

_TIMESTAMP_PATHS: tuple[str, ...] = (
    "price_timestamp",
    "timestamp",
    "timestamp_ms",
    "event_time",
    "event_time_ms",
    "snapshot.timestamp",
    "snapshot.event_time",
    "features.timestamp",
    "stats.timestamp",
    "context.timestamp",
    "state.timestamp",
    "detected_at",
    "created_at",
)

_SCOPE_KEYS = ("exchange", "market_type", "symbol", "timeframe", "exchange_symbol")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"none", "nan", "n/a", "null"}
    return False


def _to_plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_plain(value.to_dict())
        except TypeError:
            try:
                return _to_plain(value.to_dict(serialize=True))
            except Exception:
                return value
        except Exception:
            return value
    return value


def payload_to_dict(payload: Any) -> dict[str, Any]:
    plain = _to_plain(payload)
    if isinstance(plain, Mapping):
        return dict(plain)
    return {"value": plain}


def _nested_get(data: Mapping[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if isinstance(node, Mapping):
            node = node.get(part)
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _safe_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, Decimal):
        value = float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0 or result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _extract_price(payload: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for path, source in _PRICE_PATHS:
        value = _safe_float(_nested_get(payload, path))
        if value is not None:
            return value, source

    bid = _safe_float(_nested_get(payload, "best_bid")) or _safe_float(_nested_get(payload, "orderbook.best_bid"))
    ask = _safe_float(_nested_get(payload, "best_ask")) or _safe_float(_nested_get(payload, "orderbook.best_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0, "best_bid_best_ask_mid"
    return None, None


def _extract_timestamp(payload: Mapping[str, Any]) -> Any:
    for path in _TIMESTAMP_PATHS:
        value = _nested_get(payload, path)
        if not _is_missing(value):
            return _to_plain(value)
    return datetime.now(timezone.utc).isoformat()


def _infer_domain(topic: str | None, source: str | None, payload: Mapping[str, Any]) -> str | None:
    candidates = [payload.get("domain"), payload.get("analytics_type"), source, topic]
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).lower()
        for domain in (
            "orderflow",
            "price_action",
            "liquidity",
            "liquidations",
            "open_interest",
            "oi",
            "funding",
            "spoofing",
            "spreads",
            "whales",
        ):
            if domain in text:
                return "open_interest" if domain == "oi" else domain
    return None


def _ensure_scope(payload: dict[str, Any]) -> None:
    scope = payload.get("scope")
    if isinstance(scope, Mapping):
        for key in _SCOPE_KEYS:
            value = scope.get(key)
            if not _is_missing(value):
                payload.setdefault(key, value)
    scope_payload = {key: payload.get(key) for key in _SCOPE_KEYS if not _is_missing(payload.get(key))}
    if scope_payload:
        payload.setdefault("scope", scope_payload)


def ensure_strategy_payload_contract(
    payload: Any,
    *,
    topic: str | None = None,
    source: str | None = None,
    domain: str | None = None,
    require_price: bool = False,
) -> dict[str, Any]:
    """Return a strategy-ready analytics payload dict.

    The function is intentionally non-throwing for missing prices unless
    `require_price=True`; lifecycle/health events can pass through unchanged but
    still receive contract metadata.  Actionable events should pass payloads that
    contain one of the supported price aliases/nested paths.
    """
    result = payload_to_dict(payload)
    _ensure_scope(result)

    resolved_domain = domain or _infer_domain(topic, source, result)
    price, price_source = _extract_price(result)
    price_timestamp = _extract_timestamp(result)

    if price is not None:
        result.setdefault("current_price", price)
        result.setdefault("last_price", price)
        result.setdefault("reference_price", price)
        result.setdefault("price", price)
        result.setdefault("price_source", price_source or "unknown")
        result.setdefault("price_timestamp", price_timestamp)
        # SignalBuilder can use current_price/last_price, while entry_reference_price
        # explicitly marks this as analytics-derived market context rather than a
        # final trading decision.
        result.setdefault("entry_reference_price", price)
    elif require_price:
        result.setdefault("price_missing", True)
        result.setdefault("price_error", "analytics_strategy_contract_missing_price")

    result.setdefault("strategy_contract_version", STRATEGY_CONTRACT_VERSION)
    result.setdefault(
        "strategy_contract",
        {
            "version": STRATEGY_CONTRACT_VERSION,
            "domain": resolved_domain,
            "has_price": price is not None,
            "price_source": price_source,
            "price_field": "current_price" if price is not None else None,
            "expected_by": "StrategyContext/SignalBuilder",
        },
    )
    if resolved_domain:
        result.setdefault("domain", resolved_domain)
    return result
