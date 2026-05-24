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




# =============================================================================
# Domain contract adapters
# =============================================================================


def _side_to_strategy(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(getattr(value, "value", value)).strip().lower()
    if text in {"buy", "long", "bull", "bullish", "up", "bid"}:
        return "buy"
    if text in {"sell", "short", "bear", "bearish", "down", "ask"}:
        return "sell"
    return text or None


def _first_present(payload: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _nested_get(payload, path)
        if not _is_missing(value):
            return value
    return None


def _mapping_from(payload: Mapping[str, Any], *paths: str) -> dict[str, Any] | None:
    for path in paths:
        value = _nested_get(payload, path)
        value = _to_plain(value)
        if isinstance(value, Mapping) and value:
            return dict(value)
    return None


def _set_feature(feature_map: dict[str, Any], name: str, value: Any) -> None:
    if _is_missing(value):
        return
    feature_map.setdefault(name, value)


def _set_alias(payload: dict[str, Any], canonical: str, value: Any, *aliases: str) -> None:
    if _is_missing(value):
        return
    payload.setdefault(canonical, value)
    for alias in aliases:
        payload.setdefault(alias, value)


def _ensure_orderflow_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}
        payload["feature_map"] = feature_map

    cvd = _mapping_from(payload, "cvd", "cumulative_volume_delta", "stats.cvd", "context.cvd") or {}
    volume_delta = _mapping_from(payload, "volume_delta", "delta", "stats.volume_delta", "context.volume_delta") or {}
    aggressive = _mapping_from(payload, "aggressive_trades", "aggressive", "aggression", "context.aggressive_trades") or {}
    orderbook = _mapping_from(payload, "orderbook_imbalance", "orderbook", "imbalance", "context.orderbook_imbalance") or {}
    signal = _mapping_from(payload, "signal", "orderflow_signal", "setup") or {}

    delta_value = _first_present(payload, "volume_delta.volume_delta", "volume_delta", "delta", "stats.volume_delta", "context.volume_delta")
    delta_ratio = _first_present(payload, "volume_delta.delta_ratio", "cvd.delta_ratio", "delta_ratio", "stats.delta_ratio", "context.delta_ratio")
    cvd_value = _first_present(payload, "cvd.value", "cvd", "cumulative_volume_delta", "stats.cvd", "context.cvd")
    buy_volume = _first_present(payload, "buy_volume", "stats.buy_volume", "context.buy_volume")
    sell_volume = _first_present(payload, "sell_volume", "stats.sell_volume", "context.sell_volume")
    total_volume = _first_present(payload, "total_volume", "volume", "stats.total_volume", "context.total_volume")
    trades_count = _first_present(payload, "trades_count", "trades", "trade_count", "stats.trades_count", "context.trades_count")
    last_price = _first_present(payload, "last_price", "price", "close", "mark_price", "context.last_price", "stats.last_price")
    side = _side_to_strategy(_first_present(payload, "side", "direction", "bias", "signal.side"))
    score = _first_present(payload, "score", "strength", "signal.score")
    confidence = _first_present(payload, "confidence", "strength", "signal.confidence")

    if cvd_value is not None:
        cvd.setdefault("value", cvd_value)
    if delta_ratio is not None:
        cvd.setdefault("delta_ratio", delta_ratio)
        volume_delta.setdefault("delta_ratio", delta_ratio)
    if delta_value is not None:
        volume_delta.setdefault("volume_delta", delta_value)
        aggressive.setdefault("net_volume_delta", delta_value)
    if buy_volume is not None:
        aggressive.setdefault("buy_volume", buy_volume)
    if sell_volume is not None:
        aggressive.setdefault("sell_volume", sell_volume)
    if side is not None:
        signal.setdefault("side", side)
    if score is not None:
        signal.setdefault("score", score)
    if confidence is not None:
        signal.setdefault("confidence", confidence)
    if signal:
        signal.setdefault("detected", True)
        signal.setdefault("type", _first_present(payload, "signal_type", "setup_type", "type") or "orderflow_signal")
        signal.setdefault("origin", "orderflow")

    composite = _mapping_from(payload, "composite", "snapshot", "orderflow_snapshot") or {}
    if cvd:
        composite.setdefault("cvd", cvd)
        _set_alias(payload, "cvd", cvd, "cvd_snapshot", "cvd_metrics")
    if volume_delta:
        composite.setdefault("volume_delta", volume_delta)
        _set_alias(payload, "volume_delta", volume_delta, "delta", "delta_metrics", "volume_delta_snapshot")
    if aggressive:
        composite.setdefault("aggressive_trades", aggressive)
        _set_alias(payload, "aggressive_trades", aggressive, "aggressive", "aggressive_flow", "aggressive_trades_snapshot")
    if orderbook:
        composite.setdefault("orderbook_imbalance", orderbook)
        _set_alias(payload, "orderbook_imbalance", orderbook, "orderbook", "imbalance", "orderbook_snapshot")
    for key, value in {
        "trades_count": trades_count,
        "total_volume": total_volume,
        "last_price": last_price,
        "side": side,
        "score": score,
        "confidence": confidence,
    }.items():
        if value is not None:
            composite.setdefault(key, value)
            payload.setdefault(key, value)
    if composite:
        _set_alias(payload, "composite", composite, "snapshot", "orderflow_snapshot", "composite_snapshot")
    if signal:
        _set_alias(payload, "signal", signal, "orderflow_signal", "analytics_signal", "setup")

    _set_feature(feature_map, "orderflow.composite", composite or None)
    _set_feature(feature_map, "orderflow.cvd", cvd or None)
    _set_feature(feature_map, "orderflow.cvd.value", cvd.get("value"))
    _set_feature(feature_map, "orderflow.cvd.delta_ratio", cvd.get("delta_ratio"))
    _set_feature(feature_map, "orderflow.volume_delta", volume_delta or None)
    _set_feature(feature_map, "orderflow.volume_delta.volume_delta", volume_delta.get("volume_delta"))
    _set_feature(feature_map, "orderflow.volume_delta.delta_ratio", volume_delta.get("delta_ratio"))
    _set_feature(feature_map, "orderflow.aggressive_trades", aggressive or None)
    _set_feature(feature_map, "orderflow.aggressive_trades.net_volume_delta", aggressive.get("net_volume_delta"))
    _set_feature(feature_map, "orderflow.orderbook_imbalance", orderbook or None)
    _set_feature(feature_map, "orderflow.trades_count", trades_count)
    _set_feature(feature_map, "orderflow.total_volume", total_volume)
    _set_feature(feature_map, "orderflow.last_price", last_price)
    _set_feature(feature_map, "orderflow.signal", signal or None)
    _set_feature(feature_map, "orderflow.signal.side", signal.get("side"))
    _set_feature(feature_map, "orderflow.signal.score", signal.get("score"))
    _set_feature(feature_map, "orderflow.signal.confidence", signal.get("confidence"))


def _ensure_funding_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map
    sections = {
        "snapshot": _mapping_from(payload, "snapshot", "funding_snapshot"),
        "statistics": _mapping_from(payload, "statistics", "stats", "funding_statistics"),
        "regime": _mapping_from(payload, "regime", "regime_state", "funding_regime"),
        "pressure": _mapping_from(payload, "pressure", "pressure_state", "funding_pressure"),
        "extreme": _mapping_from(payload, "extreme", "extreme_event", "funding_extreme"),
        "divergence": _mapping_from(payload, "divergence", "divergence_event", "funding_divergence"),
        "flip": _mapping_from(payload, "flip", "flip_event", "funding_flip"),
        "signal": _mapping_from(payload, "signal", "funding_signal", "setup"),
    }
    if sections["snapshot"] is None:
        flat = {k: _first_present(payload, k) for k in ("funding_rate","current_rate","next_funding_rate","predicted_rate","annualized_rate","premium_index","mark_price","index_price","next_funding_time") if _first_present(payload,k) is not None}
        sections["snapshot"] = flat or None
    aliases = {
        "snapshot": ("funding_snapshot",), "statistics": ("stats","funding_statistics"), "regime": ("regime_state","funding_regime"),
        "pressure": ("pressure_state","funding_pressure"), "extreme": ("extreme_event","funding_extreme"),
        "divergence": ("divergence_event","funding_divergence"), "flip": ("flip_event","funding_flip"), "signal": ("funding_signal","setup"),
    }
    for name, section in sections.items():
        if section:
            if name in {"extreme","divergence","flip","signal"}:
                section.setdefault("detected", True)
            _set_alias(payload, name, section, *aliases[name])
            _set_feature(feature_map, f"funding.{name}", section)
            for field in ("type","score","confidence","bias","direction","level","severity"):
                _set_feature(feature_map, f"funding.{name}.{field}", section.get(field))


def _ensure_open_interest_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map
    sections = {
        "analysis": _mapping_from(payload, "analysis", "oi_analysis", "open_interest_analysis", "result"),
        "snapshot": _mapping_from(payload, "snapshot", "oi_snapshot", "open_interest_snapshot"),
        "market_context": _mapping_from(payload, "market_context", "context", "oi_context", "open_interest_context"),
        "features": _mapping_from(payload, "features", "oi_features", "open_interest_features"),
        "regime": _mapping_from(payload, "regime", "regime_result", "oi_regime", "open_interest_regime"),
        "divergence": _mapping_from(payload, "divergence", "divergence_result", "oi_divergence", "open_interest_divergence"),
        "anomaly": _mapping_from(payload, "anomaly", "anomaly_result", "oi_anomaly", "open_interest_anomaly"),
    }
    if sections["features"] is None:
        flat = {k: _first_present(payload, k) for k in ("oi_delta","oi_delta_pct","open_interest","open_interest_value","price_delta_pct","oi_pressure_score","volume_delta","long_short_ratio") if _first_present(payload,k) is not None}
        sections["features"] = flat or None
    aliases = {
        "analysis": ("oi_analysis","open_interest_analysis","result"), "snapshot": ("oi_snapshot","open_interest_snapshot"),
        "market_context": ("context","oi_context","open_interest_context"), "features": ("oi_features","open_interest_features"),
        "regime": ("regime_result","oi_regime","open_interest_regime"), "divergence": ("divergence_result","oi_divergence","open_interest_divergence"),
        "anomaly": ("anomaly_result","oi_anomaly","open_interest_anomaly"),
    }
    for name, section in sections.items():
        if section:
            if name in {"divergence","anomaly"}:
                section.setdefault("detected", True)
            _set_alias(payload, name, section, *aliases[name])
            feature_prefix = "open_interest.context" if name == "market_context" else f"open_interest.{name}"
            _set_feature(feature_map, feature_prefix, section)
            for field in ("type","score","confidence","detected","oi_delta_pct","price_delta_pct","oi_pressure_score"):
                _set_feature(feature_map, f"{feature_prefix}.{field}", section.get(field))


def _ensure_liquidations_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map
    for name, aliases in {
        "cascade": ("cascade_result", "liquidation_cascade"),
        "exhaustion": ("exhaustion_result",),
        "squeeze": ("squeeze_result",),
        "cluster": ("liquidation_cluster",),
        "signal": ("liquidation_signal", "setup"),
    }.items():
        section = _mapping_from(payload, name, *aliases)
        if section is None and name in {"cascade","signal"}:
            flat = {k: _first_present(payload, k) for k in ("confidence","intensity_score","direction","side","severity","continuation_bias","exhaustion_bias","total_notional_usd","event_count","score","confirmed") if _first_present(payload,k) is not None}
            section = flat or None
        if section:
            section.setdefault("detected", True)
            _set_alias(payload, name, section, *aliases)
            _set_feature(feature_map, f"liquidations.{name}", section)
            for field in ("confidence","intensity_score","direction","side","severity","continuation_bias","exhaustion_bias","score","confirmed"):
                _set_feature(feature_map, f"liquidations.{name}.{field}", section.get(field))


def _ensure_whales_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map
    for name, aliases in {
        "large_trade": ("large_trade_signal", "whale_large_trade"),
        "activity": ("whale_activity", "whale_activity_signal"),
        "pressure": ("whale_pressure", "whale_pressure_signal"),
        "cluster": ("whale_cluster", "whale_cluster_signal"),
        "cluster_update": ("whale_cluster_update",),
        "cluster_exhaustion": ("whale_cluster_exhaustion",),
        "liquidation_context": ("whale_liquidation_context",),
    }.items():
        section = _mapping_from(payload, name, *aliases)
        if section is None and name == "large_trade":
            flat = {k: _first_present(payload, k) for k in ("side","whale_side","price","notional","total_notional","trade_count","zscore","confidence","score","reference_price") if _first_present(payload,k) is not None}
            section = flat or None
        if section:
            section.setdefault("detected", True)
            _set_alias(payload, name, section, *aliases)
            _set_feature(feature_map, f"whales.{name}", section)
            for field in ("side","whale_side","liquidation_side","pressure_score","context_strength","cluster_score","continuation_probability","exhaustion_probability","total_notional","trade_count","notional","zscore","reference_price","confidence","score"):
                _set_feature(feature_map, f"whales.{field}", section.get(field))
                _set_feature(feature_map, f"whales.{name}.{field}", section.get(field))


def ensure_domain_strategy_contract(payload: dict[str, Any], *, topic: str | None = None, source: str | None = None, domain: str | None = None) -> None:
    resolved = domain or _infer_domain(topic, source, payload)
    if resolved == "orderflow":
        _ensure_orderflow_contract(payload)
    elif resolved == "funding":
        _ensure_funding_contract(payload)
    elif resolved == "open_interest":
        _ensure_open_interest_contract(payload)
    elif resolved == "liquidations":
        _ensure_liquidations_contract(payload)
    elif resolved == "whales":
        _ensure_whales_contract(payload)

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

    ensure_domain_strategy_contract(result, topic=topic, source=source, domain=resolved_domain)

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
