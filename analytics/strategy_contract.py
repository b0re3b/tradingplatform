from __future__ import annotations

"""
Shared analytics -> strategy payload contract helpers.

Every actionable analytics event that may be consumed by the strategy layer should
carry a stable top-level market price contract.  Individual analytics modules may
store price as `features.price`, `snapshot.mark_price`, `stats.last_price`,
`mid_price`, etc.; this helper lifts the best available value into fields that
StrategyContext/SignalBuilder can resolve consistently.
"""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from collections.abc import Mapping
from typing import Any

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


_RUNTIME_SERIALIZATION_SKIP_FIELDS = {
    "logger",
    "_logger",
    "event_bus",
    "_event_bus",
    "scheduler",
    "_scheduler",
    "lock",
    "_lock",
    "locks",
    "_locks",
}


def _to_plain(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert analytics payload/model objects to JSON-safe primitives.

    This function is intentionally defensive because it runs on the shared
    analytics -> strategy boundary.  It must not let a malformed domain payload,
    a dataclass with runtime references, or a self-referential mapping break
    strategy contract normalization.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if _seen is None:
        _seen = set()

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()

    value_id = id(value)
    if value_id in _seen:
        return "<circular_ref>"

    if isinstance(value, Mapping):
        _seen.add(value_id)
        try:
            return {
                str(k): _to_plain(v, _seen=_seen)
                for k, v in value.items()
                if str(k) not in _RUNTIME_SERIALIZATION_SKIP_FIELDS
            }
        finally:
            _seen.discard(value_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(value_id)
        try:
            return [_to_plain(v, _seen=_seen) for v in value]
        finally:
            _seen.discard(value_id)

    if is_dataclass(value) and not isinstance(value, type):
        _seen.add(value_id)
        try:
            result: dict[str, Any] = {}
            for field in fields(value):
                if field.name in _RUNTIME_SERIALIZATION_SKIP_FIELDS:
                    continue
                try:
                    result[field.name] = _to_plain(getattr(value, field.name), _seen=_seen)
                except Exception:
                    result[field.name] = repr(getattr(value, field.name, None))
            return result
        finally:
            _seen.discard(value_id)

    for method_name in ("to_payload", "to_dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue

        _seen.add(value_id)
        try:
            try:
                converted = method()
            except TypeError:
                converted = method(serialize=True)
            return _to_plain(converted, _seen=_seen)
        except Exception:
            return repr(value)
        finally:
            _seen.discard(value_id)

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

def _ensure_liquidity_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}
        payload["feature_map"] = feature_map

    snapshot = _mapping_from(
        payload,
        "snapshot",
        "liquidity_snapshot",
        "liquidity_map_snapshot",
        "map_snapshot",
        "map",
    )

    signal = _mapping_from(
        payload,
        "signal",
        "liquidity_signal",
        "analytics_signal",
        "setup",
    )

    cluster = _mapping_from(
        payload,
        "cluster",
        "stop_cluster",
        "liquidity_cluster",
    )

    clusters = _first_present(
        payload,
        "clusters",
        "stop_clusters",
        "liquidity_clusters",
        "snapshot.stop_clusters",
        "liquidity_snapshot.stop_clusters",
    )

    levels = _first_present(
        payload,
        "levels",
        "active_levels",
        "liquidity_levels",
        "source_levels",
        "snapshot.active_levels",
        "snapshot.levels",
        "liquidity_snapshot.active_levels",
    )

    equal_levels = _first_present(
        payload,
        "equal_levels",
        "equal_highs_lows",
        "snapshot.equal_levels",
        "liquidity_snapshot.equal_levels",
    )

    zones = _first_present(
        payload,
        "zones",
        "liquidity_zones",
        "snapshot.zones",
        "liquidity_snapshot.zones",
    )

    center_price = _first_present(
        payload,
        "current_price",
        "reference_price",
        "entry_reference_price",
        "last_price",
        "price",
        "center_price",
        "snapshot.current_price",
        "liquidity_snapshot.current_price",
    )

    if snapshot is None:
        # Не робимо generic snapshot із будь-якого payload.
        # Але stop_cluster.detected — це валідний liquidity event, і він має
        # достатньо даних для мінімального snapshot contract.
        low_price = _first_present(payload, "low_price")
        high_price = _first_present(payload, "high_price")
        side = _first_present(payload, "side")

        if center_price is not None and low_price is not None and high_price is not None:
            cluster_payload = cluster or {
                "exchange": _first_present(payload, "exchange"),
                "market_type": _first_present(payload, "market_type"),
                "symbol": _first_present(payload, "symbol"),
                "timeframe": _first_present(payload, "timeframe"),
                "side": _side_to_strategy(side),
                "low_price": low_price,
                "high_price": high_price,
                "center_price": center_price,
                "width": _first_present(payload, "width"),
                "width_pct": _first_present(payload, "width_pct"),
                "confidence": _first_present(payload, "confidence"),
                "estimated_stop_density": _first_present(payload, "estimated_stop_density"),
                "touches_count": _first_present(payload, "touches_count"),
                "source_level_type": _first_present(payload, "source_level_type"),
                "strength": _first_present(payload, "strength"),
                "created_at": _first_present(payload, "created_at"),
                "updated_at": _first_present(payload, "updated_at"),
                "swept_at": _first_present(payload, "swept_at"),
                "is_active": _first_present(payload, "is_active"),
                "is_swept": _first_present(payload, "is_swept"),
                "is_terminal": _first_present(payload, "is_terminal"),
                "source_levels": levels or [],
            }

            snapshot = {
                "exchange": _first_present(payload, "exchange"),
                "market_type": _first_present(payload, "market_type"),
                "symbol": _first_present(payload, "symbol"),
                "timeframe": _first_present(payload, "timeframe"),
                "exchange_symbol": _first_present(payload, "exchange_symbol"),
                "current_price": center_price,
                "active_levels": levels or [],
                "equal_levels": equal_levels or [],
                "stop_clusters": clusters or [cluster_payload],
                "clusters": clusters or [cluster_payload],
                "zones": zones or [],
                "nearest_above_level": _first_present(payload, "nearest_above_level"),
                "nearest_below_level": _first_present(payload, "nearest_below_level"),
                "strongest_cluster_above": _first_present(payload, "strongest_cluster_above"),
                "strongest_cluster_below": _first_present(payload, "strongest_cluster_below"),
                "above_liquidity_score": _first_present(payload, "above_liquidity_score"),
                "below_liquidity_score": _first_present(payload, "below_liquidity_score"),
                "liquidity_pressure_score": _first_present(payload, "liquidity_pressure_score"),
                "bias": _first_present(payload, "bias", "direction"),
                "timestamp": _first_present(payload, "timestamp", "updated_at", "created_at"),
            }

    if snapshot:
        _set_alias(
            payload,
            "snapshot",
            snapshot,
            "liquidity_snapshot",
            "liquidity_map_snapshot",
            "map_snapshot",
        )
        _set_feature(feature_map, "liquidity.snapshot", snapshot)
        _set_feature(feature_map, "liquidity.map.snapshot", snapshot)

    if clusters:
        _set_alias(payload, "clusters", clusters, "stop_clusters", "liquidity_clusters")
        _set_feature(feature_map, "liquidity.clusters", clusters)
        _set_feature(feature_map, "liquidity.stop_clusters", clusters)

    if levels:
        _set_alias(payload, "levels", levels, "active_levels", "liquidity_levels")
        _set_feature(feature_map, "liquidity.levels", levels)
        _set_feature(feature_map, "liquidity.active_levels", levels)

    if equal_levels:
        _set_alias(payload, "equal_levels", equal_levels, "equal_highs_lows")
        _set_feature(feature_map, "liquidity.equal_levels", equal_levels)

    if zones:
        _set_alias(payload, "zones", zones, "liquidity_zones")
        _set_feature(feature_map, "liquidity.zones", zones)

    if center_price is not None:
        payload.setdefault("current_price", center_price)
        payload.setdefault("last_price", center_price)
        payload.setdefault("reference_price", center_price)
        payload.setdefault("entry_reference_price", center_price)
        payload.setdefault("price", center_price)
        _set_feature(feature_map, "liquidity.current_price", center_price)

    if signal:
        signal.setdefault("detected", True)
        signal.setdefault("origin", "liquidity")
        _set_alias(payload, "signal", signal, "liquidity_signal", "analytics_signal", "setup")
        _set_feature(feature_map, "liquidity.signal", signal)
        _set_feature(feature_map, "liquidity.signal.side", signal.get("side"))
        _set_feature(feature_map, "liquidity.signal.score", signal.get("score"))
        _set_feature(feature_map, "liquidity.signal.confidence", signal.get("confidence"))
def _ensure_orderflow_contract(payload: dict[str, Any]) -> None:
    """
    Normalize analytics.orderflow events into the exact strategy contract read by
    strategy/strategies/orderflow/*.

    The important case is OrderFlowUpdate.from_stats(): concrete analyzers emit
    metric-specific flat values under payload["stats"].  Strategies do not read
    raw analytics payloads; they read FeatureSource.ORDERFLOW domain sections and
    FeatureSnapshot names, so this adapter lifts stats.* into canonical sections:

        orderflow.composite
        orderflow.cvd
        orderflow.volume_delta
        orderflow.aggressive_trades
        orderflow.orderbook_imbalance
        orderflow.signal
    """
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}
        payload["feature_map"] = feature_map

    stats = _mapping_from(payload, "stats", "context.stats") or {}
    context = _mapping_from(payload, "context") or {}
    metric = str(_first_present(payload, "metric", "stats.metric", "context.metric") or "").strip().lower()

    def first(*paths: str) -> Any:
        for path in paths:
            value = _first_present(payload, path)
            if not _is_missing(value):
                return value
            value = _nested_get(stats, path)
            if not _is_missing(value):
                return value
            value = _nested_get(context, path)
            if not _is_missing(value):
                return value
        return None

    def put(mapping: dict[str, Any], key: str, value: Any) -> None:
        if not _is_missing(value):
            mapping.setdefault(key, value)

    cvd = _mapping_from(
        payload,
        "cvd",
        "cvd_snapshot",
        "cvd_metrics",
        "cumulative_volume_delta",
        "stats.cvd",
        "context.cvd",
        "context.stats.cvd",
    ) or {}
    volume_delta = _mapping_from(
        payload,
        "volume_delta",
        "volume_delta_snapshot",
        "delta",
        "delta_metrics",
        "stats.volume_delta_section",
        "context.volume_delta",
        "context.stats.volume_delta_section",
    ) or {}
    aggressive = _mapping_from(
        payload,
        "aggressive_trades",
        "aggressive",
        "aggression",
        "aggressive_flow",
        "stats.aggressive_trades",
        "context.aggressive_trades",
        "context.stats.aggressive_trades",
    ) or {}
    orderbook = _mapping_from(
        payload,
        "orderbook_imbalance",
        "orderbook",
        "imbalance",
        "book_imbalance",
        "stats.orderbook_imbalance",
        "context.orderbook_imbalance",
        "context.stats.orderbook_imbalance",
    ) or {}
    signal = _mapping_from(payload, "signal", "orderflow_signal", "analytics_signal", "setup") or {}

    cvd_value = first("cvd.value", "cvd_value", "cvd_close", "cumulative_volume_delta")
    delta_value = first("volume_delta.volume_delta", "volume_delta", "delta")
    delta_ratio = first("cvd.delta_ratio", "volume_delta.delta_ratio", "delta_ratio", "volume_delta_ratio", "cvd_delta_ratio")

    buy_volume = first("buy_volume", "aggressive_buy_volume")
    sell_volume = first("sell_volume", "aggressive_sell_volume")
    buy_notional = first("buy_notional", "aggressive_buy_notional")
    sell_notional = first("sell_notional", "aggressive_sell_notional")
    total_volume = first("total_volume", "volume")
    if _is_missing(total_volume) and not _is_missing(buy_volume) and not _is_missing(sell_volume):
        try:
            total_volume = float(buy_volume) + float(sell_volume)
        except (TypeError, ValueError):
            total_volume = None

    total_notional = first("total_notional", "notional", "quote_volume")
    if _is_missing(total_notional) and not _is_missing(buy_notional) and not _is_missing(sell_notional):
        try:
            total_notional = float(buy_notional) + float(sell_notional)
        except (TypeError, ValueError):
            total_notional = None

    buy_ratio = first("aggressive_trades.buy_ratio", "buy_ratio", "cvd_buy_ratio", "aggressive_buy_ratio")
    sell_ratio = first("aggressive_trades.sell_ratio", "sell_ratio", "cvd_sell_ratio", "aggressive_sell_ratio")
    try:
        if _is_missing(buy_ratio) and not _is_missing(buy_volume) and total_volume and float(total_volume) > 0:
            buy_ratio = float(buy_volume) / float(total_volume)
        if _is_missing(sell_ratio) and not _is_missing(sell_volume) and total_volume and float(total_volume) > 0:
            sell_ratio = float(sell_volume) / float(total_volume)
        if _is_missing(buy_ratio) and not _is_missing(sell_ratio):
            buy_ratio = max(0.0, 1.0 - float(sell_ratio))
        if _is_missing(sell_ratio) and not _is_missing(buy_ratio):
            sell_ratio = max(0.0, 1.0 - float(buy_ratio))
    except (TypeError, ValueError):
        pass

    # CVD section: produced by cvd.updated/cvd.signal and also useful as a
    # conservative view of directional trade pressure.
    for key, value in {
        "value": cvd_value,
        "cvd_value": cvd_value,
        "cvd_open": first("cvd.cvd_open", "cvd_open"),
        "cvd_high": first("cvd.cvd_high", "cvd_high"),
        "cvd_low": first("cvd.cvd_low", "cvd_low"),
        "cvd_close": first("cvd.cvd_close", "cvd_close"),
        "cvd_change": first("cvd.cvd_change", "cvd_change"),
        "cvd_change_pct": first("cvd.cvd_change_pct", "cvd_change_pct", "change_pct"),
        "cvd_slope": first("cvd.cvd_slope", "cvd_slope", "slope"),
        "delta_ratio": delta_ratio,
        "price_change_pct": first("cvd.price_change_pct", "price_change_pct", "price_delta_pct"),
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "trades_count": first("trades_count", "trade_count", "trades"),
        "total_volume": total_volume,
        "total_notional": total_notional,
        "last_price": first("last_price", "price", "close", "mark_price"),
        "window_seconds": first("window_seconds"),
    }.items():
        put(cvd, key, value)

    # Volume-delta section.
    for key, value in {
        "volume_delta": delta_value,
        "delta_ratio": delta_ratio,
        "cumulative_volume_delta": first("volume_delta.cumulative_volume_delta", "cumulative_volume_delta", "cvd_value", "cvd_close"),
        "notional_delta": first("volume_delta.notional_delta", "notional_delta"),
        "cumulative_notional_delta": first("volume_delta.cumulative_notional_delta", "cumulative_notional_delta"),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "trades_count": first("trades_count", "trade_count", "trades"),
        "total_volume": total_volume,
        "total_notional": total_notional,
    }.items():
        put(volume_delta, key, value)

    # Aggressive-trades section.  For pure CVD/volume-delta updates we expose a
    # conservative approximation from buy/sell ratios, so strategies can still
    # resolve a complete OrderflowCompositeSnapshot without raw market reads.
    for key, value in {
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "net_volume_delta": first("aggressive_trades.net_volume_delta", "net_volume_delta", "volume_delta", "delta"),
        "net_notional_delta": first("aggressive_trades.net_notional_delta", "net_notional_delta", "notional_delta"),
        "burst_score": first("aggressive_trades.burst_score", "burst_score", "aggressive_burst_score", "aggression_score"),
        "large_buy_trades": first("aggressive_trades.large_buy_trades", "large_buy_trades"),
        "large_sell_trades": first("aggressive_trades.large_sell_trades", "large_sell_trades"),
        "aggressive_buy_count": first("aggressive_buy_count"),
        "aggressive_sell_count": first("aggressive_sell_count"),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "trades_count": first("trades_count", "trade_count", "trades"),
    }.items():
        put(aggressive, key, value)

    # Orderbook-imbalance section.
    for key, value in {
        "ratio": first("orderbook_imbalance.ratio", "orderbook_imbalance.imbalance_ratio", "imbalance_ratio", "orderbook_imbalance_ratio"),
        "diff": first("orderbook_imbalance.diff", "orderbook_imbalance.imbalance_diff", "imbalance_diff", "orderbook_imbalance_diff"),
        "imbalance_ratio": first("orderbook_imbalance.imbalance_ratio", "imbalance_ratio", "orderbook_imbalance_ratio"),
        "imbalance_diff": first("orderbook_imbalance.imbalance_diff", "imbalance_diff", "orderbook_imbalance_diff"),
        "bid_volume": first("bid_volume"),
        "ask_volume": first("ask_volume"),
        "best_bid": first("best_bid"),
        "best_ask": first("best_ask"),
        "spread": first("spread"),
        "mid_price": first("mid_price"),
        "depth_levels_used": first("depth_levels_used"),
    }.items():
        put(orderbook, key, value)

    side = _side_to_strategy(first("side", "direction", "bias", "signal.side"))
    score = first("score", "strength", "signal_score", "signal.score")
    confidence = first("confidence", "strength", "signal_confidence", "signal.confidence")
    signal_type = first("signal_type", "setup_type", "type")
    if signal or signal_type is not None or side is not None:
        signal.setdefault("detected", True)
        signal.setdefault("type", signal_type or "orderflow_signal")
        if side is not None:
            signal.setdefault("side", side)
        if score is not None:
            signal.setdefault("score", score)
        if confidence is not None:
            signal.setdefault("confidence", confidence)
        signal.setdefault("origin", "orderflow")

    composite = _mapping_from(payload, "composite", "snapshot", "orderflow_snapshot", "composite_snapshot") or {}
    for key, section in (
        ("cvd", cvd),
        ("volume_delta", volume_delta),
        ("aggressive_trades", aggressive),
        ("orderbook_imbalance", orderbook),
    ):
        if section:
            composite.setdefault(key, section)

    for key, value in {
        "exchange": first("exchange"),
        "market_type": first("market_type"),
        "symbol": first("symbol"),
        "timeframe": first("timeframe"),
        "exchange_symbol": first("exchange_symbol"),
        "timestamp": first("timestamp", "event_time"),
        "last_price": first("last_price", "price", "close", "mark_price", "mid_price"),
        "price": first("price", "last_price", "close", "mark_price", "mid_price"),
        "price_change": first("price_change"),
        "price_change_pct": first("price_change_pct", "price_delta_pct"),
        "window_seconds": first("window_seconds"),
        "trades_count": first("trades_count", "trade_count", "trades"),
        "total_volume": total_volume,
        "total_notional": total_notional,
        "metric": metric or None,
        "source_type": first("source_type"),
    }.items():
        put(composite, key, value)
        if not _is_missing(value):
            payload.setdefault(key, value)

    if cvd:
        _set_alias(payload, "cvd", cvd, "cvd_snapshot", "cvd_metrics", "cumulative_delta")
    if volume_delta:
        _set_alias(payload, "volume_delta", volume_delta, "delta", "delta_metrics", "volume_delta_snapshot")
    if aggressive:
        _set_alias(payload, "aggressive_trades", aggressive, "aggressive", "aggressive_flow", "aggressive_trades_snapshot")
    if orderbook:
        _set_alias(payload, "orderbook_imbalance", orderbook, "orderbook", "imbalance", "orderbook_snapshot")
    if composite:
        _set_alias(payload, "composite", composite, "snapshot", "orderflow_snapshot", "composite_snapshot")
    if signal:
        _set_alias(payload, "signal", signal, "orderflow_signal", "analytics_signal", "setup")

    # Stable FeatureSnapshot names used by strategy/strategies/orderflow/base.py.
    _set_feature(feature_map, "orderflow.composite", composite or None)
    _set_feature(feature_map, "orderflow.cvd", cvd or None)
    _set_feature(feature_map, "orderflow.cvd.value", cvd.get("value"))
    _set_feature(feature_map, "orderflow.cvd.delta_ratio", cvd.get("delta_ratio"))
    _set_feature(feature_map, "orderflow.cvd.cvd_change_pct", cvd.get("cvd_change_pct"))
    _set_feature(feature_map, "orderflow.cvd.cvd_slope", cvd.get("cvd_slope"))
    _set_feature(feature_map, "orderflow.cvd.price_change_pct", cvd.get("price_change_pct"))
    _set_feature(feature_map, "orderflow.volume_delta", volume_delta or None)
    _set_feature(feature_map, "orderflow.volume_delta.volume_delta", volume_delta.get("volume_delta"))
    _set_feature(feature_map, "orderflow.volume_delta.delta_ratio", volume_delta.get("delta_ratio"))
    _set_feature(feature_map, "orderflow.volume_delta.cumulative_volume_delta", volume_delta.get("cumulative_volume_delta"))
    _set_feature(feature_map, "orderflow.volume_delta.notional_delta", volume_delta.get("notional_delta"))
    _set_feature(feature_map, "orderflow.volume_delta.cumulative_notional_delta", volume_delta.get("cumulative_notional_delta"))
    _set_feature(feature_map, "orderflow.aggressive_trades", aggressive or None)
    _set_feature(feature_map, "orderflow.aggressive_trades.buy_ratio", aggressive.get("buy_ratio"))
    _set_feature(feature_map, "orderflow.aggressive_trades.sell_ratio", aggressive.get("sell_ratio"))
    _set_feature(feature_map, "orderflow.aggressive_trades.burst_score", aggressive.get("burst_score"))
    _set_feature(feature_map, "orderflow.aggressive_trades.net_volume_delta", aggressive.get("net_volume_delta"))
    _set_feature(feature_map, "orderflow.aggressive_trades.net_notional_delta", aggressive.get("net_notional_delta"))
    _set_feature(feature_map, "orderflow.aggressive_trades.large_buy_trades", aggressive.get("large_buy_trades"))
    _set_feature(feature_map, "orderflow.aggressive_trades.large_sell_trades", aggressive.get("large_sell_trades"))
    _set_feature(feature_map, "orderflow.orderbook_imbalance", orderbook or None)
    _set_feature(feature_map, "orderflow.orderbook_imbalance.ratio", orderbook.get("ratio") or orderbook.get("imbalance_ratio"))
    _set_feature(feature_map, "orderflow.orderbook_imbalance.diff", orderbook.get("diff") or orderbook.get("imbalance_diff"))
    _set_feature(feature_map, "orderflow.trades_count", composite.get("trades_count"))
    _set_feature(feature_map, "orderflow.total_volume", composite.get("total_volume"))
    _set_feature(feature_map, "orderflow.total_notional", composite.get("total_notional"))
    _set_feature(feature_map, "orderflow.last_price", composite.get("last_price"))
    _set_feature(feature_map, "orderflow.price_change_pct", composite.get("price_change_pct"))
    _set_feature(feature_map, "orderflow.signal", signal or None)
    _set_feature(feature_map, "orderflow.signal.side", signal.get("side"))
    _set_feature(feature_map, "orderflow.signal.score", signal.get("score"))
    _set_feature(feature_map, "orderflow.signal.confidence", signal.get("confidence"))


def _funding_first(section: Mapping[str, Any] | None, *paths: str) -> Any:
    if not isinstance(section, Mapping):
        return None
    return _first_present(section, *paths)


def _funding_section_type(section: Mapping[str, Any] | None, *paths: str) -> Any:
    if not isinstance(section, Mapping):
        return None
    for path in paths:
        value = _first_present(section, path)
        if not _is_missing(value):
            return value
    return None


def _normalize_funding_section(name: str, section: dict[str, Any], *, payload: Mapping[str, Any], feature_map: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize analytics.funding model field names to strategy contract keys."""
    section = dict(section or {})
    for _runtime_key in ("feature_map", "features", "strategy_contract", "strategy_contract_version"):
        section.pop(_runtime_key, None)

    def pick(*paths: str, default: Any = None) -> Any:
        for path in paths:
            value = _nested_get(section, path)
            if not _is_missing(value):
                return value
        for path in paths:
            value = _nested_get(feature_map, f"funding.{name}.{path}")
            if not _is_missing(value):
                return value
        for path in paths:
            value = _nested_get(payload, f"{name}_{path}")
            if not _is_missing(value):
                return value
        return default

    if name == "snapshot":
        aliases = {
            "funding_rate": ("funding_rate", "current_rate", "rate"),
            "current_rate": ("current_rate", "funding_rate", "rate"),
            "predicted_rate": ("predicted_rate", "predicted_funding_rate", "next_funding_rate"),
            "predicted_funding_rate": ("predicted_funding_rate", "predicted_rate", "next_funding_rate"),
            "mark_price": ("mark_price", "current_price", "reference_price"),
            "index_price": ("index_price",),
            "next_funding_time": ("next_funding_time",),
            "event_time": ("event_time", "timestamp", "updated_at", "received_at"),
        }
    elif name == "statistics":
        aliases = {
            "current_rate": ("current_rate", "funding_rate", "rate"),
            "mean_rate": ("mean_rate", "mean"),
            "median_rate": ("median_rate", "median"),
            "std_rate": ("std_rate", "std", "stdev"),
            "zscore": ("zscore", "z_score"),
            "percentile": ("percentile",),
            "sample_size": ("sample_size", "samples", "count"),
            "updated_at": ("updated_at", "event_time", "timestamp"),
        }
    elif name == "regime":
        aliases = {
            "type": ("type", "regime", "name", "state"),
            "regime": ("regime", "type", "name"),
            "bias": ("bias", "direction", "side"),
            "confidence": ("confidence", "score_confidence"),
            "score": ("score", "confidence"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    elif name == "pressure":
        aliases = {
            "score": ("score", "pressure_score", "strength", "normalized_score"),
            "pressure_score": ("pressure_score", "score", "strength", "normalized_score"),
            "level": ("level", "pressure_level", "type"),
            "direction": ("direction", "pressure_direction", "bias", "side"),
            "bias": ("bias", "direction", "pressure_direction"),
            "squeeze_probability": ("squeeze_probability", "squeeze_risk"),
            "mean_reversion_probability": ("mean_reversion_probability", "reversion_probability", "reversal_probability"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    elif name == "extreme":
        aliases = {
            "type": ("type", "extreme_type", "kind"),
            "extreme_type": ("extreme_type", "type", "kind"),
            "score": ("score", "severity", "strength", "normalized_score"),
            "severity": ("severity", "score", "strength", "normalized_score"),
            "confidence": ("confidence", "severity", "mean_reversion_probability"),
            "reversal_risk": ("reversal_risk", "is_reversal_risk", "has_reversal_risk", "mean_reversion_risk"),
            "squeeze_risk": ("squeeze_risk", "is_squeeze_risk", "has_squeeze_risk"),
            "mean_reversion_probability": ("mean_reversion_probability", "reversion_probability", "reversal_probability"),
            "squeeze_probability": ("squeeze_probability", "short_squeeze_probability", "long_squeeze_probability", "squeeze_risk"),
            "funding_rate": ("funding_rate", "current_rate", "rate"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    elif name == "divergence":
        aliases = {
            "type": ("type", "divergence_type", "kind"),
            "divergence_type": ("divergence_type", "type", "kind"),
            "score": ("score", "confidence", "strength", "signed_score"),
            "confidence": ("confidence", "score_confidence"),
            "bias": ("bias", "direction", "side", "expected_side"),
            "side": ("side", "signal_side", "expected_side", "target_side", "bias", "direction"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    elif name == "flip":
        aliases = {
            "type": ("type", "flip_type", "kind"),
            "flip_type": ("flip_type", "type", "kind"),
            "score": ("score", "confidence", "flip_magnitude"),
            "confidence": ("confidence", "score_confidence"),
            "magnitude": ("magnitude", "flip_magnitude"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    elif name == "signal":
        aliases = {
            "type": ("type", "signal_type", "setup_type"),
            "signal_type": ("signal_type", "type", "setup_type"),
            "score": ("score", "signed_score", "strength"),
            "confidence": ("confidence", "score_confidence"),
            "bias": ("bias", "direction", "side"),
            "origin": ("origin", "signal_origin"),
            "event_time": ("event_time", "timestamp", "updated_at"),
        }
    else:
        aliases = {}

    for canonical, candidates in aliases.items():
        value = pick(*candidates, default=None)
        if not _is_missing(value):
            section.setdefault(canonical, value)

    if name in {"extreme", "divergence", "flip", "signal"}:
        section.setdefault("detected", section.get("active", section.get("confirmed", True)))

    if name == "extreme":
        # FundingExtremeEvent exposes booleans but not probabilities; concrete
        # strategy thresholds use probabilities.  Use severity as the actionable
        # probability only when analytics explicitly flagged that risk.
        severity = section.get("severity", section.get("score", 0.0))
        if section.get("is_reversal_risk") is not None:
            section.setdefault("reversal_risk", section.get("is_reversal_risk"))
        if section.get("is_squeeze_risk") is not None:
            section.setdefault("squeeze_risk", section.get("is_squeeze_risk"))
        if bool(section.get("reversal_risk")) and _is_missing(section.get("mean_reversion_probability")):
            section["mean_reversion_probability"] = severity
        if bool(section.get("squeeze_risk")) and _is_missing(section.get("squeeze_probability")):
            section["squeeze_probability"] = severity

    return section


def _ensure_funding_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}
        payload["feature_map"] = feature_map

    analysis = _mapping_from(payload, "analysis", "result", "funding_analysis", "funding_result", "payload")

    def section(*paths: str) -> dict[str, Any] | None:
        result = _mapping_from(payload, *paths)
        if result is not None:
            return result
        result = _mapping_from(feature_map, *paths)
        if result is not None:
            return result
        if analysis is not None:
            return _mapping_from(analysis, *paths)
        return None

    sections = {
        "snapshot": section("snapshot", "funding_snapshot", "payload.snapshot", "payload.funding_snapshot"),
        "statistics": section("statistics", "stats", "funding_statistics", "payload.statistics", "payload.stats"),
        "regime": section("regime", "regime_state", "funding_regime", "funding_regime_state", "payload.regime_state"),
        "pressure": section("pressure", "pressure_state", "funding_pressure", "funding_pressure_state", "payload.pressure_state"),
        "extreme": section("extreme", "extreme_event", "funding_extreme", "funding_extreme_event", "payload.extreme_event"),
        "divergence": section("divergence", "divergence_event", "funding_divergence", "funding_divergence_event", "payload.divergence_event"),
        "flip": section("flip", "flip_event", "funding_flip", "funding_flip_event", "payload.flip_event"),
        "signal": section("signal", "funding_signal", "analytics_signal", "setup", "strategy_signal", "payload.signal"),
    }

    if sections["snapshot"] is None:
        flat = {
            key: _first_present(payload, key, f"feature_map.funding.{key}")
            for key in (
                "funding_rate", "current_rate", "next_funding_rate", "predicted_rate",
                "predicted_funding_rate", "annualized_rate", "premium_index",
                "mark_price", "index_price", "open_interest", "volume_24h",
                "next_funding_time", "exchange", "market_type", "symbol",
                "exchange_symbol", "timeframe", "timestamp", "event_time",
            )
            if not _is_missing(_first_present(payload, key, f"feature_map.funding.{key}"))
        }
        sections["snapshot"] = flat or None

    if sections["statistics"] is None:
        flat = {
            key: _first_present(payload, key, f"feature_map.funding.statistics.{key}")
            for key in (
                "current_rate", "mean_rate", "median_rate", "std_rate", "zscore",
                "z_score", "percentile", "min_rate", "max_rate", "sample_size",
                "samples", "window_start", "window_end", "updated_at",
            )
            if not _is_missing(_first_present(payload, key, f"feature_map.funding.statistics.{key}"))
        }
        sections["statistics"] = flat or None

    # Single-model events from FundingAnalyzer._publish_model_event use
    # FundingEventType values in the topic and `funding.<event_type>` feature_map.
    topic = str(payload.get("event_name") or payload.get("topic") or payload.get("source_topic") or "").lower()
    event_type = str(payload.get("event_type") or payload.get("type") or "").lower()
    if sections["pressure"] is None and ("pressure" in topic or event_type == "pressure"):
        sections["pressure"] = dict(payload)
    if sections["regime"] is None and ("regime" in topic or event_type == "regime"):
        sections["regime"] = dict(payload)
    if sections["extreme"] is None and ("extreme" in topic or event_type == "extreme"):
        sections["extreme"] = dict(payload)
    if sections["divergence"] is None and ("divergence" in topic or event_type == "divergence"):
        sections["divergence"] = dict(payload)
    if sections["flip"] is None and ("flip" in topic or event_type == "flip"):
        sections["flip"] = dict(payload)
    if sections["signal"] is None and ("signal" in topic or event_type == "signal"):
        sections["signal"] = dict(payload)

    aliases = {
        "snapshot": ("funding_snapshot",),
        "statistics": ("stats", "funding_statistics"),
        "regime": ("regime_state", "funding_regime", "funding_regime_state"),
        "pressure": ("pressure_state", "funding_pressure", "funding_pressure_state"),
        "extreme": ("extreme_event", "funding_extreme", "funding_extreme_event"),
        "divergence": ("divergence_event", "funding_divergence", "funding_divergence_event"),
        "flip": ("flip_event", "funding_flip", "funding_flip_event"),
        "signal": ("funding_signal", "analytics_signal", "setup"),
    }

    for name, raw_section in sections.items():
        if not raw_section:
            continue
        normalized = _normalize_funding_section(name, raw_section, payload=payload, feature_map=feature_map)
        _set_alias(payload, name, normalized, *aliases[name])
        _set_feature(feature_map, f"funding.{name}", normalized)

    normalized_sections = {name: payload.get(name) for name in sections if isinstance(payload.get(name), Mapping)}

    def add(name: str, value: Any) -> None:
        _set_feature(feature_map, name, value)

    snapshot = normalized_sections.get("snapshot", {})
    statistics = normalized_sections.get("statistics", {})
    regime = normalized_sections.get("regime", {})
    pressure = normalized_sections.get("pressure", {})
    extreme = normalized_sections.get("extreme", {})
    divergence = normalized_sections.get("divergence", {})
    flip = normalized_sections.get("flip", {})
    signal = normalized_sections.get("signal", {})

    for name, source in (
        ("funding.snapshot", snapshot),
        ("funding.statistics", statistics),
        ("funding.regime", regime),
        ("funding.pressure", pressure),
        ("funding.extreme", extreme),
        ("funding.divergence", divergence),
        ("funding.flip", flip),
        ("funding.signal", signal),
    ):
        if source:
            add(name, source)

    add("funding.rate", _first_present(snapshot, "funding_rate", "current_rate", "rate"))
    add("funding.funding_rate", _first_present(snapshot, "funding_rate", "current_rate", "rate"))
    add("funding.predicted_funding_rate", _first_present(snapshot, "predicted_funding_rate", "predicted_rate", "next_funding_rate"))
    add("funding.current_price", _first_present(snapshot, "mark_price", "current_price", "reference_price"))
    add("funding.mark_price", _first_present(snapshot, "mark_price", "current_price", "reference_price"))
    add("funding.index_price", _first_present(snapshot, "index_price"))
    add("funding.open_interest", _first_present(snapshot, "open_interest"))

    add("funding.statistics.zscore", _first_present(statistics, "zscore", "z_score"))
    add("funding.statistics.percentile", _first_present(statistics, "percentile"))
    add("funding.statistics.sample_size", _first_present(statistics, "sample_size", "samples"))

    add("funding.regime.confidence", _first_present(regime, "confidence", "score"))
    add("funding.regime.bias", _first_present(regime, "bias", "direction"))
    add("funding.regime.type", _first_present(regime, "type", "regime", "name"))

    add("funding.pressure.score", _first_present(pressure, "score", "pressure_score"))
    add("funding.pressure.level", _first_present(pressure, "level", "pressure_level"))
    add("funding.pressure.direction", _first_present(pressure, "direction", "pressure_direction", "bias"))
    add("funding.pressure.squeeze_probability", _first_present(pressure, "squeeze_probability"))
    add("funding.pressure.mean_reversion_probability", _first_present(pressure, "mean_reversion_probability", "reversion_probability"))

    add("funding.extreme.type", _first_present(extreme, "type", "extreme_type"))
    add("funding.extreme.severity", _first_present(extreme, "severity", "score"))
    add("funding.extreme.confidence", _first_present(extreme, "confidence", "severity"))
    add("funding.extreme.reversal_risk", _first_present(extreme, "reversal_risk", "is_reversal_risk"))
    add("funding.extreme.squeeze_risk", _first_present(extreme, "squeeze_risk", "is_squeeze_risk"))
    add("funding.extreme.mean_reversion_probability", _first_present(extreme, "mean_reversion_probability", "reversion_probability"))
    add("funding.extreme.squeeze_probability", _first_present(extreme, "squeeze_probability"))

    add("funding.divergence.type", _first_present(divergence, "type", "divergence_type"))
    add("funding.divergence.confidence", _first_present(divergence, "confidence", "score"))
    add("funding.divergence.score", _first_present(divergence, "score", "confidence", "signed_score"))
    add("funding.divergence.bias", _first_present(divergence, "bias", "direction", "side"))

    add("funding.flip.type", _first_present(flip, "type", "flip_type"))
    add("funding.flip.confidence", _first_present(flip, "confidence", "score"))

    add("funding.signal.type", _first_present(signal, "type", "signal_type"))
    add("funding.signal.score", _first_present(signal, "score", "signed_score"))
    add("funding.signal.confidence", _first_present(signal, "confidence"))
    add("funding.signal.bias", _first_present(signal, "bias", "direction", "side"))


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



def _ensure_price_action_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}
        payload["feature_map"] = feature_map

    state = _mapping_from(payload, "state", "snapshot", "price_action", "price_action_state", "composite") or {}
    if state:
        _set_alias(payload, "state", state, "snapshot", "price_action_state", "composite")
        _set_feature(feature_map, "price_action.state", state)
        _set_feature(feature_map, "price_action.composite", state)

    sections = {
        "market_structure": _mapping_from(payload, "market_structure", "state.market_structure", "snapshot.market_structure"),
        "support_resistance": _mapping_from(payload, "support_resistance", "state.support_resistance", "snapshot.support_resistance"),
        "fair_value_gap": _mapping_from(payload, "fair_value_gap", "fvg", "state.fair_value_gap", "snapshot.fair_value_gap"),
        "liquidity_levels": _mapping_from(payload, "liquidity_levels", "state.liquidity_levels", "snapshot.liquidity_levels"),
        "trend": _mapping_from(payload, "trend", "state.trend", "snapshot.trend"),
        "signal": _mapping_from(payload, "signal", "price_action_signal", "setup"),
    }

    # Composite state often carries nested sections only under state.*.  Lift them
    # to stable strategy feature keys so price_action strategies do not need to
    # know analytics payload internals.
    for name, section in sections.items():
        if not section:
            continue
        if name == "signal":
            section.setdefault("detected", True)
            section.setdefault("origin", "price_action")
        _set_alias(payload, name, section, f"price_action_{name}")
        _set_feature(feature_map, f"price_action.{name}", section)
        for field in (
            "bias", "direction", "trend", "regime", "score", "confidence", "strength",
            "last_price", "current_price", "updated_at", "event_type", "detected",
        ):
            _set_feature(feature_map, f"price_action.{name}.{field}", section.get(field))

    current_price = _first_present(
        payload,
        "current_price", "last_price", "price", "close", "reference_price",
        "state.current_price", "state.last_price", "state.price", "state.close",
        "state.trend.last_price", "state.market_structure.last_price",
    )
    if current_price is not None:
        payload.setdefault("current_price", current_price)
        payload.setdefault("last_price", current_price)
        payload.setdefault("reference_price", current_price)
        payload.setdefault("price", current_price)
        _set_feature(feature_map, "price_action.current_price", current_price)

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


def _ensure_spoofing_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map

    signal = _mapping_from(payload, "signal", "spoofing_signal", "setup", "event") or {}
    features = _mapping_from(payload, "features", "spoofing_features", "metrics") or {}
    detector_results = _mapping_from(payload, "detector_results", "detectors", "results") or {}

    # Detector events are often flat. Promote their fields into the canonical
    # spoofing.signal/features contract consumed by strategy/processor.py.
    for key in (
        "type", "spoofing_type", "pattern", "side", "direction", "severity",
        "status", "score", "confidence", "price_level", "wall_id", "event_time",
    ):
        value = _first_present(payload, key)
        if value is not None:
            target = "type" if key == "spoofing_type" else key
            if target == "direction":
                target = "side"
            signal.setdefault(target, _side_to_strategy(value) if target == "side" else value)

    for key in (
        "pull_ratio", "fill_ratio", "price_reaction_bps",
        "signed_price_reaction_bps", "lifetime_ms", "wall_notional",
        "pulled_notional", "cancel_to_fill_ratio", "distance_from_mid_bps",
        "layer_count", "layer_price_span_bps", "pressure_flip_strength",
    ):
        value = _first_present(payload, key)
        if value is not None:
            features.setdefault(key, value)

    if signal:
        signal.setdefault("detected", True)
        signal.setdefault("origin", "spoofing")
        _set_alias(payload, "signal", signal, "spoofing_signal", "setup")
        _set_feature(feature_map, "spoofing.signal", signal)
        for field in ("type", "pattern", "side", "severity", "status", "score", "confidence", "price_level", "wall_id", "event_time"):
            _set_feature(feature_map, f"spoofing.{field}", signal.get(field))
            _set_feature(feature_map, f"spoofing.signal.{field}", signal.get(field))

    if features:
        _set_alias(payload, "features", features, "spoofing_features", "metrics")
        _set_feature(feature_map, "spoofing.features", features)
        for field in ("pull_ratio", "fill_ratio", "price_reaction_bps", "signed_price_reaction_bps", "lifetime_ms", "wall_notional", "pulled_notional", "cancel_to_fill_ratio", "distance_from_mid_bps", "layer_count", "layer_price_span_bps", "pressure_flip_strength"):
            _set_feature(feature_map, f"spoofing.features.{field}", features.get(field))

    if detector_results:
        _set_alias(payload, "detector_results", detector_results, "detectors", "results")
        _set_feature(feature_map, "spoofing.detector_results", detector_results)


def _ensure_spreads_contract(payload: dict[str, Any]) -> None:
    feature_map = payload.setdefault("feature_map", {})
    if not isinstance(feature_map, dict):
        feature_map = {}; payload["feature_map"] = feature_map

    snapshot = _mapping_from(payload, "snapshot", "spread_snapshot", "basis_snapshot")
    signal = _mapping_from(payload, "signal", "spread_signal", "analytics_signal", "setup")
    opportunity = _mapping_from(payload, "opportunity", "spread_opportunity", "arb_opportunity")

    flat: dict[str, Any] = {}
    for key in (
        "type", "spread_type", "symbol", "exchange_a", "exchange_b",
        "market_type_a", "market_type_b", "exchange_symbol_a",
        "exchange_symbol_b", "spread_bps", "basis",
        "funding_adjusted_spread", "net_edge", "net_edge_bps",
        "zscore", "regime", "direction", "signal_type", "quote_validity",
        "has_edge", "confidence", "opportunity_key", "opportunity_status",
        "persistence_ms", "buy_exchange", "sell_exchange", "buy_market_type",
        "sell_market_type", "instrument_type",
    ):
        value = _first_present(payload, key)
        if value is not None:
            flat[key] = value

    if snapshot is None and flat:
        snapshot = dict(flat)
    if signal is None and any(key in flat for key in ("direction", "signal_type", "confidence", "score", "has_edge")):
        signal = {k: v for k, v in flat.items() if k in {"direction", "signal_type", "confidence", "has_edge", "net_edge", "net_edge_bps", "zscore"}}
    if opportunity is None and any(key in flat for key in ("opportunity_key", "opportunity_status", "net_edge", "net_edge_bps", "buy_exchange", "sell_exchange")):
        opportunity = {k: v for k, v in flat.items() if k in {"opportunity_key", "opportunity_status", "net_edge", "net_edge_bps", "buy_exchange", "sell_exchange", "buy_market_type", "sell_market_type", "persistence_ms"}}

    if snapshot:
        _set_alias(payload, "snapshot", snapshot, "spread_snapshot", "basis_snapshot")
        _set_feature(feature_map, "spreads.snapshot", snapshot)
    if signal:
        signal.setdefault("detected", True)
        signal.setdefault("origin", "spreads")
        _set_alias(payload, "signal", signal, "spread_signal", "analytics_signal", "setup")
        _set_feature(feature_map, "spreads.signal", signal)
    if opportunity:
        opportunity.setdefault("detected", True)
        _set_alias(payload, "opportunity", opportunity, "spread_opportunity", "arb_opportunity")
        _set_feature(feature_map, "spreads.opportunity", opportunity)

    merged = {}
    for container in (snapshot, signal, opportunity, flat):
        if isinstance(container, Mapping):
            merged.update(container)
    for key, value in merged.items():
        _set_feature(feature_map, f"spreads.{key}", value)


def ensure_domain_strategy_contract(payload: dict[str, Any], *, topic: str | None = None, source: str | None = None, domain: str | None = None) -> None:
    if topic:
        payload.setdefault("event_name", topic)
        payload.setdefault("topic", topic)
        payload.setdefault("source_topic", topic)
    if source:
        payload.setdefault("source", source)
    resolved = domain or _infer_domain(topic, source, payload)
    if resolved == "orderflow":
        _ensure_orderflow_contract(payload)
    elif resolved == "funding":
        _ensure_funding_contract(payload)
    elif resolved == "open_interest":
        _ensure_open_interest_contract(payload)
    elif resolved == "price_action":
        _ensure_price_action_contract(payload)
    elif resolved == "liquidations":
        _ensure_liquidations_contract(payload)
    elif resolved == "whales":
        _ensure_whales_contract(payload)
    elif resolved == "liquidity":
        _ensure_liquidity_contract(payload)
    elif resolved == "spoofing":
        _ensure_spoofing_contract(payload)
    elif resolved == "spreads":
        _ensure_spreads_contract(payload)

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