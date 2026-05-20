from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from strategy.config import StrategyConfig
from strategy.processor import SignalNormalizer


# =============================================================================
# Generic helpers
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _stringify(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if is_dataclass(value):
        return asdict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted

    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        converted = to_payload()
        if isinstance(converted, dict):
            return converted

    return {}


def _set_if_exists(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _make_config() -> StrategyConfig:
    config = StrategyConfig()
    routing = getattr(config, "routing", None)
    if routing is not None:
        _set_if_exists(routing, "enabled", True)
    return config


def _make_normalizer(config: StrategyConfig) -> SignalNormalizer:
    try:
        return SignalNormalizer(config=config)
    except TypeError:
        return SignalNormalizer()


def _call_normalize(
    normalizer: SignalNormalizer,
    *,
    event_name: str,
    payload: dict[str, Any],
    timestamp: datetime,
) -> Any:
    normalize = getattr(normalizer, "normalize_event", None)
    if not callable(normalize):
        normalize = getattr(normalizer, "normalize", None)

    assert callable(normalize), (
        "SignalNormalizer must expose normalize_event(...) or normalize(...)."
    )

    signature = inspect.signature(normalize)
    parameters = signature.parameters

    kwargs: dict[str, Any] = {}

    if "event_name" in parameters:
        kwargs["event_name"] = event_name
    elif "topic" in parameters:
        kwargs["topic"] = event_name
    elif "event" in parameters:
        kwargs["event"] = event_name

    if "payload" in parameters:
        kwargs["payload"] = payload

    if "timestamp" in parameters:
        kwargs["timestamp"] = timestamp

    if kwargs:
        try:
            return normalize(**kwargs)
        except TypeError:
            pass

    for args in (
        (event_name, payload, timestamp),
        (event_name, payload),
        (payload,),
    ):
        try:
            return normalize(*args)
        except TypeError:
            continue

    raise AssertionError(
        "Unable to call SignalNormalizer.normalize_event/normalize with supported signatures"
    )


def _normalized_source(normalized: Any) -> str:
    source = getattr(normalized, "source", None)
    if source is None:
        source = getattr(normalized, "feature_source", None)
    return _stringify(source).lower()


def _normalized_domain_data(normalized: Any) -> dict[str, Any]:
    domain_data = getattr(normalized, "domain_data", None)
    if isinstance(domain_data, dict):
        return domain_data

    domain = getattr(normalized, "domain", None)
    if isinstance(domain, dict):
        return domain

    data = _to_dict(normalized)
    for key in ("domain_data", "domain"):
        value = data.get(key)
        if isinstance(value, dict):
            return value

    return {}


def _normalized_feature_names(normalized: Any) -> set[str]:
    names: set[str] = set()

    feature_map = getattr(normalized, "feature_map", None)
    if isinstance(feature_map, dict):
        names.update(str(key) for key in feature_map.keys())

    features = getattr(normalized, "features", None)
    if isinstance(features, dict):
        names.update(str(key) for key in features.keys())

    if isinstance(features, list):
        for item in features:
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            if name is not None:
                names.add(str(name))

    data = _to_dict(normalized)
    for key in ("features", "feature_map"):
        value = data.get(key)
        if isinstance(value, dict):
            names.update(str(item) for item in value.keys())
        elif isinstance(value, list):
            for item in value:
                item_name = None
                if isinstance(item, dict):
                    item_name = item.get("name")
                else:
                    item_name = getattr(item, "name", None)
                if item_name is not None:
                    names.add(str(item_name))

    return names


def _categories_for_event(config: StrategyConfig, event_name: str) -> list[Any]:
    routing = getattr(config, "routing", None)
    assert routing is not None, "StrategyConfig.routing is missing"

    categories_for_event = getattr(routing, "categories_for_event", None)
    assert callable(categories_for_event), (
        "RoutingConfig.categories_for_event(event_name) is missing"
    )

    result = categories_for_event(event_name)
    if result is None:
        return []
    return list(result)


def _category_names(categories: list[Any]) -> set[str]:
    return {str(getattr(category, "name", category)).upper() for category in categories}


def _print_normalized_report(
    *,
    case_name: str,
    event_name: str,
    normalized: Any,
    expected_features: set[str],
    expected_domain_keys: set[str],
) -> None:
    source = _normalized_source(normalized)
    domain_data = _normalized_domain_data(normalized)
    feature_names = _normalized_feature_names(normalized)

    print("")
    print(f"========== NORMALIZER REPORT: {case_name} ==========")
    print("event:", event_name)
    print("source:", source)
    print("domain_keys:", sorted(domain_data.keys()))
    print("feature_count:", len(feature_names))
    print("expected_features:", sorted(expected_features))
    print("missing_features:", sorted(expected_features - feature_names))
    print("expected_domain_keys:", sorted(expected_domain_keys))
    print("missing_domain_keys:", sorted(expected_domain_keys - set(domain_data.keys())))
    print("sample_features:", sorted(feature_names)[:80])
    print("====================================================")
    print("")


# =============================================================================
# Representative analytics payloads
# =============================================================================


def _sample_open_interest_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "timestamp": _utcnow(),
        "confidence": 0.91,
        "score": 0.88,
        "analysis": {
            "features": {
                "oi_delta_pct": -0.12,
                "price_delta_pct": -0.035,
                "oi_pressure_score": -0.62,
                "liquidation_pressure": -0.72,
            },
            "regime": {"regime": "capitulation", "confidence": 0.91, "score": 0.88},
            "anomaly": {
                "detected": True,
                "anomaly_type": "liquidation_driven_oi_drop",
                "confidence": 0.91,
                "score": 0.88,
            },
            "divergence": {
                "detected": True,
                "divergence_type": "price_down_oi_down",
                "confidence": 0.80,
                "score": 0.75,
            },
        },
        "features": {
            "oi_delta_pct": -0.12,
            "price_delta_pct": -0.035,
            "oi_pressure_score": -0.62,
            "liquidation_pressure": -0.72,
        },
        "regime": {"regime": "capitulation", "confidence": 0.91, "score": 0.88},
        "anomaly": {
            "detected": True,
            "anomaly_type": "liquidation_driven_oi_drop",
            "confidence": 0.91,
            "score": 0.88,
        },
        "divergence": {
            "detected": True,
            "divergence_type": "price_down_oi_down",
            "confidence": 0.80,
            "score": 0.75,
        },
    }


def _sample_funding_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "timestamp": _utcnow(),
        "confidence": 0.87,
        "snapshot": {"funding_rate": -0.00035, "predicted_rate": -0.00045},
        "statistics": {"zscore": -2.4, "percentile": 0.04},
        "regime": {"regime": "extreme_negative", "confidence": 0.86},
        "pressure": {"score": 0.78, "level": "crowded_shorts", "direction": "long"},
        "extreme": {
            "type": "negative_extreme",
            "severity": 0.82,
            "mean_reversion_probability": 0.74,
            "squeeze_probability": 0.62,
        },
        "divergence": {"type": "funding_price_divergence", "confidence": 0.75, "score": 0.72},
        "flip": {"type": "negative_to_positive", "confidence": 0.70},
        "signal": {"signal_type": "squeeze_setup", "bias": "long", "score": 0.81, "confidence": 0.87},
    }


def _sample_orderflow_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "timestamp": _utcnow(),
        "confidence": 0.84,
        "composite": {
            "price_change_pct": 0.008,
            "trades_count": 1200,
            "total_volume": 480.0,
            "total_notional": 31_000_000.0,
            "cvd_delta_ratio": 0.68,
            "cvd_change_pct": 0.05,
            "cvd_slope": 0.72,
            "volume_delta_ratio": 0.64,
            "volume_delta": 140.0,
            "aggressive_buy_ratio": 0.71,
            "aggressive_sell_ratio": 0.29,
            "orderbook_imbalance_ratio": 0.63,
            "orderbook_imbalance_diff": 250.0,
        },
        "cvd": {"delta_ratio": 0.68, "cvd_change_pct": 0.05, "cvd_slope": 0.72, "price_change_pct": 0.008},
        "volume_delta": {"delta_ratio": 0.64, "volume_delta": 140.0},
        "aggressive_trades": {"buy_ratio": 0.71, "sell_ratio": 0.29},
        "orderbook_imbalance": {"ratio": 0.63, "diff": 250.0},
    }


def _sample_liquidations_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "timestamp": _utcnow(),
        "confidence": 0.83,
        "cascade": {
            "confidence": 0.83,
            "intensity_score": 0.79,
            "direction": "sell",
            "severity": "high",
            "continuation_bias": 0.31,
            "exhaustion_bias": 0.76,
            "total_notional_usd": 2_400_000.0,
            "event_count": 38,
        },
        "exhaustion": {"confidence": 0.77, "exhaustion_bias": 0.76, "bias_delta": 0.42, "confirmed": True},
        "squeeze": {"confirmed": True, "score": 0.74, "direction": "long"},
        "cluster": {
            "duration_seconds": 38,
            "avg_notional_per_event": 63_000.0,
            "side_imbalance_ratio": 0.81,
            "event_imbalance_ratio": 0.72,
            "acceleration_ratio": 1.8,
        },
    }


def _sample_liquidity_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "timestamp": _utcnow(),
        "confidence": 0.80,
        "snapshot": {
            "current_price": 65000.0,
            "above_liquidity_score": 0.72,
            "below_liquidity_score": 0.35,
            "pressure_score": 0.66,
            "bias": "up",
            "sweep_risk": {"up": 0.78, "down": 0.20},
            "magnet": {"up": 0.71, "down": 0.18},
            "nearest_above_level": {"price": 65500.0, "strength": 0.80},
            "nearest_below_level": {"price": 64200.0, "strength": 0.50},
            "active_levels": [{"price": 65500.0}],
            "stop_clusters": [{"price": 65600.0}],
            "zones": [{"low": 65400.0, "high": 65700.0}],
        },
        "signal": {"bias": "up", "score": 0.75, "confidence": 0.80},
    }


def _sample_price_action_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "timestamp": _utcnow(),
        "confidence": 0.82,
        "state": {
            "current_price": 65000.0,
            "market_structure": {"last_break_event": {"type": "bos", "side": "bullish"}, "mtf_alignment": 0.76},
            "support_resistance": {
                "last_event": {"type": "level_retested", "side": "support"},
                "nearest_support": {"price": 64200.0},
                "nearest_resistance": {"price": 65700.0},
            },
            "fair_value_gap": {
                "last_event": {"type": "fvg_retested", "side": "bullish"},
                "nearest_bullish_gap": {"low": 64600.0, "high": 64850.0},
            },
            "trend": {
                "last_signal": {"type": "trend_continuation", "side": "long"},
                "overall_trend_score": 0.78,
                "higher_timeframe_alignment": 0.70,
            },
            "liquidity_levels": {"last_event": {"type": "liquidity_swept", "side": "down"}},
        },
    }


def _sample_spoofing_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "timestamp": _utcnow(),
        "confidence": 0.86,
        "composite": {
            "signal": {
                "spoofing_type": "order_pull",
                "pattern": "fake_liquidity",
                "side": "ask",
                "severity": "high",
                "status": "detected",
                "score": 0.84,
                "confidence": 0.86,
            },
            "features": {
                "pull_ratio": 0.88,
                "fill_ratio": 0.08,
                "price_reaction_bps": 3.2,
                "lifetime_ms": 1300,
                "wall_notional": 1_200_000.0,
                "pulled_notional": 1_050_000.0,
                "cancel_to_fill_ratio": 0.91,
                "distance_from_mid_bps": 1.5,
                "layer_count": 4,
                "layer_price_span_bps": 3.1,
                "pressure_flip_strength": 0.72,
            },
            "detector_results": {
                "order_pull": {"passed": True, "score": 0.84, "confidence": 0.86},
                "fake_liquidity": {"passed": True, "score": 0.80, "confidence": 0.82},
            },
        },
    }


def _sample_spreads_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "timestamp": _utcnow(),
        "confidence": 0.81,
        "snapshot": {
            "spread_type": "spot_futures",
            "symbol": "BTCUSDT",
            "exchange_a": "binance",
            "exchange_b": "binance",
            "market_type_a": "spot",
            "market_type_b": "usdm_futures",
            "spread_bps": 38.0,
            "basis": 0.0038,
            "funding_adjusted_spread": 31.0,
            "net_edge": 26.0,
            "net_edge_bps": 26.0,
            "zscore": 2.4,
            "regime": "elevated",
            "direction": "widening",
            "quote_validity": "valid",
            "has_edge": True,
            "confidence": 0.81,
        },
        "signal": {"signal_type": "mean_reversion", "direction": "narrowing", "confidence": 0.82},
        "opportunity": {
            "opportunity_key": "binance:spot:binance:usdm_futures:BTCUSDT",
            "status": "active",
            "buy_exchange": "binance",
            "sell_exchange": "binance",
            "net_edge_bps": 26.0,
            "persistence_ms": 1500,
        },
    }


def _sample_whales_payload() -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1m",
        "timestamp": _utcnow(),
        "confidence": 0.85,
        "composite": {
            "activity": {"notional": 900_000.0, "trade_count": 6, "side": "buy", "score": 0.78},
            "pressure": {"side": "buy", "score": 0.74, "imbalance_ratio": 0.70},
            "large_trade": {"notional": 500_000.0, "zscore": 2.5, "side": "buy"},
            "cluster": {"score": 0.71, "side": "buy", "continuation_probability": 0.73, "exhaustion_probability": 0.21},
            "liquidation_context": {"side": "sell", "notional": 450_000.0, "strength": 0.66},
            "exhaustion": {"probability": 0.24, "side": "sell"},
        },
    }


CONTRACT_CASES = [
    {
        "name": "open_interest",
        "event": "analytics.oi.capitulation.detected",
        "source": "OPEN_INTEREST",
        "payload": _sample_open_interest_payload,
        "expected_features": {"open_interest.features", "open_interest.regime", "open_interest.anomaly", "open_interest.divergence"},
        "expected_domain_keys": {"features", "regime", "anomaly", "divergence"},
    },
    {
        "name": "funding",
        "event": "analytics.funding.signal",
        "source": "FUNDING",
        "payload": _sample_funding_payload,
        "expected_features": {"funding.snapshot", "funding.statistics", "funding.regime", "funding.pressure", "funding.extreme", "funding.divergence", "funding.flip", "funding.signal"},
        "expected_domain_keys": {"snapshot", "statistics", "regime", "pressure", "extreme", "divergence", "flip", "signal"},
    },
    {
        "name": "orderflow",
        "event": "analytics.orderflow.updated",
        "source": "ORDERFLOW",
        "payload": _sample_orderflow_payload,
        "expected_features": {"orderflow.composite", "orderflow.cvd", "orderflow.cvd.delta_ratio", "orderflow.volume_delta", "orderflow.volume_delta.delta_ratio", "orderflow.aggressive_trades", "orderflow.orderbook_imbalance"},
        "expected_domain_keys": {"composite", "cvd", "volume_delta", "aggressive_trades", "orderbook_imbalance"},
    },
    {
        "name": "liquidations",
        "event": "analytics.liquidations.cascade_detected",
        "source": "LIQUIDATIONS",
        "payload": _sample_liquidations_payload,
        "expected_features": {"liquidations.cascade", "liquidations.cascade.confidence", "liquidations.cascade.intensity_score", "liquidations.exhaustion", "liquidations.squeeze", "liquidations.cluster"},
        "expected_domain_keys": {"cascade", "exhaustion", "squeeze", "cluster"},
    },
    {
        "name": "liquidity",
        "event": "analytics.liquidity.map.updated",
        "source": "LIQUIDITY",
        "payload": _sample_liquidity_payload,
        "expected_features": {"liquidity.snapshot", "liquidity.map.snapshot", "liquidity.current_price", "liquidity.above_liquidity_score", "liquidity.below_liquidity_score", "liquidity.pressure_score", "liquidity.bias"},
        "expected_domain_keys": {"snapshot", "signal"},
    },
    {
        "name": "price_action",
        "event": "analytics.price_action.market_structure.updated",
        "source": "PRICE_ACTION",
        "payload": _sample_price_action_payload,
        "expected_features": {"price_action.composite", "price_action.market_structure", "price_action.support_resistance", "price_action.fair_value_gap", "price_action.trend", "price_action.current_price"},
        "expected_domain_keys": {"state", "market_structure", "support_resistance", "fair_value_gap", "trend"},
    },
    {
        "name": "spoofing",
        "event": "analytics.spoofing.detected",
        "source": "SPOOFING",
        "payload": _sample_spoofing_payload,
        "expected_features": {"spoofing.composite", "spoofing.signal", "spoofing.features", "spoofing.detector_results", "spoofing.score", "spoofing.confidence", "spoofing.features.pull_ratio", "spoofing.features.fill_ratio"},
        "expected_domain_keys": {"composite", "signal", "features", "detector_results"},
    },
    {
        "name": "spreads",
        "event": "analytics.spreads.signal.generated",
        "source": "SPREADS",
        "payload": _sample_spreads_payload,
        "expected_features": {"spreads.snapshot", "spreads.signal", "spreads.opportunity", "spreads.type", "spreads.spread_bps", "spreads.net_edge_bps", "spreads.zscore"},
        "expected_domain_keys": {"snapshot", "signal", "opportunity"},
    },
    {
        "name": "whales",
        "event": "analytics.whales.whale_activity",
        "source": "WHALES",
        "payload": _sample_whales_payload,
        "expected_features": {"whales.composite", "whales.activity", "whales.pressure", "whales.large_trade", "whales.cluster", "whales.liquidation_context", "whales.exhaustion"},
        "expected_domain_keys": {"composite", "activity", "pressure", "large_trade", "cluster", "liquidation_context", "exhaustion"},
    },
]


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda item: item["name"])
def test_routing_maps_real_analytics_topics_to_domain_and_hybrid(case: dict[str, Any]) -> None:
    config = _make_config()
    categories = _categories_for_event(config, case["event"])
    category_names = _category_names(categories)

    print("")
    print(f"========== ROUTING REPORT: {case['name']} ==========")
    print("event:", case["event"])
    print("categories:", sorted(category_names))
    print("expected_domain:", case["source"])
    print("===================================================")
    print("")

    assert case["source"] in category_names, (
        f"RoutingConfig did not route {case['event']} to {case['source']}. "
        f"categories={sorted(category_names)}"
    )
    assert "HYBRID" in category_names, (
        f"RoutingConfig did not route {case['event']} to HYBRID. "
        f"categories={sorted(category_names)}"
    )


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda item: item["name"])
def test_signal_normalizer_builds_domain_contracts_for_real_analytics_topics(case: dict[str, Any]) -> None:
    config = _make_config()
    normalizer = _make_normalizer(config)

    payload = case["payload"]()
    timestamp = payload.get("timestamp") or _utcnow()

    normalized = _call_normalize(
        normalizer,
        event_name=case["event"],
        payload=payload,
        timestamp=timestamp,
    )

    expected_features = set(case["expected_features"])
    expected_domain_keys = set(case["expected_domain_keys"])

    _print_normalized_report(
        case_name=case["name"],
        event_name=case["event"],
        normalized=normalized,
        expected_features=expected_features,
        expected_domain_keys=expected_domain_keys,
    )

    source = _normalized_source(normalized)
    feature_names = _normalized_feature_names(normalized)
    domain_data = _normalized_domain_data(normalized)

    expected_source = str(case["source"]).lower()
    assert expected_source in source, (
        f"Normalizer source mismatch for {case['event']}. "
        f"expected contains={expected_source!r}, actual={source!r}. "
        "Check analytics topic -> FeatureSource mapping."
    )

    missing_features = expected_features - feature_names
    assert not missing_features, (
        f"Normalizer did not build required contract FeatureSnapshot names for {case['name']}. "
        f"missing={sorted(missing_features)}. "
        "Check _build_contract_features(...) branch and _build_<domain>_contract_features(...)."
    )

    missing_domain_keys = expected_domain_keys - set(domain_data.keys())
    assert not missing_domain_keys, (
        f"Normalizer did not build required domain_data aliases for {case['name']}. "
        f"missing={sorted(missing_domain_keys)}. "
        "Check normalize_event(...) calls _augment_domain_data_contracts(...), "
        "and check _augment_<domain>_domain_data(...)."
    )


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda item: item["name"])
def test_hybrid_summary_features_are_present_for_each_domain_event(case: dict[str, Any]) -> None:
    config = _make_config()
    normalizer = _make_normalizer(config)

    payload = case["payload"]()
    timestamp = payload.get("timestamp") or _utcnow()

    normalized = _call_normalize(
        normalizer,
        event_name=case["event"],
        payload=payload,
        timestamp=timestamp,
    )

    feature_names = _normalized_feature_names(normalized)
    expected_hybrid = {
        "hybrid.dominant_side",
        "hybrid.alignment_score",
        "hybrid.conflict_score",
        "hybrid.confluence_score",
        "hybrid.confidence",
        "hybrid.votes",
    }

    print("")
    print(f"========== HYBRID SUMMARY REPORT: {case['name']} ==========")
    print("event:", case["event"])
    print("missing_hybrid:", sorted(expected_hybrid - feature_names))
    print("==========================================================")
    print("")

    missing_hybrid = expected_hybrid - feature_names
    assert not missing_hybrid, (
        f"Hybrid summary features are missing for {case['name']}. "
        f"missing={sorted(missing_hybrid)}. "
        "If you intentionally build hybrid summary in StrategyContextBuilder instead of SignalNormalizer, "
        "move this assertion to the StrategyContextBuilder-level test."
    )


def test_normalizer_diagnostic_matrix_for_all_domains() -> None:
    config = _make_config()
    normalizer = _make_normalizer(config)
    rows: list[dict[str, Any]] = []

    for case in CONTRACT_CASES:
        payload = case["payload"]()
        timestamp = payload.get("timestamp") or _utcnow()
        normalized = _call_normalize(
            normalizer,
            event_name=case["event"],
            payload=payload,
            timestamp=timestamp,
        )

        feature_names = _normalized_feature_names(normalized)
        domain_data = _normalized_domain_data(normalized)
        categories = _category_names(_categories_for_event(config, case["event"]))
        expected_features = set(case["expected_features"])
        expected_domain_keys = set(case["expected_domain_keys"])

        rows.append(
            {
                "domain": case["name"],
                "event": case["event"],
                "source": _normalized_source(normalized),
                "categories": sorted(categories),
                "missing_features": sorted(expected_features - feature_names),
                "missing_domain": sorted(expected_domain_keys - set(domain_data.keys())),
                "missing_hybrid": sorted(
                    {
                        "hybrid.dominant_side",
                        "hybrid.alignment_score",
                        "hybrid.conflict_score",
                        "hybrid.confluence_score",
                        "hybrid.confidence",
                        "hybrid.votes",
                    }
                    - feature_names
                ),
            }
        )

    print("")
    print("========== STRATEGY ANALYTICS PIPELINE DIAGNOSTIC MATRIX ==========")
    for row in rows:
        print("")
        print("domain:", row["domain"])
        print("event:", row["event"])
        print("source:", row["source"])
        print("categories:", row["categories"])
        print("missing_features:", row["missing_features"])
        print("missing_domain:", row["missing_domain"])
        print("missing_hybrid:", row["missing_hybrid"])
    print("==================================================================")
    print("")

    failing = [row for row in rows if row["missing_features"] or row["missing_domain"]]
    assert not failing, (
        "Some analytics domains are still not normalized into StrategyContext contracts. "
        "Run with -s and inspect the diagnostic matrix above."
    )


@pytest.mark.asyncio
async def test_optional_signal_processor_smoke_diagnostic() -> None:
    try:
        from strategy.processor import SignalProcessor
        from strategy.presets import build_default_strategy_registry
        from core.event_bus import EventBus
    except Exception as exc:
        pytest.xfail(f"Optional processor smoke imports failed: {exc!r}")

    config = _make_config()

    try:
        event_bus = EventBus()
    except Exception:
        event_bus = None

    try:
        registry = build_default_strategy_registry(config=config, event_bus=event_bus, scheduler=None)
    except TypeError:
        try:
            registry = build_default_strategy_registry(config=config)
        except TypeError:
            registry = build_default_strategy_registry()

    try:
        processor = SignalProcessor(config=config, registry=registry, event_bus=event_bus)
    except TypeError:
        try:
            processor = SignalProcessor(config=config, strategy_registry=registry, event_bus=event_bus)
        except TypeError as exc:
            pytest.xfail(f"SignalProcessor constructor signature differs: {exc!r}")

    process_event = getattr(processor, "process_event", None)
    if not callable(process_event):
        pytest.xfail("SignalProcessor.process_event(...) is not available")

    payload = _sample_open_interest_payload()
    kwargs = {
        "event_name": "analytics.oi.capitulation.detected",
        "payload": payload,
        "timestamp": payload.get("timestamp"),
        "emit": False,
    }

    signature = inspect.signature(process_event)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    try:
        batch = await process_event(**filtered_kwargs)
    except TypeError:
        batch = await process_event("analytics.oi.capitulation.detected", payload)

    print("")
    print("========== OPTIONAL PROCESSOR SMOKE ==========")
    print("batch:", batch)
    for attr in (
        "accepted",
        "emitted",
        "reasons",
        "raw_signal_count",
        "final_signal_count",
        "risk_payload_count",
        "selected_names",
        "rejected_signals",
    ):
        if hasattr(batch, attr):
            print(f"{attr}:", getattr(batch, attr))
    print("=============================================")
    print("")

    assert batch is not None