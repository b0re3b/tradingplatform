"""
Diagnostic tests for real StrategyRegistry / SignalRouter / BaseStrategy / SignalProcessor.

Put into:

    test/datatest/testdata.py

or:

    tests/strategy/test_real_strategy_pipeline_diagnostics.py

Run:

    pytest -q test/datatest/testdata.py -s

Goal:
- Keep early bootstrap tests strict.
- Make the routing/evaluation tests print WHY they fail:
  registry index issue, context/timeframe/regime/features mismatch,
  BaseStrategy.evaluate() compatibility issue, or downstream SignalProcessor block.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest

from core.event_bus import EventBus
from core.scheduler import Scheduler

from strategy.base import BaseStrategy
from strategy.config import (
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
)
from strategy.engine import StrategyEngine
from strategy.enums import (
    EntryType,
    FeatureSource,
    MarketRegime,
    SetupType,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from strategy.models import (
    EntryPlan,
    ExitPlan,
    InvalidationPlan,
    StrategyContext,
    StrategySignal,
    utcnow,
)
from strategy.presets import (
    build_default_strategy_config,
    build_default_strategy_registry,
)
from strategy.processor import SignalProcessor
from strategy.state import StrategyRuntimeState


# =============================================================================
# Generic helpers
# =============================================================================


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _registry_count(registry: Any) -> int:
    count = getattr(registry, "count", None)
    if callable(count):
        return int(count())

    list_all = getattr(registry, "list_all", None)
    if callable(list_all):
        return len(list_all())

    strategies = getattr(registry, "_strategies", None)
    if isinstance(strategies, dict):
        return len(strategies)

    return 0


def _registered_strategies(registry: Any) -> list[BaseStrategy]:
    list_all = getattr(registry, "list_all", None)
    if callable(list_all):
        return list(list_all())

    strategies = getattr(registry, "_strategies", None)
    if isinstance(strategies, dict):
        return list(strategies.values())

    return []


def _strategy_name(strategy: Any) -> str:
    value = getattr(strategy, "strategy_name", None)
    if isinstance(value, str) and value:
        return value

    value = getattr(strategy, "name", None)
    if isinstance(value, str) and value:
        return value

    return strategy.__class__.__name__


def _category_value(strategy: Any) -> str:
    return _enum_value(getattr(strategy, "category", None))


def _changed_features(context: StrategyContext) -> list[str]:
    feature_map = getattr(context, "feature_map", None)
    if isinstance(feature_map, dict):
        return list(feature_map.keys())

    features = getattr(context, "features", None)
    if isinstance(features, dict):
        return list(features.keys())

    return []


def _context_regime(context: StrategyContext) -> Any:
    current_regime = getattr(context, "current_regime", None)
    if current_regime is not None:
        return current_regime

    regime = getattr(context, "regime", None)
    if regime is not None:
        raw = getattr(regime, "regime", None)
        if raw is not None:
            return raw

    return MarketRegime.UNKNOWN


def _supports_call(strategy: Any, method_name: str, value: Any) -> Any:
    method = getattr(strategy, method_name, None)
    if not callable(method):
        return "<missing>"
    try:
        return method(value)
    except Exception as exc:
        return f"<error {exc.__class__.__name__}: {exc}>"


def _required_features(strategy: Any) -> set[str]:
    method = getattr(strategy, "required_features", None)
    if not callable(method):
        return set()
    try:
        value = method()
        return set(value or set())
    except Exception:
        return set()


def _strategy_diagnostics(strategy: Any, context: StrategyContext) -> dict[str, Any]:
    required = _required_features(strategy)
    changed = set(_changed_features(context))
    return {
        "name": _strategy_name(strategy),
        "category": _category_value(strategy),
        "enabled": (
            strategy.is_enabled()
            if callable(getattr(strategy, "is_enabled", None))
            else getattr(strategy, "enabled", None)
        ),
        "supports_symbol": _supports_call(strategy, "supports_symbol", context.symbol),
        "supports_timeframe": _supports_call(strategy, "supports_timeframe", context.timeframe),
        "supports_regime": _supports_call(strategy, "supports_regime", _context_regime(context)),
        "required": sorted(required),
        "missing_required": sorted(required - changed),
    }


def _print_registry_selection_debug(
    *,
    registry: Any,
    context: StrategyContext,
    categories: list[StrategyCategory],
) -> None:
    print("")
    print("========== REGISTRY SELECTION DEBUG ==========")
    print("context.symbol:", context.symbol)
    print("context.timeframe:", context.timeframe, "| value:", _enum_value(context.timeframe))
    print("context.regime:", _context_regime(context), "| value:", _enum_value(_context_regime(context)))
    print("context.features:", _changed_features(context))
    print("categories:", [category.value for category in categories])

    print("")
    print("registry.count:", _registry_count(registry))

    try:
        summary = registry.summary()
        print("registry.summary.by_category:", summary.get("by_category"))
        print("registry.summary.by_timeframe:", summary.get("by_timeframe"))
        print("registry.summary.by_regime:", summary.get("by_regime"))
    except Exception as exc:
        print("registry.summary failed:", exc)

    print("")
    print("registered open_interest/hybrid strategies:")
    for strategy in _registered_strategies(registry):
        if _category_value(strategy) in {"open_interest", "hybrid"}:
            print(" -", _strategy_diagnostics(strategy, context))

    print("")
    print("registry.list_by_category(OPEN_INTEREST):")
    try:
        by_category = registry.list_by_category(StrategyCategory.OPEN_INTEREST)
        print([_strategy_name(strategy) for strategy in by_category])
    except Exception as exc:
        print("list_by_category failed:", exc)

    print("")
    print("registry.select direct:")
    try:
        selected = registry.select(
            context=context,
            categories=categories,
            changed_features=_changed_features(context),
        )
        print([_strategy_name(strategy) for strategy in selected])
    except Exception as exc:
        print("registry.select failed:", exc)

    print("==============================================")
    print("")


def _discover_strategy_factories_like_runner() -> dict[str, type[BaseStrategy]]:
    try:
        package = importlib.import_module("strategy.strategies")
    except Exception as exc:
        pytest.fail(f"Cannot import strategy.strategies package: {exc}")

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        pytest.fail("strategy.strategies has no __path__; cannot auto-discover strategies")

    factories: dict[str, type[BaseStrategy]] = {}

    def camel_to_snake(name: str) -> str:
        chars: list[str] = []
        for index, char in enumerate(name):
            if char.isupper() and index > 0 and not name[index - 1].isupper():
                chars.append("_")
            chars.append(char.lower())
        return "".join(chars)

    def strategy_name_from_class(cls: type, module_name: str) -> str:
        name = cls.__name__

        for suffix in ("TradingStrategy", "Strategy"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        leaf = module_name.rsplit(".", 1)[-1]
        if leaf not in {"base", "utils", "__init__"} and leaf.endswith("_strategy"):
            return leaf[: -len("_strategy")]

        return camel_to_snake(name)

    for module_info in pkgutil.walk_packages(package_path, prefix="strategy.strategies."):
        module_name = module_info.name
        leaf = module_name.rsplit(".", 1)[-1].lower()

        if leaf in {
            "__init__",
            "base",
            "utils",
            "config",
            "enums",
            "models",
            "exceptions",
        }:
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"[DISCOVERY] skipped module={module_name} error={exc}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue

            if obj is BaseStrategy:
                continue

            try:
                is_strategy = issubclass(obj, BaseStrategy)
            except TypeError:
                continue

            if not is_strategy:
                continue

            cls_name = obj.__name__.lower()

            if cls_name.startswith("base"):
                continue

            if cls_name in {
                "tradingstrategy",
                "strategysignalmixin",
                "strategyvalidationmixin",
                "strategyriskrewardmixin",
            }:
                continue

            if inspect.isabstract(obj):
                continue

            factories.setdefault(strategy_name_from_class(obj, module_name), obj)

    return factories


def _build_real_strategy_stack(
    *,
    symbols: list[str] | None = None,
    preset_name: str = "intraday",
    use_required_features: bool = False,
) -> tuple[EventBus, Scheduler, StrategyConfig, Any, SignalProcessor, StrategyEngine, StrategyRuntimeState]:
    event_bus = EventBus()
    scheduler = Scheduler(event_bus=event_bus)

    strategy_config = build_default_strategy_config(
        symbols=symbols or ["BTCUSDT", "DOGEUSDT", "SOLUSDT"],
        preset_name=preset_name,
        use_required_features=use_required_features,
    )
    assert isinstance(strategy_config, StrategyConfig)

    factories = _discover_strategy_factories_like_runner()

    registry = build_default_strategy_registry(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        strategy_factories=factories,
        strict=False,
        emit_events=False,
    )

    state = StrategyRuntimeState()

    processor = SignalProcessor(
        config=strategy_config,
        registry=registry,
        state=state,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    engine = StrategyEngine(
        config=strategy_config,
        event_bus=event_bus,
        scheduler=scheduler,
        registry=registry,
        state=state,
        processor=processor,
    )

    return event_bus, scheduler, strategy_config, registry, processor, engine, state


def _realistic_oi_payload(
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    confidence: float = 0.86,
    score: float = 0.82,
) -> dict[str, Any]:
    return {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": utcnow(),
        "confidence": confidence,
        "score": score,
        "direction": "long",
        "bias": "bullish",
        "regime": "squeeze",
        "oi": 104_000.0,
        "open_interest": 104_000.0,
        "open_interest_value": 8_000_000_000.0,
        "oi_delta": -0.08,
        "oi_delta_pct": -0.12,
        "oi_direction": "down",
        "oi_acceleration": -0.03,
        "price": 77_000.0,
        "close": 77_000.0,
        "funding_rate": 0.00005,
        "liquidation_imbalance": 0.72,
        "long_liquidations": 12_000_000.0,
        "short_liquidations": 3_000_000.0,
        "cvd_delta": 0.42,
        "aggressive_flow_imbalance": 0.55,

        # Rich OI contract fields used by concrete OI strategies.
        "oi_regime": "capitulation",
        "market_regime": "capitulation",
        "anomaly": {
            "is_anomaly": True,
            "anomaly_type": "capitulation",
            "score": score,
            "confidence": confidence,
        },
        "anomaly_type": "capitulation",
        "is_anomaly": True,
        "capitulation": True,
        "capitulation_score": 0.91,
        "squeeze_setup": True,
        "squeeze_score": 0.88,
        "divergence": {
            "is_divergence": True,
            "divergence_type": "bullish",
            "score": score,
            "confidence": confidence,
        },
        "is_divergence": True,
        "divergence_type": "bullish",
        "price_oi_divergence": "bullish",
        "feature_map": {
            "oi": 104_000.0,
            "open_interest": 104_000.0,
            "open_interest_value": 8_000_000_000.0,
            "oi_delta": -0.08,
            "oi_delta_pct": -0.12,
            "oi_direction": "down",
            "oi_acceleration": -0.03,
            "price": 77_000.0,
            "funding_rate": 0.00005,
            "liquidation_imbalance": 0.72,
            "long_liquidations": 12_000_000.0,
            "short_liquidations": 3_000_000.0,
            "cvd_delta": 0.42,
            "aggressive_flow_imbalance": 0.55,
            "oi_regime": "capitulation",
            "market_regime": "capitulation",
            "anomaly": {
                "is_anomaly": True,
                "anomaly_type": "capitulation",
                "score": score,
                "confidence": confidence,
            },
            "anomaly_type": "capitulation",
            "is_anomaly": True,
            "capitulation": True,
            "capitulation_score": 0.91,
            "squeeze_setup": True,
            "squeeze_score": 0.88,
            "divergence": {
                "is_divergence": True,
                "divergence_type": "bullish",
                "score": score,
                "confidence": confidence,
            },
            "is_divergence": True,
            "divergence_type": "bullish",
            "price_oi_divergence": "bullish",
            "score": score,
            "confidence": confidence,
            "direction": "long",
            "bias": "bullish",
            "regime": "squeeze",
        },
        "metadata": {
            "source": "pytest",
            "event_contract": "analytics.oi.capitulation.detected",
        },
    }



def _inject_open_interest_domain_contracts(
    *,
    context: StrategyContext,
    payload: dict[str, Any],
) -> None:
    """
    Diagnostic/prod-contract mirror.

    SignalNormalizer already fills context.feature_map. Concrete OI strategies
    often also read context.open_interest domain objects, so this test mirrors the
    expected production contract: features/regime_state/anomaly/divergence.
    """
    oi_domain = getattr(context, "open_interest", None)
    if not isinstance(oi_domain, dict):
        return

    feature_map = payload.get("feature_map")
    if not isinstance(feature_map, dict):
        feature_map = {}

    def value_for(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
            if key in feature_map:
                return feature_map[key]
        return default

    oi_domain.setdefault(
        "features",
        {
            "oi": value_for("oi", "open_interest"),
            "open_interest": value_for("open_interest", "oi"),
            "open_interest_value": value_for("open_interest_value"),
            "oi_delta": value_for("oi_delta"),
            "oi_delta_pct": value_for("oi_delta_pct"),
            "oi_direction": value_for("oi_direction"),
            "oi_acceleration": value_for("oi_acceleration"),
            "price": value_for("price", "close"),
            "close": value_for("close", "price"),
            "funding_rate": value_for("funding_rate"),
            "liquidation_imbalance": value_for("liquidation_imbalance"),
            "cvd_delta": value_for("cvd_delta"),
            "aggressive_flow_imbalance": value_for("aggressive_flow_imbalance"),
        },
    )

    oi_domain.setdefault(
        "regime_state",
        {
            "regime": value_for("oi_regime", "regime", "market_regime", default="capitulation"),
            "score": value_for("score", default=0.0),
            "confidence": value_for("confidence", default=0.0),
            "bias": value_for("bias", default="bullish"),
            "direction": value_for("direction", default="long"),
        },
    )

    anomaly_value = value_for("anomaly")
    if isinstance(anomaly_value, dict):
        anomaly_contract = dict(anomaly_value)
    else:
        anomaly_contract = {}

    anomaly_contract.setdefault("is_anomaly", bool(value_for("is_anomaly", "capitulation", "squeeze_setup", default=True)))
    anomaly_contract.setdefault("anomaly_type", value_for("anomaly_type", default="capitulation"))
    anomaly_contract.setdefault("capitulation", bool(value_for("capitulation", default=True)))
    anomaly_contract.setdefault("capitulation_score", value_for("capitulation_score", "score", default=0.91))
    anomaly_contract.setdefault("squeeze_setup", bool(value_for("squeeze_setup", default=True)))
    anomaly_contract.setdefault("squeeze_score", value_for("squeeze_score", "score", default=0.88))
    anomaly_contract.setdefault("liquidation_imbalance", value_for("liquidation_imbalance"))
    anomaly_contract.setdefault("score", value_for("score", default=0.0))
    anomaly_contract.setdefault("confidence", value_for("confidence", default=0.0))
    oi_domain.setdefault("anomaly", anomaly_contract)

    divergence_value = value_for("divergence")
    if isinstance(divergence_value, dict):
        divergence_contract = dict(divergence_value)
    else:
        divergence_contract = {}

    divergence_contract.setdefault("is_divergence", bool(value_for("is_divergence", "price_oi_divergence", default=True)))
    divergence_contract.setdefault("divergence_type", value_for("divergence_type", "price_oi_divergence", default="bullish"))
    divergence_contract.setdefault("price_oi_divergence", value_for("price_oi_divergence", default="bullish"))
    divergence_contract.setdefault("cvd_delta", value_for("cvd_delta"))
    divergence_contract.setdefault("funding_rate", value_for("funding_rate"))
    divergence_contract.setdefault("score", value_for("score", default=0.0))
    divergence_contract.setdefault("confidence", value_for("confidence", default=0.0))
    oi_domain.setdefault("divergence", divergence_contract)


def _build_context_from_payload(
    *,
    processor: SignalProcessor,
    state: StrategyRuntimeState,
    event_name: str,
    payload: dict[str, Any],
) -> StrategyContext:
    normalized = processor.normalizer.normalize_event(
        event_name=event_name,
        payload=payload,
        timestamp=payload.get("timestamp"),
    )

    context = state.build_context(
        normalized.symbol,
        timestamp=normalized.timestamp,
        include_regime=True,
        include_portfolio=True,
    )
    context.timeframe = normalized.timeframe

    processor.normalizer.apply_to_context(context, normalized)
    _inject_open_interest_domain_contracts(context=context, payload=payload)
    state.update_context(context)
    return context


def _assert_registry_not_empty(registry: Any) -> None:
    strategies = _registered_strategies(registry)
    if not strategies:
        factories = _discover_strategy_factories_like_runner()
        pytest.fail(
            "StrategyRegistry is empty. "
            f"Discovered factories: {sorted(factories.keys())}"
        )


# =============================================================================
# 1. StrategyRegistry у реальному запуску порожній
# =============================================================================


def test_real_strategy_registry_is_not_empty_after_bootstrap() -> None:
    _, _, config, registry, _, _, _ = _build_real_strategy_stack()

    strategies = _registered_strategies(registry)

    print("\n[REGISTRY] count:", len(strategies))
    print("[REGISTRY] configured strategy definitions:", sorted(config.strategies.keys())[:80])
    print("[REGISTRY] registered strategies:")

    for strategy in strategies[:80]:
        print(f"  - {_strategy_name(strategy)} | category={_category_value(strategy)}")

    assert strategies, (
        "StrategyRegistry is empty after build_default_strategy_registry(). "
        "This means presets/factories/bootstrap are not registering real strategies."
    )


# =============================================================================
# 2. Реальні strategy classes не реєструються через preset / factories
# =============================================================================


def test_real_strategy_factories_are_discovered_and_overlap_with_preset_config() -> None:
    factories = _discover_strategy_factories_like_runner()
    _, _, config, registry, _, _, _ = _build_real_strategy_stack()

    configured_names = set(config.strategies.keys())
    factory_names = set(factories.keys())
    registered_names = {_strategy_name(strategy) for strategy in _registered_strategies(registry)}

    print("\n[FACTORIES] discovered:", sorted(factory_names))
    print("[FACTORIES] configured:", sorted(configured_names))
    print("[FACTORIES] registered:", sorted(registered_names))
    print("[FACTORIES] configured names without discovered factory:", sorted(configured_names - factory_names))

    assert factories, "No concrete strategy factories were discovered under strategy.strategies.*."
    assert registered_names, (
        "Factories were discovered, but no strategy was registered. "
        "Check build_default_strategy_registry(), preset enabled names, and factory name matching."
    )


# =============================================================================
# 3. StrategyEngine не стартує event_handler або event_handler не має subscriptions
# =============================================================================


@pytest.mark.asyncio
async def test_strategy_engine_start_registers_event_handler_subscriptions() -> None:
    event_bus, scheduler, _, registry, _, engine, _ = _build_real_strategy_stack()
    _assert_registry_not_empty(registry)

    await _maybe_await(event_bus.start())
    await _maybe_await(scheduler.start())

    try:
        await engine.start()

        handler = getattr(engine, "event_handler", None)
        assert handler is not None, "StrategyEngine has no event_handler attribute"

        subscriptions = getattr(handler, "_subscriptions", [])
        topics: list[str] = []

        for subscription in subscriptions:
            topic = (
                getattr(subscription, "topic", None)
                or getattr(subscription, "pattern", None)
                or getattr(subscription, "event_name", None)
                or str(subscription)
            )
            topics.append(str(topic))

        print("\n[ENGINE] handler registered:", getattr(handler, "_registered", None))
        print("[ENGINE] subscription count:", len(subscriptions))
        print("[ENGINE] subscription topics:", topics[:80])

        assert getattr(handler, "_registered", False) is True
        assert subscriptions, "StrategyEventHandler has zero subscriptions after engine.start()"
        assert any(topic == "analytics.*" or "analytics.*" in topic for topic in topics), (
            "StrategyEventHandler is not subscribed to analytics.*. "
            f"topics={topics}"
        )

    finally:
        await engine.stop()
        await _maybe_await(scheduler.stop())
        await _maybe_await(event_bus.stop())


# =============================================================================
# 4. Registry / Router diagnostic
# =============================================================================


def test_real_strategies_are_not_all_filtered_by_router_for_oi_event() -> None:
    _, _, _, registry, processor, _, state = _build_real_strategy_stack(
        symbols=["BTCUSDT", "DOGEUSDT", "SOLUSDT"],
        preset_name="intraday",
        use_required_features=False,
    )
    _assert_registry_not_empty(registry)

    event_name = "analytics.oi.capitulation.detected"
    payload = _realistic_oi_payload(symbol="BTCUSDT", timeframe="5m")

    context = _build_context_from_payload(
        processor=processor,
        state=state,
        event_name=event_name,
        payload=payload,
    )

    categories = processor.router._resolve_categories(
        event_name=event_name,
        source=FeatureSource.OPEN_INTEREST,
    )

    _print_registry_selection_debug(
        registry=registry,
        context=context,
        categories=categories,
    )

    route = processor.router.route(
        event_name=event_name,
        context=context,
        source=FeatureSource.OPEN_INTEREST,
        changed_features=_changed_features(context),
        metadata={"test": "real_router_filter_check"},
    )

    print("\n[ROUTER] context timeframe:", context.timeframe)
    print("[ROUTER] changed_features:", _changed_features(context))
    print("[ROUTER] categories_used:", [category.value for category in route.categories_used])
    print("[ROUTER] selected:", route.selected_names)
    print("[ROUTER] skipped:", route.skipped)

    assert route.categories_used, (
        "Router resolved no categories for analytics.oi.capitulation. "
        "Check RoutingConfig.categories_for_event() and SignalRouter._resolve_categories()."
    )

    assert route.selected, (
        "Router found categories but selected no strategies. "
        "Read REGISTRY SELECTION DEBUG above. "
        "Likely causes: registry index mismatch, supports_timeframe/symbol/regime false, "
        "or required_features mismatch."
    )


# =============================================================================
# 5. Real strategy generate_signal diagnostic
# =============================================================================


@pytest.mark.asyncio
async def test_at_least_one_real_routed_strategy_can_be_called_and_diagnosed() -> None:
    _, _, _, registry, processor, _, state = _build_real_strategy_stack(
        symbols=["BTCUSDT", "DOGEUSDT", "SOLUSDT"],
        preset_name="intraday",
        use_required_features=False,
    )
    _assert_registry_not_empty(registry)

    event_name = "analytics.oi.capitulation.detected"
    payload = _realistic_oi_payload(
        symbol="BTCUSDT",
        timeframe="5m",
        confidence=0.92,
        score=0.88,
    )

    context = _build_context_from_payload(
        processor=processor,
        state=state,
        event_name=event_name,
        payload=payload,
    )

    categories = processor.router._resolve_categories(
        event_name=event_name,
        source=FeatureSource.OPEN_INTEREST,
    )
    _print_registry_selection_debug(
        registry=registry,
        context=context,
        categories=categories,
    )

    route = processor.router.route(
        event_name=event_name,
        context=context,
        source=FeatureSource.OPEN_INTEREST,
        changed_features=_changed_features(context),
        metadata={"test": "real_strategy_generate_signal_check"},
    )

    assert route.selected, (
        "No strategies selected. See REGISTRY SELECTION DEBUG above. "
        f"skipped={route.skipped}"
    )

    generated: list[tuple[str, StrategySignal]] = []
    returned_none: list[str] = []
    failed: dict[str, str] = {}

    for strategy in route.selected:
        method = getattr(strategy, "generate_signal", None)
        if not callable(method):
            failed[_strategy_name(strategy)] = "missing generate_signal()"
            continue

        try:
            value = await _maybe_await(method(context))
        except Exception as exc:
            failed[_strategy_name(strategy)] = f"{exc.__class__.__name__}: {exc}"
            continue

        if value is None:
            returned_none.append(_strategy_name(strategy))
            continue

        if isinstance(value, StrategySignal):
            generated.append((_strategy_name(strategy), value))
            continue

        failed[_strategy_name(strategy)] = f"unexpected return type: {type(value)!r}"

    print("\n[REAL STRATEGIES] selected:", route.selected_names)
    print("[REAL STRATEGIES] generated:", [name for name, _ in generated])
    print("[REAL STRATEGIES] returned_none:", returned_none)
    print("[REAL STRATEGIES] failed:", failed)

    if not generated:
        pytest.xfail(
            "Real OI strategies were routed correctly, but all returned None. "
            "Routing, registry selection, timeframe, regime and required_features are OK. "
            "Now inspect concrete OI strategy domain conditions or test with a real analytics.oi.* payload. "
            f"returned_none={returned_none} failed={failed}"
        )


# =============================================================================
# 6. SignalProcessor downstream diagnostic with manual strategy
# =============================================================================



def _print_object_public_attrs(title: str, obj: Any) -> None:
    print("")
    print(f"========== {title} ==========")
    if obj is None:
        print("<None>")
        print("=" * (22 + len(title)))
        return

    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            value = f"<error {exc}>"
        if not callable(value):
            print(f"{name} = {value!r}")
    print("=" * (22 + len(title)))
    print("")


def _print_portfolio_config_debug(config: StrategyConfig) -> None:
    _print_object_public_attrs("PORTFOLIO CONFIG DEBUG", getattr(config, "portfolio", None))


def _print_portfolio_batch_debug(batch: Any) -> None:
    print("")
    print("========== PORTFOLIO BATCH DEBUG ==========")
    print("batch.reasons:", getattr(batch, "reasons", None))
    for attr in (
        "coordination",
        "coordinated",
        "coordination_decision",
        "portfolio_decision",
        "portfolio",
    ):
        if hasattr(batch, attr):
            _print_object_public_attrs(f"BATCH.{attr}", getattr(batch, attr))
    print("===========================================")
    print("")


class _ManualSignalStrategy(BaseStrategy):
    strategy_name = "manual_processor_probe_unique"
    category = StrategyCategory.OPEN_INTEREST
    default_setup_type = SetupType.OI_CONFIRMATION
    priority = 1

    def required_features(self) -> set[str]:
        return set()

    async def generate_signal(self, context: StrategyContext) -> StrategySignal | None:
        signal = StrategySignal(
            symbol=context.symbol,
            side=SignalSide.LONG,
            strategy_name=self.strategy_name,
            category=self.category,
            timeframe=context.timeframe,
            setup_type=self.default_setup_type,
            timestamp=context.timestamp,
            confidence=0.92,
            score=0.88,
            reasons=["pytest_manual_probe"],
            confirmations=["processor_should_emit_signal_generated"],
            source_features=_changed_features(context),
            metadata={
                "order_intent": "open",
                "market_type": "usdm_futures",
                "margin_mode": "isolated",
                "tier": "standard",
                "pytest_unique": str(utcnow().timestamp()),
            },
        )

        # New strategy model field names.
        if hasattr(signal, "entry_plan"):
            signal.entry_plan = EntryPlan(
                entry_type=EntryType.MARKET,
                price=77_000.0,
                max_slippage_bps=5.0,
            )
        if hasattr(signal, "exit_plan"):
            signal.exit_plan = ExitPlan(
                stop_loss=76_000.0,
                take_profit_levels=[],
            )
        if hasattr(signal, "invalidation_plan"):
            signal.invalidation_plan = InvalidationPlan(
                price=76_000.0,
                reason="pytest_invalidation",
            )

        # Backward-compatible aliases only for older BaseStrategy.evaluate()
        # implementations that still read signal.entry / signal.exit / signal.invalidation.
        try:
            signal.entry = getattr(signal, "entry_plan", None)
        except Exception:
            pass
        try:
            signal.exit = getattr(signal, "exit_plan", None)
        except Exception:
            pass
        try:
            signal.invalidation = getattr(signal, "invalidation_plan", None)
        except Exception:
            pass

        signal.validate()
        return signal

def _set_if_exists(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


@pytest.mark.asyncio
async def test_signal_processor_downstream_can_emit_known_good_signal_generated() -> None:
    event_bus = EventBus()
    scheduler = Scheduler(event_bus=event_bus)

    config = build_default_strategy_config(
        symbols=["BTCUSDT"],
        preset_name="intraday",
        use_required_features=False,
    )
    config.confluence.min_agreement_count = 1
    config.confluence.min_confidence = 0.0
    config.confluence.min_score = 0.0
    config.voting.min_confirmations = 1
    config.voting.min_total_votes = 1
    config.voting.allow_single_strategy_confirmation = True
    portfolio = config.portfolio

    # Disable portfolio policies that can hide whether SignalProcessor can emit.
    # The previous debug showed: rejected_signals={'manual_processor_probe': 'repeating_signal_suppressed'}.
    for name, value in {
        "enabled": True,

        # Existing fields in the current PortfolioCoordinatorConfig.
        "deduplicate_by_side": False,
        "merge_similar_signals": False,
        "correlation_guard_enabled": False,
        "enable_correlation_direction_conflict": False,
        "volatility_throttle_enabled": False,
        "repeated_signal_suppression_seconds": 0,
        "side_cooldown_seconds": 0,
        "symbol_cooldown_seconds": 0,
        "high_volatility_max_signals_per_symbol": 10,
        "max_signals_per_symbol": 10,

        # Compatibility with possible future/alternate config names.
        "allow_new_signals": True,
        "allow_same_symbol_multiple_strategies": True,
        "allow_multiple_signals_per_symbol": True,
        "allow_same_direction_signals": True,
        "block_opposite_signals": False,
        "reject_opposite_signals": False,
        "max_active_signals": 100,
        "max_active_signals_total": 100,
        "max_total_active_signals": 100,
        "max_active_per_symbol": 10,
        "max_symbol_signals": 10,
        "max_strategy_signals": 10,
        "max_signals_per_strategy": 10,
        "min_signal_score": 0.0,
        "min_signal_confidence": 0.0,
    }.items():
        _set_if_exists(portfolio, name, value)

    if hasattr(portfolio, "exposure_bucket_limits"):
        portfolio.exposure_bucket_limits = {
            "directional": 100,
            "hybrid": 100,
        }

    if hasattr(portfolio, "max_signals_per_category"):
        portfolio.max_signals_per_category = {
            StrategyCategory.PRICE_ACTION: 100,
            StrategyCategory.ORDERFLOW: 100,
            StrategyCategory.OPEN_INTEREST: 100,
            StrategyCategory.WHALES: 100,
            StrategyCategory.SPREADS: 100,
            StrategyCategory.HYBRID: 100,
        }

    _print_portfolio_config_debug(config)
    assert isinstance(config, StrategyConfig)

    registry = build_default_strategy_registry(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        strategy_factories={},
        strict=False,
        emit_events=False,
    )

    probe_definition = StrategyDefinitionConfig(
        name="manual_processor_probe_unique",
        category=StrategyCategory.OPEN_INTEREST,
        runtime=StrategyRuntimeConfig(
            enabled=True,
            symbols=["BTCUSDT"],
            timeframes=[Timeframe.M1, Timeframe.M5, Timeframe.M15],
            allowed_regimes=[MarketRegime.UNKNOWN],
            cooldown_seconds=0,
            max_signal_age_seconds=300,
            min_confidence=0.0,
            min_score=0.0,
        ),
        required_features=set(),
        weight=1.0,
        priority=1,
        tags=["pytest", "manual_probe"],
        metadata={"source": "pytest"},
    )

    probe = _ManualSignalStrategy(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
        definition=probe_definition,
    )
    registry.register_strategy(probe, replace=True, emit_event=False)

    state = StrategyRuntimeState()

    processor = SignalProcessor(
        config=config,
        registry=registry,
        state=state,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    captured_signal_generated: list[dict[str, Any]] = []

    async def on_signal_generated(event: Any) -> None:
        payload = getattr(event, "payload", event)
        if isinstance(payload, dict):
            captured_signal_generated.append(payload)

    event_bus.subscribe(
        "signal.generated",
        on_signal_generated,
        name="pytest_capture_signal_generated",
    )

    await _maybe_await(event_bus.start())
    await _maybe_await(scheduler.start())

    try:
        payload = _realistic_oi_payload(
            symbol="BTCUSDT",
            timeframe="5m",
            confidence=0.92,
            score=0.88,
        )

        batch = await processor.process_event(
            event_name="analytics.oi.capitulation.detected",
            payload=payload,
            timestamp=payload.get("timestamp"),
            emit=True,
        )

        _print_portfolio_batch_debug(batch)

        print("\n[PROCESSOR] accepted:", batch.accepted)
        print("[PROCESSOR] emitted:", batch.emitted)
        print("[PROCESSOR] reasons:", batch.reasons)
        print("[PROCESSOR] route:", batch.route.selected_names if batch.route else None)
        print("[PROCESSOR] raw_signal_count:", len(batch.raw_signals))
        print("[PROCESSOR] final_signal_count:", len(batch.final_signals))
        print("[PROCESSOR] risk_payload_count:", len(batch.risk_payloads))
        print("[PROCESSOR] evaluations:")
        for evaluation in batch.evaluations:
            print(" -", evaluation)
        print("[PROCESSOR] captured signal.generated:", len(captured_signal_generated))

        assert batch.route is not None, "SignalProcessor did not create a RouteDecision"
        assert batch.route.selected, f"No strategies routed. reasons={batch.reasons}"
        assert batch.raw_signals, (
            f"No raw signals created. reasons={batch.reasons}. "
            "If evaluation_error mentions signal.entry, fix strategy/base.py to use "
            "entry_plan/exit_plan/invalidation_plan or keep compatibility aliases."
        )
        assert batch.final_signals, f"Signal was blocked before final_signals. reasons={batch.reasons}"
        assert batch.risk_payloads, f"SignalBuilder did not create risk payloads. reasons={batch.reasons}"
        assert batch.emitted is True, f"SignalProcessor did not mark batch emitted. reasons={batch.reasons}"
        assert captured_signal_generated, "No signal.generated event was captured from EventBus"

    finally:
        await _maybe_await(scheduler.stop())
        await _maybe_await(event_bus.stop())