# test/liquidityenginetest/test_base_liquidity_strategy.py

from __future__ import annotations

import copy
import logging
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
    SweepStatus,
)
from analytics.liquidity.models import (
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
)
from strategy.enums import FilterDecision, SignalPriority, SignalSide, SignalStatus
from strategy.strategies.liquidity.base_liquidity_strategy import BaseLiquidityStrategy


DEFAULT_EXCHANGE = "binance"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_PRICE = 100.0

FUTURES_MARKET_TYPES = (
    "perpetual",
    "futures",
    "linear",
    "inverse",
    "swap",
    "usdm_futures",
    "coinm_futures",
)

NON_FUTURES_MARKET_TYPES = (
    "spot",
    "margin",
    "cash",
)


# ---------------------------------------------------------------------------
# Safe enum helpers
# ---------------------------------------------------------------------------

def enum_member(enum_cls: type[Enum], *names: str, default: Any | None = None) -> Any:
    for name in names:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)

    if default is not None:
        return default

    return next(iter(enum_cls))


def enum_from_value(enum_cls: type[Enum], value: Any, *fallback_names: str) -> Any:
    if isinstance(value, enum_cls):
        return value

    if value is None:
        return enum_member(enum_cls, *fallback_names)

    try:
        return enum_cls(value)
    except Exception:
        if isinstance(value, str):
            normalized = value.upper()
            if hasattr(enum_cls, normalized):
                return getattr(enum_cls, normalized)

        return enum_member(enum_cls, *fallback_names)


def buy_side() -> LiquiditySide:
    return enum_member(LiquiditySide, "BUY_SIDE", "BUY", "BID")


def sell_side() -> LiquiditySide:
    return enum_member(LiquiditySide, "SELL_SIDE", "SELL", "ASK")


def bias_up() -> LiquidityBias:
    return enum_member(
        LiquidityBias,
        "UP",
        "BULLISH",
        default=enum_member(LiquidityBias, "NEUTRAL", "NONE"),
    )


def bias_neutral() -> LiquidityBias:
    return enum_member(
        LiquidityBias,
        "NEUTRAL",
        "NONE",
        "BALANCED",
        default=bias_up(),
    )


def sweep_active() -> SweepStatus:
    return enum_member(SweepStatus, "ACTIVE", "UNSWEPT", "NONE")


def sweep_swept() -> SweepStatus:
    return enum_member(SweepStatus, "SWEPT", default=sweep_active())


def swing_high_type() -> LiquidityLevelType:
    return enum_member(
        LiquidityLevelType,
        "SWING_HIGH",
        "RESISTANCE",
        "EQUAL_HIGHS",
    )


def swing_low_type() -> LiquidityLevelType:
    return enum_member(
        LiquidityLevelType,
        "SWING_LOW",
        "SUPPORT",
        "EQUAL_LOWS",
    )


def signal_long() -> SignalSide:
    return enum_member(SignalSide, "LONG", "BUY")


def signal_new_status() -> SignalStatus:
    return enum_member(SignalStatus, "NEW", "GENERATED", "PENDING")


def signal_normal_priority() -> SignalPriority:
    return enum_member(
        SignalPriority,
        "NORMAL",
        "MEDIUM",
        "DEFAULT",
        "LOW",
    )


# ---------------------------------------------------------------------------
# Generic dataclass builders
# ---------------------------------------------------------------------------

def _annotation_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation)).lower()


def _fallback_value_for_field(name: str, annotation: Any) -> Any:
    lower = name.lower()
    annotation_name = _annotation_name(annotation)

    if lower == "exchange":
        return DEFAULT_EXCHANGE
    if lower == "market_type":
        return DEFAULT_MARKET_TYPE
    if lower == "symbol":
        return DEFAULT_SYMBOL
    if lower == "timeframe":
        return DEFAULT_TIMEFRAME

    if "timestamp" in lower or lower.endswith("_at") or lower in {"created", "updated"}:
        return datetime.now(timezone.utc)

    if lower in {"side", "liquidity_side"}:
        return buy_side()
    if lower in {"bias", "directional_bias"}:
        return bias_neutral()
    if lower in {"sweep_status", "status"}:
        return sweep_active()
    if lower in {"level_type", "type"}:
        return swing_high_type()

    if "price" in lower:
        if "low" in lower:
            return DEFAULT_PRICE * 0.99
        if "high" in lower:
            return DEFAULT_PRICE * 1.01
        if "center" in lower:
            return DEFAULT_PRICE
        return DEFAULT_PRICE

    if "score" in lower or "confidence" in lower or "strength" in lower:
        return 0.8

    if "density" in lower:
        return 0.8

    if "count" in lower or "touches" in lower or "reaction" in lower:
        return 3

    if lower.startswith("is_") or lower.startswith("has_") or annotation is bool:
        return False

    if "list" in annotation_name or "sequence" in annotation_name:
        return []

    if "dict" in annotation_name or "mapping" in annotation_name or lower == "metadata":
        return {}

    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False

    return None


def build_dataclass(model_cls: type[Any], **overrides: Any) -> Any:
    if not is_dataclass(model_cls):
        return model_cls(**overrides)

    accepted = {item.name: item for item in fields(model_cls)}
    kwargs: dict[str, Any] = {}

    for name, dataclass_field in accepted.items():
        if name in overrides:
            kwargs[name] = overrides[name]
            continue

        if dataclass_field.default is not MISSING:
            continue

        if dataclass_field.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue

        kwargs[name] = _fallback_value_for_field(name, dataclass_field.type)

    for key, value in overrides.items():
        if key in accepted:
            kwargs[key] = value

    return model_cls(**kwargs)


def clone_with(obj: Any, **changes: Any) -> Any:
    cloned = copy.deepcopy(obj)

    for key, value in changes.items():
        setattr(cloned, key, value)

    return cloned


def scope_dict(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> dict[str, str]:
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def scope_key(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> str:
    return f"{exchange}:{market_type}:{symbol}:{timeframe}"


# ---------------------------------------------------------------------------
# Local test doubles
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CapturedEvent:
    topic: str
    payload: Any = None
    source: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeEventBus:
    def __init__(self) -> None:
        self.events: list[CapturedEvent] = []

    async def emit(
        self,
        topic: str,
        payload: Any = None,
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            CapturedEvent(
                topic=topic,
                payload=payload,
                source=source,
                kwargs=kwargs,
            )
        )

    async def publish(
        self,
        topic: str,
        payload: Any = None,
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.events.append(
            CapturedEvent(
                topic=topic,
                payload=payload,
                source=source,
                kwargs=kwargs,
            )
        )

    def events_for(self, topic: str) -> list[CapturedEvent]:
        return [event for event in self.events if event.topic == topic]


@dataclass(slots=True)
class SnapshotWrapper:
    snapshot: Any = None
    value: Any = None
    data: Any = None
    payload: Any = None


@dataclass(slots=True)
class FakeLiquidityContext:
    snapshot: Any = None
    liquidity_map_snapshot: Any = None
    map_snapshot: Any = None
    last_snapshot: Any = None


@dataclass(slots=True)
class FakeStrategyContext:
    exchange: str = DEFAULT_EXCHANGE
    market_type: str = DEFAULT_MARKET_TYPE
    symbol: str = DEFAULT_SYMBOL
    timeframe: str = DEFAULT_TIMEFRAME
    current_price: float | None = DEFAULT_PRICE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    liquidity: Any = None
    features: dict[str, Any] = field(default_factory=dict)
    feature_snapshots: dict[str, Any] = field(default_factory=dict)

    price: Any = None
    portfolio: Any = None
    spread: Any = None
    regime: Any = None

    def get_feature(self, key: str, default: Any = None) -> Any:
        return self.features.get(key, default)

    def get_feature_snapshot(self, key: str, default: Any = None) -> Any:
        return self.feature_snapshots.get(key, default)

    @property
    def scope(self) -> dict[str, str]:
        return scope_dict(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    @property
    def scope_key(self) -> str:
        return scope_key(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )


class FreshnessConfig:
    def __init__(self, ttl_seconds: float | None = None) -> None:
        self.ttl_seconds = ttl_seconds

    def get_ttl(self, feature_name: str) -> float | None:
        return self.ttl_seconds


def make_config(
    *,
    enabled: bool = True,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    emit_cooldown_seconds: float = 0.0,
    ttl_seconds: float | None = 60.0,
    exchanges: set[str] | None = None,
    market_types: set[str] | None = None,
    symbols: set[str] | None = None,
    timeframes: set[str] | None = None,
) -> SimpleNamespace:
    runtime = SimpleNamespace(
        enabled=enabled,
        min_confidence=min_confidence,
        min_score=min_score,
        max_signal_age_seconds=60,
        emit_cooldown_seconds=emit_cooldown_seconds,
        exchanges=exchanges or set(),
        market_types=market_types or set(),
        symbols=symbols or set(),
        timeframes=timeframes or set(),
        allowed_regimes=None,
    )

    return SimpleNamespace(
        runtime=runtime,
        freshness=FreshnessConfig(ttl_seconds),
        builders=SimpleNamespace(require_invalidation=True),
        filters=SimpleNamespace(
            enable_portfolio_filter=False,
            enable_spread_filter=False,
        ),
        get_strategy=lambda _name: None,
        validate=lambda: None,
    )


class DummyLiquidityStrategy(BaseLiquidityStrategy):
    """
    Concrete test-only strategy.

    Important: __init__ intentionally avoids BaseStrategyComponent.__init__ and
    BaseLiquidityStrategy.__init__, because these tests target BaseLiquidityStrategy
    behavior without depending on the full app config/bootstrap.
    """

    def __init__(
        self,
        *,
        config: Any | None = None,
        event_bus: FakeEventBus | None = None,
    ) -> None:
        self.config = config or make_config()
        self.event_bus = event_bus or FakeEventBus()
        self.logger = logging.getLogger("test.dummy_liquidity_strategy")
        self._last_emitted_at: dict[str, datetime] = {}

    @property
    def strategy_name(self) -> str:
        return "dummy_liquidity_strategy"

    def evaluate(self, context: Any) -> None:
        self.validate_context(context)
        return None

    async def emit_event(
        self,
        topic: str,
        payload: Any = None,
        *,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        await self.event_bus.emit(
            topic,
            payload,
            source=source,
            **kwargs,
        )

    def log_debug(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_warning(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Local model builders
# ---------------------------------------------------------------------------

def make_level(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    side: LiquiditySide | str | None = None,
    level_type: LiquidityLevelType | str | None = None,
    sweep_status: SweepStatus | str | None = None,
    price: float = DEFAULT_PRICE * 1.02,
    confidence: float = 0.85,
    key: str | None = None,
    **overrides: Any,
) -> LiquidityLevel:
    side_value = enum_from_value(
        LiquiditySide,
        side,
        "BUY_SIDE",
        "BUY",
        "BID",
    )
    level_type_value = enum_from_value(
        LiquidityLevelType,
        level_type,
        "SWING_HIGH",
        "RESISTANCE",
        "EQUAL_HIGHS",
    )
    sweep_status_value = enum_from_value(
        SweepStatus,
        sweep_status,
        "ACTIVE",
        "UNSWEPT",
        "NONE",
    )

    now = datetime.now(timezone.utc)

    return build_dataclass(
        LiquidityLevel,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        scope=scope_dict(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        scope_key=scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        side=side_value,
        level_type=level_type_value,
        sweep_status=sweep_status_value,
        price=price,
        confidence=confidence,
        strength=confidence,
        touches_count=4,
        reaction_count=2,
        timestamp=now,
        created_at=now,
        updated_at=now,
        key=key or f"level:{exchange}:{market_type}:{symbol}:{timeframe}:{price}",
        metadata={},
        **overrides,
    )


def make_buy_level(
    *,
    price: float = DEFAULT_PRICE * 1.02,
    sweep_status: SweepStatus | str | None = None,
    **overrides: Any,
) -> LiquidityLevel:
    return make_level(
        side=buy_side(),
        level_type=swing_high_type(),
        sweep_status=sweep_status or sweep_active(),
        price=price,
        **overrides,
    )


def make_sell_level(
    *,
    price: float = DEFAULT_PRICE * 0.98,
    sweep_status: SweepStatus | str | None = None,
    **overrides: Any,
) -> LiquidityLevel:
    return make_level(
        side=sell_side(),
        level_type=swing_low_type(),
        sweep_status=sweep_status or sweep_active(),
        price=price,
        **overrides,
    )


def make_cluster(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    side: LiquiditySide | str | None = None,
    sweep_status: SweepStatus | str | None = None,
    center_price: float = DEFAULT_PRICE * 1.03,
    confidence: float = 0.85,
    key: str | None = None,
    **overrides: Any,
) -> StopCluster:
    side_value = enum_from_value(
        LiquiditySide,
        side,
        "BUY_SIDE",
        "BUY",
        "BID",
    )
    sweep_status_value = enum_from_value(
        SweepStatus,
        sweep_status,
        "ACTIVE",
        "UNSWEPT",
        "NONE",
    )
    now = datetime.now(timezone.utc)

    return build_dataclass(
        StopCluster,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        scope=scope_dict(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        scope_key=scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        side=side_value,
        sweep_status=sweep_status_value,
        center_price=center_price,
        low_price=center_price * 0.999,
        high_price=center_price * 1.001,
        confidence=confidence,
        score=confidence,
        estimated_stop_density=confidence,
        timestamp=now,
        created_at=now,
        updated_at=now,
        key=key or f"cluster:{exchange}:{market_type}:{symbol}:{timeframe}:{center_price}",
        metadata={},
        **overrides,
    )


def make_snapshot(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    current_price: float = DEFAULT_PRICE,
    timestamp: datetime | None = None,
    active_levels: list[Any] | None = None,
    equal_levels: list[Any] | None = None,
    stop_clusters: list[Any] | None = None,
    zones: list[Any] | None = None,
    nearest_above_level: Any = None,
    nearest_below_level: Any = None,
    strongest_cluster_above: Any = None,
    strongest_cluster_below: Any = None,
    bias: LiquidityBias | str | None = None,
    above_liquidity_score: float = 0.8,
    below_liquidity_score: float = 0.2,
    liquidity_pressure_score: float = 0.4,
    metadata: dict[str, Any] | None = None,
    signal: Any = None,
    **overrides: Any,
) -> LiquidityMapSnapshot:
    ts = timestamp or datetime.now(timezone.utc)

    if signal is not None and not hasattr(signal, "metadata"):
        signal.metadata = {}

    if signal is not None and getattr(signal, "metadata", None) is None:
        signal.metadata = {}

    return build_dataclass(
        LiquidityMapSnapshot,
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        created_at=ts,
        updated_at=ts,
        current_price=current_price,
        mark_price=current_price,
        last_price=current_price,
        bias=enum_from_value(
            LiquidityBias,
            bias,
            "UP",
            "BULLISH",
            "NEUTRAL",
            "NONE",
        ),
        active_levels=active_levels or [],
        equal_levels=equal_levels or [],
        stop_clusters=stop_clusters or [],
        zones=zones or [],
        nearest_above_level=nearest_above_level,
        nearest_below_level=nearest_below_level,
        strongest_cluster_above=strongest_cluster_above,
        strongest_cluster_below=strongest_cluster_below,
        above_liquidity_score=above_liquidity_score,
        below_liquidity_score=below_liquidity_score,
        liquidity_pressure_score=liquidity_pressure_score,
        signal=signal,
        metadata=metadata or {"confidence": 0.8},
        scope=scope_dict(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        scope_key=scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        ),
        **overrides,
    )


def make_valid_snapshot(**overrides: Any) -> LiquidityMapSnapshot:
    target = make_buy_level(price=DEFAULT_PRICE * 1.025)
    invalidation = make_sell_level(price=DEFAULT_PRICE * 0.985)

    return make_snapshot(
        active_levels=[target, invalidation],
        nearest_above_level=target,
        nearest_below_level=invalidation,
        above_liquidity_score=0.95,
        below_liquidity_score=0.15,
        liquidity_pressure_score=0.7,
        signal=SimpleNamespace(
            confidence=0.9,
            bias=bias_up(),
            magnet_score_up=0.9,
            sweep_risk_up=0.9,
        ),
        **overrides,
    )


def make_context(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    current_price: float | None = DEFAULT_PRICE,
    timestamp: datetime | None = None,
    snapshot: LiquidityMapSnapshot | None = None,
    snapshot_location: str = "liquidity.snapshot",
) -> FakeStrategyContext:
    context = FakeStrategyContext(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        timestamp=timestamp or datetime.now(timezone.utc),
    )

    if snapshot is None:
        return context

    if snapshot_location == "liquidity.snapshot":
        context.liquidity = FakeLiquidityContext(snapshot=snapshot)
    elif snapshot_location == "liquidity.liquidity_map_snapshot":
        context.liquidity = FakeLiquidityContext(liquidity_map_snapshot=snapshot)
    elif snapshot_location == "liquidity.map_snapshot":
        context.liquidity = FakeLiquidityContext(map_snapshot=snapshot)
    elif snapshot_location == "liquidity.last_snapshot":
        context.liquidity = FakeLiquidityContext(last_snapshot=snapshot)
    elif snapshot_location == "feature":
        context.features["liquidity_map_snapshot"] = snapshot
    elif snapshot_location == "feature_snapshot":
        context.feature_snapshots["liquidity_map_snapshot"] = snapshot
    elif snapshot_location == "feature_wrapped_snapshot":
        context.features["liquidity_map_snapshot"] = SnapshotWrapper(snapshot=snapshot)
    elif snapshot_location == "feature_wrapped_value":
        context.features["liquidity_map_snapshot"] = SnapshotWrapper(value=snapshot)
    elif snapshot_location == "feature_wrapped_data":
        context.features["liquidity_map_snapshot"] = SnapshotWrapper(data=snapshot)
    elif snapshot_location == "feature_wrapped_payload":
        context.features["liquidity_map_snapshot"] = SnapshotWrapper(payload=snapshot)
    elif snapshot_location == "mapping_payload":
        context.features["liquidity_map_snapshot"] = {"payload": snapshot}
    elif snapshot_location == "mapping_nested_payload":
        context.features["liquidity_map_snapshot"] = {"data": {"payload": snapshot}}
    elif snapshot_location == "missing":
        pass
    else:
        raise ValueError(f"Unsupported snapshot_location={snapshot_location!r}")

    return context


def make_signal_like(
    *,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    strategy_name: str = "dummy_liquidity_strategy",
    side: Any | None = None,
    status: Any | None = None,
    priority: Any | None = None,
    category: Any = None,
    score: float = 0.85,
    confidence: float = 0.8,
    timestamp: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        category=category,
        side=side or signal_long(),
        score=score,
        confidence=confidence,
        status=status or signal_new_status(),
        priority=priority or signal_normal_priority(),
        timestamp=timestamp or datetime.now(timezone.utc),
        metadata={},
        validate=lambda: None,
    )


def assert_filter_decisions_are_enums(results: list[Any]) -> None:
    assert results, "Expected at least one FilterResult"

    for result in results:
        assert isinstance(result.decision, FilterDecision), (
            f"{result.name} decision must be FilterDecision, "
            f"got {result.decision!r} ({type(result.decision).__name__})"
        )


# ---------------------------------------------------------------------------
# Tests: metadata and basic contract
# ---------------------------------------------------------------------------

def test_category_is_liquidity() -> None:
    strategy = DummyLiquidityStrategy()

    assert getattr(strategy.category, "name", None) == "LIQUIDITY"


def test_required_features_contains_liquidity_map_snapshot() -> None:
    strategy = DummyLiquidityStrategy()

    assert strategy.required_features() == {"liquidity_map_snapshot"}


def test_signal_topic_is_signal_generated() -> None:
    strategy = DummyLiquidityStrategy()

    assert strategy.SIGNAL_TOPIC == "signal.generated"


def test_strategy_enabled_by_default() -> None:
    strategy = DummyLiquidityStrategy(config=make_config(enabled=True))

    assert strategy.is_enabled() is True


def test_strategy_can_be_disabled() -> None:
    strategy = DummyLiquidityStrategy(config=make_config(enabled=False))

    assert strategy.is_enabled() is False


# ---------------------------------------------------------------------------
# Tests: snapshot extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "snapshot_location",
    [
        "liquidity.snapshot",
        "liquidity.liquidity_map_snapshot",
        "liquidity.map_snapshot",
        "liquidity.last_snapshot",
        "feature",
        "feature_snapshot",
        "feature_wrapped_snapshot",
        "feature_wrapped_value",
        "feature_wrapped_data",
        "feature_wrapped_payload",
        "mapping_payload",
        "mapping_nested_payload",
    ],
)
def test_extract_snapshot_from_all_supported_locations(snapshot_location: str) -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot, snapshot_location=snapshot_location)

    assert strategy._extract_snapshot(context) is snapshot


def test_extract_snapshot_returns_none_when_missing() -> None:
    strategy = DummyLiquidityStrategy()
    context = make_context(snapshot=None)

    assert strategy._extract_snapshot(context) is None


def test_extract_snapshot_ignores_malformed_wrapper() -> None:
    strategy = DummyLiquidityStrategy()
    context = make_context(snapshot=None)
    context.features["liquidity_map_snapshot"] = {"data": {"payload": object()}}

    assert strategy._extract_snapshot(context) is None


def test_extract_snapshot_prefers_domain_snapshot_over_feature_snapshot() -> None:
    strategy = DummyLiquidityStrategy()

    domain_snapshot = make_valid_snapshot(symbol=DEFAULT_SYMBOL)
    feature_snapshot = make_valid_snapshot(symbol="ETHUSDT")

    context = make_context(
        snapshot=domain_snapshot,
        snapshot_location="liquidity.snapshot",
    )
    context.features["liquidity_map_snapshot"] = feature_snapshot

    assert strategy._extract_snapshot(context) is domain_snapshot


# ---------------------------------------------------------------------------
# Tests: scope, futures-only, allow lists
# ---------------------------------------------------------------------------

def test_base_context_is_valid_for_matching_futures_scope() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is True


@pytest.mark.parametrize(
    ("field_name", "context_value"),
    [
        ("exchange", "bybit"),
        ("market_type", "linear"),
        ("symbol", "ETHUSDT"),
        ("timeframe", "5m"),
    ],
)
def test_base_context_rejects_scope_mismatch(
    field_name: str,
    context_value: str,
) -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()

    kwargs = {
        "exchange": snapshot.exchange,
        "market_type": snapshot.market_type,
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "snapshot": snapshot,
    }
    kwargs[field_name] = context_value

    context = make_context(**kwargs)

    assert strategy._base_context_is_valid(context, snapshot) is False


@pytest.mark.parametrize("market_type", FUTURES_MARKET_TYPES)
def test_accepts_supported_futures_market_types(market_type: str) -> None:
    strategy = DummyLiquidityStrategy(
        config=make_config(market_types={market_type}),
    )
    snapshot = make_valid_snapshot(market_type=market_type)
    context = make_context(market_type=market_type, snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is True


@pytest.mark.parametrize("market_type", NON_FUTURES_MARKET_TYPES)
def test_rejects_non_futures_market_types(market_type: str) -> None:
    strategy = DummyLiquidityStrategy(
        config=make_config(market_types={market_type}),
    )
    snapshot = make_valid_snapshot(market_type=market_type)
    context = make_context(market_type=market_type, snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is False


def test_rejects_disallowed_exchange() -> None:
    strategy = DummyLiquidityStrategy(
        config=make_config(exchanges={"bybit"}),
    )
    snapshot = make_valid_snapshot(exchange=DEFAULT_EXCHANGE)
    context = make_context(exchange=DEFAULT_EXCHANGE, snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is False


def test_rejects_disallowed_symbol() -> None:
    strategy = DummyLiquidityStrategy(
        config=make_config(symbols={"ETHUSDT"}),
    )
    snapshot = make_valid_snapshot(symbol=DEFAULT_SYMBOL)
    context = make_context(symbol=DEFAULT_SYMBOL, snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is False


def test_rejects_disallowed_timeframe() -> None:
    strategy = DummyLiquidityStrategy(
        config=make_config(timeframes={"5m"}),
    )
    snapshot = make_valid_snapshot(timeframe=DEFAULT_TIMEFRAME)
    context = make_context(timeframe=DEFAULT_TIMEFRAME, snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is False


# ---------------------------------------------------------------------------
# Tests: freshness / TTL
# ---------------------------------------------------------------------------

def test_accepts_fresh_snapshot() -> None:
    now = datetime.now(timezone.utc)

    strategy = DummyLiquidityStrategy(
        config=make_config(ttl_seconds=60),
    )
    snapshot = make_valid_snapshot(timestamp=now - timedelta(seconds=10))
    context = make_context(snapshot=snapshot, timestamp=now)

    assert strategy._base_context_is_valid(context, snapshot) is True


def test_rejects_stale_snapshot() -> None:
    now = datetime.now(timezone.utc)

    strategy = DummyLiquidityStrategy(
        config=make_config(ttl_seconds=30),
    )
    snapshot = make_valid_snapshot(timestamp=now - timedelta(seconds=3600))
    context = make_context(snapshot=snapshot, timestamp=now)

    assert strategy._base_context_is_valid(context, snapshot) is False


def test_rejects_snapshot_unreasonably_in_future_because_age_abs_is_used() -> None:
    now = datetime.now(timezone.utc)

    strategy = DummyLiquidityStrategy(
        config=make_config(ttl_seconds=30),
    )
    snapshot = make_valid_snapshot(timestamp=now + timedelta(hours=3))
    context = make_context(snapshot=snapshot, timestamp=now)

    assert strategy._base_context_is_valid(context, snapshot) is False


def test_disables_staleness_check_when_ttl_is_none() -> None:
    now = datetime.now(timezone.utc)

    strategy = DummyLiquidityStrategy(
        config=make_config(ttl_seconds=None),
    )
    snapshot = make_valid_snapshot(timestamp=now - timedelta(days=365))
    context = make_context(snapshot=snapshot, timestamp=now)

    assert strategy._base_context_is_valid(context, snapshot) is True


# ---------------------------------------------------------------------------
# Tests: common filters
# ---------------------------------------------------------------------------

def test_common_pre_filters_return_filter_decision_enums() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    results = strategy._run_common_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert_filter_decisions_are_enums(results)


def test_common_pre_filters_do_not_block_clean_context() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    results = strategy._run_common_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert not any(result.blocked for result in results)


def test_common_pre_filters_block_invalid_price() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    results = strategy._run_common_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=0.0,
    )

    assert any(
        result.name == "price_validation_filter" and result.blocked
        for result in results
    )


def test_common_pre_filters_block_scope_mismatch() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(symbol="ETHUSDT", snapshot=snapshot)

    results = strategy._run_common_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert any(
        result.name == "liquidity_scope_filter" and result.blocked
        for result in results
    )


def test_common_pre_filters_block_non_futures_market_type() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot(market_type="spot")
    context = make_context(market_type="spot", snapshot=snapshot)

    results = strategy._run_common_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert any(
        result.name == "futures_market_filter" and result.blocked
        for result in results
    )


# ---------------------------------------------------------------------------
# Tests: current price resolving
# ---------------------------------------------------------------------------

def test_resolve_current_price_from_snapshot() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot(current_price=123.45)
    context = make_context(snapshot=snapshot)

    assert strategy._resolve_current_price(context, snapshot) == pytest.approx(123.45)


def test_resolve_current_price_prefers_context_mid_price() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot(current_price=100.0)
    context = make_context(snapshot=snapshot)
    context.price = SimpleNamespace(mid_price=101.25)

    assert strategy._resolve_current_price(context, snapshot) == pytest.approx(101.25)


@pytest.mark.parametrize(
    "bad_price",
    [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_resolve_current_price_returns_none_for_invalid_prices(
    bad_price: float | None,
) -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot(current_price=bad_price)  # type: ignore[arg-type]
    context = make_context(snapshot=snapshot)
    context.price = SimpleNamespace(
        mid_price=bad_price,
        last_price=bad_price,
        mark_price=bad_price,
        index_price=bad_price,
    )

    assert strategy._resolve_current_price(context, snapshot) is None


# ---------------------------------------------------------------------------
# Tests: target helpers
# ---------------------------------------------------------------------------

def test_collect_targets_above_excludes_swept_levels_by_default() -> None:
    strategy = DummyLiquidityStrategy()

    swept_target = make_buy_level(
        price=DEFAULT_PRICE * 1.025,
        sweep_status=sweep_swept(),
    )
    active_target = make_buy_level(
        price=DEFAULT_PRICE * 1.03,
        sweep_status=sweep_active(),
    )
    snapshot = make_valid_snapshot(
        active_levels=[swept_target, active_target],
        nearest_above_level=swept_target,
    )

    targets = strategy._collect_targets_above(snapshot, DEFAULT_PRICE)

    assert active_target in targets
    assert swept_target not in targets


def test_collect_targets_below_excludes_swept_levels_by_default() -> None:
    strategy = DummyLiquidityStrategy()

    swept_target = make_sell_level(
        price=DEFAULT_PRICE * 0.975,
        sweep_status=sweep_swept(),
    )
    active_target = make_sell_level(
        price=DEFAULT_PRICE * 0.97,
        sweep_status=sweep_active(),
    )
    snapshot = make_valid_snapshot(
        active_levels=[swept_target, active_target],
        nearest_below_level=swept_target,
    )

    targets = strategy._collect_targets_below(snapshot, DEFAULT_PRICE)

    assert active_target in targets
    assert swept_target not in targets


def test_collect_targets_above_can_include_swept_when_explicitly_requested() -> None:
    strategy = DummyLiquidityStrategy()

    swept_target = make_buy_level(
        price=DEFAULT_PRICE * 1.025,
        sweep_status=sweep_swept(),
    )
    snapshot = make_valid_snapshot(
        active_levels=[swept_target],
        nearest_above_level=swept_target,
    )

    targets = strategy._collect_targets_above(
        snapshot,
        DEFAULT_PRICE,
        include_swept=True,
    )

    assert swept_target in targets


def test_dedupe_liquidity_items_removes_duplicate_keys() -> None:
    strategy = DummyLiquidityStrategy()

    first = make_buy_level(price=101.0, key="duplicate-key")
    second = make_buy_level(price=102.0, key="duplicate-key")

    result = strategy._dedupe_liquidity_items([first, second])

    assert result == [first]


def test_reference_price_prefers_price_attribute() -> None:
    strategy = DummyLiquidityStrategy()
    item = SimpleNamespace(price=101.0, center_price=102.0)

    assert strategy._reference_price(item) == pytest.approx(101.0)


def test_reference_price_falls_back_to_center_price() -> None:
    strategy = DummyLiquidityStrategy()
    item = SimpleNamespace(center_price=102.0)

    assert strategy._reference_price(item) == pytest.approx(102.0)


def test_reference_price_falls_back_to_low_high_midpoint() -> None:
    strategy = DummyLiquidityStrategy()
    item = SimpleNamespace(low_price=90.0, high_price=110.0)

    assert strategy._reference_price(item) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Tests: metadata and payload helpers
# ---------------------------------------------------------------------------

def test_build_liquidity_signal_metadata_contains_scope_counts_target_and_evidence() -> None:
    strategy = DummyLiquidityStrategy()

    target = make_buy_level(price=DEFAULT_PRICE * 1.025)
    evidence = make_sell_level(price=DEFAULT_PRICE * 0.985)

    snapshot = make_valid_snapshot(
        active_levels=[target, evidence],
        nearest_above_level=target,
        nearest_below_level=evidence,
    )

    metadata = strategy._build_liquidity_signal_metadata(
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
        target=target,
        evidence=evidence,
        setup_name="dummy_setup",
        extra={"custom_key": "custom_value"},
    )

    assert metadata["setup_name"] == "dummy_setup"
    assert metadata["exchange"] == DEFAULT_EXCHANGE
    assert metadata["market_type"] == DEFAULT_MARKET_TYPE
    assert metadata["symbol"] == DEFAULT_SYMBOL
    assert metadata["timeframe"] == DEFAULT_TIMEFRAME
    assert metadata["scope_key"] == snapshot.scope_key
    assert metadata["active_levels_count"] == 2
    assert metadata["target"] is not None
    assert metadata["evidence"] is not None
    assert metadata["custom_key"] == "custom_value"


def test_to_payload_handles_none_dataclass_enum_and_plain_object() -> None:
    strategy = DummyLiquidityStrategy()

    level = make_buy_level()
    plain = SimpleNamespace(a=1, b="x")

    assert strategy._to_payload(None) is None
    assert isinstance(strategy._to_payload(level), dict)

    plain_payload = strategy._to_payload(plain)
    assert plain_payload["a"] == 1
    assert plain_payload["b"] == "x"


# ---------------------------------------------------------------------------
# Tests: signal/context validation
# ---------------------------------------------------------------------------

def test_validate_signal_context_pair_accepts_matching_signal_and_context() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)
    signal = make_signal_like(category=strategy.category)

    strategy._validate_signal_context_pair(signal, context)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("symbol", "ETHUSDT"),
        ("timeframe", "5m"),
        ("strategy_name", "wrong_strategy"),
    ],
)
def test_validate_signal_context_pair_rejects_signal_mismatch(
    field_name: str,
    bad_value: Any,
) -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    signal_kwargs = {"category": strategy.category}
    signal_kwargs[field_name] = bad_value
    signal = make_signal_like(**signal_kwargs)

    with pytest.raises(ValueError):
        strategy._validate_signal_context_pair(signal, context)


def test_validate_signal_context_pair_rejects_snapshot_context_scope_mismatch() -> None:
    strategy = DummyLiquidityStrategy()

    snapshot = make_valid_snapshot(symbol="ETHUSDT")
    context = make_context(symbol=DEFAULT_SYMBOL, snapshot=snapshot)
    signal = make_signal_like(category=strategy.category)

    with pytest.raises(ValueError):
        strategy._validate_signal_context_pair(signal, context)


# ---------------------------------------------------------------------------
# Tests: emit_signal contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_signal_publishes_signal_generated_payload() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(event_bus=event_bus)

    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)
    signal = make_signal_like(category=strategy.category)

    result = await strategy.emit_signal(signal, context)

    assert result is signal

    events = event_bus.events_for("signal.generated")
    assert len(events) == 1

    event = events[0]
    payload = event.payload

    assert event.topic == "signal.generated"
    assert event.source == "dummy_liquidity_strategy"

    assert payload["symbol"] == DEFAULT_SYMBOL
    assert payload["strategy_name"] == "dummy_liquidity_strategy"
    assert payload["timeframe"] == DEFAULT_TIMEFRAME
    assert payload["side"] == strategy._value(signal.side)
    assert payload["score"] == pytest.approx(0.85)
    assert payload["confidence"] == pytest.approx(0.8)
    assert payload["status"] == strategy._value(signal.status)
    assert payload["source"] == "dummy_liquidity_strategy"

    assert payload["signal"] is signal
    assert isinstance(payload["signal_payload"], dict)

    assert payload["analytics"]["liquidity"]["exchange"] == DEFAULT_EXCHANGE
    assert payload["analytics"]["liquidity"]["market_type"] == DEFAULT_MARKET_TYPE
    assert payload["analytics"]["liquidity"]["symbol"] == DEFAULT_SYMBOL
    assert payload["analytics"]["liquidity"]["timeframe"] == DEFAULT_TIMEFRAME


@pytest.mark.asyncio
async def test_emit_signal_without_snapshot_still_publishes_lightweight_payload() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(event_bus=event_bus)

    context = make_context(snapshot=None)
    signal = make_signal_like(category=strategy.category)

    result = await strategy.emit_signal(signal, context)

    assert result is signal

    events = event_bus.events_for("signal.generated")
    assert len(events) == 1

    payload = events[0].payload
    assert "analytics" not in payload
    assert payload["signal"] is signal
    assert isinstance(payload["signal_payload"], dict)


@pytest.mark.asyncio
async def test_emit_signal_rejects_symbol_mismatch_without_emitting() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(event_bus=event_bus)

    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)
    signal = make_signal_like(symbol="ETHUSDT", category=strategy.category)

    with pytest.raises(ValueError):
        await strategy.emit_signal(signal, context)

    assert event_bus.events_for("signal.generated") == []


@pytest.mark.asyncio
async def test_emit_signal_rejects_timeframe_mismatch_without_emitting() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(event_bus=event_bus)

    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)
    signal = make_signal_like(timeframe="5m", category=strategy.category)

    with pytest.raises(ValueError):
        await strategy.emit_signal(signal, context)

    assert event_bus.events_for("signal.generated") == []


@pytest.mark.asyncio
async def test_emit_signal_rejects_strategy_name_mismatch_without_emitting() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(event_bus=event_bus)

    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)
    signal = make_signal_like(
        strategy_name="wrong_strategy",
        category=strategy.category,
    )

    with pytest.raises(ValueError):
        await strategy.emit_signal(signal, context)

    assert event_bus.events_for("signal.generated") == []


@pytest.mark.asyncio
async def test_emit_signal_suppresses_second_signal_inside_emit_cooldown() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(
        event_bus=event_bus,
        config=make_config(emit_cooldown_seconds=60),
    )

    ts = datetime.now(timezone.utc)
    snapshot = make_valid_snapshot(timestamp=ts)
    context = make_context(snapshot=snapshot, timestamp=ts)
    signal = make_signal_like(category=strategy.category, timestamp=ts)

    first = await strategy.emit_signal(signal, context)
    second = await strategy.emit_signal(signal, context)

    assert first is signal
    assert second is None
    assert len(event_bus.events_for("signal.generated")) == 1


@pytest.mark.asyncio
async def test_emit_signal_allows_second_signal_after_cooldown_expires() -> None:
    event_bus = FakeEventBus()
    strategy = DummyLiquidityStrategy(
        event_bus=event_bus,
        config=make_config(emit_cooldown_seconds=60),
    )

    first_ts = datetime.now(timezone.utc)
    second_ts = first_ts + timedelta(seconds=61)

    first_snapshot = make_valid_snapshot(timestamp=first_ts)
    second_snapshot = make_valid_snapshot(timestamp=second_ts)

    first_context = make_context(snapshot=first_snapshot, timestamp=first_ts)
    second_context = make_context(snapshot=second_snapshot, timestamp=second_ts)

    signal = make_signal_like(category=strategy.category, timestamp=first_ts)

    first = await strategy.emit_signal(signal, first_context)
    second = await strategy.emit_signal(signal, second_context)

    assert first is signal
    assert second is signal
    assert len(event_bus.events_for("signal.generated")) == 2


def test_cooldown_key_contains_strategy_exchange_market_symbol_timeframe() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_valid_snapshot()
    context = make_context(snapshot=snapshot)

    key = strategy._cooldown_key(
        DEFAULT_SYMBOL,
        DEFAULT_TIMEFRAME,
        context=context,
    )

    assert "dummy_liquidity_strategy" in key
    assert DEFAULT_EXCHANGE in key
    assert DEFAULT_MARKET_TYPE in key
    assert DEFAULT_SYMBOL in key
    assert DEFAULT_TIMEFRAME in key


# ---------------------------------------------------------------------------
# Defensive behavior
# ---------------------------------------------------------------------------

def test_base_context_validation_does_not_raise_on_empty_snapshot() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_snapshot(
        active_levels=[],
        equal_levels=[],
        stop_clusters=[],
        zones=[],
        nearest_above_level=None,
        nearest_below_level=None,
        strongest_cluster_above=None,
        strongest_cluster_below=None,
    )
    context = make_context(snapshot=snapshot)

    assert strategy._base_context_is_valid(context, snapshot) is True


def test_collect_targets_above_does_not_raise_on_empty_snapshot() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_snapshot(
        active_levels=[],
        stop_clusters=[],
        nearest_above_level=None,
        strongest_cluster_above=None,
    )

    assert strategy._collect_targets_above(snapshot, DEFAULT_PRICE) == []


def test_collect_targets_below_does_not_raise_on_empty_snapshot() -> None:
    strategy = DummyLiquidityStrategy()
    snapshot = make_snapshot(
        active_levels=[],
        stop_clusters=[],
        nearest_below_level=None,
        strongest_cluster_below=None,
    )

    assert strategy._collect_targets_below(snapshot, DEFAULT_PRICE) == []


def test_liquidity_item_is_terminal_or_swept_detects_swept_level() -> None:
    strategy = DummyLiquidityStrategy()
    swept = make_buy_level(sweep_status=sweep_swept())

    assert strategy._liquidity_item_is_terminal_or_swept(swept) is True


def test_liquidity_item_is_terminal_or_swept_does_not_mark_active_level_terminal() -> None:
    strategy = DummyLiquidityStrategy()
    active = make_buy_level(sweep_status=sweep_active())

    assert strategy._liquidity_item_is_terminal_or_swept(active) is False