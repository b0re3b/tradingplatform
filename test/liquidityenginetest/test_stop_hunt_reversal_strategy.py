# test/liquidityenginetest/test_stop_hunt_reversal_strategy.py

from __future__ import annotations

import logging
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MethodType, SimpleNamespace
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
    LiquidityZone,
    StopCluster,
)
from strategy.enums import FilterDecision, SignalSide
from strategy.strategies.liquidity.stop_hunt_reversal_strategy import (
    StopHuntReversalStrategy,
)


DEFAULT_EXCHANGE = "binance"
DEFAULT_MARKET_TYPE = "usdm_futures"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_PRICE = 100.0

NON_FUTURES_MARKET_TYPES = ("spot", "margin", "cash")


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


def bias_down() -> LiquidityBias:
    return enum_member(
        LiquidityBias,
        "DOWN",
        "BEARISH",
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
    return enum_member(SweepStatus, "ACTIVE", "UNSWEPT", "NOT_SWEPT", "NONE")


def sweep_swept() -> SweepStatus:
    return enum_member(SweepStatus, "SWEPT", default=sweep_active())


def sweep_partially_swept() -> SweepStatus:
    return enum_member(
        SweepStatus,
        "PARTIALLY_SWEPT",
        "PARTIAL",
        default=sweep_swept(),
    )


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


def signal_short() -> SignalSide:
    return enum_member(SignalSide, "SHORT", "SELL")


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
    def __init__(self, ttl_seconds: float | None = 60.0) -> None:
        self.ttl_seconds = ttl_seconds

    def get_ttl(self, feature_name: str) -> float | None:
        return self.ttl_seconds


def make_config(
    *,
    enabled: bool = True,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
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
        cooldown_seconds=0,
        emit_cooldown_seconds=0,
        signal_cooldown_seconds=0,
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


# ---------------------------------------------------------------------------
# Model builders
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
    price: float = DEFAULT_PRICE,
    confidence: float = 0.90,
    touches_count: int = 5,
    reaction_count: int = 3,
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
        "NOT_SWEPT",
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
        touches_count=touches_count,
        reaction_count=reaction_count,
        timestamp=now,
        created_at=now,
        updated_at=now,
        key=key or f"level:{exchange}:{market_type}:{symbol}:{timeframe}:{price}",
        metadata={},
        **overrides,
    )


def make_buy_level(
    *,
    price: float = DEFAULT_PRICE * 1.015,
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
    price: float = DEFAULT_PRICE * 0.985,
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
    center_price: float = DEFAULT_PRICE,
    confidence: float = 0.90,
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
        "NOT_SWEPT",
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


def make_zone(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    side: LiquiditySide | str | None = None,
    center_price: float = DEFAULT_PRICE,
    score: float = 0.90,
    **overrides: Any,
) -> LiquidityZone:
    side_value = enum_from_value(
        LiquiditySide,
        side,
        "BUY_SIDE",
        "BUY",
        "BID",
    )

    return build_dataclass(
        LiquidityZone,
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
        center_price=center_price,
        low_price=center_price * 0.998,
        high_price=center_price * 1.002,
        score=score,
        confidence=score,
        metadata={},
        **overrides,
    )


def make_analytics_signal(
    *,
    bias: LiquidityBias | str | None = None,
    confidence: float = 0.90,
    **overrides: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        bias=enum_from_value(
            LiquidityBias,
            bias,
            "NEUTRAL",
            "NONE",
            "UP",
            "BULLISH",
        ),
        confidence=confidence,
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
    above_liquidity_score: float = 0.5,
    below_liquidity_score: float = 0.5,
    liquidity_pressure_score: float = 0.0,
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
            "NEUTRAL",
            "NONE",
            "UP",
            "BULLISH",
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
        metadata=metadata or {"confidence": 0.85},
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


def make_sell_side_sweep_reclaim_snapshot(
    *,
    current_price: float = DEFAULT_PRICE,
    swept_price: float = DEFAULT_PRICE * 0.985,
    target_price: float = DEFAULT_PRICE * 1.025,
    use_cluster: bool = False,
    partial: bool = False,
    **overrides: Any,
) -> LiquidityMapSnapshot:
    status = sweep_partially_swept() if partial else sweep_swept()

    swept_level = make_sell_level(
        price=swept_price,
        sweep_status=status,
        confidence=0.94,
        touches_count=6,
        reaction_count=4,
    )
    swept_cluster = make_cluster(
        side=sell_side(),
        center_price=swept_price,
        sweep_status=status,
        confidence=0.93,
    )
    target = make_buy_level(
        price=target_price,
        sweep_status=sweep_active(),
        confidence=0.88,
    )
    zone = make_zone(
        side=sell_side(),
        center_price=swept_price,
        score=0.90,
    )

    defaults: dict[str, Any] = {
        "current_price": current_price,
        "active_levels": [target],
        "equal_levels": [swept_level],
        "stop_clusters": [swept_cluster] if use_cluster else [],
        "zones": [zone],
        "nearest_below_level": swept_level,
        "nearest_above_level": target,
        "strongest_cluster_below": swept_cluster if use_cluster else None,
        "bias": bias_down(),
        "above_liquidity_score": 0.82,
        "below_liquidity_score": 0.24,
        "liquidity_pressure_score": -0.65,
        "signal": make_analytics_signal(
            bias=bias_down(),
            confidence=0.90,
            sweep_risk_down=0.92,
            magnet_score_up=0.78,
        ),
        "metadata": {"confidence": 0.90},
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


def make_buy_side_sweep_rejection_snapshot(
    *,
    current_price: float = DEFAULT_PRICE,
    swept_price: float = DEFAULT_PRICE * 1.015,
    target_price: float = DEFAULT_PRICE * 0.975,
    use_cluster: bool = False,
    partial: bool = False,
    **overrides: Any,
) -> LiquidityMapSnapshot:
    status = sweep_partially_swept() if partial else sweep_swept()

    swept_level = make_buy_level(
        price=swept_price,
        sweep_status=status,
        confidence=0.94,
        touches_count=6,
        reaction_count=4,
    )
    swept_cluster = make_cluster(
        side=buy_side(),
        center_price=swept_price,
        sweep_status=status,
        confidence=0.93,
    )
    target = make_sell_level(
        price=target_price,
        sweep_status=sweep_active(),
        confidence=0.88,
    )
    zone = make_zone(
        side=buy_side(),
        center_price=swept_price,
        score=0.90,
    )

    defaults: dict[str, Any] = {
        "current_price": current_price,
        "active_levels": [target],
        "equal_levels": [swept_level],
        "stop_clusters": [swept_cluster] if use_cluster else [],
        "zones": [zone],
        "nearest_above_level": swept_level,
        "nearest_below_level": target,
        "strongest_cluster_above": swept_cluster if use_cluster else None,
        "bias": bias_up(),
        "above_liquidity_score": 0.24,
        "below_liquidity_score": 0.82,
        "liquidity_pressure_score": 0.65,
        "signal": make_analytics_signal(
            bias=bias_up(),
            confidence=0.90,
            sweep_risk_up=0.92,
            magnet_score_down=0.78,
        ),
        "metadata": {"confidence": 0.90},
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


def make_unswept_stop_hunt_like_snapshot(
    *,
    direction: str = "long",
    current_price: float = DEFAULT_PRICE,
    **overrides: Any,
) -> LiquidityMapSnapshot:
    if direction == "long":
        unswept = make_sell_level(
            price=current_price * 0.985,
            sweep_status=sweep_active(),
            confidence=0.95,
        )
        target = make_buy_level(price=current_price * 1.025)
        defaults: dict[str, Any] = {
            "current_price": current_price,
            "active_levels": [unswept, target],
            "equal_levels": [unswept],
            "nearest_below_level": unswept,
            "nearest_above_level": target,
            "bias": bias_down(),
            "liquidity_pressure_score": -0.65,
            "signal": make_analytics_signal(bias=bias_down(), confidence=0.90),
        }
        defaults.update(overrides)
        return make_snapshot(**defaults)

    unswept = make_buy_level(
        price=current_price * 1.015,
        sweep_status=sweep_active(),
        confidence=0.95,
    )
    target = make_sell_level(price=current_price * 0.975)
    defaults = {
        "current_price": current_price,
        "active_levels": [unswept, target],
        "equal_levels": [unswept],
        "nearest_above_level": unswept,
        "nearest_below_level": target,
        "bias": bias_up(),
        "liquidity_pressure_score": 0.65,
        "signal": make_analytics_signal(bias=bias_up(), confidence=0.90),
    }
    defaults.update(overrides)
    return make_snapshot(**defaults)


def make_context(
    *,
    exchange: str = DEFAULT_EXCHANGE,
    market_type: str = DEFAULT_MARKET_TYPE,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    current_price: float | None = DEFAULT_PRICE,
    timestamp: datetime | None = None,
    snapshot: LiquidityMapSnapshot | None = None,
) -> FakeStrategyContext:
    context = FakeStrategyContext(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        timeframe=timeframe,
        current_price=current_price,
        timestamp=timestamp or datetime.now(timezone.utc),
    )

    if snapshot is not None:
        context.liquidity = FakeLiquidityContext(snapshot=snapshot)

    return context


# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------

def make_strategy(
    *,
    config: Any | None = None,
    event_bus: FakeEventBus | None = None,
) -> StopHuntReversalStrategy:
    strategy = StopHuntReversalStrategy.__new__(StopHuntReversalStrategy)
    strategy.config = config or make_config()
    strategy.event_bus = event_bus or FakeEventBus()
    strategy.logger = logging.getLogger("test.stop_hunt_reversal_strategy")
    strategy._last_emitted_at = {}

    strategy.validate_context = MethodType(lambda self, context: None, strategy)

    async def emit_event(
        self: StopHuntReversalStrategy,
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

    strategy.emit_event = MethodType(emit_event, strategy)

    strategy.log_debug = MethodType(lambda self, *args, **kwargs: None, strategy)
    strategy.log_info = MethodType(lambda self, *args, **kwargs: None, strategy)
    strategy.log_warning = MethodType(lambda self, *args, **kwargs: None, strategy)
    strategy.log_error = MethodType(lambda self, *args, **kwargs: None, strategy)

    return strategy


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_no_signal(signal: Any) -> None:
    assert signal is None


def assert_signal_is_valid(signal: Any) -> None:
    assert signal is not None
    assert signal.strategy_name == "stop_hunt_reversal_strategy"
    assert signal.symbol == DEFAULT_SYMBOL
    assert signal.timeframe == DEFAULT_TIMEFRAME
    assert signal.confidence > 0.0
    assert signal.score > 0.0

    assert getattr(signal, "entry_plan", None) is not None
    assert getattr(signal, "exit_plan", None) is not None
    assert getattr(signal, "invalidation_plan", None) is not None
    assert getattr(signal, "execution_plan", None) is not None

    validator = getattr(signal, "validate", None)
    assert callable(validator)
    validator()


def assert_filter_decisions_are_enums(results: list[Any]) -> None:
    assert results, "Expected at least one FilterResult"

    for result in results:
        assert isinstance(result.decision, FilterDecision), (
            f"{getattr(result, 'name', '<unknown>')} decision must be FilterDecision, "
            f"got {result.decision!r} ({type(result.decision).__name__})"
        )


def merged_signal_metadata(signal: Any) -> dict[str, Any]:
    metadata = getattr(signal, "metadata", {}) or {}
    entry_metadata = getattr(signal.entry_plan, "metadata", {}) or {}
    execution_metadata = getattr(signal.execution_plan, "metadata", {}) or {}

    return {
        **metadata,
        **entry_metadata,
        **execution_metadata,
    }


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_strategy_name_is_stop_hunt_reversal_strategy() -> None:
    strategy = make_strategy()

    assert strategy.strategy_name == "stop_hunt_reversal_strategy"


def test_returns_none_when_strategy_is_disabled() -> None:
    strategy = make_strategy(config=make_config(enabled=False))
    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


def test_returns_none_without_liquidity_snapshot() -> None:
    strategy = make_strategy()
    context = make_context(snapshot=None)

    assert_no_signal(strategy.evaluate(context))


@pytest.mark.parametrize(
    "bad_price",
    [None, 0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_returns_none_when_all_current_price_sources_are_invalid(
    bad_price: float | None,
) -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(current_price=DEFAULT_PRICE)

    snapshot.current_price = bad_price  # type: ignore[assignment]
    if hasattr(snapshot, "mark_price"):
        snapshot.mark_price = bad_price  # type: ignore[assignment]
    if hasattr(snapshot, "last_price"):
        snapshot.last_price = bad_price  # type: ignore[assignment]

    context = make_context(snapshot=snapshot, current_price=bad_price)
    context.price = SimpleNamespace(
        mid_price=bad_price,
        last_price=bad_price,
        mark_price=bad_price,
        index_price=bad_price,
    )

    assert_no_signal(strategy.evaluate(context))


# ---------------------------------------------------------------------------
# Base guards: futures / scope / freshness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("market_type", NON_FUTURES_MARKET_TYPES)
def test_rejects_non_futures_market_type(market_type: str) -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(market_type=market_type)
    context = make_context(market_type=market_type, snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


@pytest.mark.parametrize(
    ("field_name", "context_value"),
    [
        ("exchange", "bybit"),
        ("market_type", "linear"),
        ("symbol", "ETHUSDT"),
        ("timeframe", "5m"),
    ],
)
def test_rejects_snapshot_context_scope_mismatch(
    field_name: str,
    context_value: str,
) -> None:
    strategy = make_strategy()
    snapshot = make_sell_side_sweep_reclaim_snapshot()

    kwargs = {
        "exchange": snapshot.exchange,
        "market_type": snapshot.market_type,
        "symbol": snapshot.symbol,
        "timeframe": snapshot.timeframe,
        "snapshot": snapshot,
    }
    kwargs[field_name] = context_value

    context = make_context(**kwargs)

    assert_no_signal(strategy.evaluate(context))


def test_rejects_stale_snapshot() -> None:
    now = datetime.now(timezone.utc)

    strategy = make_strategy(config=make_config(ttl_seconds=30))
    snapshot = make_sell_side_sweep_reclaim_snapshot(timestamp=now - timedelta(hours=1))
    context = make_context(snapshot=snapshot, timestamp=now)

    assert_no_signal(strategy.evaluate(context))


# ---------------------------------------------------------------------------
# Evidence requirements
# ---------------------------------------------------------------------------

def test_does_not_signal_from_active_unswept_sell_side_liquidity() -> None:
    strategy = make_strategy()

    snapshot = make_unswept_stop_hunt_like_snapshot(direction="long")
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


def test_does_not_signal_from_active_unswept_buy_side_liquidity() -> None:
    strategy = make_strategy()

    snapshot = make_unswept_stop_hunt_like_snapshot(direction="short")
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


def test_rejects_sell_side_sweep_without_reclaim() -> None:
    strategy = make_strategy()

    swept_level = make_sell_level(
        price=DEFAULT_PRICE * 0.985,
        sweep_status=sweep_swept(),
        confidence=0.95,
    )
    target = make_buy_level(price=DEFAULT_PRICE * 1.025)

    snapshot = make_snapshot(
        current_price=DEFAULT_PRICE * 0.980,
        active_levels=[target],
        equal_levels=[swept_level],
        nearest_below_level=swept_level,
        nearest_above_level=target,
        bias=bias_down(),
        liquidity_pressure_score=-0.70,
        signal=make_analytics_signal(bias=bias_down(), confidence=0.90),
    )
    context = make_context(snapshot=snapshot, current_price=DEFAULT_PRICE * 0.980)

    assert_no_signal(strategy.evaluate(context))


def test_rejects_buy_side_sweep_without_rejection() -> None:
    strategy = make_strategy()

    swept_level = make_buy_level(
        price=DEFAULT_PRICE * 1.015,
        sweep_status=sweep_swept(),
        confidence=0.95,
    )
    target = make_sell_level(price=DEFAULT_PRICE * 0.975)

    snapshot = make_snapshot(
        current_price=DEFAULT_PRICE * 1.020,
        active_levels=[target],
        equal_levels=[swept_level],
        nearest_above_level=swept_level,
        nearest_below_level=target,
        bias=bias_up(),
        liquidity_pressure_score=0.70,
        signal=make_analytics_signal(bias=bias_up(), confidence=0.90),
    )
    context = make_context(snapshot=snapshot, current_price=DEFAULT_PRICE * 1.020)

    assert_no_signal(strategy.evaluate(context))


def test_rejects_swept_evidence_too_far_from_current_price() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(
        current_price=DEFAULT_PRICE,
        swept_price=DEFAULT_PRICE * 0.50,
        target_price=DEFAULT_PRICE * 1.025,
    )
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


def test_accepts_partially_swept_sell_side_evidence() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(partial=True)
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)
    assert signal.side == signal_long()


def test_accepts_partially_swept_buy_side_evidence() -> None:
    strategy = make_strategy()

    snapshot = make_buy_side_sweep_rejection_snapshot(partial=True)
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)
    assert signal.side == signal_short()


# ---------------------------------------------------------------------------
# Positive behavior
# ---------------------------------------------------------------------------

def test_generates_long_after_sell_side_sweep_and_reclaim() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)
    assert signal.side == signal_long()


def test_generates_short_after_buy_side_sweep_and_rejection() -> None:
    strategy = make_strategy()

    snapshot = make_buy_side_sweep_rejection_snapshot()
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)
    assert signal.side == signal_short()


def test_can_use_swept_level_as_evidence() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(use_cluster=False)
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)

    metadata = merged_signal_metadata(signal)
    assert metadata["setup_name"] == "stop_hunt_reversal"
    assert "evidence" in metadata


def test_can_use_swept_cluster_as_evidence() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(
        use_cluster=True,
        equal_levels=[],
        nearest_below_level=None,
    )
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)

    metadata = merged_signal_metadata(signal)
    assert metadata["setup_name"] == "stop_hunt_reversal"
    assert "evidence" in metadata


def test_generated_signal_contains_stop_hunt_metadata() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot(use_cluster=True)
    context = make_context(snapshot=snapshot)

    signal = strategy.evaluate(context)

    assert_signal_is_valid(signal)

    metadata = merged_signal_metadata(signal)

    assert metadata["setup_name"] == "stop_hunt_reversal"
    assert metadata["strategy_name"] == "stop_hunt_reversal_strategy"
    assert metadata["exchange"] == DEFAULT_EXCHANGE
    assert metadata["market_type"] == DEFAULT_MARKET_TYPE
    assert metadata["scope_key"] == f"{DEFAULT_EXCHANGE}:{DEFAULT_MARKET_TYPE}:{DEFAULT_SYMBOL}:{DEFAULT_TIMEFRAME}"
    assert "target_price" in metadata or "target" in metadata


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_signal_is_rejected_when_below_runtime_min_confidence() -> None:
    strategy = make_strategy(
        config=make_config(
            min_confidence=1.01,
            min_score=0.0,
        )
    )

    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


def test_signal_is_rejected_when_below_runtime_min_score() -> None:
    strategy = make_strategy(
        config=make_config(
            min_confidence=0.0,
            min_score=99.0,
        )
    )

    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    assert_no_signal(strategy.evaluate(context))


# ---------------------------------------------------------------------------
# Private helper / regression checks
# ---------------------------------------------------------------------------

def test_pre_filters_use_filter_decision_enum_not_strings() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    results = strategy._run_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert_filter_decisions_are_enums(results)


def test_pre_filters_block_when_no_swept_or_partially_swept_evidence_exists() -> None:
    strategy = make_strategy()

    snapshot = make_unswept_stop_hunt_like_snapshot(direction="long")
    context = make_context(snapshot=snapshot)

    results = strategy._run_pre_filters(
        context=context,
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert_filter_decisions_are_enums(results)
    assert any(result.blocked for result in results)
    assert any(
        "swept" in result.reason.lower() or "evidence" in result.reason.lower()
        for result in results
        if result.blocked
    )


def test_direction_for_sell_side_sweep_is_long() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot()

    side = strategy._side_from_swept_evidence(
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert side == signal_long()


def test_direction_for_buy_side_sweep_is_short() -> None:
    strategy = make_strategy()

    snapshot = make_buy_side_sweep_rejection_snapshot()

    side = strategy._side_from_swept_evidence(
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
    )

    assert side == signal_short()


def test_target_for_long_reversal_is_above_current_price() -> None:
    strategy = make_strategy()

    snapshot = make_sell_side_sweep_reclaim_snapshot()

    target = strategy._target_for_side(
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
        side=signal_long(),
    )

    assert target is not None
    assert strategy._reference_price(target) > DEFAULT_PRICE


def test_target_for_short_reversal_is_below_current_price() -> None:
    strategy = make_strategy()

    snapshot = make_buy_side_sweep_rejection_snapshot()

    target = strategy._target_for_side(
        snapshot=snapshot,
        current_price=DEFAULT_PRICE,
        side=signal_short(),
    )

    assert target is not None
    assert strategy._reference_price(target) < DEFAULT_PRICE


# ---------------------------------------------------------------------------
# Emit contract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_and_emit_publishes_signal_generated() -> None:
    event_bus = FakeEventBus()
    strategy = make_strategy(event_bus=event_bus)

    snapshot = make_sell_side_sweep_reclaim_snapshot()
    context = make_context(snapshot=snapshot)

    signal = await strategy.evaluate_and_emit(context)

    assert_signal_is_valid(signal)

    events = event_bus.events_for("signal.generated")
    assert len(events) == 1

    payload = events[0].payload

    assert payload["symbol"] == DEFAULT_SYMBOL
    assert payload["strategy_name"] == "stop_hunt_reversal_strategy"
    assert payload["signal"] is signal
    assert "signal_payload" in payload
    assert payload["analytics"]["liquidity"]["exchange"] == DEFAULT_EXCHANGE
    assert payload["analytics"]["liquidity"]["market_type"] == DEFAULT_MARKET_TYPE


@pytest.mark.asyncio
async def test_evaluate_and_emit_does_not_emit_when_evaluate_returns_none() -> None:
    event_bus = FakeEventBus()
    strategy = make_strategy(event_bus=event_bus)

    snapshot = make_unswept_stop_hunt_like_snapshot(direction="long")
    context = make_context(snapshot=snapshot)

    signal = await strategy.evaluate_and_emit(context)

    assert signal is None
    assert event_bus.events_for("signal.generated") == []