from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any

from core.event_bus import EventBus
from core.logger import get_logger

from analytics.liquidity.enums import LiquiditySide
from analytics.liquidity.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidityLevel,
    LiquidityMapSnapshot,
    LiquidityZone,
    StopCluster,
    make_liquidity_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)

try:
    from strategy.base import BaseStrategyComponent as _StrategyBase
except ImportError:  # backward compatibility while strategy package is being migrated
    from strategy.base import StrategyComponent as _StrategyBase

from strategy.config import StrategyConfig, StrategyDefinitionConfig
from strategy.enums import FilterDecision, StrategyCategory
from strategy.models import FilterResult, StrategySignal


class BaseLiquidityStrategy(_StrategyBase, ABC):
    """
    Production-ready base class для strategy/strategies/liquidity.

    Відповідальність:
    - працює тільки з готовим analytics.liquidity.models.LiquidityMapSnapshot;
    - не викликає analytics detectors / LiquidityMap напряму;
    - не читає exchange/data cache напряму;
    - не містить execution/risk логіки;
    - валідує повний scope: exchange + market_type + symbol + timeframe;
    - enforce-ить futures/perpetual-only режим;
    - централізує extraction snapshot-а зі StrategyContext-like object;
    - централізує common filters, freshness checks, signal emission і metadata;
    - надає helpers для дочірніх liquidity strategies:
      sweep/magnet metrics, nearest/strongest targets, zones, score helpers.

    Очікуваний production flow:
        analytics.liquidity.* -> StrategyContextBuilder -> StrategyContext
        -> BaseLiquidityStrategy subclasses -> signal.generated
        -> risk -> execution
    """

    SIGNAL_TOPIC = "signal.generated"

    SNAPSHOT_FEATURE_NAME = "liquidity_map_snapshot"

    SNAPSHOT_DOMAIN_KEYS: tuple[str, ...] = (
        "snapshot",
        "liquidity_map_snapshot",
        "map_snapshot",
        "last_snapshot",
    )

    SNAPSHOT_FEATURE_KEYS: tuple[str, ...] = (
        "liquidity_map_snapshot",
        "liquidity.snapshot",
        "liquidity_snapshot",
        "liquidity.map.snapshot",
        "analytics.liquidity.map.snapshot",
        "analytics.liquidity.map.updated",
    )

    FUTURES_MARKET_TYPES: frozenset[str] = frozenset(
        {
            "perpetual",
            "futures",
            "linear",
            "inverse",
            "swap",
            "usdm_futures",
            "coinm_futures",
        }
    )

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=logger
            or get_logger(
                __name__,
                service_name="strategy",
                event_type="liquidity_strategy",
            ),
        )

        self.validate_config()
        self._last_emitted_at: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Strategy metadata
    # ------------------------------------------------------------------

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.LIQUIDITY

    @property
    def _strategy_cfg(self) -> StrategyDefinitionConfig | None:
        getter = getattr(self.config, "get_strategy", None)
        if not callable(getter):
            return None
        return getter(self.strategy_name)

    @property
    def _runtime(self) -> Any:
        strategy_cfg = self._strategy_cfg
        return strategy_cfg.runtime if strategy_cfg is not None else self.config.runtime

    @property
    def priority(self) -> int:
        strategy_cfg = self._strategy_cfg
        return int(strategy_cfg.priority) if strategy_cfg is not None else 100

    def is_enabled(self) -> bool:
        strategy_cfg = self._strategy_cfg
        if strategy_cfg is None:
            return bool(getattr(self.config.runtime, "enabled", True))
        return bool(strategy_cfg.runtime.enabled)

    def required_features(self) -> set[str]:
        strategy_cfg = self._strategy_cfg

        if strategy_cfg is None:
            return {self.SNAPSHOT_FEATURE_NAME}

        configured = set(getattr(strategy_cfg, "required_features", set()) or set())
        configured.add(self.SNAPSHOT_FEATURE_NAME)
        return configured

    # ------------------------------------------------------------------
    # Public emit contract
    # ------------------------------------------------------------------

    async def emit_signal(
        self,
        signal: StrategySignal,
        context: Any,
        **emit_kwargs: Any,
    ) -> StrategySignal | None:
        """
        Єдиний стандарт публікації StrategySignal.

        Payload містить:
        - lightweight fields для routing;
        - сам signal object для internal pipeline;
        - payload-safe signal_payload;
        - full analytics liquidity scope/metadata, якщо snapshot є в context.
        """
        self._validate_signal_context_pair(signal=signal, context=context)

        context_ts = self._context_timestamp(context)
        if self._is_on_emit_cooldown(
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            timestamp=context_ts,
        ):
            self.log_debug(
                "Liquidity strategy signal suppressed by cooldown",
                symbol=signal.symbol,
                timeframe=self._value(signal.timeframe),
                strategy_name=self.strategy_name,
            )
            return None

        snapshot = self._extract_snapshot(context)
        signal_payload = self._to_payload(signal)

        payload: dict[str, Any] = {
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "category": self._value(signal.category),
            "timeframe": self._value(signal.timeframe),
            "side": self._value(signal.side),
            "score": self._safe_float(signal.score, 0.0),
            "confidence": self._clamp01(signal.confidence),
            "status": self._value(signal.status),
            "priority": self._value(signal.priority),
            "source": self.strategy_name,
            "signal": signal,
            "signal_payload": signal_payload,
        }

        if snapshot is not None:
            payload["analytics"] = {
                "liquidity": self._build_liquidity_signal_metadata(
                    snapshot=snapshot,
                    current_price=self._resolve_current_price(context, snapshot),
                )
            }

        await self.emit_event(
            self.SIGNAL_TOPIC,
            payload,
            source=self.strategy_name,
            **emit_kwargs,
        )

        self._last_emitted_at[
            self._cooldown_key(signal.symbol, signal.timeframe, context=context)
        ] = context_ts

        self.log_info(
            "Liquidity strategy signal emitted",
            symbol=signal.symbol,
            timeframe=self._value(signal.timeframe),
            side=self._value(signal.side),
            confidence=self._clamp01(signal.confidence),
            score=self._safe_float(signal.score, 0.0),
            strategy_name=self.strategy_name,
        )

        return signal

    # ------------------------------------------------------------------
    # Context / snapshot extraction
    # ------------------------------------------------------------------

    def _extract_snapshot(self, context: Any) -> LiquidityMapSnapshot | None:
        """
        Дістає LiquidityMapSnapshot зі StrategyContext-like object.

        Підтримує:
        - context.liquidity.snapshot / liquidity_map_snapshot / map_snapshot;
        - context.get_feature(...);
        - context.get_feature_snapshot(...);
        - wrappers: .value / .data / .snapshot / .payload;
        - mapping wrappers: value/data/snapshot/payload.
        """
        liquidity_context = getattr(context, "liquidity", None)

        domain_candidates = [
            self._mapping_or_attr_get(liquidity_context, key)
            for key in self.SNAPSHOT_DOMAIN_KEYS
        ]

        feature_candidates: list[Any] = []

        for key in self.SNAPSHOT_FEATURE_KEYS:
            getter = getattr(context, "get_feature", None)
            if callable(getter):
                try:
                    feature_candidates.append(getter(key))
                except Exception:
                    self.log_debug(
                        "Failed to read liquidity feature from context",
                        symbol=getattr(context, "symbol", None),
                        feature_name=key,
                        strategy_name=self.strategy_name,
                    )

            snapshot_getter = getattr(context, "get_feature_snapshot", None)
            if callable(snapshot_getter):
                try:
                    feature_candidates.append(snapshot_getter(key))
                except Exception:
                    pass

        for candidate in [*domain_candidates, *feature_candidates]:
            snapshot = self._unwrap_snapshot_candidate(candidate)
            if snapshot is not None:
                return snapshot

        return None

    def _unwrap_snapshot_candidate(
        self,
        candidate: Any,
    ) -> LiquidityMapSnapshot | None:
        if isinstance(candidate, LiquidityMapSnapshot):
            return candidate

        if candidate is None:
            return None

        if isinstance(candidate, Mapping):
            for key in ("snapshot", "value", "data", "payload"):
                nested = candidate.get(key)
                if isinstance(nested, LiquidityMapSnapshot):
                    return nested
                nested_unwrapped = self._unwrap_snapshot_candidate(nested)
                if nested_unwrapped is not None:
                    return nested_unwrapped

        for attr in ("snapshot", "value", "data", "payload"):
            nested = getattr(candidate, attr, None)
            if isinstance(nested, LiquidityMapSnapshot):
                return nested
            nested_unwrapped = self._unwrap_snapshot_candidate(nested)
            if nested_unwrapped is not None:
                return nested_unwrapped

        return None

    # ------------------------------------------------------------------
    # Base validation / filters
    # ------------------------------------------------------------------

    def _base_context_is_valid(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        if not self._snapshot_matches_context(context=context, snapshot=snapshot):
            return False

        if not self._is_futures_market_type(snapshot.market_type):
            self.log_debug(
                "Liquidity strategy skipped: non-futures market_type",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_exchange_allowed(snapshot.exchange):
            self.log_debug(
                "Liquidity strategy skipped: exchange not allowed",
                exchange=snapshot.exchange,
                symbol=snapshot.symbol,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_market_type_allowed(snapshot.market_type):
            self.log_debug(
                "Liquidity strategy skipped: market_type not allowed",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_symbol_allowed(snapshot.symbol):
            self.log_debug(
                "Liquidity strategy skipped: symbol not allowed",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_timeframe_allowed(snapshot.timeframe):
            self.log_debug(
                "Liquidity strategy skipped: timeframe not allowed",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                strategy_name=self.strategy_name,
            )
            return False

        if not self._is_regime_allowed(context):
            self.log_debug(
                "Liquidity strategy skipped: regime not allowed",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                regime=self._value(
                    self._mapping_or_attr_get(
                        getattr(context, "regime", None),
                        "regime",
                    )
                ),
                strategy_name=self.strategy_name,
            )
            return False

        if self._snapshot_is_stale(context, snapshot):
            self.log_debug(
                "Liquidity strategy skipped: stale liquidity snapshot",
                exchange=snapshot.exchange,
                market_type=snapshot.market_type,
                symbol=snapshot.symbol,
                timeframe=snapshot.timeframe,
                snapshot_ts=snapshot.timestamp.isoformat(),
                context_ts=self._context_timestamp(context).isoformat(),
                strategy_name=self.strategy_name,
            )
            return False

        return True

    def _run_common_pre_filters(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        results: list[FilterResult] = []

        optional_filters: tuple[FilterResult | None, ...] = (
            self._scope_filter(context=context, snapshot=snapshot),
            self._futures_market_filter(snapshot),
            self._portfolio_filter(context),
            self._spread_filter(context),
            self._liquidity_strength_filter(snapshot),
        )

        for filter_result in optional_filters:
            if filter_result is not None:
                results.append(filter_result)

        results.append(self._snapshot_presence_filter(snapshot))
        results.append(self._price_validation_filter(current_price))

        return results

    def _scope_filter(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult:
        if self._snapshot_matches_context(context=context, snapshot=snapshot):
            return FilterResult(
                name="liquidity_scope_filter",
                decision=FilterDecision.PASS,
                reason=f"Liquidity snapshot scope matches context: {snapshot.scope_key}",
            )

        return FilterResult(
            name="liquidity_scope_filter",
            decision=FilterDecision.BLOCK,
            reason=(
                "Liquidity snapshot scope does not match context: "
                f"snapshot={snapshot.scope}, context={self._context_scope(context)}"
            ),
        )

    def _futures_market_filter(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult:
        if self._is_futures_market_type(snapshot.market_type):
            return FilterResult(
                name="futures_market_filter",
                decision=FilterDecision.PASS,
                reason=f"Futures market_type accepted: {snapshot.market_type}",
            )

        return FilterResult(
            name="futures_market_filter",
            decision=FilterDecision.BLOCK,
            reason=f"Non-futures market_type rejected: {snapshot.market_type}",
        )

    def _portfolio_filter(self, context: Any) -> FilterResult | None:
        filters_cfg = getattr(self.config, "filters", None)
        if filters_cfg is None or not getattr(filters_cfg, "enable_portfolio_filter", False):
            return None

        portfolio = getattr(context, "portfolio", None)
        if portfolio is None:
            return FilterResult(
                name="portfolio_filter",
                decision=FilterDecision.PASS,
                reason="Portfolio context unavailable; filter skipped safely",
            )

        exposure = self._safe_float(
            self._mapping_or_attr_get(portfolio, "exposure_pct"),
            0.0,
        )
        max_exposure = self._safe_float(
            getattr(filters_cfg, "max_portfolio_exposure_pct", 1.0),
            1.0,
        )

        if max_exposure > 0 and exposure > max_exposure:
            return FilterResult(
                name="portfolio_filter",
                decision=FilterDecision.BLOCK,
                reason=f"Portfolio exposure too high: {exposure:.4f} > {max_exposure:.4f}",
            )

        return FilterResult(
            name="portfolio_filter",
            decision=FilterDecision.PASS,
            reason="Portfolio exposure within limits",
        )

    def _spread_filter(self, context: Any) -> FilterResult | None:
        filters_cfg = getattr(self.config, "filters", None)
        if filters_cfg is None or not getattr(filters_cfg, "enable_spread_filter", False):
            return None

        price_context = getattr(context, "price", None)
        spread = self._safe_float(
            self._mapping_or_attr_get(price_context, "spread_bps"),
            0.0,
        )

        max_spread = self._safe_float(
            getattr(filters_cfg, "max_spread_bps", 0.0),
            0.0,
        )

        if max_spread > 0 and spread > max_spread:
            return FilterResult(
                name="spread_filter",
                decision=FilterDecision.BLOCK,
                reason=f"Spread too high: {spread:.2f} bps",
            )

        return FilterResult(
            name="spread_filter",
            decision=FilterDecision.PASS,
            reason="Spread within threshold",
        )

    def _liquidity_strength_filter(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult | None:
        filters_cfg = getattr(self.config, "filters", None)
        if filters_cfg is None or not getattr(filters_cfg, "enable_liquidity_filter", False):
            return None

        strongest_liquidity = max(
            self._clamp01(snapshot.above_liquidity_score),
            self._clamp01(snapshot.below_liquidity_score),
        )

        min_score = self._safe_float(
            getattr(filters_cfg, "min_liquidity_score", 0.0),
            0.0,
        )

        if strongest_liquidity < min_score:
            return FilterResult(
                name="liquidity_strength_filter",
                decision=FilterDecision.BLOCK,
                reason=f"Liquidity score too low: {strongest_liquidity:.4f}",
            )

        return FilterResult(
            name="liquidity_strength_filter",
            decision=FilterDecision.PASS,
            reason=f"Liquidity score OK: {strongest_liquidity:.4f}",
        )

    def _snapshot_presence_filter(
        self,
        snapshot: LiquidityMapSnapshot,
    ) -> FilterResult:
        if not snapshot.has_levels():
            return FilterResult(
                name="liquidity_snapshot_presence",
                decision=FilterDecision.BLOCK,
                reason="Liquidity snapshot has no active levels or clusters",
            )

        return FilterResult(
            name="liquidity_snapshot_presence",
            decision=FilterDecision.PASS,
            reason="Liquidity snapshot contains active liquidity structures",
        )

    def _price_validation_filter(self, current_price: float) -> FilterResult:
        price = self._safe_positive_float(current_price)

        if price is None:
            return FilterResult(
                name="price_validation",
                decision=FilterDecision.BLOCK,
                reason="Current price must be positive and finite",
            )

        return FilterResult(
            name="price_validation",
            decision=FilterDecision.PASS,
            reason="Current price is valid",
        )

    # ------------------------------------------------------------------
    # Scope / freshness helpers
    # ------------------------------------------------------------------

    def _snapshot_matches_context(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        context_scope = self._context_scope(context)

        expected_key = make_liquidity_key(
            exchange=context_scope["exchange"],
            market_type=context_scope["market_type"],
            symbol=context_scope["symbol"],
            timeframe=context_scope["timeframe"],
        )

        return snapshot.liquidity_key == expected_key

    def _context_scope(self, context: Any) -> dict[str, str]:
        symbol = self._context_symbol(context)
        timeframe = self._context_timeframe(context)
        exchange = self._context_exchange(context)
        market_type = self._context_market_type(context)

        return {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "timeframe": timeframe,
        }

    def _context_symbol(self, context: Any) -> str:
        return normalize_symbol(getattr(context, "symbol", None))

    def _context_timeframe(self, context: Any) -> str:
        return normalize_timeframe(getattr(context, "timeframe", DEFAULT_TIMEFRAME))

    def _context_exchange(self, context: Any) -> str:
        value = getattr(context, "exchange", None)

        if value is None:
            market = getattr(context, "market", None)
            value = self._mapping_or_attr_get(market, "exchange")

        if value is None:
            liquidity = getattr(context, "liquidity", None)
            value = self._mapping_or_attr_get(liquidity, "exchange")

        return normalize_exchange(value or DEFAULT_EXCHANGE)

    def _context_market_type(self, context: Any) -> str:
        value = getattr(context, "market_type", None)

        if value is None:
            market = getattr(context, "market", None)
            value = self._mapping_or_attr_get(market, "market_type")

        if value is None:
            liquidity = getattr(context, "liquidity", None)
            value = self._mapping_or_attr_get(liquidity, "market_type")

        return normalize_market_type(value or DEFAULT_MARKET_TYPE)

    def _context_timestamp(self, context: Any) -> datetime:
        timestamp = getattr(context, "timestamp", None)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)

        return datetime.now(timezone.utc)

    def _snapshot_is_stale(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
    ) -> bool:
        snapshot_getter = getattr(context, "get_feature_snapshot", None)
        if callable(snapshot_getter):
            try:
                feature = snapshot_getter(self.SNAPSHOT_FEATURE_NAME)
            except Exception:
                feature = None

            if feature is not None and hasattr(feature, "is_stale"):
                try:
                    return bool(feature.is_stale(self._context_timestamp(context)))
                except Exception:
                    self.log_debug(
                        "Feature freshness check failed, falling back to TTL",
                        symbol=snapshot.symbol,
                        timeframe=snapshot.timeframe,
                        strategy_name=self.strategy_name,
                    )

        ttl = self._resolve_snapshot_ttl_seconds()
        if ttl is None or ttl <= 0:
            return False

        age_seconds = abs(
            (self._context_timestamp(context) - snapshot.timestamp).total_seconds()
        )
        return age_seconds > ttl

    def _resolve_snapshot_ttl_seconds(self) -> float | None:
        freshness = getattr(self.config, "freshness", None)
        if freshness is None:
            return None

        try:
            ttl = freshness.get_ttl(self.SNAPSHOT_FEATURE_NAME)
        except Exception:
            return None

        return self._safe_float(ttl)

    # ------------------------------------------------------------------
    # Runtime allow-lists
    # ------------------------------------------------------------------

    def _is_exchange_allowed(self, exchange: str) -> bool:
        runtime = self._runtime
        allowed = set(getattr(runtime, "exchanges", set()) or set())
        allowed = {normalize_exchange(item) for item in allowed}
        return not allowed or normalize_exchange(exchange) in allowed

    def _is_market_type_allowed(self, market_type: str) -> bool:
        runtime = self._runtime
        allowed = set(getattr(runtime, "market_types", set()) or set())
        allowed = {normalize_market_type(item) for item in allowed}
        return not allowed or normalize_market_type(market_type) in allowed

    def _is_symbol_allowed(self, symbol: str) -> bool:
        runtime = self._runtime
        allowed = set(getattr(runtime, "symbols", set()) or set())
        allowed = {str(item).strip().upper() for item in allowed if str(item).strip()}
        return not allowed or normalize_symbol(symbol) in allowed

    def _is_timeframe_allowed(self, timeframe: Any) -> bool:
        runtime = self._runtime
        allowed_timeframes = getattr(runtime, "timeframes", None)

        if not allowed_timeframes:
            return True

        current = normalize_timeframe(self._value(timeframe))
        allowed = {normalize_timeframe(self._value(item)) for item in allowed_timeframes}

        return current in allowed

    def _is_regime_allowed(self, context: Any) -> bool:
        runtime = self._runtime
        allowed_regimes = getattr(runtime, "allowed_regimes", None)

        if not allowed_regimes:
            return True

        regime_context = getattr(context, "regime", None)
        regime = self._mapping_or_attr_get(regime_context, "regime")
        if regime is None:
            return True

        current = self._value(regime)
        allowed = {self._value(item) for item in allowed_regimes}

        return current in allowed

    def _is_futures_market_type(self, market_type: Any) -> bool:
        return normalize_market_type(market_type) in self.FUTURES_MARKET_TYPES

    # ------------------------------------------------------------------
    # Current price / analytics metrics
    # ------------------------------------------------------------------

    def _resolve_current_price(
        self,
        context: Any,
        snapshot: LiquidityMapSnapshot,
    ) -> float | None:
        price_context = getattr(context, "price", None)

        candidates = (
            self._mapping_or_attr_get(price_context, "mid_price"),
            self._mapping_or_attr_get(price_context, "last_price"),
            self._mapping_or_attr_get(price_context, "mark_price"),
            self._mapping_or_attr_get(price_context, "index_price"),
            getattr(snapshot, "current_price", None),
        )

        for raw_price in candidates:
            price = self._safe_positive_float(raw_price)
            if price is not None:
                return price

        return None

    def _magnet_score_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="magnet_score_up",
            metadata_key="magnet_score_up",
        )

    def _magnet_score_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="magnet_score_down",
            metadata_key="magnet_score_down",
        )

    def _sweep_risk_up(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="sweep_risk_up",
            metadata_key="sweep_risk_up",
        )

    def _sweep_risk_down(self, snapshot: LiquidityMapSnapshot) -> float:
        return self._snapshot_metric(
            snapshot=snapshot,
            signal_attr="sweep_risk_down",
            metadata_key="sweep_risk_down",
        )

    def _snapshot_metric(
        self,
        snapshot: LiquidityMapSnapshot,
        signal_attr: str,
        metadata_key: str,
    ) -> float:
        if snapshot.signal is not None:
            value = getattr(snapshot.signal, signal_attr, None)
            resolved = self._safe_float(value)
            if resolved is not None:
                return self._clamp01(resolved)

        metadata = getattr(snapshot, "metadata", None) or {}
        resolved = self._safe_float(metadata.get(metadata_key))

        return self._clamp01(resolved or 0.0)

    # ------------------------------------------------------------------
    # Liquidity target helpers for subclasses
    # ------------------------------------------------------------------

    def _reference_price(self, item: Any) -> float:
        if item is None:
            return 0.0

        for attr in ("price", "center_price", "mid_price"):
            value = getattr(item, attr, None)
            price = self._safe_positive_float(value)
            if price is not None:
                return price

        low = self._safe_positive_float(getattr(item, "low_price", None))
        high = self._safe_positive_float(getattr(item, "high_price", None))
        if low is not None and high is not None:
            return (low + high) / 2.0

        return 0.0

    def _distance_pct(self, price_a: float, price_b: float) -> float:
        price_a = self._safe_float(price_a, 0.0)
        price_b = self._safe_float(price_b, 0.0)

        if price_b == 0:
            return 0.0

        return abs(price_a - price_b) / abs(price_b)

    def _nearest_directional_liquidity(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> LiquidityLevel | StopCluster | None:
        getter = getattr(snapshot, "get_nearest_directional_liquidity", None)
        if callable(getter):
            try:
                return getter(side)
            except Exception:
                pass

        if side == LiquiditySide.BUY_SIDE:
            return snapshot.nearest_above_level

        if side == LiquiditySide.SELL_SIDE:
            return snapshot.nearest_below_level

        return None

    def _strongest_directional_cluster(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> StopCluster | None:
        getter = getattr(snapshot, "get_strongest_directional_cluster", None)
        if callable(getter):
            try:
                return getter(side)
            except Exception:
                pass

        if side == LiquiditySide.BUY_SIDE:
            return snapshot.strongest_cluster_above

        if side == LiquiditySide.SELL_SIDE:
            return snapshot.strongest_cluster_below

        return None

    def _directional_levels(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        *,
        include_terminal: bool = False,
    ) -> list[LiquidityLevel]:
        levels = [
            level
            for level in snapshot.active_levels
            if level.side == side
        ]

        if include_terminal:
            terminal = [
                level
                for level in [*snapshot.active_levels, *snapshot.equal_levels]
                if level.side == side and level not in levels
            ]
            levels.extend(terminal)

        return levels

    def _directional_clusters(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        *,
        include_swept: bool = False,
    ) -> list[StopCluster]:
        clusters = [
            cluster
            for cluster in snapshot.stop_clusters
            if cluster.side == side
        ]

        if not include_swept:
            clusters = [
                cluster
                for cluster in clusters
                if not self._cluster_is_swept(cluster)
            ]

        return clusters

    def _directional_zones(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
    ) -> list[LiquidityZone]:
        return [
            zone
            for zone in snapshot.zones
            if zone.side in {side, LiquiditySide.BOTH}
        ]

    def _best_zone_for_side(
        self,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> LiquidityZone | None:
        zones = [
            zone
            for zone in self._directional_zones(snapshot, side)
            if (
                side == LiquiditySide.BUY_SIDE
                and zone.center_price > current_price
            )
            or (
                side == LiquiditySide.SELL_SIDE
                and zone.center_price < current_price
            )
        ]

        if not zones:
            return None

        return max(zones, key=lambda zone: self._clamp01(getattr(zone, "score", 0.0)))

    def _collect_targets_above(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        *,
        include_swept: bool = False,
    ) -> list[LiquidityLevel | StopCluster]:
        candidates: list[LiquidityLevel | StopCluster] = []

        for item in (
            snapshot.nearest_above_level,
            snapshot.strongest_cluster_above,
            *snapshot.active_levels,
            *snapshot.stop_clusters,
        ):
            if item is None:
                continue

            ref_price = self._reference_price(item)
            if ref_price <= current_price:
                continue

            if not include_swept and self._liquidity_item_is_terminal_or_swept(item):
                continue

            candidates.append(item)

        return sorted(
            self._dedupe_liquidity_items(candidates),
            key=self._reference_price,
        )

    def _collect_targets_below(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        *,
        include_swept: bool = False,
    ) -> list[LiquidityLevel | StopCluster]:
        candidates: list[LiquidityLevel | StopCluster] = []

        for item in (
            snapshot.nearest_below_level,
            snapshot.strongest_cluster_below,
            *snapshot.active_levels,
            *snapshot.stop_clusters,
        ):
            if item is None:
                continue

            ref_price = self._reference_price(item)
            if ref_price >= current_price:
                continue

            if not include_swept and self._liquidity_item_is_terminal_or_swept(item):
                continue

            candidates.append(item)

        return sorted(
            self._dedupe_liquidity_items(candidates),
            key=self._reference_price,
            reverse=True,
        )

    def _dedupe_liquidity_items(
        self,
        items: Sequence[LiquidityLevel | StopCluster],
    ) -> list[LiquidityLevel | StopCluster]:
        seen: set[str] = set()
        result: list[LiquidityLevel | StopCluster] = []

        for item in items:
            key = getattr(item, "key", None)
            if key is None:
                key = (
                    f"{getattr(item, 'scope_key', '')}:"
                    f"{item.__class__.__name__}:"
                    f"{self._reference_price(item):.12f}"
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _liquidity_item_is_terminal_or_swept(
        self,
        item: LiquidityLevel | StopCluster,
    ) -> bool:
        if isinstance(item, LiquidityLevel):
            return item.is_terminal() or item.is_swept()

        if isinstance(item, StopCluster):
            return self._cluster_is_swept(item)

        return False

    def _cluster_is_swept(self, cluster: StopCluster) -> bool:
        checker = getattr(cluster, "is_swept", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return getattr(cluster, "swept_at", None) is not None

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _build_liquidity_signal_metadata(
        self,
        snapshot: LiquidityMapSnapshot,
        current_price: float | None = None,
        *,
        target: LiquidityLevel | StopCluster | None = None,
        evidence: LiquidityLevel | StopCluster | None = None,
        setup_name: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "setup_name": setup_name,
            "exchange": snapshot.exchange,
            "market_type": snapshot.market_type,
            "symbol": snapshot.symbol,
            "timeframe": snapshot.timeframe,
            "scope": snapshot.scope,
            "scope_key": snapshot.scope_key,
            "liquidity_key": snapshot.liquidity_key,
            "snapshot_timestamp": snapshot.timestamp.isoformat(),
            "current_price": current_price if current_price is not None else snapshot.current_price,
            "bias": self._value(snapshot.bias),
            "above_liquidity_score": self._clamp01(snapshot.above_liquidity_score),
            "below_liquidity_score": self._clamp01(snapshot.below_liquidity_score),
            "liquidity_pressure_score": self._clamp_signed(
                snapshot.liquidity_pressure_score
            ),
            "magnet_score_up": self._magnet_score_up(snapshot),
            "magnet_score_down": self._magnet_score_down(snapshot),
            "sweep_risk_up": self._sweep_risk_up(snapshot),
            "sweep_risk_down": self._sweep_risk_down(snapshot),
            "active_levels_count": len(snapshot.active_levels),
            "equal_levels_count": len(snapshot.equal_levels),
            "stop_clusters_count": len(snapshot.stop_clusters),
            "zones_count": len(snapshot.zones),
            "nearest_above": self._to_payload(snapshot.nearest_above_level),
            "nearest_below": self._to_payload(snapshot.nearest_below_level),
            "strongest_cluster_above": self._to_payload(snapshot.strongest_cluster_above),
            "strongest_cluster_below": self._to_payload(snapshot.strongest_cluster_below),
            "target": self._to_payload(target),
            "evidence": self._to_payload(evidence),
        }

        if snapshot.signal is not None:
            metadata["liquidity_signal"] = self._to_payload(snapshot.signal)

        if extra:
            metadata.update(dict(extra))

        return metadata

    # ------------------------------------------------------------------
    # Signal/context validation
    # ------------------------------------------------------------------

    def _validate_signal_context_pair(
        self,
        signal: StrategySignal,
        context: Any,
    ) -> None:
        context_scope = self._context_scope(context)

        if signal.symbol != context_scope["symbol"]:
            raise ValueError(
                f"Signal/context symbol mismatch: "
                f"signal={signal.symbol}, context={context_scope['symbol']}"
            )

        if self._value(signal.timeframe) != context_scope["timeframe"]:
            raise ValueError(
                f"Signal/context timeframe mismatch: "
                f"signal={self._value(signal.timeframe)}, "
                f"context={context_scope['timeframe']}"
            )

        if signal.strategy_name != self.strategy_name:
            raise ValueError(
                f"Signal strategy mismatch: "
                f"signal={signal.strategy_name}, strategy={self.strategy_name}"
            )

        snapshot = self._extract_snapshot(context)
        if snapshot is not None and not self._snapshot_matches_context(context, snapshot):
            raise ValueError(
                f"Signal/context liquidity scope mismatch: "
                f"snapshot={snapshot.scope}, context={context_scope}"
            )

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------

    def _is_on_emit_cooldown(
        self,
        symbol: str,
        timeframe: Any,
        timestamp: datetime,
    ) -> bool:
        runtime = self._runtime
        cooldown_seconds = float(getattr(runtime, "emit_cooldown_seconds", 0) or 0)

        if cooldown_seconds <= 0:
            return False

        last_emitted_at = self._last_emitted_at.get(
            self._cooldown_key(symbol, timeframe)
        )
        if last_emitted_at is None:
            return False

        return timestamp - last_emitted_at < timedelta(seconds=cooldown_seconds)

    def _cooldown_key(
        self,
        symbol: str,
        timeframe: Any,
        *,
        context: Any | None = None,
    ) -> str:
        exchange = DEFAULT_EXCHANGE
        market_type = DEFAULT_MARKET_TYPE

        if context is not None:
            scope = self._context_scope(context)
            exchange = scope["exchange"]
            market_type = scope["market_type"]

        return (
            f"{self.strategy_name}:"
            f"{normalize_exchange(exchange)}:"
            f"{normalize_market_type(market_type)}:"
            f"{normalize_symbol(symbol)}:"
            f"{normalize_timeframe(self._value(timeframe))}"
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _resolve_regime(self, context: Any) -> Any | None:
        regime_context = getattr(context, "regime", None)
        if regime_context is None:
            return None
        return self._mapping_or_attr_get(regime_context, "regime") or regime_context

    def _mapping_or_attr_get(
        self,
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if obj is None:
            return default

        if isinstance(obj, Mapping):
            return obj.get(key, default)

        return getattr(obj, key, default)

    def _safe_float(
        self,
        value: Any,
        default: float | None = None,
    ) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default

        if not isfinite(result):
            return default

        return result

    def _safe_positive_float(self, value: Any) -> float | None:
        result = self._safe_float(value)
        if result is None or result <= 0:
            return None
        return result

    def _clamp01(self, value: Any) -> float:
        resolved = self._safe_float(value, 0.0)
        if resolved is None:
            return 0.0
        return max(0.0, min(resolved, 1.0))

    def _clamp_signed(self, value: Any) -> float:
        resolved = self._safe_float(value, 0.0)
        if resolved is None:
            return 0.0
        return max(-1.0, min(resolved, 1.0))

    def _value(self, value: Any) -> Any:
        return getattr(value, "value", value)

    def _to_payload(self, value: Any) -> Any:
        if value is None:
            return None

        to_event_payload = getattr(value, "to_event_payload", None)
        if callable(to_event_payload):
            try:
                return to_event_payload()
            except Exception:
                pass

        to_payload = getattr(value, "to_payload", None)
        if callable(to_payload):
            try:
                return to_payload()
            except Exception:
                pass

        if isinstance(value, Mapping):
            return {
                str(key): self._to_payload(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._to_payload(item) for item in value]

        if hasattr(value, "__dict__"):
            return {
                key: self._to_payload(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }

        return self._value(value)