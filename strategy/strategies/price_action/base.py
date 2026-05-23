# trading_system/strategy/strategies/price_action/base.py

from __future__ import annotations
import logging

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    EntryType,
    ExitType,
    FeatureSource,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
    TriggerType,
)
from ...exceptions import StrategyConfigError, StrategyEvaluationError
from ...models import FeatureSnapshot, StrategyContext, StrategySignal, clamp
from ..base_strategy import TradingStrategy
from .utils import (
    as_dict,
    extract_last_event,
    extract_last_update,
    extract_price_action_module,
    extract_price_action_state,
    extract_scope,
    first_non_empty,
    get_path,
    layer_confidence,
    layer_strength,
    parse_datetime,
    price_action_domain,
    price_action_item,
    price_action_path,
    scope_matches_context,
    select_primary_layer,
    select_secondary_layer,
    serialize_for_metadata,
    to_bool,
    to_float,
    to_int,
    to_str,
    unit_score,
)


# =============================================================================
# Feature contract
# =============================================================================


@dataclass(frozen=True, slots=True)
class PriceActionFeatureNames:
    """
    Stable feature names expected in StrategyContext.

    StrategyContextBuilder / SignalNormalizer should populate these from
    analytics.price_action.* payloads. Concrete strategies may also read
    equivalent values from FeatureSource.PRICE_ACTION domain_data aliases.
    """
    _logger = logging.getLogger(__name__ + ".PriceActionFeatureNames")

    COMPOSITE: str = "price_action.composite"

    MARKET_STRUCTURE: str = "price_action.market_structure"
    MARKET_STRUCTURE_INTERNAL: str = "price_action.market_structure.internal"
    MARKET_STRUCTURE_EXTERNAL: str = "price_action.market_structure.external"
    MARKET_STRUCTURE_LAST_BREAK_EVENT: str = (
        "price_action.market_structure.last_break_event"
    )
    MARKET_STRUCTURE_MTF_ALIGNMENT: str = (
        "price_action.market_structure.mtf_alignment"
    )

    SUPPORT_RESISTANCE: str = "price_action.support_resistance"
    SUPPORT_RESISTANCE_INTERNAL: str = "price_action.support_resistance.internal"
    SUPPORT_RESISTANCE_EXTERNAL: str = "price_action.support_resistance.external"
    SUPPORT_RESISTANCE_LAST_EVENT: str = (
        "price_action.support_resistance.last_event"
    )
    SUPPORT_RESISTANCE_NEAREST_SUPPORT: str = (
        "price_action.support_resistance.nearest_support"
    )
    SUPPORT_RESISTANCE_NEAREST_RESISTANCE: str = (
        "price_action.support_resistance.nearest_resistance"
    )

    FAIR_VALUE_GAP: str = "price_action.fair_value_gap"
    FVG: str = "price_action.fvg"
    FVG_INTERNAL: str = "price_action.fair_value_gap.internal"
    FVG_EXTERNAL: str = "price_action.fair_value_gap.external"
    FVG_LAST_EVENT: str = "price_action.fair_value_gap.last_event"
    FVG_NEAREST_BULLISH_GAP: str = (
        "price_action.fair_value_gap.nearest_bullish_gap"
    )
    FVG_NEAREST_BEARISH_GAP: str = (
        "price_action.fair_value_gap.nearest_bearish_gap"
    )

    TREND: str = "price_action.trend"
    TREND_INTERNAL: str = "price_action.trend.internal"
    TREND_EXTERNAL: str = "price_action.trend.external"
    TREND_LAST_SIGNAL: str = "price_action.trend.last_signal"
    TREND_INTERNAL_EXTERNAL_ALIGNMENT: str = (
        "price_action.trend.internal_external_alignment"
    )
    TREND_HIGHER_TIMEFRAME_ALIGNMENT: str = (
        "price_action.trend.higher_timeframe_alignment"
    )
    TREND_OVERALL_SCORE: str = "price_action.trend.overall_trend_score"

    LIQUIDITY_LEVELS: str = "price_action.liquidity_levels"

    LAST_PRICE: str = "price_action.last_price"
    CURRENT_PRICE: str = "price_action.current_price"
    TIMESTAMP: str = "price_action.timestamp"

    @classmethod
    def all(cls) -> set[str]:
        _strategy_logger = getattr(cls, "_logger", None) or logging.getLogger(__name__ + ".PriceActionFeatureNames")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionFeatureNames.all")
        instance = cls()
        return {
            getattr(instance, item.name)
            for item in fields(cls)
            if isinstance(getattr(instance, item.name), str)
            and getattr(instance, item.name).strip()
        }


PRICE_ACTION_FEATURES = PriceActionFeatureNames()


# =============================================================================
# Scope
# =============================================================================


@dataclass(frozen=True, slots=True)
class PriceActionStrategyScope:
    """
    Futures price-action scope used only for metadata and normalization.

    Concrete strategies still make decisions from StrategyContext.
    """
    _logger = logging.getLogger(__name__ + ".PriceActionStrategyScope")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str | None = None

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionStrategyScope.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "PriceActionStrategyScope.symbol cannot be empty"
            )

        object.__setattr__(self, "exchange", exchange)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "exchange_symbol", exchange_symbol)

    @property
    def key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionStrategyScope.key")
        return f"{self.exchange}:{self.market_type}:{self.symbol}:{self.timeframe}"

    @property
    def legacy_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionStrategyScope.legacy_key")
        return f"{self.symbol}:{self.exchange}"

    def to_dict(self) -> dict[str, str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionStrategyScope.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol or self.symbol,
            "key": self.key,
            "legacy_key": self.legacy_key,
        }


# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class PriceActionStrategyConfig:
    """
    Domain config shared by concrete price-action strategies.

    Runtime enabled/symbol/timeframe/regime checks belong to StrategyConfig /
    StrategyDefinitionConfig. This config keeps price-action-specific defaults
    and quality thresholds.
    """
    _logger = logging.getLogger(__name__ + ".PriceActionStrategyConfig")

    default_market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES
    default_margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED
    default_order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN
    default_trade_tier: StrategyTradeTier = StrategyTradeTier.T2

    default_entry_type: EntryType = EntryType.MARKET
    default_exit_types: tuple[ExitType, ...] = (
        ExitType.TAKE_PROFIT,
        ExitType.STOP_LOSS,
        ExitType.INVALIDATION,
    )

    prefer_external_layer: bool = True

    min_context_confidence: float = 0.0
    min_signal_confidence: float = 0.50
    min_signal_score: float = 0.35

    stale_feature_max_age_seconds: float | None = None

    requested_leverage: float | None = None
    max_slippage_bps: float | None = None
    entry_timeout_seconds: int | None = None
    max_holding_seconds: int | None = None

    attach_price_action_context_metadata: bool = True
    attach_scope_metadata: bool = True
    attach_feature_values_metadata: bool = True

    tag_price_action: str = "price_action"
    tag_market_structure: str = "market_structure"
    tag_support_resistance: str = "support_resistance"
    tag_fvg: str = "fvg"
    tag_trend: str = "trend"
    tag_liquidity: str = "liquidity"
    tag_reversal: str = "reversal"
    tag_continuation: str = "continuation"
    tag_breakout: str = "breakout"
    tag_retest: str = "retest"
    tag_reaction: str = "reaction"

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionStrategyConfig.validate")
        bounded = {
            "min_context_confidence": self.min_context_confidence,
            "min_signal_confidence": self.min_signal_confidence,
            "min_signal_score": self.min_signal_score,
        }

        for name, value in bounded.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{name} must be between 0.0 and 1.0")

        if (
            self.stale_feature_max_age_seconds is not None
            and self.stale_feature_max_age_seconds <= 0
        ):
            raise StrategyConfigError("stale_feature_max_age_seconds must be > 0")

        if self.requested_leverage is not None and self.requested_leverage <= 0:
            raise StrategyConfigError("requested_leverage must be > 0")

        if self.max_slippage_bps is not None and self.max_slippage_bps < 0:
            raise StrategyConfigError("max_slippage_bps must be >= 0")

        if self.entry_timeout_seconds is not None and self.entry_timeout_seconds <= 0:
            raise StrategyConfigError("entry_timeout_seconds must be > 0")

        if self.max_holding_seconds is not None and self.max_holding_seconds <= 0:
            raise StrategyConfigError("max_holding_seconds must be > 0")

        for attr in (
            "tag_price_action",
            "tag_market_structure",
            "tag_support_resistance",
            "tag_fvg",
            "tag_trend",
            "tag_liquidity",
            "tag_reversal",
            "tag_continuation",
            "tag_breakout",
            "tag_retest",
            "tag_reaction",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")


# =============================================================================
# Composite snapshot
# =============================================================================


@dataclass(slots=True)
class PriceActionCompositeSnapshot:
    """
    Strategy-level normalized view over analytics.price_action state.

    This is intentionally not an analytics model. It is a strategy-side
    projection that gives concrete price-action strategies one stable contract.
    """
    _logger = logging.getLogger(__name__ + ".PriceActionCompositeSnapshot")

    exchange: str
    market_type: str
    symbol: str
    timeframe: str

    exchange_symbol: str | None = None
    timestamp: datetime | None = None
    source: str = "unknown"

    current_price: float | None = None
    last_price: float | None = None

    market_structure: dict[str, Any] = field(default_factory=dict)
    support_resistance: dict[str, Any] = field(default_factory=dict)
    fair_value_gap: dict[str, Any] = field(default_factory=dict)
    trend: dict[str, Any] = field(default_factory=dict)
    liquidity_levels: dict[str, Any] = field(default_factory=dict)

    raw_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.__post_init__")
        exchange = str(self.exchange or "unknown").strip().lower()
        market_type = str(
            self.market_type or StrategyMarketType.USDM_FUTURES.value
        ).strip()
        symbol = str(self.symbol or "").strip().upper()
        timeframe = str(self.timeframe or Timeframe.M1.value).strip().lower()
        exchange_symbol = str(self.exchange_symbol or symbol).strip().upper()

        if not symbol:
            raise StrategyEvaluationError(
                "PriceActionCompositeSnapshot.symbol cannot be empty"
            )

        self.exchange = exchange
        self.market_type = market_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_symbol = exchange_symbol

        self.timestamp = parse_datetime(self.timestamp)

        self.current_price = to_float(self.current_price)
        self.last_price = to_float(self.last_price)

        self.market_structure = as_dict(self.market_structure)
        self.support_resistance = as_dict(self.support_resistance)
        self.fair_value_gap = as_dict(self.fair_value_gap)
        self.trend = as_dict(self.trend)
        self.liquidity_levels = as_dict(self.liquidity_levels)
        self.raw_state = as_dict(self.raw_state)
        self.metadata = as_dict(self.metadata)

    @property
    def scope(self) -> PriceActionStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.scope")
        return PriceActionStrategyScope(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            exchange_symbol=self.exchange_symbol,
        )

    @property
    def scope_key(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.scope_key")
        return self.scope.key

    def module(self, name: str) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.module")
        normalized = str(name or "").strip().lower()

        if normalized in {"market_structure", "structure"}:
            return self.market_structure

        if normalized in {"support_resistance", "sr"}:
            return self.support_resistance

        if normalized in {"fair_value_gap", "fvg"}:
            return self.fair_value_gap

        if normalized == "trend":
            return self.trend

        if normalized in {"liquidity", "liquidity_levels"}:
            return self.liquidity_levels

        return {}

    def has_module(self, name: str) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.has_module")
        return bool(self.module(name))

    def last_update(self) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.last_update")
        candidates = [
            self.timestamp,
            extract_last_update(self.market_structure),
            extract_last_update(self.support_resistance),
            extract_last_update(self.fair_value_gap),
            extract_last_update(self.trend),
            extract_last_update(self.liquidity_levels),
            extract_last_update(self.raw_state),
        ]
        return next((item for item in candidates if item is not None), None)

    def to_dict(self) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionCompositeSnapshot.to_dict")
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_symbol": self.exchange_symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "current_price": self.current_price,
            "last_price": self.last_price,
            "market_structure": serialize_for_metadata(self.market_structure),
            "support_resistance": serialize_for_metadata(self.support_resistance),
            "fair_value_gap": serialize_for_metadata(self.fair_value_gap),
            "trend": serialize_for_metadata(self.trend),
            "liquidity_levels": serialize_for_metadata(self.liquidity_levels),
            "raw_state": serialize_for_metadata(self.raw_state),
            "metadata": serialize_for_metadata(self.metadata),
            "scope": self.scope.to_dict(),
            "scope_key": self.scope_key,
        }


# =============================================================================
# Base price-action strategy
# =============================================================================


class PriceActionTradingStrategy(TradingStrategy):
    """
    Base class for concrete strategy/strategies/price_action/* classes.

    Responsibilities:
    - read price-action analytics data from StrategyContext only;
    - provide helper methods for module/layer extraction and scoring;
    - build internal StrategySignal objects through TradingStrategy helpers;
    - attach futures/price-action metadata for SignalProcessor.

    Forbidden:
    - no direct analytics.price_action.* EventBus subscriptions;
    - no SignalContext;
    - no manual StrategyEvaluation construction;
    - no EventBus emit of signal.generated;
    - no RiskManager / Execution calls;
    - no raw market data reads.
    """
    _logger = logging.getLogger(__name__ + ".PriceActionTradingStrategy")

    component_namespace = "strategy.price_action"
    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    feature_names = PRICE_ACTION_FEATURES

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        price_action_config: PriceActionStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.__init__")
        self.price_action_config = price_action_config or PriceActionStrategyConfig()
        self.price_action_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            service_name=service_name,
        )

    def validate_config(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.validate_config")
        super().validate_config()
        self.price_action_config.validate()

    # ------------------------------------------------------------------
    # No-signal diagnostics
    # ------------------------------------------------------------------

    def remember_no_signal(self, reason: str, **metadata: Any) -> None:
        """
        Store the exact reason why generate_signal() returned None.

        BaseStrategy.evaluate() should consume these fields when building a
        failed StrategyEvaluation. Keeping the helper here also makes
        price-action strategies safe in deployments where the root BaseStrategy
        patch has not been applied yet.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.remember_no_signal")
        normalized = str(reason or "").strip() or "no_signal_generated"
        self._last_no_signal_reason = normalized
        self._last_no_signal_metadata = dict(metadata)

    def clear_no_signal_reason(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.clear_no_signal_reason")
        self._last_no_signal_reason = None
        self._last_no_signal_metadata = {}

    def consume_no_signal_reason(self) -> tuple[list[str], dict[str, Any]]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.consume_no_signal_reason")
        reason = getattr(self, "_last_no_signal_reason", None) or "no_signal_generated"
        metadata = dict(getattr(self, "_last_no_signal_metadata", {}) or {})
        self.clear_no_signal_reason()
        return [reason], metadata

    # ------------------------------------------------------------------
    # Context / domain access
    # ------------------------------------------------------------------

    def price_action_domain(self, context: StrategyContext) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_domain")
        self.validate_context(context)
        return price_action_domain(context)

    def price_action_item(
        self,
        context: StrategyContext,
        key: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_item")
        self.validate_context(context)
        return price_action_item(context, key, default)

    def price_action_path(
        self,
        context: StrategyContext,
        path: str,
        default: Any = None,
    ) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_path")
        self.validate_context(context)

        if not isinstance(path, str) or not path.strip():
            raise StrategyEvaluationError("price_action path cannot be empty")

        return price_action_path(context, path, default)

    def price_action_float(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float | None = None,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_float")
        return to_float(self.price_action_path(context, path, default), default)

    def price_action_int(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: int | None = None,
    ) -> int | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_int")
        return to_int(self.price_action_path(context, path, default), default)

    def price_action_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_score")
        return unit_score(self.price_action_path(context, path, default), default)

    def price_action_signed_score(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: float = 0.0,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_signed_score")
        value = self.price_action_float(context, path, default=default)
        return clamp(float(value if value is not None else default), -1.0, 1.0)

    def price_action_bool(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: bool = False,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_bool")
        return to_bool(self.price_action_path(context, path, default), default)

    def price_action_str(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: str | None = None,
    ) -> str | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_str")
        return to_str(self.price_action_path(context, path, default), default)

    def price_action_datetime(
        self,
        context: StrategyContext,
        path: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_datetime")
        return parse_datetime(self.price_action_path(context, path, default))

    def price_action_feature_snapshot(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> FeatureSnapshot | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_feature_snapshot")
        self.validate_context(context)

        if not isinstance(feature_name, str) or not feature_name.strip():
            raise StrategyEvaluationError("feature_name cannot be empty")

        features_map = getattr(context, "features", None)
        if isinstance(features_map, Mapping):
            raw = features_map.get(feature_name)
            if isinstance(raw, FeatureSnapshot):
                return raw

        return None

    def price_action_feature_age_seconds(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_feature_age_seconds")
        snapshot = self.price_action_feature_snapshot(context, feature_name)
        if snapshot is None:
            return None
        return snapshot.age_seconds(context.timestamp)

    def price_action_feature_is_stale(
        self,
        context: StrategyContext,
        feature_name: str,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_feature_is_stale")
        max_age = self.price_action_config.stale_feature_max_age_seconds
        if max_age is None:
            return False

        age = self.price_action_feature_age_seconds(context, feature_name)
        if age is None:
            return False

        return age > max_age

    def has_any_price_action_data(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] = (),
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.has_any_price_action_data")
        self.validate_context(context)

        if self.price_action_domain(context):
            return True

        return any(context.has_feature(name) for name in feature_names)

    def has_stale_price_action_features(
        self,
        context: StrategyContext,
        feature_names: tuple[str, ...] | None = None,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.has_stale_price_action_features")
        names = feature_names or tuple(self.required_features())

        return any(
            self.price_action_feature_is_stale(context, feature_name)
            for feature_name in names
        )

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def price_action_scope(self, context: StrategyContext) -> PriceActionStrategyScope:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_scope")
        domain = self.price_action_domain(context)

        exchange = (
            to_str(domain.get("exchange"))
            or to_str(get_path(domain, "scope.exchange"))
            or to_str(context.metadata.get("exchange"))
            or "unknown"
        )
        market_type = (
            to_str(domain.get("market_type"))
            or to_str(get_path(domain, "scope.market_type"))
            or to_str(context.metadata.get("market_type"))
            or self.price_action_config.default_market_type.value
        )
        exchange_symbol = (
            to_str(domain.get("exchange_symbol"))
            or to_str(get_path(domain, "scope.exchange_symbol"))
            or to_str(context.metadata.get("exchange_symbol"))
            or context.symbol
        )

        timeframe_value = getattr(context.timeframe, "value", context.timeframe)

        return PriceActionStrategyScope(
            exchange=exchange,
            market_type=market_type,
            symbol=context.symbol,
            timeframe=str(timeframe_value or Timeframe.M1.value),
            exchange_symbol=exchange_symbol,
        )

    # ------------------------------------------------------------------
    # Snapshot / module resolution
    # ------------------------------------------------------------------

    def resolve_price_action_snapshot(
        self,
        context: StrategyContext,
    ) -> PriceActionCompositeSnapshot | None:
        """
        Resolve normalized price-action composite snapshot from StrategyContext.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.resolve_price_action_snapshot")
        self.validate_context(context)

        scope = self.price_action_scope(context)
        state = extract_price_action_state(context)

        if not state:
            state = self._build_state_from_features(context)

        if not state:
            return None

        if not scope_matches_context(context, state):
            return None

        return self._snapshot_from_state(
            state,
            context=context,
            scope=scope,
            source=to_str(state.get("_source_feature"), "context.domain") or "context.domain",
        )

    def resolve_price_action_module(
        self,
        context: StrategyContext,
        module_name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """
        Resolve one price-action module from StrategyContext.

        Examples:
            market_structure
            support_resistance
            fair_value_gap / fvg
            trend
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.resolve_price_action_module")
        self.validate_context(context)

        snapshot = self.resolve_price_action_snapshot(context)
        if snapshot is not None:
            module = snapshot.module(module_name)
            if module:
                return module

        return extract_price_action_module(
            context,
            module_name,
            aliases=aliases,
        )

    def select_primary_layer(
        self,
        module_payload: Any,
        *,
        prefer_external_layer: bool | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.select_primary_layer")
        return select_primary_layer(
            module_payload,
            prefer_external_layer=(
                self.price_action_config.prefer_external_layer
                if prefer_external_layer is None
                else prefer_external_layer
            ),
        )

    def select_secondary_layer(
        self,
        module_payload: Any,
        *,
        prefer_external_layer: bool | None = None,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.select_secondary_layer")
        return select_secondary_layer(
            module_payload,
            prefer_external_layer=(
                self.price_action_config.prefer_external_layer
                if prefer_external_layer is None
                else prefer_external_layer
            ),
        )

    def layer_confidence(self, layer: Any) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.layer_confidence")
        return layer_confidence(layer)

    def layer_strength(self, layer: Any) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.layer_strength")
        return layer_strength(layer)

    def last_module_event(self, module_payload: Any) -> dict[str, Any] | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.last_module_event")
        return extract_last_event(module_payload)

    def module_last_update(self, module_payload: Any) -> datetime | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.module_last_update")
        return extract_last_update(module_payload)

    # ------------------------------------------------------------------
    # Signal builder
    # ------------------------------------------------------------------

    def build_price_action_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        priority: SignalPriority = SignalPriority.MEDIUM,
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        """
        Build internal StrategySignal with price-action/futures metadata.

        Final risk-ready payload conversion belongs to SignalProcessor /
        SignalBuilder, not to this domain strategy.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.build_price_action_signal")
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self.strategy_name}: price-action signal side must be LONG or SHORT"
            )

        scope = self.price_action_scope(context)

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("domain", FeatureSource.PRICE_ACTION.value)
        signal_metadata.setdefault("price_action_strategy_version", "2.0.0")
        signal_metadata.setdefault(
            "order_intent",
            self.price_action_config.default_order_intent.value,
        )
        signal_metadata.setdefault(
            "margin_mode",
            self.price_action_config.default_margin_mode.value,
        )
        signal_metadata.setdefault(
            "market_type",
            self.price_action_config.default_market_type.value,
        )
        signal_metadata.setdefault(
            "tier",
            self.price_action_config.default_trade_tier.value,
        )

        if self.price_action_config.requested_leverage is not None:
            signal_metadata.setdefault(
                "requested_leverage",
                float(self.price_action_config.requested_leverage),
            )

        if self.price_action_config.max_slippage_bps is not None:
            signal_metadata.setdefault(
                "max_slippage_bps",
                float(self.price_action_config.max_slippage_bps),
            )

        if self.price_action_config.entry_timeout_seconds is not None:
            signal_metadata.setdefault(
                "entry_timeout_seconds",
                int(self.price_action_config.entry_timeout_seconds),
            )

        if self.price_action_config.max_holding_seconds is not None:
            signal_metadata.setdefault(
                "max_holding_seconds",
                int(self.price_action_config.max_holding_seconds),
            )

        if self.price_action_config.attach_scope_metadata:
            signal_metadata.setdefault("scope", scope.to_dict())

        if self.price_action_config.attach_price_action_context_metadata:
            signal_metadata.setdefault(
                "price_action_context",
                self.price_action_context_metadata(context),
            )

        if self.price_action_config.metadata:
            signal_metadata.setdefault(
                "price_action_config_metadata",
                serialize_for_metadata(self.price_action_config.metadata),
            )

        final_reasons = list(
            dict.fromkeys(
                [
                    "price_action_strategy_signal",
                    *(reasons or []),
                ]
            )
        )
        final_confirmations = list(dict.fromkeys(confirmations or []))
        final_features = list(dict.fromkeys(source_features or []))

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=setup_type or self.default_setup_type,
            reasons=final_reasons,
            confirmations=final_confirmations,
            source_features=final_features,
            metadata=serialize_for_metadata(signal_metadata),
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            status=status,
        )

        signal.validate()
        return signal

    def price_action_context_metadata(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        """
        Compact serialized price-action context for StrategySignal.metadata.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy.price_action_context_metadata")
        metadata: dict[str, Any] = {}

        snapshot = self.resolve_price_action_snapshot(context)
        if snapshot is not None:
            metadata["snapshot"] = snapshot.to_dict()

        if self.price_action_config.attach_feature_values_metadata:
            metadata["feature_values"] = {
                "current_price": self.price_action_path(
                    context,
                    "current_price",
                    None,
                ),
                "last_price": self.price_action_path(
                    context,
                    "last_price",
                    None,
                ),
                "market_structure_bias": self.price_action_path(
                    context,
                    "market_structure.external.bias",
                    None,
                ),
                "trend_direction": self.price_action_path(
                    context,
                    "trend.external.direction",
                    None,
                ),
                "trend_strength": self.price_action_path(
                    context,
                    "trend.external.trend_strength",
                    None,
                ),
                "support_resistance_last_event": self.price_action_path(
                    context,
                    "support_resistance.last_event.event_type",
                    None,
                ),
                "fvg_last_event": self.price_action_path(
                    context,
                    "fair_value_gap.last_event.event_type",
                    None,
                ),
            }

        return serialize_for_metadata(metadata)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_state_from_features(
        self,
        context: StrategyContext,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy._build_state_from_features")
        state = {
            "current_price": self._feature_value(
                context,
                PRICE_ACTION_FEATURES.CURRENT_PRICE,
            ),
            "last_price": self._feature_value(
                context,
                PRICE_ACTION_FEATURES.LAST_PRICE,
            ),
            "timestamp": self._feature_value(
                context,
                PRICE_ACTION_FEATURES.TIMESTAMP,
            ),
            "market_structure": {
                "internal": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.MARKET_STRUCTURE_INTERNAL,
                ),
                "external": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.MARKET_STRUCTURE_EXTERNAL,
                ),
                "last_break_event": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.MARKET_STRUCTURE_LAST_BREAK_EVENT,
                ),
                "mtf_alignment": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.MARKET_STRUCTURE_MTF_ALIGNMENT,
                ),
            },
            "support_resistance": {
                "internal": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_INTERNAL,
                ),
                "external": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_EXTERNAL,
                ),
                "last_event": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_LAST_EVENT,
                ),
                "nearest_support": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_NEAREST_SUPPORT,
                ),
                "nearest_resistance": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.SUPPORT_RESISTANCE_NEAREST_RESISTANCE,
                ),
            },
            "fair_value_gap": {
                "internal": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.FVG_INTERNAL,
                ),
                "external": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.FVG_EXTERNAL,
                ),
                "last_event": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.FVG_LAST_EVENT,
                ),
                "nearest_bullish_gap": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.FVG_NEAREST_BULLISH_GAP,
                ),
                "nearest_bearish_gap": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.FVG_NEAREST_BEARISH_GAP,
                ),
            },
            "trend": {
                "internal": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_INTERNAL,
                ),
                "external": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_EXTERNAL,
                ),
                "last_signal": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_LAST_SIGNAL,
                ),
                "internal_external_alignment": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_INTERNAL_EXTERNAL_ALIGNMENT,
                ),
                "higher_timeframe_alignment": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_HIGHER_TIMEFRAME_ALIGNMENT,
                ),
                "overall_trend_score": self._feature_value(
                    context,
                    PRICE_ACTION_FEATURES.TREND_OVERALL_SCORE,
                ),
            },
            "liquidity_levels": self._feature_value(
                context,
                PRICE_ACTION_FEATURES.LIQUIDITY_LEVELS,
            ),
        }

        return state if self._has_any_value(state) else {}

    def _snapshot_from_state(
        self,
        state: Mapping[str, Any],
        *,
        context: StrategyContext,
        scope: PriceActionStrategyScope,
        source: str,
    ) -> PriceActionCompositeSnapshot:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy._snapshot_from_state")
        market_structure = (
            as_dict(get_path(state, "market_structure"))
            or as_dict(get_path(state, "structure"))
        )
        support_resistance = (
            as_dict(get_path(state, "support_resistance"))
            or as_dict(get_path(state, "sr"))
        )
        fair_value_gap = (
            as_dict(get_path(state, "fair_value_gap"))
            or as_dict(get_path(state, "fvg"))
        )
        trend = as_dict(get_path(state, "trend"))
        liquidity_levels = (
            as_dict(get_path(state, "liquidity_levels"))
            or as_dict(get_path(state, "liquidity"))
        )

        timestamp = (
            parse_datetime(first_non_empty(state.get("timestamp"), state.get("time")))
            or extract_last_update(state)
            or context.timestamp
        )

        current_price = first_non_empty(
            state.get("current_price"),
            state.get("last_price"),
            get_path(state, "price.current"),
            get_path(state, "market.current_price"),
            get_path(market_structure, "current_price"),
            get_path(trend, "current_price"),
        )
        last_price = first_non_empty(
            state.get("last_price"),
            state.get("close"),
            get_path(state, "price.last"),
            get_path(market_structure, "last_price"),
            get_path(trend, "last_price"),
            current_price,
        )

        return PriceActionCompositeSnapshot(
            exchange=scope.exchange,
            market_type=scope.market_type,
            symbol=scope.symbol,
            timeframe=scope.timeframe,
            exchange_symbol=scope.exchange_symbol,
            timestamp=timestamp,
            source=source,
            current_price=to_float(current_price),
            last_price=to_float(last_price),
            market_structure=market_structure,
            support_resistance=support_resistance,
            fair_value_gap=fair_value_gap,
            trend=trend,
            liquidity_levels=liquidity_levels,
            raw_state=dict(state),
            metadata={
                "scope": scope.to_dict(),
                "source": source,
                "payload_scope": extract_scope(state),
            },
        )

    @staticmethod
    def _feature_value(context: StrategyContext, feature_name: str) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".PriceActionTradingStrategy._feature_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy._feature_value")
        if not isinstance(feature_name, str) or not feature_name.strip():
            return None

        if not context.has_feature(feature_name):
            return None

        value = context.get_feature(feature_name)

        if isinstance(value, FeatureSnapshot):
            return value.value

        return value

    @staticmethod
    def _has_any_value(value: Any) -> bool:
        _strategy_logger = logging.getLogger(__name__ + ".PriceActionTradingStrategy._has_any_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering PriceActionTradingStrategy._has_any_value")
        if value is None:
            return False

        if isinstance(value, Mapping):
            return any(PriceActionTradingStrategy._has_any_value(item) for item in value.values())

        if isinstance(value, (list, tuple, set)):
            return any(PriceActionTradingStrategy._has_any_value(item) for item in value)

        return value is not None


# Backward-compatible alias while concrete price-action strategies are migrated.
PriceActionStrategyBase = PriceActionTradingStrategy