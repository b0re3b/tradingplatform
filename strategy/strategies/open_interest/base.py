from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analytics.open_interest.enums import OIDivergenceType, OIRegime
from analytics.open_interest.models import OIFeatures
from core.logger import get_logger
from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
from strategy.enums import ConfidenceGrade, MarketRegime, SignalPriority, SignalStrength, StrategyCategory
from strategy.models import SignalContext, StrategySignal


@dataclass(slots=True)
class OIBasePayload:
    """Базовий payload для open-interest стратегій."""

    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class OpenInterestStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Base class для strategy/strategies/open_interest.

    Призначення:
    - уніфікувати інтеграцію з core.logger;
    - прибрати дублювання helper-логіки;
    - централізувати OI feature extraction;
    - централізувати score/confidence helpers;
    - не створювати EventBus напряму, а приймати його ззовні.
    """

    STRATEGY_NAME: str = "open_interest_base_strategy"
    DEFAULT_PRIORITY: int = 80
    REQUIRED_FEATURES: set[str] = set()

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=get_logger(
                __name__,
                service_name="strategy",
                event_type=self.STRATEGY_NAME,
            ),
        )
        self.validate_config()

    @property
    def priority(self) -> int:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return self.DEFAULT_PRIORITY

    # ------------------------------------------------------------------
    # Common validation helpers
    # ------------------------------------------------------------------

    def is_strategy_runtime_allowed(self, context: SignalContext) -> tuple[bool, str | None]:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is None:
            return False, "strategy_config_not_found"

        if not strategy_cfg.runtime.enabled:
            return False, "strategy_disabled"

        if strategy_cfg.runtime.symbols and context.symbol not in strategy_cfg.runtime.symbols:
            return False, "symbol_not_enabled_for_strategy"

        if strategy_cfg.runtime.timeframes and context.timeframe not in strategy_cfg.runtime.timeframes:
            return False, "timeframe_not_enabled_for_strategy"

        return True, None

    def is_market_regime_allowed(self, context: SignalContext) -> tuple[bool, str | None]:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is None:
            return False, "strategy_config_not_found"

        regime = self.get_market_regime(context)
        if strategy_cfg.runtime.allowed_regimes and regime not in strategy_cfg.runtime.allowed_regimes:
            return False, "regime_not_allowed"

        return True, None

    def context_has_any_oi_data(self, context: SignalContext, keys: tuple[str, ...]) -> bool:
        if context.open_interest:
            return True
        return any(context.has_feature(name) for name in keys)

    def has_stale_required_features(self, context: SignalContext) -> bool:
        for name in self.REQUIRED_FEATURES:
            if context.has_feature(name) and context.feature_is_stale(name):
                return True
        return False

    # ------------------------------------------------------------------
    # Common extraction helpers
    # ------------------------------------------------------------------

    def get_market_regime(self, context: SignalContext) -> MarketRegime:
        return context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN

    def extract_oi_features(self, context: SignalContext) -> OIFeatures | None:
        raw = context.open_interest.get("features") if context.open_interest else None
        if isinstance(raw, OIFeatures):
            return raw

        if not isinstance(raw, dict):
            return None

        try:
            return OIFeatures(**raw)
        except Exception:
            self.logger.exception(
                "Failed to parse OIFeatures | symbol=%s strategy=%s",
                getattr(context, "symbol", "UNKNOWN"),
                self.STRATEGY_NAME,
            )
            return None

    def extract_domain_section(
        self,
        context: SignalContext,
        section: str,
        *,
        prefix_nested_keys: bool = False,
    ) -> dict[str, Any]:
        domain = context.open_interest or {}
        raw: dict[str, Any] = {}

        nested = domain.get(section)
        if isinstance(nested, dict):
            if prefix_nested_keys:
                raw.update({f"{section}_{key}": value for key, value in nested.items()})
            else:
                raw.update(nested)

        return raw

    def merge_domain_keys(
        self,
        *,
        raw: dict[str, Any],
        domain: dict[str, Any],
        keys: tuple[str, ...],
    ) -> None:
        for key in keys:
            if key in domain and key not in raw:
                raw[key] = domain[key]

    def merge_aliases(
        self,
        *,
        raw: dict[str, Any],
        domain: dict[str, Any],
        aliases: dict[str, str],
    ) -> None:
        for src_key, dst_key in aliases.items():
            if src_key in domain and dst_key not in raw:
                raw[dst_key] = domain[src_key]

    def merge_feature_aliases(
        self,
        *,
        raw: dict[str, Any],
        context: SignalContext,
        feature_aliases: dict[str, str],
    ) -> None:
        for feature_name, dst_key in feature_aliases.items():
            if dst_key not in raw and context.has_feature(feature_name):
                raw[dst_key] = context.get_feature(feature_name)

    # ------------------------------------------------------------------
    # Common parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_int(value: Any, default: int | None = None) -> int | None:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def safe_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    @classmethod
    def normalize_reasons(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, tuple):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        return [text] if text else []

    @classmethod
    def parse_oi_regime(cls, value: Any) -> OIRegime:
        if isinstance(value, OIRegime):
            return value

        text = cls.safe_str(value).upper()
        if not text:
            return OIRegime.NEUTRAL

        for item in OIRegime:
            if item.value.upper() == text:
                return item

        return OIRegime.NEUTRAL

    @classmethod
    def parse_divergence_type(cls, value: Any) -> OIDivergenceType:
        if isinstance(value, OIDivergenceType):
            return value

        text = cls.safe_str(value).upper()
        if not text:
            return OIDivergenceType.NONE

        for item in OIDivergenceType:
            if item.value.upper() == text:
                return item

        return OIDivergenceType.NONE

    # ------------------------------------------------------------------
    # Common scoring helpers
    # ------------------------------------------------------------------

    def category_weight(self) -> float:
        if hasattr(self.config, "get_category_weight"):
            return self.config.get_category_weight(StrategyCategory.OPEN_INTEREST)
        return self.config.weighting.category_weights.get(StrategyCategory.OPEN_INTEREST, 1.0)

    def strategy_weight(self) -> float:
        if hasattr(self.config, "get_strategy_weight"):
            return self.config.get_strategy_weight(self.STRATEGY_NAME, default=1.0)

        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        return strategy_cfg.weight if strategy_cfg is not None else 1.0

    def regime_adjustment(self, regime: MarketRegime) -> float:
        if hasattr(self.config, "get_regime_adjustment"):
            return self.config.get_regime_adjustment(regime)
        return self.config.weighting.regime_adjustments.get(regime, 1.0)

    def weighted_score(self, base_score: float, regime: MarketRegime) -> float:
        return self.clamp(
            base_score
            * self.category_weight()
            * self.strategy_weight()
            * self.regime_adjustment(regime)
        )

    def confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    @staticmethod
    def strength_from_score(score: float) -> SignalStrength:
        if score >= 0.90:
            return SignalStrength.EXTREME
        if score >= 0.75:
            return SignalStrength.STRONG
        if score >= 0.55:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @staticmethod
    def priority_from_score(score: float) -> SignalPriority:
        if score >= 0.90:
            return SignalPriority.CRITICAL
        if score >= 0.75:
            return SignalPriority.HIGH
        if score >= 0.50:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    # ------------------------------------------------------------------
    # Common signal enrichment helpers
    # ------------------------------------------------------------------

    def add_required_source_features(self, signal: StrategySignal, context: SignalContext) -> None:
        for feature_name in self.REQUIRED_FEATURES:
            if context.has_feature(feature_name):
                signal.add_source_feature(feature_name)

    def append_oi_feature_reasons(self, signal: StrategySignal, features: OIFeatures) -> None:
        if features.volume_ratio is not None:
            signal.add_reason(f"volume_ratio:{features.volume_ratio:.4f}")
        if features.oi_delta_pct is not None:
            signal.add_reason(f"oi_delta_pct:{features.oi_delta_pct:.4f}")
        if features.price_delta_pct is not None:
            signal.add_reason(f"price_delta_pct:{features.price_delta_pct:.4f}")
        if features.oi_pressure_score is not None:
            signal.add_reason(f"oi_pressure_score:{features.oi_pressure_score:.4f}")
        if features.oi_price_efficiency is not None:
            signal.add_reason(f"oi_price_efficiency:{features.oi_price_efficiency:.4f}")

    def append_oi_feature_metadata(self, signal: StrategySignal, features: OIFeatures) -> None:
        signal.metadata.update(
            {
                "oi": features.oi,
                "oi_delta": features.oi_delta,
                "oi_delta_pct": features.oi_delta_pct,
                "oi_ma_fast": getattr(features, "oi_ma_fast", None),
                "oi_ma_slow": getattr(features, "oi_ma_slow", None),
                "oi_zscore": features.oi_zscore,
                "oi_velocity": features.oi_velocity,
                "oi_acceleration": features.oi_acceleration,
                "price": features.price,
                "price_delta": getattr(features, "price_delta", None),
                "price_delta_pct": features.price_delta_pct,
                "volume": getattr(features, "volume", None),
                "volume_ratio": features.volume_ratio,
                "funding_rate": features.funding_rate,
                "liquidation_imbalance": features.liquidation_imbalance,
                "aggressive_flow_imbalance": features.aggressive_flow_imbalance,
                "oi_change_per_volume": features.oi_change_per_volume,
                "oi_price_efficiency": features.oi_price_efficiency,
                "oi_pressure_score": features.oi_pressure_score,
                "oi_direction": getattr(features.oi_direction, "value", None),
                "price_direction": getattr(features.price_direction, "value", None),
            }
        )
