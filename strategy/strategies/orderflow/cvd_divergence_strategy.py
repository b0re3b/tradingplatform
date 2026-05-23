# trading_system/strategy/strategies/orderflow/cvd_divergence_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategyMetadata, StrategySignal
from .base import (
    ORDERFLOW_FEATURES,
    OrderflowCompositeSnapshot,
    OrderflowStrategyConfig,
    OrderflowTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    confidence_from_components,
    cvd_divergence_filter_reason,
    cvd_divergence_side_from_snapshot,
    cvd_source_features,
    extract_aggressive_buy_ratio,
    extract_aggressive_net_notional_delta,
    extract_aggressive_sell_ratio,
    extract_cvd_change_pct,
    extract_cvd_delta_ratio,
    extract_cvd_slope,
    extract_cvd_value,
    extract_event_time,
    extract_large_buy_trades,
    extract_large_sell_trades,
    extract_orderbook_imbalance_diff,
    extract_price_change_pct,
    extract_total_notional,
    extract_total_volume,
    extract_trades_count,
    freshness_score,
    is_directional_side,
    is_stale,
    magnitude_score,
    percent_score,
    ratio_score,
    serialize_for_metadata,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class CvdDivergencePayload:
    """
    Normalized strategy-level payload для CVD divergence.

    Source of truth:
        StrategyContext / FeatureSource.ORDERFLOW

    Preferred normalized form:
        OrderflowCompositeSnapshot

    Strategy idea:
        bullish divergence:
            price_change_pct < 0 while CVD strengthens;

        bearish divergence:
            price_change_pct > 0 while CVD weakens.
    """
    _logger = logging.getLogger(__name__ + ".CvdDivergencePayload")

    snapshot: OrderflowCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def price_change_pct(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.price_change_pct")
        return extract_price_change_pct(self.snapshot)

    @property
    def cvd_change_pct(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.cvd_change_pct")
        return extract_cvd_change_pct(self.snapshot)

    @property
    def cvd_delta_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.cvd_delta_ratio")
        return extract_cvd_delta_ratio(self.snapshot)

    @property
    def cvd_slope(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.cvd_slope")
        return extract_cvd_slope(self.snapshot)

    @property
    def trades_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.trades_count")
        return extract_trades_count(self.snapshot)

    @property
    def total_volume(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.total_volume")
        return extract_total_volume(self.snapshot)

    @property
    def total_notional(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergencePayload.total_notional")
        return extract_total_notional(self.snapshot)


@dataclass(slots=True)
class CvdDivergenceStrategyConfig(OrderflowStrategyConfig):
    """
    Unified CVD divergence strategy config.

    Strategy idea:
    - read normalized CVD/orderflow context from StrategyContext;
    - detect directional CVD divergence;
    - build internal reversal StrategySignal;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """
    _logger = logging.getLogger(__name__ + ".CvdDivergenceStrategyConfig")

    require_fresh_cvd: bool = True
    require_actionable_side: bool = True

    min_abs_price_change_pct: float = 0.05
    min_abs_cvd_change_pct: float = 0.05
    min_abs_delta_ratio: float = 0.08
    min_abs_cvd_slope: float = 0.0

    min_trades_count: int = 12
    min_strength_for_signal: float = 0.25

    bullish_divergence_score_threshold: float = 0.55
    bearish_divergence_score_threshold: float = 0.55

    min_orderbook_alignment_bonus_threshold: float = 0.05
    min_aggressive_confirmation_ratio: float = 0.52
    min_large_trade_confirmation_count: int = 0

    score_price_weight: float = 0.22
    score_cvd_change_weight: float = 0.24
    score_delta_ratio_weight: float = 0.24
    score_cvd_slope_weight: float = 0.10
    score_participation_weight: float = 0.08
    score_context_weight: float = 0.07
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    orderbook_alignment_bonus: float = 0.04
    aggressive_confirmation_bonus: float = 0.04
    large_trade_confirmation_bonus: float = 0.03
    volume_participation_bonus: float = 0.03

    tag_cvd_divergence: str = "cvd_divergence"
    tag_bullish_divergence: str = "bullish_cvd_divergence"
    tag_bearish_divergence: str = "bearish_cvd_divergence"
    tag_reversal: str = "reversal"
    tag_absorption: str = "absorption"
    tag_delta_dislocation: str = "delta_dislocation"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_orderflow_features: tuple[str, ...] = (
        ORDERFLOW_FEATURES.CVD,
        ORDERFLOW_FEATURES.CVD_DELTA_RATIO,
        ORDERFLOW_FEATURES.CVD_CHANGE_PCT,
        ORDERFLOW_FEATURES.CVD_SLOPE,
        ORDERFLOW_FEATURES.CVD_PRICE_CHANGE_PCT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategyConfig.validate")
        OrderflowStrategyConfig.validate(self)

        non_negative_fields = {
            "min_abs_price_change_pct": self.min_abs_price_change_pct,
            "min_abs_cvd_change_pct": self.min_abs_cvd_change_pct,
            "min_abs_delta_ratio": self.min_abs_delta_ratio,
            "min_abs_cvd_slope": self.min_abs_cvd_slope,
            "bullish_divergence_score_threshold": self.bullish_divergence_score_threshold,
            "bearish_divergence_score_threshold": self.bearish_divergence_score_threshold,
            "min_orderbook_alignment_bonus_threshold": self.min_orderbook_alignment_bonus_threshold,
            "min_aggressive_confirmation_ratio": self.min_aggressive_confirmation_ratio,
            "orderbook_alignment_bonus": self.orderbook_alignment_bonus,
            "aggressive_confirmation_bonus": self.aggressive_confirmation_bonus,
            "large_trade_confirmation_bonus": self.large_trade_confirmation_bonus,
            "volume_participation_bonus": self.volume_participation_bonus,
        }

        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        unit_fields = {
            "min_strength_for_signal": self.min_strength_for_signal,
            "min_aggressive_confirmation_ratio": self.min_aggressive_confirmation_ratio,
            "orderbook_alignment_bonus": self.orderbook_alignment_bonus,
            "aggressive_confirmation_bonus": self.aggressive_confirmation_bonus,
            "large_trade_confirmation_bonus": self.large_trade_confirmation_bonus,
            "volume_participation_bonus": self.volume_participation_bonus,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_trades_count < 1:
            raise StrategyConfigError("min_trades_count must be >= 1")

        if self.min_large_trade_confirmation_count < 0:
            raise StrategyConfigError("min_large_trade_confirmation_count must be >= 0")

        score_weights = {
            "score_price_weight": self.score_price_weight,
            "score_cvd_change_weight": self.score_cvd_change_weight,
            "score_delta_ratio_weight": self.score_delta_ratio_weight,
            "score_cvd_slope_weight": self.score_cvd_slope_weight,
            "score_participation_weight": self.score_participation_weight,
            "score_context_weight": self.score_context_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }

        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_cvd_divergence",
            "tag_bullish_divergence",
            "tag_bearish_divergence",
            "tag_reversal",
            "tag_absorption",
            "tag_delta_dislocation",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_orderflow_features:
            raise StrategyConfigError("required_orderflow_features cannot be empty")

        for feature in self.required_orderflow_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_orderflow_features cannot contain empty feature names"
                )


class CvdDivergenceStrategy(OrderflowTradingStrategy):
    """
    Unified CVD divergence strategy.

    Input:
        StrategyContext with FeatureSource.ORDERFLOW domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".CvdDivergenceStrategy")

    component_namespace = "strategy.orderflow.cvd_divergence"
    category: StrategyCategory = StrategyCategory.ORDERFLOW
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        orderflow_config: CvdDivergenceStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy.__init__")
        resolved_orderflow_config = (
            orderflow_config or CvdDivergenceStrategyConfig()
        )
        resolved_orderflow_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            orderflow_config=resolved_orderflow_config,
            service_name=service_name,
        )

        self.cvd_config: CvdDivergenceStrategyConfig = resolved_orderflow_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy.strategy_name")
        return "cvd_divergence"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            tags=[
                self.cvd_config.tag_orderflow,
                self.cvd_config.tag_cvd,
                self.cvd_config.tag_cvd_divergence,
                self.cvd_config.tag_reversal,
                self.cvd_config.tag_absorption,
                self.cvd_config.tag_delta_dislocation,
                "analytics_orderflow",
            ],
            version="2.0.0",
            description=(
                "Detects bullish/bearish CVD divergence from normalized "
                "orderflow StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.RANGING,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.orderflow",
                "strategy_type": "cvd_divergence",
                "base_class": "OrderflowTradingStrategy",
                "canonical_payload": "OrderflowCompositeSnapshot",
                "uses_cvd": True,
                "uses_volume_delta": False,
                "uses_aggressive_trades": True,
                "uses_orderbook": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.cvd_config.required_orderflow_features
        )

    async def generate_signal(
            self,
            context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy.generate_signal")
        self.validate_context_requirements(context)

        required_features = tuple(self.cvd_config.required_orderflow_features)

        if not self.has_any_orderflow_data(context, required_features):
            self.remember_no_signal(
                "missing_orderflow_cvd_divergence_contract",
                orderflow_domain_keys=sorted(self.orderflow_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_orderflow_features(context, required_features):
            self.remember_no_signal(
                "stale_orderflow_cvd_divergence_features",
                required_features=sorted(required_features),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            self.remember_no_signal(
                "orderflow_cvd_divergence_payload_not_resolved",
                orderflow_domain=self.orderflow_domain(context),
                required_features=sorted(self.required_features()),
            )
            return None

        if (
                self.cvd_config.require_fresh_cvd
                and is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.cvd_config.stale_feature_max_age_seconds,
        )
        ):
            self.remember_no_signal(
                "stale_orderflow_cvd_divergence_snapshot",
                event_time=payload.event_time.isoformat() if payload.event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.cvd_config.stale_feature_max_age_seconds,
            )
            return None

        common_rejection = cvd_divergence_filter_reason(
            payload.snapshot,
            min_abs_price_change_pct=self.cvd_config.min_abs_price_change_pct,
            min_abs_cvd_change_pct=self.cvd_config.min_abs_cvd_change_pct,
            min_abs_delta_ratio=self.cvd_config.min_abs_delta_ratio,
            min_abs_cvd_slope=self.cvd_config.min_abs_cvd_slope,
            min_trades_count=self.cvd_config.min_trades_count,
            min_strength_for_signal=self.cvd_config.min_strength_for_signal,
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "orderflow_cvd_divergence_quality_filter_failed",
                filter_reason=common_rejection,
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
                price_change_pct=payload.price_change_pct,
                cvd_change_pct=payload.cvd_change_pct,
                cvd_delta_ratio=payload.cvd_delta_ratio,
                cvd_slope=payload.cvd_slope,
                trades_count=payload.trades_count,
                total_volume=payload.total_volume,
                total_notional=payload.total_notional,
            )
            return None

        side = payload.side
        if self.cvd_config.require_actionable_side and not is_directional_side(side):
            self.remember_no_signal(
                "orderflow_cvd_divergence_side_not_directional",
                side=serialize_for_metadata(side),
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        side_score_threshold = (
            self.cvd_config.bullish_divergence_score_threshold
            if side is SignalSide.LONG
            else self.cvd_config.bearish_divergence_score_threshold
        )

        min_score = max(
            self.cvd_config.min_signal_score,
            side_score_threshold,
        )
        if breakdown.score < min_score:
            self.remember_no_signal(
                "orderflow_cvd_divergence_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_score=min_score,
                side_score_threshold=side_score_threshold,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.cvd_config.min_signal_confidence:
            self.remember_no_signal(
                "orderflow_cvd_divergence_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.cvd_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "cvd_divergence_signal",
                    f"side:{side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "orderflow_setup_family": "cvd_divergence",
            "orderflow_strategy_version": "2.0.0",
            "contract": "orderflow",
            "contract_version": "strategy-domain-v1",
            "primary_section": "cvd",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": (
                payload.event_time.isoformat()
                if payload.event_time is not None
                else None
            ),
            "tags": tags,
            "divergence_side": side.value,
            "price_change_pct": payload.price_change_pct,
            "cvd_change_pct": payload.cvd_change_pct,
            "cvd_delta_ratio": payload.cvd_delta_ratio,
            "cvd_slope": payload.cvd_slope,
            "cvd_value": extract_cvd_value(payload.snapshot),
            "trades_count": payload.trades_count,
            "total_volume": payload.total_volume,
            "total_notional": payload.total_notional,
            "aggressive_buy_ratio": extract_aggressive_buy_ratio(payload.snapshot),
            "aggressive_sell_ratio": extract_aggressive_sell_ratio(payload.snapshot),
            "aggressive_net_notional_delta": extract_aggressive_net_notional_delta(
                payload.snapshot
            ),
            "large_buy_trades": extract_large_buy_trades(payload.snapshot),
            "large_sell_trades": extract_large_sell_trades(payload.snapshot),
            "orderbook_imbalance_diff": extract_orderbook_imbalance_diff(
                payload.snapshot
            ),
        }

        return self.build_orderflow_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.cvd_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.cvd_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> CvdDivergencePayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._extract_payload")
        snapshot = self.resolve_orderflow_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        if snapshot.trades_count < self.cvd_config.min_trades_count:
            return None

        side = cvd_divergence_side_from_snapshot(
            snapshot,
            min_abs_price_change_pct=self.cvd_config.min_abs_price_change_pct,
            min_abs_cvd_change_pct=self.cvd_config.min_abs_cvd_change_pct,
            min_abs_delta_ratio=self.cvd_config.min_abs_delta_ratio,
            min_abs_cvd_slope=self.cvd_config.min_abs_cvd_slope,
        )
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "bullish_cvd_divergence"
            if side is SignalSide.LONG
            else "bearish_cvd_divergence",
            f"price_change_pct:{extract_price_change_pct(snapshot):.6f}",
            f"cvd_change_pct:{extract_cvd_change_pct(snapshot):.6f}",
            f"cvd_delta_ratio:{extract_cvd_delta_ratio(snapshot):.6f}",
            f"cvd_slope:{extract_cvd_slope(snapshot):.6f}",
        ]

        return CvdDivergencePayload(
            snapshot=snapshot,
            side=side,
            event_time=event_time,
            reasons=reasons,
            raw=self.orderflow_domain(context),
        )

    # ------------------------------------------------------------------
    # Scoring / confidence
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: CvdDivergencePayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._build_score_breakdown")
        snapshot = payload.snapshot
        side = payload.side

        price_component = percent_score(
            abs(payload.price_change_pct),
            scale=max(self.cvd_config.min_abs_price_change_pct * 4.0, 0.01),
        )
        cvd_change_component = percent_score(
            abs(payload.cvd_change_pct),
            scale=max(self.cvd_config.min_abs_cvd_change_pct * 4.0, 0.01),
        )
        delta_ratio_component = ratio_score(
            abs(payload.cvd_delta_ratio),
            scale=max(self.cvd_config.min_abs_delta_ratio * 4.0, 0.01),
        )
        cvd_slope_component = ratio_score(
            abs(payload.cvd_slope),
            scale=max(self.cvd_config.min_abs_cvd_slope * 4.0, 1.0),
        )
        participation_component = self._participation_component(snapshot)
        context_component = self._context_component(snapshot, side)
        fresh_score = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.cvd_config.stale_feature_max_age_seconds,
        )

        components = {
            "price": price_component,
            "cvd_change": cvd_change_component,
            "delta_ratio": delta_ratio_component,
            "cvd_slope": cvd_slope_component,
            "participation": participation_component,
            "context": context_component,
            "freshness": fresh_score,
        }
        weights = {
            "price": self.cvd_config.score_price_weight,
            "cvd_change": self.cvd_config.score_cvd_change_weight,
            "delta_ratio": self.cvd_config.score_delta_ratio_weight,
            "cvd_slope": self.cvd_config.score_cvd_slope_weight,
            "participation": self.cvd_config.score_participation_weight,
            "context": self.cvd_config.score_context_weight,
            "freshness": self.cvd_config.score_freshness_weight,
        }

        primary = weighted_score(
            {
                "price": price_component,
                "cvd_change": cvd_change_component,
                "delta_ratio": delta_ratio_component,
                "cvd_slope": cvd_slope_component,
            },
            {
                "price": 0.25,
                "cvd_change": 0.30,
                "delta_ratio": 0.30,
                "cvd_slope": 0.15,
            },
        )

        score = weighted_score(components, weights, default=primary)
        confidence = confidence_from_components(
            primary=primary,
            context=context_component,
            confirmation=participation_component,
            freshness=fresh_score,
            primary_weight=self.cvd_config.confidence_primary_weight,
            context_weight=self.cvd_config.confidence_context_weight,
            confirmation_weight=self.cvd_config.confidence_confirmation_weight,
            freshness_weight=self.cvd_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "price_cvd_dislocation",
            f"cvd_delta_ratio:{payload.cvd_delta_ratio:.6f}",
            f"cvd_change_pct:{payload.cvd_change_pct:.6f}",
        ]

        orderbook_bonus = self._orderbook_alignment_bonus(snapshot, side)
        if orderbook_bonus > 0:
            score += orderbook_bonus
            confidence += min(0.03, orderbook_bonus)
            confirmations.append("orderbook_supports_cvd_divergence")

        aggressive_bonus = self._aggressive_confirmation_bonus(snapshot, side)
        if aggressive_bonus > 0:
            score += aggressive_bonus
            confidence += min(0.03, aggressive_bonus)
            confirmations.append("aggressive_flow_supports_cvd_divergence")

        large_trade_bonus = self._large_trade_confirmation_bonus(snapshot, side)
        if large_trade_bonus > 0:
            score += large_trade_bonus
            confirmations.append("large_trades_support_cvd_divergence")

        volume_bonus = self._volume_participation_bonus(snapshot)
        if volume_bonus > 0:
            score += volume_bonus
            confidence += min(0.02, volume_bonus)
            confirmations.append("sufficient_volume_participation")

        if side is SignalSide.LONG:
            confirmations.append("bullish_cvd_divergence")
        elif side is SignalSide.SHORT:
            confirmations.append("bearish_cvd_divergence")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _participation_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._participation_component")
        trades_component = unit_score(
            snapshot.trades_count / max(self.cvd_config.min_trades_count * 3, 1)
        )
        volume_component = magnitude_score(
            snapshot.total_volume,
            scale=max(self.cvd_config.min_total_volume * 3.0, 1.0),
        )
        notional_component = magnitude_score(
            snapshot.total_notional,
            scale=max(self.cvd_config.min_total_notional * 3.0, 1.0),
        )

        return weighted_score(
            {
                "trades": trades_component,
                "volume": volume_component,
                "notional": notional_component,
            },
            {
                "trades": 0.50,
                "volume": 0.25,
                "notional": 0.25,
            },
        )

    def _context_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._context_component")
        orderbook = max(
            0.0,
            self._orderbook_alignment_score(snapshot, side),
        )
        aggressive = max(
            0.0,
            self._aggressive_alignment_score(snapshot, side),
        )
        large_trades = max(
            0.0,
            self._large_trade_alignment_score(snapshot, side),
        )

        return weighted_score(
            {
                "orderbook": orderbook,
                "aggressive": aggressive,
                "large_trades": large_trades,
            },
            {
                "orderbook": 0.30,
                "aggressive": 0.50,
                "large_trades": 0.20,
            },
        )

    def _orderbook_alignment_score(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._orderbook_alignment_score")
        imbalance = extract_orderbook_imbalance_diff(snapshot)

        if side is SignalSide.LONG:
            return unit_score((imbalance + 1.0) / 2.0)

        if side is SignalSide.SHORT:
            return unit_score((1.0 - imbalance) / 2.0)

        return 0.0

    def _aggressive_alignment_score(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._aggressive_alignment_score")
        buy_ratio = extract_aggressive_buy_ratio(snapshot)
        sell_ratio = extract_aggressive_sell_ratio(snapshot)
        aggressive_delta = extract_aggressive_net_notional_delta(snapshot)

        if side is SignalSide.LONG:
            return weighted_score(
                {
                    "ratio": buy_ratio,
                    "delta": unit_score((aggressive_delta + 1.0) / 2.0),
                },
                {
                    "ratio": 0.70,
                    "delta": 0.30,
                },
            )

        if side is SignalSide.SHORT:
            return weighted_score(
                {
                    "ratio": sell_ratio,
                    "delta": unit_score((1.0 - aggressive_delta) / 2.0),
                },
                {
                    "ratio": 0.70,
                    "delta": 0.30,
                },
            )

        return 0.0

    def _large_trade_alignment_score(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._large_trade_alignment_score")
        large_buy = extract_large_buy_trades(snapshot)
        large_sell = extract_large_sell_trades(snapshot)
        total = large_buy + large_sell

        if total <= 0:
            return 0.0

        if side is SignalSide.LONG:
            return unit_score(large_buy / total)

        if side is SignalSide.SHORT:
            return unit_score(large_sell / total)

        return 0.0

    def _orderbook_alignment_bonus(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._orderbook_alignment_bonus")
        threshold = self.cvd_config.min_orderbook_alignment_bonus_threshold
        imbalance = extract_orderbook_imbalance_diff(snapshot)

        if side is SignalSide.LONG and imbalance >= threshold:
            return self.cvd_config.orderbook_alignment_bonus

        if side is SignalSide.SHORT and imbalance <= -threshold:
            return self.cvd_config.orderbook_alignment_bonus

        return 0.0

    def _aggressive_confirmation_bonus(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._aggressive_confirmation_bonus")
        threshold = self.cvd_config.min_aggressive_confirmation_ratio

        if side is SignalSide.LONG:
            if extract_aggressive_buy_ratio(snapshot) >= threshold:
                return self.cvd_config.aggressive_confirmation_bonus

        if side is SignalSide.SHORT:
            if extract_aggressive_sell_ratio(snapshot) >= threshold:
                return self.cvd_config.aggressive_confirmation_bonus

        return 0.0

    def _large_trade_confirmation_bonus(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._large_trade_confirmation_bonus")
        min_count = self.cvd_config.min_large_trade_confirmation_count
        if min_count <= 0:
            return 0.0

        if side is SignalSide.LONG:
            if extract_large_buy_trades(snapshot) >= min_count:
                return self.cvd_config.large_trade_confirmation_bonus

        if side is SignalSide.SHORT:
            if extract_large_sell_trades(snapshot) >= min_count:
                return self.cvd_config.large_trade_confirmation_bonus

        return 0.0

    def _volume_participation_bonus(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._volume_participation_bonus")
        if self.cvd_config.min_total_volume > 0:
            if snapshot.total_volume >= self.cvd_config.min_total_volume:
                return self.cvd_config.volume_participation_bonus

        if self.cvd_config.min_total_notional > 0:
            if snapshot.total_notional >= self.cvd_config.min_total_notional:
                return self.cvd_config.volume_participation_bonus

        return 0.0

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: CvdDivergencePayload) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._source_features")
        features = [
            *cvd_source_features(),
            ORDERFLOW_FEATURES.TRADES_COUNT,
            ORDERFLOW_FEATURES.TOTAL_VOLUME,
            ORDERFLOW_FEATURES.TOTAL_NOTIONAL,
        ]

        if payload.snapshot.has_aggressive_flow:
            features.extend(
                [
                    ORDERFLOW_FEATURES.AGGRESSIVE_TRADES,
                    ORDERFLOW_FEATURES.AGGRESSIVE_BUY_RATIO,
                    ORDERFLOW_FEATURES.AGGRESSIVE_SELL_RATIO,
                    ORDERFLOW_FEATURES.AGGRESSIVE_NET_NOTIONAL_DELTA,
                    ORDERFLOW_FEATURES.LARGE_BUY_TRADES,
                    ORDERFLOW_FEATURES.LARGE_SELL_TRADES,
                ]
            )

        if payload.snapshot.has_orderbook:
            features.extend(
                [
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_DIFF,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_RATIO,
                ]
            )

        return list(dict.fromkeys(features))

    def _tags(self, payload: CvdDivergencePayload) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering CvdDivergenceStrategy._tags")
        tags = [
            self.cvd_config.tag_orderflow,
            self.cvd_config.tag_cvd,
            self.cvd_config.tag_cvd_divergence,
            self.cvd_config.tag_reversal,
            self.cvd_config.tag_delta_dislocation,
        ]

        if payload.side is SignalSide.LONG:
            tags.append(self.cvd_config.tag_bullish_divergence)

        if payload.side is SignalSide.SHORT:
            tags.append(self.cvd_config.tag_bearish_divergence)

        if payload.snapshot.has_aggressive_flow:
            tags.append(self.cvd_config.tag_aggressive_flow)

        if payload.snapshot.has_orderbook:
            tags.append(self.cvd_config.tag_orderbook)

        return list(dict.fromkeys(tags))