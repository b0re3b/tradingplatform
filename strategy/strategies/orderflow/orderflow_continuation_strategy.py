# trading_system/strategy/strategies/orderflow/orderflow_continuation_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    ORDERFLOW_FEATURES,
    OrderflowCompositeSnapshot,
    OrderflowStrategyConfig,
    OrderflowTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    confidence_from_components,
    continuation_filter_reason,
    continuation_side_from_snapshot,
    continuation_source_features,
    extract_aggressive_burst_score,
    extract_aggressive_buy_ratio,
    extract_aggressive_net_notional_delta,
    extract_aggressive_net_volume_delta,
    extract_aggressive_sell_ratio,
    extract_cumulative_notional_delta,
    extract_cumulative_volume_delta,
    extract_cvd_change_pct,
    extract_cvd_delta_ratio,
    extract_cvd_slope,
    extract_event_time,
    extract_large_buy_trades,
    extract_large_sell_trades,
    extract_notional_delta,
    extract_orderbook_imbalance_diff,
    extract_orderbook_imbalance_ratio,
    extract_price_change_pct,
    extract_total_notional,
    extract_total_volume,
    extract_trades_count,
    extract_volume_delta,
    extract_volume_delta_ratio,
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


@dataclass(slots=True)
class OrderflowContinuationPayload:
    """
    Normalized strategy-level payload для orderflow continuation.

    Source of truth:
        StrategyContext / FeatureSource.ORDERFLOW

    Preferred normalized form:
        OrderflowCompositeSnapshot

    Strategy idea:
        LONG continuation:
            price up + CVD up + volume delta up + aggressive buyers dominate;

        SHORT continuation:
            price down + CVD down + volume delta down + aggressive sellers dominate.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowContinuationPayload")

    snapshot: OrderflowCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def price_change_pct(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.price_change_pct")
        return extract_price_change_pct(self.snapshot)

    @property
    def cvd_delta_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.cvd_delta_ratio")
        return extract_cvd_delta_ratio(self.snapshot)

    @property
    def cvd_change_pct(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.cvd_change_pct")
        return extract_cvd_change_pct(self.snapshot)

    @property
    def cvd_slope(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.cvd_slope")
        return extract_cvd_slope(self.snapshot)

    @property
    def volume_delta_ratio(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.volume_delta_ratio")
        return extract_volume_delta_ratio(self.snapshot)

    @property
    def volume_delta(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.volume_delta")
        return extract_volume_delta(self.snapshot)

    @property
    def cumulative_volume_delta(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.cumulative_volume_delta")
        return extract_cumulative_volume_delta(self.snapshot)

    @property
    def trades_count(self) -> int:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.trades_count")
        return extract_trades_count(self.snapshot)

    @property
    def total_volume(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.total_volume")
        return extract_total_volume(self.snapshot)

    @property
    def total_notional(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationPayload.total_notional")
        return extract_total_notional(self.snapshot)


@dataclass(slots=True)
class OrderflowContinuationStrategyConfig(OrderflowStrategyConfig):
    """
    Unified orderflow continuation strategy config.

    Strategy idea:
    - read normalized composite orderflow context from StrategyContext;
    - detect directional continuation confirmed by CVD, volume delta,
      aggressive flow and orderbook context;
    - build internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowContinuationStrategyConfig")

    require_fresh_snapshot: bool = True
    require_actionable_side: bool = True

    min_trades_count: int = 10
    min_total_volume: float = 0.0

    min_abs_price_change_pct: float = 0.03

    min_cvd_delta_ratio: float = 0.08
    min_cvd_change_pct: float = 0.03
    min_cvd_slope: float = 0.0

    min_volume_delta_ratio: float = 0.10
    min_volume_delta_abs: float = 0.0
    min_cumulative_volume_delta_abs: float = 0.0

    min_aggressive_buy_ratio: float = 0.55
    min_aggressive_sell_ratio: float = 0.55
    min_aggressive_burst_score: float = 0.0
    min_large_aggressive_trades: int = 0

    min_orderbook_imbalance_ratio: float = 0.05

    min_score_for_signal: float = 0.45
    min_confidence_for_signal: float = 0.50

    price_weight: float = 0.11
    cvd_delta_weight: float = 0.14
    cvd_change_weight: float = 0.09
    volume_delta_weight: float = 0.13
    cumulative_volume_weight: float = 0.07
    notional_delta_weight: float = 0.10
    cumulative_notional_weight: float = 0.05
    aggression_ratio_weight: float = 0.12
    aggressive_net_weight: float = 0.08
    large_trade_weight: float = 0.04
    orderbook_weight: float = 0.04
    burst_weight: float = 0.03

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_volume_bonus: float = 0.03
    strong_cvd_bonus: float = 0.04
    strong_aggression_bonus: float = 0.04
    orderbook_alignment_bonus: float = 0.03
    large_trade_bonus: float = 0.03
    burst_bonus: float = 0.02

    strong_cvd_ratio_threshold: float = 0.20
    strong_volume_delta_ratio_threshold: float = 0.20
    strong_aggressive_ratio_threshold: float = 0.65
    strong_burst_threshold: float = 0.50

    tag_orderflow_continuation: str = "orderflow_continuation"
    tag_long_continuation: str = "long_continuation"
    tag_short_continuation: str = "short_continuation"
    tag_pressure: str = "pressure"
    tag_aggressive_flow: str = "aggressive_flow"
    tag_cvd_confirmation: str = "cvd_confirmation"
    tag_volume_delta_confirmation: str = "volume_delta_confirmation"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.CONTINUATION

    required_orderflow_features: tuple[str, ...] = (
        ORDERFLOW_FEATURES.CVD_DELTA_RATIO,
        ORDERFLOW_FEATURES.VOLUME_DELTA_RATIO,
        ORDERFLOW_FEATURES.AGGRESSIVE_BUY_RATIO,
        ORDERFLOW_FEATURES.AGGRESSIVE_SELL_RATIO,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategyConfig.validate")
        OrderflowStrategyConfig.validate(self)

        non_negative_fields = {
            "min_abs_price_change_pct": self.min_abs_price_change_pct,
            "min_cvd_delta_ratio": self.min_cvd_delta_ratio,
            "min_cvd_change_pct": self.min_cvd_change_pct,
            "min_cvd_slope": self.min_cvd_slope,
            "min_volume_delta_ratio": self.min_volume_delta_ratio,
            "min_volume_delta_abs": self.min_volume_delta_abs,
            "min_cumulative_volume_delta_abs": self.min_cumulative_volume_delta_abs,
            "min_aggressive_burst_score": self.min_aggressive_burst_score,
            "min_orderbook_imbalance_ratio": self.min_orderbook_imbalance_ratio,
            "min_score_for_signal": self.min_score_for_signal,
            "strong_volume_bonus": self.strong_volume_bonus,
            "strong_cvd_bonus": self.strong_cvd_bonus,
            "strong_aggression_bonus": self.strong_aggression_bonus,
            "orderbook_alignment_bonus": self.orderbook_alignment_bonus,
            "large_trade_bonus": self.large_trade_bonus,
            "burst_bonus": self.burst_bonus,
            "strong_cvd_ratio_threshold": self.strong_cvd_ratio_threshold,
            "strong_volume_delta_ratio_threshold": self.strong_volume_delta_ratio_threshold,
            "strong_aggressive_ratio_threshold": self.strong_aggressive_ratio_threshold,
            "strong_burst_threshold": self.strong_burst_threshold,
        }

        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        unit_fields = {
            "min_aggressive_buy_ratio": self.min_aggressive_buy_ratio,
            "min_aggressive_sell_ratio": self.min_aggressive_sell_ratio,
            "min_confidence_for_signal": self.min_confidence_for_signal,
            "strong_volume_bonus": self.strong_volume_bonus,
            "strong_cvd_bonus": self.strong_cvd_bonus,
            "strong_aggression_bonus": self.strong_aggression_bonus,
            "orderbook_alignment_bonus": self.orderbook_alignment_bonus,
            "large_trade_bonus": self.large_trade_bonus,
            "burst_bonus": self.burst_bonus,
            "strong_cvd_ratio_threshold": self.strong_cvd_ratio_threshold,
            "strong_volume_delta_ratio_threshold": self.strong_volume_delta_ratio_threshold,
            "strong_aggressive_ratio_threshold": self.strong_aggressive_ratio_threshold,
            "strong_burst_threshold": self.strong_burst_threshold,
        }

        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        if self.min_trades_count < 1:
            raise StrategyConfigError("min_trades_count must be >= 1")

        if self.min_total_volume < 0:
            raise StrategyConfigError("min_total_volume must be >= 0")

        if self.min_large_aggressive_trades < 0:
            raise StrategyConfigError("min_large_aggressive_trades must be >= 0")

        score_weights = {
            "price_weight": self.price_weight,
            "cvd_delta_weight": self.cvd_delta_weight,
            "cvd_change_weight": self.cvd_change_weight,
            "volume_delta_weight": self.volume_delta_weight,
            "cumulative_volume_weight": self.cumulative_volume_weight,
            "notional_delta_weight": self.notional_delta_weight,
            "cumulative_notional_weight": self.cumulative_notional_weight,
            "aggression_ratio_weight": self.aggression_ratio_weight,
            "aggressive_net_weight": self.aggressive_net_weight,
            "large_trade_weight": self.large_trade_weight,
            "orderbook_weight": self.orderbook_weight,
            "burst_weight": self.burst_weight,
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
            "tag_orderflow_continuation",
            "tag_long_continuation",
            "tag_short_continuation",
            "tag_pressure",
            "tag_aggressive_flow",
            "tag_cvd_confirmation",
            "tag_volume_delta_confirmation",
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


class OrderflowContinuationStrategy(OrderflowTradingStrategy):
    """
    Unified orderflow continuation strategy.

    Input:
        StrategyContext with FeatureSource.ORDERFLOW domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".OrderflowContinuationStrategy")

    component_namespace = "strategy.orderflow.continuation"
    category: StrategyCategory = StrategyCategory.ORDERFLOW
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        orderflow_config: OrderflowContinuationStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy.__init__")
        resolved_orderflow_config = (
            orderflow_config or OrderflowContinuationStrategyConfig()
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

        self.continuation_config: OrderflowContinuationStrategyConfig = (
            resolved_orderflow_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy.strategy_name")
        return "orderflow_continuation"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            tags=[
                self.continuation_config.tag_orderflow,
                self.continuation_config.tag_orderflow_continuation,
                self.continuation_config.tag_continuation,
                self.continuation_config.tag_pressure,
                self.continuation_config.tag_aggressive_flow,
                self.continuation_config.tag_cvd_confirmation,
                self.continuation_config.tag_volume_delta_confirmation,
                "analytics_orderflow",
            ],
            version="2.0.0",
            description=(
                "Detects orderflow continuation using price movement, CVD, "
                "volume delta, aggressive trades and orderbook imbalance from "
                "normalized StrategyContext."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.orderflow",
                "strategy_type": "orderflow_continuation",
                "base_class": "OrderflowTradingStrategy",
                "canonical_payload": "OrderflowCompositeSnapshot",
                "uses_cvd": True,
                "uses_volume_delta": True,
                "uses_aggressive_trades": True,
                "uses_orderbook": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.continuation_config.required_orderflow_features
        )

    async def generate_signal(
            self,
            context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy.generate_signal")
        self.validate_context_requirements(context)

        required_features = tuple(self.continuation_config.required_orderflow_features)

        if not self.has_any_orderflow_data(context, required_features):
            self.remember_no_signal(
                "missing_orderflow_continuation_contract",
                orderflow_domain_keys=sorted(self.orderflow_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_orderflow_features(context, required_features):
            self.remember_no_signal(
                "stale_orderflow_continuation_features",
                required_features=sorted(required_features),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            self.remember_no_signal(
                "orderflow_continuation_payload_not_resolved",
                orderflow_domain=self.orderflow_domain(context),
                required_features=sorted(self.required_features()),
            )
            return None

        if (
                self.continuation_config.require_fresh_snapshot
                and is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.continuation_config.stale_feature_max_age_seconds,
        )
        ):
            self.remember_no_signal(
                "stale_orderflow_continuation_snapshot",
                event_time=payload.event_time.isoformat() if payload.event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=(
                    self.continuation_config.stale_feature_max_age_seconds
                ),
            )
            return None

        common_rejection = continuation_filter_reason(
            payload.snapshot,
            min_trades_count=self.continuation_config.min_trades_count,
            min_total_volume=self.continuation_config.min_total_volume,
            min_abs_price_change_pct=(
                self.continuation_config.min_abs_price_change_pct
            ),
            min_cvd_delta_ratio=self.continuation_config.min_cvd_delta_ratio,
            min_volume_delta_ratio=(
                self.continuation_config.min_volume_delta_ratio
            ),
            min_aggressive_buy_ratio=(
                self.continuation_config.min_aggressive_buy_ratio
            ),
            min_aggressive_sell_ratio=(
                self.continuation_config.min_aggressive_sell_ratio
            ),
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "orderflow_continuation_quality_filter_failed",
                filter_reason=common_rejection,
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
                trades_count=payload.trades_count,
                total_volume=payload.total_volume,
                total_notional=payload.total_notional,
                price_change_pct=payload.price_change_pct,
                cvd_delta_ratio=payload.cvd_delta_ratio,
                volume_delta_ratio=payload.volume_delta_ratio,
            )
            return None

        side = payload.side
        if (
                self.continuation_config.require_actionable_side
                and not is_directional_side(side)
        ):
            self.remember_no_signal(
                "orderflow_continuation_side_not_directional",
                side=serialize_for_metadata(side),
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        min_score = max(
            self.continuation_config.min_signal_score,
            self.continuation_config.min_score_for_signal,
        )
        if breakdown.score < min_score:
            self.remember_no_signal(
                "orderflow_continuation_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_score=min_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        min_confidence = max(
            self.continuation_config.min_signal_confidence,
            self.continuation_config.min_confidence_for_signal,
        )
        if breakdown.confidence < min_confidence:
            self.remember_no_signal(
                "orderflow_continuation_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_confidence=min_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "orderflow_continuation_signal",
                    f"side:{side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "orderflow_setup_family": "orderflow_continuation",
            "orderflow_strategy_version": "2.0.0",
            "contract": "orderflow",
            "contract_version": "strategy-domain-v1",
            "primary_section": "composite",
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
            "continuation_side": side.value,
            "price_change_pct": payload.price_change_pct,
            "cvd_delta_ratio": payload.cvd_delta_ratio,
            "cvd_change_pct": payload.cvd_change_pct,
            "cvd_slope": payload.cvd_slope,
            "volume_delta_ratio": payload.volume_delta_ratio,
            "volume_delta": payload.volume_delta,
            "cumulative_volume_delta": payload.cumulative_volume_delta,
            "notional_delta": extract_notional_delta(payload.snapshot),
            "cumulative_notional_delta": extract_cumulative_notional_delta(
                payload.snapshot
            ),
            "aggressive_buy_ratio": extract_aggressive_buy_ratio(payload.snapshot),
            "aggressive_sell_ratio": extract_aggressive_sell_ratio(payload.snapshot),
            "aggressive_burst_score": extract_aggressive_burst_score(payload.snapshot),
            "aggressive_net_volume_delta": extract_aggressive_net_volume_delta(
                payload.snapshot
            ),
            "aggressive_net_notional_delta": extract_aggressive_net_notional_delta(
                payload.snapshot
            ),
            "large_buy_trades": extract_large_buy_trades(payload.snapshot),
            "large_sell_trades": extract_large_sell_trades(payload.snapshot),
            "orderbook_imbalance_ratio": extract_orderbook_imbalance_ratio(
                payload.snapshot
            ),
            "orderbook_imbalance_diff": extract_orderbook_imbalance_diff(
                payload.snapshot
            ),
            "trades_count": payload.trades_count,
            "total_volume": payload.total_volume,
            "total_notional": payload.total_notional,
        }

        return self.build_orderflow_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.continuation_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.continuation_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> OrderflowContinuationPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._extract_payload")
        snapshot = self.resolve_orderflow_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        if snapshot.trades_count < self.continuation_config.min_trades_count:
            return None

        if snapshot.total_volume < self.continuation_config.min_total_volume:
            return None

        side = continuation_side_from_snapshot(
            snapshot,
            min_abs_price_change_pct=self.continuation_config.min_abs_price_change_pct,
            min_cvd_delta_ratio=self.continuation_config.min_cvd_delta_ratio,
            min_volume_delta_ratio=self.continuation_config.min_volume_delta_ratio,
            min_aggressive_buy_ratio=self.continuation_config.min_aggressive_buy_ratio,
            min_aggressive_sell_ratio=self.continuation_config.min_aggressive_sell_ratio,
            max_orderbook_contradiction=self.continuation_config.min_orderbook_imbalance_ratio,
        )
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "long_orderflow_continuation"
            if side is SignalSide.LONG
            else "short_orderflow_continuation",
            f"price_change_pct:{extract_price_change_pct(snapshot):.6f}",
            f"cvd_delta_ratio:{extract_cvd_delta_ratio(snapshot):.6f}",
            f"volume_delta_ratio:{extract_volume_delta_ratio(snapshot):.6f}",
            f"aggressive_buy_ratio:{extract_aggressive_buy_ratio(snapshot):.6f}",
            f"aggressive_sell_ratio:{extract_aggressive_sell_ratio(snapshot):.6f}",
        ]

        return OrderflowContinuationPayload(
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
        payload: OrderflowContinuationPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._build_score_breakdown")
        snapshot = payload.snapshot
        side = payload.side

        components = {
            "price": percent_score(
                abs(payload.price_change_pct),
                scale=max(self.continuation_config.min_abs_price_change_pct * 4.0, 0.01),
            ),
            "cvd_delta": ratio_score(
                abs(payload.cvd_delta_ratio),
                scale=max(self.continuation_config.min_cvd_delta_ratio * 4.0, 0.01),
            ),
            "cvd_change": percent_score(
                abs(payload.cvd_change_pct),
                scale=max(self.continuation_config.min_cvd_change_pct * 4.0, 0.01),
            ),
            "volume_delta": ratio_score(
                abs(payload.volume_delta_ratio),
                scale=max(self.continuation_config.min_volume_delta_ratio * 4.0, 0.01),
            ),
            "cumulative_volume": magnitude_score(
                abs(payload.cumulative_volume_delta),
                scale=max(abs(snapshot.total_volume), 1.0),
            ),
            "notional_delta": ratio_score(
                abs(self._notional_delta_ratio(snapshot)),
                scale=0.45,
            ),
            "cumulative_notional": magnitude_score(
                abs(extract_cumulative_notional_delta(snapshot)),
                scale=max(abs(snapshot.total_notional), 1.0),
            ),
            "aggression_ratio": (
                extract_aggressive_buy_ratio(snapshot)
                if side is SignalSide.LONG
                else extract_aggressive_sell_ratio(snapshot)
            ),
            "aggressive_net": ratio_score(
                abs(snapshot.directional_aggressive_notional_delta(side)),
                scale=max(abs(snapshot.total_notional), 1.0),
            ),
            "large_trades": self._large_trade_component(snapshot, side),
            "orderbook": self._orderbook_component(snapshot, side),
            "burst": ratio_score(
                extract_aggressive_burst_score(snapshot),
                scale=1.0,
            ),
        }

        weights = {
            "price": self.continuation_config.price_weight,
            "cvd_delta": self.continuation_config.cvd_delta_weight,
            "cvd_change": self.continuation_config.cvd_change_weight,
            "volume_delta": self.continuation_config.volume_delta_weight,
            "cumulative_volume": self.continuation_config.cumulative_volume_weight,
            "notional_delta": self.continuation_config.notional_delta_weight,
            "cumulative_notional": self.continuation_config.cumulative_notional_weight,
            "aggression_ratio": self.continuation_config.aggression_ratio_weight,
            "aggressive_net": self.continuation_config.aggressive_net_weight,
            "large_trades": self.continuation_config.large_trade_weight,
            "orderbook": self.continuation_config.orderbook_weight,
            "burst": self.continuation_config.burst_weight,
        }

        primary = weighted_score(
            {
                "price": components["price"],
                "cvd": components["cvd_delta"],
                "volume_delta": components["volume_delta"],
                "aggression": components["aggression_ratio"],
            },
            {
                "price": 0.20,
                "cvd": 0.30,
                "volume_delta": 0.25,
                "aggression": 0.25,
            },
        )
        context_component = weighted_score(
            {
                "orderbook": components["orderbook"],
                "large_trades": components["large_trades"],
                "burst": components["burst"],
            },
            {
                "orderbook": 0.35,
                "large_trades": 0.30,
                "burst": 0.35,
            },
        )
        confirmation_component = weighted_score(
            {
                "cvd_change": components["cvd_change"],
                "notional_delta": components["notional_delta"],
                "aggressive_net": components["aggressive_net"],
                "cumulative": components["cumulative_volume"],
            },
            {
                "cvd_change": 0.25,
                "notional_delta": 0.25,
                "aggressive_net": 0.30,
                "cumulative": 0.20,
            },
        )
        fresh_score = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.continuation_config.stale_feature_max_age_seconds,
        )

        score = weighted_score(components, weights, default=primary)
        confidence = confidence_from_components(
            primary=primary,
            context=context_component,
            confirmation=confirmation_component,
            freshness=fresh_score,
            primary_weight=self.continuation_config.confidence_primary_weight,
            context_weight=self.continuation_config.confidence_context_weight,
            confirmation_weight=self.continuation_config.confidence_confirmation_weight,
            freshness_weight=self.continuation_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "price_flow_continuation",
            f"side:{side.value}",
            f"cvd_delta_ratio:{payload.cvd_delta_ratio:.6f}",
            f"volume_delta_ratio:{payload.volume_delta_ratio:.6f}",
        ]

        bonus = self._confirmation_bonus(snapshot, side)
        if bonus > 0:
            score += bonus
            confidence += min(0.06, bonus)
            confirmations.append("strong_orderflow_continuation_confirmation")

        if side is SignalSide.LONG:
            confirmations.append("buyer_aggression_dominates")
        elif side is SignalSide.SHORT:
            confirmations.append("seller_aggression_dominates")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _confirmation_bonus(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._confirmation_bonus")
        bonus = 0.0

        if abs(extract_cvd_delta_ratio(snapshot)) >= self.continuation_config.strong_cvd_ratio_threshold:
            bonus += self.continuation_config.strong_cvd_bonus

        if abs(extract_volume_delta_ratio(snapshot)) >= self.continuation_config.strong_volume_delta_ratio_threshold:
            bonus += self.continuation_config.strong_volume_bonus

        if side is SignalSide.LONG:
            if extract_aggressive_buy_ratio(snapshot) >= self.continuation_config.strong_aggressive_ratio_threshold:
                bonus += self.continuation_config.strong_aggression_bonus
        elif side is SignalSide.SHORT:
            if extract_aggressive_sell_ratio(snapshot) >= self.continuation_config.strong_aggressive_ratio_threshold:
                bonus += self.continuation_config.strong_aggression_bonus

        if self._orderbook_supports_side(snapshot, side):
            bonus += self.continuation_config.orderbook_alignment_bonus

        if snapshot.directional_large_trades(side) >= self.continuation_config.min_large_aggressive_trades:
            if self.continuation_config.min_large_aggressive_trades > 0:
                bonus += self.continuation_config.large_trade_bonus

        if extract_aggressive_burst_score(snapshot) >= self.continuation_config.strong_burst_threshold:
            bonus += self.continuation_config.burst_bonus

        return bonus

    def _large_trade_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._large_trade_component")
        min_trades = max(self.continuation_config.min_large_aggressive_trades, 1)
        return unit_score(snapshot.directional_large_trades(side) / (min_trades * 3.0))

    def _orderbook_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._orderbook_component")
        imbalance = extract_orderbook_imbalance_diff(snapshot)

        if side is SignalSide.LONG:
            return unit_score((imbalance + 1.0) / 2.0)

        if side is SignalSide.SHORT:
            return unit_score((1.0 - imbalance) / 2.0)

        return 0.0

    def _notional_delta_ratio(
        self,
        snapshot: OrderflowCompositeSnapshot,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._notional_delta_ratio")
        total_notional = max(abs(snapshot.total_notional), 1.0)
        return max(-1.0, min(1.0, extract_notional_delta(snapshot) / total_notional))

    def _orderbook_supports_side(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._orderbook_supports_side")
        threshold = self.continuation_config.min_orderbook_imbalance_ratio
        imbalance = extract_orderbook_imbalance_diff(snapshot)

        if side is SignalSide.LONG:
            return imbalance >= threshold

        if side is SignalSide.SHORT:
            return imbalance <= -threshold

        return False

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OrderflowContinuationPayload) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._source_features")
        features = [
            *continuation_source_features(),
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
                    ORDERFLOW_FEATURES.AGGRESSIVE_BURST_SCORE,
                    ORDERFLOW_FEATURES.AGGRESSIVE_NET_VOLUME_DELTA,
                    ORDERFLOW_FEATURES.AGGRESSIVE_NET_NOTIONAL_DELTA,
                    ORDERFLOW_FEATURES.LARGE_BUY_TRADES,
                    ORDERFLOW_FEATURES.LARGE_SELL_TRADES,
                ]
            )

        if payload.snapshot.has_orderbook:
            features.extend(
                [
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_RATIO,
                    ORDERFLOW_FEATURES.ORDERBOOK_IMBALANCE_DIFF,
                ]
            )

        return list(dict.fromkeys(features))

    def _tags(self, payload: OrderflowContinuationPayload) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering OrderflowContinuationStrategy._tags")
        tags = [
            self.continuation_config.tag_orderflow,
            self.continuation_config.tag_orderflow_continuation,
            self.continuation_config.tag_continuation,
            self.continuation_config.tag_cvd_confirmation,
            self.continuation_config.tag_volume_delta_confirmation,
            self.continuation_config.tag_pressure,
            self.continuation_config.tag_aggressive_flow,
        ]

        if payload.side is SignalSide.LONG:
            tags.append(self.continuation_config.tag_long_continuation)

        if payload.side is SignalSide.SHORT:
            tags.append(self.continuation_config.tag_short_continuation)

        if payload.snapshot.has_orderbook:
            tags.append(self.continuation_config.tag_orderbook)

        return list(dict.fromkeys(tags))