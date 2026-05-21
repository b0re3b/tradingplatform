# trading_system/strategy/strategies/orderflow/orderflow_reversal_strategy.py

from __future__ import annotations

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
    reversal_filter_reason,
    reversal_side_from_snapshot,
    reversal_source_features,
    serialize_for_metadata,
    unit_score,
    weighted_score,
)


@dataclass(slots=True)
class OrderflowReversalPayload:
    """
    Normalized strategy-level payload для orderflow reversal.

    Source of truth:
        StrategyContext / FeatureSource.ORDERFLOW

    Preferred normalized form:
        OrderflowCompositeSnapshot

    Strategy idea:
        LONG reversal:
            price still weak / down, but CVD + volume delta + aggressive
            buyer flow already turn positive.

        SHORT reversal:
            price still strong / up, but CVD + volume delta + aggressive
            seller flow already turn negative.
    """

    snapshot: OrderflowCompositeSnapshot
    side: SignalSide

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def price_change_pct(self) -> float:
        return extract_price_change_pct(self.snapshot)

    @property
    def cvd_delta_ratio(self) -> float:
        return extract_cvd_delta_ratio(self.snapshot)

    @property
    def cvd_change_pct(self) -> float:
        return extract_cvd_change_pct(self.snapshot)

    @property
    def cvd_slope(self) -> float:
        return extract_cvd_slope(self.snapshot)

    @property
    def volume_delta_ratio(self) -> float:
        return extract_volume_delta_ratio(self.snapshot)

    @property
    def volume_delta(self) -> float:
        return extract_volume_delta(self.snapshot)

    @property
    def notional_delta(self) -> float:
        return extract_notional_delta(self.snapshot)

    @property
    def aggressive_net_notional_delta(self) -> float:
        return extract_aggressive_net_notional_delta(self.snapshot)

    @property
    def trades_count(self) -> int:
        return extract_trades_count(self.snapshot)

    @property
    def total_volume(self) -> float:
        return extract_total_volume(self.snapshot)

    @property
    def total_notional(self) -> float:
        return extract_total_notional(self.snapshot)


@dataclass(slots=True)
class OrderflowReversalStrategyConfig(OrderflowStrategyConfig):
    """
    Unified orderflow reversal strategy config.

    Strategy idea:
    - read normalized composite orderflow context from StrategyContext;
    - detect absorption / reversal pressure;
    - build internal StrategySignal only;
    - leave routing, filtering, confluence, portfolio coordination and
      risk-ready conversion to SignalProcessor.
    """

    require_fresh_snapshot: bool = True
    require_actionable_side: bool = True

    min_trades_count: int = 10
    min_total_volume: float = 0.0

    min_abs_price_change_pct: float = 0.03

    min_abs_cvd_delta_ratio: float = 0.08
    min_abs_volume_delta_ratio: float = 0.10
    min_abs_cvd_change_pct: float = 0.03

    min_aggressive_buy_ratio_for_long: float = 0.52
    min_aggressive_sell_ratio_for_short: float = 0.52

    min_bullish_imbalance_for_long: float = 0.02
    min_bearish_imbalance_for_short: float = 0.02

    require_orderbook_confirmation: bool = False
    require_aggressive_confirmation: bool = True

    min_score_for_signal: float = 0.48
    min_confidence_for_signal: float = 0.52

    price_weight: float = 0.10
    cvd_ratio_weight: float = 0.14
    volume_ratio_weight: float = 0.13
    cvd_change_weight: float = 0.10
    cvd_slope_weight: float = 0.07
    notional_weight: float = 0.10
    aggressive_notional_weight: float = 0.11
    absorption_weight: float = 0.10
    aggression_ratio_weight: float = 0.07
    large_trade_weight: float = 0.04
    orderbook_weight: float = 0.03
    burst_weight: float = 0.01

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    absorption_bonus: float = 0.05
    orderbook_flip_bonus: float = 0.04
    aggressive_confirmation_bonus: float = 0.04
    large_trade_bonus: float = 0.03
    burst_bonus: float = 0.02

    strong_absorption_threshold: float = 0.60
    strong_aggressive_ratio_threshold: float = 0.62
    strong_burst_threshold: float = 0.50

    tag_orderflow_reversal: str = "orderflow_reversal"
    tag_long_reversal: str = "long_reversal"
    tag_short_reversal: str = "short_reversal"
    tag_absorption: str = "absorption"
    tag_aggressive_flow: str = "aggressive_flow"
    tag_cvd_reversal: str = "cvd_reversal"
    tag_volume_delta_reversal: str = "volume_delta_reversal"

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    required_orderflow_features: tuple[str, ...] = (
        ORDERFLOW_FEATURES.CVD_DELTA_RATIO,
        ORDERFLOW_FEATURES.VOLUME_DELTA_RATIO,
        ORDERFLOW_FEATURES.AGGRESSIVE_BUY_RATIO,
        ORDERFLOW_FEATURES.AGGRESSIVE_SELL_RATIO,
    )

    def validate(self) -> None:
        OrderflowStrategyConfig.validate(self)

        non_negative_fields = {
            "min_abs_price_change_pct": self.min_abs_price_change_pct,
            "min_abs_cvd_delta_ratio": self.min_abs_cvd_delta_ratio,
            "min_abs_volume_delta_ratio": self.min_abs_volume_delta_ratio,
            "min_abs_cvd_change_pct": self.min_abs_cvd_change_pct,
            "min_bullish_imbalance_for_long": self.min_bullish_imbalance_for_long,
            "min_bearish_imbalance_for_short": self.min_bearish_imbalance_for_short,
            "min_score_for_signal": self.min_score_for_signal,
            "absorption_bonus": self.absorption_bonus,
            "orderbook_flip_bonus": self.orderbook_flip_bonus,
            "aggressive_confirmation_bonus": self.aggressive_confirmation_bonus,
            "large_trade_bonus": self.large_trade_bonus,
            "burst_bonus": self.burst_bonus,
            "strong_absorption_threshold": self.strong_absorption_threshold,
            "strong_aggressive_ratio_threshold": self.strong_aggressive_ratio_threshold,
            "strong_burst_threshold": self.strong_burst_threshold,
        }

        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        unit_fields = {
            "min_aggressive_buy_ratio_for_long": self.min_aggressive_buy_ratio_for_long,
            "min_aggressive_sell_ratio_for_short": self.min_aggressive_sell_ratio_for_short,
            "min_confidence_for_signal": self.min_confidence_for_signal,
            "absorption_bonus": self.absorption_bonus,
            "orderbook_flip_bonus": self.orderbook_flip_bonus,
            "aggressive_confirmation_bonus": self.aggressive_confirmation_bonus,
            "large_trade_bonus": self.large_trade_bonus,
            "burst_bonus": self.burst_bonus,
            "strong_absorption_threshold": self.strong_absorption_threshold,
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

        score_weights = {
            "price_weight": self.price_weight,
            "cvd_ratio_weight": self.cvd_ratio_weight,
            "volume_ratio_weight": self.volume_ratio_weight,
            "cvd_change_weight": self.cvd_change_weight,
            "cvd_slope_weight": self.cvd_slope_weight,
            "notional_weight": self.notional_weight,
            "aggressive_notional_weight": self.aggressive_notional_weight,
            "absorption_weight": self.absorption_weight,
            "aggression_ratio_weight": self.aggression_ratio_weight,
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
            "tag_orderflow_reversal",
            "tag_long_reversal",
            "tag_short_reversal",
            "tag_absorption",
            "tag_aggressive_flow",
            "tag_cvd_reversal",
            "tag_volume_delta_reversal",
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


class OrderflowReversalStrategy(OrderflowTradingStrategy):
    """
    Unified orderflow reversal strategy.

    Input:
        StrategyContext with FeatureSource.ORDERFLOW domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.orderflow.reversal"
    category: StrategyCategory = StrategyCategory.ORDERFLOW
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        orderflow_config: OrderflowReversalStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_orderflow_config = (
            orderflow_config or OrderflowReversalStrategyConfig()
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

        self.reversal_config: OrderflowReversalStrategyConfig = (
            resolved_orderflow_config
        )

    @property
    def strategy_name(self) -> str:
        return "orderflow_reversal"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.ORDERFLOW,
            timeframe=Timeframe.M1,
            tags=[
                self.reversal_config.tag_orderflow,
                self.reversal_config.tag_orderflow_reversal,
                self.reversal_config.tag_reversal,
                self.reversal_config.tag_absorption,
                self.reversal_config.tag_aggressive_flow,
                self.reversal_config.tag_cvd_reversal,
                self.reversal_config.tag_volume_delta_reversal,
                "analytics_orderflow",
            ],
            version="2.0.0",
            description=(
                "Detects orderflow reversal / absorption using price weakness or "
                "strength against CVD, volume delta, aggressive trades and "
                "optional orderbook imbalance from normalized StrategyContext."
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
                "strategy_type": "orderflow_reversal",
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
        base_required = super().required_features()
        return set(base_required).union(
            self.reversal_config.required_orderflow_features
        )

    async def generate_signal(
            self,
            context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        required_features = tuple(self.reversal_config.required_orderflow_features)

        if not self.has_any_orderflow_data(context, required_features):
            self.remember_no_signal(
                "missing_orderflow_reversal_contract",
                orderflow_domain_keys=sorted(self.orderflow_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_orderflow_features(context, required_features):
            self.remember_no_signal(
                "stale_orderflow_reversal_features",
                required_features=sorted(required_features),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            self.remember_no_signal(
                "orderflow_reversal_payload_not_resolved",
                orderflow_domain=self.orderflow_domain(context),
                required_features=sorted(self.required_features()),
            )
            return None

        if (
                self.reversal_config.require_fresh_snapshot
                and is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
        )
        ):
            self.remember_no_signal(
                "stale_orderflow_reversal_snapshot",
                event_time=payload.event_time.isoformat() if payload.event_time else None,
                context_timestamp=context.timestamp.isoformat(),
                stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
            )
            return None

        common_rejection = reversal_filter_reason(
            payload.snapshot,
            min_trades_count=self.reversal_config.min_trades_count,
            min_total_volume=self.reversal_config.min_total_volume,
            min_abs_price_change_pct=self.reversal_config.min_abs_price_change_pct,
            min_abs_cvd_delta_ratio=self.reversal_config.min_abs_cvd_delta_ratio,
            min_abs_volume_delta_ratio=self.reversal_config.min_abs_volume_delta_ratio,
            min_aggressive_buy_ratio_for_long=(
                self.reversal_config.min_aggressive_buy_ratio_for_long
            ),
            min_aggressive_sell_ratio_for_short=(
                self.reversal_config.min_aggressive_sell_ratio_for_short
            ),
            require_aggressive_confirmation=(
                self.reversal_config.require_aggressive_confirmation
            ),
            require_orderbook_confirmation=(
                self.reversal_config.require_orderbook_confirmation
            ),
        )
        if common_rejection is not None:
            self.remember_no_signal(
                "orderflow_reversal_quality_filter_failed",
                filter_reason=common_rejection,
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
                trades_count=payload.trades_count,
                total_volume=payload.total_volume,
                total_notional=payload.total_notional,
                price_change_pct=payload.price_change_pct,
                cvd_delta_ratio=payload.cvd_delta_ratio,
                volume_delta_ratio=payload.volume_delta_ratio,
                notional_delta=payload.notional_delta,
                aggressive_net_notional_delta=payload.aggressive_net_notional_delta,
            )
            return None

        side = payload.side
        if (
                self.reversal_config.require_actionable_side
                and not is_directional_side(side)
        ):
            self.remember_no_signal(
                "orderflow_reversal_side_not_directional",
                side=serialize_for_metadata(side),
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        min_score = max(
            self.reversal_config.min_signal_score,
            self.reversal_config.min_score_for_signal,
        )
        if breakdown.score < min_score:
            self.remember_no_signal(
                "orderflow_reversal_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_score=min_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        min_confidence = max(
            self.reversal_config.min_signal_confidence,
            self.reversal_config.min_confidence_for_signal,
        )
        if breakdown.confidence < min_confidence:
            self.remember_no_signal(
                "orderflow_reversal_confidence_below_minimum",
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
                    "orderflow_reversal_signal",
                    f"side:{side.value}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "orderflow_setup_family": "orderflow_reversal",
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
            "reversal_side": side.value,
            "price_change_pct": payload.price_change_pct,
            "cvd_delta_ratio": payload.cvd_delta_ratio,
            "cvd_change_pct": payload.cvd_change_pct,
            "cvd_slope": payload.cvd_slope,
            "volume_delta_ratio": payload.volume_delta_ratio,
            "volume_delta": payload.volume_delta,
            "notional_delta": payload.notional_delta,
            "aggressive_net_notional_delta": payload.aggressive_net_notional_delta,
            "aggressive_net_volume_delta": extract_aggressive_net_volume_delta(
                payload.snapshot
            ),
            "aggressive_buy_ratio": extract_aggressive_buy_ratio(payload.snapshot),
            "aggressive_sell_ratio": extract_aggressive_sell_ratio(payload.snapshot),
            "aggressive_burst_score": extract_aggressive_burst_score(payload.snapshot),
            "large_buy_trades": extract_large_buy_trades(payload.snapshot),
            "large_sell_trades": extract_large_sell_trades(payload.snapshot),
            "orderbook_imbalance_ratio": extract_orderbook_imbalance_ratio(
                payload.snapshot
            ),
            "orderbook_imbalance_diff": extract_orderbook_imbalance_diff(
                payload.snapshot
            ),
            "cumulative_volume_delta": extract_cumulative_volume_delta(
                payload.snapshot
            ),
            "cumulative_notional_delta": extract_cumulative_notional_delta(
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
            setup_type=self.reversal_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.reversal_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> OrderflowReversalPayload | None:
        snapshot = self.resolve_orderflow_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        if snapshot.trades_count < self.reversal_config.min_trades_count:
            return None

        if snapshot.total_volume < self.reversal_config.min_total_volume:
            return None

        side = reversal_side_from_snapshot(
            snapshot,
            min_abs_price_change_pct=self.reversal_config.min_abs_price_change_pct,
            min_abs_cvd_delta_ratio=self.reversal_config.min_abs_cvd_delta_ratio,
            min_abs_volume_delta_ratio=self.reversal_config.min_abs_volume_delta_ratio,
            min_aggressive_buy_ratio_for_long=(
                self.reversal_config.min_aggressive_buy_ratio_for_long
            ),
            min_aggressive_sell_ratio_for_short=(
                self.reversal_config.min_aggressive_sell_ratio_for_short
            ),
            require_aggressive_confirmation=(
                self.reversal_config.require_aggressive_confirmation
            ),
            require_orderbook_confirmation=(
                self.reversal_config.require_orderbook_confirmation
            ),
            min_bullish_imbalance_for_long=(
                self.reversal_config.min_bullish_imbalance_for_long
            ),
            min_bearish_imbalance_for_short=(
                self.reversal_config.min_bearish_imbalance_for_short
            ),
        )
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "long_orderflow_reversal"
            if side is SignalSide.LONG
            else "short_orderflow_reversal",
            f"price_change_pct:{extract_price_change_pct(snapshot):.6f}",
            f"cvd_delta_ratio:{extract_cvd_delta_ratio(snapshot):.6f}",
            f"volume_delta_ratio:{extract_volume_delta_ratio(snapshot):.6f}",
            f"notional_delta:{extract_notional_delta(snapshot):.6f}",
            f"aggressive_net_notional_delta:{extract_aggressive_net_notional_delta(snapshot):.6f}",
        ]

        return OrderflowReversalPayload(
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
        payload: OrderflowReversalPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot
        side = payload.side

        components = {
            "price": percent_score(
                abs(payload.price_change_pct),
                scale=max(self.reversal_config.min_abs_price_change_pct * 4.0, 0.01),
            ),
            "cvd_ratio": ratio_score(
                abs(payload.cvd_delta_ratio),
                scale=max(self.reversal_config.min_abs_cvd_delta_ratio * 4.0, 0.01),
            ),
            "volume_ratio": ratio_score(
                abs(payload.volume_delta_ratio),
                scale=max(self.reversal_config.min_abs_volume_delta_ratio * 4.0, 0.01),
            ),
            "cvd_change": percent_score(
                abs(payload.cvd_change_pct),
                scale=max(self.reversal_config.min_abs_cvd_change_pct * 4.0, 0.01),
            ),
            "cvd_slope": magnitude_score(
                abs(payload.cvd_slope),
                scale=10.0,
            ),
            "notional": ratio_score(
                abs(self._notional_delta_ratio(snapshot)),
                scale=0.40,
            ),
            "aggressive_notional": ratio_score(
                abs(snapshot.directional_aggressive_notional_delta(side)),
                scale=max(abs(snapshot.total_notional), 1.0),
            ),
            "absorption": self._absorption_component(snapshot, side),
            "aggression_ratio": (
                extract_aggressive_buy_ratio(snapshot)
                if side is SignalSide.LONG
                else extract_aggressive_sell_ratio(snapshot)
            ),
            "large_trades": self._large_trade_component(snapshot, side),
            "orderbook": self._orderbook_component(snapshot, side),
            "burst": ratio_score(
                extract_aggressive_burst_score(snapshot),
                scale=1.0,
            ),
        }

        weights = {
            "price": self.reversal_config.price_weight,
            "cvd_ratio": self.reversal_config.cvd_ratio_weight,
            "volume_ratio": self.reversal_config.volume_ratio_weight,
            "cvd_change": self.reversal_config.cvd_change_weight,
            "cvd_slope": self.reversal_config.cvd_slope_weight,
            "notional": self.reversal_config.notional_weight,
            "aggressive_notional": self.reversal_config.aggressive_notional_weight,
            "absorption": self.reversal_config.absorption_weight,
            "aggression_ratio": self.reversal_config.aggression_ratio_weight,
            "large_trades": self.reversal_config.large_trade_weight,
            "orderbook": self.reversal_config.orderbook_weight,
            "burst": self.reversal_config.burst_weight,
        }

        primary = weighted_score(
            {
                "cvd": components["cvd_ratio"],
                "volume_delta": components["volume_ratio"],
                "notional": components["notional"],
                "absorption": components["absorption"],
            },
            {
                "cvd": 0.30,
                "volume_delta": 0.25,
                "notional": 0.20,
                "absorption": 0.25,
            },
        )
        context_component = weighted_score(
            {
                "aggression_ratio": components["aggression_ratio"],
                "large_trades": components["large_trades"],
                "orderbook": components["orderbook"],
                "burst": components["burst"],
            },
            {
                "aggression_ratio": 0.40,
                "large_trades": 0.20,
                "orderbook": 0.25,
                "burst": 0.15,
            },
        )
        confirmation_component = weighted_score(
            {
                "cvd_change": components["cvd_change"],
                "aggressive_notional": components["aggressive_notional"],
                "absorption": components["absorption"],
            },
            {
                "cvd_change": 0.30,
                "aggressive_notional": 0.35,
                "absorption": 0.35,
            },
        )
        fresh_score = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.reversal_config.stale_feature_max_age_seconds,
        )

        score = weighted_score(components, weights, default=primary)
        confidence = confidence_from_components(
            primary=primary,
            context=context_component,
            confirmation=confirmation_component,
            freshness=fresh_score,
            primary_weight=self.reversal_config.confidence_primary_weight,
            context_weight=self.reversal_config.confidence_context_weight,
            confirmation_weight=self.reversal_config.confidence_confirmation_weight,
            freshness_weight=self.reversal_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "orderflow_absorption_reversal",
            f"side:{side.value}",
            f"cvd_delta_ratio:{payload.cvd_delta_ratio:.6f}",
            f"volume_delta_ratio:{payload.volume_delta_ratio:.6f}",
        ]

        bonus = self._confirmation_bonus(snapshot, side)
        if bonus > 0:
            score += bonus
            confidence += min(0.06, bonus)
            confirmations.append("strong_orderflow_reversal_confirmation")

        if side is SignalSide.LONG:
            confirmations.append("buyer_absorption_detected")
        elif side is SignalSide.SHORT:
            confirmations.append("seller_absorption_detected")

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
        bonus = 0.0

        if self._absorption_component(snapshot, side) >= self.reversal_config.strong_absorption_threshold:
            bonus += self.reversal_config.absorption_bonus

        if self._orderbook_supports_side(snapshot, side):
            bonus += self.reversal_config.orderbook_flip_bonus

        if side is SignalSide.LONG:
            if extract_aggressive_buy_ratio(snapshot) >= self.reversal_config.strong_aggressive_ratio_threshold:
                bonus += self.reversal_config.aggressive_confirmation_bonus
        elif side is SignalSide.SHORT:
            if extract_aggressive_sell_ratio(snapshot) >= self.reversal_config.strong_aggressive_ratio_threshold:
                bonus += self.reversal_config.aggressive_confirmation_bonus

        if snapshot.directional_large_trades(side) > 0:
            bonus += self.reversal_config.large_trade_bonus

        if extract_aggressive_burst_score(snapshot) >= self.reversal_config.strong_burst_threshold:
            bonus += self.reversal_config.burst_bonus

        return bonus

    def _absorption_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
        """
        Absorption score:
        - LONG: price down, but signed delta / notional / aggressive flow turn up.
        - SHORT: price up, but signed delta / notional / aggressive flow turn down.
        """
        price = extract_price_change_pct(snapshot)
        cvd_delta = extract_cvd_delta_ratio(snapshot)
        volume_delta = extract_volume_delta_ratio(snapshot)
        notional_delta = extract_notional_delta(snapshot)
        aggressive_delta = extract_aggressive_net_notional_delta(snapshot)

        if side is SignalSide.LONG:
            return weighted_score(
                {
                    "price_weakness": unit_score(abs(min(0.0, price))),
                    "cvd_turn": unit_score(max(0.0, cvd_delta)),
                    "volume_turn": unit_score(max(0.0, volume_delta)),
                    "notional_turn": unit_score(max(0.0, notional_delta) / max(abs(snapshot.total_notional), 1.0)),
                    "aggressive_turn": unit_score(max(0.0, aggressive_delta) / max(abs(snapshot.total_notional), 1.0)),
                },
                {
                    "price_weakness": 0.20,
                    "cvd_turn": 0.25,
                    "volume_turn": 0.20,
                    "notional_turn": 0.15,
                    "aggressive_turn": 0.20,
                },
            )

        if side is SignalSide.SHORT:
            return weighted_score(
                {
                    "price_strength": unit_score(max(0.0, price)),
                    "cvd_turn": unit_score(abs(min(0.0, cvd_delta))),
                    "volume_turn": unit_score(abs(min(0.0, volume_delta))),
                    "notional_turn": unit_score(abs(min(0.0, notional_delta)) / max(abs(snapshot.total_notional), 1.0)),
                    "aggressive_turn": unit_score(abs(min(0.0, aggressive_delta)) / max(abs(snapshot.total_notional), 1.0)),
                },
                {
                    "price_strength": 0.20,
                    "cvd_turn": 0.25,
                    "volume_turn": 0.20,
                    "notional_turn": 0.15,
                    "aggressive_turn": 0.20,
                },
            )

        return 0.0

    def _large_trade_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
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

    def _orderbook_component(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> float:
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
        total_notional = max(abs(snapshot.total_notional), 1.0)
        return max(-1.0, min(1.0, extract_notional_delta(snapshot) / total_notional))

    def _orderbook_supports_side(
        self,
        snapshot: OrderflowCompositeSnapshot,
        side: SignalSide,
    ) -> bool:
        imbalance = extract_orderbook_imbalance_diff(snapshot)

        if side is SignalSide.LONG:
            return imbalance >= self.reversal_config.min_bullish_imbalance_for_long

        if side is SignalSide.SHORT:
            return imbalance <= -self.reversal_config.min_bearish_imbalance_for_short

        return False

    # ------------------------------------------------------------------
    # Source features / tags
    # ------------------------------------------------------------------

    def _source_features(self, payload: OrderflowReversalPayload) -> list[str]:
        features = [
            *reversal_source_features(),
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

    def _tags(self, payload: OrderflowReversalPayload) -> list[str]:
        tags = [
            self.reversal_config.tag_orderflow,
            self.reversal_config.tag_orderflow_reversal,
            self.reversal_config.tag_reversal,
            self.reversal_config.tag_absorption,
            self.reversal_config.tag_aggressive_flow,
            self.reversal_config.tag_cvd_reversal,
            self.reversal_config.tag_volume_delta_reversal,
        ]

        if payload.side is SignalSide.LONG:
            tags.append(self.reversal_config.tag_long_reversal)

        if payload.side is SignalSide.SHORT:
            tags.append(self.reversal_config.tag_short_reversal)

        if payload.snapshot.has_orderbook:
            tags.append(self.reversal_config.tag_orderbook)

        return list(dict.fromkeys(tags))