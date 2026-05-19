# trading_system/strategy/strategies/whales/whale_accumulation_strategy.py

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
    WHALES_FEATURES,
    WhaleCompositeSnapshot,
    WhalesStrategyConfig,
    WhalesTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    average_score,
    cluster_context_score,
    confidence_from_components,
    continuation_context_score,
    extract_event_time,
    freshness_score,
    is_directional_side,
    is_stale,
    large_trade_score,
    low_exhaustion_score,
    normalize_label,
    serialize_for_metadata,
    side_label_to_signal_side,
    unit_score,
    whale_accumulation_source_features,
    whale_activity_score,
    whale_pressure_score,
    weighted_score,
)


@dataclass(slots=True)
class WhaleAccumulationPayload:
    """
    Normalized strategy-level payload для whale accumulation.

    Direction convention:
    - buy-side accumulation -> LONG.
    """

    snapshot: WhaleCompositeSnapshot
    side: SignalSide
    accumulation_side: str

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WhaleAccumulationStrategyConfig(WhalesStrategyConfig):
    """
    Unified whale accumulation strategy config.

    Strategy idea:
    - buy-side whale activity наростає до breakout;
    - pressure/cluster підтримують bid-side accumulation;
    - continuation probability достатня;
    - exhaustion probability низька;
    - strategy returns internal StrategySignal only.
    """

    min_accumulation_score: float = 0.62
    min_accumulation_confidence: float = 0.56

    min_activity_notional: float = 250_000.0
    min_activity_trade_count: int = 3
    min_pressure_imbalance_ratio: float = 0.58
    min_pressure_score: float = 0.48
    min_cluster_score: float = 0.50
    min_continuation_probability: float = 0.55
    max_exhaustion_probability: float = 0.50

    min_large_trade_notional: float = 200_000.0
    min_large_trade_zscore: float = 1.5

    require_buy_side: bool = True
    require_activity_confirmation: bool = True
    require_pressure_confirmation: bool = True
    require_cluster_confirmation: bool = False
    require_continuation_confirmation: bool = False

    use_large_trade_confirmation: bool = True
    block_sell_side_pressure: bool = True
    block_high_exhaustion_probability: bool = True

    require_activity_same_side: bool = True
    require_pressure_same_side: bool = True
    require_cluster_same_side: bool = False
    require_large_trade_same_side: bool = False

    score_activity_weight: float = 0.28
    score_pressure_weight: float = 0.24
    score_cluster_weight: float = 0.16
    score_continuation_weight: float = 0.14
    score_large_trade_weight: float = 0.08
    score_exhaustion_weight: float = 0.05
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_activity_bonus: float = 0.04
    strong_pressure_bonus: float = 0.04
    cluster_confirmation_bonus: float = 0.04
    continuation_confirmation_bonus: float = 0.04
    large_trade_confirmation_bonus: float = 0.03
    low_exhaustion_bonus: float = 0.03

    strong_activity_multiplier: float = 2.0
    strong_pressure_threshold: float = 0.72
    strong_continuation_threshold: float = 0.72
    low_exhaustion_threshold: float = 0.30

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.CONTINUATION

    tag_whale_accumulation: str = "whale_accumulation"
    tag_buy_side_accumulation: str = "buy_side_accumulation"
    tag_pre_breakout_positioning: str = "pre_breakout_positioning"
    tag_cluster_support: str = "cluster_support"
    tag_continuation_bias: str = "continuation_bias"
    tag_low_exhaustion: str = "low_exhaustion"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    accumulation_rr_hint: float | None = None

    required_whales_features: tuple[str, ...] = (
        WHALES_FEATURES.ACTIVITY,
        WHALES_FEATURES.PRESSURE,
    )

    def validate(self) -> None:
        WhalesStrategyConfig.validate(self)

        unit_fields = {
            "min_accumulation_score": self.min_accumulation_score,
            "min_accumulation_confidence": self.min_accumulation_confidence,
            "min_pressure_imbalance_ratio": self.min_pressure_imbalance_ratio,
            "min_pressure_score": self.min_pressure_score,
            "min_cluster_score": self.min_cluster_score,
            "min_continuation_probability": self.min_continuation_probability,
            "max_exhaustion_probability": self.max_exhaustion_probability,
            "strong_activity_bonus": self.strong_activity_bonus,
            "strong_pressure_bonus": self.strong_pressure_bonus,
            "cluster_confirmation_bonus": self.cluster_confirmation_bonus,
            "continuation_confirmation_bonus": self.continuation_confirmation_bonus,
            "large_trade_confirmation_bonus": self.large_trade_confirmation_bonus,
            "low_exhaustion_bonus": self.low_exhaustion_bonus,
            "strong_pressure_threshold": self.strong_pressure_threshold,
            "strong_continuation_threshold": self.strong_continuation_threshold,
            "low_exhaustion_threshold": self.low_exhaustion_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_activity_notional": self.min_activity_notional,
            "min_large_trade_notional": self.min_large_trade_notional,
            "min_large_trade_zscore": self.min_large_trade_zscore,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if self.min_activity_trade_count < 0:
            raise StrategyConfigError("min_activity_trade_count must be >= 0")

        if self.strong_activity_multiplier < 0:
            raise StrategyConfigError("strong_activity_multiplier must be >= 0")

        score_weights = {
            "score_activity_weight": self.score_activity_weight,
            "score_pressure_weight": self.score_pressure_weight,
            "score_cluster_weight": self.score_cluster_weight,
            "score_continuation_weight": self.score_continuation_weight,
            "score_large_trade_weight": self.score_large_trade_weight,
            "score_exhaustion_weight": self.score_exhaustion_weight,
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

        hint_fields = {
            "execution_entry_offset_bps_hint": self.execution_entry_offset_bps_hint,
            "execution_stop_buffer_bps_hint": self.execution_stop_buffer_bps_hint,
            "execution_take_profit_bps_hint": self.execution_take_profit_bps_hint,
            "accumulation_rr_hint": self.accumulation_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_whale_accumulation",
            "tag_buy_side_accumulation",
            "tag_pre_breakout_positioning",
            "tag_cluster_support",
            "tag_continuation_bias",
            "tag_low_exhaustion",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_whales_features:
            raise StrategyConfigError("required_whales_features cannot be empty")


class WhaleAccumulationStrategy(WhalesTradingStrategy):
    """
    Unified whale accumulation strategy.

    Input:
        StrategyContext with FeatureSource.WHALES domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.whales.accumulation"
    category: StrategyCategory = StrategyCategory.WHALES
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        whales_config: WhaleAccumulationStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_whales_config = whales_config or WhaleAccumulationStrategyConfig()
        resolved_whales_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            whales_config=resolved_whales_config,
            service_name=service_name,
        )

        self.accumulation_config: WhaleAccumulationStrategyConfig = (
            resolved_whales_config
        )

    @property
    def strategy_name(self) -> str:
        return "whale_accumulation"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.WHALES,
            timeframe=Timeframe.M1,
            tags=[
                self.accumulation_config.tag_whales,
                self.accumulation_config.tag_accumulation,
                self.accumulation_config.tag_whale_accumulation,
                self.accumulation_config.tag_buy_side_accumulation,
                self.accumulation_config.tag_pre_breakout_positioning,
                "analytics_whales",
            ],
            version="2.0.0",
            description=(
                "Interprets buy-side whale accumulation / pre-breakout positioning "
                "from normalized StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.TRENDING_UP,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.whales",
                "strategy_type": "whale_accumulation",
                "base_class": "WhalesTradingStrategy",
                "canonical_payload": "WhaleCompositeSnapshot",
                "uses_activity": True,
                "uses_pressure": True,
                "uses_cluster": True,
                "uses_continuation_probability": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(
            self.accumulation_config.required_whales_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_whales_data(
            context,
            tuple(self.accumulation_config.required_whales_features),
        ):
            return None

        if self.has_stale_whales_features(
            context,
            tuple(self.accumulation_config.required_whales_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.accumulation_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self.accepts_whale_snapshot(
            payload.snapshot,
            require_futures_market_type=self.accumulation_config.require_futures_market_type,
            min_confidence=self.accumulation_config.min_confidence,
        ):
            return None

        if not self._passes_accumulation_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        rejection = self.whale_quality_rejection_reason(
            snapshot=payload.snapshot,
            score=breakdown.score,
            confidence=breakdown.confidence,
            required_inputs=self._required_inputs(),
        )
        if rejection is not None:
            return None

        if breakdown.score < self.accumulation_config.min_accumulation_score:
            return None

        if breakdown.confidence < self.accumulation_config.min_accumulation_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "whale_accumulation_signal",
                    f"side:{payload.side.value}",
                    f"accumulation_side:{payload.accumulation_side}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "whales_setup_family": "whale_accumulation",
            "whales_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "accumulation_side": payload.accumulation_side,
            "dominant_side": payload.snapshot.dominant_side,
            "whale_side": payload.snapshot.whale_side,
            "cluster_side": payload.snapshot.cluster_side,
            "imbalance_ratio": payload.snapshot.imbalance_ratio,
            "pressure_score": payload.snapshot.pressure_score,
            "cluster_score": payload.snapshot.cluster_score,
            "continuation_probability": payload.snapshot.continuation_probability,
            "exhaustion_probability": payload.snapshot.exhaustion_probability,
            "total_notional": payload.snapshot.total_notional,
            "trade_count": payload.snapshot.trade_count,
            "large_trade_notional": payload.snapshot.large_trade_notional,
            "large_trade_zscore": payload.snapshot.large_trade_zscore,
            "reference_price": payload.snapshot.reference_price,
            "execution_hints": self._execution_hints(),
        }

        return self.build_whale_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.accumulation_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.accumulation_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> WhaleAccumulationPayload | None:
        snapshot = self.resolve_whale_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        accumulation_side = self._resolve_accumulation_side(snapshot)
        if accumulation_side != "buy":
            return None

        side = side_label_to_signal_side(accumulation_side)
        if not is_directional_side(side):
            return None

        event_time = (
            extract_event_time(snapshot.activity)
            or extract_event_time(snapshot.pressure)
            or extract_event_time(snapshot.cluster_update)
            or extract_event_time(snapshot.cluster)
            or extract_event_time(snapshot.large_trade)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "whale_accumulation_context",
            f"accumulation_side:{accumulation_side}",
            f"imbalance_ratio:{snapshot.imbalance_ratio:.4f}",
            f"continuation_probability:{snapshot.continuation_probability}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        return WhaleAccumulationPayload(
            snapshot=snapshot,
            side=side,
            accumulation_side=accumulation_side,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw=snapshot.inputs(),
        )

    def _resolve_accumulation_side(
        self,
        snapshot: WhaleCompositeSnapshot,
    ) -> str:
        for candidate in (
            snapshot.whale_side,
            snapshot.dominant_side,
            snapshot.cluster_side,
        ):
            label = normalize_label(candidate)
            if label in {"buy", "sell"}:
                return label

        activity_side = normalize_label(
            snapshot.activity.get("side")
            or snapshot.activity.get("dominant_side")
        )
        if activity_side in {"buy", "sell"}:
            return activity_side

        large_trade_side = normalize_label(
            snapshot.large_trade.get("side")
            or snapshot.large_trade.get("trade_side")
        )
        if large_trade_side in {"buy", "sell"}:
            return large_trade_side

        return "unknown"

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _required_inputs(self) -> tuple[str, ...]:
        required = []

        if self.accumulation_config.require_activity_confirmation:
            required.append("activity")

        if self.accumulation_config.require_pressure_confirmation:
            required.append("pressure")

        if self.accumulation_config.require_cluster_confirmation:
            required.append("cluster")

        return tuple(required)

    def _passes_accumulation_filters(
        self,
        payload: WhaleAccumulationPayload,
    ) -> bool:
        snapshot = payload.snapshot

        if self.accumulation_config.require_buy_side:
            if payload.accumulation_side != "buy":
                return False

        if self.accumulation_config.block_sell_side_pressure:
            if snapshot.dominant_side == "sell" or snapshot.whale_side == "sell":
                return False

        if self.accumulation_config.require_activity_confirmation:
            if not snapshot.has_activity:
                return False

            if snapshot.total_notional < self.accumulation_config.min_activity_notional:
                return False

            if snapshot.trade_count < self.accumulation_config.min_activity_trade_count:
                return False

        if self.accumulation_config.require_pressure_confirmation:
            if not snapshot.has_pressure:
                return False

            if snapshot.imbalance_ratio < self.accumulation_config.min_pressure_imbalance_ratio:
                return False

            if snapshot.pressure_score < self.accumulation_config.min_pressure_score:
                return False

        if self.accumulation_config.require_cluster_confirmation:
            if not snapshot.has_cluster:
                return False

            cluster_score = snapshot.cluster_score or 0.0
            if cluster_score < self.accumulation_config.min_cluster_score:
                return False

        if self.accumulation_config.require_continuation_confirmation:
            continuation_probability = snapshot.continuation_probability or 0.0
            if continuation_probability < self.accumulation_config.min_continuation_probability:
                return False

        if self.accumulation_config.block_high_exhaustion_probability:
            exhaustion_probability = snapshot.exhaustion_probability
            if exhaustion_probability is not None:
                if exhaustion_probability > self.accumulation_config.max_exhaustion_probability:
                    return False

        if self.accumulation_config.require_activity_same_side and snapshot.has_activity:
            activity_side = normalize_label(
                snapshot.activity.get("side")
                or snapshot.activity.get("dominant_side")
            )
            if activity_side in {"buy", "sell"} and activity_side != payload.accumulation_side:
                return False

        if self.accumulation_config.require_pressure_same_side and snapshot.has_pressure:
            if snapshot.dominant_side in {"buy", "sell"}:
                if snapshot.dominant_side != payload.accumulation_side:
                    return False

        if self.accumulation_config.require_cluster_same_side and snapshot.has_cluster:
            if snapshot.cluster_side in {"buy", "sell"}:
                if snapshot.cluster_side != payload.accumulation_side:
                    return False

        if self.accumulation_config.require_large_trade_same_side and snapshot.has_large_trade:
            large_trade_side = normalize_label(
                snapshot.large_trade.get("side")
                or snapshot.large_trade.get("trade_side")
            )
            if large_trade_side in {"buy", "sell"} and large_trade_side != payload.accumulation_side:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: WhaleAccumulationPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot
        inputs = snapshot.inputs()

        activity_component = whale_activity_score(
            snapshot.activity,
            min_notional=self.accumulation_config.min_activity_notional,
            min_trade_count=self.accumulation_config.min_activity_trade_count,
        )
        pressure_component = whale_pressure_score(
            snapshot.pressure,
            min_imbalance_ratio=self.accumulation_config.min_pressure_imbalance_ratio,
        )
        cluster_component = cluster_context_score(
            inputs,
            min_cluster_score=self.accumulation_config.min_cluster_score,
        )
        continuation_component = continuation_context_score(
            inputs,
            min_continuation_probability=self.accumulation_config.min_continuation_probability,
        )
        large_trade_component = (
            large_trade_score(
                snapshot.large_trade,
                min_notional=self.accumulation_config.min_large_trade_notional,
                min_zscore=self.accumulation_config.min_large_trade_zscore,
            )
            if self.accumulation_config.use_large_trade_confirmation
            else 0.0
        )
        exhaustion_component = low_exhaustion_score(
            inputs,
            max_exhaustion_probability=self.accumulation_config.max_exhaustion_probability,
        )
        freshness_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.accumulation_config.stale_feature_max_age_seconds,
        )

        components = {
            "activity": activity_component,
            "pressure": pressure_component,
            "cluster": cluster_component,
            "continuation": continuation_component,
            "large_trade": large_trade_component,
            "exhaustion": exhaustion_component,
            "freshness": freshness_component,
        }
        weights = {
            "activity": self.accumulation_config.score_activity_weight,
            "pressure": self.accumulation_config.score_pressure_weight,
            "cluster": self.accumulation_config.score_cluster_weight,
            "continuation": self.accumulation_config.score_continuation_weight,
            "large_trade": self.accumulation_config.score_large_trade_weight,
            "exhaustion": self.accumulation_config.score_exhaustion_weight,
            "freshness": self.accumulation_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=activity_component)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, activity_component, pressure_component),
            context=average_score(cluster_component, continuation_component),
            confirmation=average_score(large_trade_component, exhaustion_component),
            freshness=freshness_component,
            primary_weight=self.accumulation_config.confidence_primary_weight,
            context_weight=self.accumulation_config.confidence_context_weight,
            confirmation_weight=self.accumulation_config.confidence_confirmation_weight,
            freshness_weight=self.accumulation_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "whale_accumulation_context",
            f"accumulation_side:{payload.accumulation_side}",
            f"side:{payload.side.value}",
        ]

        if snapshot.total_notional >= (
            self.accumulation_config.min_activity_notional
            * self.accumulation_config.strong_activity_multiplier
        ):
            score += self.accumulation_config.strong_activity_bonus
            confirmations.append("strong_whale_activity")

        if snapshot.imbalance_ratio >= self.accumulation_config.strong_pressure_threshold:
            score += self.accumulation_config.strong_pressure_bonus
            confirmations.append("strong_buy_side_pressure")

        if cluster_component >= 1.0:
            score += self.accumulation_config.cluster_confirmation_bonus
            confirmations.append("cluster_supports_accumulation")

        if (
            snapshot.continuation_probability is not None
            and snapshot.continuation_probability >= self.accumulation_config.strong_continuation_threshold
        ):
            score += self.accumulation_config.continuation_confirmation_bonus
            confirmations.append("continuation_bias_confirmed")

        if large_trade_component > 0.0:
            score += self.accumulation_config.large_trade_confirmation_bonus
            confirmations.append("large_trade_accumulation_confirmation")

        if (
            snapshot.exhaustion_probability is None
            or snapshot.exhaustion_probability <= self.accumulation_config.low_exhaustion_threshold
        ):
            score += self.accumulation_config.low_exhaustion_bonus
            confirmations.append("low_exhaustion_risk")

        if snapshot.dominant_side == payload.accumulation_side:
            confirmations.append("pressure_same_as_accumulation_side")

        if snapshot.cluster_side == payload.accumulation_side:
            confirmations.append("cluster_same_as_accumulation_side")

        activity_side = normalize_label(
            snapshot.activity.get("side")
            or snapshot.activity.get("dominant_side")
        )
        if activity_side == payload.accumulation_side:
            confirmations.append("activity_same_as_accumulation_side")

        large_trade_side = normalize_label(
            snapshot.large_trade.get("side")
            or snapshot.large_trade.get("trade_side")
        )
        if large_trade_side == payload.accumulation_side:
            confirmations.append("large_trade_same_as_accumulation_side")

        if snapshot.total_notional > 0:
            reasons.append(f"total_notional:{snapshot.total_notional:.2f}")

        if snapshot.trade_count > 0:
            reasons.append(f"trade_count:{snapshot.trade_count}")

        if snapshot.large_trade_notional > 0:
            reasons.append(f"large_trade_notional:{snapshot.large_trade_notional:.2f}")

        if snapshot.continuation_probability is not None:
            reasons.append(
                f"continuation_probability:{snapshot.continuation_probability:.4f}"
            )

        if snapshot.exhaustion_probability is not None:
            reasons.append(
                f"exhaustion_probability:{snapshot.exhaustion_probability:.4f}"
            )

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: WhaleAccumulationPayload,
    ) -> list[str]:
        features = [
            *whale_accumulation_source_features(),
            WHALES_FEATURES.ACTIVITY,
            WHALES_FEATURES.PRESSURE,
            WHALES_FEATURES.LARGE_TRADE,
            WHALES_FEATURES.CLUSTER,
            WHALES_FEATURES.CLUSTER_UPDATE,
            WHALES_FEATURES.CLUSTER_EXHAUSTION,
            WHALES_FEATURES.DOMINANT_SIDE,
            WHALES_FEATURES.WHALE_SIDE,
            WHALES_FEATURES.CLUSTER_SIDE,
            WHALES_FEATURES.IMBALANCE_RATIO,
            WHALES_FEATURES.PRESSURE_SCORE,
            WHALES_FEATURES.CLUSTER_SCORE,
            WHALES_FEATURES.CONTINUATION_PROBABILITY,
            WHALES_FEATURES.EXHAUSTION_PROBABILITY,
            WHALES_FEATURES.TOTAL_NOTIONAL,
            WHALES_FEATURES.TRADE_COUNT,
            WHALES_FEATURES.LARGE_TRADE_NOTIONAL,
            WHALES_FEATURES.LARGE_TRADE_ZSCORE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: WhaleAccumulationPayload,
    ) -> list[str]:
        tags = [
            self.accumulation_config.tag_whales,
            self.accumulation_config.tag_accumulation,
            self.accumulation_config.tag_whale_accumulation,
            self.accumulation_config.tag_buy_side_accumulation,
            self.accumulation_config.tag_pre_breakout_positioning,
            f"side:{payload.side.value}",
            f"accumulation_side:{payload.accumulation_side}",
        ]

        if payload.snapshot.cluster_score is not None:
            tags.append(self.accumulation_config.tag_cluster_support)

        if payload.snapshot.continuation_probability is not None:
            tags.append(self.accumulation_config.tag_continuation_bias)

        if (
            payload.snapshot.exhaustion_probability is None
            or payload.snapshot.exhaustion_probability <= self.accumulation_config.low_exhaustion_threshold
        ):
            tags.append(self.accumulation_config.tag_low_exhaustion)

        if payload.snapshot.market_type:
            tags.append(f"market_type:{payload.snapshot.market_type}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.accumulation_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.accumulation_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.accumulation_config.execution_take_profit_bps_hint,
            "risk_reward": self.accumulation_config.accumulation_rr_hint,
        }