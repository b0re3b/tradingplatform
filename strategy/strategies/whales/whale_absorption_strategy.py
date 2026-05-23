# trading_system/strategy/strategies/whales/whale_absorption_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler
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
    exhaustion_context_score,
    extract_event_time,
    freshness_score,
    is_directional_side,
    is_stale,
    large_trade_score,
    liquidation_context_score,
    normalize_label,
    opposite_side,
    serialize_for_metadata,
    side_label_to_signal_side,
    unit_score,
    whale_absorption_source_features,
    whale_activity_score,
    whale_pressure_score,
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
class WhaleAbsorptionPayload:
    """
    Normalized strategy-level payload для whale absorption reversal.

    Direction convention:
    - whale buy absorption + sell-side liquidation/exhaustion -> LONG;
    - whale sell absorption + buy-side liquidation/exhaustion -> SHORT.
    """
    _logger = logging.getLogger(__name__ + ".WhaleAbsorptionPayload")

    snapshot: WhaleCompositeSnapshot
    side: SignalSide
    whale_side: str
    absorbed_side: str

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WhaleAbsorptionStrategyConfig(WhalesStrategyConfig):
    """
    Unified whale absorption strategy config.

    Strategy idea:
    - dominant whale pressure shows buy/sell absorption;
    - opposite liquidation/exhaustion flow confirms forced-flow absorption;
    - cluster/activity/large trade may strengthen the setup;
    - strategy returns internal StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".WhaleAbsorptionStrategyConfig")

    min_absorption_score: float = 0.64
    min_absorption_confidence: float = 0.58

    min_pressure_imbalance_ratio: float = 0.62
    min_pressure_score: float = 0.50
    min_context_strength: float = 0.55
    min_cluster_score: float = 0.45
    min_exhaustion_probability: float = 0.55

    min_liquidation_notional: float = 150_000.0
    min_activity_notional: float = 250_000.0
    min_activity_trade_count: int = 2
    min_large_trade_notional: float = 200_000.0
    min_large_trade_zscore: float = 1.5

    require_pressure_context: bool = True
    require_opposite_liquidation: bool = True
    require_liquidation_context: bool = True
    require_cluster_confirmation: bool = True
    require_exhaustion_confirmation: bool = False

    use_activity_confirmation: bool = True
    use_large_trade_confirmation: bool = True
    require_activity_same_side: bool = False
    require_large_trade_same_side: bool = False

    score_pressure_weight: float = 0.28
    score_liquidation_weight: float = 0.24
    score_cluster_weight: float = 0.16
    score_exhaustion_weight: float = 0.12
    score_activity_weight: float = 0.08
    score_large_trade_weight: float = 0.06
    score_freshness_weight: float = 0.06

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_pressure_bonus: float = 0.04
    strong_liquidation_bonus: float = 0.04
    cluster_confirmation_bonus: float = 0.04
    exhaustion_confirmation_bonus: float = 0.04
    activity_confirmation_bonus: float = 0.03
    large_trade_confirmation_bonus: float = 0.03

    strong_pressure_threshold: float = 0.75
    strong_context_threshold: float = 0.75
    strong_liquidation_multiplier: float = 2.0

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.REVERSAL

    tag_whale_absorption: str = "whale_absorption"
    tag_absorption_reversal: str = "absorption_reversal"
    tag_liquidation_absorption: str = "liquidation_absorption"
    tag_opposite_liquidation: str = "opposite_liquidation"
    tag_cluster_confirmed: str = "cluster_confirmed"
    tag_exhaustion_confirmed: str = "exhaustion_confirmed"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    absorption_rr_hint: float | None = None

    required_whales_features: tuple[str, ...] = (
        WHALES_FEATURES.PRESSURE,
        WHALES_FEATURES.LIQUIDATION_CONTEXT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategyConfig.validate")
        WhalesStrategyConfig.validate(self)

        unit_fields = {
            "min_absorption_score": self.min_absorption_score,
            "min_absorption_confidence": self.min_absorption_confidence,
            "min_pressure_imbalance_ratio": self.min_pressure_imbalance_ratio,
            "min_pressure_score": self.min_pressure_score,
            "min_context_strength": self.min_context_strength,
            "min_cluster_score": self.min_cluster_score,
            "min_exhaustion_probability": self.min_exhaustion_probability,
            "strong_pressure_bonus": self.strong_pressure_bonus,
            "strong_liquidation_bonus": self.strong_liquidation_bonus,
            "cluster_confirmation_bonus": self.cluster_confirmation_bonus,
            "exhaustion_confirmation_bonus": self.exhaustion_confirmation_bonus,
            "activity_confirmation_bonus": self.activity_confirmation_bonus,
            "large_trade_confirmation_bonus": self.large_trade_confirmation_bonus,
            "strong_pressure_threshold": self.strong_pressure_threshold,
            "strong_context_threshold": self.strong_context_threshold,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_liquidation_notional": self.min_liquidation_notional,
            "min_activity_notional": self.min_activity_notional,
            "min_large_trade_notional": self.min_large_trade_notional,
            "min_large_trade_zscore": self.min_large_trade_zscore,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if self.min_activity_trade_count < 0:
            raise StrategyConfigError("min_activity_trade_count must be >= 0")

        if self.strong_liquidation_multiplier < 0:
            raise StrategyConfigError("strong_liquidation_multiplier must be >= 0")

        score_weights = {
            "score_pressure_weight": self.score_pressure_weight,
            "score_liquidation_weight": self.score_liquidation_weight,
            "score_cluster_weight": self.score_cluster_weight,
            "score_exhaustion_weight": self.score_exhaustion_weight,
            "score_activity_weight": self.score_activity_weight,
            "score_large_trade_weight": self.score_large_trade_weight,
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
            "absorption_rr_hint": self.absorption_rr_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_whale_absorption",
            "tag_absorption_reversal",
            "tag_liquidation_absorption",
            "tag_opposite_liquidation",
            "tag_cluster_confirmed",
            "tag_exhaustion_confirmed",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_whales_features:
            raise StrategyConfigError("required_whales_features cannot be empty")


class WhaleAbsorptionStrategy(WhalesTradingStrategy):
    """
    Unified whale absorption reversal strategy.

    Input:
        StrategyContext with FeatureSource.WHALES domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".WhaleAbsorptionStrategy")

    component_namespace = "strategy.whales.absorption"
    category: StrategyCategory = StrategyCategory.WHALES
    default_setup_type: SetupType = SetupType.REVERSAL

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        whales_config: WhaleAbsorptionStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy.__init__")
        resolved_whales_config = whales_config or WhaleAbsorptionStrategyConfig()
        resolved_whales_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            whales_config=resolved_whales_config,
            service_name=service_name,
        )

        self.absorption_config: WhaleAbsorptionStrategyConfig = resolved_whales_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy.strategy_name")
        return "whale_absorption"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.WHALES,
            timeframe=Timeframe.M1,
            tags=[
                self.absorption_config.tag_whales,
                self.absorption_config.tag_absorption,
                self.absorption_config.tag_whale_absorption,
                self.absorption_config.tag_absorption_reversal,
                self.absorption_config.tag_liquidation_absorption,
                "analytics_whales",
            ],
            version="2.0.0",
            description=(
                "Interprets whale absorption against opposite liquidation/exhaustion "
                "flow from normalized StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.whales",
                "strategy_type": "whale_absorption",
                "base_class": "WhalesTradingStrategy",
                "canonical_payload": "WhaleCompositeSnapshot",
                "uses_pressure": True,
                "uses_liquidation_context": True,
                "uses_cluster_exhaustion": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.absorption_config.required_whales_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_whales_data(
            context,
            tuple(self.absorption_config.required_whales_features),
        ):
            return None

        if self.has_stale_whales_features(
            context,
            tuple(self.absorption_config.required_whales_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.absorption_config.stale_feature_max_age_seconds,
        ):
            return None

        if not self.accepts_whale_snapshot(
            payload.snapshot,
            require_futures_market_type=self.absorption_config.require_futures_market_type,
            min_confidence=self.absorption_config.min_confidence,
        ):
            return None

        if not self._passes_absorption_filters(payload):
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

        if breakdown.score < self.absorption_config.min_absorption_score:
            return None

        if breakdown.confidence < self.absorption_config.min_absorption_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "whale_absorption_signal",
                    f"side:{payload.side.value}",
                    f"whale_side:{payload.whale_side}",
                    f"absorbed_side:{payload.absorbed_side}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "whales_setup_family": "whale_absorption",
            "whales_strategy_version": "2.0.0",
            "contract": "whales",
            "contract_version": "strategy-domain-v1",
            "primary_section": "pressure",
            "secondary_section": "liquidation_context",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "whale_side": payload.whale_side,
            "absorbed_side": payload.absorbed_side,
            "dominant_side": payload.snapshot.dominant_side,
            "liquidation_side": payload.snapshot.liquidation_side,
            "exhausted_side": payload.snapshot.exhausted_side,
            "cluster_side": payload.snapshot.cluster_side,
            "imbalance_ratio": payload.snapshot.imbalance_ratio,
            "pressure_score": payload.snapshot.pressure_score,
            "context_strength": payload.snapshot.context_strength,
            "cluster_score": payload.snapshot.cluster_score,
            "continuation_probability": payload.snapshot.continuation_probability,
            "exhaustion_probability": payload.snapshot.exhaustion_probability,
            "total_notional": payload.snapshot.total_notional,
            "liquidation_notional": payload.snapshot.liquidation_notional,
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
            setup_type=self.absorption_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.absorption_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> WhaleAbsorptionPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._extract_payload")
        snapshot = self.resolve_whale_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        whale_side = self._resolve_absorbing_whale_side(snapshot)
        if whale_side not in {"buy", "sell"}:
            return None

        side = side_label_to_signal_side(whale_side)
        if not is_directional_side(side):
            return None

        absorbed_side = opposite_side(whale_side)
        if absorbed_side == "unknown":
            return None

        event_time = (
            extract_event_time(snapshot.liquidation_context)
            or extract_event_time(snapshot.cluster_exhaustion)
            or extract_event_time(snapshot.pressure)
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "whale_absorption_context",
            f"whale_side:{whale_side}",
            f"absorbed_side:{absorbed_side}",
            f"imbalance_ratio:{snapshot.imbalance_ratio:.4f}",
            f"context_strength:{snapshot.context_strength:.4f}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        return WhaleAbsorptionPayload(
            snapshot=snapshot,
            side=side,
            whale_side=whale_side,
            absorbed_side=absorbed_side,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw=snapshot.inputs(),
        )

    def _resolve_absorbing_whale_side(
        self,
        snapshot: WhaleCompositeSnapshot,
    ) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._resolve_absorbing_whale_side")
        for candidate in (
            snapshot.whale_side,
            snapshot.dominant_side,
            snapshot.cluster_side,
        ):
            label = normalize_label(candidate)
            if label in {"buy", "sell"}:
                return label
        return "unknown"

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _required_inputs(self) -> tuple[str, ...]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._required_inputs")
        required = ["pressure"]

        if self.absorption_config.require_liquidation_context:
            required.append("liquidation_context")

        if self.absorption_config.require_cluster_confirmation:
            required.append("cluster")

        return tuple(required)

    def _passes_absorption_filters(
        self,
        payload: WhaleAbsorptionPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._passes_absorption_filters")
        snapshot = payload.snapshot

        if self.absorption_config.require_pressure_context:
            if not snapshot.has_pressure:
                return False

        if self.absorption_config.require_liquidation_context:
            if not snapshot.has_liquidation_context:
                return False

        if snapshot.imbalance_ratio < self.absorption_config.min_pressure_imbalance_ratio:
            return False

        if snapshot.pressure_score < self.absorption_config.min_pressure_score:
            return False

        if self.absorption_config.require_liquidation_context:
            if snapshot.context_strength < self.absorption_config.min_context_strength:
                return False

            if snapshot.liquidation_notional < self.absorption_config.min_liquidation_notional:
                return False

        if self.absorption_config.require_opposite_liquidation:
            if snapshot.liquidation_side not in {"unknown", ""}:
                if snapshot.liquidation_side != payload.absorbed_side:
                    return False

        if self.absorption_config.require_cluster_confirmation:
            cluster_score = snapshot.cluster_score or 0.0
            if cluster_score < self.absorption_config.min_cluster_score:
                return False

            if snapshot.cluster_side not in {"unknown", ""}:
                if snapshot.cluster_side != payload.whale_side:
                    return False

        if self.absorption_config.require_exhaustion_confirmation:
            exhaustion_probability = snapshot.exhaustion_probability or 0.0
            if exhaustion_probability < self.absorption_config.min_exhaustion_probability:
                return False

            if snapshot.exhausted_side not in {"unknown", ""}:
                if snapshot.exhausted_side != payload.absorbed_side:
                    return False

        if self.absorption_config.require_activity_same_side and snapshot.has_activity:
            activity_side = normalize_label(snapshot.activity.get("side") or snapshot.activity.get("dominant_side"))
            if activity_side in {"buy", "sell"} and activity_side != payload.whale_side:
                return False

        if self.absorption_config.require_large_trade_same_side and snapshot.has_large_trade:
            large_trade_side = normalize_label(snapshot.large_trade.get("side") or snapshot.large_trade.get("trade_side"))
            if large_trade_side in {"buy", "sell"} and large_trade_side != payload.whale_side:
                return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: WhaleAbsorptionPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._build_score_breakdown")
        snapshot = payload.snapshot
        inputs = snapshot.inputs()

        pressure_component = whale_pressure_score(
            snapshot.pressure,
            min_imbalance_ratio=self.absorption_config.min_pressure_imbalance_ratio,
        )
        liquidation_component = liquidation_context_score(
            snapshot.liquidation_context,
            min_notional=self.absorption_config.min_liquidation_notional,
            min_context_strength=self.absorption_config.min_context_strength,
        )
        cluster_component = cluster_context_score(
            inputs,
            min_cluster_score=self.absorption_config.min_cluster_score,
        )
        exhaustion_component = exhaustion_context_score(
            inputs,
            min_exhaustion_probability=self.absorption_config.min_exhaustion_probability,
        )
        activity_component = (
            whale_activity_score(
                snapshot.activity,
                min_notional=self.absorption_config.min_activity_notional,
                min_trade_count=self.absorption_config.min_activity_trade_count,
            )
            if self.absorption_config.use_activity_confirmation
            else 0.0
        )
        large_trade_component = (
            large_trade_score(
                snapshot.large_trade,
                min_notional=self.absorption_config.min_large_trade_notional,
                min_zscore=self.absorption_config.min_large_trade_zscore,
            )
            if self.absorption_config.use_large_trade_confirmation
            else 0.0
        )
        freshness_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.absorption_config.stale_feature_max_age_seconds,
        )

        components = {
            "pressure": pressure_component,
            "liquidation": liquidation_component,
            "cluster": cluster_component,
            "exhaustion": exhaustion_component,
            "activity": activity_component,
            "large_trade": large_trade_component,
            "freshness": freshness_component,
        }
        weights = {
            "pressure": self.absorption_config.score_pressure_weight,
            "liquidation": self.absorption_config.score_liquidation_weight,
            "cluster": self.absorption_config.score_cluster_weight,
            "exhaustion": self.absorption_config.score_exhaustion_weight,
            "activity": self.absorption_config.score_activity_weight,
            "large_trade": self.absorption_config.score_large_trade_weight,
            "freshness": self.absorption_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=pressure_component)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, pressure_component),
            context=average_score(liquidation_component, cluster_component),
            confirmation=average_score(
                exhaustion_component,
                activity_component,
                large_trade_component,
            ),
            freshness=freshness_component,
            primary_weight=self.absorption_config.confidence_primary_weight,
            context_weight=self.absorption_config.confidence_context_weight,
            confirmation_weight=self.absorption_config.confidence_confirmation_weight,
            freshness_weight=self.absorption_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "whale_absorption_context",
            f"whale_side:{payload.whale_side}",
            f"absorbed_side:{payload.absorbed_side}",
            f"side:{payload.side.value}",
        ]

        if snapshot.imbalance_ratio >= self.absorption_config.strong_pressure_threshold:
            score += self.absorption_config.strong_pressure_bonus
            confirmations.append("strong_whale_pressure")

        if snapshot.context_strength >= self.absorption_config.strong_context_threshold:
            score += self.absorption_config.strong_liquidation_bonus
            confirmations.append("strong_liquidation_context")

        if snapshot.liquidation_notional >= (
            self.absorption_config.min_liquidation_notional
            * self.absorption_config.strong_liquidation_multiplier
        ):
            score += self.absorption_config.strong_liquidation_bonus
            confirmations.append("large_opposite_liquidation")

        if cluster_component >= 1.0:
            score += self.absorption_config.cluster_confirmation_bonus
            confirmations.append("cluster_confirms_absorption")

        if exhaustion_component >= 1.0:
            score += self.absorption_config.exhaustion_confirmation_bonus
            confirmations.append("exhaustion_confirms_absorption")

        if activity_component > 0.0:
            score += self.absorption_config.activity_confirmation_bonus
            confirmations.append("whale_activity_confirmation")

        if large_trade_component > 0.0:
            score += self.absorption_config.large_trade_confirmation_bonus
            confirmations.append("large_trade_confirmation")

        if snapshot.liquidation_side == payload.absorbed_side:
            confirmations.append("opposite_liquidation_side")

        if snapshot.exhausted_side == payload.absorbed_side:
            confirmations.append("opposite_side_exhausted")

        if snapshot.cluster_side == payload.whale_side:
            confirmations.append("cluster_same_as_whale_side")

        if snapshot.total_notional > 0:
            reasons.append(f"total_notional:{snapshot.total_notional:.2f}")

        if snapshot.liquidation_notional > 0:
            reasons.append(f"liquidation_notional:{snapshot.liquidation_notional:.2f}")

        if snapshot.large_trade_notional > 0:
            reasons.append(f"large_trade_notional:{snapshot.large_trade_notional:.2f}")

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
        payload: WhaleAbsorptionPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._source_features")
        features = [
            *whale_absorption_source_features(),
            WHALES_FEATURES.PRESSURE,
            WHALES_FEATURES.LIQUIDATION_CONTEXT,
            WHALES_FEATURES.CLUSTER,
            WHALES_FEATURES.CLUSTER_UPDATE,
            WHALES_FEATURES.CLUSTER_EXHAUSTION,
            WHALES_FEATURES.ACTIVITY,
            WHALES_FEATURES.LARGE_TRADE,
            WHALES_FEATURES.DOMINANT_SIDE,
            WHALES_FEATURES.WHALE_SIDE,
            WHALES_FEATURES.LIQUIDATION_SIDE,
            WHALES_FEATURES.EXHAUSTED_SIDE,
            WHALES_FEATURES.CLUSTER_SIDE,
            WHALES_FEATURES.IMBALANCE_RATIO,
            WHALES_FEATURES.PRESSURE_SCORE,
            WHALES_FEATURES.CONTEXT_STRENGTH,
            WHALES_FEATURES.CLUSTER_SCORE,
            WHALES_FEATURES.EXHAUSTION_PROBABILITY,
            WHALES_FEATURES.TOTAL_NOTIONAL,
            WHALES_FEATURES.LIQUIDATION_NOTIONAL,
            WHALES_FEATURES.LARGE_TRADE_NOTIONAL,
            WHALES_FEATURES.LARGE_TRADE_ZSCORE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: WhaleAbsorptionPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._tags")
        tags = [
            self.absorption_config.tag_whales,
            self.absorption_config.tag_absorption,
            self.absorption_config.tag_whale_absorption,
            self.absorption_config.tag_absorption_reversal,
            self.absorption_config.tag_liquidation_absorption,
            f"side:{payload.side.value}",
            f"whale_side:{payload.whale_side}",
            f"absorbed_side:{payload.absorbed_side}",
        ]

        if payload.snapshot.liquidation_side == payload.absorbed_side:
            tags.append(self.absorption_config.tag_opposite_liquidation)

        if payload.snapshot.cluster_score is not None:
            tags.append(self.absorption_config.tag_cluster_confirmed)

        if payload.snapshot.exhaustion_probability is not None:
            tags.append(self.absorption_config.tag_exhaustion_confirmed)

        if payload.snapshot.market_type:
            tags.append(f"market_type:{payload.snapshot.market_type}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering WhaleAbsorptionStrategy._execution_hints")
        return {
            "entry_offset_bps": self.absorption_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.absorption_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.absorption_config.execution_take_profit_bps_hint,
            "risk_reward": self.absorption_config.absorption_rr_hint,
        }