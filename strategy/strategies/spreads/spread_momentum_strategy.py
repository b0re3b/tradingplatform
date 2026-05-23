# trading_system/strategy/strategies/spreads/spread_momentum_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import SpreadRegime, SpreadType
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    SPREADS_FEATURES,
    SpreadCompositeSnapshot,
    SpreadsStrategyConfig,
    SpreadsTradingStrategy,
)
from .utils import (
    DECIMAL_ONE,
    DECIMAL_ZERO,
    ScoreBreakdown,
    average_score,
    confidence_from_components,
    cross_exchange_to_signal_side,
    edge_component,
    extract_timestamp,
    freshness_score,
    has_tradeable_edge,
    is_anomaly_signal,
    is_directional_side,
    is_mean_reversion_signal,
    is_regime_shift_signal,
    is_stale,
    is_widening_signal,
    normalize_label,
    quote_component,
    regime_component,
    serialize_for_metadata,
    spread_direction_to_signal_side,
    spread_momentum_source_features,
    spread_quality_filter_reason,
    unit_score,
    weighted_score,
    zscore_component,
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
class SpreadMomentumPayload:
    """
    Normalized strategy-level payload for spread momentum / continuation.

    This strategy is intentionally separate from mean reversion:
    - WIDENING means follow spread expansion;
    - COMPRESSING means follow spread compression;
    - exact multi-leg construction remains SignalProcessor/SignalBuilder concern.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMomentumPayload")

    snapshot: SpreadCompositeSnapshot
    side: SignalSide

    momentum_direction: str
    edge: Decimal
    abs_edge: Decimal
    abs_zscore: Decimal

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpreadMomentumStrategyConfig(SpreadsStrategyConfig):
    """
    Generic spread momentum config.

    Strategy idea:
    - analytics.spreads indicates widening/compression continuation;
    - spread direction is directional and supported;
    - quote quality and edge remain tradeable;
    - strategy returns internal StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMomentumStrategyConfig")

    min_momentum_score: float = 0.62
    min_momentum_confidence: float = 0.56

    min_abs_spread_bps: Decimal = Decimal("2")
    min_abs_edge: Decimal = Decimal("0")
    min_zscore: Decimal = Decimal("0.75")
    max_zscore: Decimal | None = Decimal("5.5")

    require_valid_quote: bool = True
    require_tradeable_edge: bool = True
    require_direction: bool = True
    require_momentum_signal: bool = False

    allow_widening: bool = True
    allow_compressing: bool = True
    allow_spot_futures: bool = True
    allow_cross_exchange: bool = True
    allow_unknown_spread_type: bool = False

    reject_mean_reversion_signal: bool = True
    reject_extreme_dislocation: bool = False
    allow_anomaly_signal: bool = True
    allow_regime_shift_signal: bool = True

    allowed_regimes: set[str] = field(
        default_factory=lambda: {
            SpreadRegime.ELEVATED.value,
            SpreadRegime.EXTREME.value,
            SpreadRegime.DISLOCATED.value,
            SpreadRegime.NORMAL.value,
        }
    )

    score_direction_weight: float = 0.24
    score_edge_weight: float = 0.22
    score_zscore_weight: float = 0.18
    score_regime_weight: float = 0.12
    score_confirmation_weight: float = 0.14
    score_quote_weight: float = 0.05
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    widening_bonus: float = 0.04
    compressing_bonus: float = 0.04
    strong_edge_bonus: float = 0.04
    zscore_momentum_bonus: float = 0.03
    regime_shift_bonus: float = 0.03
    anomaly_bonus: float = 0.02

    strong_edge_multiplier: Decimal = Decimal("2")
    strong_zscore_multiplier: Decimal = Decimal("2")

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.CONTINUATION

    tag_spread_momentum: str = "spread_momentum"
    tag_widening: str = "spread_widening"
    tag_compressing: str = "spread_compressing"
    tag_continuation: str = "continuation"
    tag_directional_spread: str = "directional_spread"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    momentum_tp_multiplier_hint: float | None = None

    required_spreads_features: tuple[str, ...] = (
        SPREADS_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategyConfig.validate")
        SpreadsStrategyConfig.validate(self)

        if not 0.0 <= float(self.min_momentum_score) <= 1.0:
            raise StrategyConfigError("min_momentum_score must be between 0.0 and 1.0")

        if not 0.0 <= float(self.min_momentum_confidence) <= 1.0:
            raise StrategyConfigError("min_momentum_confidence must be between 0.0 and 1.0")

        if self.min_abs_spread_bps < DECIMAL_ZERO:
            raise StrategyConfigError("min_abs_spread_bps must be >= 0")

        if self.min_abs_edge < DECIMAL_ZERO:
            raise StrategyConfigError("min_abs_edge must be >= 0")

        if self.min_zscore < DECIMAL_ZERO:
            raise StrategyConfigError("min_zscore must be >= 0")

        if self.max_zscore is not None and self.max_zscore <= self.min_zscore:
            raise StrategyConfigError("max_zscore must be greater than min_zscore")

        if self.strong_edge_multiplier < DECIMAL_ZERO:
            raise StrategyConfigError("strong_edge_multiplier must be >= 0")

        if self.strong_zscore_multiplier < DECIMAL_ZERO:
            raise StrategyConfigError("strong_zscore_multiplier must be >= 0")

        score_weights = {
            "score_direction_weight": self.score_direction_weight,
            "score_edge_weight": self.score_edge_weight,
            "score_zscore_weight": self.score_zscore_weight,
            "score_regime_weight": self.score_regime_weight,
            "score_confirmation_weight": self.score_confirmation_weight,
            "score_quote_weight": self.score_quote_weight,
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

        unit_fields = {
            "widening_bonus": self.widening_bonus,
            "compressing_bonus": self.compressing_bonus,
            "strong_edge_bonus": self.strong_edge_bonus,
            "zscore_momentum_bonus": self.zscore_momentum_bonus,
            "regime_shift_bonus": self.regime_shift_bonus,
            "anomaly_bonus": self.anomaly_bonus,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        hint_fields = {
            "execution_entry_offset_bps_hint": self.execution_entry_offset_bps_hint,
            "execution_stop_buffer_bps_hint": self.execution_stop_buffer_bps_hint,
            "execution_take_profit_bps_hint": self.execution_take_profit_bps_hint,
            "momentum_tp_multiplier_hint": self.momentum_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_spread_momentum",
            "tag_widening",
            "tag_compressing",
            "tag_continuation",
            "tag_directional_spread",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spreads_features:
            raise StrategyConfigError("required_spreads_features cannot be empty")


class SpreadMomentumStrategy(SpreadsTradingStrategy):
    """
    Unified generic spread momentum strategy.

    Input:
        StrategyContext with FeatureSource.SPREADS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMomentumStrategy")

    component_namespace = "strategy.spreads.momentum"
    category: StrategyCategory = StrategyCategory.SPREADS
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spreads_config: SpreadMomentumStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy.__init__")
        resolved_spreads_config = spreads_config or SpreadMomentumStrategyConfig()
        resolved_spreads_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spreads_config=resolved_spreads_config,
            service_name=service_name,
        )

        self.momentum_config: SpreadMomentumStrategyConfig = resolved_spreads_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy.strategy_name")
        return "spread_momentum"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPREADS,
            timeframe=Timeframe.M1,
            tags=[
                self.momentum_config.tag_spreads,
                self.momentum_config.tag_momentum,
                self.momentum_config.tag_spread_momentum,
                self.momentum_config.tag_continuation,
                self.momentum_config.tag_directional_spread,
                "analytics_spreads",
            ],
            version="2.0.0",
            description=(
                "Interprets widening/compressing spread momentum from normalized "
                "StrategyContext and returns internal StrategySignal."
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
                "source": "analytics.spreads",
                "strategy_type": "spread_momentum",
                "base_class": "SpreadsTradingStrategy",
                "canonical_payload": "SpreadCompositeSnapshot",
                "uses_direction": True,
                "uses_widening_compressing": True,
                "opposite_of_mean_reversion": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.momentum_config.required_spreads_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_spreads_data(
            context,
            tuple(self.momentum_config.required_spreads_features),
        ):
            return None

        if self.has_stale_spreads_features(
            context,
            tuple(self.momentum_config.required_spreads_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.momentum_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = spread_quality_filter_reason(
            payload.snapshot.to_signal_payload(),
            min_score=self.momentum_config.min_score,
            min_confidence=max(
                self.momentum_config.min_confidence,
                self.momentum_config.min_momentum_confidence,
            ),
            require_valid_quote=self.momentum_config.require_valid_quote,
            require_edge=self.momentum_config.require_tradeable_edge,
            allowed_regimes=self.momentum_config.allowed_regimes,
            stale_after_seconds=self.momentum_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spread_snapshot(
            payload.snapshot,
            require_valid_quote=self.momentum_config.require_valid_quote,
            require_edge=self.momentum_config.require_tradeable_edge,
        ):
            return None

        if not self._passes_spread_type_filters(payload.snapshot):
            return None

        if not self._passes_momentum_filters(payload):
            return None

        if not self._passes_confirmation_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.momentum_config.min_momentum_score:
            return None

        if breakdown.confidence < self.momentum_config.min_momentum_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "spread_momentum_signal",
                    f"side:{payload.side.value}",
                    f"momentum_direction:{payload.momentum_direction}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spreads_setup_family": "spread_momentum",
            "spreads_strategy_version": "2.0.0",
            "contract": "spreads",
            "contract_version": "strategy-domain-v1",
            "primary_section": "snapshot",
            "secondary_section": "signal",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "spread_type": normalize_label(payload.snapshot.spread_type),
            "momentum_direction": payload.momentum_direction,
            "mapped_side": payload.side.value,
            "edge": str(payload.edge),
            "abs_edge": str(payload.abs_edge),
            "spread_bps": str(payload.snapshot.spread_bps)
            if payload.snapshot.spread_bps is not None
            else None,
            "basis": str(payload.snapshot.basis)
            if payload.snapshot.basis is not None
            else None,
            "funding_adjusted_spread": str(payload.snapshot.funding_adjusted_spread)
            if payload.snapshot.funding_adjusted_spread is not None
            else None,
            "net_edge": str(payload.snapshot.net_edge)
            if payload.snapshot.net_edge is not None
            else None,
            "net_edge_bps": str(payload.snapshot.net_edge_bps)
            if payload.snapshot.net_edge_bps is not None
            else None,
            "zscore": str(payload.snapshot.zscore)
            if payload.snapshot.zscore is not None
            else None,
            "abs_zscore": str(payload.abs_zscore),
            "regime": normalize_label(payload.snapshot.regime),
            "direction": normalize_label(payload.snapshot.direction),
            "signal_type": normalize_label(payload.snapshot.signal_type),
            "quote_validity": normalize_label(payload.snapshot.quote_validity),
            "has_edge": payload.snapshot.tradeable_edge,
            "leg_semantics": self._leg_semantics(payload),
            "execution_hints": self._execution_hints(),
        }

        return self.build_spread_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.momentum_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.momentum_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> SpreadMomentumPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._extract_payload")
        snapshot = self.resolve_spread_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        edge = self._edge(snapshot)
        if edge is None or edge == DECIMAL_ZERO:
            return None

        direction = self._momentum_direction(snapshot)
        if direction == "UNKNOWN":
            return None

        side = self._side(snapshot, direction)
        if self.momentum_config.require_direction and not is_directional_side(side):
            return None

        event_time = (
            extract_timestamp(snapshot.to_signal_payload())
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "spread_momentum_context",
            f"spread_type:{normalize_label(snapshot.spread_type)}",
            f"momentum_direction:{direction}",
            f"edge:{edge}",
            f"abs_zscore:{snapshot.abs_zscore}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        return SpreadMomentumPayload(
            snapshot=snapshot,
            side=side,
            momentum_direction=direction,
            edge=edge,
            abs_edge=abs(edge),
            abs_zscore=snapshot.abs_zscore,
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw={
                "snapshot": snapshot.raw_snapshot,
                "signal": snapshot.raw_signal,
                "opportunity": snapshot.raw_opportunity,
            },
        )

    def _edge(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> Decimal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._edge")
        for candidate in (
            snapshot.net_edge_bps,
            snapshot.net_edge,
            snapshot.funding_adjusted_spread,
            snapshot.spread_bps,
            snapshot.basis,
        ):
            if candidate is not None:
                return candidate
        return None

    def _momentum_direction(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._momentum_direction")
        label = normalize_label(snapshot.direction)

        if label == "widening":
            return "WIDENING"

        if label in {"compressing", "compression"}:
            return "COMPRESSING"

        if is_widening_signal(snapshot.raw_signal):
            return "WIDENING"

        signal_label = normalize_label(snapshot.signal_type)
        if signal_label in {"compressing", "compression", "spread_compressing"}:
            return "COMPRESSING"

        return "UNKNOWN"

    def _side(
        self,
        snapshot: SpreadCompositeSnapshot,
        direction: str,
    ) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._side")
        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            side = cross_exchange_to_signal_side(
                snapshot.raw_opportunity or snapshot.to_signal_payload()
            )
            if is_directional_side(side):
                return side

        if snapshot.direction is not None:
            side = spread_direction_to_signal_side(snapshot.direction)
            if is_directional_side(side):
                return side

        if direction == "WIDENING":
            return SignalSide.LONG

        if direction == "COMPRESSING":
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_spread_type_filters(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._passes_spread_type_filters")
        if snapshot.spread_type is None:
            return self.momentum_config.allow_unknown_spread_type

        if snapshot.spread_type is SpreadType.SPOT_FUTURES:
            return self.momentum_config.allow_spot_futures

        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            return self.momentum_config.allow_cross_exchange

        return self.momentum_config.allow_unknown_spread_type

    def _passes_momentum_filters(
        self,
        payload: SpreadMomentumPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._passes_momentum_filters")
        snapshot = payload.snapshot

        if self.momentum_config.require_valid_quote and not snapshot.is_quote_valid:
            return False

        if self.momentum_config.require_tradeable_edge and not has_tradeable_edge(
            snapshot.to_signal_payload()
        ):
            return False

        if payload.momentum_direction == "WIDENING" and not self.momentum_config.allow_widening:
            return False

        if payload.momentum_direction == "COMPRESSING" and not self.momentum_config.allow_compressing:
            return False

        if payload.abs_edge < self.momentum_config.min_abs_edge:
            return False

        if snapshot.spread_bps is not None:
            if abs(snapshot.spread_bps) < self.momentum_config.min_abs_spread_bps:
                return False

        if payload.abs_zscore < self.momentum_config.min_zscore:
            return False

        if self.momentum_config.max_zscore is not None:
            if payload.abs_zscore > self.momentum_config.max_zscore:
                return False

        if self.momentum_config.allowed_regimes:
            if normalize_label(snapshot.regime) not in {
                item.lower()
                for item in self.momentum_config.allowed_regimes
            }:
                return False

        if self.momentum_config.reject_extreme_dislocation:
            if snapshot.regime is SpreadRegime.DISLOCATED:
                return False

        return True

    def _passes_confirmation_filters(
        self,
        payload: SpreadMomentumPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._passes_confirmation_filters")
        snapshot = payload.snapshot

        if self.momentum_config.require_momentum_signal:
            if payload.momentum_direction == "WIDENING":
                if not is_widening_signal(snapshot.raw_signal):
                    return False

            if payload.momentum_direction == "COMPRESSING":
                signal_label = normalize_label(snapshot.signal_type)
                if signal_label not in {
                    "compressing",
                    "compression",
                    "spread_compressing",
                    "momentum",
                    "continuation",
                }:
                    return False

        if self.momentum_config.reject_mean_reversion_signal:
            if is_mean_reversion_signal(snapshot.raw_signal):
                return False

        if is_regime_shift_signal(snapshot.raw_signal) and not self.momentum_config.allow_regime_shift_signal:
            return False

        if is_anomaly_signal(snapshot.raw_signal) and not self.momentum_config.allow_anomaly_signal:
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: SpreadMomentumPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        direction_component = self._direction_component(payload)
        edge_scale = max(
            self.momentum_config.min_abs_edge,
            self.momentum_config.min_abs_spread_bps,
            DECIMAL_ONE,
        )
        edge_component_value = edge_component(
            snapshot.to_signal_payload(),
            min_edge=edge_scale,
            scale=edge_scale * Decimal("3"),
        )
        z_component = zscore_component(
            snapshot.to_signal_payload(),
            entry_zscore=max(self.momentum_config.min_zscore, DECIMAL_ONE),
            stop_zscore=self.momentum_config.max_zscore,
        )
        regime_component_value = regime_component(snapshot.to_signal_payload())
        confirmation_component = self._confirmation_component(payload)
        quote_component_value = quote_component(snapshot.to_signal_payload())
        freshness_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.momentum_config.stale_feature_max_age_seconds,
        )

        components = {
            "direction": direction_component,
            "edge": edge_component_value,
            "zscore": z_component,
            "regime": regime_component_value,
            "confirmation": confirmation_component,
            "quote": quote_component_value,
            "freshness": freshness_component,
        }
        weights = {
            "direction": self.momentum_config.score_direction_weight,
            "edge": self.momentum_config.score_edge_weight,
            "zscore": self.momentum_config.score_zscore_weight,
            "regime": self.momentum_config.score_regime_weight,
            "confirmation": self.momentum_config.score_confirmation_weight,
            "quote": self.momentum_config.score_quote_weight,
            "freshness": self.momentum_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=direction_component)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, direction_component),
            context=average_score(edge_component_value, regime_component_value),
            confirmation=average_score(confirmation_component, z_component),
            freshness=freshness_component,
            primary_weight=self.momentum_config.confidence_primary_weight,
            context_weight=self.momentum_config.confidence_context_weight,
            confirmation_weight=self.momentum_config.confidence_confirmation_weight,
            freshness_weight=self.momentum_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "spread_momentum_context",
            f"momentum_direction:{payload.momentum_direction}",
            f"side:{payload.side.value}",
            f"abs_edge:{payload.abs_edge}",
            f"abs_zscore:{payload.abs_zscore}",
        ]

        if payload.momentum_direction == "WIDENING":
            score += self.momentum_config.widening_bonus
            confirmations.append("widening_momentum")

        if payload.momentum_direction == "COMPRESSING":
            score += self.momentum_config.compressing_bonus
            confirmations.append("compressing_momentum")

        if payload.abs_edge >= (
            self.momentum_config.min_abs_edge
            * self.momentum_config.strong_edge_multiplier
        ):
            score += self.momentum_config.strong_edge_bonus
            confirmations.append("strong_spread_edge")

        if payload.abs_zscore >= (
            self.momentum_config.min_zscore
            * self.momentum_config.strong_zscore_multiplier
        ):
            score += self.momentum_config.zscore_momentum_bonus
            confirmations.append("strong_zscore_momentum")

        if is_regime_shift_signal(snapshot.raw_signal):
            score += self.momentum_config.regime_shift_bonus
            confirmations.append("regime_shift_momentum")

        if is_anomaly_signal(snapshot.raw_signal):
            score += self.momentum_config.anomaly_bonus
            confirmations.append("spread_anomaly_momentum")

        if snapshot.spread_bps is not None:
            reasons.append(f"spread_bps:{snapshot.spread_bps}")

        if snapshot.direction is not None:
            reasons.append(f"direction:{normalize_label(snapshot.direction)}")

        if snapshot.net_edge is not None:
            reasons.append(f"net_edge:{snapshot.net_edge}")

        if snapshot.net_edge_bps is not None:
            reasons.append(f"net_edge_bps:{snapshot.net_edge_bps}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _direction_component(
        self,
        payload: SpreadMomentumPayload,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._direction_component")
        if payload.momentum_direction in {"WIDENING", "COMPRESSING"}:
            return 1.0
        return 0.0

    def _confirmation_component(
        self,
        payload: SpreadMomentumPayload,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._confirmation_component")
        snapshot = payload.snapshot

        widening = 1.0 if is_widening_signal(snapshot.raw_signal) else 0.0
        compressing = 1.0 if normalize_label(snapshot.signal_type) in {
            "compressing",
            "compression",
            "spread_compressing",
        } else 0.0

        momentum_match = (
            widening
            if payload.momentum_direction == "WIDENING"
            else compressing
        )

        components = {
            "momentum_match": momentum_match,
            "regime_shift": 0.75 if is_regime_shift_signal(snapshot.raw_signal) else 0.0,
            "anomaly": 0.60 if is_anomaly_signal(snapshot.raw_signal) else 0.0,
            "not_mean_reversion": 0.0 if is_mean_reversion_signal(snapshot.raw_signal) else 1.0,
            "quote": quote_component(snapshot.to_signal_payload()),
        }

        return weighted_score(
            components,
            {
                "momentum_match": 0.35,
                "regime_shift": 0.18,
                "anomaly": 0.12,
                "not_mean_reversion": 0.20,
                "quote": 0.15,
            },
            default=0.0,
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: SpreadMomentumPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._source_features")
        features = [
            *spread_momentum_source_features(),
            SPREADS_FEATURES.SNAPSHOT,
            SPREADS_FEATURES.SIGNAL,
            SPREADS_FEATURES.SPREAD_TYPE,
            SPREADS_FEATURES.SYMBOL,
            SPREADS_FEATURES.EXCHANGE_A,
            SPREADS_FEATURES.EXCHANGE_B,
            SPREADS_FEATURES.SPREAD_BPS,
            SPREADS_FEATURES.BASIS,
            SPREADS_FEATURES.FUNDING_ADJUSTED_SPREAD,
            SPREADS_FEATURES.NET_EDGE,
            SPREADS_FEATURES.NET_EDGE_BPS,
            SPREADS_FEATURES.ZSCORE,
            SPREADS_FEATURES.REGIME,
            SPREADS_FEATURES.DIRECTION,
            SPREADS_FEATURES.SIGNAL_TYPE,
            SPREADS_FEATURES.QUOTE_VALIDITY,
            SPREADS_FEATURES.HAS_EDGE,
            SPREADS_FEATURES.CONFIDENCE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: SpreadMomentumPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._tags")
        tags = [
            self.momentum_config.tag_spreads,
            self.momentum_config.tag_momentum,
            self.momentum_config.tag_spread_momentum,
            self.momentum_config.tag_continuation,
            self.momentum_config.tag_directional_spread,
            f"side:{payload.side.value}",
            f"direction:{payload.momentum_direction.lower()}",
        ]

        if payload.momentum_direction == "WIDENING":
            tags.append(self.momentum_config.tag_widening)

        if payload.momentum_direction == "COMPRESSING":
            tags.append(self.momentum_config.tag_compressing)

        if payload.snapshot.spread_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spread_type)}")

        if payload.snapshot.regime is not None:
            tags.append(f"regime:{normalize_label(payload.snapshot.regime)}")

        if payload.snapshot.signal_type is not None:
            tags.append(f"signal:{normalize_label(payload.snapshot.signal_type)}")

        return list(dict.fromkeys(tags))

    def _leg_semantics(
        self,
        payload: SpreadMomentumPayload,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._leg_semantics")
        snapshot = payload.snapshot

        return {
            "momentum_direction": payload.momentum_direction,
            "spread_type": normalize_label(snapshot.spread_type),
            "generic_side": payload.side.value,
            "leg_a": {
                "exchange": snapshot.exchange_a,
                "market_type": snapshot.market_type_a,
                "symbol": snapshot.exchange_symbol_a,
            },
            "leg_b": {
                "exchange": snapshot.exchange_b,
                "market_type": snapshot.market_type_b,
                "symbol": snapshot.exchange_symbol_b,
            },
            "note": (
                "StrategySignal.side is generic LONG/SHORT. Exact multi-leg "
                "construction for spread momentum is owned by SignalProcessor/SignalBuilder."
            ),
        }

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMomentumStrategy._execution_hints")
        return {
            "entry_offset_bps": self.momentum_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.momentum_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.momentum_config.execution_take_profit_bps_hint,
            "momentum_tp_multiplier": self.momentum_config.momentum_tp_multiplier_hint,
        }