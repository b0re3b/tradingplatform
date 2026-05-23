# trading_system/strategy/strategies/spreads/spread_mean_reversion_strategy.py

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
    basis_to_bias,
    basis_to_signal_side,
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
    spread_mean_reversion_source_features,
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
class SpreadMeanReversionPayload:
    """
    Normalized strategy-level payload for generic spread mean reversion.

    This strategy is intentionally broader than spot_futures_basis:
    - SPOT_FUTURES can use basis/funding-adjusted edge semantics;
    - CROSS_EXCHANGE can use generic cross-exchange side/leg semantics;
    - final multi-leg construction still belongs to SignalProcessor/SignalBuilder.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMeanReversionPayload")

    snapshot: SpreadCompositeSnapshot
    side: SignalSide

    spread_bias: str
    edge: Decimal
    abs_edge: Decimal
    abs_zscore: Decimal

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpreadMeanReversionStrategyConfig(SpreadsStrategyConfig):
    """
    Generic spread mean-reversion config.

    Strategy idea:
    - z-score or spread dislocation is stretched;
    - regime indicates elevated/extreme/dislocated spread;
    - optional analytics mean-reversion/regime/anomaly signal confirms setup;
    - strategy returns internal StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMeanReversionStrategyConfig")

    entry_zscore: Decimal = Decimal("2.0")
    stop_zscore: Decimal = Decimal("4.5")

    min_abs_edge: Decimal = Decimal("0")
    min_abs_spread_bps: Decimal = Decimal("0")

    require_valid_quote: bool = True
    require_tradeable_edge: bool = True
    require_mean_reversion_signal: bool = False
    require_directional_side: bool = True

    allow_spot_futures: bool = True
    allow_cross_exchange: bool = True
    allow_unknown_spread_type: bool = False

    allow_regime_shift_entry: bool = True
    allow_anomaly_entry: bool = True
    allow_widening_entry: bool = False
    widening_requires_wait: bool = True

    allowed_regimes: set[str] = field(
        default_factory=lambda: {
            SpreadRegime.ELEVATED.value,
            SpreadRegime.EXTREME.value,
            SpreadRegime.DISLOCATED.value,
        }
    )

    score_zscore_weight: float = 0.30
    score_edge_weight: float = 0.22
    score_regime_weight: float = 0.18
    score_confirmation_weight: float = 0.15
    score_quote_weight: float = 0.08
    score_freshness_weight: float = 0.07

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    zscore_entry_bonus: float = 0.04
    strong_edge_bonus: float = 0.04
    extreme_regime_bonus: float = 0.04
    dislocated_regime_bonus: float = 0.05
    mean_reversion_confirmation_bonus: float = 0.05
    anomaly_confirmation_bonus: float = 0.03
    regime_shift_confirmation_bonus: float = 0.03

    strong_edge_multiplier: Decimal = Decimal("2")

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    tag_spread_mean_reversion: str = "spread_mean_reversion"
    tag_zscore_reversion: str = "zscore_reversion"
    tag_dislocation: str = "spread_dislocation"
    tag_short_spread: str = "short_spread"
    tag_long_spread: str = "long_spread"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    mean_reversion_tp_multiplier_hint: float | None = None

    required_spreads_features: tuple[str, ...] = (
        SPREADS_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategyConfig.validate")
        SpreadsStrategyConfig.validate(self)

        if self.entry_zscore <= DECIMAL_ZERO:
            raise StrategyConfigError("entry_zscore must be > 0")

        if self.stop_zscore <= self.entry_zscore:
            raise StrategyConfigError("stop_zscore must be greater than entry_zscore")

        if self.min_abs_edge < DECIMAL_ZERO:
            raise StrategyConfigError("min_abs_edge must be >= 0")

        if self.min_abs_spread_bps < DECIMAL_ZERO:
            raise StrategyConfigError("min_abs_spread_bps must be >= 0")

        if self.strong_edge_multiplier < DECIMAL_ZERO:
            raise StrategyConfigError("strong_edge_multiplier must be >= 0")

        score_weights = {
            "score_zscore_weight": self.score_zscore_weight,
            "score_edge_weight": self.score_edge_weight,
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
            "zscore_entry_bonus": self.zscore_entry_bonus,
            "strong_edge_bonus": self.strong_edge_bonus,
            "extreme_regime_bonus": self.extreme_regime_bonus,
            "dislocated_regime_bonus": self.dislocated_regime_bonus,
            "mean_reversion_confirmation_bonus": self.mean_reversion_confirmation_bonus,
            "anomaly_confirmation_bonus": self.anomaly_confirmation_bonus,
            "regime_shift_confirmation_bonus": self.regime_shift_confirmation_bonus,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        hint_fields = {
            "execution_entry_offset_bps_hint": self.execution_entry_offset_bps_hint,
            "execution_stop_buffer_bps_hint": self.execution_stop_buffer_bps_hint,
            "execution_take_profit_bps_hint": self.execution_take_profit_bps_hint,
            "mean_reversion_tp_multiplier_hint": self.mean_reversion_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_spread_mean_reversion",
            "tag_zscore_reversion",
            "tag_dislocation",
            "tag_short_spread",
            "tag_long_spread",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spreads_features:
            raise StrategyConfigError("required_spreads_features cannot be empty")


class SpreadMeanReversionStrategy(SpreadsTradingStrategy):
    """
    Unified generic spread mean-reversion strategy.

    Input:
        StrategyContext with FeatureSource.SPREADS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".SpreadMeanReversionStrategy")

    component_namespace = "strategy.spreads.mean_reversion"
    category: StrategyCategory = StrategyCategory.SPREADS
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spreads_config: SpreadMeanReversionStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy.__init__")
        resolved_spreads_config = spreads_config or SpreadMeanReversionStrategyConfig()
        resolved_spreads_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spreads_config=resolved_spreads_config,
            service_name=service_name,
        )

        self.mean_reversion_config: SpreadMeanReversionStrategyConfig = (
            resolved_spreads_config
        )

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy.strategy_name")
        return "spread_mean_reversion"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPREADS,
            timeframe=Timeframe.M1,
            tags=[
                self.mean_reversion_config.tag_spreads,
                self.mean_reversion_config.tag_mean_reversion,
                self.mean_reversion_config.tag_spread_mean_reversion,
                self.mean_reversion_config.tag_zscore_reversion,
                self.mean_reversion_config.tag_dislocation,
                "analytics_spreads",
            ],
            version="2.0.0",
            description=(
                "Interprets generic spread dislocation / z-score mean reversion "
                "from normalized StrategyContext and returns internal StrategySignal."
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
                "strategy_type": "spread_mean_reversion",
                "base_class": "SpreadsTradingStrategy",
                "canonical_payload": "SpreadCompositeSnapshot",
                "uses_zscore": True,
                "uses_regime": True,
                "duplicates_signal_processor_confluence": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(
            self.mean_reversion_config.required_spreads_features
        )

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_spreads_data(
            context,
            tuple(self.mean_reversion_config.required_spreads_features),
        ):
            return None

        if self.has_stale_spreads_features(
            context,
            tuple(self.mean_reversion_config.required_spreads_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.mean_reversion_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = spread_quality_filter_reason(
            payload.snapshot.to_signal_payload(),
            min_score=self.mean_reversion_config.min_score,
            min_confidence=self.mean_reversion_config.min_confidence,
            require_valid_quote=self.mean_reversion_config.require_valid_quote,
            require_edge=self.mean_reversion_config.require_tradeable_edge,
            allowed_regimes=self.mean_reversion_config.allowed_regimes,
            stale_after_seconds=self.mean_reversion_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spread_snapshot(
            payload.snapshot,
            require_valid_quote=self.mean_reversion_config.require_valid_quote,
            require_edge=self.mean_reversion_config.require_tradeable_edge,
        ):
            return None

        if not self._passes_spread_type_filters(payload.snapshot):
            return None

        if not self._passes_mean_reversion_filters(payload):
            return None

        if not self._passes_confirmation_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.mean_reversion_config.min_score:
            return None

        if breakdown.confidence < self.mean_reversion_config.min_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "spread_mean_reversion_signal",
                    f"side:{payload.side.value}",
                    f"spread_bias:{payload.spread_bias}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spreads_setup_family": "spread_mean_reversion",
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
            "spread_bias": payload.spread_bias,
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
            setup_type=self.mean_reversion_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.mean_reversion_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> SpreadMeanReversionPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._extract_payload")
        snapshot = self.resolve_spread_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        edge = self._edge(snapshot)
        if edge is None or edge == DECIMAL_ZERO:
            return None

        side = self._side(snapshot)
        if self.mean_reversion_config.require_directional_side and not is_directional_side(side):
            return None

        spread_bias = self._spread_bias(snapshot, side)
        if spread_bias == "UNKNOWN":
            return None

        event_time = (
            extract_timestamp(snapshot.to_signal_payload())
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "spread_mean_reversion_context",
            f"spread_type:{normalize_label(snapshot.spread_type)}",
            f"spread_bias:{spread_bias}",
            f"edge:{edge}",
            f"abs_zscore:{snapshot.abs_zscore}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        return SpreadMeanReversionPayload(
            snapshot=snapshot,
            side=side,
            spread_bias=spread_bias,
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
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._edge")
        for candidate in (
            snapshot.funding_adjusted_spread,
            snapshot.net_edge,
            snapshot.net_edge_bps,
            snapshot.basis,
            snapshot.spread_bps,
        ):
            if candidate is not None:
                return candidate
        return None

    def _side(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._side")
        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            return cross_exchange_to_signal_side(
                snapshot.raw_opportunity or snapshot.to_signal_payload()
            )

        return basis_to_signal_side(snapshot.to_signal_payload())

    def _spread_bias(
        self,
        snapshot: SpreadCompositeSnapshot,
        side: SignalSide,
    ) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._spread_bias")
        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            if side is SignalSide.LONG:
                return "LONG_A_SHORT_B"
            if side is SignalSide.SHORT:
                return "SHORT_A_LONG_B"
            return "UNKNOWN"

        bias = basis_to_bias(snapshot.to_signal_payload())
        return bias or "UNKNOWN"

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_spread_type_filters(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._passes_spread_type_filters")
        if snapshot.spread_type is None:
            return self.mean_reversion_config.allow_unknown_spread_type

        if snapshot.spread_type is SpreadType.SPOT_FUTURES:
            return self.mean_reversion_config.allow_spot_futures

        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            return self.mean_reversion_config.allow_cross_exchange

        return self.mean_reversion_config.allow_unknown_spread_type

    def _passes_mean_reversion_filters(
        self,
        payload: SpreadMeanReversionPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._passes_mean_reversion_filters")
        snapshot = payload.snapshot

        if self.mean_reversion_config.require_valid_quote and not snapshot.is_quote_valid:
            return False

        if self.mean_reversion_config.require_tradeable_edge and not has_tradeable_edge(
            snapshot.to_signal_payload()
        ):
            return False

        if payload.abs_zscore < self.mean_reversion_config.entry_zscore:
            return False

        if payload.abs_zscore >= self.mean_reversion_config.stop_zscore:
            return False

        if payload.abs_edge < self.mean_reversion_config.min_abs_edge:
            return False

        if snapshot.spread_bps is not None:
            if abs(snapshot.spread_bps) < self.mean_reversion_config.min_abs_spread_bps:
                return False

        if self.mean_reversion_config.allowed_regimes:
            if normalize_label(snapshot.regime) not in {
                item.lower()
                for item in self.mean_reversion_config.allowed_regimes
            }:
                return False

        return True

    def _passes_confirmation_filters(
        self,
        payload: SpreadMeanReversionPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._passes_confirmation_filters")
        signal = payload.snapshot.raw_signal

        if self.mean_reversion_config.require_mean_reversion_signal:
            if not is_mean_reversion_signal(signal):
                return False

        if is_widening_signal(signal):
            if not self.mean_reversion_config.allow_widening_entry:
                return False

            if self.mean_reversion_config.widening_requires_wait:
                return False

        if is_regime_shift_signal(signal) and not self.mean_reversion_config.allow_regime_shift_entry:
            return False

        if is_anomaly_signal(signal) and not self.mean_reversion_config.allow_anomaly_entry:
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: SpreadMeanReversionPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        z_component = zscore_component(
            snapshot.to_signal_payload(),
            entry_zscore=self.mean_reversion_config.entry_zscore,
            stop_zscore=self.mean_reversion_config.stop_zscore,
        )
        edge_scale = max(
            self.mean_reversion_config.min_abs_edge,
            self.mean_reversion_config.min_abs_spread_bps,
            DECIMAL_ONE,
        )
        e_component = edge_component(
            snapshot.to_signal_payload(),
            min_edge=edge_scale,
            scale=edge_scale * Decimal("3"),
        )
        r_component = regime_component(snapshot.to_signal_payload())
        c_component = self._confirmation_component(snapshot)
        q_component = quote_component(snapshot.to_signal_payload())
        f_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.mean_reversion_config.stale_feature_max_age_seconds,
        )

        components = {
            "zscore": z_component,
            "edge": e_component,
            "regime": r_component,
            "confirmation": c_component,
            "quote": q_component,
            "freshness": f_component,
        }
        weights = {
            "zscore": self.mean_reversion_config.score_zscore_weight,
            "edge": self.mean_reversion_config.score_edge_weight,
            "regime": self.mean_reversion_config.score_regime_weight,
            "confirmation": self.mean_reversion_config.score_confirmation_weight,
            "quote": self.mean_reversion_config.score_quote_weight,
            "freshness": self.mean_reversion_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=z_component)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, z_component),
            context=average_score(e_component, r_component),
            confirmation=average_score(c_component, q_component),
            freshness=f_component,
            primary_weight=self.mean_reversion_config.confidence_primary_weight,
            context_weight=self.mean_reversion_config.confidence_context_weight,
            confirmation_weight=self.mean_reversion_config.confidence_confirmation_weight,
            freshness_weight=self.mean_reversion_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "spread_mean_reversion_context",
            f"spread_bias:{payload.spread_bias}",
            f"side:{payload.side.value}",
            f"abs_zscore:{payload.abs_zscore}",
            f"abs_edge:{payload.abs_edge}",
        ]

        if payload.abs_zscore >= self.mean_reversion_config.entry_zscore:
            score += self.mean_reversion_config.zscore_entry_bonus
            confirmations.append("zscore_entry_threshold_passed")

        if payload.abs_edge >= (
            self.mean_reversion_config.min_abs_edge
            * self.mean_reversion_config.strong_edge_multiplier
        ):
            score += self.mean_reversion_config.strong_edge_bonus
            confirmations.append("strong_spread_edge")

        if snapshot.regime is SpreadRegime.EXTREME:
            score += self.mean_reversion_config.extreme_regime_bonus
            confirmations.append("extreme_spread_regime")

        if snapshot.regime is SpreadRegime.DISLOCATED:
            score += self.mean_reversion_config.dislocated_regime_bonus
            confirmations.append("dislocated_spread_regime")

        if is_mean_reversion_signal(snapshot.raw_signal):
            score += self.mean_reversion_config.mean_reversion_confirmation_bonus
            confirmations.append("mean_reversion_signal_confirmation")

        if is_anomaly_signal(snapshot.raw_signal):
            score += self.mean_reversion_config.anomaly_confirmation_bonus
            confirmations.append("spread_anomaly_confirmation")

        if is_regime_shift_signal(snapshot.raw_signal):
            score += self.mean_reversion_config.regime_shift_confirmation_bonus
            confirmations.append("regime_shift_confirmation")

        if snapshot.spread_bps is not None:
            reasons.append(f"spread_bps:{snapshot.spread_bps}")

        if snapshot.basis is not None:
            reasons.append(f"basis:{snapshot.basis}")

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

    def _confirmation_component(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._confirmation_component")
        signal = snapshot.raw_signal

        components = {
            "mean_reversion": 1.0 if is_mean_reversion_signal(signal) else 0.0,
            "regime_shift": 0.75 if is_regime_shift_signal(signal) else 0.0,
            "anomaly": 0.70 if is_anomaly_signal(signal) else 0.0,
            "not_widening": 0.0 if is_widening_signal(signal) else 1.0,
            "quote": quote_component(snapshot.to_signal_payload()),
        }

        return weighted_score(
            components,
            {
                "mean_reversion": 0.35,
                "regime_shift": 0.18,
                "anomaly": 0.17,
                "not_widening": 0.15,
                "quote": 0.15,
            },
            default=0.0,
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: SpreadMeanReversionPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._source_features")
        features = [
            *spread_mean_reversion_source_features(),
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
        payload: SpreadMeanReversionPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._tags")
        tags = [
            self.mean_reversion_config.tag_spreads,
            self.mean_reversion_config.tag_mean_reversion,
            self.mean_reversion_config.tag_spread_mean_reversion,
            self.mean_reversion_config.tag_zscore_reversion,
            self.mean_reversion_config.tag_dislocation,
            f"side:{payload.side.value}",
            f"bias:{payload.spread_bias.lower()}",
        ]

        if payload.side is SignalSide.SHORT:
            tags.append(self.mean_reversion_config.tag_short_spread)

        if payload.side is SignalSide.LONG:
            tags.append(self.mean_reversion_config.tag_long_spread)

        if payload.snapshot.spread_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spread_type)}")

        if payload.snapshot.regime is not None:
            tags.append(f"regime:{normalize_label(payload.snapshot.regime)}")

        if payload.snapshot.signal_type is not None:
            tags.append(f"signal:{normalize_label(payload.snapshot.signal_type)}")

        return list(dict.fromkeys(tags))

    def _leg_semantics(
        self,
        payload: SpreadMeanReversionPayload,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._leg_semantics")
        snapshot = payload.snapshot

        if snapshot.spread_type is SpreadType.CROSS_EXCHANGE:
            return {
                "spread_bias": payload.spread_bias,
                "spread_type": SpreadType.CROSS_EXCHANGE.value,
                "exchange_a": snapshot.exchange_a,
                "exchange_b": snapshot.exchange_b,
                "market_type_a": snapshot.market_type_a,
                "market_type_b": snapshot.market_type_b,
                "note": (
                    "For cross-exchange spreads, exact long/short leg construction "
                    "is owned by SignalProcessor/SignalBuilder."
                ),
            }

        if payload.spread_bias == "SHORT_BASIS":
            primary_action = "short_futures_or_basis_leg"
            hedge_action = "long_spot_or_underlying_leg"
        elif payload.spread_bias == "LONG_BASIS":
            primary_action = "long_futures_or_basis_leg"
            hedge_action = "short_spot_or_underlying_leg"
        else:
            primary_action = "generic_spread_reversion"
            hedge_action = "generic_spread_hedge"

        return {
            "spread_bias": payload.spread_bias,
            "spread_type": normalize_label(snapshot.spread_type),
            "primary_action": primary_action,
            "hedge_action": hedge_action,
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
                "construction is owned by SignalProcessor/SignalBuilder."
            ),
        }

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpreadMeanReversionStrategy._execution_hints")
        return {
            "entry_offset_bps": self.mean_reversion_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.mean_reversion_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.mean_reversion_config.execution_take_profit_bps_hint,
            "mean_reversion_tp_multiplier": (
                self.mean_reversion_config.mean_reversion_tp_multiplier_hint
            ),
        }