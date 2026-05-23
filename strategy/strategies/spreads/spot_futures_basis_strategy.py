# trading_system/strategy/strategies/spreads/spot_futures_basis_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import (
    InstrumentType,
    SpreadRegime,
    SpreadType,
)
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
    decimal_abs,
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
    spot_futures_contract_error,
    spot_futures_source_features,
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
class SpotFuturesBasisPayload:
    """
    Normalized strategy-level payload for spot/futures basis mean reversion.

    Direction convention:
    - positive funding-adjusted edge / basis -> SHORT_BASIS -> SignalSide.SHORT;
    - negative funding-adjusted edge / basis -> LONG_BASIS -> SignalSide.LONG.

    Concrete leg construction remains a SignalProcessor / SignalBuilder concern.
    """
    _logger = logging.getLogger(__name__ + ".SpotFuturesBasisPayload")

    snapshot: SpreadCompositeSnapshot
    side: SignalSide
    basis_bias: str

    edge: Decimal
    abs_edge: Decimal
    abs_zscore: Decimal

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpotFuturesBasisStrategyConfig(SpreadsStrategyConfig):
    """
    Unified spot/futures basis strategy config.

    Strategy idea:
    - read normalized spot/futures spread snapshot/signal from StrategyContext;
    - require valid spot/futures contract and quote quality;
    - require basis/funding-adjusted edge and z-score dislocation;
    - map basis bias to internal StrategySignal side;
    - return StrategySignal only.
    """
    _logger = logging.getLogger(__name__ + ".SpotFuturesBasisStrategyConfig")

    entry_zscore: Decimal = Decimal("2.0")
    stop_zscore: Decimal = Decimal("4.5")

    min_funding_adjusted_edge: Decimal = Decimal("0")
    min_basis_abs: Decimal = Decimal("0")

    require_valid_quote: bool = True
    require_snapshot_edge: bool = True
    require_spot_futures_contract: bool = True

    require_mean_reversion_signal: bool = False
    require_regime_shift_confirmation: bool = False
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
    allowed_spot_exchanges: set[str] = field(default_factory=set)
    allowed_futures_exchanges: set[str] = field(default_factory=set)

    score_zscore_weight: float = 0.30
    score_edge_weight: float = 0.24
    score_regime_weight: float = 0.16
    score_confirmation_weight: float = 0.15
    score_quote_weight: float = 0.10
    score_freshness_weight: float = 0.05

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    zscore_entry_bonus: float = 0.04
    extreme_regime_bonus: float = 0.04
    dislocated_regime_bonus: float = 0.05
    funding_adjusted_bonus: float = 0.04
    mean_reversion_confirmation_bonus: float = 0.04
    anomaly_confirmation_bonus: float = 0.03
    regime_shift_confirmation_bonus: float = 0.03

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    tag_spot_futures_basis: str = "spot_futures_basis"
    tag_basis_mean_reversion: str = "basis_mean_reversion"
    tag_funding_adjusted: str = "funding_adjusted"
    tag_short_basis: str = "short_basis"
    tag_long_basis: str = "long_basis"
    tag_zscore_entry: str = "zscore_entry"

    execution_entry_offset_bps_hint: float | None = None
    execution_stop_buffer_bps_hint: float | None = None
    execution_take_profit_bps_hint: float | None = None
    basis_tp_multiplier_hint: float | None = None

    required_spreads_features: tuple[str, ...] = (
        SPREADS_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategyConfig.validate")
        SpreadsStrategyConfig.validate(self)

        if self.entry_zscore <= DECIMAL_ZERO:
            raise StrategyConfigError("entry_zscore must be > 0")

        if self.stop_zscore <= self.entry_zscore:
            raise StrategyConfigError("stop_zscore must be greater than entry_zscore")

        if self.min_funding_adjusted_edge < DECIMAL_ZERO:
            raise StrategyConfigError("min_funding_adjusted_edge must be >= 0")

        if self.min_basis_abs < DECIMAL_ZERO:
            raise StrategyConfigError("min_basis_abs must be >= 0")

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
            "extreme_regime_bonus": self.extreme_regime_bonus,
            "dislocated_regime_bonus": self.dislocated_regime_bonus,
            "funding_adjusted_bonus": self.funding_adjusted_bonus,
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
            "basis_tp_multiplier_hint": self.basis_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_spot_futures_basis",
            "tag_basis_mean_reversion",
            "tag_funding_adjusted",
            "tag_short_basis",
            "tag_long_basis",
            "tag_zscore_entry",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spreads_features:
            raise StrategyConfigError("required_spreads_features cannot be empty")


class SpotFuturesBasisStrategy(SpreadsTradingStrategy):
    """
    Unified spot/futures basis mean-reversion strategy.

    Input:
        StrategyContext with FeatureSource.SPREADS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """
    _logger = logging.getLogger(__name__ + ".SpotFuturesBasisStrategy")

    component_namespace = "strategy.spreads.spot_futures_basis"
    category: StrategyCategory = StrategyCategory.SPREADS
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spreads_config: SpotFuturesBasisStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy.__init__")
        resolved_spreads_config = spreads_config or SpotFuturesBasisStrategyConfig()
        resolved_spreads_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spreads_config=resolved_spreads_config,
            service_name=service_name,
        )

        self.basis_config: SpotFuturesBasisStrategyConfig = resolved_spreads_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy.strategy_name")
        return "spot_futures_basis"

    @property
    def metadata(self) -> StrategyMetadata:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy.metadata")
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPREADS,
            timeframe=Timeframe.M1,
            tags=[
                self.basis_config.tag_spreads,
                self.basis_config.tag_spot_futures,
                self.basis_config.tag_basis,
                self.basis_config.tag_spot_futures_basis,
                self.basis_config.tag_basis_mean_reversion,
                self.basis_config.tag_funding_adjusted,
                "analytics_spreads",
            ],
            version="2.0.0",
            description=(
                "Interprets spot/futures basis dislocation and funding-adjusted "
                "edge from normalized StrategyContext and returns internal StrategySignal."
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
                "strategy_type": "spot_futures_basis",
                "base_class": "SpreadsTradingStrategy",
                "canonical_payload": "SpreadCompositeSnapshot",
                "uses_spot_futures": True,
                "uses_funding_adjusted_edge": True,
                "uses_zscore": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(self.basis_config.required_spreads_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy.generate_signal")
        self.validate_context_requirements(context)

        if not self.has_any_spreads_data(
            context,
            tuple(self.basis_config.required_spreads_features),
        ):
            return None

        if self.has_stale_spreads_features(
            context,
            tuple(self.basis_config.required_spreads_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.basis_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = spread_quality_filter_reason(
            payload.snapshot.to_signal_payload(),
            min_score=self.basis_config.min_score,
            min_confidence=self.basis_config.min_confidence,
            require_valid_quote=self.basis_config.require_valid_quote,
            require_edge=self.basis_config.require_snapshot_edge,
            allowed_regimes=self.basis_config.allowed_regimes,
            stale_after_seconds=self.basis_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spread_snapshot(
            payload.snapshot,
            require_valid_quote=self.basis_config.require_valid_quote,
            require_edge=self.basis_config.require_snapshot_edge,
        ):
            return None

        if not self._passes_contract_filters(payload.snapshot):
            return None

        if not self._passes_basis_filters(payload):
            return None

        if not self._passes_confirmation_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.basis_config.min_score:
            return None

        if breakdown.confidence < self.basis_config.min_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "spot_futures_basis_signal",
                    f"side:{payload.side.value}",
                    f"basis_bias:{payload.basis_bias}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "spreads_setup_family": "spot_futures_basis",
            "spreads_strategy_version": "2.0.0",
            "contract": "spreads",
            "contract_version": "strategy-domain-v1",
            "primary_section": "snapshot",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "spread_type": normalize_label(payload.snapshot.spread_type),
            "basis_bias": payload.basis_bias,
            "mapped_side": payload.side.value,
            "edge": str(payload.edge),
            "abs_edge": str(payload.abs_edge),
            "basis": str(payload.snapshot.basis)
            if payload.snapshot.basis is not None
            else None,
            "funding_adjusted_spread": str(payload.snapshot.funding_adjusted_spread)
            if payload.snapshot.funding_adjusted_spread is not None
            else None,
            "spread_bps": str(payload.snapshot.spread_bps)
            if payload.snapshot.spread_bps is not None
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
            setup_type=self.basis_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.basis_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> SpotFuturesBasisPayload | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._extract_payload")
        snapshot = self.resolve_spread_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        edge = self._basis_edge(snapshot)
        if edge is None or edge == DECIMAL_ZERO:
            return None

        side = basis_to_signal_side(snapshot.to_signal_payload())
        if not is_directional_side(side):
            return None

        bias = basis_to_bias(snapshot.to_signal_payload())
        if bias is None:
            return None

        event_time = (
            extract_timestamp(snapshot.to_signal_payload())
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "spot_futures_basis_context",
            f"spread_type:{normalize_label(snapshot.spread_type)}",
            f"basis_bias:{bias}",
            f"edge:{edge}",
            f"abs_zscore:{snapshot.abs_zscore}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        return SpotFuturesBasisPayload(
            snapshot=snapshot,
            side=side,
            basis_bias=bias,
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

    def _basis_edge(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> Decimal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._basis_edge")
        for candidate in (
            snapshot.funding_adjusted_spread,
            snapshot.basis,
            snapshot.spread_bps,
        ):
            if candidate is not None:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_contract_filters(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._passes_contract_filters")
        if not self.basis_config.require_spot_futures_contract:
            return True

        if snapshot.spread_type is not None and snapshot.spread_type is not SpreadType.SPOT_FUTURES:
            return False

        contract_error = spot_futures_contract_error(snapshot.to_signal_payload())
        if contract_error is not None:
            return False

        if self.basis_config.allowed_spot_exchanges:
            if snapshot.exchange_a not in {
                exchange.lower()
                for exchange in self.basis_config.allowed_spot_exchanges
            }:
                return False

        if self.basis_config.allowed_futures_exchanges:
            if snapshot.exchange_b not in {
                exchange.lower()
                for exchange in self.basis_config.allowed_futures_exchanges
            }:
                return False

        return True

    def _passes_basis_filters(
        self,
        payload: SpotFuturesBasisPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._passes_basis_filters")
        snapshot = payload.snapshot

        if self.basis_config.require_valid_quote and not snapshot.is_quote_valid:
            return False

        if self.basis_config.require_snapshot_edge and not has_tradeable_edge(
            snapshot.to_signal_payload()
        ):
            return False

        if payload.abs_zscore < self.basis_config.entry_zscore:
            return False

        if payload.abs_zscore >= self.basis_config.stop_zscore:
            return False

        if self.basis_config.min_funding_adjusted_edge > DECIMAL_ZERO:
            funding_edge = decimal_abs(snapshot.funding_adjusted_spread)
            if funding_edge < self.basis_config.min_funding_adjusted_edge:
                return False

        if self.basis_config.min_basis_abs > DECIMAL_ZERO:
            basis_abs = decimal_abs(snapshot.basis or snapshot.spread_bps)
            if basis_abs < self.basis_config.min_basis_abs:
                return False

        if self.basis_config.allowed_regimes:
            if normalize_label(snapshot.regime) not in {
                item.lower()
                for item in self.basis_config.allowed_regimes
            }:
                return False

        return True

    def _passes_confirmation_filters(
        self,
        payload: SpotFuturesBasisPayload,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._passes_confirmation_filters")
        signal = payload.snapshot.raw_signal

        if self.basis_config.require_mean_reversion_signal:
            if not is_mean_reversion_signal(signal):
                return False

        if self.basis_config.require_regime_shift_confirmation:
            if not is_regime_shift_signal(signal):
                return False

        if is_widening_signal(signal):
            if not self.basis_config.allow_widening_entry:
                return False

            if self.basis_config.widening_requires_wait:
                return False

        if is_regime_shift_signal(signal) and not self.basis_config.allow_regime_shift_entry:
            return False

        if is_anomaly_signal(signal) and not self.basis_config.allow_anomaly_entry:
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: SpotFuturesBasisPayload,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._build_score_breakdown")
        snapshot = payload.snapshot

        z_component = zscore_component(
            snapshot.to_signal_payload(),
            entry_zscore=self.basis_config.entry_zscore,
            stop_zscore=self.basis_config.stop_zscore,
        )
        edge_scale = max(
            self.basis_config.min_funding_adjusted_edge,
            self.basis_config.min_basis_abs,
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
            stale_after_seconds=self.basis_config.stale_feature_max_age_seconds,
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
            "zscore": self.basis_config.score_zscore_weight,
            "edge": self.basis_config.score_edge_weight,
            "regime": self.basis_config.score_regime_weight,
            "confirmation": self.basis_config.score_confirmation_weight,
            "quote": self.basis_config.score_quote_weight,
            "freshness": self.basis_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=z_component)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, z_component),
            context=average_score(e_component, r_component),
            confirmation=average_score(c_component, q_component),
            freshness=f_component,
            primary_weight=self.basis_config.confidence_primary_weight,
            context_weight=self.basis_config.confidence_context_weight,
            confirmation_weight=self.basis_config.confidence_confirmation_weight,
            freshness_weight=self.basis_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "spot_futures_basis_context",
            f"basis_bias:{payload.basis_bias}",
            f"side:{payload.side.value}",
            f"abs_zscore:{payload.abs_zscore}",
            f"abs_edge:{payload.abs_edge}",
        ]

        if payload.abs_zscore >= self.basis_config.entry_zscore:
            score += self.basis_config.zscore_entry_bonus
            confirmations.append("zscore_entry_threshold_passed")

        if snapshot.regime is SpreadRegime.EXTREME:
            score += self.basis_config.extreme_regime_bonus
            confirmations.append("extreme_spread_regime")

        if snapshot.regime is SpreadRegime.DISLOCATED:
            score += self.basis_config.dislocated_regime_bonus
            confirmations.append("dislocated_spread_regime")

        if snapshot.funding_adjusted_spread is not None:
            score += self.basis_config.funding_adjusted_bonus
            confirmations.append("funding_adjusted_edge_available")

        if is_mean_reversion_signal(snapshot.raw_signal):
            score += self.basis_config.mean_reversion_confirmation_bonus
            confirmations.append("mean_reversion_signal_confirmation")

        if is_anomaly_signal(snapshot.raw_signal):
            score += self.basis_config.anomaly_confirmation_bonus
            confirmations.append("spread_anomaly_confirmation")

        if is_regime_shift_signal(snapshot.raw_signal):
            score += self.basis_config.regime_shift_confirmation_bonus
            confirmations.append("regime_shift_confirmation")

        if snapshot.spread_bps is not None:
            reasons.append(f"spread_bps:{snapshot.spread_bps}")

        if snapshot.basis is not None:
            reasons.append(f"basis:{snapshot.basis}")

        if snapshot.funding_adjusted_spread is not None:
            reasons.append(f"funding_adjusted_spread:{snapshot.funding_adjusted_spread}")

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
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._confirmation_component")
        signal = snapshot.raw_signal

        components = {
            "mean_reversion": 1.0 if is_mean_reversion_signal(signal) else 0.0,
            "regime_shift": 0.75 if is_regime_shift_signal(signal) else 0.0,
            "anomaly": 0.70 if is_anomaly_signal(signal) else 0.0,
            "not_widening": 0.0 if is_widening_signal(signal) else 1.0,
        }

        return weighted_score(
            components,
            {
                "mean_reversion": 0.40,
                "regime_shift": 0.20,
                "anomaly": 0.20,
                "not_widening": 0.20,
            },
            default=0.0,
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: SpotFuturesBasisPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._source_features")
        features = [
            *spot_futures_source_features(),
            SPREADS_FEATURES.SNAPSHOT,
            SPREADS_FEATURES.SIGNAL,
            SPREADS_FEATURES.SPREAD_TYPE,
            SPREADS_FEATURES.SYMBOL,
            SPREADS_FEATURES.EXCHANGE_A,
            SPREADS_FEATURES.EXCHANGE_B,
            SPREADS_FEATURES.MARKET_TYPE_A,
            SPREADS_FEATURES.MARKET_TYPE_B,
            SPREADS_FEATURES.SPREAD_BPS,
            SPREADS_FEATURES.BASIS,
            SPREADS_FEATURES.FUNDING_ADJUSTED_SPREAD,
            SPREADS_FEATURES.ZSCORE,
            SPREADS_FEATURES.REGIME,
            SPREADS_FEATURES.DIRECTION,
            SPREADS_FEATURES.QUOTE_VALIDITY,
            SPREADS_FEATURES.HAS_EDGE,
            SPREADS_FEATURES.CONFIDENCE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: SpotFuturesBasisPayload,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._tags")
        tags = [
            self.basis_config.tag_spreads,
            self.basis_config.tag_spot_futures,
            self.basis_config.tag_basis,
            self.basis_config.tag_spot_futures_basis,
            self.basis_config.tag_basis_mean_reversion,
            self.basis_config.tag_zscore_entry,
            f"side:{payload.side.value}",
            f"bias:{payload.basis_bias.lower()}",
        ]

        if payload.basis_bias == "SHORT_BASIS":
            tags.append(self.basis_config.tag_short_basis)

        if payload.basis_bias == "LONG_BASIS":
            tags.append(self.basis_config.tag_long_basis)

        if payload.snapshot.funding_adjusted_spread is not None:
            tags.append(self.basis_config.tag_funding_adjusted)

        if payload.snapshot.regime is not None:
            tags.append(f"regime:{normalize_label(payload.snapshot.regime)}")

        if payload.snapshot.signal_type is not None:
            tags.append(f"signal:{normalize_label(payload.snapshot.signal_type)}")

        return list(dict.fromkeys(tags))

    def _leg_semantics(
        self,
        payload: SpotFuturesBasisPayload,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._leg_semantics")
        snapshot = payload.snapshot

        if payload.basis_bias == "SHORT_BASIS":
            primary_action = "short_futures_or_basis_leg"
            hedge_action = "long_spot_or_underlying_leg"
        elif payload.basis_bias == "LONG_BASIS":
            primary_action = "long_futures_or_basis_leg"
            hedge_action = "short_spot_or_underlying_leg"
        else:
            primary_action = "unknown"
            hedge_action = "unknown"

        return {
            "basis_bias": payload.basis_bias,
            "primary_action": primary_action,
            "hedge_action": hedge_action,
            "spot_leg": {
                "exchange": snapshot.exchange_a,
                "market_type": snapshot.market_type_a,
                "symbol": snapshot.exchange_symbol_a,
                "instrument_type": InstrumentType.SPOT.value,
            },
            "futures_leg": {
                "exchange": snapshot.exchange_b,
                "market_type": snapshot.market_type_b,
                "symbol": snapshot.exchange_symbol_b,
                "instrument_type": "derivative",
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
            _strategy_logger.debug("Entering SpotFuturesBasisStrategy._execution_hints")
        return {
            "entry_offset_bps": self.basis_config.execution_entry_offset_bps_hint,
            "stop_buffer_bps": self.basis_config.execution_stop_buffer_bps_hint,
            "take_profit_bps": self.basis_config.execution_take_profit_bps_hint,
            "basis_tp_multiplier": self.basis_config.basis_tp_multiplier_hint,
        }