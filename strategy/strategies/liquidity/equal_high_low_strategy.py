# trading_system/strategy/strategies/liquidity/equal_high_low_strategy.py

from __future__ import annotations
import logging

from dataclasses import dataclass, field
from typing import Any

from analytics.liquidity.enums import (
    LiquidityBias,
    LiquidityLevelType,
    LiquiditySide,
)
from analytics.liquidity.models import (
    EqualLevel,
    LiquidityLevel,
    LiquidityMapSnapshot,
    StopCluster,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    LIQUIDITY_FEATURES,
    LiquidityStrategyConfig,
    LiquidityTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    best_zone_for_side,
    collect_targets_above,
    collect_targets_below,
    compactness_score,
    compactness_width_pct,
    confidence_from_components,
    distance_pct,
    distance_score,
    expected_equal_level_side,
    is_directional_side,
    is_equal_level,
    is_partially_swept_item,
    is_swept_item,
    is_terminal_item,
    is_valid_equal_reaction_level,
    level_quality,
    magnet_score_down,
    magnet_score_up,
    reference_price,
    serialize_for_metadata,
    signed_score,
    sweep_risk_down,
    sweep_risk_up,
    unit_score,
    weighted_score,
)
from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    FilterDecision,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
)
from ...exceptions import StrategyConfigError
from ...models import FilterResult, StrategyContext, StrategySignal, TargetPlan


@dataclass(slots=True)
class EqualLevelCandidate:
    """
    Internal equal highs / equal lows reaction candidate.

    Це локальний DTO для candidate selection, не runtime state.
    """
    _logger = logging.getLogger(__name__ + ".EqualLevelCandidate")

    side: SignalSide
    level: EqualLevel | LiquidityLevel
    edge: float
    target: LiquidityLevel | StopCluster | None = field(default=None)

    @property
    def level_price(self) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualLevelCandidate.level_price")
        return reference_price(self.level)


@dataclass(slots=True)
class EqualHighLowStrategyConfig(LiquidityStrategyConfig):
    """
    Unified equal highs / equal lows reaction strategy config.

    Strategy idea:
    - LONG from active sell-side equal lows below/near current price;
    - SHORT from active buy-side equal highs above/near current price;
    - swept / partially swept equal levels are ignored by default because they
      belong to StopHuntReversalStrategy;
    - return internal StrategySignal only; SignalProcessor owns final emission.
    """
    _logger = logging.getLogger(__name__ + ".EqualHighLowStrategyConfig")

    allow_swept_equal_levels: bool = False

    long_candidate_max_overshoot: float = 1.0030
    short_candidate_min_undershoot: float = 0.9970

    min_edge: float = 0.20
    max_level_distance_pct: float = 0.0450
    max_target_distance_pct: float = 0.0800

    fallback_stop_pct: float = 0.0040
    long_stop_offset: float = 0.9985
    short_stop_offset: float = 1.0015

    min_touches_count: int = 2
    min_reaction_count: int = 0

    require_valid_equal_level_type: bool = True
    require_directional_side_match: bool = True
    allow_signal_without_target: bool = True
    reject_terminal_targets: bool = True

    score_edge_weight: float = 0.34
    score_level_quality_weight: float = 0.26
    score_distance_weight: float = 0.16
    score_target_weight: float = 0.10
    score_zone_weight: float = 0.08
    score_context_weight: float = 0.06

    confidence_edge_weight: float = 0.30
    confidence_level_quality_weight: float = 0.26
    confidence_distance_weight: float = 0.16
    confidence_target_weight: float = 0.12
    confidence_context_weight: float = 0.10
    confidence_zone_weight: float = 0.06

    high_priority_score: float = 0.82
    critical_priority_score: float = 0.92

    tag_equal_high_low: str = "equal_high_low"
    tag_equal_lows: str = "equal_lows_reaction"
    tag_equal_highs: str = "equal_highs_reaction"
    tag_reaction: str = "liquidity_reaction"
    tag_reversal: str = "reversal"
    tag_structure_quality: str = "structure_quality"
    tag_target_available: str = "target_available"

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    required_liquidity_features: tuple[str, ...] = (
        LIQUIDITY_FEATURES.SNAPSHOT,
    )

    def validate(self) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategyConfig.validate")
        LiquidityStrategyConfig.validate(self)

        bounded_fields = {
            "long_candidate_max_overshoot": self.long_candidate_max_overshoot,
            "short_candidate_min_undershoot": self.short_candidate_min_undershoot,
            "min_edge": self.min_edge,
            "max_level_distance_pct": self.max_level_distance_pct,
            "max_target_distance_pct": self.max_target_distance_pct,
            "fallback_stop_pct": self.fallback_stop_pct,
            "high_priority_score": self.high_priority_score,
            "critical_priority_score": self.critical_priority_score,
        }

        for field_name, value in bounded_fields.items():
            if not 0.0 <= float(value) <= 2.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 2.0")

        if not 0.0 <= self.min_edge <= 1.0:
            raise StrategyConfigError("min_edge must be between 0.0 and 1.0")

        if self.max_level_distance_pct <= 0:
            raise StrategyConfigError("max_level_distance_pct must be > 0")

        if self.max_target_distance_pct <= 0:
            raise StrategyConfigError("max_target_distance_pct must be > 0")

        if self.fallback_stop_pct <= 0:
            raise StrategyConfigError("fallback_stop_pct must be > 0")

        if self.long_stop_offset <= 0:
            raise StrategyConfigError("long_stop_offset must be > 0")

        if self.short_stop_offset <= 0:
            raise StrategyConfigError("short_stop_offset must be > 0")

        if self.min_touches_count < 0:
            raise StrategyConfigError("min_touches_count must be >= 0")

        if self.min_reaction_count < 0:
            raise StrategyConfigError("min_reaction_count must be >= 0")

        weights = {
            "score_edge_weight": self.score_edge_weight,
            "score_level_quality_weight": self.score_level_quality_weight,
            "score_distance_weight": self.score_distance_weight,
            "score_target_weight": self.score_target_weight,
            "score_zone_weight": self.score_zone_weight,
            "score_context_weight": self.score_context_weight,
            "confidence_edge_weight": self.confidence_edge_weight,
            "confidence_level_quality_weight": self.confidence_level_quality_weight,
            "confidence_distance_weight": self.confidence_distance_weight,
            "confidence_target_weight": self.confidence_target_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_zone_weight": self.confidence_zone_weight,
        }

        for field_name, value in weights.items():
            if value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if (
            self.score_edge_weight
            + self.score_level_quality_weight
            + self.score_distance_weight
            + self.score_target_weight
            + self.score_zone_weight
            + self.score_context_weight
        ) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if (
            self.confidence_edge_weight
            + self.confidence_level_quality_weight
            + self.confidence_distance_weight
            + self.confidence_target_weight
            + self.confidence_context_weight
            + self.confidence_zone_weight
        ) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_equal_high_low",
            "tag_equal_lows",
            "tag_equal_highs",
            "tag_reaction",
            "tag_reversal",
            "tag_structure_quality",
            "tag_target_available",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_liquidity_features:
            raise StrategyConfigError("required_liquidity_features cannot be empty")

        for feature in self.required_liquidity_features:
            if not isinstance(feature, str) or not feature.strip():
                raise StrategyConfigError(
                    "required_liquidity_features cannot contain empty feature names"
                )


class EqualHighLowStrategy(LiquidityTradingStrategy):
    """
    Unified equal highs / equal lows reaction strategy.

    Input:
        StrategyContext with FeatureSource.LIQUIDITY domain data and LiquidityMapSnapshot.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    """
    _logger = logging.getLogger(__name__ + ".EqualHighLowStrategy")

    component_namespace = "strategy.liquidity.equal_high_low"
    category: StrategyCategory = StrategyCategory.LIQUIDITY
    default_setup_type: SetupType = SetupType.MEAN_REVERSION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        liquidity_config: EqualHighLowStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy.__init__")
        resolved_liquidity_config = liquidity_config or EqualHighLowStrategyConfig()
        resolved_liquidity_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            liquidity_config=resolved_liquidity_config,
            service_name=service_name,
        )

        self.equal_config: EqualHighLowStrategyConfig = resolved_liquidity_config

    @property
    def strategy_name(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy.strategy_name")
        return "equal_high_low_strategy"

    def required_features(self) -> set[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy.required_features")
        base_required = super().required_features()
        return set(base_required).union(self.equal_config.required_liquidity_features)

    async def generate_signal(
            self,
            context: StrategyContext,
    ) -> StrategySignal | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy.generate_signal")
        self.validate_context_requirements(context)

        snapshot = self.liquidity_snapshot(context)
        if snapshot is None:
            self.remember_no_signal(
                "missing_liquidity_snapshot_contract",
                liquidity_domain_keys=sorted(self.liquidity_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        if not self.base_context_is_valid(context=context, snapshot=snapshot):
            self.remember_no_signal(
                "invalid_liquidity_base_context",
                liquidity_domain_keys=sorted(self.liquidity_domain(context).keys()),
                snapshot=serialize_for_metadata(snapshot),
                required_features=sorted(self.required_features()),
            )
            return None

        current_price = self.current_price(context=context, snapshot=snapshot)
        if current_price is None or current_price <= 0:
            self.remember_no_signal(
                "invalid_liquidity_current_price",
                current_price=current_price,
                liquidity_domain_keys=sorted(self.liquidity_domain(context).keys()),
            )
            return None

        filters = self._run_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )
        blocked_filters = [item for item in filters if item.blocked]
        if blocked_filters:
            self.remember_no_signal(
                "equal_high_low_pre_filters_blocked",
                filters=[item.to_dict() for item in filters],
                blocked_filters=[item.name for item in blocked_filters],
                current_price=current_price,
            )
            return None

        candidate = self._find_best_candidate(
            snapshot=snapshot,
            current_price=current_price,
        )
        if candidate is None:
            self.remember_no_signal(
                "equal_high_low_candidate_not_found",
                current_price=current_price,
                equal_levels=serialize_for_metadata(
                    getattr(snapshot, "equal_levels", None)
                ),
                active_levels=serialize_for_metadata(
                    getattr(snapshot, "active_levels", None)
                ),
                stop_clusters=serialize_for_metadata(
                    getattr(snapshot, "stop_clusters", None)
                ),
            )
            return None

        side = candidate.side
        if not is_directional_side(side):
            self.remember_no_signal(
                "equal_high_low_side_not_directional",
                side=serialize_for_metadata(side),
                candidate=serialize_for_metadata(candidate),
            )
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
            candidate=candidate,
        )

        if breakdown.score < self.equal_config.min_signal_score:
            self.remember_no_signal(
                "equal_high_low_score_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_score=self.equal_config.min_signal_score,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.equal_config.min_signal_confidence:
            self.remember_no_signal(
                "equal_high_low_confidence_below_minimum",
                score=breakdown.score,
                confidence=breakdown.confidence,
                min_signal_confidence=self.equal_config.min_signal_confidence,
                score_breakdown=breakdown.to_dict(),
            )
            return None

        stop_loss = self._resolve_stop_price(
            side=side,
            current_price=current_price,
            level=candidate.level,
        )
        take_profit = (
            reference_price(candidate.target)
            if candidate.target is not None
            else None
        )

        target_plans = self._target_plans(
            current_price=current_price,
            side=side,
            target=candidate.target,
            stop_loss=stop_loss,
        )

        reasons = list(
            dict.fromkeys(
                [
                    self._build_primary_reason(
                        level=candidate.level,
                        side=side,
                        current_price=current_price,
                    ),
                    self._build_target_reason(candidate.target),
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(
            dict.fromkeys(
                [
                    *self._build_confirmations(
                        snapshot=snapshot,
                        level=candidate.level,
                        side=side,
                        current_price=current_price,
                        target=candidate.target,
                    ),
                    *breakdown.confirmations,
                ]
            )
        )
        source_features = self._source_features(candidate)

        metadata = {
            "liquidity_setup_family": "equal_high_low_reaction",
            "liquidity_strategy_version": "2.0.0",
            "contract": "liquidity",
            "primary_section": "snapshot",
            "strategy_contract_role": "decision_module",
            "score_breakdown": breakdown.to_dict(),
            "tags": self._tags(candidate=candidate, snapshot=snapshot),
            "side": side.value,
            "current_price": current_price,
            "level": self._level_metadata(candidate.level),
            "level_price": candidate.level_price,
            "level_type": serialize_for_metadata(
                getattr(candidate.level, "level_type", None)
            ),
            "target": self._target_metadata(candidate.target),
            "target_price": take_profit,
            "stop_loss": stop_loss,
            "edge": candidate.edge,
            "bias": serialize_for_metadata(getattr(snapshot, "bias", None)),
            "liquidity_pressure_score": signed_score(
                getattr(snapshot, "liquidity_pressure_score", 0.0)
            ),
            "above_liquidity_score": unit_score(
                getattr(snapshot, "above_liquidity_score", 0.0)
            ),
            "below_liquidity_score": unit_score(
                getattr(snapshot, "below_liquidity_score", 0.0)
            ),
            "magnet_score_up": magnet_score_up(snapshot),
            "magnet_score_down": magnet_score_down(snapshot),
            "sweep_risk_up": sweep_risk_up(snapshot),
            "sweep_risk_down": sweep_risk_down(snapshot),
            "filters": [item.to_dict() for item in filters],
            "target_plans": [plan.to_dict() for plan in target_plans],
        }

        return self.build_liquidity_signal(
            context=context,
            side=side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.equal_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self._resolve_priority(
                score=breakdown.score,
                confidence=breakdown.confidence,
            ),
            snapshot=snapshot,
            current_price=current_price,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _run_pre_filters(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[FilterResult]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._run_pre_filters")
        results = self.run_common_pre_filters(
            context=context,
            snapshot=snapshot,
            current_price=current_price,
        )

        valid_levels = self._valid_equal_levels(
            snapshot=snapshot,
            current_price=current_price,
        )

        if valid_levels:
            results.append(
                FilterResult(
                    name="equal_high_low_candidate_presence",
                    decision=FilterDecision.PASS,
                    reason=f"Valid equal highs/lows candidates found: {len(valid_levels)}",
                )
            )
        else:
            results.append(
                FilterResult(
                    name="equal_high_low_candidate_presence",
                    decision=FilterDecision.BLOCK,
                    reason="No valid active equal highs/lows candidates found",
                )
            )

        for item in results:
            item.validate()

        return results

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_best_candidate(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> EqualLevelCandidate | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._find_best_candidate")
        candidates: list[EqualLevelCandidate] = []

        for level in self._valid_equal_levels(
            snapshot=snapshot,
            current_price=current_price,
        ):
            side = self._side_from_equal_level(level=level, current_price=current_price)
            if not is_directional_side(side):
                continue

            edge = self._edge_for_level(
                snapshot=snapshot,
                level=level,
                side=side,
                current_price=current_price,
            )
            if edge < self.equal_config.min_edge:
                continue

            target = self._target_for_side(
                snapshot=snapshot,
                current_price=current_price,
                side=side,
            )

            if target is None and not self.equal_config.allow_signal_without_target:
                continue

            candidates.append(
                EqualLevelCandidate(
                    side=side,
                    level=level,
                    edge=edge,
                    target=target,
                )
            )

        if not candidates:
            return None

        return max(candidates, key=self._candidate_rank)

    def _valid_equal_levels(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
    ) -> list[EqualLevel | LiquidityLevel]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._valid_equal_levels")
        levels = list(getattr(snapshot, "equal_levels", []) or [])

        result: list[EqualLevel | LiquidityLevel] = []
        for level in levels:
            if not self._is_valid_equal_reaction_level(level):
                continue

            if not self._level_distance_ok(level=level, current_price=current_price):
                continue

            if self.equal_config.min_touches_count > 0:
                touches = int(getattr(level, "touches_count", 0) or 0)
                if touches < self.equal_config.min_touches_count:
                    continue

            if self.equal_config.min_reaction_count > 0:
                reactions = int(getattr(level, "reaction_count", 0) or 0)
                if reactions < self.equal_config.min_reaction_count:
                    continue

            result.append(level)

        return result

    def _is_valid_equal_reaction_level(
        self,
        level: EqualLevel | LiquidityLevel,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._is_valid_equal_reaction_level")
        if self.equal_config.require_valid_equal_level_type:
            if not is_equal_level(level):
                return False

        if not is_valid_equal_reaction_level(
            level,
            allow_swept=self.equal_config.allow_swept_equal_levels,
        ):
            return False

        if self.equal_config.require_directional_side_match:
            expected = expected_equal_level_side(level)
            if expected is not None and getattr(level, "side", None) != expected:
                return False

        if not self.equal_config.allow_swept_equal_levels:
            if is_swept_item(level) or is_partially_swept_item(level):
                return False

        return True

    def _level_distance_ok(
        self,
        *,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._level_distance_ok")
        price = reference_price(level)
        if price <= 0 or current_price <= 0:
            return False

        if distance_pct(price, current_price) > self.equal_config.max_level_distance_pct:
            return False

        side = self._side_from_equal_level(level=level, current_price=current_price)

        if side is SignalSide.LONG:
            # Equal lows should be below or slightly above current price.
            return price <= current_price * self.equal_config.long_candidate_max_overshoot

        if side is SignalSide.SHORT:
            # Equal highs should be above or slightly below current price.
            return price >= current_price * self.equal_config.short_candidate_min_undershoot

        return False

    def _side_from_equal_level(
        self,
        *,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
    ) -> SignalSide:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._side_from_equal_level")
        level_type = getattr(level, "level_type", None)
        liquidity_side = getattr(level, "side", None)

        if level_type == LiquidityLevelType.EQUAL_LOWS:
            return SignalSide.LONG

        if level_type == LiquidityLevelType.EQUAL_HIGHS:
            return SignalSide.SHORT

        if liquidity_side == LiquiditySide.SELL_SIDE:
            return SignalSide.LONG

        if liquidity_side == LiquiditySide.BUY_SIDE:
            return SignalSide.SHORT

        price = reference_price(level)
        if price <= 0 or current_price <= 0:
            return SignalSide.UNKNOWN

        if price < current_price:
            return SignalSide.LONG

        if price > current_price:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _candidate_rank(
        self,
        candidate: EqualLevelCandidate,
    ) -> tuple[float, float, int, int, float]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._candidate_rank")
        level = candidate.level
        confidence, touches, reactions, compactness_rank = level_quality(level)
        return (
            candidate.edge,
            confidence,
            touches,
            reactions,
            compactness_rank,
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def _target_for_side(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> LiquidityLevel | StopCluster | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._target_for_side")
        if side is SignalSide.LONG:
            candidates = collect_targets_above(snapshot, current_price)
        elif side is SignalSide.SHORT:
            candidates = collect_targets_below(snapshot, current_price)
        else:
            return None

        valid = [
            item
            for item in candidates
            if self._is_valid_target(
                item=item,
                current_price=current_price,
                side=side,
            )
        ]

        if not valid:
            return None

        if side is SignalSide.LONG:
            return min(valid, key=reference_price)

        return max(valid, key=reference_price)

    def _is_valid_target(
        self,
        *,
        item: LiquidityLevel | StopCluster,
        current_price: float,
        side: SignalSide,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._is_valid_target")
        price = reference_price(item)
        if price <= 0 or current_price <= 0:
            return False

        if side is SignalSide.LONG and price <= current_price:
            return False

        if side is SignalSide.SHORT and price >= current_price:
            return False

        if distance_pct(price, current_price) > self.equal_config.max_target_distance_pct:
            return False

        if self.equal_config.reject_terminal_targets and is_terminal_item(item):
            return False

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        candidate: EqualLevelCandidate,
    ) -> ScoreBreakdown:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._build_score_breakdown")
        level = candidate.level
        side = candidate.side
        target = candidate.target

        edge = unit_score(candidate.edge)
        level_quality_score = self._level_quality_score(level)
        distance_component = self._level_distance_score(
            level=level,
            current_price=current_price,
        )
        target_score = self._target_quality_score(
            target=target,
            current_price=current_price,
        )
        zone_alignment = self._zone_alignment_score(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )
        context_score = self._context_score(
            snapshot=snapshot,
            side=side,
        )

        score = weighted_score(
            {
                "edge": edge,
                "level_quality": level_quality_score,
                "distance": distance_component,
                "target": target_score,
                "zone": zone_alignment,
                "context": context_score,
            },
            {
                "edge": self.equal_config.score_edge_weight,
                "level_quality": self.equal_config.score_level_quality_weight,
                "distance": self.equal_config.score_distance_weight,
                "target": self.equal_config.score_target_weight,
                "zone": self.equal_config.score_zone_weight,
                "context": self.equal_config.score_context_weight,
            },
            default=edge,
        )

        confidence_primary = weighted_score(
            {
                "edge": edge,
                "level_quality": level_quality_score,
                "distance": distance_component,
                "target": target_score,
                "context": context_score,
                "zone": zone_alignment,
            },
            {
                "edge": self.equal_config.confidence_edge_weight,
                "level_quality": self.equal_config.confidence_level_quality_weight,
                "distance": self.equal_config.confidence_distance_weight,
                "target": self.equal_config.confidence_target_weight,
                "context": self.equal_config.confidence_context_weight,
                "zone": self.equal_config.confidence_zone_weight,
            },
            default=edge,
        )

        confidence = confidence_from_components(
            primary=confidence_primary,
            context=context_score,
            confirmation=level_quality_score,
            freshness=1.0,
        )

        reasons = [
            f"edge:{edge:.3f}",
            f"level_quality:{level_quality_score:.3f}",
            f"distance_score:{distance_component:.3f}",
            f"target_score:{target_score:.3f}",
            f"context_score:{context_score:.3f}",
        ]

        confirmations: list[str] = []

        if int(getattr(level, "touches_count", 0) or 0) >= 3:
            confirmations.append("multiple_touches_confirm_equal_level_importance")

        if int(getattr(level, "reaction_count", 0) or 0) >= 2:
            confirmations.append("repeated_reactions_confirm_structure_validity")

        if compactness_score(level) >= 0.70:
            confirmations.append("compact_equal_level_structure_confirmed")

        if target is not None:
            confirmations.append("clear_liquidity_target_available")

        if zone_alignment > 0:
            confirmations.append("zone_alignment_confirmed")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "edge": edge,
                "level_quality": level_quality_score,
                "distance_score": distance_component,
                "target_score": target_score,
                "zone_alignment": zone_alignment,
                "context_score": context_score,
                "compactness_score": compactness_score(level),
            },
            weights={
                "score_edge_weight": self.equal_config.score_edge_weight,
                "score_level_quality_weight": self.equal_config.score_level_quality_weight,
                "score_distance_weight": self.equal_config.score_distance_weight,
                "score_target_weight": self.equal_config.score_target_weight,
                "score_zone_weight": self.equal_config.score_zone_weight,
                "score_context_weight": self.equal_config.score_context_weight,
            },
            reasons=reasons,
            confirmations=confirmations,
        ).normalize()

    def _edge_for_level(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        level: EqualLevel | LiquidityLevel,
        side: SignalSide,
        current_price: float,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._edge_for_level")
        level_quality_score = self._level_quality_score(level)
        distance_component = self._level_distance_score(
            level=level,
            current_price=current_price,
        )
        context = self._context_score(snapshot=snapshot, side=side)
        zone_alignment = self._zone_alignment_score(
            snapshot=snapshot,
            current_price=current_price,
            side=side,
        )

        return unit_score(
            0.42 * level_quality_score
            + 0.24 * distance_component
            + 0.20 * context
            + 0.14 * zone_alignment
        )

    @staticmethod
    def _level_quality_score(level: EqualLevel | LiquidityLevel) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".EqualHighLowStrategy._level_quality_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._level_quality_score")
        confidence = unit_score(getattr(level, "confidence", 0.0))
        touches = max(int(getattr(level, "touches_count", 0) or 0), 0)
        reactions = max(int(getattr(level, "reaction_count", 0) or 0), 0)

        return unit_score(
            0.42 * confidence
            + 0.20 * min(touches / 6.0, 1.0)
            + 0.18 * min(reactions / 4.0, 1.0)
            + 0.20 * compactness_score(level)
        )

    def _level_distance_score(
        self,
        *,
        level: EqualLevel | LiquidityLevel,
        current_price: float,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._level_distance_score")
        return distance_score(
            price=reference_price(level),
            current_price=current_price,
            max_distance_pct=self.equal_config.max_level_distance_pct,
        )

    def _target_quality_score(
        self,
        *,
        target: LiquidityLevel | StopCluster | None,
        current_price: float,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._target_quality_score")
        if target is None:
            return 0.0

        price = reference_price(target)
        if price <= 0 or current_price <= 0:
            return 0.0

        distance_component = distance_score(
            price=price,
            current_price=current_price,
            max_distance_pct=self.equal_config.max_target_distance_pct,
        )

        if isinstance(target, StopCluster):
            return unit_score(
                0.45 * unit_score(getattr(target, "confidence", 0.0))
                + 0.30 * unit_score(getattr(target, "estimated_stop_density", 0.0))
                + 0.25 * distance_component
            )

        if isinstance(target, LiquidityLevel):
            touches = max(int(getattr(target, "touches_count", 0) or 0), 0)
            reactions = max(int(getattr(target, "reaction_count", 0) or 0), 0)

            return unit_score(
                0.48 * unit_score(getattr(target, "confidence", 0.0))
                + 0.24 * distance_component
                + 0.14 * min(touches / 6.0, 1.0)
                + 0.14 * min(reactions / 4.0, 1.0)
            )

        return 0.0

    def _context_score(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._context_score")
        pressure = signed_score(getattr(snapshot, "liquidity_pressure_score", 0.0))

        if side is SignalSide.LONG:
            pressure_alignment = unit_score(max(pressure, 0.0))
            bias_bonus = 0.12 if getattr(snapshot, "bias", None) == LiquidityBias.DOWN else 0.0
            magnet = magnet_score_up(snapshot)
            sweep_risk = sweep_risk_down(snapshot)

        elif side is SignalSide.SHORT:
            pressure_alignment = unit_score(max(-pressure, 0.0))
            bias_bonus = 0.12 if getattr(snapshot, "bias", None) == LiquidityBias.UP else 0.0
            magnet = magnet_score_down(snapshot)
            sweep_risk = sweep_risk_up(snapshot)

        else:
            return 0.0

        return unit_score(
            0.34 * pressure_alignment
            + 0.24 * magnet
            + 0.20 * sweep_risk
            + bias_bonus
        )

    def _zone_alignment_score(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        current_price: float,
        side: SignalSide,
    ) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._zone_alignment_score")
        liquidity_side = (
            LiquiditySide.BUY_SIDE
            if side is SignalSide.LONG
            else LiquiditySide.SELL_SIDE
        )
        zone = best_zone_for_side(
            snapshot=snapshot,
            side=liquidity_side,
            current_price=current_price,
        )
        if zone is None:
            return 0.0

        return unit_score(getattr(zone, "score", 0.0))

    # ------------------------------------------------------------------
    # Trade levels
    # ------------------------------------------------------------------

    def _resolve_stop_price(
        self,
        *,
        side: SignalSide,
        current_price: float,
        level: EqualLevel | LiquidityLevel,
    ) -> float | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._resolve_stop_price")
        level_price = reference_price(level)
        if current_price <= 0 or level_price <= 0:
            return None

        if side is SignalSide.LONG:
            if level_price < current_price:
                return level_price * self.equal_config.long_stop_offset
            return current_price * (1.0 - self.equal_config.fallback_stop_pct)

        if side is SignalSide.SHORT:
            if level_price > current_price:
                return level_price * self.equal_config.short_stop_offset
            return current_price * (1.0 + self.equal_config.fallback_stop_pct)

        return None

    def _target_plans(
        self,
        *,
        current_price: float,
        side: SignalSide,
        target: LiquidityLevel | StopCluster | None,
        stop_loss: float | None,
    ) -> list[TargetPlan]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._target_plans")
        result: list[TargetPlan] = []

        target_price = reference_price(target) if target is not None else 0.0
        if target_price > 0:
            result.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=1.0,
                    rr=self._compute_rr(
                        current_price=current_price,
                        stop_price=stop_loss,
                        target_price=target_price,
                        side=side,
                    ),
                    label="equal_high_low_target",
                    metadata={
                        "source": "opposite_liquidity_target",
                        "target_type": self._target_type(target),
                    },
                )
            )

        for item in result:
            item.validate()

        return result

    @staticmethod
    def _compute_rr(
        *,
        current_price: float,
        stop_price: float | None,
        target_price: float | None,
        side: SignalSide,
    ) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".EqualHighLowStrategy._compute_rr")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._compute_rr")
        if current_price <= 0 or stop_price is None or target_price is None:
            return None

        risk = abs(current_price - stop_price)
        reward = (
            target_price - current_price
            if side is SignalSide.LONG
            else current_price - target_price
        )

        if risk <= 0 or reward <= 0:
            return None

        return reward / risk

    # ------------------------------------------------------------------
    # Reasons / confirmations / metadata
    # ------------------------------------------------------------------

    def _resolve_priority(
        self,
        *,
        score: float,
        confidence: float,
    ) -> SignalPriority:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._resolve_priority")
        combined = unit_score(0.55 * score + 0.45 * confidence)

        if combined >= self.equal_config.critical_priority_score:
            return SignalPriority.URGENT

        if combined >= self.equal_config.high_priority_score:
            return SignalPriority.HIGH

        return self.equal_config.default_priority

    def _source_features(self, candidate: EqualLevelCandidate) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._source_features")
        features = [
            LIQUIDITY_FEATURES.SNAPSHOT,
            LIQUIDITY_FEATURES.MAP_SNAPSHOT,
            LIQUIDITY_FEATURES.PRESSURE_SCORE,
            LIQUIDITY_FEATURES.BIAS,
            LIQUIDITY_FEATURES.EQUAL_LEVELS,
        ]

        if candidate.side is SignalSide.LONG:
            features.extend(
                [
                    LIQUIDITY_FEATURES.BELOW_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.MAGNET_UP,
                    LIQUIDITY_FEATURES.SWEEP_RISK_DOWN,
                    LIQUIDITY_FEATURES.NEAREST_ABOVE_LEVEL,
                ]
            )

        if candidate.side is SignalSide.SHORT:
            features.extend(
                [
                    LIQUIDITY_FEATURES.ABOVE_LIQUIDITY_SCORE,
                    LIQUIDITY_FEATURES.MAGNET_DOWN,
                    LIQUIDITY_FEATURES.SWEEP_RISK_UP,
                    LIQUIDITY_FEATURES.NEAREST_BELOW_LEVEL,
                ]
            )

        if candidate.target is not None:
            if isinstance(candidate.target, StopCluster):
                features.append(LIQUIDITY_FEATURES.STOP_CLUSTERS)
            elif isinstance(candidate.target, LiquidityLevel):
                features.append(LIQUIDITY_FEATURES.ACTIVE_LEVELS)

        return list(dict.fromkeys(features))

    def _tags(
        self,
        *,
        candidate: EqualLevelCandidate,
        snapshot: LiquidityMapSnapshot,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._tags")
        tags = [
            self.equal_config.tag_liquidity,
            self.equal_config.tag_equal_levels,
            self.equal_config.tag_equal_high_low,
            self.equal_config.tag_reaction,
            self.equal_config.tag_reversal,
            self.equal_config.tag_structure_quality,
        ]

        level_type = getattr(candidate.level, "level_type", None)
        if level_type == LiquidityLevelType.EQUAL_LOWS:
            tags.append(self.equal_config.tag_equal_lows)

        if level_type == LiquidityLevelType.EQUAL_HIGHS:
            tags.append(self.equal_config.tag_equal_highs)

        if candidate.target is not None:
            tags.append(self.equal_config.tag_target_available)
            tags.append(f"target_type:{self._target_type(candidate.target)}")

        bias = getattr(snapshot, "bias", None)
        if bias is not None:
            tags.append(f"bias:{serialize_for_metadata(bias)}")

        return list(dict.fromkeys(tags))

    def _build_primary_reason(
        self,
        *,
        level: EqualLevel | LiquidityLevel,
        side: SignalSide,
        current_price: float,
    ) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._build_primary_reason")
        prefix = (
            "Equal lows reaction -> long setup"
            if side is SignalSide.LONG
            else "Equal highs reaction -> short setup"
        )

        return (
            f"{prefix}: level={reference_price(level):.6f}, "
            f"current_price={current_price:.6f}, "
            f"confidence={unit_score(getattr(level, 'confidence', 0.0)):.3f}, "
            f"touches={int(getattr(level, 'touches_count', 0) or 0)}, "
            f"reactions={int(getattr(level, 'reaction_count', 0) or 0)}, "
            f"compactness_width_pct={compactness_width_pct(level):.6f}, "
            f"sweep_status={serialize_for_metadata(getattr(level, 'sweep_status', None))}"
        )

    def _build_target_reason(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._build_target_reason")
        if target is None:
            return "No explicit target found; signal based on equal highs/lows structure quality"

        if isinstance(target, StopCluster):
            return (
                f"Nearest target is stop cluster at {reference_price(target):.6f} "
                f"(confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f})"
            )

        return (
            f"Nearest target is liquidity level at {reference_price(target):.6f} "
            f"(type={self._target_type(target)}, "
            f"confidence={unit_score(getattr(target, 'confidence', 0.0)):.3f}, "
            f"sweep_status={serialize_for_metadata(getattr(target, 'sweep_status', None))})"
        )

    def _build_confirmations(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        level: EqualLevel | LiquidityLevel,
        side: SignalSide,
        current_price: float,
        target: LiquidityLevel | StopCluster | None,
    ) -> list[str]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._build_confirmations")
        confirmations: list[str] = []

        if int(getattr(level, "touches_count", 0) or 0) >= 3:
            confirmations.append("Multiple touches confirm equal level importance")

        if int(getattr(level, "reaction_count", 0) or 0) >= 2:
            confirmations.append("Repeated reactions confirm structure validity")

        if compactness_score(level) >= 0.70:
            confirmations.append("Compact equal highs/lows structure")

        if side is SignalSide.LONG:
            if getattr(snapshot, "bias", None) == LiquidityBias.DOWN:
                confirmations.append("Counter-bias long setup from equal lows")
            if self._has_high_quality_zone(
                snapshot=snapshot,
                side=LiquiditySide.BUY_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality buy-side zone ahead")

        if side is SignalSide.SHORT:
            if getattr(snapshot, "bias", None) == LiquidityBias.UP:
                confirmations.append("Counter-bias short setup from equal highs")
            if self._has_high_quality_zone(
                snapshot=snapshot,
                side=LiquiditySide.SELL_SIDE,
                current_price=current_price,
            ):
                confirmations.append("High-quality sell-side zone ahead")

        if target is not None:
            confirmations.append("Clear liquidity target available")

        return confirmations

    def _has_high_quality_zone(
        self,
        *,
        snapshot: LiquidityMapSnapshot,
        side: LiquiditySide,
        current_price: float,
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._has_high_quality_zone")
        zone = best_zone_for_side(
            snapshot=snapshot,
            side=side,
            current_price=current_price,
        )
        if zone is None:
            return False

        return unit_score(getattr(zone, "score", 0.0)) >= 0.60

    @staticmethod
    def _target_type(target: LiquidityLevel | StopCluster | None) -> str | None:
        _strategy_logger = logging.getLogger(__name__ + ".EqualHighLowStrategy._target_type")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._target_type")
        if target is None:
            return None

        if isinstance(target, StopCluster):
            return "stop_cluster"

        if isinstance(target, LiquidityLevel):
            level_type = getattr(target, "level_type", None)
            if hasattr(level_type, "value"):
                return str(level_type.value)
            return str(level_type)

        return target.__class__.__name__

    def _level_metadata(
        self,
        level: EqualLevel | LiquidityLevel,
    ) -> dict[str, Any]:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._level_metadata")
        return {
            "type": self._target_type(level),
            "price": reference_price(level),
            "confidence": unit_score(getattr(level, "confidence", 0.0)),
            "touches_count": int(getattr(level, "touches_count", 0) or 0),
            "reaction_count": int(getattr(level, "reaction_count", 0) or 0),
            "compactness_width_pct": compactness_width_pct(level),
            "compactness_score": compactness_score(level),
            "side": serialize_for_metadata(getattr(level, "side", None)),
            "sweep_status": serialize_for_metadata(getattr(level, "sweep_status", None)),
            "raw": serialize_for_metadata(level),
        }

    def _target_metadata(
        self,
        target: LiquidityLevel | StopCluster | None,
    ) -> dict[str, Any] | None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering EqualHighLowStrategy._target_metadata")
        if target is None:
            return None

        return {
            "type": self._target_type(target),
            "price": reference_price(target),
            "confidence": unit_score(getattr(target, "confidence", 0.0)),
            "strength": serialize_for_metadata(getattr(target, "strength", None)),
            "side": serialize_for_metadata(getattr(target, "side", None)),
            "sweep_status": serialize_for_metadata(getattr(target, "sweep_status", None)),
            "raw": serialize_for_metadata(target),
        }


__all__ = [
    "EqualHighLowStrategy",
    "EqualHighLowStrategyConfig",
    "EqualLevelCandidate",
]