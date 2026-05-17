from __future__ import annotations

from core.logger import get_logger
from risk.config import (
    ExecutionCostConfig,
    LeveragePolicyConfig,
    TierModelConfig,
    TierRiskConfig,
)
from risk.enums import (
    ExecutionQuality,
    LiquidityClass,
    RiskDecisionType,
    RiskLevel,
    RiskMode,
    RiskViolationType,
    TradeTier,
)
from risk.models import (
    ExecutionCostEstimate,
    ExpectedValueSnapshot,
    RiskCheckResult,
    RiskEvaluationRequest,
    RiskViolation,
    TierRiskProfile,
)
from risk.state import RiskState
from risk.utils import (
    is_finite_number,
    calculate_cost_to_reward_ratio,
    calculate_expected_value,
    calculate_reward_distance,
    calculate_risk_reward_ratio,
    calculate_side_aware_stop_distance,
)


class TierRiskGuard:
    """
    Validates and resolves requested trade tier.

    Responsibilities:
    - choose default tier if request.tier is missing;
    - block tier if current RiskMode does not allow new positions;
    - downgrade tier according to RiskMode;
    - return TierRiskProfile for downstream sizing and guards.

    Does not publish events and does not mutate RiskState.
    """

    def __init__(
        self,
        config: TierModelConfig,
        *,
        service_name: str = "risk.tier_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="tier_guard",
        )

    def resolve_profile(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        mode: RiskMode | None = None,
    ) -> TierRiskProfile:
        risk_mode = mode or state.risk_mode
        requested_tier = request.tier or self._config.default_tier

        if requested_tier not in self._config.tiers:
            raise ValueError(f"Unsupported trade tier: {requested_tier!r}")

        if risk_mode.is_terminal:
            raise ValueError(f"Risk mode {risk_mode.value} does not allow new positions")

        if risk_mode is RiskMode.REDUCE_ONLY and request.order_intent.increases_risk:
            raise ValueError("REDUCE_ONLY mode does not allow risk-increasing orders")

        max_tier = self._config.max_tier_by_mode.get(risk_mode, TradeTier.T1)
        final_tier = requested_tier
        downgraded = False
        reason: str | None = None

        if requested_tier.rank > max_tier.rank:
            final_tier = max_tier
            downgraded = True
            reason = f"tier_downgraded_by_risk_mode:{risk_mode.value}"

        tier_config = self._config.tiers[final_tier]

        self._validate_tier_allowed_by_mode(
            tier=final_tier,
            tier_config=tier_config,
            mode=risk_mode,
        )

        return TierRiskProfile(
            requested_tier=requested_tier,
            final_tier=final_tier,
            risk_units=tier_config.risk_units,
            min_rr=tier_config.min_rr,
            min_expected_value=tier_config.min_expected_value,
            max_cost_to_reward_pct=tier_config.max_cost_to_reward_pct,
            default_leverage=tier_config.default_leverage,
            max_leverage=tier_config.max_leverage,
            downgraded=downgraded,
            reason=reason,
            metadata={
                "risk_mode": risk_mode.value,
                "max_tier_for_mode": max_tier.value,
            },
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        state: RiskState,
        *,
        mode: RiskMode | None = None,
    ) -> RiskCheckResult:
        try:
            profile = self.resolve_profile(request, state, mode=mode)
            decision = (
                RiskDecisionType.DOWNGRADE_TIER
                if profile.downgraded
                else RiskDecisionType.ALLOW
            )

            violations: list[RiskViolation] = []
            if profile.downgraded:
                violations.append(
                    RiskViolation(
                        violation_type=RiskViolationType.TIER_DOWNGRADED,
                        level=RiskLevel.WARNING,
                        message=profile.reason or "Tier downgraded by risk policy",
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=profile.requested_tier,
                        metadata={
                            "requested_tier": profile.requested_tier.value,
                            "final_tier": profile.final_tier.value,
                        },
                    )
                )

            return RiskCheckResult(
                passed=True,
                decision=decision,
                violations=violations,
                adjusted_tier=profile.final_tier,
                risk_mode=mode or state.risk_mode,
                reason=profile.reason,
                metadata={"tier_profile": profile},
            )

        except ValueError as exc:
            self._logger.warning(
                "Tier check failed | symbol=%s reason=%s",
                request.symbol,
                str(exc),
                extra={"symbol": request.symbol},
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.TIER_NOT_ALLOWED,
                        level=RiskLevel.CRITICAL,
                        message=str(exc),
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=request.tier,
                    )
                ],
                risk_mode=mode or state.risk_mode,
                reason=str(exc),
            )

    @staticmethod
    def _validate_tier_allowed_by_mode(
        *,
        tier: TradeTier,
        tier_config: TierRiskConfig,
        mode: RiskMode,
    ) -> None:
        if mode is RiskMode.CAUTION and not tier_config.allow_in_caution:
            raise ValueError(f"Tier {tier.value} is not allowed in CAUTION mode")

        if mode is RiskMode.SAFE_MODE and not tier_config.allow_in_safe_mode:
            raise ValueError(f"Tier {tier.value} is not allowed in SAFE_MODE")

        if mode in {RiskMode.NORMAL, RiskMode.CAUTION, RiskMode.SAFE_MODE}:
            return

        if mode is RiskMode.REDUCE_ONLY:
            raise ValueError("Risk-increasing tiers are not allowed in REDUCE_ONLY mode")

        if mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}:
            raise ValueError(f"Tier {tier.value} is not allowed in {mode.value}")


class RiskRewardGuard:
    """
    Validates stop-loss, take-profit, RR and expected value.

    Responsibilities:
    - side-aware stop-loss validation;
    - side-aware take-profit validation;
    - RR check;
    - optional expected value check.

    Does not calculate position size.
    """

    def __init__(
        self,
        *,
        service_name: str = "risk.risk_reward_guard",
    ) -> None:
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="risk_reward_guard",
        )

    def evaluate(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
    ) -> ExpectedValueSnapshot:
        stop_distance = calculate_side_aware_stop_distance(
            side=request.side,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
        )

        if stop_distance is None:
            raise ValueError("Stop loss is required for risk/reward validation")

        reward_distance = calculate_reward_distance(
            side=request.side,
            entry_price=request.entry_price,
            take_profit=request.take_profit,
        )

        if reward_distance is None:
            raise ValueError("Take profit is required for risk/reward validation")

        rr = calculate_risk_reward_ratio(
            side=request.side,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
        )

        if rr is None:
            raise ValueError("Risk/reward ratio cannot be calculated")

        expected_loss = (
            request.expected_loss
            if request.expected_loss is not None
            else stop_distance
        )
        expected_reward = (
            request.expected_reward
            if request.expected_reward is not None
            else reward_distance
        )

        self._validate_non_negative_finite(
            expected_loss,
            field_name="expected_loss",
            allow_zero=False,
        )
        self._validate_non_negative_finite(
            expected_reward,
            field_name="expected_reward",
            allow_zero=False,
        )

        expected_cost = self._resolve_expected_cost(request)
        win_probability = request.expected_win_probability

        if win_probability is not None:
            self._validate_probability(win_probability)

        expected_value: float | None = None
        expected_value_after_cost: float | None = None

        if win_probability is not None:
            expected_value_after_cost = calculate_expected_value(
                expected_reward=expected_reward,
                expected_loss=expected_loss,
                win_probability=win_probability,
                expected_cost=expected_cost,
            )
            expected_value = calculate_expected_value(
                expected_reward=expected_reward,
                expected_loss=expected_loss,
                win_probability=win_probability,
                expected_cost=0.0,
            )

            self._validate_optional_finite(
                expected_value,
                field_name="expected_value",
            )
            self._validate_optional_finite(
                expected_value_after_cost,
                field_name="expected_value_after_cost",
            )

        cost_to_reward = calculate_cost_to_reward_ratio(
            expected_cost=expected_cost,
            expected_reward=expected_reward,
        )
        self._validate_optional_finite(
            cost_to_reward,
            field_name="cost_to_reward_ratio",
        )

        return ExpectedValueSnapshot(
            expected_reward=expected_reward,
            expected_loss=expected_loss,
            expected_cost=expected_cost,
            win_probability=win_probability,
            risk_reward_ratio=rr,
            cost_to_reward_ratio=cost_to_reward,
            expected_value=expected_value,
            expected_value_after_cost=expected_value_after_cost,
            metadata={
                "stop_distance": stop_distance,
                "reward_distance": reward_distance,
                "tier": tier_profile.final_tier.value,
            },
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
    ) -> RiskCheckResult:
        try:
            snapshot = self.evaluate(request, tier_profile)
        except ValueError as exc:
            violation_type = self._classify_validation_error(str(exc))

            self._logger.warning(
                "Risk/reward validation failed | symbol=%s tier=%s reason=%s",
                request.symbol,
                tier_profile.final_tier.value,
                str(exc),
                extra={
                    "symbol": request.symbol,
                    "tier": tier_profile.final_tier.value,
                },
            )

            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=violation_type,
                        level=RiskLevel.CRITICAL,
                        message=str(exc),
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=tier_profile.final_tier,
                    )
                ],
                reason=str(exc),
            )

        violations: list[RiskViolation] = []

        if snapshot.risk_reward_ratio is None or snapshot.risk_reward_ratio < tier_profile.min_rr:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.RISK_REWARD_TOO_LOW,
                    level=RiskLevel.CRITICAL,
                    message="Risk/reward ratio is below tier minimum",
                    current_value=snapshot.risk_reward_ratio,
                    limit_value=tier_profile.min_rr,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                    metadata={
                        "expected_reward": snapshot.expected_reward,
                        "expected_loss": snapshot.expected_loss,
                    },
                )
            )

        if (
            snapshot.expected_value_after_cost is not None
            and snapshot.expected_value_after_cost < tier_profile.min_expected_value
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXPECTED_VALUE_NEGATIVE,
                    level=RiskLevel.CRITICAL,
                    message="Expected value after cost is below tier minimum",
                    current_value=snapshot.expected_value_after_cost,
                    limit_value=tier_profile.min_expected_value,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                )
            )

        if violations:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=violations,
                reason="Risk/reward validation failed",
                metadata={"expected_value_snapshot": snapshot},
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={"expected_value_snapshot": snapshot},
        )

    @staticmethod
    def _resolve_expected_cost(request: RiskEvaluationRequest) -> float:
        if request.execution_cost is not None:
            cost = request.execution_cost.total_cost
        elif request.expected_cost is not None:
            cost = request.expected_cost
        else:
            return 0.0

        RiskRewardGuard._validate_non_negative_finite(
            cost,
            field_name="expected_cost",
            allow_zero=True,
        )
        return max(0.0, cost)

    @staticmethod
    def _validate_probability(value: float) -> None:
        if not is_finite_number(value) or value < 0.0 or value > 1.0:
            raise ValueError("expected_win_probability must be a finite number in [0, 1]")

    @staticmethod
    def _validate_non_negative_finite(
        value: float,
        *,
        field_name: str,
        allow_zero: bool,
    ) -> None:
        if not is_finite_number(value):
            raise ValueError(f"{field_name} must be finite")

        if allow_zero:
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0")
            return

        if value <= 0.0:
            raise ValueError(f"{field_name} must be > 0")

    @staticmethod
    def _validate_optional_finite(value: float | None, *, field_name: str) -> None:
        if value is not None and not is_finite_number(value):
            raise ValueError(f"{field_name} must be finite")

    @staticmethod
    def _classify_validation_error(message: str) -> RiskViolationType:
        normalized = message.lower()

        if "finite" in normalized or "probability" in normalized or "expected_" in normalized:
            return RiskViolationType.INVALID_REQUEST

        if "stop loss is required" in normalized or "stop_loss is required" in normalized:
            return RiskViolationType.STOP_LOSS_MISSING

        if "take profit is required" in normalized or "take_profit is required" in normalized:
            return RiskViolationType.TAKE_PROFIT_MISSING

        if "stop_loss" in normalized and "long" in normalized:
            return RiskViolationType.STOP_LOSS_SIDE_INVALID

        if "stop_loss" in normalized and "short" in normalized:
            return RiskViolationType.STOP_LOSS_SIDE_INVALID

        if "take_profit" in normalized and "long" in normalized:
            return RiskViolationType.TAKE_PROFIT_SIDE_INVALID

        if "take_profit" in normalized and "short" in normalized:
            return RiskViolationType.TAKE_PROFIT_SIDE_INVALID

        return RiskViolationType.STOP_DISTANCE_INVALID


class ExecutionCostGuard:
    """
    Validates execution cost, spread/slippage and execution quality.

    Especially important for micro-scalping where costs can destroy edge.
    """

    def __init__(
        self,
        config: ExecutionCostConfig,
        *,
        service_name: str = "risk.execution_cost_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="execution_cost_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
        ev_snapshot: ExpectedValueSnapshot,
        *,
        mode: RiskMode = RiskMode.NORMAL,
    ) -> RiskCheckResult:
        if not self._config.enabled:
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.ALLOW,
                metadata={"enabled": False},
            )

        violations: list[RiskViolation] = []

        execution_cost = request.execution_cost or ExecutionCostEstimate(
            other_cost=max(0.0, request.expected_cost or 0.0),
            quality=request.execution_quality,
        )

        if not self._is_execution_quality_allowed(execution_cost.quality):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXECUTION_QUALITY_TOO_LOW,
                    level=RiskLevel.CRITICAL,
                    message="Execution quality is below configured minimum",
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                    metadata={
                        "quality": execution_cost.quality.value,
                        "min_quality": self._config.min_execution_quality.value,
                    },
                )
            )

        if (
            self._config.require_spread_guard
            and self._config.max_spread_pct is not None
            and execution_cost.spread_pct is not None
            and execution_cost.spread_pct > self._config.max_spread_pct
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SPREAD_TOO_WIDE,
                    level=RiskLevel.CRITICAL,
                    message="Spread exceeds configured maximum",
                    current_value=execution_cost.spread_pct,
                    limit_value=self._config.max_spread_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                )
            )

        if (
            self._config.require_slippage_guard
            and self._config.max_slippage_pct is not None
            and execution_cost.slippage_pct is not None
            and execution_cost.slippage_pct > self._config.max_slippage_pct
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.SLIPPAGE_TOO_HIGH,
                    level=RiskLevel.CRITICAL,
                    message="Slippage exceeds configured maximum",
                    current_value=execution_cost.slippage_pct,
                    limit_value=self._config.max_slippage_pct,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                )
            )

        max_cost_to_reward = self._resolve_max_cost_to_reward(
            tier=tier_profile.final_tier,
            mode=mode,
            tier_profile=tier_profile,
        )

        cost_ratio = ev_snapshot.cost_to_reward_ratio
        if cost_ratio is None:
            cost_ratio = calculate_cost_to_reward_ratio(
                expected_cost=execution_cost.total_cost,
                expected_reward=ev_snapshot.expected_reward,
            )

        if not is_finite_number(cost_ratio):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXECUTION_COST_TOO_HIGH,
                    level=RiskLevel.CRITICAL,
                    message="Execution cost-to-reward ratio must be finite",
                    current_value=cost_ratio,
                    limit_value=max_cost_to_reward,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                    metadata={
                        "expected_cost": execution_cost.total_cost,
                        "expected_reward": ev_snapshot.expected_reward,
                    },
                )
            )
        elif cost_ratio > max_cost_to_reward:
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXECUTION_COST_TOO_HIGH,
                    level=RiskLevel.CRITICAL,
                    message="Execution cost is too high relative to expected reward",
                    current_value=cost_ratio,
                    limit_value=max_cost_to_reward,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                    metadata={
                        "expected_cost": execution_cost.total_cost,
                        "expected_reward": ev_snapshot.expected_reward,
                    },
                )
            )

        if (
            ev_snapshot.expected_value_after_cost is not None
            and not is_finite_number(ev_snapshot.expected_value_after_cost)
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXPECTED_VALUE_NEGATIVE,
                    level=RiskLevel.CRITICAL,
                    message="Expected value after cost must be finite",
                    current_value=ev_snapshot.expected_value_after_cost,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                )
            )

        if (
            self._config.require_positive_ev_after_cost
            and ev_snapshot.expected_value_after_cost is not None
            and is_finite_number(ev_snapshot.expected_value_after_cost)
            and ev_snapshot.expected_value_after_cost <= 0
        ):
            violations.append(
                RiskViolation(
                    violation_type=RiskViolationType.EXPECTED_VALUE_NEGATIVE,
                    level=RiskLevel.CRITICAL,
                    message="Expected value after cost must be positive",
                    current_value=ev_snapshot.expected_value_after_cost,
                    limit_value=0.0,
                    symbol=request.symbol,
                    strategy_name=request.strategy_name,
                    tier=tier_profile.final_tier,
                )
            )

        if violations:
            self._logger.warning(
                "Execution cost check failed | symbol=%s tier=%s violations=%s",
                request.symbol,
                tier_profile.final_tier.value,
                len(violations),
                extra={
                    "symbol": request.symbol,
                    "tier": tier_profile.final_tier.value,
                },
            )
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=violations,
                reason="Execution cost validation failed",
                metadata={
                    "execution_cost": execution_cost,
                    "cost_to_reward_ratio": cost_ratio,
                    "max_cost_to_reward": max_cost_to_reward,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            metadata={
                "execution_cost": execution_cost,
                "cost_to_reward_ratio": cost_ratio,
                "max_cost_to_reward": max_cost_to_reward,
            },
        )

    def _resolve_max_cost_to_reward(
        self,
        *,
        tier: TradeTier,
        mode: RiskMode,
        tier_profile: TierRiskProfile,
    ) -> float:
        tier_limit = self._config.max_cost_to_reward_by_tier.get(
            tier,
            tier_profile.max_cost_to_reward_pct,
        )

        if mode is RiskMode.SAFE_MODE:
            return min(tier_limit, self._config.safe_mode_max_cost_to_reward_pct)

        if mode is RiskMode.NORMAL:
            return min(tier_limit, self._config.default_max_cost_to_reward_pct)

        if mode is RiskMode.CAUTION:
            return min(tier_limit, self._config.default_max_cost_to_reward_pct)

        return min(tier_limit, self._config.default_max_cost_to_reward_pct)

    def _is_execution_quality_allowed(self, quality: ExecutionQuality) -> bool:
        if quality is ExecutionQuality.BLOCKED:
            return False

        quality_rank = {
            ExecutionQuality.BLOCKED: 0,
            ExecutionQuality.POOR: 1,
            ExecutionQuality.ACCEPTABLE: 2,
            ExecutionQuality.GOOD: 3,
            ExecutionQuality.EXCELLENT: 4,
        }

        return quality_rank[quality] >= quality_rank[self._config.min_execution_quality]


class LeverageGuard:
    """
    Adaptive leverage guard.

    Leverage depends on:
    - requested leverage;
    - tier cap;
    - liquidity class cap;
    - risk mode cap;
    - symbol cap;
    - strategy cap;
    - execution quality.
    """

    def __init__(
        self,
        config: LeveragePolicyConfig,
        *,
        service_name: str = "risk.leverage_guard",
    ) -> None:
        self._config = config
        self._logger = get_logger(
            __name__,
            service=service_name,
            event_type="leverage_guard",
        )

    def check(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
        *,
        mode: RiskMode = RiskMode.NORMAL,
    ) -> RiskCheckResult:
        requested = self._resolve_requested_leverage(request, tier_profile)

        if not is_finite_number(requested) or requested <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.LEVERAGE_NOT_ALLOWED,
                        level=RiskLevel.CRITICAL,
                        message="Requested leverage must be > 0",
                        current_value=requested,
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=tier_profile.final_tier,
                    )
                ],
                reason="Invalid leverage",
            )

        max_allowed = self.resolve_max_leverage(
            request,
            tier_profile,
            mode=mode,
        )

        if max_allowed <= 0:
            return RiskCheckResult(
                passed=False,
                decision=RiskDecisionType.DENY,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.LEVERAGE_NOT_ALLOWED,
                        level=RiskLevel.CRITICAL,
                        message="Leverage is not allowed in current risk context",
                        current_value=max_allowed,
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=tier_profile.final_tier,
                    )
                ],
                reason="Leverage not allowed",
            )

        if requested > max_allowed:
            self._logger.warning(
                "Leverage capped | symbol=%s tier=%s requested=%s allowed=%s",
                request.symbol,
                tier_profile.final_tier.value,
                requested,
                max_allowed,
                extra={
                    "symbol": request.symbol,
                    "tier": tier_profile.final_tier.value,
                },
            )
            return RiskCheckResult(
                passed=True,
                decision=RiskDecisionType.REDUCE_SIZE,
                adjusted_leverage=max_allowed,
                violations=[
                    RiskViolation(
                        violation_type=RiskViolationType.MAX_LEVERAGE_EXCEEDED,
                        level=RiskLevel.WARNING,
                        message="Requested leverage exceeds allowed limit and was capped",
                        current_value=requested,
                        limit_value=max_allowed,
                        symbol=request.symbol,
                        strategy_name=request.strategy_name,
                        tier=tier_profile.final_tier,
                    )
                ],
                metadata={
                    "requested_leverage": requested,
                    "allowed_leverage": max_allowed,
                },
            )

        return RiskCheckResult(
            passed=True,
            decision=RiskDecisionType.ALLOW,
            adjusted_leverage=requested,
            metadata={
                "requested_leverage": requested,
                "allowed_leverage": max_allowed,
            },
        )

    def _resolve_requested_leverage(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
    ) -> float:
        # None means no explicit user/strategy request, so defaults may be used.
        # Numeric zero/NaN/inf are explicit invalid inputs and must fail closed.
        if request.requested_leverage is not None:
            return request.requested_leverage

        if tier_profile.default_leverage is not None:
            return tier_profile.default_leverage

        return self._config.default_leverage

    def resolve_max_leverage(
        self,
        request: RiskEvaluationRequest,
        tier_profile: TierRiskProfile,
        *,
        mode: RiskMode,
    ) -> float:
        caps: list[float] = [
            self._config.default_leverage,
            tier_profile.max_leverage,
            self._config.max_leverage_by_tier.get(
                tier_profile.final_tier,
                tier_profile.max_leverage,
            ),
            self._config.max_leverage_by_liquidity.get(
                request.liquidity_class,
                self._config.default_leverage,
            ),
        ]

        symbol_cap = self._config.per_symbol_max_leverage.get(request.symbol)
        if symbol_cap is not None:
            caps.append(symbol_cap)

        if request.strategy_name:
            strategy_cap = self._config.per_strategy_max_leverage.get(request.strategy_name)
            if strategy_cap is not None:
                caps.append(strategy_cap)

        if mode is RiskMode.SAFE_MODE:
            caps.append(self._config.safe_mode_max_leverage)
        elif mode is RiskMode.CAUTION:
            caps.append(self._config.caution_max_leverage)
        elif mode is RiskMode.REDUCE_ONLY:
            caps.append(self._config.reduce_only_max_leverage)
        elif mode in {RiskMode.HALTED, RiskMode.EMERGENCY_STOP}:
            return 0.0

        if request.liquidity_class in {
            LiquidityClass.LOW,
            LiquidityClass.ILLIQUID,
            LiquidityClass.SHITCOIN,
        }:
            caps.append(self._config.low_liquidity_max_leverage)

        if request.liquidity_class is LiquidityClass.SHITCOIN:
            caps.append(self._config.shitcoin_max_leverage)

        if request.execution_quality in {
            ExecutionQuality.POOR,
            ExecutionQuality.BLOCKED,
        }:
            caps.append(min(caps) if caps else 1.0)
            caps.append(1.0)

        finite_caps = [cap for cap in caps if is_finite_number(cap) and cap >= 0.0]
        if not finite_caps:
            return 0.0

        return max(0.0, min(finite_caps))


__all__ = [
    "ExecutionCostGuard",
    "LeverageGuard",
    "RiskRewardGuard",
    "TierRiskGuard",
]