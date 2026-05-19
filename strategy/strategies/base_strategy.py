# trading_system/strategy/strategies/base_strategy.py

from __future__ import annotations

from abc import ABC
from typing import Any

from ..base import BaseStrategy
from ..enums import (
    EntryType,
    ExitType,
    FeatureSource,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    StrategyExecutionQuality,
    StrategyLiquidityClass,
    StrategyMarginMode,
    StrategyMarketType,
    StrategyOrderIntent,
    StrategyTradeTier,
    Timeframe,
    TriggerType,
)
from ..exceptions import StrategyEvaluationError
from ..models import (
    EntryPlan,
    ExecutionCostPayload,
    ExecutionPlanDraft,
    ExitPlan,
    InvalidationPlan,
    StrategyContext,
    StrategySignal,
    TargetPlan,
    clamp,
    confidence_to_grade,
    confidence_to_strength,
    utcnow,
)


class StrategyMixinSupport:
    """
    Small helper base for mixins.

    Mixins are used together with BaseStrategy, but static analyzers do not know
    that self has strategy_name / validate_context / validate_context_requirements.
    These wrappers avoid unresolved attribute warnings without weakening runtime
    behavior.
    """

    @property
    def _strategy_name_for_errors(self) -> str:
        return str(getattr(self, "strategy_name", self.__class__.__name__))

    def _validate_context_for_mixin(self, context: StrategyContext) -> None:
        validator = getattr(self, "validate_context", None)
        if not callable(validator):
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: validate_context() is not available"
            )
        validator(context)

    def _validate_context_requirements_for_mixin(self, context: StrategyContext) -> None:
        validator = getattr(self, "validate_context_requirements", None)
        if not callable(validator):
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: validate_context_requirements() is not available"
            )
        validator(context)


class StrategySignalMixin(StrategyMixinSupport):
    """
    Helper methods for concrete strategies.

    Concrete strategies use these helpers to build internal StrategySignal
    objects. They must not emit signal.generated directly. SignalProcessor /
    SignalRouter converts StrategySignal into RiskReadySignalPayload for
    RiskManager.
    """

    def build_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trigger_type: TriggerType = TriggerType.PRIMARY,
        origin: SignalOrigin = SignalOrigin.SINGLE_STRATEGY,
        priority: SignalPriority = SignalPriority.MEDIUM,
        status: SignalStatus = SignalStatus.NEW,
    ) -> StrategySignal:
        """
        Build a validated StrategySignal from StrategyContext.

        This is only an internal strategy-layer signal. It may still need
        SignalBuilder to attach entry/exit/execution plans before routing to risk.
        """
        self._validate_context_requirements_for_mixin(context)

        confidence_f = self._normalize_unit_interval(confidence, "confidence")
        score_f = self._normalize_non_negative(score, "score")

        strategy_name = self._strategy_name_for_errors
        category = getattr(self, "category", StrategyCategory.HYBRID)
        default_setup_type = getattr(self, "default_setup_type", SetupType.UNKNOWN)

        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=strategy_name,
            category=category,
            timeframe=context.timeframe,
            setup_type=setup_type or default_setup_type,
            timestamp=context.timestamp or utcnow(),
            confidence=confidence_f,
            score=score_f,
            confidence_grade=confidence_to_grade(confidence_f),
            strength=confidence_to_strength(confidence_f),
            status=status,
            trigger_type=trigger_type,
            origin=origin,
            priority=priority,
            reasons=list(reasons or []),
            confirmations=list(confirmations or []),
            source_features=list(source_features or []),
            regime=self._context_regime_for_mixin(context),
            metadata=dict(metadata or {}),
        )

        signal.metadata.setdefault("strategy_name", strategy_name)
        signal.metadata.setdefault("category", category.value)
        signal.metadata.setdefault("timeframe", context.timeframe.value)
        signal.metadata.setdefault("setup_type", signal.setup_type.value)
        signal.metadata.setdefault("signal_id", signal.signal_id)

        signal.validate()
        return signal

    def build_directional_signal(
        self,
        *,
        context: StrategyContext,
        side: SignalSide,
        confidence: float,
        score: float,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        setup_type: SetupType | None = None,
        reasons: list[str] | None = None,
        confirmations: list[str] | None = None,
        source_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        priority: SignalPriority = SignalPriority.MEDIUM,
        tier: StrategyTradeTier | None = None,
        order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN,
        requested_leverage: float | None = None,
        margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED,
        liquidity_class: StrategyLiquidityClass | None = None,
        execution_quality: StrategyExecutionQuality | None = None,
        market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES,
    ) -> StrategySignal:
        """
        Build a directional signal and optionally attach trade plan metadata.

        Final validation that entry/stop/take-profit are risk-ready belongs to
        SignalBuilder / SignalProcessor before signal.generated is emitted.
        """
        if side not in {SignalSide.LONG, SignalSide.SHORT}:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: directional signal side must be LONG or SHORT"
            )

        signal_metadata = dict(metadata or {})
        signal_metadata.setdefault("order_intent", order_intent.value)
        signal_metadata.setdefault("margin_mode", margin_mode.value)
        signal_metadata.setdefault("market_type", market_type.value)

        if tier is not None:
            signal_metadata.setdefault("tier", tier.value)

        if liquidity_class is not None:
            signal_metadata.setdefault("liquidity_class", liquidity_class.value)

        if execution_quality is not None:
            signal_metadata.setdefault("execution_quality", execution_quality.value)

        if requested_leverage is not None:
            if requested_leverage <= 0:
                raise StrategyEvaluationError("requested_leverage must be > 0")
            signal_metadata.setdefault("requested_leverage", float(requested_leverage))

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=signal_metadata,
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=priority,
        )

        if entry_price is not None or stop_loss is not None or take_profit is not None:
            self.attach_basic_trade_plan(
                signal=signal,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                requested_leverage=requested_leverage,
            )

        signal.validate()
        return signal

    def attach_basic_trade_plan(
        self,
        *,
        signal: StrategySignal,
        entry_price: float | None,
        stop_loss: float | None,
        take_profit: float | None = None,
        requested_leverage: float | None = None,
        entry_type: EntryType = EntryType.LIMIT,
    ) -> StrategySignal:
        """
        Attach basic entry/exit/execution draft to an existing signal.

        This is still a draft. RiskManager decides final size/leverage/budget.
        """
        signal.validate()

        entry = EntryPlan(
            entry_type=entry_type,
            price=entry_price,
        )

        targets: list[TargetPlan] = []
        if take_profit is not None:
            targets.append(
                TargetPlan(
                    price=float(take_profit),
                    size_fraction=1.0,
                    rr=None,
                    label="primary",
                )
            )

        exit_plan = ExitPlan(
            exit_types=[ExitType.STOP_LOSS],
            stop_loss=stop_loss,
            take_profit_levels=targets,
            partial_exit_enabled=False,
        )

        if take_profit is not None:
            exit_plan.exit_types.append(ExitType.TAKE_PROFIT)

        invalidation = InvalidationPlan(
            price=stop_loss,
            reason="strategy_invalidation",
        )

        execution_plan = ExecutionPlanDraft(
            symbol=signal.symbol,
            side=signal.side,
            entry=entry,
            exit=exit_plan,
            invalidation=invalidation,
            leverage=requested_leverage,
            reduce_only=False,
        )

        signal.entry_plan = entry
        signal.exit_plan = exit_plan
        signal.invalidation_plan = invalidation
        signal.execution_plan = execution_plan

        signal.metadata.setdefault("entry_price", entry_price)
        signal.metadata.setdefault("stop_loss", stop_loss)
        signal.metadata.setdefault("take_profit", take_profit)

        signal.validate()
        return signal

    @staticmethod
    def _normalize_unit_interval(value: float, field_name: str) -> float:
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if not 0.0 <= value_f <= 1.0:
            raise StrategyEvaluationError(f"{field_name} must be between 0.0 and 1.0")

        return value_f

    @staticmethod
    def _normalize_non_negative(value: float, field_name: str) -> float:
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if value_f < 0.0:
            raise StrategyEvaluationError(f"{field_name} must be >= 0.0")

        return value_f

    @staticmethod
    def _context_regime_for_mixin(context: StrategyContext) -> Any:
        current_regime = getattr(context, "current_regime", None)
        if current_regime is not None:
            return current_regime

        regime = getattr(context, "regime", None)
        if regime is not None:
            value = getattr(regime, "regime", None)
            if value is not None:
                return value

        from ..enums import MarketRegime

        return MarketRegime.UNKNOWN


class StrategyValidationMixin(StrategyMixinSupport):
    """
    Validation helpers for concrete strategies.

    These helpers validate StrategyContext content only. They do not read data
    caches, analytics services, risk state, or execution state directly.
    """

    def require_feature(self, context: StrategyContext, name: str) -> Any:
        self._validate_context_for_mixin(context)

        if not name.strip():
            raise StrategyEvaluationError("feature name cannot be empty")

        if not context.has_feature(name):
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: missing required feature '{name}' "
                f"for symbol {context.symbol}"
            )

        return context.get_feature(name)

    def optional_feature(
        self,
        context: StrategyContext,
        name: str,
        default: Any = None,
    ) -> Any:
        self._validate_context_for_mixin(context)

        if not name.strip():
            raise StrategyEvaluationError("feature name cannot be empty")

        if not context.has_feature(name):
            return default

        return context.get_feature(name)

    def require_domain_value(
        self,
        context: StrategyContext,
        source: FeatureSource,
        key: str,
    ) -> Any:
        self._validate_context_for_mixin(context)

        if not key.strip():
            raise StrategyEvaluationError("domain key cannot be empty")

        domain = context.domain_dict(source)
        if key not in domain:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: missing domain value "
                f"{source.value}.{key} for symbol {context.symbol}"
            )

        return domain[key]

    def optional_domain_value(
        self,
        context: StrategyContext,
        source: FeatureSource,
        key: str,
        default: Any = None,
    ) -> Any:
        self._validate_context_for_mixin(context)

        if not key.strip():
            raise StrategyEvaluationError("domain key cannot be empty")

        return context.domain_dict(source).get(key, default)

    def require_price(self, context: StrategyContext) -> float:
        self._validate_context_for_mixin(context)

        price = getattr(context, "price", None)
        if price is None:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: context.price is required"
            )

        value = (
            getattr(price, "last", None)
            or getattr(price, "close", None)
            or getattr(price, "mark_price", None)
            or getattr(price, "price", None)
        )

        if value is None:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: context.price does not contain usable price"
            )

        value_f = float(value)
        if value_f <= 0:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: context price must be > 0"
            )

        return value_f

    def has_any_feature(
        self,
        context: StrategyContext,
        names: set[str] | list[str] | tuple[str, ...],
    ) -> bool:
        self._validate_context_for_mixin(context)
        return any(context.has_feature(name) for name in names)

    def has_all_features(
        self,
        context: StrategyContext,
        names: set[str] | list[str] | tuple[str, ...],
    ) -> bool:
        self._validate_context_for_mixin(context)
        return all(context.has_feature(name) for name in names)

    @staticmethod
    def require_positive(value: float | int | None, field_name: str) -> float:
        if value is None:
            raise StrategyEvaluationError(f"{field_name} is required")

        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if value_f <= 0:
            raise StrategyEvaluationError(f"{field_name} must be > 0")

        return value_f

    @staticmethod
    def require_non_negative(value: float | int | None, field_name: str) -> float:
        if value is None:
            raise StrategyEvaluationError(f"{field_name} is required")

        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if value_f < 0:
            raise StrategyEvaluationError(f"{field_name} must be >= 0")

        return value_f

    @staticmethod
    def require_unit_interval(value: float | int | None, field_name: str) -> float:
        if value is None:
            raise StrategyEvaluationError(f"{field_name} is required")

        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if not 0.0 <= value_f <= 1.0:
            raise StrategyEvaluationError(f"{field_name} must be between 0.0 and 1.0")

        return value_f


class StrategyRiskRewardMixin(StrategyMixinSupport):
    """
    Pure local helpers for estimating entry/stop/target quality.

    These methods do not replace RiskManager checks. They only help concrete
    strategies produce cleaner StrategySignal metadata before SignalProcessor
    builds the final risk-ready payload.
    """

    @staticmethod
    def calculate_stop_distance(
        *,
        entry: float,
        stop: float,
        side: SignalSide,
    ) -> float:
        entry_f = StrategyValidationMixin.require_positive(entry, "entry")
        stop_f = StrategyValidationMixin.require_positive(stop, "stop")

        if side is SignalSide.LONG:
            distance = entry_f - stop_f
        elif side is SignalSide.SHORT:
            distance = stop_f - entry_f
        else:
            raise StrategyEvaluationError("side must be LONG or SHORT")

        return max(0.0, distance)

    @staticmethod
    def calculate_reward_distance(
        *,
        entry: float,
        target: float,
        side: SignalSide,
    ) -> float:
        entry_f = StrategyValidationMixin.require_positive(entry, "entry")
        target_f = StrategyValidationMixin.require_positive(target, "target")

        if side is SignalSide.LONG:
            distance = target_f - entry_f
        elif side is SignalSide.SHORT:
            distance = entry_f - target_f
        else:
            raise StrategyEvaluationError("side must be LONG or SHORT")

        return max(0.0, distance)

    def calculate_rr(
        self,
        *,
        entry: float,
        stop: float,
        target: float,
        side: SignalSide,
    ) -> float:
        risk = self.calculate_stop_distance(
            entry=entry,
            stop=stop,
            side=side,
        )
        reward = self.calculate_reward_distance(
            entry=entry,
            target=target,
            side=side,
        )

        if risk <= 0:
            return 0.0

        return max(0.0, reward / risk)

    def is_valid_stop(
        self,
        *,
        entry: float,
        stop: float,
        side: SignalSide,
    ) -> bool:
        return self.calculate_stop_distance(
            entry=entry,
            stop=stop,
            side=side,
        ) > 0.0

    def is_valid_target(
        self,
        *,
        entry: float,
        target: float,
        side: SignalSide,
    ) -> bool:
        return self.calculate_reward_distance(
            entry=entry,
            target=target,
            side=side,
        ) > 0.0

    def validate_trade_geometry(
        self,
        *,
        entry: float,
        stop: float,
        target: float | None,
        side: SignalSide,
        min_rr: float | None = None,
    ) -> dict[str, float]:
        entry_f = StrategyValidationMixin.require_positive(entry, "entry")
        stop_f = StrategyValidationMixin.require_positive(stop, "stop")

        stop_distance = self.calculate_stop_distance(
            entry=entry_f,
            stop=stop_f,
            side=side,
        )

        if stop_distance <= 0:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: invalid stop geometry for {side.value}"
            )

        result: dict[str, float] = {
            "entry": entry_f,
            "stop": stop_f,
            "stop_distance": stop_distance,
        }

        if target is not None:
            target_f = StrategyValidationMixin.require_positive(target, "target")
            reward_distance = self.calculate_reward_distance(
                entry=entry_f,
                target=target_f,
                side=side,
            )

            if reward_distance <= 0:
                raise StrategyEvaluationError(
                    f"{self._strategy_name_for_errors}: invalid target geometry for {side.value}"
                )

            rr = reward_distance / stop_distance

            if min_rr is not None and rr < min_rr:
                raise StrategyEvaluationError(
                    f"{self._strategy_name_for_errors}: RR {rr:.4f} is below required {min_rr:.4f}"
                )

            result.update(
                {
                    "target": target_f,
                    "reward_distance": reward_distance,
                    "rr": rr,
                }
            )

        return result

    @staticmethod
    def estimate_expected_value(
        *,
        win_probability: float,
        expected_reward: float,
        expected_loss: float,
        expected_cost: float = 0.0,
    ) -> float:
        probability = StrategyValidationMixin.require_unit_interval(
            win_probability,
            "win_probability",
        )
        reward = StrategyValidationMixin.require_positive(
            expected_reward,
            "expected_reward",
        )
        loss = StrategyValidationMixin.require_positive(
            expected_loss,
            "expected_loss",
        )

        cost = 0.0 if expected_cost is None else float(expected_cost)
        if cost < 0:
            raise StrategyEvaluationError("expected_cost must be >= 0")

        return probability * reward - (1.0 - probability) * loss - cost


class StrategyExecutionMixin:
    """
    Helpers for strategy-side execution metadata.

    Execution remains responsible for real order placement. Risk remains
    responsible for final leverage, size and budget approval.
    """

    def build_execution_cost_payload(
        self,
        *,
        spread_cost: float = 0.0,
        slippage_cost: float = 0.0,
        fee_cost: float = 0.0,
        funding_cost: float = 0.0,
        other_cost: float = 0.0,
        spread_pct: float | None = None,
        slippage_pct: float | None = None,
        quality: StrategyExecutionQuality = StrategyExecutionQuality.ACCEPTABLE,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionCostPayload:
        payload = ExecutionCostPayload(
            spread_cost=max(0.0, float(spread_cost)),
            slippage_cost=max(0.0, float(slippage_cost)),
            fee_cost=max(0.0, float(fee_cost)),
            funding_cost=max(0.0, float(funding_cost)),
            other_cost=max(0.0, float(other_cost)),
            spread_pct=spread_pct,
            slippage_pct=slippage_pct,
            quality=quality,
            metadata=dict(metadata or {}),
        )
        payload.validate()
        return payload

    @staticmethod
    def execution_quality_from_cost_to_reward(
        cost_to_reward: float,
    ) -> StrategyExecutionQuality:
        value = max(0.0, float(cost_to_reward))

        if value <= 0.03:
            return StrategyExecutionQuality.EXCELLENT
        if value <= 0.06:
            return StrategyExecutionQuality.GOOD
        if value <= 0.10:
            return StrategyExecutionQuality.ACCEPTABLE
        if value <= 0.20:
            return StrategyExecutionQuality.POOR
        return StrategyExecutionQuality.BLOCKED

    @staticmethod
    def liquidity_class_from_score(score: float) -> StrategyLiquidityClass:
        value = clamp(float(score), 0.0, 1.0)

        if value >= 0.90:
            return StrategyLiquidityClass.TOP
        if value >= 0.75:
            return StrategyLiquidityClass.HIGH
        if value >= 0.50:
            return StrategyLiquidityClass.NORMAL
        if value >= 0.30:
            return StrategyLiquidityClass.LOW
        if value >= 0.15:
            return StrategyLiquidityClass.ILLIQUID
        return StrategyLiquidityClass.SHITCOIN

    @staticmethod
    def trade_tier_from_priority_score(score: float) -> StrategyTradeTier:
        value = clamp(float(score), 0.0, 1.0)

        if value >= 0.88:
            return StrategyTradeTier.T4
        if value >= 0.74:
            return StrategyTradeTier.T3
        if value >= 0.58:
            return StrategyTradeTier.T2
        return StrategyTradeTier.T1


class TradingStrategy(
    StrategySignalMixin,
    StrategyValidationMixin,
    StrategyRiskRewardMixin,
    StrategyExecutionMixin,
    BaseStrategy,
    ABC,
):
    """
    Base class for concrete domain strategies.

    Concrete strategies should subclass this class and implement:

        async def generate_signal(self, context: StrategyContext) -> StrategySignal | None

    Responsibilities of concrete strategies:
    - read only StrategyContext;
    - detect setup;
    - return StrategySignal or None;
    - optionally attach draft entry/exit metadata.

    Forbidden responsibilities:
    - no direct analytics/data/risk/execution calls;
    - no signal.generated emit;
    - no final position sizing;
    - no budget/exposure checks;
    - no order placement.
    """

    category: StrategyCategory = StrategyCategory.HYBRID
    default_setup_type: SetupType = SetupType.UNKNOWN
    default_timeframe: Timeframe = Timeframe.M1

    def context_feature_score(
        self,
        context: StrategyContext,
        feature_name: str,
        *,
        default: float = 0.0,
    ) -> float:
        value = self.optional_feature(context, feature_name, default)

        try:
            return clamp(float(value), 0.0, 1.0)
        except (TypeError, ValueError):
            return default

    def build_priority_metadata(
        self,
        *,
        setup_quality: float,
        confluence_score: float = 0.0,
        liquidity_score: float = 0.0,
        risk_reward_score: float = 0.0,
        execution_quality_score: float = 0.0,
        regime_alignment_score: float = 0.0,
        freshness_score: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build normalized priority metadata.

        SignalScorer may later overwrite/refine this, but concrete strategies
        can provide useful first-pass components.
        """
        components = {
            "setup_quality": clamp(float(setup_quality), 0.0, 1.0),
            "confluence_score": clamp(float(confluence_score), 0.0, 1.0),
            "liquidity_score": clamp(float(liquidity_score), 0.0, 1.0),
            "risk_reward_score": clamp(float(risk_reward_score), 0.0, 1.0),
            "execution_quality_score": clamp(float(execution_quality_score), 0.0, 1.0),
            "regime_alignment_score": clamp(float(regime_alignment_score), 0.0, 1.0),
            "freshness_score": clamp(float(freshness_score), 0.0, 1.0),
        }

        priority_score = (
            0.25 * components["setup_quality"]
            + 0.20 * components["confluence_score"]
            + 0.15 * components["liquidity_score"]
            + 0.15 * components["risk_reward_score"]
            + 0.10 * components["execution_quality_score"]
            + 0.10 * components["regime_alignment_score"]
            + 0.05 * components["freshness_score"]
        )

        result: dict[str, Any] = {
            "priority_score": clamp(priority_score, 0.0, 1.0),
            "priority_components": components,
        }

        if extra:
            result.update(extra)

        return result

    def build_trade_metadata(
        self,
        *,
        tier: StrategyTradeTier | None = None,
        order_intent: StrategyOrderIntent = StrategyOrderIntent.OPEN,
        liquidity_class: StrategyLiquidityClass | None = None,
        execution_quality: StrategyExecutionQuality | None = None,
        margin_mode: StrategyMarginMode = StrategyMarginMode.ISOLATED,
        market_type: StrategyMarketType = StrategyMarketType.USDM_FUTURES,
        requested_leverage: float | None = None,
        exchange: str | None = None,
        expected_reward: float | None = None,
        expected_loss: float | None = None,
        expected_win_probability: float | None = None,
        expected_cost: float | None = None,
        execution_cost: ExecutionCostPayload | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build metadata that SignalProcessor can later convert into
        RiskReadySignalPayload fields.
        """
        metadata: dict[str, Any] = {
            "order_intent": order_intent.value,
            "margin_mode": margin_mode.value,
            "market_type": market_type.value,
        }

        if tier is not None:
            metadata["tier"] = tier.value

        if liquidity_class is not None:
            metadata["liquidity_class"] = liquidity_class.value

        if execution_quality is not None:
            metadata["execution_quality"] = execution_quality.value

        if requested_leverage is not None:
            if requested_leverage <= 0:
                raise StrategyEvaluationError("requested_leverage must be > 0")
            metadata["requested_leverage"] = float(requested_leverage)

        if exchange is not None:
            metadata["exchange"] = exchange

        if expected_reward is not None:
            metadata["expected_reward"] = float(expected_reward)

        if expected_loss is not None:
            metadata["expected_loss"] = float(expected_loss)

        if expected_win_probability is not None:
            metadata["expected_win_probability"] = self.require_unit_interval(
                expected_win_probability,
                "expected_win_probability",
            )

        if expected_cost is not None:
            if expected_cost < 0:
                raise StrategyEvaluationError("expected_cost must be >= 0")
            metadata["expected_cost"] = float(expected_cost)

        if execution_cost is not None:
            execution_cost.validate()
            metadata["execution_cost"] = execution_cost.to_dict()

        if extra:
            metadata.update(extra)

        return metadata


__all__ = [
    "TradingStrategy",
    "StrategySignalMixin",
    "StrategyValidationMixin",
    "StrategyRiskRewardMixin",
    "StrategyExecutionMixin",
]