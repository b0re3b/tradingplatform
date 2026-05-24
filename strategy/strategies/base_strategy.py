# trading_system/strategy/strategies/base_strategy.py

from __future__ import annotations
import logging

from abc import ABC
from dataclasses import dataclass
from enum import Enum
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


@dataclass(slots=True, frozen=True)
class StrategyRRProfile:
    """
    Strategy-layer risk/reward intent attached to every generated signal.

    The profile is not a RiskManager decision and does not size positions. It only
    gives SignalBuilder a strategy-owned RR intent so fallback TP generation does
    not silently use an incompatible global default.
    """

    min_rr: float
    base_rr: float
    max_rr: float
    source: str
    target_price: float | None = None
    stop_loss: float | None = None
    entry_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_rr": float(self.min_rr),
            "base_rr": float(self.base_rr),
            "max_rr": float(self.max_rr),
            "source": self.source,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "entry_price": self.entry_price,
        }


class StrategyMixinSupport:
    """
    Small helper base for mixins.

    Mixins are used together with BaseStrategy, but static analyzers do not know
    that self has strategy_name / validate_context / validate_context_requirements.
    These wrappers avoid unresolved attribute warnings without weakening runtime
    behavior.
    """
    _logger = logging.getLogger(__name__ + ".StrategyMixinSupport")

    @property
    def _strategy_name_for_errors(self) -> str:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyMixinSupport._strategy_name_for_errors")
        return str(getattr(self, "strategy_name", self.__class__.__name__))

    def _validate_context_for_mixin(self, context: StrategyContext) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyMixinSupport._validate_context_for_mixin")
        validator = getattr(self, "validate_context", None)
        if not callable(validator):
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: validate_context() is not available"
            )
        validator(context)

    def _validate_context_requirements_for_mixin(self, context: StrategyContext) -> None:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyMixinSupport._validate_context_requirements_for_mixin")
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
    _logger = logging.getLogger(__name__ + ".StrategySignalMixin")

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin.build_signal")
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

        self.attach_rr_profile(signal=signal, context=context)

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin.build_directional_signal")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin.attach_basic_trade_plan")
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

        self.attach_rr_profile(
            signal=signal,
            context=None,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        rr_profile = signal.metadata.get("rr_profile")
        if isinstance(rr_profile, dict) and targets:
            rr_value = rr_profile.get("base_rr")
            if rr_value is not None:
                try:
                    targets[0].rr = float(rr_value)
                except (TypeError, ValueError):
                    pass

        signal.validate()
        return signal

    def attach_rr_profile(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> StrategySignal:
        """
        Attach min/base/max RR intent to every strategy signal.

        Priority:
        1. Explicit metadata / attached trade plan prices.
        2. Market-derived target candidates already present in StrategyContext
           or domain metadata.
        3. Tier-aware fallback profile.

        This keeps concrete strategies as decision modules and prevents the
        downstream SignalBuilder from falling back to an RR below the selected
        strategy tier's minimum.
        """
        metadata = signal.metadata

        entry = self._first_float(
            entry_price,
            metadata.get("entry_price"),
            metadata.get("entry"),
            metadata.get("entry_reference_price"),
            metadata.get("reference_price"),
            metadata.get("current_price"),
            metadata.get("price"),
            self._context_price(context),
        )
        stop = self._first_float(
            stop_loss,
            metadata.get("stop_loss"),
            metadata.get("invalidation_price"),
            metadata.get("invalid_price"),
            metadata.get("invalidation", {}).get("price") if isinstance(metadata.get("invalidation"), dict) else None,
        )
        target = self._first_float(
            take_profit,
            metadata.get("take_profit"),
            metadata.get("target_price"),
            metadata.get("primary_target"),
            metadata.get("expected_target"),
        )

        tier_value = self._enum_value(metadata.get("tier"))
        min_rr = self._first_float(
            metadata.get("min_rr"),
            metadata.get("minimum_rr"),
            metadata.get("required_rr"),
            metadata.get("risk_reward_min"),
            self._tier_min_rr(tier_value),
            default=1.8,
        )
        if min_rr is None or min_rr <= 0:
            min_rr = 1.8

        configured_base = self._first_float(
            metadata.get("base_rr"),
            metadata.get("rr"),
            metadata.get("risk_reward"),
            metadata.get("risk_reward_ratio"),
            getattr(getattr(self, "config", None), "default_rr_ratio", None),
            default=min_rr,
        )
        if configured_base is None or configured_base <= 0:
            configured_base = min_rr

        max_candidate = self._max_possible_rr(
            signal=signal,
            context=context,
            entry_price=entry,
            stop_loss=stop,
            explicit_target=target,
        )

        if max_candidate is not None and max_candidate > 0:
            max_rr = max(float(max_candidate), float(min_rr))
            base_rr = min(max(float(configured_base), float(min_rr)), max_rr)
            source = "market_target"
        else:
            max_rr = max(float(configured_base), float(min_rr))
            base_rr = max(float(configured_base), float(min_rr))
            source = "tier_fallback"

        profile = StrategyRRProfile(
            min_rr=float(min_rr),
            base_rr=float(base_rr),
            max_rr=float(max_rr),
            source=source,
            target_price=target,
            stop_loss=stop,
            entry_price=entry,
        )

        metadata["rr_profile"] = profile.to_dict()
        metadata["min_rr"] = float(profile.min_rr)
        metadata["base_rr"] = float(profile.base_rr)
        metadata["max_rr"] = float(profile.max_rr)
        metadata["rr"] = float(profile.base_rr)
        metadata.setdefault("rr_source", profile.source)

        if target is not None:
            metadata.setdefault("target_price", target)
        if entry is not None:
            metadata.setdefault("entry_price", entry)
        if stop is not None:
            metadata.setdefault("stop_loss", stop)

        return signal

    def _max_possible_rr(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None,
        entry_price: float | None,
        stop_loss: float | None,
        explicit_target: float | None,
    ) -> float | None:
        if entry_price is None or stop_loss is None:
            return None

        risk_distance = abs(float(entry_price) - float(stop_loss))
        if risk_distance <= 0:
            return None

        candidates: list[tuple[float, str]] = []
        if explicit_target is not None:
            candidates.append((float(explicit_target), "explicit"))

        candidates.extend(self._rr_target_candidates(signal.metadata))
        if context is not None:
            candidates.extend(self._rr_target_candidates(getattr(context, "features", None)))
            # StrategyContext implementations expose domain data differently
            # across packages; use common accessors without requiring a specific
            # context internals contract.
            for source in FeatureSource:
                try:
                    domain = context.domain_dict(source)
                except Exception:
                    continue
                candidates.extend(self._rr_target_candidates(domain))

        best: float | None = None
        for price, _source in candidates:
            rr = self._calculate_rr(
                side=signal.side,
                entry_price=float(entry_price),
                stop_loss=float(stop_loss),
                target_price=price,
            )
            if rr is None or rr <= 0:
                continue
            best = rr if best is None else max(best, rr)

        return best

    @classmethod
    def _rr_target_candidates(cls, value: Any) -> list[tuple[float, str]]:
        candidates: list[tuple[float, str]] = []
        cls._collect_rr_target_candidates(value, candidates, path="")
        return candidates

    @classmethod
    def _collect_rr_target_candidates(
        cls,
        value: Any,
        candidates: list[tuple[float, str]],
        *,
        path: str,
        depth: int = 0,
    ) -> None:
        if value is None or depth > 5:
            return

        if isinstance(value, dict):
            for key, item in value.items():
                key_s = str(key)
                child_path = f"{path}.{key_s}" if path else key_s
                if cls._looks_like_target_price_key(key_s):
                    item_f = cls._safe_float(item)
                    if item_f is not None and item_f > 0:
                        candidates.append((item_f, child_path))
                cls._collect_rr_target_candidates(item, candidates, path=child_path, depth=depth + 1)
            return

        if isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                cls._collect_rr_target_candidates(item, candidates, path=f"{path}[{index}]", depth=depth + 1)
            return

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                cls._collect_rr_target_candidates(to_dict(), candidates, path=path, depth=depth + 1)
            except Exception:
                return

    @staticmethod
    def _looks_like_target_price_key(key: str) -> bool:
        normalized = key.lower().strip()
        if not normalized:
            return False
        positive_tokens = (
            "take_profit",
            "target_price",
            "primary_target",
            "secondary_target",
            "final_target",
            "expected_target",
            "liquidity_target",
            "sweep_target",
            "reversion_target",
            "breakout_target",
            "continuation_target",
            "fair_value",
            "vwap",
            "poc",
            "value_area_high",
            "value_area_low",
            "swing_high",
            "swing_low",
            "resistance",
            "support",
        )
        negative_tokens = (
            "stop",
            "invalidation",
            "entry",
            "mark_price",
            "index_price",
            "last_price",
            "current_price",
        )
        return any(token in normalized for token in positive_tokens) and not any(
            token in normalized for token in negative_tokens
        )

    @staticmethod
    def _calculate_rr(
        *,
        side: SignalSide,
        entry_price: float,
        stop_loss: float,
        target_price: float,
    ) -> float | None:
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            return None

        if side == SignalSide.LONG:
            reward_distance = target_price - entry_price
        elif side == SignalSide.SHORT:
            reward_distance = entry_price - target_price
        else:
            return None

        if reward_distance <= 0:
            return None
        return reward_distance / risk_distance

    @classmethod
    def _first_float(cls, *values: Any, default: float | None = None) -> float | None:
        for value in values:
            value_f = cls._safe_float(value)
            if value_f is not None:
                return value_f
        return default

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, Enum):
            value = value.value
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f

    @staticmethod
    def _enum_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @staticmethod
    def _tier_min_rr(tier: str | None) -> float | None:
        if tier is None:
            return None
        normalized = tier.lower().strip()
        # Keep this map strategy-local. RiskManager remains the source of truth;
        # these values are used only to keep strategy fallback RR compatible with
        # the default risk tier profiles.
        return {
            "t0": 1.0,
            "t1": 1.2,
            "t2": 1.8,
            "t3": 2.2,
            "t4": 2.8,
            "scalp": 1.2,
            "base": 1.8,
            "standard": 1.8,
            "swing": 2.2,
            "high_conviction": 2.8,
        }.get(normalized)

    @staticmethod
    def _context_price(context: StrategyContext | None) -> float | None:
        if context is None:
            return None
        for attr in ("price", "current_price", "last_price", "mark_price"):
            value = getattr(context, attr, None)
            value_f = StrategySignalMixin._safe_float(value)
            if value_f is not None:
                return value_f
        try:
            for key in ("price", "current_price", "last_price", "mark_price"):
                value_f = StrategySignalMixin._safe_float(context.get_feature(key))
                if value_f is not None:
                    return value_f
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_unit_interval(value: float, field_name: str) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".StrategySignalMixin._normalize_unit_interval")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin._normalize_unit_interval")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if not 0.0 <= value_f <= 1.0:
            raise StrategyEvaluationError(f"{field_name} must be between 0.0 and 1.0")

        return value_f

    @staticmethod
    def _normalize_non_negative(value: float, field_name: str) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".StrategySignalMixin._normalize_non_negative")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin._normalize_non_negative")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise StrategyEvaluationError(f"{field_name} must be numeric") from exc

        if value_f < 0.0:
            raise StrategyEvaluationError(f"{field_name} must be >= 0.0")

        return value_f

    @staticmethod
    def _context_regime_for_mixin(context: StrategyContext) -> Any:
        _strategy_logger = logging.getLogger(__name__ + ".StrategySignalMixin._context_regime_for_mixin")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategySignalMixin._context_regime_for_mixin")
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
    _logger = logging.getLogger(__name__ + ".StrategyValidationMixin")

    def require_feature(self, context: StrategyContext, name: str) -> Any:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_feature")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.optional_feature")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_domain_value")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.optional_domain_value")
        self._validate_context_for_mixin(context)

        if not key.strip():
            raise StrategyEvaluationError("domain key cannot be empty")

        return context.domain_dict(source).get(key, default)

    def require_price(self, context: StrategyContext) -> float:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_price")
        self._validate_context_for_mixin(context)

        price = getattr(context, "price", None)
        if price is None:
            raise StrategyEvaluationError(
                f"{self._strategy_name_for_errors}: context.price is required"
            )

        value = self._extract_price_value(price)

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

    @staticmethod
    def _extract_price_value(price: Any) -> float | None:
        _strategy_logger = logging.getLogger(__name__ + ".StrategyValidationMixin._extract_price_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin._extract_price_value")
        if isinstance(price, dict):
            candidates = (
                "last_price",
                "last",
                "close",
                "mark_price",
                "index_price",
                "price",
                "mid",
                "mid_price",
            )
            for key in candidates:
                value = price.get(key)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if value_f > 0:
                    return value_f
            return None

        for attr in (
                "last_price",
                "last",
                "close",
                "mark_price",
                "index_price",
                "price",
                "mid",
                "mid_price",
        ):
            value = getattr(price, attr, None)
            if value is None:
                continue

            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue

            if value_f > 0:
                return value_f

        return None

    def has_any_feature(
        self,
        context: StrategyContext,
        names: set[str] | list[str] | tuple[str, ...],
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.has_any_feature")
        self._validate_context_for_mixin(context)
        return any(context.has_feature(name) for name in names)

    def has_all_features(
        self,
        context: StrategyContext,
        names: set[str] | list[str] | tuple[str, ...],
    ) -> bool:
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.has_all_features")
        self._validate_context_for_mixin(context)
        return all(context.has_feature(name) for name in names)

    @staticmethod
    def require_positive(value: float | int | None, field_name: str) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".StrategyValidationMixin.require_positive")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_positive")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyValidationMixin.require_non_negative")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_non_negative")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyValidationMixin.require_unit_interval")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyValidationMixin.require_unit_interval")
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
    _logger = logging.getLogger(__name__ + ".StrategyRiskRewardMixin")

    @staticmethod
    def calculate_stop_distance(
        *,
        entry: float,
        stop: float,
        side: SignalSide,
    ) -> float:
        _strategy_logger = logging.getLogger(__name__ + ".StrategyRiskRewardMixin.calculate_stop_distance")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.calculate_stop_distance")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyRiskRewardMixin.calculate_reward_distance")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.calculate_reward_distance")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.calculate_rr")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.is_valid_stop")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.is_valid_target")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.validate_trade_geometry")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyRiskRewardMixin.estimate_expected_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyRiskRewardMixin.estimate_expected_value")
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
    _logger = logging.getLogger(__name__ + ".StrategyExecutionMixin")

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyExecutionMixin.build_execution_cost_payload")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyExecutionMixin.execution_quality_from_cost_to_reward")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyExecutionMixin.execution_quality_from_cost_to_reward")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyExecutionMixin.liquidity_class_from_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyExecutionMixin.liquidity_class_from_score")
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
        _strategy_logger = logging.getLogger(__name__ + ".StrategyExecutionMixin.trade_tier_from_priority_score")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering StrategyExecutionMixin.trade_tier_from_priority_score")
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
    _logger = logging.getLogger(__name__ + ".TradingStrategy")

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradingStrategy.context_feature_score")
        value = self.optional_feature(context, feature_name, default)

        # StrategyContext.get_feature() returns FeatureSnapshot in the current
        # strategy model.  Older helper code tried float(snapshot), which always
        # failed and silently returned default=0.0.  That can make concrete
        # strategies score every setup as zero and return no passed signals.
        raw_value = self._feature_snapshot_numeric_value(value, default=default)

        try:
            return clamp(float(raw_value), 0.0, 1.0)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _feature_snapshot_numeric_value(value: Any, *, default: float = 0.0) -> Any:
        """
        Extract a numeric value from FeatureSnapshot-like objects.

        Preference:
        1. normalized_value because it is already scaled to the strategy domain;
        2. value for raw FeatureSnapshot payloads;
        3. the value itself for primitive floats/ints/strings.
        """
        _strategy_logger = logging.getLogger(__name__ + ".TradingStrategy._feature_snapshot_numeric_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradingStrategy._feature_snapshot_numeric_value")
        if value is None:
            return default

        normalized_value = getattr(value, "normalized_value", None)
        if normalized_value is not None:
            return normalized_value

        snapshot_value = getattr(value, "value", None)
        if snapshot_value is not None:
            return snapshot_value

        return value

    @staticmethod
    def _extract_price_value(price: Any) -> float | int | str | None:
        """
        Extract a usable last/current price from PriceSnapshot-like objects.

        PriceSnapshot in strategy.models uses last_price.  The previous helper
        checked only last/close/mark_price/price, so valid StrategyContext.price
        objects built by SignalNormalizer/StrategyContextBuilder could be
        rejected as unusable.
        """
        _strategy_logger = logging.getLogger(__name__ + ".TradingStrategy._extract_price_value")
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradingStrategy._extract_price_value")
        if price is None:
            return None

        if isinstance(price, dict):
            for key in (
                "last_price",
                "last",
                "close",
                "mark_price",
                "index_price",
                "price",
                "mid",
                "mid_price",
            ):
                value = price.get(key)
                if value is not None:
                    return value
            return None

        for attr in (
            "last_price",
            "last",
            "close",
            "mark_price",
            "index_price",
            "price",
            "mid",
            "mid_price",
        ):
            value = getattr(price, attr, None)
            if value is not None:
                return value

        return None

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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradingStrategy.build_priority_metadata")
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
        _strategy_logger = getattr(self, "logger", None) or getattr(self, "_logger", None) or logging.getLogger(__name__ + "." + self.__class__.__name__)
        if _strategy_logger.isEnabledFor(logging.DEBUG):
            _strategy_logger.debug("Entering TradingStrategy.build_trade_metadata")
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