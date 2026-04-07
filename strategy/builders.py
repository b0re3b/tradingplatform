from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ContextAwareComponent
from .config import BuilderConfig, StrategyConfig
from .context import StrategyContext
from .enums import EntryType, ExitType, SignalSide
from .exceptions import BuilderError
from .models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    InvalidationPlan,
    StrategySignal,
    TargetPlan,
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass(slots=True)
class BuildEvaluation:
    """
    Результат побудови execution/trade plan.
    """

    signal: StrategySignal
    context_symbol: str

    entry: EntryPlan | None = None
    invalidation: InvalidationPlan | None = None
    targets: list[TargetPlan] = field(default_factory=list)
    exit_plan: ExitPlan | None = None
    execution_plan: ExecutionPlanDraft | None = None

    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.accepted = False
        if reason not in self.reasons:
            self.reasons.append(reason)


class BaseBuilder(ContextAwareComponent):
    """
    Базовий клас для builder-компонентів.
    """

    builder_name: str = "base_builder"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def builders_config(self) -> BuilderConfig:
        return self.config.builders

    @property
    def name(self) -> str:
        return self.builder_name

    def validate_signal_context(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> None:
        self.validate_context(context)
        signal.validate()

        if signal.symbol != context.symbol:
            raise BuilderError(
                f"{self.name}: signal symbol '{signal.symbol}' != context symbol '{context.symbol}'"
            )

        if not signal.is_directional:
            raise BuilderError(f"{self.name}: only directional signals are supported")

    def resolve_reference_price(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> float:
        if signal.entry_plan is not None and signal.entry_plan.price is not None:
            return float(signal.entry_plan.price)

        if context.price is not None:
            if context.price.mid_price is not None:
                return float(context.price.mid_price)
            if context.price.last_price is not None:
                return float(context.price.last_price)

        entry_price = context.get_feature("entry_price")
        if isinstance(entry_price, (int, float)) and entry_price > 0:
            return float(entry_price)

        raise BuilderError(f"{self.name}: unable to resolve reference price")

    def resolve_slippage_bps(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> float | None:
        if signal.entry_plan is not None and signal.entry_plan.max_slippage_bps is not None:
            return signal.entry_plan.max_slippage_bps

        value = context.get_feature("max_slippage_bps")
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)

        return None


class EntryBuilder(BaseBuilder):
    builder_name = "entry_builder"

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> EntryPlan:
        self.validate_signal_context(signal=signal, context=context)

        if signal.entry_plan is not None:
            signal.entry_plan.validate()
            return signal.entry_plan

        entry_type = self._resolve_entry_type(signal=signal, context=context)
        reference_price = self.resolve_reference_price(signal=signal, context=context)
        entry_price = self._resolve_entry_price(
            signal=signal,
            context=context,
            entry_type=entry_type,
            reference_price=reference_price,
        )

        plan = EntryPlan(
            entry_type=entry_type,
            price=entry_price,
            timeout_seconds=self._resolve_timeout_seconds(signal=signal, context=context),
            max_slippage_bps=self.resolve_slippage_bps(signal=signal, context=context),
            confirmation_required=self._resolve_confirmation_required(signal=signal, context=context),
            notes=self._build_notes(signal=signal, context=context, entry_type=entry_type),
            metadata={
                "builder": self.name,
                "reference_price": reference_price,
            },
        )
        plan.validate()
        return plan

    def _resolve_entry_type(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> EntryType:
        if signal.entry_plan is not None:
            return signal.entry_plan.entry_type

        feature_value = context.get_feature("entry_type")
        if isinstance(feature_value, str):
            try:
                return EntryType(feature_value)
            except ValueError:
                pass

        return self.builders_config.default_entry_type

    def _resolve_entry_price(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry_type: EntryType,
        reference_price: float,
    ) -> float:
        if entry_type == EntryType.MARKET:
            return reference_price

        pullback_price = context.get_feature("pullback_entry_price")
        breakout_price = context.get_feature("breakout_entry_price")
        passive_price = context.get_feature("passive_entry_price")

        if entry_type == EntryType.PULLBACK and isinstance(pullback_price, (int, float)) and pullback_price > 0:
            return float(pullback_price)

        if entry_type == EntryType.BREAKOUT_CONFIRMATION and isinstance(breakout_price, (int, float)) and breakout_price > 0:
            return float(breakout_price)

        if entry_type == EntryType.PASSIVE and isinstance(passive_price, (int, float)) and passive_price > 0:
            return float(passive_price)

        return reference_price

    def _resolve_timeout_seconds(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> int | None:
        value = context.get_feature("entry_timeout_seconds")
        if isinstance(value, int) and value > 0:
            return value

        if signal.entry_plan is not None:
            return signal.entry_plan.timeout_seconds

        return None

    def _resolve_confirmation_required(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> bool:
        value = context.get_feature("entry_confirmation_required")
        if isinstance(value, bool):
            return value

        if signal.entry_plan is not None:
            return signal.entry_plan.confirmation_required

        return False

    def _build_notes(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry_type: EntryType,
    ) -> list[str]:
        notes: list[str] = [f"entry_type:{entry_type.value}"]

        if signal.reasons:
            notes.append(f"reasons:{','.join(signal.reasons[:3])}")

        if context.price is not None and context.price.spread_bps is not None:
            notes.append(f"spread_bps:{context.price.spread_bps}")

        return notes


class InvalidationBuilder(BaseBuilder):
    builder_name = "invalidation_builder"

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry: EntryPlan | None = None,
    ) -> InvalidationPlan:
        self.validate_signal_context(signal=signal, context=context)

        if signal.invalidation_plan is not None:
            signal.invalidation_plan.validate()
            return signal.invalidation_plan

        if not self.builders_config.require_invalidation:
            return InvalidationPlan(
                price=None,
                reason="invalidation_not_required",
                metadata={"builder": self.name},
            )

        entry_price = entry.price if entry is not None and entry.price is not None else self.resolve_reference_price(
            signal=signal,
            context=context,
        )

        invalidation_price = self._resolve_invalidation_price(
            signal=signal,
            context=context,
            entry_price=entry_price,
        )

        plan = InvalidationPlan(
            price=invalidation_price,
            reason=self._resolve_invalidation_reason(signal=signal, context=context),
            timeout_seconds=self._resolve_invalidation_timeout(signal=signal, context=context),
            conditions=self._resolve_conditions(signal=signal, context=context),
            metadata={
                "builder": self.name,
                "entry_price": entry_price,
            },
        )
        plan.validate()
        return plan

    def _resolve_invalidation_price(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry_price: float,
    ) -> float | None:
        explicit = context.get_feature("invalidation_price")
        if isinstance(explicit, (int, float)) and explicit > 0:
            return float(explicit)

        stop_loss = context.get_feature("stop_loss")
        if isinstance(stop_loss, (int, float)) and stop_loss > 0:
            return float(stop_loss)

        structure_low = context.get_feature("structure_low")
        structure_high = context.get_feature("structure_high")
        liquidity_low = context.get_feature("liquidity_low")
        liquidity_high = context.get_feature("liquidity_high")

        if signal.is_long:
            for candidate in (structure_low, liquidity_low):
                if isinstance(candidate, (int, float)) and 0 < float(candidate) < entry_price:
                    return float(candidate)

            fallback_pct = context.get_feature("fallback_stop_pct", 0.003)
            if isinstance(fallback_pct, (int, float)) and fallback_pct > 0:
                return entry_price * (1.0 - float(fallback_pct))

        if signal.is_short:
            for candidate in (structure_high, liquidity_high):
                if isinstance(candidate, (int, float)) and float(candidate) > entry_price:
                    return float(candidate)

            fallback_pct = context.get_feature("fallback_stop_pct", 0.003)
            if isinstance(fallback_pct, (int, float)) and fallback_pct > 0:
                return entry_price * (1.0 + float(fallback_pct))

        return None

    def _resolve_invalidation_reason(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> str:
        explicit_reason = context.get_feature("invalidation_reason")
        if isinstance(explicit_reason, str) and explicit_reason.strip():
            return explicit_reason

        if signal.is_long:
            return "bullish_setup_invalidated"
        return "bearish_setup_invalidated"

    def _resolve_invalidation_timeout(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> int | None:
        value = context.get_feature("invalidation_timeout_seconds")
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _resolve_conditions(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> list[str]:
        conditions = context.get_feature("invalidation_conditions")
        if isinstance(conditions, list):
            return [str(item) for item in conditions]

        if signal.is_long:
            return ["loss_of_bullish_structure"]
        return ["loss_of_bearish_structure"]


class TargetBuilder(BaseBuilder):
    builder_name = "target_builder"

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        entry: EntryPlan,
        invalidation: InvalidationPlan | None,
    ) -> list[TargetPlan]:
        self.validate_signal_context(signal=signal, context=context)
        entry.validate()
        if invalidation is not None:
            invalidation.validate()

        explicit_targets = self._explicit_targets(context)
        if explicit_targets:
            for target in explicit_targets:
                target.validate()
            return explicit_targets

        if entry.price is None:
            raise BuilderError(f"{self.name}: entry price is required for target building")

        risk_distance = self._resolve_risk_distance(
            entry_price=entry.price,
            invalidation=invalidation,
            context=context,
        )
        rr_ratio = self._resolve_rr_ratio(context=context)

        targets = self._build_rr_targets(
            signal=signal,
            entry_price=entry.price,
            risk_distance=risk_distance,
            rr_ratio=rr_ratio,
        )

        for target in targets:
            target.validate()
        return targets

    def _explicit_targets(self, context: StrategyContext) -> list[TargetPlan]:
        raw = context.get_feature("targets")
        if not isinstance(raw, list):
            return []

        result: list[TargetPlan] = []
        for item in raw:
            if isinstance(item, dict):
                price = item.get("price")
                if not isinstance(price, (int, float)) or price <= 0:
                    continue
                result.append(
                    TargetPlan(
                        price=float(price),
                        size_fraction=float(item.get("size_fraction", 1.0)),
                        rr=float(item["rr"]) if isinstance(item.get("rr"), (int, float)) else None,
                        label=item.get("label"),
                        metadata=item.get("metadata", {}),
                    )
                )
            elif isinstance(item, (int, float)) and item > 0:
                result.append(TargetPlan(price=float(item)))
        return result

    def _resolve_risk_distance(
        self,
        *,
        entry_price: float,
        invalidation: InvalidationPlan | None,
        context: StrategyContext,
    ) -> float:
        if invalidation is not None and invalidation.price is not None:
            risk_distance = abs(entry_price - invalidation.price)
            if risk_distance > 0:
                return risk_distance

        atr = context.get_feature("atr")
        if isinstance(atr, (int, float)) and atr > 0:
            return float(atr)

        fallback_pct = context.get_feature("fallback_target_stop_pct", 0.003)
        if isinstance(fallback_pct, (int, float)) and fallback_pct > 0:
            return entry_price * float(fallback_pct)

        raise BuilderError(f"{self.name}: unable to resolve risk distance")

    def _resolve_rr_ratio(self, *, context: StrategyContext) -> float:
        rr = context.get_feature("rr_ratio")
        if isinstance(rr, (int, float)) and rr > 0:
            return float(rr)

        return self.builders_config.default_rr_ratio

    def _build_rr_targets(
        self,
        *,
        signal: StrategySignal,
        entry_price: float,
        risk_distance: float,
        rr_ratio: float,
    ) -> list[TargetPlan]:
        levels = self.builders_config.default_partial_tp_levels
        if not levels:
            levels = [1.0]

        base_target_distance = risk_distance * rr_ratio
        targets: list[TargetPlan] = []

        cumulative = 1.0
        if len(levels) == 1:
            target_price = self._project_price(
                side=signal.side,
                entry_price=entry_price,
                distance=base_target_distance,
            )
            targets.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=1.0,
                    rr=rr_ratio,
                    label="tp1",
                    metadata={"builder": self.name},
                )
            )
            return targets

        # Якщо кілька TP — розтягуємо їх по 0.5R, 1R, 1.5R ... відносно базового rr
        for index, size_fraction in enumerate(levels, start=1):
            multiplier = index / len(levels)
            projected_distance = base_target_distance * multiplier
            target_price = self._project_price(
                side=signal.side,
                entry_price=entry_price,
                distance=projected_distance,
            )
            targets.append(
                TargetPlan(
                    price=target_price,
                    size_fraction=float(size_fraction),
                    rr=rr_ratio * multiplier,
                    label=f"tp{index}",
                    metadata={"builder": self.name},
                )
            )

        return targets

    def _project_price(
        self,
        *,
        side: SignalSide,
        entry_price: float,
        distance: float,
    ) -> float:
        if side == SignalSide.LONG:
            return entry_price + distance
        if side == SignalSide.SHORT:
            return entry_price - distance
        raise BuilderError(f"{self.name}: unsupported side for projection: {side}")


class ExitBuilder(BaseBuilder):
    builder_name = "exit_builder"

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        invalidation: InvalidationPlan | None,
        targets: list[TargetPlan],
    ) -> ExitPlan:
        self.validate_signal_context(signal=signal, context=context)

        if signal.exit_plan is not None:
            signal.exit_plan.validate()
            return signal.exit_plan

        stop_loss = invalidation.price if invalidation is not None else None
        trailing_distance = self._resolve_trailing_distance(context)
        max_holding_seconds = self._resolve_max_holding_seconds(context)

        exit_types = [ExitType.STOP_LOSS, ExitType.TAKE_PROFIT]
        if trailing_distance is not None:
            exit_types.append(ExitType.TRAILING_STOP)

        if max_holding_seconds is not None:
            exit_types.append(ExitType.TIME_EXIT)

        plan = ExitPlan(
            exit_types=exit_types,
            stop_loss=stop_loss,
            take_profit_levels=targets,
            trailing_distance=trailing_distance,
            max_holding_seconds=max_holding_seconds,
            partial_exit_enabled=self.builders_config.enable_partial_take_profit and len(targets) > 1,
            metadata={"builder": self.name},
        )
        plan.validate()
        return plan

    def _resolve_trailing_distance(self, context: StrategyContext) -> float | None:
        trailing = context.get_feature("trailing_distance")
        if isinstance(trailing, (int, float)) and trailing > 0:
            return float(trailing)

        trailing_bps = context.get_feature("trailing_bps")
        if isinstance(trailing_bps, (int, float)) and trailing_bps > 0:
            mid_price = None
            if context.price is not None:
                mid_price = context.price.mid_price
            if mid_price is not None and mid_price > 0:
                return mid_price * float(trailing_bps) / 10000.0

        return None

    def _resolve_max_holding_seconds(self, context: StrategyContext) -> int | None:
        value = context.get_feature("max_holding_seconds")
        if isinstance(value, int) and value > 0:
            return value
        return None


class ExecutionPlanBuilder(BaseBuilder):
    builder_name = "execution_plan_builder"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.entry_builder = EntryBuilder(config=config, event_bus=event_bus, logger=logger)
        self.invalidation_builder = InvalidationBuilder(config=config, event_bus=event_bus, logger=logger)
        self.target_builder = TargetBuilder(config=config, event_bus=event_bus, logger=logger)
        self.exit_builder = ExitBuilder(config=config, event_bus=event_bus, logger=logger)

    def build(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        attach_to_signal: bool = True,
    ) -> BuildEvaluation:
        self.validate_signal_context(signal=signal, context=context)

        evaluation = BuildEvaluation(
            signal=signal,
            context_symbol=context.symbol,
        )

        try:
            entry = self.entry_builder.build(
                signal=signal,
                context=context,
            )
            evaluation.entry = entry

            invalidation = self.invalidation_builder.build(
                signal=signal,
                context=context,
                entry=entry,
            )
            evaluation.invalidation = invalidation

            targets = self.target_builder.build(
                signal=signal,
                context=context,
                entry=entry,
                invalidation=invalidation,
            )
            evaluation.targets = targets

            exit_plan = self.exit_builder.build(
                signal=signal,
                context=context,
                invalidation=invalidation,
                targets=targets,
            )
            evaluation.exit_plan = exit_plan

            execution_plan = ExecutionPlanDraft(
                symbol=signal.symbol,
                side=signal.side,
                entry=entry,
                exit=exit_plan,
                invalidation=invalidation,
                leverage=self._resolve_leverage(context),
                reduce_only=self._resolve_reduce_only(context),
                post_only=self._resolve_post_only(entry),
                expected_holding_seconds=exit_plan.max_holding_seconds,
                notes=self._build_notes(signal=signal, context=context),
                metadata={
                    "builder": self.name,
                    "signal_strategy": signal.strategy_name,
                    "signal_confidence": signal.confidence,
                    "signal_score": signal.score,
                },
            )
            execution_plan.validate()
            evaluation.execution_plan = execution_plan

            if attach_to_signal:
                signal.entry_plan = entry
                signal.invalidation_plan = invalidation
                signal.exit_plan = exit_plan
                signal.execution_plan = execution_plan

        except Exception as exc:
            evaluation.reject(str(exc))
            raise BuilderError(f"{self.name}: build failed: {exc}") from exc

        return evaluation

    def _resolve_leverage(self, context: StrategyContext) -> float | None:
        value = context.get_feature("leverage")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return None

    def _resolve_reduce_only(self, context: StrategyContext) -> bool:
        value = context.get_feature("reduce_only")
        if isinstance(value, bool):
            return value
        return False

    def _resolve_post_only(self, entry: EntryPlan) -> bool:
        return entry.entry_type in {EntryType.LIMIT, EntryType.PASSIVE}

    def _build_notes(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> list[str]:
        notes: list[str] = [
            f"strategy:{signal.strategy_name}",
            f"confidence:{signal.confidence:.4f}",
            f"score:{signal.score:.4f}",
        ]

        if signal.combined_from:
            notes.append(f"combined_from:{','.join(signal.combined_from)}")

        if context.regime is not None:
            notes.append(f"regime:{context.regime.regime.value}")

        return notes