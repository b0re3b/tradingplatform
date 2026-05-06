# strategy/strategies/whales/whale_absorption_strategy.py

from __future__ import annotations

from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from strategy.config import StrategyConfig
from strategy.enums import (
    EntryType,
    ExitType,
    SetupType,
    SignalOrigin,
    SignalSide,
    StrategyCategory,
    TriggerType,
)
from strategy.models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    FilterResult,
    InvalidationPlan,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
    TargetPlan,
)
from strategy.strategies.whales.base import (
    LoggerLike,
    WhaleStrategyBase,
    WhaleStrategyEventConfig,
)


class WhaleAbsorptionStrategy(WhaleStrategyBase):
    """
    Whale absorption strategy.

    Ідея:
        Стратегія шукає ситуації, коли великі учасники — whales —
        абсорбують агресивний потік з протилежного боку ринку.

    Bullish сценарій:
        - домінує whale buy pressure;
        - є sell-side liquidation / aggressive sell pressure;
        - sell-side cluster демонструє exhaustion;
        - є контекст для реверсу / поглинання.

    Bearish сценарій:
        - домінує whale sell pressure;
        - є buy-side liquidation / aggressive buy pressure;
        - buy-side cluster демонструє exhaustion;
        - є контекст для реверсу / поглинання.

    Джерела даних:
        - context.whales
        - context.feature_map

    Очікувані whale-сутності:
        - whale_pressure
        - whale_liquidation_context
        - whale_cluster
        - whale_cluster_update
        - whale_cluster_exhaustion
    """

    DEFAULT_STRATEGY_NAME = "whale_absorption_strategy"

    def __init__(
        self,
        *,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        event_config: WhaleStrategyEventConfig | None = None,
        logger: LoggerLike | None = None,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            event_config=event_config,
            logger=logger,
            strategy_name=strategy_name,
        )

    @property
    def required_features(self) -> set[str]:
        definition = self._strategy_definition
        if definition is not None and definition.required_features:
            return set(definition.required_features)

        return {
            "whale_pressure",
            "whale_liquidation_context",
            "whale_cluster",
            "whale_cluster_update",
            "whale_cluster_exhaustion",
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        """
        Основний sync-вхід у стратегію.

        Повертає:
            - StrategyEvaluation з signal, якщо setup валідний;
            - StrategyEvaluation(passed=False), якщо setup відсутній
              або не пройшов фільтри.
        """
        try:
            self.validate_context(context)

            evaluation = StrategyEvaluation(
                strategy_name=self.strategy_name,
                symbol=context.symbol,
                timestamp=context.timestamp,
                passed=False,
            )

            if not self._runtime_config.enabled:
                evaluation.reasons.append("Strategy disabled")
                return evaluation

            inputs = self._extract_inputs(context)
            signal_side = self._determine_signal_side(inputs)

            if signal_side == SignalSide.UNKNOWN:
                evaluation.reasons.append("No whale absorption setup detected")
                return evaluation

            score = self._calculate_score(
                inputs=inputs,
                side=signal_side,
            )
            confidence = self._calculate_confidence(
                inputs=inputs,
                side=signal_side,
            )

            evaluation.score = score
            evaluation.confidence = confidence

            if score < self._min_absorption_score:
                evaluation.reasons.append(
                    f"Score below threshold: "
                    f"{score:.4f} < {self._min_absorption_score:.4f}"
                )
                return evaluation

            if confidence < self._runtime_config.min_confidence:
                evaluation.reasons.append(
                    f"Confidence below threshold: "
                    f"{confidence:.4f} < {self._runtime_config.min_confidence:.4f}"
                )
                return evaluation

            signal = self._build_signal(
                context=context,
                inputs=inputs,
                side=signal_side,
                score=score,
                confidence=confidence,
            )

            filter_results = self._run_filters(
                context=context,
                signal=signal,
                inputs=inputs,
            )

            for result in filter_results:
                signal.add_filter_result(result)

            if not signal.passed_filters:
                evaluation.reasons.extend(
                    [
                        f"{result.name}: {result.reason}"
                        for result in filter_results
                        if result.blocked
                    ]
                )
                evaluation.signal = signal
                return evaluation

            if signal.score < self._runtime_config.min_score:
                evaluation.reasons.append(
                    f"Signal score below runtime threshold: "
                    f"{signal.score:.4f} < {self._runtime_config.min_score:.4f}"
                )
                evaluation.signal = signal
                return evaluation

            evaluation.signal = signal
            evaluation.passed = True
            evaluation.reasons.extend(signal.reasons)
            return evaluation

        except Exception as exc:
            raise self._wrap_evaluation_error(
                context=context,
                exc=exc,
            ) from exc

    # =========================================================================
    # Thresholds / tunables
    # =========================================================================

    @property
    def _min_pressure_imbalance_ratio(self) -> float:
        return float(
            self._metadata.get(
                "min_pressure_imbalance_ratio",
                0.62,
            )
        )

    @property
    def _min_context_strength(self) -> float:
        return float(
            self._metadata.get(
                "min_context_strength",
                0.55,
            )
        )

    @property
    def _min_cluster_score(self) -> float:
        return float(
            self._metadata.get(
                "min_cluster_score",
                0.45,
            )
        )

    @property
    def _min_exhaustion_probability(self) -> float:
        return float(
            self._metadata.get(
                "min_exhaustion_probability",
                0.55,
            )
        )

    @property
    def _min_absorption_score(self) -> float:
        return float(
            self._metadata.get(
                "min_absorption_score",
                0.55,
            )
        )

    @property
    def _require_opposite_liquidation(self) -> bool:
        return bool(
            self._metadata.get(
                "require_opposite_liquidation",
                True,
            )
        )

    @property
    def _require_cluster_confirmation(self) -> bool:
        return bool(
            self._metadata.get(
                "require_cluster_confirmation",
                True,
            )
        )

    @property
    def _require_exhaustion_confirmation(self) -> bool:
        return bool(
            self._metadata.get(
                "require_exhaustion_confirmation",
                False,
            )
        )

    @property
    def _default_stop_buffer_bps(self) -> float:
        return float(
            self._metadata.get(
                "default_stop_buffer_bps",
                25.0,
            )
        )

    @property
    def _default_rr_ratio(self) -> float:
        return float(
            self._metadata.get(
                "default_rr_ratio",
                self.config.builders.default_rr_ratio,
            )
        )

    # =========================================================================
    # Input extraction
    # =========================================================================

    def _extract_inputs(
        self,
        context: SignalContext,
    ) -> dict[str, dict[str, Any]]:
        return {
            "pressure": self._resolve_payload(
                context,
                names=(
                    "whale_pressure",
                    "whale_pressure_signal",
                    "analytics.whales.whale_pressure",
                ),
            ),
            "liquidation_context": self._resolve_payload(
                context,
                names=(
                    "whale_liquidation_context",
                    "whale_liquidation_context_signal",
                    "analytics.whales.whale_liquidation_context",
                ),
            ),
            "cluster": self._resolve_payload(
                context,
                names=(
                    "whale_cluster",
                    "whale_cluster_signal",
                    "analytics.whales.whale_cluster",
                ),
            ),
            "cluster_update": self._resolve_payload(
                context,
                names=(
                    "whale_cluster_update",
                    "whale_cluster_update_signal",
                    "analytics.whales.whale_cluster_update",
                ),
            ),
            "cluster_exhaustion": self._resolve_payload(
                context,
                names=(
                    "whale_cluster_exhaustion",
                    "whale_cluster_exhaustion_signal",
                    "analytics.whales.whale_cluster_exhaustion",
                ),
            ),
        }

    # =========================================================================
    # Setup detection
    # =========================================================================

    def _determine_signal_side(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> SignalSide:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        dominant_side = str(
            pressure.get("dominant_side", "")
        ).lower()

        whale_side = str(
            liq_ctx.get("whale_side", "")
        ).lower()

        liquidation_side = str(
            liq_ctx.get("liquidation_side", "")
        ).lower()

        cluster_side = str(
            cluster.get("cluster_side")
            or cluster_update.get("cluster_side")
            or cluster_exhaustion.get("cluster_side")
            or ""
        ).lower()

        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability")
        )

        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score")
        )

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
        )

        context_strength = self._safe_float(
            liq_ctx.get("context_strength")
        )

        if (
            imbalance_ratio is None
            or imbalance_ratio < self._min_pressure_imbalance_ratio
        ):
            return SignalSide.UNKNOWN

        if (
            context_strength is None
            or context_strength < self._min_context_strength
        ):
            return SignalSide.UNKNOWN

        if self._require_cluster_confirmation:
            if (
                cluster_score is None
                or cluster_score < self._min_cluster_score
            ):
                return SignalSide.UNKNOWN

        if self._require_exhaustion_confirmation:
            if (
                exhaustion_probability is None
                or exhaustion_probability < self._min_exhaustion_probability
            ):
                return SignalSide.UNKNOWN

        bullish_absorption = (
            dominant_side == "buy"
            and whale_side == "buy"
            and (
                not self._require_opposite_liquidation
                or liquidation_side == "sell"
            )
            and (
                cluster_side in {"sell", "unknown", ""}
                or (
                    exhaustion_probability is not None
                    and exhaustion_probability >= self._min_exhaustion_probability
                )
            )
        )

        bearish_absorption = (
            dominant_side == "sell"
            and whale_side == "sell"
            and (
                not self._require_opposite_liquidation
                or liquidation_side == "buy"
            )
            and (
                cluster_side in {"buy", "unknown", ""}
                or (
                    exhaustion_probability is not None
                    and exhaustion_probability >= self._min_exhaustion_probability
                )
            )
        )

        if bullish_absorption:
            return SignalSide.LONG

        if bearish_absorption:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    # =========================================================================
    # Scoring
    # =========================================================================

    def _calculate_score(
        self,
        *,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
    ) -> float:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        context_strength = self._safe_float(
            liq_ctx.get("context_strength"),
            default=0.0,
        )
        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )
        continuation_probability = self._safe_float(
            cluster.get("continuation_probability")
            or cluster_update.get("continuation_probability"),
            default=0.0,
        )

        base_score = (
            imbalance_ratio * 0.35
            + context_strength * 0.30
            + cluster_score * 0.15
            + exhaustion_probability * 0.15
            + (1.0 - continuation_probability) * 0.05
        )

        if side == SignalSide.UNKNOWN:
            return 0.0

        return self._clamp(
            base_score,
            0.0,
            1.0,
        )

    def _calculate_confidence(
        self,
        *,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
    ) -> float:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        context_strength = self._safe_float(
            liq_ctx.get("context_strength"),
            default=0.0,
        )
        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )

        confidence = (
            imbalance_ratio * 0.30
            + context_strength * 0.30
            + cluster_score * 0.20
            + exhaustion_probability * 0.20
        )

        if side == SignalSide.UNKNOWN:
            return 0.0

        return self._clamp(
            confidence,
            0.0,
            1.0,
        )

    # =========================================================================
    # Signal building
    # =========================================================================

    def _build_signal(
        self,
        *,
        context: SignalContext,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
        score: float,
        confidence: float,
    ) -> StrategySignal:
        signal = StrategySignal(
            symbol=context.symbol,
            side=side,
            strategy_name=self.strategy_name,
            category=StrategyCategory.WHALES,
            timeframe=context.timeframe,
            setup_type=SetupType.ABSORPTION,
            timestamp=context.timestamp,
            confidence=confidence,
            score=score,
            strength=self._resolve_strength(
                score,
                confidence,
            ),
            confidence_grade=self._resolve_confidence_grade(
                confidence,
            ),
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._map_priority(self.priority),
            regime=self._resolve_regime(context),
            metadata={
                "strategy_type": "whale_absorption",
                "inputs_present": {
                    key: bool(value)
                    for key, value in inputs.items()
                },
            },
        )

        self._append_reasons(
            signal=signal,
            inputs=inputs,
            side=side,
        )
        self._append_confirmations(
            signal=signal,
            inputs=inputs,
            side=side,
        )
        self._append_source_features(signal)

        if context.price is not None:
            execution_plan = self._build_execution_plan(
                context=context,
                side=side,
                confidence=confidence,
            )

            if execution_plan is not None:
                signal.execution_plan = execution_plan
                signal.entry_plan = execution_plan.entry
                signal.exit_plan = execution_plan.exit
                signal.invalidation_plan = execution_plan.invalidation

        return signal

    def _append_reasons(
        self,
        *,
        signal: StrategySignal,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
    ) -> None:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        context_strength = self._safe_float(
            liq_ctx.get("context_strength"),
            default=0.0,
        )
        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )

        signal.add_reason(
            f"Whale absorption detected on {side.value}"
        )
        signal.add_reason(
            f"Pressure imbalance ratio={imbalance_ratio:.4f}"
        )
        signal.add_reason(
            f"Liquidation context strength={context_strength:.4f}"
        )

        if cluster_score > 0:
            signal.add_reason(
                f"Cluster score={cluster_score:.4f}"
            )

        if exhaustion_probability > 0:
            signal.add_reason(
                f"Exhaustion probability={exhaustion_probability:.4f}"
            )

    def _append_confirmations(
        self,
        *,
        signal: StrategySignal,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
    ) -> None:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        dominant_side = str(
            pressure.get("dominant_side", "")
        ).lower()

        whale_side = str(
            liq_ctx.get("whale_side", "")
        ).lower()

        liquidation_side = str(
            liq_ctx.get("liquidation_side", "")
        ).lower()

        cluster_side = str(
            cluster.get("cluster_side")
            or cluster_update.get("cluster_side")
            or cluster_exhaustion.get("cluster_side")
            or ""
        ).lower()

        if dominant_side:
            signal.add_confirmation(
                f"Dominant whale pressure={dominant_side}"
            )

        if whale_side:
            signal.add_confirmation(
                f"Whale side={whale_side}"
            )

        if liquidation_side:
            signal.add_confirmation(
                f"Liquidation side={liquidation_side}"
            )

        if cluster_side:
            signal.add_confirmation(
                f"Cluster side={cluster_side}"
            )

        if side == SignalSide.LONG and dominant_side == "buy":
            signal.add_confirmation(
                "Buy-side whales absorbing sell pressure"
            )

        if side == SignalSide.SHORT and dominant_side == "sell":
            signal.add_confirmation(
                "Sell-side whales absorbing buy pressure"
            )

    def _append_source_features(
        self,
        signal: StrategySignal,
    ) -> None:
        signal.add_source_feature("whale_pressure")
        signal.add_source_feature("whale_liquidation_context")
        signal.add_source_feature("whale_cluster")
        signal.add_source_feature("whale_cluster_update")
        signal.add_source_feature("whale_cluster_exhaustion")

    # =========================================================================
    # Execution plan
    # =========================================================================

    def _build_execution_plan(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
        confidence: float,
    ) -> ExecutionPlanDraft | None:
        price = self._resolve_reference_price(context)
        if price is None or price <= 0:
            return None

        entry_type = self.config.builders.default_entry_type
        rr_ratio = self._default_rr_ratio
        stop_buffer_fraction = self._default_stop_buffer_bps / 10_000.0

        entry = EntryPlan(
            entry_type=entry_type,
            price=(
                price
                if entry_type
                in {
                    EntryType.LIMIT,
                    EntryType.STOP,
                    EntryType.PULLBACK,
                }
                else None
            ),
            confirmation_required=confidence < 0.75,
            notes=[
                "Generated by WhaleAbsorptionStrategy",
            ],
        )

        if side == SignalSide.LONG:
            stop_loss = price * (1.0 - stop_buffer_fraction)
            take_profit = price + (price - stop_loss) * rr_ratio
            invalidation_price = stop_loss
            invalidation_reason = "Bullish whale absorption invalidated"
        else:
            stop_loss = price * (1.0 + stop_buffer_fraction)
            take_profit = price - (stop_loss - price) * rr_ratio
            invalidation_price = stop_loss
            invalidation_reason = "Bearish whale absorption invalidated"

        exit_plan = ExitPlan(
            exit_types=[
                ExitType.STOP_LOSS,
                ExitType.TAKE_PROFIT,
                ExitType.INVALIDATION,
            ],
            stop_loss=stop_loss,
            take_profit_levels=[
                TargetPlan(
                    price=take_profit,
                    size_fraction=1.0,
                    rr=rr_ratio,
                    label="main_target",
                )
            ],
            partial_exit_enabled=self.config.builders.enable_partial_take_profit,
        )

        invalidation = InvalidationPlan(
            price=invalidation_price,
            reason=invalidation_reason,
            conditions=[
                "whale_absorption_signal_flip",
                "opposite_whale_pressure_confirmation",
            ],
        )

        return ExecutionPlanDraft(
            symbol=context.symbol,
            side=side,
            entry=entry,
            exit=exit_plan,
            invalidation=invalidation,
            expected_holding_seconds=self._suggest_holding_seconds(context),
            notes=[
                "Execution draft generated from whale absorption setup",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "rr_ratio": rr_ratio,
                "reference_price": price,
            },
        )

    # =========================================================================
    # Filters
    # =========================================================================

    def _run_filters(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal,
        inputs: dict[str, dict[str, Any]],
    ) -> list[FilterResult]:
        return self._run_common_filters(
            context=context,
            signal=signal,
        )