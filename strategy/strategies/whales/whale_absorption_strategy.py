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
    Futures whale absorption strategy.

    Ідея:
        Стратегія шукає ситуацію, коли великі учасники поглинають
        агресивний протилежний потік, часто на фоні liquidation pressure
        або exhaustion протилежного whale/cluster боку.

    Bullish absorption:
        - dominant whale pressure = buy;
        - whale_side/context side = buy;
        - liquidation_side / exhausted side = sell;
        - sell-side liquidation або sell-side exhaustion підтверджує виснаження;
        - buy-side whales поглинають sell pressure.

    Bearish absorption:
        - dominant whale pressure = sell;
        - whale_side/context side = sell;
        - liquidation_side / exhausted side = buy;
        - buy-side liquidation або buy-side exhaustion підтверджує виснаження;
        - sell-side whales поглинають buy pressure.

    Основні analytics.whales inputs:
        - whale_pressure;
        - whale_liquidation_context;
        - whale_cluster;
        - whale_cluster_update;
        - whale_cluster_exhaustion.

    Optional confirmations:
        - whale_activity;
        - large_trade.

    Важливо:
        - scope/freshness/futures validation виконується в WhaleStrategyBase
          через _resolve_payload();
        - клас поки лишається самодостатнім evaluator-ом;
        - пізніше filters/scoring/signal building можна буде винести в
          SignalProcessor.
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
        Основний sync evaluation-вхід.

        Повертає:
            - StrategyEvaluation(passed=True, signal=...), якщо absorption setup
              валідний;
            - StrategyEvaluation(passed=False), якщо setup відсутній або
              заблокований фільтрами.
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

            missing_required = self._missing_required_inputs(inputs)
            if missing_required:
                evaluation.reasons.append(
                    "Missing required whale analytics inputs: "
                    + ", ".join(sorted(missing_required))
                )
                return evaluation

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
                0.56,
            )
        )

    @property
    def _min_liquidation_notional(self) -> float:
        return float(
            self._metadata.get(
                "min_liquidation_notional",
                150_000.0,
            )
        )

    @property
    def _min_activity_notional(self) -> float:
        return float(
            self._metadata.get(
                "min_activity_notional",
                250_000.0,
            )
        )

    @property
    def _min_large_trade_notional(self) -> float:
        return float(
            self._metadata.get(
                "min_large_trade_notional",
                200_000.0,
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
    def _require_futures_liquidation_context(self) -> bool:
        return bool(
            self._metadata.get(
                "require_futures_liquidation_context",
                True,
            )
        )

    @property
    def _use_activity_confirmation(self) -> bool:
        return bool(
            self._metadata.get(
                "use_activity_confirmation",
                True,
            )
        )

    @property
    def _use_large_trade_confirmation(self) -> bool:
        return bool(
            self._metadata.get(
                "use_large_trade_confirmation",
                True,
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
        """
        Дістає whale analytics inputs із context.whales / context.feature_map.

        _resolve_payload() уже виконує:
            - object -> dict conversion;
            - scope validation;
            - freshness validation;
            - futures market_type validation;
            - optional dropping invalid payloads.
        """
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
            "activity": self._resolve_payload(
                context,
                names=(
                    "whale_activity",
                    "whale_activity_signal",
                    "analytics.whales.whale_activity",
                ),
            ),
            "large_trade": self._resolve_payload(
                context,
                names=(
                    "large_trade",
                    "large_trade_signal",
                    "analytics.whales.large_trade",
                ),
            ),
        }

    def _missing_required_inputs(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> set[str]:
        required = {
            "pressure",
            "liquidation_context",
        }

        if self._require_cluster_confirmation:
            required.add("cluster_or_cluster_update")

        if self._require_exhaustion_confirmation:
            required.add("cluster_exhaustion")

        missing: set[str] = set()

        if not inputs.get("pressure"):
            missing.add("whale_pressure")

        if not inputs.get("liquidation_context"):
            missing.add("whale_liquidation_context")

        if (
            "cluster_or_cluster_update" in required
            and not inputs.get("cluster")
            and not inputs.get("cluster_update")
        ):
            missing.add("whale_cluster/whale_cluster_update")

        if "cluster_exhaustion" in required and not inputs.get("cluster_exhaustion"):
            missing.add("whale_cluster_exhaustion")

        return missing

    # =========================================================================
    # Setup detection
    # =========================================================================

    def _determine_signal_side(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> SignalSide:
        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]

        dominant_side = self._side_value(
            pressure.get("dominant_side")
            or pressure.get("side")
        )

        whale_side = self._side_value(
            liq_ctx.get("whale_side")
            or liq_ctx.get("dominant_side")
            or liq_ctx.get("absorbing_side")
            or liq_ctx.get("side")
        )

        liquidation_side = self._side_value(
            liq_ctx.get("liquidation_side")
            or liq_ctx.get("liquidated_side")
            or liq_ctx.get("opposite_side")
        )

        exhausted_side = self._resolve_exhausted_side(inputs)
        cluster_side = self._resolve_cluster_side(inputs)

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
            or pressure.get("pressure_imbalance_ratio")
        )

        context_strength = self._safe_float(
            liq_ctx.get("context_strength")
            or liq_ctx.get("liquidation_context_strength")
            or liq_ctx.get("strength")
        )

        liquidation_notional = self._resolve_liquidation_notional(liq_ctx)
        cluster_score = self._resolve_cluster_score(inputs)
        exhaustion_probability = self._resolve_exhaustion_probability(inputs)

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

        if self._require_futures_liquidation_context:
            if liquidation_notional < self._min_liquidation_notional:
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

        bullish_absorption = self._is_bullish_absorption(
            dominant_side=dominant_side,
            whale_side=whale_side,
            liquidation_side=liquidation_side,
            exhausted_side=exhausted_side,
            cluster_side=cluster_side,
            exhaustion_probability=exhaustion_probability,
        )

        bearish_absorption = self._is_bearish_absorption(
            dominant_side=dominant_side,
            whale_side=whale_side,
            liquidation_side=liquidation_side,
            exhausted_side=exhausted_side,
            cluster_side=cluster_side,
            exhaustion_probability=exhaustion_probability,
        )

        if bullish_absorption:
            return SignalSide.LONG

        if bearish_absorption:
            return SignalSide.SHORT

        return SignalSide.UNKNOWN

    def _is_bullish_absorption(
        self,
        *,
        dominant_side: str,
        whale_side: str,
        liquidation_side: str,
        exhausted_side: str,
        cluster_side: str,
        exhaustion_probability: float | None,
    ) -> bool:
        if dominant_side != "buy":
            return False

        if whale_side not in {"buy", "unknown", ""}:
            return False

        if self._require_opposite_liquidation and liquidation_side != "sell":
            return False

        opposite_side_confirmed = (
            liquidation_side == "sell"
            or exhausted_side == "sell"
            or cluster_side in {"sell", "unknown", ""}
            or (
                exhaustion_probability is not None
                and exhaustion_probability >= self._min_exhaustion_probability
            )
        )

        return opposite_side_confirmed

    def _is_bearish_absorption(
        self,
        *,
        dominant_side: str,
        whale_side: str,
        liquidation_side: str,
        exhausted_side: str,
        cluster_side: str,
        exhaustion_probability: float | None,
    ) -> bool:
        if dominant_side != "sell":
            return False

        if whale_side not in {"sell", "unknown", ""}:
            return False

        if self._require_opposite_liquidation and liquidation_side != "buy":
            return False

        opposite_side_confirmed = (
            liquidation_side == "buy"
            or exhausted_side == "buy"
            or cluster_side in {"buy", "unknown", ""}
            or (
                exhaustion_probability is not None
                and exhaustion_probability >= self._min_exhaustion_probability
            )
        )

        return opposite_side_confirmed

    # =========================================================================
    # Scoring
    # =========================================================================

    def _calculate_score(
        self,
        *,
        inputs: dict[str, dict[str, Any]],
        side: SignalSide,
    ) -> float:
        if side == SignalSide.UNKNOWN:
            return 0.0

        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
            or pressure.get("pressure_imbalance_ratio"),
            default=0.0,
        ) or 0.0

        context_strength = self._safe_float(
            liq_ctx.get("context_strength")
            or liq_ctx.get("liquidation_context_strength")
            or liq_ctx.get("strength"),
            default=0.0,
        ) or 0.0

        cluster_score = self._resolve_cluster_score(inputs) or 0.0
        exhaustion_probability = self._resolve_exhaustion_probability(inputs) or 0.0
        continuation_probability = self._resolve_continuation_probability(inputs) or 0.0

        liquidation_score = self._normalize_notional(
            self._resolve_liquidation_notional(liq_ctx),
            self._min_liquidation_notional,
        )

        activity_score = self._resolve_activity_confirmation_score(inputs)
        large_trade_score = self._resolve_large_trade_confirmation_score(inputs)

        # Absorption favor:
        # - high pressure imbalance;
        # - strong liquidation context;
        # - visible opposite-side exhaustion;
        # - cluster context;
        # - liquidation notional;
        # - lower continuation probability on exhausted side.
        base_score = (
            imbalance_ratio * 0.25
            + context_strength * 0.25
            + cluster_score * 0.15
            + exhaustion_probability * 0.15
            + liquidation_score * 0.10
            + (1.0 - continuation_probability) * 0.05
            + activity_score * 0.03
            + large_trade_score * 0.02
        )

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
        if side == SignalSide.UNKNOWN:
            return 0.0

        pressure = inputs["pressure"]
        liq_ctx = inputs["liquidation_context"]

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
            or pressure.get("pressure_imbalance_ratio"),
            default=0.0,
        ) or 0.0

        context_strength = self._safe_float(
            liq_ctx.get("context_strength")
            or liq_ctx.get("liquidation_context_strength")
            or liq_ctx.get("strength"),
            default=0.0,
        ) or 0.0

        cluster_score = self._resolve_cluster_score(inputs) or 0.0
        exhaustion_probability = self._resolve_exhaustion_probability(inputs) or 0.0
        liquidation_score = self._normalize_notional(
            self._resolve_liquidation_notional(liq_ctx),
            self._min_liquidation_notional,
        )

        activity_score = self._resolve_activity_confirmation_score(inputs)
        large_trade_score = self._resolve_large_trade_confirmation_score(inputs)

        confidence = (
            imbalance_ratio * 0.27
            + context_strength * 0.27
            + cluster_score * 0.18
            + exhaustion_probability * 0.16
            + liquidation_score * 0.07
            + activity_score * 0.03
            + large_trade_score * 0.02
        )

        return self._clamp(
            confidence,
            0.0,
            1.0,
        )

    def _resolve_activity_confirmation_score(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float:
        if not self._use_activity_confirmation:
            return 0.0

        activity = inputs.get("activity") or {}
        if not activity:
            return 0.0

        total_notional = self._safe_float(
            activity.get("total_notional")
            or activity.get("notional")
            or activity.get("volume_notional"),
            default=0.0,
        ) or 0.0

        trade_count = self._safe_int(
            activity.get("trade_count")
            or activity.get("large_trade_count")
            or activity.get("count"),
            default=0,
        )

        notional_score = self._normalize_notional(
            total_notional,
            self._min_activity_notional,
        )

        count_score = self._clamp(
            trade_count / 3.0,
            0.0,
            1.0,
        )

        return self._clamp(
            notional_score * 0.75 + count_score * 0.25,
            0.0,
            1.0,
        )

    def _resolve_large_trade_confirmation_score(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float:
        if not self._use_large_trade_confirmation:
            return 0.0

        large_trade = inputs.get("large_trade") or {}
        if not large_trade:
            return 0.0

        notional = self._safe_float(
            large_trade.get("notional")
            or large_trade.get("trade_notional")
            or large_trade.get("total_notional"),
            default=0.0,
        ) or 0.0

        zscore = self._safe_float(
            large_trade.get("zscore")
            or large_trade.get("z_score")
            or large_trade.get("notional_zscore"),
            default=0.0,
        ) or 0.0

        notional_score = self._normalize_notional(
            notional,
            self._min_large_trade_notional,
        )

        zscore_score = self._clamp(
            zscore / 5.0,
            0.0,
            1.0,
        )

        return self._clamp(
            notional_score * 0.7 + zscore_score * 0.3,
            0.0,
            1.0,
        )

    def _normalize_notional(
        self,
        value: float,
        threshold: float,
    ) -> float:
        return self._clamp(
            value / max(threshold, 1.0),
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
                "market_scope": self._build_market_scope_metadata(context, inputs),
                "inputs_present": {
                    key: bool(value)
                    for key, value in inputs.items()
                },
                "inputs_used": {
                    "pressure": True,
                    "liquidation_context": True,
                    "cluster": bool(inputs.get("cluster")),
                    "cluster_update": bool(inputs.get("cluster_update")),
                    "cluster_exhaustion": bool(inputs.get("cluster_exhaustion")),
                    "activity": bool(inputs.get("activity")),
                    "large_trade": bool(inputs.get("large_trade")),
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
                inputs=inputs,
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

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
            or pressure.get("pressure_imbalance_ratio"),
            default=0.0,
        ) or 0.0

        context_strength = self._safe_float(
            liq_ctx.get("context_strength")
            or liq_ctx.get("liquidation_context_strength")
            or liq_ctx.get("strength"),
            default=0.0,
        ) or 0.0

        liquidation_notional = self._resolve_liquidation_notional(liq_ctx)
        cluster_score = self._resolve_cluster_score(inputs) or 0.0
        exhaustion_probability = self._resolve_exhaustion_probability(inputs) or 0.0
        continuation_probability = self._resolve_continuation_probability(inputs) or 0.0

        signal.add_reason(
            f"Whale absorption detected on {side.value}"
        )
        signal.add_reason(
            f"Pressure imbalance ratio={imbalance_ratio:.4f}"
        )
        signal.add_reason(
            f"Liquidation context strength={context_strength:.4f}"
        )
        signal.add_reason(
            f"Liquidation notional={liquidation_notional:.2f}"
        )

        if cluster_score > 0:
            signal.add_reason(
                f"Cluster score={cluster_score:.4f}"
            )

        if exhaustion_probability > 0:
            signal.add_reason(
                f"Exhaustion probability={exhaustion_probability:.4f}"
            )

        if continuation_probability > 0:
            signal.add_reason(
                f"Continuation probability={continuation_probability:.4f}"
            )

        activity_score = self._resolve_activity_confirmation_score(inputs)
        if activity_score > 0:
            signal.add_reason(
                f"Whale activity confirmation score={activity_score:.4f}"
            )

        large_trade_score = self._resolve_large_trade_confirmation_score(inputs)
        if large_trade_score > 0:
            signal.add_reason(
                f"Large trade confirmation score={large_trade_score:.4f}"
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

        dominant_side = self._side_value(
            pressure.get("dominant_side")
            or pressure.get("side")
        )

        whale_side = self._side_value(
            liq_ctx.get("whale_side")
            or liq_ctx.get("dominant_side")
            or liq_ctx.get("absorbing_side")
            or liq_ctx.get("side")
        )

        liquidation_side = self._side_value(
            liq_ctx.get("liquidation_side")
            or liq_ctx.get("liquidated_side")
            or liq_ctx.get("opposite_side")
        )

        cluster_side = self._resolve_cluster_side(inputs)
        exhausted_side = self._resolve_exhausted_side(inputs)

        if dominant_side:
            signal.add_confirmation(
                f"Dominant whale pressure={dominant_side}"
            )

        if whale_side:
            signal.add_confirmation(
                f"Whale absorbing side={whale_side}"
            )

        if liquidation_side:
            signal.add_confirmation(
                f"Liquidation side={liquidation_side}"
            )

        if cluster_side:
            signal.add_confirmation(
                f"Cluster side={cluster_side}"
            )

        if exhausted_side:
            signal.add_confirmation(
                f"Exhausted side={exhausted_side}"
            )

        if side == SignalSide.LONG:
            signal.add_confirmation(
                "Buy-side whales absorbing sell pressure"
            )

        elif side == SignalSide.SHORT:
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
        signal.add_source_feature("whale_activity")
        signal.add_source_feature("large_trade")

    # =========================================================================
    # Execution plan
    # =========================================================================

    def _build_execution_plan(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
        confidence: float,
        inputs: dict[str, dict[str, Any]],
    ) -> ExecutionPlanDraft | None:
        price = self._resolve_reference_price(context)
        if price is None or price <= 0:
            return None

        entry_type = self._resolve_entry_type()
        rr_ratio = self._default_rr_ratio
        stop_buffer_fraction = self._resolve_stop_buffer_fraction(inputs)

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
                "liquidation_context_strength_drop",
                "cluster_exhaustion_invalidated",
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
                "Execution draft generated from futures whale absorption setup",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "rr_ratio": rr_ratio,
                "reference_price": price,
                "stop_buffer_fraction": stop_buffer_fraction,
                "stop_buffer_bps": stop_buffer_fraction * 10_000.0,
                "setup": "whale_absorption",
            },
        )

    def _resolve_entry_type(self) -> EntryType:
        configured = self.config.builders.default_entry_type

        if configured in {
            EntryType.MARKET,
            EntryType.LIMIT,
            EntryType.PULLBACK,
            EntryType.STOP,
        }:
            return configured

        return EntryType.MARKET

    def _resolve_stop_buffer_fraction(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float:
        base_buffer_bps = self._default_stop_buffer_bps

        exhaustion_probability = self._resolve_exhaustion_probability(inputs) or 0.0
        context_strength = self._safe_float(
            inputs["liquidation_context"].get("context_strength")
            or inputs["liquidation_context"].get("liquidation_context_strength")
            or inputs["liquidation_context"].get("strength"),
            default=0.0,
        ) or 0.0

        # Stronger absorption can use a slightly tighter stop,
        # weaker/noisier context gets more room.
        tightening_factor = self._clamp(
            1.0 - (exhaustion_probability * 0.15 + context_strength * 0.10),
            0.75,
            1.15,
        )

        buffer_bps = base_buffer_bps * tightening_factor

        return max(
            buffer_bps / 10_000.0,
            0.0001,
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
        results = self._run_common_filters(
            context=context,
            signal=signal,
        )

        results.extend(
            self._run_absorption_specific_filters(
                signal=signal,
                inputs=inputs,
            )
        )

        return results

    def _run_absorption_specific_filters(
        self,
        *,
        signal: StrategySignal,
        inputs: dict[str, dict[str, Any]],
    ) -> list[FilterResult]:
        results: list[FilterResult] = []

        liq_ctx = inputs["liquidation_context"]
        liquidation_notional = self._resolve_liquidation_notional(liq_ctx)

        if self._require_futures_liquidation_context:
            if liquidation_notional < self._min_liquidation_notional:
                results.append(
                    FilterResult(
                        name="whale_absorption_liquidation_filter",
                        decision=FilterDecision.BLOCK,
                        reason=(
                            "Liquidation notional below threshold: "
                            f"{liquidation_notional:.2f} < {self._min_liquidation_notional:.2f}"
                        ),
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="whale_absorption_liquidation_filter",
                        decision=FilterDecision.PASS,
                        reason=(
                            "Liquidation context confirmed: "
                            f"{liquidation_notional:.2f}"
                        ),
                    )
                )

        if self._require_exhaustion_confirmation:
            exhaustion_probability = self._resolve_exhaustion_probability(inputs)
            if (
                exhaustion_probability is None
                or exhaustion_probability < self._min_exhaustion_probability
            ):
                results.append(
                    FilterResult(
                        name="whale_absorption_exhaustion_filter",
                        decision=FilterDecision.BLOCK,
                        reason=(
                            "Exhaustion probability below threshold: "
                            f"{(exhaustion_probability or 0.0):.4f} "
                            f"< {self._min_exhaustion_probability:.4f}"
                        ),
                    )
                )
            else:
                results.append(
                    FilterResult(
                        name="whale_absorption_exhaustion_filter",
                        decision=FilterDecision.PASS,
                        reason=(
                            "Exhaustion confirmed: "
                            f"{exhaustion_probability:.4f}"
                        ),
                    )
                )

        return results

    # =========================================================================
    # Analytics payload helpers
    # =========================================================================

    def _resolve_cluster_score(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float | None:
        cluster = inputs.get("cluster") or {}
        cluster_update = inputs.get("cluster_update") or {}
        cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

        return self._first_float(
            cluster.get("cluster_score"),
            cluster.get("score"),
            cluster_update.get("cluster_score"),
            cluster_update.get("score"),
            cluster_exhaustion.get("cluster_score"),
            cluster_exhaustion.get("score"),
        )

    def _resolve_exhaustion_probability(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float | None:
        cluster = inputs.get("cluster") or {}
        cluster_update = inputs.get("cluster_update") or {}
        cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

        return self._first_float(
            cluster_exhaustion.get("exhaustion_probability"),
            cluster_exhaustion.get("probability"),
            cluster_update.get("exhaustion_probability"),
            cluster.get("exhaustion_probability"),
        )

    def _resolve_continuation_probability(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> float | None:
        cluster = inputs.get("cluster") or {}
        cluster_update = inputs.get("cluster_update") or {}

        return self._first_float(
            cluster.get("continuation_probability"),
            cluster_update.get("continuation_probability"),
        )

    def _resolve_cluster_side(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> str:
        cluster = inputs.get("cluster") or {}
        cluster_update = inputs.get("cluster_update") or {}
        cluster_exhaustion = inputs.get("cluster_exhaustion") or {}

        return self._side_value(
            cluster.get("cluster_side")
            or cluster.get("side")
            or cluster_update.get("cluster_side")
            or cluster_update.get("side")
            or cluster_exhaustion.get("cluster_side")
            or cluster_exhaustion.get("side")
        )

    def _resolve_exhausted_side(
        self,
        inputs: dict[str, dict[str, Any]],
    ) -> str:
        cluster_exhaustion = inputs.get("cluster_exhaustion") or {}
        cluster_update = inputs.get("cluster_update") or {}
        cluster = inputs.get("cluster") or {}

        return self._side_value(
            cluster_exhaustion.get("exhausted_side")
            or cluster_exhaustion.get("cluster_side")
            or cluster_update.get("exhausted_side")
            or cluster.get("exhausted_side")
        )

    def _resolve_liquidation_notional(
        self,
        liq_ctx: dict[str, Any],
    ) -> float:
        return (
            self._first_float(
                liq_ctx.get("liquidation_notional"),
                liq_ctx.get("total_liquidation_notional"),
                liq_ctx.get("notional"),
                liq_ctx.get("total_notional"),
                liq_ctx.get("opposite_liquidation_notional"),
            )
            or 0.0
        )

    def _build_market_scope_metadata(
        self,
        context: SignalContext,
        inputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        pressure = inputs.get("pressure") or {}
        liq_ctx = inputs.get("liquidation_context") or {}
        scope = pressure.get("scope") if isinstance(pressure.get("scope"), dict) else {}

        return {
            "symbol": context.symbol,
            "timeframe": str(context.timeframe),
            "exchange": (
                pressure.get("exchange")
                or liq_ctx.get("exchange")
                or scope.get("exchange")
            ),
            "market_type": (
                pressure.get("market_type")
                or liq_ctx.get("market_type")
                or scope.get("market_type")
            ),
        }

    def _side_value(
        self,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        normalized = str(value).strip().lower()

        if normalized in {"buy", "bid", "long", "bull", "bullish", "b"}:
            return "buy"

        if normalized in {"sell", "ask", "short", "bear", "bearish", "s"}:
            return "sell"

        if normalized in {"unknown", "none", "neutral"}:
            return "unknown"

        return normalized

    def _first_float(
        self,
        *values: Any,
    ) -> float | None:
        for value in values:
            resolved = self._safe_float(value, default=None)
            if resolved is not None:
                return resolved
        return None