# strategy/strategies/whales/whale_breakout_strategy.py

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


class WhaleBreakoutStrategy(WhaleStrategyBase):
    """
    Whale breakout strategy.

    Ідея:
        Стратегія шукає продовження руху, коли whale-активність
        не абсорбує протилежний потік, а штовхає ринок у напрямку
        breakout / continuation.

    Bullish breakout:
        - buy whale pressure домінує;
        - whale activity по buy-side підтверджує агресію;
        - cluster side = buy;
        - continuation_probability достатньо висока;
        - exhaustion_probability не надто висока.

    Bearish breakout:
        - sell whale pressure домінує;
        - whale activity по sell-side підтверджує агресію;
        - cluster side = sell;
        - continuation_probability достатньо висока;
        - exhaustion_probability не надто висока.

    Джерела даних:
        - context.whales
        - context.feature_map

    Очікувані whale-сутності:
        - whale_activity
        - whale_pressure
        - whale_cluster
        - whale_cluster_update
        - whale_cluster_exhaustion
    """

    DEFAULT_STRATEGY_NAME = "whale_breakout_strategy"

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
            "whale_activity",
            "whale_pressure",
            "whale_cluster",
            "whale_cluster_update",
            "whale_cluster_exhaustion",
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        """
        Основний sync-вхід у breakout-стратегію.

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
                evaluation.reasons.append("No whale breakout setup detected")
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

            if score < self._min_breakout_score:
                evaluation.reasons.append(
                    f"Score below threshold: "
                    f"{score:.4f} < {self._min_breakout_score:.4f}"
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
    def _min_activity_notional(self) -> float:
        return float(
            self._metadata.get(
                "min_activity_notional",
                300_000.0,
            )
        )

    @property
    def _min_activity_trade_count(self) -> int:
        return int(
            self._metadata.get(
                "min_activity_trade_count",
                3,
            )
        )

    @property
    def _min_pressure_imbalance_ratio(self) -> float:
        return float(
            self._metadata.get(
                "min_pressure_imbalance_ratio",
                0.64,
            )
        )

    @property
    def _min_cluster_score(self) -> float:
        return float(
            self._metadata.get(
                "min_cluster_score",
                0.55,
            )
        )

    @property
    def _min_continuation_probability(self) -> float:
        return float(
            self._metadata.get(
                "min_continuation_probability",
                0.60,
            )
        )

    @property
    def _max_exhaustion_probability(self) -> float:
        return float(
            self._metadata.get(
                "max_exhaustion_probability",
                0.55,
            )
        )

    @property
    def _min_breakout_score(self) -> float:
        return float(
            self._metadata.get(
                "min_breakout_score",
                0.58,
            )
        )

    @property
    def _require_activity_confirmation(self) -> bool:
        return bool(
            self._metadata.get(
                "require_activity_confirmation",
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
    def _default_stop_buffer_bps(self) -> float:
        return float(
            self._metadata.get(
                "default_stop_buffer_bps",
                20.0,
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
            "activity": self._resolve_payload(
                context,
                names=(
                    "whale_activity",
                    "whale_activity_signal",
                    "analytics.whales.whale_activity",
                ),
            ),
            "pressure": self._resolve_payload(
                context,
                names=(
                    "whale_pressure",
                    "whale_pressure_signal",
                    "analytics.whales.whale_pressure",
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
        activity = inputs["activity"]
        pressure = inputs["pressure"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        activity_side = str(
            activity.get("side", "")
        ).lower()

        activity_trade_count = self._safe_int(
            activity.get("trade_count"),
            default=0,
        )

        activity_total_notional = self._safe_float(
            activity.get("total_notional"),
            default=0.0,
        )

        dominant_side = str(
            pressure.get("dominant_side", "")
        ).lower()

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio")
        )

        cluster_side = str(
            cluster.get("cluster_side")
            or cluster_update.get("cluster_side")
            or cluster_exhaustion.get("cluster_side")
            or ""
        ).lower()

        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score")
        )

        continuation_probability = self._safe_float(
            cluster.get("continuation_probability")
            or cluster_update.get("continuation_probability")
        )

        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability")
        )

        if (
            imbalance_ratio is None
            or imbalance_ratio < self._min_pressure_imbalance_ratio
        ):
            return SignalSide.UNKNOWN

        if self._require_activity_confirmation:
            if activity_trade_count < self._min_activity_trade_count:
                return SignalSide.UNKNOWN

            if (
                activity_total_notional is None
                or activity_total_notional < self._min_activity_notional
            ):
                return SignalSide.UNKNOWN

        if self._require_cluster_confirmation:
            if (
                cluster_score is None
                or cluster_score < self._min_cluster_score
            ):
                return SignalSide.UNKNOWN

            if (
                continuation_probability is None
                or continuation_probability < self._min_continuation_probability
            ):
                return SignalSide.UNKNOWN

        if (
            exhaustion_probability is not None
            and exhaustion_probability > self._max_exhaustion_probability
        ):
            return SignalSide.UNKNOWN

        bullish_breakout = (
            dominant_side == "buy"
            and activity_side in {"buy", ""}
            and cluster_side in {"buy", ""}
        )

        bearish_breakout = (
            dominant_side == "sell"
            and activity_side in {"sell", ""}
            and cluster_side in {"sell", ""}
        )

        if bullish_breakout:
            return SignalSide.LONG

        if bearish_breakout:
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
        activity = inputs["activity"]
        pressure = inputs["pressure"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        activity_total_notional = self._safe_float(
            activity.get("total_notional"),
            default=0.0,
        )
        activity_trade_count = self._safe_int(
            activity.get("trade_count"),
            default=0,
        )

        activity_score = self._normalize_activity(
            total_notional=activity_total_notional or 0.0,
            trade_count=activity_trade_count,
        )

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score"),
            default=0.0,
        )
        continuation_probability = self._safe_float(
            cluster.get("continuation_probability")
            or cluster_update.get("continuation_probability"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )

        base_score = (
            (activity_score or 0.0) * 0.25
            + (imbalance_ratio or 0.0) * 0.25
            + (cluster_score or 0.0) * 0.20
            + (continuation_probability or 0.0) * 0.25
            + (1.0 - (exhaustion_probability or 0.0)) * 0.05
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
        activity = inputs["activity"]
        pressure = inputs["pressure"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        activity_total_notional = self._safe_float(
            activity.get("total_notional"),
            default=0.0,
        )
        activity_trade_count = self._safe_int(
            activity.get("trade_count"),
            default=0,
        )

        activity_score = self._normalize_activity(
            total_notional=activity_total_notional or 0.0,
            trade_count=activity_trade_count,
        )

        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        continuation_probability = self._safe_float(
            cluster.get("continuation_probability")
            or cluster_update.get("continuation_probability"),
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
            (activity_score or 0.0) * 0.25
            + (imbalance_ratio or 0.0) * 0.25
            + (continuation_probability or 0.0) * 0.30
            + (cluster_score or 0.0) * 0.10
            + (1.0 - (exhaustion_probability or 0.0)) * 0.10
        )

        if side == SignalSide.UNKNOWN:
            return 0.0

        return self._clamp(
            confidence,
            0.0,
            1.0,
        )

    def _normalize_activity(
        self,
        *,
        total_notional: float,
        trade_count: int,
    ) -> float:
        notional_part = self._clamp(
            total_notional / max(self._min_activity_notional, 1.0),
            0.0,
            2.0,
        )

        count_part = self._clamp(
            trade_count / max(self._min_activity_trade_count, 1),
            0.0,
            2.0,
        )

        return self._clamp(
            (notional_part * 0.7 + count_part * 0.3) / 2.0,
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
            setup_type=SetupType.BREAKOUT,
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
                "strategy_type": "whale_breakout",
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
        activity = inputs["activity"]
        pressure = inputs["pressure"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        trade_count = self._safe_int(
            activity.get("trade_count"),
            default=0,
        )
        total_notional = self._safe_float(
            activity.get("total_notional"),
            default=0.0,
        )
        imbalance_ratio = self._safe_float(
            pressure.get("imbalance_ratio"),
            default=0.0,
        )
        cluster_score = self._safe_float(
            cluster.get("cluster_score")
            or cluster_update.get("cluster_score"),
            default=0.0,
        )
        continuation_probability = self._safe_float(
            cluster.get("continuation_probability")
            or cluster_update.get("continuation_probability"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )

        signal.add_reason(
            f"Whale breakout detected on {side.value}"
        )
        signal.add_reason(
            f"Whale activity trades={trade_count}"
        )
        signal.add_reason(
            f"Whale activity notional={(total_notional or 0.0):.2f}"
        )
        signal.add_reason(
            f"Pressure imbalance ratio={(imbalance_ratio or 0.0):.4f}"
        )
        signal.add_reason(
            f"Cluster score={(cluster_score or 0.0):.4f}"
        )
        signal.add_reason(
            f"Continuation probability={(continuation_probability or 0.0):.4f}"
        )

        if exhaustion_probability is not None and exhaustion_probability > 0:
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
        activity = inputs["activity"]
        pressure = inputs["pressure"]
        cluster = inputs["cluster"]
        cluster_update = inputs["cluster_update"]
        cluster_exhaustion = inputs["cluster_exhaustion"]

        activity_side = str(
            activity.get("side", "")
        ).lower()

        dominant_side = str(
            pressure.get("dominant_side", "")
        ).lower()

        cluster_side = str(
            cluster.get("cluster_side")
            or cluster_update.get("cluster_side")
            or cluster_exhaustion.get("cluster_side")
            or ""
        ).lower()

        if activity_side:
            signal.add_confirmation(
                f"Whale activity side={activity_side}"
            )

        if dominant_side:
            signal.add_confirmation(
                f"Dominant whale pressure={dominant_side}"
            )

        if cluster_side:
            signal.add_confirmation(
                f"Cluster side={cluster_side}"
            )

        if side == SignalSide.LONG:
            signal.add_confirmation(
                "Buy-side whales supporting bullish breakout"
            )

        elif side == SignalSide.SHORT:
            signal.add_confirmation(
                "Sell-side whales supporting bearish breakout"
            )

    def _append_source_features(
        self,
        signal: StrategySignal,
    ) -> None:
        signal.add_source_feature("whale_activity")
        signal.add_source_feature("whale_pressure")
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

        entry_type = self._resolve_entry_type()
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
            confirmation_required=confidence < 0.72,
            notes=[
                "Generated by WhaleBreakoutStrategy",
            ],
        )

        if side == SignalSide.LONG:
            stop_loss = price * (1.0 - stop_buffer_fraction)
            take_profit = price + (price - stop_loss) * rr_ratio
            invalidation_price = stop_loss
            invalidation_reason = "Bullish whale breakout invalidated"
        else:
            stop_loss = price * (1.0 + stop_buffer_fraction)
            take_profit = price - (stop_loss - price) * rr_ratio
            invalidation_price = stop_loss
            invalidation_reason = "Bearish whale breakout invalidated"

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
                "whale_breakout_signal_flip",
                "continuation_probability_drop",
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
                "Execution draft generated from whale breakout setup",
            ],
            metadata={
                "strategy_name": self.strategy_name,
                "rr_ratio": rr_ratio,
                "reference_price": price,
            },
        )

    def _resolve_entry_type(self) -> EntryType:
        configured = self.config.builders.default_entry_type

        if configured in {
            EntryType.MARKET,
            EntryType.STOP,
            EntryType.BREAKOUT_CONFIRMATION,
            EntryType.LIMIT,
            EntryType.PULLBACK,
        }:
            return configured

        return EntryType.MARKET

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