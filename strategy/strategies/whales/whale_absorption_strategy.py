from __future__ import annotations

from typing import Any, Mapping

from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
from strategy.enums import (
    ConfidenceGrade,
    EntryType,
    ExitType,
    FilterDecision,
    MarketRegime,
    SetupType,
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStrength,
    StrategyCategory,
    TriggerType,
)
from strategy.exceptions import StrategyEvaluationError
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


class WhaleAbsorptionStrategy(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Whale absorption strategy.

    Ідея:
        стратегія шукає ситуації, коли великі учасники (whales)
        абсорбують агресивний потік з протилежного боку ринку.

    Базовий bullish сценарій:
        - домінує whale buy pressure
        - ліквідації / агресія з боку sell
        - є ознаки exhaustion у sell-side cluster
        - є контекст для реверсу / поглинання

    Базовий bearish сценарій:
        - домінує whale sell pressure
        - ліквідації / агресія з боку buy
        - є ознаки exhaustion у buy-side cluster
        - є контекст для реверсу / поглинання

    Стратегія читає whale-дані з:
        - context.whales
        - context.feature_map

    Очікувані whale-сутності в контексті:
        - whale_pressure
        - whale_liquidation_context
        - whale_cluster
        - whale_cluster_update
        - whale_cluster_exhaustion

    Зауваження:
        ця реалізація вже готова до використання як baseline,
        але конкретні пороги можна підкручувати через
        StrategyDefinitionConfig.metadata.
    """

    DEFAULT_STRATEGY_NAME = "whale_absorption_strategy"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
        logger: Any | None = None,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=logger,
        )
        self.strategy_name = strategy_name
        self.validate_config()

    @property
    def name(self) -> str:
        return self.strategy_name

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.WHALES

    @property
    def priority(self) -> int:
        definition = self._strategy_definition
        if definition is None:
            return 100
        return definition.priority

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

    @property
    def _strategy_definition(self):
        return self.config.get_strategy(self.strategy_name)

    @property
    def _runtime_config(self):
        definition = self._strategy_definition
        if definition is not None:
            return definition.runtime
        return self.config.runtime

    @property
    def _metadata(self) -> dict[str, Any]:
        definition = self._strategy_definition
        if definition is None:
            return {}
        return dict(definition.metadata)

    def evaluate(self, context: SignalContext) -> StrategyEvaluation:
        """
        Основний вхід у стратегію.

        Повертає StrategyEvaluation:
            - signal, якщо сетап валідний
            - passed=False, якщо сетап відсутній або не пройшов фільтри
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

            score = self._calculate_score(inputs=inputs, side=signal_side)
            confidence = self._calculate_confidence(inputs=inputs, side=signal_side)

            evaluation.score = score
            evaluation.confidence = confidence

            if score < self._min_absorption_score:
                evaluation.reasons.append(
                    f"Score below threshold: {score:.4f} < {self._min_absorption_score:.4f}"
                )
                return evaluation

            if confidence < self._runtime_config.min_confidence:
                evaluation.reasons.append(
                    f"Confidence below threshold: {confidence:.4f} < {self._runtime_config.min_confidence:.4f}"
                )
                return evaluation

            signal = self._build_signal(
                context=context,
                inputs=inputs,
                side=signal_side,
                score=score,
                confidence=confidence,
            )

            filter_results = self._run_filters(context=context, signal=signal, inputs=inputs)
            for result in filter_results:
                signal.add_filter_result(result)

            if not signal.passed_filters:
                evaluation.reasons.extend(
                    [f"{result.name}: {result.reason}" for result in filter_results if result.blocked]
                )
                evaluation.signal = signal
                return evaluation

            if signal.score < self._runtime_config.min_score:
                evaluation.reasons.append(
                    f"Signal score below runtime threshold: {signal.score:.4f} < {self._runtime_config.min_score:.4f}"
                )
                evaluation.signal = signal
                return evaluation

            evaluation.signal = signal
            evaluation.passed = True
            evaluation.reasons.extend(signal.reasons)
            return evaluation

        except Exception as exc:
            raise StrategyEvaluationError(
                f"{self.strategy_name} failed for symbol={context.symbol}: {exc}"
            ) from exc

    # =========================================================================
    # Thresholds / tunables
    # =========================================================================

    @property
    def _min_pressure_imbalance_ratio(self) -> float:
        return float(self._metadata.get("min_pressure_imbalance_ratio", 0.62))

    @property
    def _min_context_strength(self) -> float:
        return float(self._metadata.get("min_context_strength", 0.55))

    @property
    def _min_cluster_score(self) -> float:
        return float(self._metadata.get("min_cluster_score", 0.45))

    @property
    def _min_exhaustion_probability(self) -> float:
        return float(self._metadata.get("min_exhaustion_probability", 0.55))

    @property
    def _min_absorption_score(self) -> float:
        return float(self._metadata.get("min_absorption_score", 0.55))

    @property
    def _require_opposite_liquidation(self) -> bool:
        return bool(self._metadata.get("require_opposite_liquidation", True))

    @property
    def _require_cluster_confirmation(self) -> bool:
        return bool(self._metadata.get("require_cluster_confirmation", True))

    @property
    def _require_exhaustion_confirmation(self) -> bool:
        return bool(self._metadata.get("require_exhaustion_confirmation", False))

    @property
    def _default_stop_buffer_bps(self) -> float:
        return float(self._metadata.get("default_stop_buffer_bps", 25.0))

    @property
    def _default_rr_ratio(self) -> float:
        return float(self._metadata.get("default_rr_ratio", self.config.builders.default_rr_ratio))

    # =========================================================================
    # Core extraction
    # =========================================================================

    def _extract_inputs(self, context: SignalContext) -> dict[str, dict[str, Any]]:
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

    def _resolve_payload(
        self,
        context: SignalContext,
        *,
        names: tuple[str, ...],
    ) -> dict[str, Any]:
        for name in names:
            value = context.whales.get(name)
            resolved = self._object_to_dict(value)
            if resolved:
                return resolved

        for name in names:
            feature_value = context.get_feature(name)
            resolved = self._object_to_dict(feature_value)
            if resolved:
                return resolved

            snapshot = context.get_feature_snapshot(name)
            if snapshot is not None:
                resolved = self._object_to_dict(snapshot.value)
                if resolved:
                    return resolved

        return {}

    def _object_to_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_event") and callable(value.to_event):
            try:
                result = value.to_event()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                return {}
        return {}

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

        dominant_side = str(pressure.get("dominant_side", "")).lower()
        whale_side = str(liq_ctx.get("whale_side", "")).lower()
        liquidation_side = str(liq_ctx.get("liquidation_side", "")).lower()

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
        imbalance_ratio = self._safe_float(pressure.get("imbalance_ratio"))
        context_strength = self._safe_float(liq_ctx.get("context_strength"))

        if imbalance_ratio is None or imbalance_ratio < self._min_pressure_imbalance_ratio:
            return SignalSide.UNKNOWN

        if context_strength is None or context_strength < self._min_context_strength:
            return SignalSide.UNKNOWN

        if self._require_cluster_confirmation:
            if cluster_score is None or cluster_score < self._min_cluster_score:
                return SignalSide.UNKNOWN

        if self._require_exhaustion_confirmation:
            if exhaustion_probability is None or exhaustion_probability < self._min_exhaustion_probability:
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

        imbalance_ratio = self._safe_float(pressure.get("imbalance_ratio"), default=0.0)
        context_strength = self._safe_float(liq_ctx.get("context_strength"), default=0.0)
        cluster_score = self._safe_float(
            cluster.get("cluster_score") or cluster_update.get("cluster_score"),
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

        return self._clamp(base_score, 0.0, 1.0)

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

        imbalance_ratio = self._safe_float(pressure.get("imbalance_ratio"), default=0.0)
        context_strength = self._safe_float(liq_ctx.get("context_strength"), default=0.0)
        cluster_score = self._safe_float(
            cluster.get("cluster_score") or cluster_update.get("cluster_score"),
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

        return self._clamp(confidence, 0.0, 1.0)

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
            strength=self._resolve_strength(score, confidence),
            confidence_grade=self._resolve_confidence_grade(confidence),
            trigger_type=TriggerType.PRIMARY,
            origin=SignalOrigin.SINGLE_STRATEGY,
            priority=self._map_priority(self.priority),
            regime=self._resolve_regime(context),
            metadata={
                "strategy_type": "whale_absorption",
                "inputs_present": {
                    key: bool(value) for key, value in inputs.items()
                },
            },
        )

        self._append_reasons(signal=signal, inputs=inputs, side=side)
        self._append_confirmations(signal=signal, inputs=inputs, side=side)
        self._append_source_features(signal=signal)

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

        imbalance_ratio = self._safe_float(pressure.get("imbalance_ratio"), default=0.0)
        context_strength = self._safe_float(liq_ctx.get("context_strength"), default=0.0)
        cluster_score = self._safe_float(
            cluster.get("cluster_score") or cluster_update.get("cluster_score"),
            default=0.0,
        )
        exhaustion_probability = self._safe_float(
            cluster_exhaustion.get("exhaustion_probability")
            or cluster_update.get("exhaustion_probability")
            or cluster.get("exhaustion_probability"),
            default=0.0,
        )

        signal.add_reason(f"Whale absorption detected on {side.value}")
        signal.add_reason(f"Pressure imbalance ratio={imbalance_ratio:.4f}")
        signal.add_reason(f"Liquidation context strength={context_strength:.4f}")

        if cluster_score > 0:
            signal.add_reason(f"Cluster score={cluster_score:.4f}")
        if exhaustion_probability > 0:
            signal.add_reason(f"Exhaustion probability={exhaustion_probability:.4f}")

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

        dominant_side = str(pressure.get("dominant_side", "")).lower()
        whale_side = str(liq_ctx.get("whale_side", "")).lower()
        liquidation_side = str(liq_ctx.get("liquidation_side", "")).lower()
        cluster_side = str(
            cluster.get("cluster_side")
            or cluster_update.get("cluster_side")
            or cluster_exhaustion.get("cluster_side")
            or ""
        ).lower()

        if dominant_side:
            signal.add_confirmation(f"Dominant whale pressure={dominant_side}")
        if whale_side:
            signal.add_confirmation(f"Whale side={whale_side}")
        if liquidation_side:
            signal.add_confirmation(f"Liquidation side={liquidation_side}")
        if cluster_side:
            signal.add_confirmation(f"Cluster side={cluster_side}")

        if side == SignalSide.LONG and dominant_side == "buy":
            signal.add_confirmation("Buy-side whales absorbing sell pressure")
        if side == SignalSide.SHORT and dominant_side == "sell":
            signal.add_confirmation("Sell-side whales absorbing buy pressure")

    def _append_source_features(self, signal: StrategySignal) -> None:
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
            price=price if entry_type in {EntryType.LIMIT, EntryType.STOP, EntryType.PULLBACK} else None,
            confirmation_required=confidence < 0.75,
            notes=["Generated by WhaleAbsorptionStrategy"],
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
            exit_types=[ExitType.STOP_LOSS, ExitType.TAKE_PROFIT, ExitType.INVALIDATION],
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

    def _resolve_reference_price(self, context: SignalContext) -> float | None:
        if context.price is None:
            return None
        if context.price.mid_price is not None and context.price.mid_price > 0:
            return context.price.mid_price
        if context.price.last_price is not None and context.price.last_price > 0:
            return context.price.last_price
        if context.price.mark_price is not None and context.price.mark_price > 0:
            return context.price.mark_price
        return None

    def _suggest_holding_seconds(self, context: SignalContext) -> int:
        mapping = {
            "1s": 60,
            "5s": 180,
            "15s": 300,
            "1m": 900,
            "3m": 1800,
            "5m": 3600,
            "15m": 4 * 3600,
            "30m": 6 * 3600,
            "1h": 12 * 3600,
            "4h": 24 * 3600,
            "1d": 3 * 24 * 3600,
        }
        return mapping.get(str(context.timeframe), 1800)

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
        results: list[FilterResult] = []

        results.extend(self._run_regime_filter(context, signal))
        results.extend(self._run_spread_filter(context))
        results.extend(self._run_liquidity_filter(context))
        results.extend(self._run_volatility_filter(context))

        return results

    def _run_regime_filter(
        self,
        context: SignalContext,
        signal: StrategySignal,
    ) -> list[FilterResult]:
        if not self.config.filters.enable_regime_filter:
            return []

        allowed_regimes = set(self._runtime_config.allowed_regimes)
        regime = self._resolve_regime(context)

        if not allowed_regimes:
            return []

        if MarketRegime.UNKNOWN in allowed_regimes and regime == MarketRegime.UNKNOWN:
            return [
                FilterResult(
                    name="regime_filter",
                    decision=FilterDecision.WARN,
                    reason="Regime unknown but allowed by runtime config",
                )
            ]

        if regime not in allowed_regimes:
            return [
                FilterResult(
                    name="regime_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Regime {regime.value} is not allowed",
                )
            ]

        return [
            FilterResult(
                name="regime_filter",
                decision=FilterDecision.PASS,
                reason=f"Regime {regime.value} allowed",
            )
        ]

    def _run_spread_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_spread_filter:
            return []

        spread_bps = None
        if context.price is not None:
            spread_bps = context.price.spread_bps

        if spread_bps is None:
            return [
                FilterResult(
                    name="spread_filter",
                    decision=FilterDecision.WARN,
                    reason="Spread unavailable",
                )
            ]

        if spread_bps > self.config.filters.max_spread_bps:
            return [
                FilterResult(
                    name="spread_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Spread too high: {spread_bps:.4f} bps",
                )
            ]

        return [
            FilterResult(
                name="spread_filter",
                decision=FilterDecision.PASS,
                reason=f"Spread acceptable: {spread_bps:.4f} bps",
            )
        ]

    def _run_liquidity_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_liquidity_filter:
            return []

        liquidity_score = self._safe_float(
            context.get_feature("liquidity_score"),
            default=None,
        )
        if liquidity_score is None:
            return [
                FilterResult(
                    name="liquidity_filter",
                    decision=FilterDecision.WARN,
                    reason="Liquidity score unavailable",
                )
            ]

        if liquidity_score < self.config.filters.min_liquidity_score:
            return [
                FilterResult(
                    name="liquidity_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Liquidity score too low: {liquidity_score:.4f}",
                )
            ]

        return [
            FilterResult(
                name="liquidity_filter",
                decision=FilterDecision.PASS,
                reason=f"Liquidity score acceptable: {liquidity_score:.4f}",
            )
        ]

    def _run_volatility_filter(self, context: SignalContext) -> list[FilterResult]:
        if not self.config.filters.enable_volatility_filter:
            return []

        volatility_zscore = self._safe_float(
            context.get_feature("volatility_zscore"),
            default=None,
        )
        if volatility_zscore is None:
            return [
                FilterResult(
                    name="volatility_filter",
                    decision=FilterDecision.WARN,
                    reason="Volatility z-score unavailable",
                )
            ]

        if volatility_zscore > self.config.filters.max_volatility_zscore:
            return [
                FilterResult(
                    name="volatility_filter",
                    decision=FilterDecision.BLOCK,
                    reason=f"Volatility too high: {volatility_zscore:.4f}",
                )
            ]

        return [
            FilterResult(
                name="volatility_filter",
                decision=FilterDecision.PASS,
                reason=f"Volatility acceptable: {volatility_zscore:.4f}",
            )
        ]

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_regime(self, context: SignalContext) -> MarketRegime:
        if context.regime is None:
            return MarketRegime.UNKNOWN
        return context.regime.regime

    def _resolve_strength(self, score: float, confidence: float) -> SignalStrength:
        composite = (score + confidence) / 2.0
        if composite >= 0.90:
            return SignalStrength.EXTREME
        if composite >= 0.75:
            return SignalStrength.STRONG
        if composite >= 0.55:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _resolve_confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    def _map_priority(self, priority: int) -> SignalPriority:
        if priority <= 25:
            return SignalPriority.CRITICAL
        if priority <= 50:
            return SignalPriority.HIGH
        if priority <= 100:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    def _safe_float(self, value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))