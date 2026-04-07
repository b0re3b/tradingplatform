from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import StrategyComponent
from .config import ConfluenceConfig, StrategyConfig
from .context import StrategyContext
from .enums import (
    SignalOrigin,
    SignalPriority,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    TriggerType,
)
from .exceptions import ConfluenceError
from .models import (
    ConfluenceResult,
    EntryPlan,
    ExitPlan,
    InvalidationPlan,
    StrategySignal,
)
from .scoring import StrategyScoring


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass(slots=True)
class ConfluenceEvaluation:
    """
    Підсумок роботи confluence engine.

    - raw_signals: усі отримані сигнали
    - eligible_signals: сигнали після базового pre-check
    - accepted_signals: сигнали, що підтримують фінальний consensus
    - rejected_signals: сигнали, які були відкинуті
    - result: агрегований ConfluenceResult
    - merged_signal: фінальний комбінований StrategySignal | None
    """

    symbol: str
    timestamp: datetime

    raw_signals: list[StrategySignal] = field(default_factory=list)
    eligible_signals: list[StrategySignal] = field(default_factory=list)
    accepted_signals: list[StrategySignal] = field(default_factory=list)
    rejected_signals: dict[str, str] = field(default_factory=dict)

    result: ConfluenceResult | None = None
    merged_signal: StrategySignal | None = None

    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.result is not None and self.result.accepted

    @property
    def selected_strategy_names(self) -> list[str]:
        return [signal.strategy_name for signal in self.accepted_signals]


class ConfluenceEngine(StrategyComponent):
    """
    Відповідає за об'єднання кількох strategy signals у фінальний confluence result.

    Пайплайн:
    1. validate signals
    2. pre-filter
    3. score_signals через StrategyScoring
    4. перевірка confluence rules
    5. optional merge into final StrategySignal
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()
        self.scoring = StrategyScoring(config=config, event_bus=event_bus, logger=logger)

    @property
    def confluence_config(self) -> ConfluenceConfig:
        return self.config.confluence

    def evaluate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
        merge_signal: bool = True,
    ) -> ConfluenceEvaluation:
        if not signals:
            raise ConfluenceError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = self._ensure_same_symbol(signals)
        timestamp = self._resolve_timestamp(signals, context)

        evaluation = ConfluenceEvaluation(
            symbol=symbol,
            timestamp=timestamp,
            raw_signals=list(signals),
        )

        eligible, rejected = self._pre_filter_signals(signals=signals, context=context)
        evaluation.eligible_signals = eligible
        evaluation.rejected_signals = rejected

        if not eligible:
            evaluation.reasons.append("no_eligible_signals")
            evaluation.result = ConfluenceResult(
                symbol=symbol,
                timestamp=timestamp,
                accepted=False,
                reasons=["no_eligible_signals"],
            )
            return evaluation

        result = self.scoring.score_signals(
            signals=eligible,
            context=context,
        )

        evaluation.result = result

        accepted_signals, rejected_by_side = self._select_consensus_signals(
            signals=eligible,
            dominant_side=result.side,
        )
        evaluation.accepted_signals = accepted_signals
        evaluation.rejected_signals.update(rejected_by_side)

        acceptance_reasons = self._evaluate_acceptance(
            result=result,
            accepted_signals=accepted_signals,
        )
        if acceptance_reasons:
            result.accepted = False
            for reason in acceptance_reasons:
                if reason not in result.reasons:
                    result.reasons.append(reason)
                if reason not in evaluation.reasons:
                    evaluation.reasons.append(reason)

        if merge_signal and result.accepted and accepted_signals:
            evaluation.merged_signal = self._merge_signals(
                signals=accepted_signals,
                result=result,
                context=context,
            )

        self.log_debug(
            "Confluence evaluation completed",
            symbol=symbol,
            accepted=result.accepted,
            side=str(result.side),
            confidence=result.confidence,
            score=result.score,
            selected=evaluation.selected_strategy_names,
            rejected=evaluation.rejected_signals,
        )

        return evaluation

    def merge_only(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> StrategySignal:
        evaluation = self.evaluate(
            signals=signals,
            context=context,
            merge_signal=True,
        )
        if evaluation.merged_signal is None:
            raise ConfluenceError("unable to build merged signal from provided signals")
        return evaluation.merged_signal

    def explain(self, evaluation: ConfluenceEvaluation) -> dict[str, Any]:
        result = evaluation.result
        return {
            "symbol": evaluation.symbol,
            "timestamp": evaluation.timestamp.isoformat(),
            "accepted": evaluation.accepted,
            "raw_signals": [signal.strategy_name for signal in evaluation.raw_signals],
            "eligible_signals": [signal.strategy_name for signal in evaluation.eligible_signals],
            "accepted_signals": [signal.strategy_name for signal in evaluation.accepted_signals],
            "rejected_signals": evaluation.rejected_signals,
            "reasons": evaluation.reasons,
            "result": None if result is None else {
                "side": str(result.side),
                "score": result.score,
                "confidence": result.confidence,
                "accepted": result.accepted,
                "strategy_names": result.strategy_names,
                "reasons": result.reasons,
                "confirmations": result.confirmations,
                "conflicts": [
                    {
                        "type": str(conflict.conflict_type),
                        "source": conflict.source,
                        "message": conflict.message,
                        "penalty": conflict.penalty,
                    }
                    for conflict in result.conflicts
                ],
                "metadata": result.metadata,
            },
            "merged_signal": None if evaluation.merged_signal is None else {
                "strategy_name": evaluation.merged_signal.strategy_name,
                "side": str(evaluation.merged_signal.side),
                "score": evaluation.merged_signal.score,
                "confidence": evaluation.merged_signal.confidence,
                "combined_from": evaluation.merged_signal.combined_from,
            },
        }

    def _ensure_same_symbol(self, signals: list[StrategySignal]) -> str:
        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise ConfluenceError("all signals must belong to the same symbol")
        return symbol

    def _resolve_timestamp(
        self,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> datetime:
        if context is not None:
            return context.timestamp
        return max(signal.timestamp for signal in signals)

    def _pre_filter_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        eligible: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            reason = self._pre_filter_reason(signal=signal, context=context)
            if reason is not None:
                rejected[signal.strategy_name] = reason
                continue
            eligible.append(signal)

        return eligible, rejected

    def _pre_filter_reason(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None,
    ) -> str | None:
        if signal.status in {
            SignalStatus.REJECTED,
            SignalStatus.CANCELLED,
            SignalStatus.EXPIRED,
            SignalStatus.FAILED,
        }:
            return f"signal_status:{signal.status}"

        if not signal.is_directional:
            return "non_directional_signal"

        if signal.confidence < 0:
            return "negative_confidence"

        if signal.score < 0:
            return "negative_score"

        if context is not None and signal.symbol != context.symbol:
            return "symbol_mismatch"

        if context is not None:
            if signal.timeframe != context.timeframe:
                # soft rule: не банимо, якщо хочеш жорстко, можна змінити
                pass

        return None

    def _select_consensus_signals(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if dominant_side not in {SignalSide.LONG, SignalSide.SHORT}:
            return [], {signal.strategy_name: "no_dominant_side" for signal in signals}

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            if signal.side == dominant_side:
                accepted.append(signal)
            else:
                rejected[signal.strategy_name] = "not_in_consensus_side"

        return accepted, rejected

    def _evaluate_acceptance(
        self,
        *,
        result: ConfluenceResult,
        accepted_signals: list[StrategySignal],
    ) -> list[str]:
        reasons: list[str] = []

        if not result.accepted:
            reasons.append("scoring_rejected")

        if result.side not in {SignalSide.LONG, SignalSide.SHORT}:
            reasons.append("invalid_final_side")

        if len(accepted_signals) < self.confluence_config.min_agreement_count:
            reasons.append("not_enough_agreement")

        if result.confidence < self.confluence_config.min_confidence:
            reasons.append("confidence_below_threshold")

        if result.score < self.confluence_config.min_score:
            reasons.append("score_below_threshold")

        return reasons

    def _merge_signals(
        self,
        *,
        signals: list[StrategySignal],
        result: ConfluenceResult,
        context: StrategyContext | None,
    ) -> StrategySignal:
        if not signals:
            raise ConfluenceError("cannot merge empty signal list")

        primary = self._select_primary_signal(signals)

        merged_entry = self._merge_entry_plan(signals)
        merged_exit = self._merge_exit_plan(signals)
        merged_invalidation = self._merge_invalidation_plan(signals)

        merged = StrategySignal(
            symbol=primary.symbol,
            side=result.side,
            strategy_name="ConfluenceEngine",
            category=StrategyCategory.HYBRID,
            timeframe=context.timeframe if context is not None else primary.timeframe,
            setup_type=primary.setup_type,
            timestamp=context.timestamp if context is not None else primary.timestamp,
            confidence=result.confidence,
            score=result.score,
            strength=result.strength,
            confidence_grade=result.confidence_grade,
            status=SignalStatus.CONFIRMED if result.accepted else SignalStatus.REJECTED,
            trigger_type=TriggerType.CONFLUENCE,
            origin=SignalOrigin.CONFLUENCE,
            priority=self._resolve_priority(signals),
            reasons=self._merge_reasons(signals, result),
            confirmations=self._merge_confirmations(signals, result),
            source_features=self._merge_source_features(signals),
            combined_from=[signal.strategy_name for signal in signals],
            conflicts=result.conflicts,
            filter_results=self._merge_filter_results(signals),
            entry_plan=merged_entry,
            exit_plan=merged_exit,
            invalidation_plan=merged_invalidation,
            regime=context.current_regime if context is not None else primary.regime,
            metadata={
                "primary_strategy": primary.strategy_name,
                "confluence_result": {
                    "strategy_names": result.strategy_names,
                    "accepted": result.accepted,
                    "score": result.score,
                    "confidence": result.confidence,
                },
            },
        )
        merged.validate()
        return merged

    def _select_primary_signal(self, signals: list[StrategySignal]) -> StrategySignal:
        """
        Primary signal:
        1. highest confidence
        2. highest score
        """
        return max(
            signals,
            key=lambda signal: (signal.confidence, signal.score),
        )

    def _resolve_priority(self, signals: list[StrategySignal]) -> SignalPriority:
        priorities = [signal.priority for signal in signals]

        if SignalPriority.CRITICAL in priorities:
            return SignalPriority.CRITICAL
        if SignalPriority.HIGH in priorities:
            return SignalPriority.HIGH
        if SignalPriority.MEDIUM in priorities:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    def _merge_reasons(
        self,
        signals: list[StrategySignal],
        result: ConfluenceResult,
    ) -> list[str]:
        merged: list[str] = []

        for signal in signals:
            for reason in signal.reasons:
                if reason not in merged:
                    merged.append(reason)

        for reason in result.reasons:
            if reason not in merged:
                merged.append(reason)

        return merged

    def _merge_confirmations(
        self,
        signals: list[StrategySignal],
        result: ConfluenceResult,
    ) -> list[str]:
        merged: list[str] = []

        for signal in signals:
            for confirmation in signal.confirmations:
                if confirmation not in merged:
                    merged.append(confirmation)

        for confirmation in result.confirmations:
            if confirmation not in merged:
                merged.append(confirmation)

        return merged

    def _merge_source_features(self, signals: list[StrategySignal]) -> list[str]:
        merged: list[str] = []

        for signal in signals:
            for feature_name in signal.source_features:
                if feature_name not in merged:
                    merged.append(feature_name)

        return merged

    def _merge_filter_results(self, signals: list[StrategySignal]):
        merged = []
        seen: set[tuple[str, str]] = set()

        for signal in signals:
            for item in signal.filter_results:
                key = (item.name, str(item.decision))
                if key in seen:
                    continue
                merged.append(item)
                seen.add(key)

        return merged

    def _merge_entry_plan(self, signals: list[StrategySignal]) -> EntryPlan | None:
        candidates = [signal.entry_plan for signal in signals if signal.entry_plan is not None]
        if not candidates:
            return None

        priced = [item for item in candidates if item.price is not None]
        selected = min(priced, key=lambda item: item.price) if priced else candidates[0]
        selected.validate()
        return selected

    def _merge_exit_plan(self, signals: list[StrategySignal]) -> ExitPlan | None:
        candidates = [signal.exit_plan for signal in signals if signal.exit_plan is not None]
        if not candidates:
            return None

        selected = max(
            candidates,
            key=lambda item: len(item.take_profit_levels),
        )
        selected.validate()
        return selected

    def _merge_invalidation_plan(
        self,
        signals: list[StrategySignal],
    ) -> InvalidationPlan | None:
        candidates = [
            signal.invalidation_plan
            for signal in signals
            if signal.invalidation_plan is not None
        ]
        if not candidates:
            return None

        priced = [item for item in candidates if item.price is not None]
        selected = priced[0] if priced else candidates[0]
        selected.validate()
        return selected