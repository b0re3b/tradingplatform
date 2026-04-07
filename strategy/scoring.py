from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import StrategyComponent
from .config import ConflictConfig, StrategyConfig, VotingConfig, WeightingConfig
from .context import StrategyContext
from .enums import (
    ConfidenceGrade,
    ConflictType,
    MarketRegime,
    SignalSide,
    SignalStrength,
    StrategyCategory,
)
from .exceptions import ConflictResolutionError, ScoringError
from .models import (
    ConflictRecord,
    ConfluenceResult,
    StrategySignal,
    confidence_to_grade,
    confidence_to_strength,
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass(slots=True)
class WeightedSignal:
    """
    StrategySignal після застосування ваг.
    """

    signal: StrategySignal
    base_weight: float
    regime_weight: float
    strategy_weight: float
    final_weight: float
    weighted_score: float
    weighted_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal.validate()
        if self.base_weight < 0:
            raise ScoringError("WeightedSignal.base_weight must be >= 0")
        if self.regime_weight < 0:
            raise ScoringError("WeightedSignal.regime_weight must be >= 0")
        if self.strategy_weight < 0:
            raise ScoringError("WeightedSignal.strategy_weight must be >= 0")
        if self.final_weight < 0:
            raise ScoringError("WeightedSignal.final_weight must be >= 0")


@dataclass(slots=True)
class VoteSummary:
    """
    Підсумок voting stage.
    """

    total_votes: int = 0
    long_votes: int = 0
    short_votes: int = 0
    flat_votes: int = 0

    confirmation_count: int = 0
    primary_count: int = 0

    dominant_side: SignalSide = SignalSide.UNKNOWN
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConflictSummary:
    """
    Підсумок conflict resolution.
    """

    accepted: bool = True
    total_penalty: float = 0.0
    conflicts: list[ConflictRecord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        conflict.validate()
        self.conflicts.append(conflict)
        self.total_penalty += conflict.penalty


@dataclass(slots=True)
class ScoreBreakdown:
    """
    Детальний результат агрегації score/confidence.
    """

    symbol: str
    side: SignalSide = SignalSide.UNKNOWN

    raw_score_sum: float = 0.0
    weighted_score_sum: float = 0.0
    weighted_confidence_sum: float = 0.0
    total_weight: float = 0.0

    confidence: float = 0.0
    score: float = 0.0
    confidence_grade: ConfidenceGrade = ConfidenceGrade.VERY_LOW
    strength: SignalStrength = SignalStrength.WEAK

    strategy_names: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ScoringError("ScoreBreakdown.symbol cannot be empty")
        if self.total_weight < 0:
            raise ScoringError("ScoreBreakdown.total_weight must be >= 0")

    def add_reason(self, reason: str) -> None:
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def add_confirmation(self, confirmation: str) -> None:
        if confirmation and confirmation not in self.confirmations:
            self.confirmations.append(confirmation)


class WeightingEngine(StrategyComponent):
    """
    Застосовує ваги до сирих strategy signals.

    Враховує:
    - вагу категорії
    - вагу strategy definition
    - adjustment по market regime
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def weighting_config(self) -> WeightingConfig:
        return self.config.weighting

    def apply(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None = None,
    ) -> WeightedSignal:
        signal.validate()

        regime = context.current_regime if context is not None else signal.regime
        category_weight = self._category_weight(signal.category)
        regime_weight = self._regime_weight(regime)
        strategy_weight = self.config.get_strategy_weight(signal.strategy_name, default=1.0)

        final_weight = category_weight * regime_weight * strategy_weight
        weighted_score = signal.score * final_weight
        weighted_confidence = signal.confidence * final_weight

        result = WeightedSignal(
            signal=signal,
            base_weight=category_weight,
            regime_weight=regime_weight,
            strategy_weight=strategy_weight,
            final_weight=final_weight,
            weighted_score=weighted_score,
            weighted_confidence=weighted_confidence,
            metadata={
                "regime": str(regime),
                "category": str(signal.category),
            },
        )
        result.validate()
        return result

    def apply_many(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> list[WeightedSignal]:
        return [self.apply(signal=signal, context=context) for signal in signals]

    def _category_weight(self, category: StrategyCategory) -> float:
        return self.weighting_config.category_weights.get(category, 1.0)

    def _regime_weight(self, regime: MarketRegime) -> float:
        return self.weighting_config.regime_adjustments.get(regime, 1.0)


class VotingEngine(StrategyComponent):
    """
    Відповідає за side voting та acceptance logic.

    Перевіряє:
    - dominant side
    - min confirmations
    - min total votes
    - require_primary_trigger
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def voting_config(self) -> VotingConfig:
        return self.config.voting

    def summarize(self, signals: list[StrategySignal]) -> VoteSummary:
        if not signals:
            return VoteSummary(
                total_votes=0,
                accepted=False,
                dominant_side=SignalSide.UNKNOWN,
                reasons=["no_signals"],
            )

        summary = VoteSummary()

        for signal in signals:
            signal.validate()
            if signal.side == SignalSide.LONG:
                summary.long_votes += 1
            elif signal.side == SignalSide.SHORT:
                summary.short_votes += 1
            else:
                summary.flat_votes += 1

            if signal.trigger_type.value == "confirmation":
                summary.confirmation_count += 1
            if signal.trigger_type.value == "primary":
                summary.primary_count += 1

        summary.total_votes = len(signals)
        summary.dominant_side = self._dominant_side(summary)

        reasons: list[str] = []

        if summary.total_votes < self.voting_config.min_total_votes:
            reasons.append("not_enough_total_votes")

        if summary.confirmation_count < self.voting_config.min_confirmations:
            reasons.append("not_enough_confirmations")

        if self.voting_config.require_primary_trigger and summary.primary_count < 1:
            reasons.append("primary_trigger_required")

        if summary.dominant_side == SignalSide.UNKNOWN:
            reasons.append("no_dominant_side")

        summary.accepted = not reasons
        summary.reasons = reasons
        return summary

    def _dominant_side(self, summary: VoteSummary) -> SignalSide:
        if summary.long_votes > summary.short_votes and summary.long_votes > 0:
            return SignalSide.LONG
        if summary.short_votes > summary.long_votes and summary.short_votes > 0:
            return SignalSide.SHORT
        if summary.flat_votes > 0 and summary.long_votes == 0 and summary.short_votes == 0:
            return SignalSide.FLAT
        return SignalSide.UNKNOWN


class ConflictResolver(StrategyComponent):
    """
    Виявляє та штрафує конфлікти між strategy signals.

    Підтримує:
    - side conflict
    - regime conflict
    - timeframe/category conflicts (як soft conflicts)
    - reject_on_side_conflict
    - reject_on_regime_conflict
    - max_total_penalty
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def conflict_config(self) -> ConflictConfig:
        return self.config.conflict

    def resolve(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
        context: StrategyContext | None = None,
    ) -> ConflictSummary:
        summary = ConflictSummary()

        if not signals:
            summary.accepted = False
            summary.reasons.append("no_signals_for_conflict_resolution")
            return summary

        try:
            self._detect_side_conflicts(
                signals=signals,
                dominant_side=dominant_side,
                summary=summary,
            )
            self._detect_regime_conflicts(
                signals=signals,
                context=context,
                summary=summary,
            )
            self._detect_timeframe_conflicts(
                signals=signals,
                summary=summary,
            )
        except Exception as exc:
            raise ConflictResolutionError(f"Conflict resolution failed: {exc}") from exc

        reasons: list[str] = list(summary.reasons)

        if self.conflict_config.reject_on_side_conflict and any(
            conflict.conflict_type == ConflictType.SIDE_CONFLICT for conflict in summary.conflicts
        ):
            reasons.append("rejected_on_side_conflict")

        if self.conflict_config.reject_on_regime_conflict and any(
            conflict.conflict_type == ConflictType.REGIME_CONFLICT for conflict in summary.conflicts
        ):
            reasons.append("rejected_on_regime_conflict")

        if summary.total_penalty > self.conflict_config.max_total_penalty:
            reasons.append("conflict_penalty_too_high")

        summary.accepted = not reasons
        summary.reasons = reasons
        return summary

    def _detect_side_conflicts(
        self,
        *,
        signals: list[StrategySignal],
        dominant_side: SignalSide,
        summary: ConflictSummary,
    ) -> None:
        if dominant_side not in {SignalSide.LONG, SignalSide.SHORT}:
            return

        opposite_side = SignalSide.SHORT if dominant_side == SignalSide.LONG else SignalSide.LONG

        for signal in signals:
            if signal.side == opposite_side:
                summary.add_conflict(
                    ConflictRecord(
                        conflict_type=ConflictType.SIDE_CONFLICT,
                        source=signal.strategy_name,
                        message=f"signal side {signal.side} conflicts with dominant side {dominant_side}",
                        penalty=0.15,
                        metadata={
                            "dominant_side": str(dominant_side),
                            "signal_side": str(signal.side),
                        },
                    )
                )

    def _detect_regime_conflicts(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
        summary: ConflictSummary,
    ) -> None:
        if context is None:
            return

        current_regime = context.current_regime
        if current_regime == MarketRegime.UNKNOWN:
            return

        for signal in signals:
            if signal.regime not in {MarketRegime.UNKNOWN, current_regime}:
                summary.add_conflict(
                    ConflictRecord(
                        conflict_type=ConflictType.REGIME_CONFLICT,
                        source=signal.strategy_name,
                        message=f"signal regime {signal.regime} conflicts with context regime {current_regime}",
                        penalty=0.10,
                        metadata={
                            "context_regime": str(current_regime),
                            "signal_regime": str(signal.regime),
                        },
                    )
                )

    def _detect_timeframe_conflicts(
        self,
        *,
        signals: list[StrategySignal],
        summary: ConflictSummary,
    ) -> None:
        timeframes = {signal.timeframe for signal in signals}
        if len(timeframes) > 1:
            summary.add_conflict(
                ConflictRecord(
                    conflict_type=ConflictType.TIMEFRAME_CONFLICT,
                    source="timeframe_consistency",
                    message="multiple timeframes detected in signal set",
                    penalty=0.05,
                    metadata={
                        "timeframes": [str(item) for item in sorted(timeframes, key=str)],
                    },
                )
            )


class ConfidenceScorer(StrategyComponent):
    """
    Підсумкова агрегація score/confidence по weighted signals.

    Враховує:
    - weighted score
    - weighted confidence
    - confirmation bonus
    - conflict penalty
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    def aggregate(
        self,
        *,
        symbol: str,
        weighted_signals: list[WeightedSignal],
        vote_summary: VoteSummary,
        conflict_summary: ConflictSummary,
    ) -> ScoreBreakdown:
        if not symbol.strip():
            raise ScoringError("symbol cannot be empty")

        breakdown = ScoreBreakdown(
            symbol=symbol,
            side=vote_summary.dominant_side,
        )

        if not weighted_signals:
            breakdown.add_reason("no_weighted_signals")
            return breakdown

        for item in weighted_signals:
            item.validate()
            signal = item.signal

            breakdown.raw_score_sum += signal.score
            breakdown.weighted_score_sum += item.weighted_score
            breakdown.weighted_confidence_sum += item.weighted_confidence
            breakdown.total_weight += item.final_weight

            if signal.strategy_name not in breakdown.strategy_names:
                breakdown.strategy_names.append(signal.strategy_name)

            for reason in signal.reasons:
                breakdown.add_reason(reason)

            for confirmation in signal.confirmations:
                breakdown.add_confirmation(confirmation)

        if breakdown.total_weight > 0:
            breakdown.score = breakdown.weighted_score_sum / breakdown.total_weight
            breakdown.confidence = breakdown.weighted_confidence_sum / breakdown.total_weight
        else:
            breakdown.score = 0.0
            breakdown.confidence = 0.0

        breakdown.confidence += self.config.confluence.confirmation_bonus * vote_summary.confirmation_count
        breakdown.confidence -= conflict_summary.total_penalty
        breakdown.confidence = clamp(breakdown.confidence, 0.0, 1.0)

        breakdown.confidence_grade = confidence_to_grade(breakdown.confidence)
        breakdown.strength = confidence_to_strength(breakdown.confidence)

        if vote_summary.reasons:
            for reason in vote_summary.reasons:
                breakdown.add_reason(f"voting:{reason}")

        if conflict_summary.reasons:
            for reason in conflict_summary.reasons:
                breakdown.add_reason(f"conflict:{reason}")

        breakdown.metadata = {
            "raw_score_sum": breakdown.raw_score_sum,
            "weighted_score_sum": breakdown.weighted_score_sum,
            "weighted_confidence_sum": breakdown.weighted_confidence_sum,
            "total_weight": breakdown.total_weight,
            "confirmation_count": vote_summary.confirmation_count,
            "conflict_penalty": conflict_summary.total_penalty,
        }

        breakdown.validate()
        return breakdown

    def to_confluence_result(
        self,
        *,
        symbol: str,
        weighted_signals: list[WeightedSignal],
        vote_summary: VoteSummary,
        conflict_summary: ConflictSummary,
    ) -> ConfluenceResult:
        breakdown = self.aggregate(
            symbol=symbol,
            weighted_signals=weighted_signals,
            vote_summary=vote_summary,
            conflict_summary=conflict_summary,
        )

        accepted = (
            vote_summary.accepted
            and conflict_summary.accepted
            and breakdown.confidence >= self.config.confluence.min_confidence
            and breakdown.score >= self.config.confluence.min_score
        )

        result = ConfluenceResult(
            symbol=symbol,
            timestamp=weighted_signals[0].signal.timestamp if weighted_signals else contextless_now(),
            side=breakdown.side,
            score=breakdown.score,
            confidence=breakdown.confidence,
            confidence_grade=breakdown.confidence_grade,
            strength=breakdown.strength,
            strategy_names=breakdown.strategy_names,
            reasons=breakdown.reasons,
            confirmations=breakdown.confirmations,
            conflicts=conflict_summary.conflicts,
            accepted=accepted,
            metadata={
                **breakdown.metadata,
                "vote_summary": {
                    "total_votes": vote_summary.total_votes,
                    "long_votes": vote_summary.long_votes,
                    "short_votes": vote_summary.short_votes,
                    "flat_votes": vote_summary.flat_votes,
                    "dominant_side": str(vote_summary.dominant_side),
                    "accepted": vote_summary.accepted,
                    "reasons": vote_summary.reasons,
                },
                "conflict_summary": {
                    "accepted": conflict_summary.accepted,
                    "total_penalty": conflict_summary.total_penalty,
                    "reasons": conflict_summary.reasons,
                },
            },
        )
        result.validate()
        return result


class StrategyScoring(StrategyComponent):
    """
    Фасад над усім scoring pipeline.

    Послідовність:
    1. weighting
    2. voting
    3. conflict resolution
    4. confidence aggregation
    """

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)

        self.weighting = WeightingEngine(config=config, event_bus=event_bus, logger=logger)
        self.voting = VotingEngine(config=config, event_bus=event_bus, logger=logger)
        self.conflicts = ConflictResolver(config=config, event_bus=event_bus, logger=logger)
        self.confidence = ConfidenceScorer(config=config, event_bus=event_bus, logger=logger)

    def score_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> ConfluenceResult:
        if not signals:
            raise ScoringError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise ScoringError("all signals must belong to the same symbol")

        weighted_signals = self.weighting.apply_many(
            signals=signals,
            context=context,
        )

        vote_summary = self.voting.summarize(signals)
        conflict_summary = self.conflicts.resolve(
            signals=signals,
            dominant_side=vote_summary.dominant_side,
            context=context,
        )

        return self.confidence.to_confluence_result(
            symbol=symbol,
            weighted_signals=weighted_signals,
            vote_summary=vote_summary,
            conflict_summary=conflict_summary,
        )


def contextless_now():
    from datetime import datetime
    return datetime.utcnow()