from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

from .base import ContextAwareComponent
from .config import FilterConfig, StrategyConfig
from .context import StrategyContext
from .enums import FilterDecision, MarketRegime
from .exceptions import FilterExecutionError
from .models import FilterResult, StrategySignal


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass(slots=True)
class FilterEvaluation:
    """
    Підсумок роботи пайплайна фільтрів.
    """

    signal: StrategySignal
    context_symbol: str
    timestamp: datetime = field(default_factory=utcnow)

    results: list[FilterResult] = field(default_factory=list)
    accepted: bool = True
    blocking_filters: list[str] = field(default_factory=list)
    warning_filters: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: FilterResult) -> None:
        result.validate()
        self.results.append(result)

        if result.decision == FilterDecision.BLOCK:
            self.accepted = False
            self.blocking_filters.append(result.name)
            if result.reason:
                self.reasons.append(result.reason)

        elif result.decision == FilterDecision.WARN:
            self.warning_filters.append(result.name)
            if result.reason:
                self.reasons.append(result.reason)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warning_filters)

    @property
    def has_blocks(self) -> bool:
        return bool(self.blocking_filters)


class BaseFilter(ContextAwareComponent, ABC):
    """
    Базовий клас для всіх strategy filters.
    """

    filter_name: str = "base_filter"

    def __init__(
        self,
        config: StrategyConfig,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    @property
    def filters_config(self) -> FilterConfig:
        return self.config.filters

    @property
    def name(self) -> str:
        return self.filter_name

    def is_enabled(self) -> bool:
        return True

    def evaluate(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        self.validate_context(context)
        signal.validate()

        try:
            result = self._evaluate(signal=signal, context=context)
        except Exception as exc:
            raise FilterExecutionError(
                f"{self.name}: filter evaluation failed: {exc}"
            ) from exc

        result.validate()
        return result

    @abstractmethod
    def _evaluate(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
    ) -> FilterResult:
        """
        Реальна логіка фільтра.
        """

    def pass_result(
        self,
        reason: str | None = None,
        *,
        score_impact: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FilterResult:
        return FilterResult(
            name=self.name,
            decision=FilterDecision.PASS,
            reason=reason,
            score_impact=score_impact,
            metadata=metadata or {},
        )

    def warn_result(
        self,
        reason: str,
        *,
        score_impact: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FilterResult:
        return FilterResult(
            name=self.name,
            decision=FilterDecision.WARN,
            reason=reason,
            score_impact=score_impact,
            metadata=metadata or {},
        )

    def block_result(
        self,
        reason: str,
        *,
        score_impact: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> FilterResult:
        return FilterResult(
            name=self.name,
            decision=FilterDecision.BLOCK,
            reason=reason,
            score_impact=score_impact,
            metadata=metadata or {},
        )

    def skip_result(
        self,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> FilterResult:
        return FilterResult(
            name=self.name,
            decision=FilterDecision.SKIP,
            reason=reason,
            score_impact=0.0,
            metadata=metadata or {},
        )


class RegimeFilter(BaseFilter):
    filter_name = "regime_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_regime_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("regime_filter_disabled")

        current_regime = context.current_regime
        if current_regime == MarketRegime.UNKNOWN:
            return self.warn_result(
                "unknown_market_regime",
                score_impact=-0.05,
                metadata={"regime": str(current_regime)},
            )

        if current_regime == MarketRegime.ILLIQUID:
            return self.block_result(
                "illiquid_market_regime",
                score_impact=-0.25,
                metadata={"regime": str(current_regime)},
            )

        if current_regime == MarketRegime.NEWS_DRIVEN:
            return self.warn_result(
                "news_driven_market_regime",
                score_impact=-0.15,
                metadata={"regime": str(current_regime)},
            )

        if current_regime == MarketRegime.RISK_OFF:
            return self.warn_result(
                "risk_off_market_regime",
                score_impact=-0.10,
                metadata={"regime": str(current_regime)},
            )

        return self.pass_result(
            "regime_ok",
            metadata={"regime": str(current_regime)},
        )


class VolatilityFilter(BaseFilter):
    filter_name = "volatility_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_volatility_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("volatility_filter_disabled")

        volatility_zscore = context.get_feature("volatility_zscore")
        if volatility_zscore is None:
            volatility_zscore = context.get_feature("realized_volatility_zscore")

        if volatility_zscore is None:
            return self.warn_result(
                "volatility_data_missing",
                score_impact=-0.03,
            )

        if not isinstance(volatility_zscore, (int, float)):
            return self.warn_result(
                "invalid_volatility_value",
                score_impact=-0.05,
                metadata={"value": volatility_zscore},
            )

        value = float(volatility_zscore)
        threshold = self.filters_config.max_volatility_zscore

        if value > threshold * 1.5:
            return self.block_result(
                "volatility_too_high",
                score_impact=-0.25,
                metadata={"volatility_zscore": value, "threshold": threshold},
            )

        if value > threshold:
            return self.warn_result(
                "elevated_volatility",
                score_impact=-0.10,
                metadata={"volatility_zscore": value, "threshold": threshold},
            )

        return self.pass_result(
            "volatility_ok",
            metadata={"volatility_zscore": value, "threshold": threshold},
        )


class LiquidityFilter(BaseFilter):
    filter_name = "liquidity_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_liquidity_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("liquidity_filter_disabled")

        liquidity_score = context.get_feature("liquidity_score")
        if liquidity_score is None:
            liquidity_score = context.liquidity.get("liquidity_score")

        if liquidity_score is None:
            return self.warn_result(
                "liquidity_data_missing",
                score_impact=-0.05,
            )

        if not isinstance(liquidity_score, (int, float)):
            return self.warn_result(
                "invalid_liquidity_score",
                score_impact=-0.05,
                metadata={"value": liquidity_score},
            )

        value = float(liquidity_score)
        threshold = self.filters_config.min_liquidity_score

        if value < threshold * 0.5:
            return self.block_result(
                "liquidity_too_low",
                score_impact=-0.25,
                metadata={"liquidity_score": value, "threshold": threshold},
            )

        if value < threshold:
            return self.warn_result(
                "suboptimal_liquidity",
                score_impact=-0.10,
                metadata={"liquidity_score": value, "threshold": threshold},
            )

        return self.pass_result(
            "liquidity_ok",
            metadata={"liquidity_score": value, "threshold": threshold},
        )


class SpreadFilter(BaseFilter):
    filter_name = "spread_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_spread_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("spread_filter_disabled")

        spread_bps = None
        if context.price is not None:
            spread_bps = context.price.spread_bps

        if spread_bps is None:
            spread_bps = context.get_feature("spread_bps")

        if spread_bps is None:
            return self.warn_result(
                "spread_data_missing",
                score_impact=-0.03,
            )

        if not isinstance(spread_bps, (int, float)):
            return self.warn_result(
                "invalid_spread_value",
                score_impact=-0.05,
                metadata={"value": spread_bps},
            )

        value = float(spread_bps)
        threshold = self.filters_config.max_spread_bps

        if value > threshold * 2:
            return self.block_result(
                "spread_too_wide",
                score_impact=-0.25,
                metadata={"spread_bps": value, "threshold": threshold},
            )

        if value > threshold:
            return self.warn_result(
                "spread_elevated",
                score_impact=-0.10,
                metadata={"spread_bps": value, "threshold": threshold},
            )

        return self.pass_result(
            "spread_ok",
            metadata={"spread_bps": value, "threshold": threshold},
        )


class FundingFilter(BaseFilter):
    filter_name = "funding_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_funding_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("funding_filter_disabled")

        funding_alignment = context.get_feature("funding_alignment")
        if funding_alignment is None:
            funding_alignment = context.funding.get("funding_alignment")

        funding_rate = context.get_feature("funding_rate")
        if funding_rate is None:
            funding_rate = context.funding.get("funding_rate")

        if funding_alignment is None and funding_rate is None:
            return self.warn_result(
                "funding_data_missing",
                score_impact=-0.02,
            )

        threshold = self.filters_config.min_funding_alignment

        if funding_alignment is not None and isinstance(funding_alignment, (int, float)):
            alignment = float(funding_alignment)
            if alignment < threshold:
                return self.warn_result(
                    "poor_funding_alignment",
                    score_impact=-0.08,
                    metadata={
                        "funding_alignment": alignment,
                        "threshold": threshold,
                        "funding_rate": funding_rate,
                    },
                )

        if funding_rate is not None and isinstance(funding_rate, (int, float)):
            rate = float(funding_rate)
            if abs(rate) > 0.01:
                return self.warn_result(
                    "extreme_funding_rate",
                    score_impact=-0.08,
                    metadata={
                        "funding_rate": rate,
                    },
                )

        return self.pass_result(
            "funding_ok",
            metadata={
                "funding_alignment": funding_alignment,
                "funding_rate": funding_rate,
            },
        )


class SessionFilter(BaseFilter):
    filter_name = "session_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_session_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("session_filter_disabled")

        session_name = context.get_feature("session_name")
        if session_name is None:
            session_name = context.extra.get("session_name")

        if session_name is not None:
            if isinstance(session_name, str):
                normalized = session_name.strip().lower()
                if normalized in {"dead", "inactive", "closed"}:
                    return self.block_result(
                        "inactive_trading_session",
                        score_impact=-0.20,
                        metadata={"session_name": session_name},
                    )
                if normalized in {"transition", "off_hours"}:
                    return self.warn_result(
                        "weak_trading_session",
                        score_impact=-0.08,
                        metadata={"session_name": session_name},
                    )
                return self.pass_result(
                    "session_ok",
                    metadata={"session_name": session_name},
                )

        # fallback за UTC-годиною
        hour = context.timestamp.hour
        if 21 <= hour or hour < 1:
            return self.warn_result(
                "late_session_utc",
                score_impact=-0.05,
                metadata={"hour_utc": hour},
            )

        return self.pass_result(
            "session_ok_fallback",
            metadata={"hour_utc": hour},
        )


class NewsFilter(BaseFilter):
    filter_name = "news_filter"

    def is_enabled(self) -> bool:
        return self.filters_config.enable_news_filter

    def _evaluate(self, *, signal: StrategySignal, context: StrategyContext) -> FilterResult:
        if not self.is_enabled():
            return self.skip_result("news_filter_disabled")

        news_risk = context.get_feature("news_risk")
        if news_risk is None:
            news_risk = context.extra.get("news_risk")

        if news_risk is None:
            return self.warn_result(
                "news_risk_unknown",
                score_impact=-0.03,
            )

        if isinstance(news_risk, str):
            normalized = news_risk.strip().lower()
            if normalized in {"high", "critical"}:
                return self.block_result(
                    "high_news_risk",
                    score_impact=-0.25,
                    metadata={"news_risk": news_risk},
                )
            if normalized in {"medium", "elevated"}:
                return self.warn_result(
                    "elevated_news_risk",
                    score_impact=-0.10,
                    metadata={"news_risk": news_risk},
                )
            return self.pass_result(
                "news_risk_ok",
                metadata={"news_risk": news_risk},
            )

        if isinstance(news_risk, (int, float)):
            value = float(news_risk)
            if value >= 0.8:
                return self.block_result(
                    "high_news_risk_numeric",
                    score_impact=-0.25,
                    metadata={"news_risk": value},
                )
            if value >= 0.5:
                return self.warn_result(
                    "elevated_news_risk_numeric",
                    score_impact=-0.10,
                    metadata={"news_risk": value},
                )
            return self.pass_result(
                "news_risk_ok_numeric",
                metadata={"news_risk": value},
            )

        return self.warn_result(
            "invalid_news_risk_value",
            score_impact=-0.05,
            metadata={"news_risk": news_risk},
        )


class FilterPipeline(ContextAwareComponent):
    """
    Оркестратор усіх strategy filters.

    Відповідає за:
    - побудову дефолтного пайплайна
    - послідовне виконання фільтрів
    - early stop на BLOCK
    - optional warning aggregation
    - прикріплення filter_results до signal
    """

    def __init__(
        self,
        config: StrategyConfig,
        filters: list[BaseFilter] | None = None,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.filters: list[BaseFilter] = filters or self._build_default_filters()

    def _build_default_filters(self) -> list[BaseFilter]:
        return [
            RegimeFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            VolatilityFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            LiquidityFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            SpreadFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            FundingFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            SessionFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
            NewsFilter(config=self.config, event_bus=self.event_bus, logger=self.logger),
        ]

    def evaluate(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext,
        stop_on_block: bool = True,
        attach_to_signal: bool = True,
    ) -> FilterEvaluation:
        self.validate_context(context)
        signal.validate()

        evaluation = FilterEvaluation(
            signal=signal,
            context_symbol=context.symbol,
            timestamp=context.timestamp,
        )

        for strategy_filter in self.filters:
            try:
                result = strategy_filter.evaluate(
                    signal=signal,
                    context=context,
                )
            except Exception as exc:
                raise FilterExecutionError(
                    f"Filter pipeline failed on '{strategy_filter.name}': {exc}"
                ) from exc

            evaluation.add_result(result)

            if attach_to_signal:
                signal.add_filter_result(result)

            if result.decision == FilterDecision.BLOCK and stop_on_block:
                break

        return evaluation

    def evaluate_many(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
        stop_on_block: bool = True,
        attach_to_signal: bool = True,
    ) -> list[FilterEvaluation]:
        return [
            self.evaluate(
                signal=signal,
                context=context,
                stop_on_block=stop_on_block,
                attach_to_signal=attach_to_signal,
            )
            for signal in signals
        ]

    def accepted_signals(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext,
        stop_on_block: bool = True,
        attach_to_signal: bool = True,
    ) -> tuple[list[StrategySignal], dict[str, FilterEvaluation]]:
        evaluations = self.evaluate_many(
            signals=signals,
            context=context,
            stop_on_block=stop_on_block,
            attach_to_signal=attach_to_signal,
        )

        accepted: list[StrategySignal] = []
        evaluation_map: dict[str, FilterEvaluation] = {}

        for evaluation in evaluations:
            strategy_name = evaluation.signal.strategy_name
            evaluation_map[strategy_name] = evaluation

            if evaluation.accepted:
                accepted.append(evaluation.signal)

        return accepted, evaluation_map

    def explain(self, evaluation: FilterEvaluation) -> dict[str, Any]:
        return {
            "strategy_name": evaluation.signal.strategy_name,
            "symbol": evaluation.signal.symbol,
            "accepted": evaluation.accepted,
            "blocking_filters": evaluation.blocking_filters,
            "warning_filters": evaluation.warning_filters,
            "reasons": evaluation.reasons,
            "results": [
                {
                    "name": result.name,
                    "decision": str(result.decision),
                    "reason": result.reason,
                    "score_impact": result.score_impact,
                    "metadata": result.metadata,
                }
                for result in evaluation.results
            ],
            "metadata": evaluation.metadata,
        }