from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import StrategyComponent
from .config import PortfolioCoordinatorConfig, StrategyConfig
from .context import StrategyContext
from .enums import SignalPriority, SignalSide, SignalStatus, StrategyCategory
from .exceptions import PortfolioCoordinationError
from .models import StrategySignal
from .state import StrategyState


def utcnow() -> datetime:
    return datetime.utcnow()


@dataclass(slots=True)
class CoordinationDecision:
    """
    Результат координації одного або кількох сигналів на рівні портфеля.
    """

    symbol: str
    timestamp: datetime

    raw_signals: list[StrategySignal] = field(default_factory=list)
    accepted_signals: list[StrategySignal] = field(default_factory=list)
    rejected_signals: dict[str, str] = field(default_factory=dict)
    merged_signals: list[StrategySignal] = field(default_factory=list)

    throttled_signals: dict[str, str] = field(default_factory=dict)
    suppressed_signals: dict[str, str] = field(default_factory=dict)

    accepted: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_signals(self) -> list[StrategySignal]:
        if self.merged_signals:
            return self.merged_signals
        return self.accepted_signals

    @property
    def selected_names(self) -> list[str]:
        return [signal.strategy_name for signal in self.final_signals]


class PortfolioCoordinator(StrategyComponent):
    """
    Координує фінальні сигнали перед передачею в risk/execution layer.

    Підтримує:
    - symbol cooldown
    - side cooldown
    - blocked symbols
    - repeated signal suppression
    - deduplicate by side
    - limit by strategy category
    - symbol throttling by volatility regime
    - max signals per symbol
    - exposure buckets
    - correlation guard
    - correlation direction conflict
    - merge similar signals
    """

    def __init__(
        self,
        config: StrategyConfig,
        state: StrategyState,
        event_bus=None,
        logger=None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()
        self.state = state

    @property
    def portfolio_config(self) -> PortfolioCoordinatorConfig:
        return self.config.portfolio

    def coordinate(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None = None,
    ) -> CoordinationDecision:
        if not signals:
            raise PortfolioCoordinationError("signals cannot be empty")

        for signal in signals:
            signal.validate()

        symbol = self._ensure_same_symbol(signals)
        now = context.timestamp if context is not None else utcnow()

        decision = CoordinationDecision(
            symbol=symbol,
            timestamp=now,
            raw_signals=list(signals),
        )

        accepted, rejected = self._apply_prechecks(
            signals=signals,
            context=context,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_prechecks")
            return decision

        accepted, suppressed = self._suppress_repeating_signals(
            symbol=symbol,
            signals=accepted,
            now=now,
        )
        decision.accepted_signals = accepted
        decision.suppressed_signals.update(suppressed)
        decision.rejected_signals.update(suppressed)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_repeat_suppression")
            return decision

        accepted, rejected_after_dedup = self._deduplicate_signals(
            signals=accepted,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_after_dedup)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_deduplication")
            return decision

        accepted, rejected_after_category = self._apply_category_limits(
            signals=accepted,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_after_category)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_category_limits")
            return decision

        accepted, throttled = self._apply_volatility_throttle(
            symbol=symbol,
            signals=accepted,
            context=context,
        )
        decision.accepted_signals = accepted
        decision.throttled_signals.update(throttled)
        decision.rejected_signals.update(throttled)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_volatility_throttle")
            return decision

        accepted, rejected_after_limits = self._apply_symbol_limits(
            symbol=symbol,
            signals=accepted,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_after_limits)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_symbol_limits")
            return decision

        accepted, rejected_after_exposure = self._apply_exposure_buckets(
            symbol=symbol,
            signals=accepted,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_after_exposure)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_exposure_buckets")
            return decision

        accepted, rejected_after_correlation = self._apply_correlation_guard(
            symbol=symbol,
            signals=accepted,
        )
        decision.accepted_signals = accepted
        decision.rejected_signals.update(rejected_after_correlation)

        if not accepted:
            decision.accepted = False
            decision.reasons.append("no_signals_after_correlation_guard")
            return decision

        merged = (
            self._merge_similar_signals(accepted)
            if self.portfolio_config.merge_similar_signals
            else accepted
        )
        decision.merged_signals = merged

        if not merged:
            decision.accepted = False
            decision.reasons.append("no_signals_after_merge")
            return decision

        if self.portfolio_config.enabled:
            self._update_state_after_acceptance(
                symbol=symbol,
                signals=merged,
            )

        return decision

    def coordinate_one(
        self,
        *,
        signal: StrategySignal,
        context: StrategyContext | None = None,
    ) -> CoordinationDecision:
        return self.coordinate(signals=[signal], context=context)

    def explain(self, decision: CoordinationDecision) -> dict[str, Any]:
        return {
            "symbol": decision.symbol,
            "timestamp": decision.timestamp.isoformat(),
            "accepted": decision.accepted,
            "raw_signals": [signal.strategy_name for signal in decision.raw_signals],
            "accepted_signals": [signal.strategy_name for signal in decision.accepted_signals],
            "merged_signals": [signal.strategy_name for signal in decision.merged_signals],
            "rejected_signals": decision.rejected_signals,
            "throttled_signals": decision.throttled_signals,
            "suppressed_signals": decision.suppressed_signals,
            "reasons": decision.reasons,
            "metadata": decision.metadata,
        }

    def _ensure_same_symbol(self, signals: list[StrategySignal]) -> str:
        symbol = signals[0].symbol
        if any(signal.symbol != symbol for signal in signals):
            raise PortfolioCoordinationError("all signals must belong to the same symbol")
        return symbol

    def _apply_prechecks(
        self,
        *,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        symbol = signals[0].symbol
        now = context.timestamp if context is not None else utcnow()

        if self.state.portfolio.is_symbol_blocked(symbol):
            for signal in signals:
                rejected[signal.strategy_name] = "symbol_blocked"
            return [], rejected

        if self._is_symbol_on_cooldown(symbol=symbol, now=now):
            for signal in signals:
                rejected[signal.strategy_name] = "symbol_on_cooldown"
            return [], rejected

        for signal in signals:
            if self._is_side_on_cooldown(symbol=symbol, side=signal.side, now=now):
                rejected[signal.strategy_name] = f"side_on_cooldown:{signal.side.value}"
                continue

            if signal.status in {
                SignalStatus.REJECTED,
                SignalStatus.CANCELLED,
                SignalStatus.EXPIRED,
                SignalStatus.FAILED,
            }:
                rejected[signal.strategy_name] = f"invalid_signal_status:{signal.status.value}"
                continue

            if not signal.is_directional:
                rejected[signal.strategy_name] = "non_directional_signal"
                continue

            accepted.append(signal)

        return accepted, rejected

    def _suppress_repeating_signals(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
        now: datetime,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        suppression_window = self.portfolio_config.repeated_signal_suppression_seconds
        if suppression_window <= 0:
            return signals, {}

        symbol_state = self.state.get_symbol_state(symbol)
        if symbol_state is None:
            return signals, {}

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            previous = symbol_state.last_signal_by_side.get(signal.side.value)
            if previous is None:
                accepted.append(signal)
                continue

            delta = (now - previous.timestamp).total_seconds()
            same_setup = previous.setup_type == signal.setup_type
            same_strategy = previous.strategy_name == signal.strategy_name

            if delta <= suppression_window and (same_setup or same_strategy):
                rejected[signal.strategy_name] = "repeating_signal_suppressed"
                continue

            accepted.append(signal)

        return accepted, rejected

    def _deduplicate_signals(
        self,
        *,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if not self.portfolio_config.deduplicate_by_side:
            return signals, {}

        best_by_side: dict[SignalSide, StrategySignal] = {}
        rejected: dict[str, str] = {}

        for signal in signals:
            existing = best_by_side.get(signal.side)
            if existing is None:
                best_by_side[signal.side] = signal
                continue

            better = self._better_signal(
                existing=existing,
                challenger=signal,
            )
            if better is signal:
                rejected[existing.strategy_name] = f"deduplicated_by_side:{signal.strategy_name}"
                best_by_side[signal.side] = signal
            else:
                rejected[signal.strategy_name] = f"deduplicated_by_side:{existing.strategy_name}"

        accepted = list(best_by_side.values())
        accepted.sort(key=lambda item: (-item.confidence, -item.score, item.strategy_name))
        return accepted, rejected

    def _apply_category_limits(
        self,
        *,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        limits = self.portfolio_config.max_signals_per_category
        if not limits:
            return signals, {}

        grouped: dict[StrategyCategory, list[StrategySignal]] = {}
        for signal in signals:
            grouped.setdefault(signal.category, []).append(signal)

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for category, category_signals in grouped.items():
            limit = limits.get(category)
            ordered = sorted(
                category_signals,
                key=lambda item: (-item.confidence, -item.score, item.strategy_name),
            )

            if limit is None:
                accepted.extend(ordered)
                continue

            accepted.extend(ordered[:limit])
            for signal in ordered[limit:]:
                rejected[signal.strategy_name] = f"category_limit_reached:{category.value}"

        accepted.sort(key=lambda item: (-item.confidence, -item.score, item.strategy_name))
        return accepted, rejected

    def _apply_volatility_throttle(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
        context: StrategyContext | None,
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if not self.portfolio_config.volatility_throttle_enabled:
            return signals, {}

        if context is None:
            return signals, {}

        volatility_zscore = context.get_feature("volatility_zscore")
        if volatility_zscore is None:
            volatility_zscore = context.get_feature("realized_volatility_zscore")

        if not isinstance(volatility_zscore, (int, float)):
            return signals, {}

        value = float(volatility_zscore)
        threshold = self.portfolio_config.volatility_throttle_threshold

        if value <= threshold:
            return signals, {}

        limit = self.portfolio_config.high_volatility_max_signals_per_symbol
        ordered = sorted(
            signals,
            key=lambda item: (-item.confidence, -item.score, item.strategy_name),
        )

        accepted = ordered[:limit]
        rejected = {
            signal.strategy_name: "volatility_throttled"
            for signal in ordered[limit:]
        }
        return accepted, rejected

    def _apply_symbol_limits(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        limit = self.portfolio_config.max_signals_per_symbol
        if limit < 1:
            raise PortfolioCoordinationError("max_signals_per_symbol must be >= 1")

        current_count = self.state.portfolio.get_signal_count(symbol)
        available_slots = max(0, limit - current_count)

        ordered = sorted(
            signals,
            key=lambda item: (-item.confidence, -item.score, item.strategy_name),
        )

        accepted = ordered[:available_slots]
        rejected = {
            signal.strategy_name: "symbol_signal_limit_reached"
            for signal in ordered[available_slots:]
        }

        return accepted, rejected

    def _apply_exposure_buckets(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        limits = self.portfolio_config.exposure_bucket_limits
        if not limits:
            return signals, {}

        snapshot = self.state.portfolio.snapshot
        symbol_exposure = snapshot.symbol_exposure

        accepted: list[StrategySignal] = []
        rejected: dict[str, str] = {}

        for signal in signals:
            bucket = self._resolve_exposure_bucket(symbol)
            if bucket is None:
                accepted.append(signal)
                continue

            current = int(symbol_exposure.get(bucket, 0.0))
            limit = limits.get(bucket)
            if limit is None:
                accepted.append(signal)
                continue

            if current >= limit:
                rejected[signal.strategy_name] = f"exposure_bucket_limit:{bucket}"
                continue

            accepted.append(signal)

        return accepted, rejected

    def _apply_correlation_guard(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> tuple[list[StrategySignal], dict[str, str]]:
        if not self.portfolio_config.correlation_guard_enabled:
            return signals, {}

        snapshot = self.state.portfolio.snapshot
        correlated_group = None

        for group_name, symbols in snapshot.correlation_groups.items():
            if symbol in symbols:
                correlated_group = (group_name, set(symbols))
                break

        if correlated_group is None:
            return signals, {}

        group_name, symbols_in_group = correlated_group

        active_group_states = {
            active_symbol: self.state.get_symbol_state(active_symbol)
            for active_symbol in self.state.symbols.keys()
            if active_symbol in symbols_in_group and active_symbol != symbol
        }

        active_group_states = {
            key: value
            for key, value in active_group_states.items()
            if value is not None and value.last_signal is not None and value.last_signal.is_active
        }

        if not active_group_states:
            return signals, {}

        if self.portfolio_config.enable_correlation_direction_conflict:
            accepted: list[StrategySignal] = []
            rejected: dict[str, str] = {}

            for signal in signals:
                conflict_found = False
                for _, state in active_group_states.items():
                    last_signal = state.last_signal
                    if last_signal is None:
                        continue

                    if last_signal.side != signal.side:
                        rejected[signal.strategy_name] = f"correlation_direction_conflict:{group_name}"
                        conflict_found = True
                        break

                if not conflict_found:
                    accepted.append(signal)

            if accepted:
                return accepted, rejected
            return [], rejected

        best_signal = max(signals, key=lambda item: (item.confidence, item.score))
        rejected = {
            signal.strategy_name: f"correlation_guard:{group_name}"
            for signal in signals
            if signal is not best_signal
        }
        return [best_signal], rejected

    def _merge_similar_signals(
        self,
        signals: list[StrategySignal],
    ) -> list[StrategySignal]:
        if not signals:
            return []

        grouped: dict[tuple[SignalSide, str], list[StrategySignal]] = {}

        for signal in signals:
            key = (signal.side, signal.setup_type.value)
            grouped.setdefault(key, []).append(signal)

        merged: list[StrategySignal] = []
        for group_signals in grouped.values():
            if len(group_signals) == 1:
                merged.append(group_signals[0])
                continue

            primary = max(group_signals, key=lambda item: (item.confidence, item.score))

            merged_signal = StrategySignal(
                symbol=primary.symbol,
                side=primary.side,
                strategy_name="PortfolioCoordinator",
                category=primary.category,
                timeframe=primary.timeframe,
                setup_type=primary.setup_type,
                timestamp=primary.timestamp,
                confidence=max(signal.confidence for signal in group_signals),
                score=max(signal.score for signal in group_signals),
                strength=primary.strength,
                confidence_grade=primary.confidence_grade,
                status=primary.status,
                trigger_type=primary.trigger_type,
                origin=primary.origin,
                priority=self._highest_priority(group_signals),
                reasons=self._merge_reasons(group_signals),
                confirmations=self._merge_confirmations(group_signals),
                source_features=self._merge_source_features(group_signals),
                combined_from=[signal.strategy_name for signal in group_signals],
                conflicts=self._merge_conflicts(group_signals),
                filter_results=self._merge_filter_results(group_signals),
                entry_plan=primary.entry_plan,
                exit_plan=primary.exit_plan,
                invalidation_plan=primary.invalidation_plan,
                execution_plan=primary.execution_plan,
                regime=primary.regime,
                metadata={
                    "merged_by": "portfolio_coordinator",
                    "merged_count": len(group_signals),
                    "primary_strategy": primary.strategy_name,
                },
            )
            merged_signal.validate()
            merged.append(merged_signal)

        merged.sort(key=lambda item: (-item.confidence, -item.score, item.strategy_name))
        return merged

    def _update_state_after_acceptance(
        self,
        *,
        symbol: str,
        signals: list[StrategySignal],
    ) -> None:
        symbol_state = self.state.get_or_create_symbol_state(symbol)

        for signal in signals:
            self.state.update_signal(signal, active=True)
            self.state.portfolio.increment_signal_count(symbol)
            symbol_state.remember_signal(signal)

            side_cooldown_seconds = self.portfolio_config.side_cooldown_seconds
            if side_cooldown_seconds > 0:
                symbol_state.add_side_cooldown(
                    side=signal.side.value,
                    seconds=side_cooldown_seconds,
                    reason=f"side_cooldown_after_{signal.side.value}_signal",
                )

        cooldown_seconds = self.portfolio_config.symbol_cooldown_seconds
        if cooldown_seconds > 0:
            symbol_state.add_cooldown(
                strategy_name="__symbol__",
                seconds=cooldown_seconds,
                reason="portfolio_coordinator_symbol_cooldown",
            )

    def _is_symbol_on_cooldown(
        self,
        *,
        symbol: str,
        now: datetime,
    ) -> bool:
        symbol_state = self.state.get_symbol_state(symbol)
        if symbol_state is None:
            return False

        cooldown = symbol_state.cooldowns.get("__symbol__")
        if cooldown is None:
            return False

        return cooldown.is_active(now)

    def _is_side_on_cooldown(
        self,
        *,
        symbol: str,
        side: SignalSide,
        now: datetime,
    ) -> bool:
        symbol_state = self.state.get_symbol_state(symbol)
        if symbol_state is None:
            return False
        return symbol_state.is_side_on_cooldown(side.value, now)

    def _better_signal(
        self,
        *,
        existing: StrategySignal,
        challenger: StrategySignal,
    ) -> StrategySignal:
        existing_override = self.portfolio_config.priority_overrides.get(existing.strategy_name, 0)
        challenger_override = self.portfolio_config.priority_overrides.get(challenger.strategy_name, 0)

        existing_key = (
            existing_override,
            existing.confidence,
            existing.score,
            self._priority_rank(existing.priority),
        )
        challenger_key = (
            challenger_override,
            challenger.confidence,
            challenger.score,
            self._priority_rank(challenger.priority),
        )
        return challenger if challenger_key > existing_key else existing

    def _priority_rank(self, priority: SignalPriority) -> int:
        ranking = {
            SignalPriority.LOW: 1,
            SignalPriority.MEDIUM: 2,
            SignalPriority.HIGH: 3,
            SignalPriority.CRITICAL: 4,
        }
        return ranking.get(priority, 0)

    def _highest_priority(self, signals: list[StrategySignal]) -> SignalPriority:
        priorities = [signal.priority for signal in signals]
        if not priorities:
            return SignalPriority.MEDIUM
        return max(priorities, key=self._priority_rank)

    def _resolve_exposure_bucket(self, symbol: str) -> str | None:
        symbol_upper = symbol.upper()

        if "BTC" in symbol_upper:
            return "btc"
        if "ETH" in symbol_upper:
            return "eth"
        return "alts"

    def _merge_reasons(self, signals: list[StrategySignal]) -> list[str]:
        merged: list[str] = []
        for signal in signals:
            for reason in signal.reasons:
                if reason not in merged:
                    merged.append(reason)
        return merged

    def _merge_confirmations(self, signals: list[StrategySignal]) -> list[str]:
        merged: list[str] = []
        for signal in signals:
            for confirmation in signal.confirmations:
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

    def _merge_conflicts(self, signals: list[StrategySignal]):
        merged = []
        seen: set[tuple[str, str, str]] = set()

        for signal in signals:
            for conflict in signal.conflicts:
                key = (
                    conflict.source,
                    conflict.conflict_type.value,
                    conflict.message,
                )
                if key in seen:
                    continue
                merged.append(conflict)
                seen.add(key)

        return merged

    def _merge_filter_results(self, signals: list[StrategySignal]):
        merged = []
        seen: set[tuple[str, str, str | None]] = set()

        for signal in signals:
            for result in signal.filter_results:
                key = (result.name, result.decision.value, result.reason)
                if key in seen:
                    continue
                merged.append(result)
                seen.add(key)

        return merged