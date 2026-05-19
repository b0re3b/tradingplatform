# tests/strategy/test_strategy_registry_and_state.py

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from strategy.config import StrategyConfig
from strategy.enums import (
    FeatureSource,
    MarketRegime,
    SetupType,
    SignalSide,
    SignalStatus,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import (
    StrategyRegistrationError,
    StrategyStateError,
    UnsupportedStrategyError,
    ValidationError,
)
from strategy.models import (
    FeatureSnapshot,
    StrategyContext,
    StrategyEvaluation,
    StrategySignal,
    utcnow,
)
from strategy.registry import StrategyRegistry
from strategy.state import (
    SignalState,
    StrategyContextStore,
    StrategyCooldownState,
    StrategyMetricsState,
    StrategyRuntimeState,
)

from conftest import DummyStrategy, make_runtime_config


# =============================================================================
# Local strategy subclasses for registry index coverage
# =============================================================================


class OpenInterestDummyStrategy(DummyStrategy):
    category = StrategyCategory.OPEN_INTEREST
    default_setup_type = SetupType.OI_CONFIRMATION


class HybridDummyStrategy(DummyStrategy):
    category = StrategyCategory.HYBRID
    default_setup_type = SetupType.HYBRID


# =============================================================================
# Local helpers
# =============================================================================


def _names(strategies: list[Any]) -> list[str]:
    return [strategy.strategy_name for strategy in strategies]


def _make_registry(
    *,
    config: StrategyConfig,
    event_bus: Any | None = None,
    scheduler: Any | None = None,
) -> StrategyRegistry:
    return StrategyRegistry(
        config=config,
        event_bus=event_bus,
        scheduler=scheduler,
    )


def _make_strategy(
    *,
    strategy_cls: type[DummyStrategy] = DummyStrategy,
    name: str,
    category: StrategyCategory = StrategyCategory.ORDERFLOW,
    config: StrategyConfig,
    make_definition,
    mock_event_bus,
    mock_scheduler,
    symbols: list[str] | None = None,
    timeframes: list[Timeframe] | None = None,
    regimes: list[MarketRegime] | None = None,
    required_features: tuple[str, ...] = ("orderflow_imbalance",),
    priority: int = 10,
    enabled: bool = True,
) -> DummyStrategy:
    runtime = make_runtime_config(
        enabled=enabled,
        symbols=symbols if symbols is not None else ["BTCUSDT"],
        timeframes=timeframes
        if timeframes is not None
        else [Timeframe.M1, Timeframe.M5, Timeframe.M15],
        allowed_regimes=regimes if regimes is not None else [MarketRegime.UNKNOWN],
    )
    definition = make_definition(
        name=name,
        category=category,
        runtime=runtime,
        required_features=required_features,
        priority=priority,
    )
    config.upsert_strategy(definition)

    return strategy_cls(
        config=config,
        event_bus=mock_event_bus,
        scheduler=mock_scheduler,
        definition=definition,
    )


def _context_with_features(
    make_context,
    make_feature,
    *,
    symbol: str = "BTCUSDT",
    timeframe: Timeframe = Timeframe.M1,
    feature_names: tuple[str, ...] = ("orderflow_imbalance",),
    source: FeatureSource = FeatureSource.ORDERFLOW,
) -> StrategyContext:
    return make_context(
        symbol=symbol,
        timeframe=timeframe,
        features=[
            make_feature(
                name=name,
                source=source,
                symbol=symbol,
            )
            for name in feature_names
        ],
    )


# =============================================================================
# StrategyRegistry: lifecycle and registration
# =============================================================================


class TestStrategyRegistryLifecycle:
    def test_registry_initial_state(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
            scheduler=mock_scheduler,
        )

        assert registry.is_empty()
        assert registry.count() == 0
        assert registry.list_all() == []
        assert registry.list_names() == []
        assert registry.categories() == []
        assert registry.features() == []
        assert not registry.is_registered
        assert not registry.is_started

    def test_register_marks_registry_registered(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        registry.register()

        assert registry.is_registered
        assert registry.subscriptions_count == 0

    @pytest.mark.asyncio()
    async def test_start_and_stop_emit_lifecycle_events(
        self,
        strategy_config: StrategyConfig,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        await registry.start()

        assert registry.is_started
        assert registry.is_registered
        assert mock_event_bus.topic_emitted("strategy.registry.started")

        await registry.stop()

        assert not registry.is_started
        assert not registry.is_registered
        assert mock_event_bus.topic_emitted("strategy.registry.stopped")

    @pytest.mark.asyncio()
    async def test_start_stop_without_event_bus_is_safe(
        self,
        strategy_config: StrategyConfig,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        await registry.start()
        await registry.stop()

        assert not registry.is_started


class TestStrategyRegistryRegistration:
    def test_register_strategy_indexes_everything(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        registry.register_strategy(dummy_strategy)

        assert registry.count() == 1
        assert not registry.is_empty()
        assert registry.get(dummy_strategy.strategy_name) is dummy_strategy
        assert registry.require(dummy_strategy.strategy_name) is dummy_strategy
        assert registry.has(dummy_strategy.strategy_name)

        assert registry.list_all() == [dummy_strategy]
        assert registry.list_names() == [dummy_strategy.strategy_name]
        assert registry.list_by_category(StrategyCategory.ORDERFLOW) == [dummy_strategy]
        assert registry.list_by_timeframe(Timeframe.M1) == [dummy_strategy]
        assert registry.list_by_feature("orderflow_imbalance") == [dummy_strategy]
        assert registry.list_by_symbol("BTCUSDT") == [dummy_strategy]
        assert registry.list_by_regime(MarketRegime.TRENDING_UP) == [dummy_strategy]

        assert registry.categories() == [StrategyCategory.ORDERFLOW]
        assert registry.features() == ["orderflow_imbalance"]

        assert mock_event_bus.nowait_topic_emitted(
            "strategy.registry.strategy_registered"
        )
        payload = mock_event_bus.nowait_payloads(
            "strategy.registry.strategy_registered"
        )[0]
        assert payload["strategy_name"] == dummy_strategy.strategy_name
        assert payload["category"] == "orderflow"
        assert payload["total"] == 1

    def test_register_strategy_can_skip_event_emit(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        registry.register_strategy(dummy_strategy, emit_event=False)

        assert registry.count() == 1
        assert not mock_event_bus.nowait_topic_emitted(
            "strategy.registry.strategy_registered"
        )

    def test_register_duplicate_without_replace_raises(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        registry.register_strategy(dummy_strategy)

        with pytest.raises(StrategyRegistrationError, match="already registered"):
            registry.register_strategy(dummy_strategy)

    def test_register_duplicate_with_replace_rebuilds_indexes(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )

        first = _make_strategy(
            name="replace_me",
            category=StrategyCategory.ORDERFLOW,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            symbols=["BTCUSDT"],
            timeframes=[Timeframe.M1],
            required_features=("orderflow_imbalance",),
            priority=30,
        )
        second = _make_strategy(
            strategy_cls=OpenInterestDummyStrategy,
            name="replace_me",
            category=StrategyCategory.OPEN_INTEREST,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            symbols=["ETHUSDT"],
            timeframes=[Timeframe.H1],
            required_features=("open_interest",),
            priority=5,
        )

        registry.register_strategy(first)
        registry.register_strategy(second, replace=True)

        assert registry.count() == 1
        assert registry.get("replace_me") is second
        assert registry.list_by_category(StrategyCategory.ORDERFLOW) == []
        assert registry.list_by_category(StrategyCategory.OPEN_INTEREST) == [second]
        assert registry.list_by_timeframe(Timeframe.M1) == []
        assert registry.list_by_timeframe(Timeframe.H1) == [second]
        assert registry.list_by_feature("orderflow_imbalance") == []
        assert registry.list_by_feature("open_interest") == [second]
        assert registry.list_by_symbol("ETHUSDT") == [second]

        btc_context = _context_with_features(
            make_context,
            make_feature,
            symbol="BTCUSDT",
            timeframe=Timeframe.M1,
            feature_names=("orderflow_imbalance",),
        )
        eth_context = _context_with_features(
            make_context,
            make_feature,
            symbol="ETHUSDT",
            timeframe=Timeframe.H1,
            feature_names=("open_interest",),
            source=FeatureSource.OPEN_INTEREST,
        )

        assert registry.select(context=btc_context) == []
        assert registry.select(context=eth_context) == [second]

    def test_register_many_keeps_priority_order(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        slow = _make_strategy(
            name="slow",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            priority=50,
        )
        fast = _make_strategy(
            name="fast",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            priority=10,
        )

        registry.register_many([slow, fast])

        assert _names(registry.list_all()) == ["fast", "slow"]
        assert registry.list_names() == ["fast", "slow"]

    def test_register_many_stops_at_first_duplicate(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        first = _make_strategy(
            name="first",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
        )
        duplicate = _make_strategy(
            name="first",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
        )
        second = _make_strategy(
            name="second",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
        )

        with pytest.raises(StrategyRegistrationError):
            registry.register_many([first, duplicate, second])

        assert registry.count() == 1
        assert registry.has("first")
        assert not registry.has("second")

    def test_unregister_strategy_removes_all_indexes_and_emits_event(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )
        registry.register_strategy(dummy_strategy)

        removed = registry.unregister_strategy(dummy_strategy.strategy_name)

        assert removed is dummy_strategy
        assert registry.count() == 0
        assert registry.get(dummy_strategy.strategy_name) is None
        assert registry.list_by_category(StrategyCategory.ORDERFLOW) == []
        assert registry.list_by_timeframe(Timeframe.M1) == []
        assert registry.list_by_feature("orderflow_imbalance") == []
        assert registry.categories() == []
        assert registry.features() == []

        assert mock_event_bus.nowait_topic_emitted(
            "strategy.registry.strategy_unregistered"
        )
        payload = mock_event_bus.nowait_payloads(
            "strategy.registry.strategy_unregistered"
        )[0]
        assert payload["strategy_name"] == dummy_strategy.strategy_name
        assert payload["total"] == 0

    @pytest.mark.parametrize("name", ["missing", "", "   "])
    def test_unregister_unknown_or_empty_strategy_raises(
        self,
        strategy_config: StrategyConfig,
        name: str,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        with pytest.raises(StrategyRegistrationError):
            registry.unregister_strategy(name)

    def test_clear_removes_all_registry_state(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )
        registry.register_strategy(dummy_strategy)

        registry.clear()

        assert registry.is_empty()
        assert registry.count() == 0
        assert registry.list_all() == []
        assert registry.categories() == []
        assert registry.features() == []
        assert registry.summary()["total"] == 0
        assert mock_event_bus.nowait_topic_emitted("strategy.registry.cleared")

    def test_get_and_require_validation(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
    ) -> None:
        registry = _make_registry(config=strategy_config)
        registry.register_strategy(dummy_strategy)

        assert registry.get("") is None
        assert registry.get("   ") is None
        assert registry.get("missing") is None

        with pytest.raises(StrategyRegistrationError):
            registry.require("")

        with pytest.raises(UnsupportedStrategyError):
            registry.require("missing")


# =============================================================================
# StrategyRegistry: selection
# =============================================================================


class TestStrategyRegistrySelection:
    def test_select_returns_only_context_applicable_strategies(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        selected_strategy = _make_strategy(
            name="selected",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            symbols=["BTCUSDT"],
            timeframes=[Timeframe.M1],
            required_features=("orderflow_imbalance",),
            priority=20,
        )
        wrong_symbol = _make_strategy(
            name="wrong_symbol",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            symbols=["ETHUSDT"],
            timeframes=[Timeframe.M1],
            required_features=("orderflow_imbalance",),
            priority=10,
        )
        missing_feature = _make_strategy(
            name="missing_feature",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            symbols=["BTCUSDT"],
            timeframes=[Timeframe.M1],
            required_features=("cvd_delta",),
            priority=5,
        )

        registry.register_many([selected_strategy, wrong_symbol, missing_feature])

        context = _context_with_features(
            make_context,
            make_feature,
            symbol="BTCUSDT",
            timeframe=Timeframe.M1,
            feature_names=("orderflow_imbalance",),
        )

        assert registry.select(context=context) == [selected_strategy]

    def test_select_by_category_source_and_changed_features(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        orderflow = _make_strategy(
            name="orderflow_selected",
            category=StrategyCategory.ORDERFLOW,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            required_features=("orderflow_imbalance",),
            priority=20,
        )
        open_interest = _make_strategy(
            strategy_cls=OpenInterestDummyStrategy,
            name="oi_selected",
            category=StrategyCategory.OPEN_INTEREST,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            required_features=("open_interest",),
            priority=10,
        )

        registry.register_many([orderflow, open_interest])

        orderflow_context = _context_with_features(
            make_context,
            make_feature,
            feature_names=("orderflow_imbalance",),
            source=FeatureSource.ORDERFLOW,
        )
        oi_context = _context_with_features(
            make_context,
            make_feature,
            feature_names=("open_interest",),
            source=FeatureSource.OPEN_INTEREST,
        )

        assert registry.select(
            context=orderflow_context,
            categories={StrategyCategory.ORDERFLOW},
            changed_features={"orderflow_imbalance"},
            source=FeatureSource.ORDERFLOW,
        ) == [orderflow]

        assert registry.select(
            context=oi_context,
            categories={StrategyCategory.OPEN_INTEREST},
            changed_features={"open_interest"},
            source=FeatureSource.OPEN_INTEREST,
        ) == [open_interest]

    def test_select_falls_back_to_union_when_strict_intersection_is_empty(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        by_category = _make_strategy(
            name="by_category",
            category=StrategyCategory.ORDERFLOW,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            required_features=("orderflow_imbalance",),
            priority=20,
        )
        by_feature = _make_strategy(
            strategy_cls=OpenInterestDummyStrategy,
            name="by_feature",
            category=StrategyCategory.OPEN_INTEREST,
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            required_features=("open_interest",),
            priority=10,
        )

        registry.register_many([by_category, by_feature])

        context = _context_with_features(
            make_context,
            make_feature,
            feature_names=("orderflow_imbalance", "open_interest"),
        )

        selected = registry.select(
            context=context,
            categories={StrategyCategory.ORDERFLOW},
            changed_features={"open_interest"},
        )

        assert _names(selected) == ["by_feature", "by_category"]

    def test_select_skips_disabled_unless_include_disabled(
        self,
        strategy_config: StrategyConfig,
        make_definition,
        mock_event_bus,
        mock_scheduler,
        make_context,
        make_feature,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        disabled = _make_strategy(
            name="disabled",
            config=strategy_config,
            make_definition=make_definition,
            mock_event_bus=mock_event_bus,
            mock_scheduler=mock_scheduler,
            enabled=False,
            required_features=("orderflow_imbalance",),
        )

        registry.register_strategy(disabled)

        context = _context_with_features(
            make_context,
            make_feature,
            feature_names=("orderflow_imbalance",),
        )

        assert registry.select(context=context) == []
        assert registry.select(context=context, include_disabled=True) == [disabled]

    def test_select_for_event_requires_event_name(
        self,
        strategy_config: StrategyConfig,
        strategy_context: StrategyContext,
    ) -> None:
        registry = _make_registry(config=strategy_config)

        with pytest.raises(StrategyRegistrationError, match="event_name cannot be empty"):
            registry.select_for_event(
                context=strategy_context,
                event_name="",
            )

    def test_select_for_event_delegates_to_select(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        strategy_context: StrategyContext,
    ) -> None:
        registry = _make_registry(config=strategy_config)
        registry.register_strategy(dummy_strategy)

        selected = registry.select_for_event(
            context=strategy_context,
            event_name="analytics.orderflow.updated",
            categories={StrategyCategory.ORDERFLOW},
            changed_features={"orderflow_imbalance"},
            source=FeatureSource.ORDERFLOW,
        )

        assert selected == [dummy_strategy]

    def test_registry_selection_does_not_evaluate_or_emit_signals(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
        strategy_context: StrategyContext,
        mock_event_bus,
    ) -> None:
        registry = _make_registry(
            config=strategy_config,
            event_bus=mock_event_bus,
        )
        registry.register_strategy(dummy_strategy)

        selected = registry.select(context=strategy_context)

        assert selected == [dummy_strategy]
        assert dummy_strategy.generate_calls == 0
        assert not mock_event_bus.topic_emitted("signal.generated")
        assert not mock_event_bus.nowait_topic_emitted("signal.generated")

    def test_summary_contains_indexes(
        self,
        strategy_config: StrategyConfig,
        dummy_strategy,
    ) -> None:
        registry = _make_registry(config=strategy_config)
        registry.register()
        registry.register_strategy(dummy_strategy, emit_event=False)

        summary = registry.summary()

        assert summary["total"] == 1
        assert summary["strategies"] == [dummy_strategy.strategy_name]
        assert summary["by_category"]["orderflow"] == [dummy_strategy.strategy_name]
        assert dummy_strategy.strategy_name in summary["by_timeframe"]["1m"]
        assert summary["feature_index"]["orderflow_imbalance"] == [
            dummy_strategy.strategy_name
        ]
        assert summary["by_symbol"]["BTCUSDT"] == [dummy_strategy.strategy_name]
        assert summary["registered"] is True
        assert summary["started"] is False


# =============================================================================
# SignalState
# =============================================================================


class TestSignalState:
    def test_signal_state_requires_positive_history_size(self) -> None:
        with pytest.raises(ValidationError):
            SignalState(max_history_size=0)

    def test_remember_indexes_active_signal(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()

        state.remember(strategy_signal)

        signal_id = strategy_signal.signal_id
        assert state.get_by_signal_id(signal_id) is strategy_signal
        assert state.get_active() == [strategy_signal]
        assert state.get_active_for_symbol(strategy_signal.symbol) == [strategy_signal]
        assert state.get_active_for_strategy(strategy_signal.strategy_name) == [
            strategy_signal
        ]
        assert state.get_last_for_symbol(strategy_signal.symbol) is strategy_signal
        assert (
            state.get_last_for_strategy(strategy_signal.strategy_name)
            is strategy_signal
        )
        assert (
            state.get_last_for_symbol_side(strategy_signal.symbol, strategy_signal.side)
            is strategy_signal
        )
        assert list(state.history) == [strategy_signal]
        assert state.updated_at is not None

    def test_remember_terminal_signal_is_not_active_by_default(
        self,
        make_signal,
    ) -> None:
        state = SignalState()
        signal = make_signal(status=SignalStatus.REJECTED)

        state.remember(signal)

        assert state.get_active() == []
        assert list(state.rejected_signals) == [signal]
        assert state.get_last_for_symbol(signal.symbol) is signal
        assert state.get_by_signal_id(signal.signal_id) is signal

    @pytest.mark.parametrize(
        ("status", "bucket_name"),
        [
            (SignalStatus.REJECTED, "rejected_signals"),
            (SignalStatus.EXPIRED, "expired_signals"),
            (SignalStatus.CONFIRMED, "confirmed_signals"),
            (SignalStatus.EXECUTED, "executed_signals"),
            (SignalStatus.FAILED, "failed_signals"),
            (SignalStatus.CANCELLED, "cancelled_signals"),
        ],
    )
    def test_status_buckets(
        self,
        make_signal,
        status: SignalStatus,
        bucket_name: str,
    ) -> None:
        state = SignalState()
        signal = make_signal(status=status)

        state.remember(signal)

        bucket = getattr(state, bucket_name)
        assert list(bucket) == [signal]

    def test_find_signal_priority(
        self,
        make_signal,
    ) -> None:
        state = SignalState()

        by_id = make_signal(
            symbol="BTCUSDT",
            strategy_name="by_id",
            side=SignalSide.LONG,
            metadata={"test_case": "by_id"},
        )
        by_symbol_side = make_signal(
            symbol="ETHUSDT",
            strategy_name="by_symbol_side",
            side=SignalSide.SHORT,
            metadata={"test_case": "by_symbol_side"},
        )
        by_symbol = make_signal(
            symbol="SOLUSDT",
            strategy_name="by_symbol",
            side=SignalSide.LONG,
            metadata={"test_case": "by_symbol"},
        )
        by_strategy = make_signal(
            symbol="XRPUSDT",
            strategy_name="by_strategy",
            side=SignalSide.LONG,
            metadata={"test_case": "by_strategy"},
        )

        for signal in [by_id, by_symbol_side, by_symbol, by_strategy]:
            state.remember(signal)

        assert (
            state.find_signal(
                signal_id=by_id.signal_id,
                symbol="ETHUSDT",
                strategy_name="by_symbol_side",
                side=SignalSide.SHORT,
            )
            is by_id
        )
        assert (
            state.find_signal(
                symbol="ETHUSDT",
                side=SignalSide.SHORT,
            )
            is by_symbol_side
        )
        assert state.find_signal(symbol="SOLUSDT") is by_symbol
        assert state.find_signal(strategy_name="by_strategy") is by_strategy
        assert state.find_signal(signal_id="missing") is None

    def test_mark_status_by_signal_id_reindexes_and_adds_reason_metadata(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()
        state.remember(strategy_signal)

        updated = state.mark_status_by_signal_id(
            strategy_signal.signal_id,
            status=SignalStatus.REJECTED,
            reason="risk_blocked",
            metadata={"risk_code": "max_drawdown"},
        )

        assert updated is strategy_signal
        assert strategy_signal.status is SignalStatus.REJECTED
        assert "risk_blocked" in strategy_signal.reasons
        assert strategy_signal.metadata["risk_code"] == "max_drawdown"
        assert state.get_active() == []
        assert list(state.rejected_signals)[-1] is strategy_signal

    def test_mark_status_by_missing_signal_id_returns_none(self) -> None:
        state = SignalState()

        assert (
            state.mark_status_by_signal_id(
                "missing",
                status=SignalStatus.REJECTED,
            )
            is None
        )

    def test_mark_status_from_payload_by_id(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()
        state.remember(strategy_signal)

        updated = state.mark_status_from_payload(
            payload={
                "signal_id": strategy_signal.signal_id,
                "reason": "confirmed_by_risk",
                "metadata": {"risk_manager": "unit"},
            },
            status=SignalStatus.CONFIRMED,
            default_reason="confirmed",
        )

        assert updated is strategy_signal
        assert strategy_signal.status is SignalStatus.CONFIRMED
        assert "confirmed_by_risk" in strategy_signal.reasons
        assert strategy_signal.metadata["risk_manager"] == "unit"
        assert (
            strategy_signal.metadata["last_status_event_payload"]["signal_id"]
            == strategy_signal.signal_id
        )
        assert state.get_active() == [strategy_signal]

    def test_mark_status_from_payload_falls_back_to_symbol_side(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()
        state.remember(strategy_signal)

        updated = state.mark_status_from_payload(
            payload={
                "symbol": strategy_signal.symbol,
                "side": strategy_signal.side.value,
            },
            status=SignalStatus.EXECUTED,
            default_reason="order_filled",
        )

        assert updated is strategy_signal
        assert strategy_signal.status is SignalStatus.EXECUTED
        assert "order_filled" in strategy_signal.reasons
        assert state.get_active() == []
        assert list(state.executed_signals)[-1] is strategy_signal

    def test_mark_status_from_payload_returns_none_for_unknown_signal(self) -> None:
        state = SignalState()

        updated = state.mark_status_from_payload(
            payload={
                "symbol": "BTCUSDT",
                "side": "long",
            },
            status=SignalStatus.REJECTED,
            default_reason="not_found",
        )

        assert updated is None

    def test_mark_rejected_and_expired_helpers(
        self,
        make_signal,
    ) -> None:
        state = SignalState()
        rejected = make_signal(strategy_name="rejected_signal")
        expired = make_signal(strategy_name="expired_signal")

        state.remember(rejected)
        state.remember(expired)

        state.mark_rejected(rejected, reason="blocked")
        state.mark_expired(expired, reason="ttl")

        assert rejected.status is SignalStatus.REJECTED
        assert expired.status is SignalStatus.EXPIRED
        assert "blocked" in rejected.reasons
        assert "ttl" in expired.reasons
        assert rejected not in state.get_active()
        assert expired not in state.get_active()

    def test_remove_active_by_signal_and_key(
        self,
        make_signal,
    ) -> None:
        state = SignalState()
        first = make_signal(strategy_name="first")
        second = make_signal(strategy_name="second")

        state.remember(first)
        state.remember(second)

        state.remove_active(first)
        assert first not in state.get_active()
        assert second in state.get_active()

        state.remove_active_by_key(
            symbol=second.symbol,
            strategy_name=second.strategy_name,
            side=second.side,
        )
        assert state.get_active() == []

    def test_history_list_filters_and_limits(
        self,
        make_signal,
    ) -> None:
        state = SignalState()
        first = make_signal(
            symbol="BTCUSDT",
            strategy_name="first",
            timestamp=utcnow() - timedelta(seconds=10),
        )
        second = make_signal(
            symbol="ETHUSDT",
            strategy_name="second",
            timestamp=utcnow(),
        )

        state.remember(first)
        state.remember(second)

        assert state.history_list(limit=1) == [second]
        assert state.history_list(symbol="BTCUSDT") == [first]
        assert state.history_list(strategy_name="second") == [second]
        assert state.history_list(signal_id=first.signal_id) == [first]

    def test_history_is_bounded(
        self,
        make_signal,
    ) -> None:
        state = SignalState(max_history_size=2)

        first = make_signal(strategy_name="first")
        second = make_signal(strategy_name="second")
        third = make_signal(strategy_name="third")

        state.remember(first)
        state.remember(second)
        state.remember(third)

        assert list(state.history) == [second, third]

    def test_prune_inactive_removes_terminal_active_entries(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()
        state.remember(strategy_signal, active=True)

        strategy_signal.status = SignalStatus.EXECUTED
        state.prune_inactive()

        assert state.get_active() == []

    def test_prune_older_than_rebuilds_indexes(
        self,
        make_signal,
    ) -> None:
        state = SignalState()
        old = make_signal(
            strategy_name="old",
            timestamp=utcnow() - timedelta(days=1),
        )
        fresh = make_signal(
            strategy_name="fresh",
            timestamp=utcnow(),
        )

        state.remember(old)
        state.remember(fresh)

        removed = state.prune_older_than(utcnow() - timedelta(minutes=1))

        assert removed["history"] == 1
        assert list(state.history) == [fresh]
        assert state.get_by_signal_id(old.signal_id) is None
        assert state.get_by_signal_id(fresh.signal_id) is fresh
        assert state.get_active() == [fresh]

    def test_clear_symbol_removes_all_symbol_related_state(
        self,
        make_signal,
    ) -> None:
        state = SignalState()

        btc = make_signal(symbol="BTCUSDT", strategy_name="btc")
        eth = make_signal(symbol="ETHUSDT", strategy_name="eth")

        state.remember(btc)
        state.remember(eth)

        state.clear_symbol("BTCUSDT")

        assert state.get_by_signal_id(btc.signal_id) is None
        assert state.get_last_for_symbol("BTCUSDT") is None
        assert state.history_list(symbol="BTCUSDT") == []
        assert state.get_by_signal_id(eth.signal_id) is eth

    def test_clear_and_summary(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = SignalState()
        state.remember(strategy_signal)

        summary = state.summary()

        assert summary["active"] == 1
        assert summary["signal_by_id"] == 1
        assert summary["history"] == 1
        assert summary["symbols"] == [strategy_signal.symbol]
        assert summary["strategies"] == [strategy_signal.strategy_name]

        state.clear()

        assert state.summary()["active"] == 0
        assert state.summary()["history"] == 0
        assert state.get_active() == []


# =============================================================================
# StrategyContextStore
# =============================================================================


class TestStrategyContextStore:
    def test_set_context_updates_context_and_features(
        self,
        strategy_context: StrategyContext,
    ) -> None:
        store = StrategyContextStore()

        store.set_context(strategy_context)

        assert store.get_context(strategy_context.symbol) is strategy_context
        assert store.updated_at is not None
        assert store.updated_at_by_symbol[strategy_context.symbol] is not None

        for feature_name, feature in strategy_context.feature_map.items():
            assert store.get_feature(strategy_context.symbol, feature_name) is feature

    def test_put_feature_updates_existing_context(
        self,
        make_context,
        make_feature,
    ) -> None:
        store = StrategyContextStore()
        context = make_context(features=[])

        store.set_context(context)

        snapshot = make_feature(
            name="cvd_delta",
            symbol=context.symbol,
            freshness_seconds=30,
        )
        store.put_feature(snapshot)

        assert store.get_feature(context.symbol, "cvd_delta") is snapshot
        assert context.get_feature("cvd_delta") == snapshot.value
        assert context.freshness_map["cvd_delta"] == 30
        assert context.timestamp == snapshot.timestamp

    def test_build_context_from_store_features(
        self,
        make_feature,
    ) -> None:
        store = StrategyContextStore()
        snapshot = make_feature(name="cvd_delta", symbol="BTCUSDT")

        store.put_feature(snapshot)

        context = store.build_context(
            "BTCUSDT",
            timeframe=Timeframe.M5,
        )

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M5
        assert context.get_feature("cvd_delta") == snapshot.value
        assert store.get_context("BTCUSDT") is context

    def test_build_context_rejects_empty_symbol(self) -> None:
        store = StrategyContextStore()

        with pytest.raises(StrategyStateError, match="symbol cannot be empty"):
            store.build_context("")

    def test_prune_stale_features_removes_expired_features(
        self,
        make_context,
        make_feature,
    ) -> None:
        store = StrategyContextStore()
        expired = make_feature(
            name="expired_feature",
            timestamp=utcnow() - timedelta(seconds=300),
            freshness_seconds=10,
        )
        fresh = make_feature(
            name="fresh_feature",
            timestamp=utcnow(),
            freshness_seconds=60,
        )
        context = make_context(features=[expired, fresh])

        store.set_context(context)

        removed = store.prune_stale_features()

        assert removed == 1
        assert store.get_feature("BTCUSDT", "expired_feature") is None
        assert store.get_feature("BTCUSDT", "fresh_feature") is fresh
        assert "expired_feature" not in context.feature_map

    def test_remove_symbol_and_clear(
        self,
        strategy_context: StrategyContext,
    ) -> None:
        store = StrategyContextStore()
        store.set_context(strategy_context)

        store.remove_symbol(strategy_context.symbol)

        assert store.get_context(strategy_context.symbol) is None
        assert store.features.get(strategy_context.symbol) is None

        store.set_context(strategy_context)
        store.clear()

        assert store.contexts == {}
        assert store.features == {}
        assert store.regimes == {}

    def test_validate_detects_context_key_mismatch(
        self,
        strategy_context: StrategyContext,
    ) -> None:
        store = StrategyContextStore()
        store.contexts["ETHUSDT"] = strategy_context

        with pytest.raises(ValidationError):
            store.validate()

    def test_summary(self, strategy_context: StrategyContext) -> None:
        store = StrategyContextStore()
        store.set_context(strategy_context)

        summary = store.summary()

        assert summary["contexts"] == 1
        assert summary["features_symbols"] == 1
        assert summary["features_total"] == len(strategy_context.feature_map)


# =============================================================================
# StrategyCooldownState
# =============================================================================


class TestStrategyCooldownState:
    def test_add_strategy_and_side_cooldowns(self) -> None:
        state = StrategyCooldownState()

        state.add_strategy_cooldown(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
            seconds=30,
            reason="recent_signal",
        )
        state.add_side_cooldown(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            seconds=30,
            reason="side_signal_accepted",
        )

        assert state.is_strategy_blocked(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
        )
        assert state.is_side_blocked(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
        )
        assert state.updated_at is not None

    def test_non_positive_cooldowns_are_ignored(self) -> None:
        state = StrategyCooldownState()

        state.add_strategy_cooldown(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
            seconds=0,
        )
        state.add_side_cooldown(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
            seconds=-1,
        )

        assert not state.is_strategy_blocked(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
        )
        assert not state.is_side_blocked(
            symbol="BTCUSDT",
            side=SignalSide.LONG,
        )

    def test_expired_cooldowns_are_not_active(self) -> None:
        state = StrategyCooldownState()
        now = utcnow()

        state.add_strategy_cooldown(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
            seconds=1,
        )
        state.add_side_cooldown(
            symbol="BTCUSDT",
            side=SignalSide.SHORT,
            seconds=1,
        )

        future = now + timedelta(seconds=10)

        assert not state.is_strategy_blocked(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
            now=future,
        )
        assert not state.is_side_blocked(
            symbol="BTCUSDT",
            side=SignalSide.SHORT,
            now=future,
        )

    def test_validate_accepts_valid_cooldowns(self) -> None:
        state = StrategyCooldownState()
        state.add_strategy_cooldown(
            symbol="BTCUSDT",
            strategy_name="cvd_divergence",
            seconds=30,
        )

        state.validate()


# =============================================================================
# StrategyMetricsState
# =============================================================================


class TestStrategyMetricsState:
    def test_record_passed_and_failed_evaluations(
        self,
        make_signal,
        make_evaluation,
    ) -> None:
        metrics = StrategyMetricsState()

        passed_signal = make_signal(strategy_name="passed_strategy")
        passed = make_evaluation(
            signal=passed_signal,
            strategy_name=passed_signal.strategy_name,
            symbol=passed_signal.symbol,
            passed=True,
            score=passed_signal.score,
            confidence=passed_signal.confidence,
        )
        failed = make_evaluation(
            signal=None,
            strategy_name="failed_strategy",
            symbol="BTCUSDT",
            passed=False,
        )

        metrics.record_evaluation(passed)
        metrics.record_evaluation(failed)

        assert metrics.evaluations_total == 2
        assert metrics.evaluations_passed == 1
        assert metrics.evaluations_failed == 1
        assert metrics.signals_generated == 1
        assert metrics.evaluations_by_strategy["passed_strategy"] == 1
        assert metrics.evaluations_by_strategy["failed_strategy"] == 1
        assert metrics.signals_by_strategy["passed_strategy"] == 1
        assert metrics.signals_by_symbol["BTCUSDT"] == 1
        assert metrics.pass_rate == 0.5

    def test_record_signal_status_metrics(
        self,
        make_signal,
    ) -> None:
        metrics = StrategyMetricsState()

        rejected = make_signal(status=SignalStatus.REJECTED)
        expired = make_signal(status=SignalStatus.EXPIRED)

        metrics.record_rejected_signal(rejected)
        metrics.record_expired_signal(expired)
        metrics.record_applicability_skip(strategy_name="cvd_divergence")
        metrics.record_error(strategy_name="cvd_divergence")

        assert metrics.signals_rejected == 1
        assert metrics.signals_expired == 1
        assert metrics.applicability_skipped == 1
        assert metrics.errors_total == 1
        assert metrics.errors_by_strategy["cvd_divergence"] == 1
        assert metrics.updated_at is not None

    def test_reset_and_summary(
        self,
        make_signal,
        make_evaluation,
    ) -> None:
        metrics = StrategyMetricsState()
        signal = make_signal()
        evaluation = make_evaluation(signal=signal, passed=True)

        metrics.record_evaluation(evaluation)

        summary = metrics.summary()

        assert summary["evaluations_total"] == 1
        assert summary["evaluations_passed"] == 1
        assert summary["signals_generated"] == 1
        assert summary["signals_by_category"]["orderflow"] == 1

        metrics.reset()

        assert metrics.evaluations_total == 0
        assert metrics.signals_generated == 0
        assert metrics.summary()["evaluations_total"] == 0


# =============================================================================
# StrategyRuntimeState
# =============================================================================


class TestStrategyRuntimeState:
    def test_runtime_state_validates_empty_symbol_sets(self) -> None:
        state = StrategyRuntimeState()
        state.validate()

        state.blocked_symbols.add("")

        with pytest.raises(ValidationError):
            state.validate()

        state.blocked_symbols.clear()
        state.active_symbols.add(" ")

        with pytest.raises(ValidationError):
            state.validate()

    def test_update_context_tracks_active_symbol(
        self,
        strategy_context: StrategyContext,
    ) -> None:
        state = StrategyRuntimeState()

        state.update_context(strategy_context)

        assert state.contexts.get_context(strategy_context.symbol) is strategy_context
        assert strategy_context.symbol in state.active_symbols
        assert state.updated_at is not None

    def test_build_context_tracks_active_symbol(
        self,
        make_feature,
    ) -> None:
        state = StrategyRuntimeState()
        feature = make_feature(name="cvd_delta", symbol="BTCUSDT")
        state.contexts.put_feature(feature)

        context = state.build_context(
            "BTCUSDT",
            timeframe=Timeframe.M5,
        )

        assert context.symbol == "BTCUSDT"
        assert context.timeframe is Timeframe.M5
        assert context.get_feature("cvd_delta") == feature.value
        assert "BTCUSDT" in state.active_symbols

    def test_update_signal_tracks_signal_and_symbol(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = StrategyRuntimeState()

        state.update_signal(strategy_signal)

        assert state.signals.get_by_signal_id(strategy_signal.signal_id) is strategy_signal
        assert strategy_signal.symbol in state.active_symbols
        assert state.updated_at is not None

    def test_update_evaluation_records_metrics_and_signal(
        self,
        strategy_signal: StrategySignal,
        make_evaluation,
    ) -> None:
        state = StrategyRuntimeState()
        evaluation = make_evaluation(
            signal=strategy_signal,
            strategy_name=strategy_signal.strategy_name,
            symbol=strategy_signal.symbol,
            passed=True,
        )

        state.update_evaluation(evaluation)

        assert state.metrics.evaluations_total == 1
        assert state.metrics.evaluations_passed == 1
        assert state.metrics.signals_generated == 1
        assert state.signals.get_by_signal_id(strategy_signal.signal_id) is strategy_signal
        assert strategy_signal.symbol in state.active_symbols

    def test_update_failed_evaluation_does_not_store_signal(
        self,
        make_evaluation,
    ) -> None:
        state = StrategyRuntimeState()
        evaluation = make_evaluation(
            signal=None,
            strategy_name="failed_strategy",
            symbol="BTCUSDT",
            passed=False,
        )

        state.update_evaluation(evaluation)

        assert state.metrics.evaluations_total == 1
        assert state.metrics.evaluations_failed == 1
        assert state.signals.summary()["history"] == 0
        assert state.active_symbols == set()

    def test_mark_signal_status_by_signal_id(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_signal(strategy_signal)

        updated = state.mark_signal_status(
            signal_id=strategy_signal.signal_id,
            status=SignalStatus.CONFIRMED,
            reason="risk_confirmed",
            metadata={"risk": "ok"},
        )

        assert updated is strategy_signal
        assert strategy_signal.status is SignalStatus.CONFIRMED
        assert "risk_confirmed" in strategy_signal.reasons
        assert strategy_signal.metadata["risk"] == "ok"
        assert state.signals.get_active() == [strategy_signal]

    def test_mark_signal_status_returns_none_for_missing_signal(self) -> None:
        state = StrategyRuntimeState()

        assert (
            state.mark_signal_status(
                signal_id="missing",
                status=SignalStatus.REJECTED,
            )
            is None
        )

    def test_mark_signal_status_by_symbol_strategy_side(
        self,
        strategy_signal: StrategySignal,
    ) -> None:
        state = StrategyRuntimeState()
        state.update_signal(strategy_signal)

        updated = state.mark_signal_status(
            symbol=strategy_signal.symbol,
            strategy_name=strategy_signal.strategy_name,
            side=strategy_signal.side,
            status=SignalStatus.EXECUTED,
            reason="filled",
        )

        assert updated is strategy_signal
        assert strategy_signal.status is SignalStatus.EXECUTED
        assert "filled" in strategy_signal.reasons
        assert state.signals.get_active() == []

    def test_runtime_state_summary(
        self,
        strategy_context: StrategyContext,
        strategy_signal: StrategySignal,
        make_evaluation,
    ) -> None:
        state = StrategyRuntimeState()

        state.update_context(strategy_context)
        state.update_signal(strategy_signal)
        state.update_evaluation(
            make_evaluation(
                signal=strategy_signal,
                strategy_name=strategy_signal.strategy_name,
                symbol=strategy_signal.symbol,
                passed=True,
            )
        )

        summary = state.summary()

        assert summary["signals"]["history"] >= 1
        assert summary["contexts"]["contexts"] == 1
        assert summary["metrics"]["evaluations_total"] == 1
        assert strategy_signal.symbol in summary["active_symbols"]

    def test_runtime_state_has_no_event_bus_side_effects(
        self,
        strategy_signal: StrategySignal,
        mock_event_bus,
    ) -> None:
        state = StrategyRuntimeState()

        state.update_signal(strategy_signal)
        state.mark_signal_status(
            signal_id=strategy_signal.signal_id,
            status=SignalStatus.REJECTED,
            reason="blocked",
        )

        assert mock_event_bus.emit_calls == 0
        assert mock_event_bus.nowait_calls == 0
        assert not mock_event_bus.topic_emitted("signal.generated")
        assert not mock_event_bus.nowait_topic_emitted("signal.generated")