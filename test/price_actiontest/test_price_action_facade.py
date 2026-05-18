# tests/analytics/price_action/test_price_action_facade.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pytest

from core.event_bus import EventPriority

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.models import PriceActionCompositeState, PriceActionScope

try:
    from analytics.price_action.price_action_analyzer import (
        PriceActionAnalyzer,
        PriceActionAnalyzerConfig,
    )
except ImportError:  # pragma: no cover - only for projects that re-export facade
    from analytics.price_action import PriceActionAnalyzer, PriceActionAnalyzerConfig


# ---------------------------------------------------------------------------
# Local test doubles
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FakeChildState(PriceActionScope):
    """
    Minimal scoped child state.

    It intentionally inherits PriceActionScope because PriceActionAnalyzer
    validates child state scope before aggregating it into the composite state.
    """

    last_price: float | None = None
    last_update: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        PriceActionScope.__post_init__(self)

        if self.last_price is not None:
            self.last_price = float(self.last_price)

        if self.last_update is not None and self.last_update.tzinfo is None:
            self.last_update = self.last_update.replace(tzinfo=timezone.utc)

        self.metadata = dict(self.metadata or {})


class FakeChildAnalyzer(BasePriceActionModule[FakeChildState]):
    """
    Minimal child analyzer double.

    It uses the real BasePriceActionModule so facade tests still exercise:
    - constructor DI contract;
    - full futures scope;
    - BasePriceActionConfig validation;
    - snapshot envelope shape;
    - register/reset/shutdown behavior.

    It avoids domain-specific calculations from the real child analyzers.
    """

    def __init__(
        self,
        *,
        exchange: str = "binance",
        market_type: str = "usdm_futures",
        symbol: str,
        exchange_symbol: str = "BTCUSDT",
        timeframe: str,
        event_bus,
        module_name: str,
        price: float | None = None,
        updated_at: datetime | None = None,
        fail_register: bool = False,
        fail_reset: bool = False,
        fail_shutdown: bool = False,
    ) -> None:
        self._requested_module_name = module_name
        self.register_calls = 0
        self.unregister_calls = 0
        self.reset_calls = 0
        self.shutdown_calls = 0
        self.fail_register = fail_register
        self.fail_reset = fail_reset
        self.fail_shutdown = fail_shutdown

        super().__init__(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            scheduler=None,
            config=BasePriceActionConfig(
                emit_events=False,
                publish_snapshots=False,
                subscribe_market_candles=False,
                require_event_scope=True,
                event_namespace=f"analytics.price_action.fake.{module_name}",
            ),
            service_name=f"analytics.price_action.fake.{module_name}",
        )

        # Keep facade-facing names stable and readable.
        self.module_name = module_name
        self._state = FakeChildState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            last_price=price,
            last_update=updated_at,
            metadata={"child_module": module_name},
        )

    def register(self) -> None:
        self.register_calls += 1
        if self.fail_register:
            raise RuntimeError(f"{self.module_name} register failure")
        self._registered = True

    def unregister(self) -> None:
        self.unregister_calls += 1
        self._registered = False
        self._subscriptions.clear()
        self._scheduled_job_ids.clear()

    def reset(self) -> None:
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError(f"{self.module_name} reset failure")

        self._state = FakeChildState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            last_price=None,
            last_update=None,
            metadata={"reset": True, "child_module": self.module_name},
        )

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"{self.module_name} shutdown failure")
        self._registered = False
        self._shutdown = True

    def get_state(self) -> FakeChildState:
        return self._state

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "registered": self._registered,
                "shutdown": self._shutdown,
                "register_calls": self.register_calls,
                "unregister_calls": self.unregister_calls,
                "reset_calls": self.reset_calls,
                "shutdown_calls": self.shutdown_calls,
            },
        )


class EmitRecorder:
    """
    Async EventBus.emit replacement.
    """

    def __init__(self, *, accepted: bool = True, fail: bool = False) -> None:
        self.accepted = accepted
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        priority: EventPriority,
        source: str,
        correlation_id: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> bool:
        self.calls.append(
            {
                "topic": topic,
                "payload": dict(payload),
                "priority": priority,
                "source": source,
                "correlation_id": correlation_id,
                "headers": dict(headers or {}),
            }
        )

        if self.fail:
            raise RuntimeError("forced emit failure")

        return self.accepted


# ---------------------------------------------------------------------------
# Local builders
# ---------------------------------------------------------------------------

def make_facade_config(**overrides: Any) -> PriceActionAnalyzerConfig:
    defaults: dict[str, Any] = {
        "emit_events": True,
        "event_namespace": "analytics.price_action",
        "publish_snapshots": False,
        "snapshot_interval_seconds": None,

        # Facade itself must not subscribe to market candles. Children own
        # market.candle.closed / market.candles.updated subscriptions.
        "subscribe_market_candles": False,
        "require_event_scope": True,

        "auto_register_modules": True,
        "shutdown_child_modules": True,
        "reset_child_modules": True,

        "publish_on_module_update": True,
        "publish_composite_snapshot_on_module_update": False,

        "enable_market_structure": True,
        "enable_support_resistance": True,
        "enable_fair_value_gap": True,
        "enable_liquidity_levels": True,
        "enable_trend": True,

        "market_structure_updated_topic": "analytics.price_action.market_structure.updated",
        "support_resistance_updated_topic": "analytics.price_action.support_resistance.updated",
        "fair_value_gap_updated_topic": "analytics.price_action.fair_value_gap.updated",
        "liquidity_levels_updated_topic": "analytics.price_action.liquidity_levels.updated",
        "trend_updated_topic": "analytics.price_action.trend.updated",
    }
    defaults.update(overrides)
    return PriceActionAnalyzerConfig(**defaults)


def make_child(
    *,
    event_bus,
    module_name: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
    price: float | None = None,
    updated_at: datetime | None = None,
    fail_register: bool = False,
    fail_reset: bool = False,
    fail_shutdown: bool = False,
) -> FakeChildAnalyzer:
    return FakeChildAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        module_name=module_name,
        price=price,
        updated_at=updated_at,
        fail_register=fail_register,
        fail_reset=fail_reset,
        fail_shutdown=fail_shutdown,
    )


def make_child_set(
    event_bus,
    symbol: str,
    timeframe: str,
    *,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
) -> dict[str, FakeChildAnalyzer]:
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    return {
        "market_structure": make_child(
            event_bus=event_bus,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            module_name="market_structure",
            price=100.0,
            updated_at=base_time,
        ),
        "support_resistance": make_child(
            event_bus=event_bus,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            module_name="support_resistance",
            price=101.0,
            updated_at=base_time + timedelta(minutes=1),
        ),
        "fair_value_gap": make_child(
            event_bus=event_bus,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            module_name="fair_value_gap",
            price=102.0,
            updated_at=base_time + timedelta(minutes=2),
        ),
        "liquidity_levels": make_child(
            event_bus=event_bus,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            module_name="liquidity_levels",
            price=103.0,
            updated_at=base_time + timedelta(minutes=3),
        ),
        "trend": make_child(
            event_bus=event_bus,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            exchange_symbol=exchange_symbol,
            timeframe=timeframe,
            module_name="trend",
            price=104.0,
            updated_at=base_time + timedelta(minutes=4),
        ),
    }


def make_facade(
    *,
    event_bus,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    exchange_symbol: str = "BTCUSDT",
    config: PriceActionAnalyzerConfig | None = None,
    children: dict[str, FakeChildAnalyzer] | None = None,
) -> PriceActionAnalyzer:
    children = children or make_child_set(
        event_bus,
        symbol,
        timeframe,
        exchange=exchange,
        market_type=market_type,
        exchange_symbol=exchange_symbol,
    )

    return PriceActionAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        config=config or make_facade_config(),
        market_structure=children.get("market_structure"),
        support_resistance=children.get("support_resistance"),
        fair_value_gap=children.get("fair_value_gap"),
        liquidity_levels=children.get("liquidity_levels"),
        trend=children.get("trend"),
    )


def scoped_payload(
    *,
    module_name: str,
    last_price: float = 100.0,
    exchange: str = "binance",
    market_type: str = "usdm_futures",
    symbol: str = "BTCUSDT",
    exchange_symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": exchange_symbol,
        "timeframe": timeframe,
        "key": [exchange, market_type, symbol, timeframe],
        "state": {
            "exchange": exchange,
            "market_type": market_type,
            "symbol": symbol,
            "exchange_symbol": exchange_symbol,
            "timeframe": timeframe,
            "key": [exchange, market_type, symbol, timeframe],
            "last_price": last_price,
            "metadata": {
                "child_module": module_name,
                "source": "pytest.scoped_payload",
            },
        },
    }
    payload.update(extra)
    return payload


def metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = snapshot.get("metadata")
    assert isinstance(result, dict)
    return result


def emitted_topics(recorder: EmitRecorder) -> list[str]:
    return [call["topic"] for call in recorder.calls]


def assert_payload_scope(payload: Mapping[str, Any]) -> None:
    assert payload["exchange"] == "binance"
    assert payload["market_type"] == "usdm_futures"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["exchange_symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "1m"
    assert payload["key"] == ["binance", "usdm_futures", "BTCUSDT", "1m"]


# ---------------------------------------------------------------------------
# Config / construction vulnerabilities
# ---------------------------------------------------------------------------

class TestPriceActionFacadeConfigAndConstruction:
    def test_config_rejects_auto_register_with_no_enabled_modules(self) -> None:
        with pytest.raises(
            ValueError,
            match="at least one price action module must be enabled",
        ):
            make_facade_config(
                auto_register_modules=True,
                enable_market_structure=False,
                enable_support_resistance=False,
                enable_fair_value_gap=False,
                enable_liquidity_levels=False,
                enable_trend=False,
            ).validate()

    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            (
                "market_structure_updated_topic",
                "market_structure_updated_topic must not be empty",
            ),
            (
                "support_resistance_updated_topic",
                "support_resistance_updated_topic must not be empty",
            ),
            (
                "fair_value_gap_updated_topic",
                "fair_value_gap_updated_topic must not be empty",
            ),
            (
                "liquidity_levels_updated_topic",
                "liquidity_levels_updated_topic must not be empty",
            ),
            (
                "trend_updated_topic",
                "trend_updated_topic must not be empty",
            ),
        ],
    )
    def test_config_rejects_empty_enabled_child_update_topics(
        self,
        field_name: str,
        expected: str,
    ) -> None:
        config = make_facade_config(**{field_name: " . "})

        with pytest.raises(ValueError, match=expected):
            config.validate()

    def test_constructor_uses_full_futures_scope_and_aggregates_latest_child_price(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
        )

        state = facade.get_state()

        assert isinstance(state, PriceActionCompositeState)
        assert state.exchange == "binance"
        assert state.market_type == "usdm_futures"
        assert state.symbol == symbol
        assert state.exchange_symbol == symbol
        assert state.timeframe == timeframe
        assert state.key == ("binance", "usdm_futures", symbol, timeframe)

        assert state.last_price == 104.0
        assert state.last_update == datetime(2026, 1, 1, 12, 4, tzinfo=timezone.utc)
        assert state.metadata["enabled_modules"] == [
            "market_structure",
            "support_resistance",
            "fair_value_gap",
            "liquidity_levels",
            "trend",
        ]

    def test_disabled_module_with_explicit_injected_child_must_not_bypass_enable_flag(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        config = make_facade_config(
            enable_liquidity_levels=False,
            enable_trend=False,
            auto_register_modules=True,
        )

        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            children=children,
        )

        assert "liquidity_levels" not in facade.get_child_analyzers()
        assert "trend" not in facade.get_child_analyzers()
        assert facade.liquidity_levels is None
        assert facade.trend is None

    def test_constructor_rejects_injected_child_with_wrong_exchange_scope(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        children["trend"] = make_child(
            event_bus=event_bus,
            exchange="bybit",
            market_type="usdm_futures",
            symbol=symbol,
            exchange_symbol=symbol,
            timeframe=timeframe,
            module_name="trend",
            price=104.0,
            updated_at=datetime(2026, 1, 1, 12, 4, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match="Child price action module scope mismatch"):
            make_facade(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                children=children,
            )

    def test_constructor_rejects_injected_child_with_wrong_market_type_scope(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        children["fair_value_gap"] = make_child(
            event_bus=event_bus,
            exchange="binance",
            market_type="linear",
            symbol=symbol,
            exchange_symbol=symbol,
            timeframe=timeframe,
            module_name="fair_value_gap",
            price=102.0,
            updated_at=datetime(2026, 1, 1, 12, 2, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match="Child price action module scope mismatch"):
            make_facade(
                event_bus=event_bus,
                symbol=symbol,
                timeframe=timeframe,
                children=children,
            )


# ---------------------------------------------------------------------------
# Registration lifecycle vulnerabilities
# ---------------------------------------------------------------------------

class TestPriceActionFacadeRegistration:
    def test_register_subscribes_to_child_update_topics_and_auto_registers_children_once(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(auto_register_modules=True),
            children=children,
        )

        facade.register()
        first_subscription_patterns = [sub.pattern for sub in facade._subscriptions]

        facade.register()
        second_subscription_patterns = [sub.pattern for sub in facade._subscriptions]

        assert facade._registered is True
        assert first_subscription_patterns == second_subscription_patterns
        assert first_subscription_patterns == [
            "analytics.price_action.market_structure.updated",
            "analytics.price_action.support_resistance.updated",
            "analytics.price_action.fair_value_gap.updated",
            "analytics.price_action.liquidity_levels.updated",
            "analytics.price_action.trend.updated",
        ]

        assert {name: child.register_calls for name, child in children.items()} == {
            "market_structure": 1,
            "support_resistance": 1,
            "fair_value_gap": 1,
            "liquidity_levels": 1,
            "trend": 1,
        }
        assert all(child._registered for child in children.values())

    def test_register_does_not_auto_register_children_when_disabled(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(auto_register_modules=False),
            children=children,
        )

        facade.register()

        assert facade._registered is True
        assert all(child.register_calls == 0 for child in children.values())
        assert metadata(facade.snapshot())["registered_modules"] == []

    def test_register_failure_in_one_child_rolls_back_facade_and_children(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        children["fair_value_gap"].fail_register = True

        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(auto_register_modules=True),
            children=children,
        )

        with pytest.raises(RuntimeError, match="fair_value_gap register failure"):
            facade.register()

        assert facade._registered is False
        assert facade._subscriptions == []

        assert children["market_structure"].register_calls == 1
        assert children["support_resistance"].register_calls == 1
        assert children["market_structure"].unregister_calls == 1
        assert children["support_resistance"].unregister_calls == 1

        assert all(child._registered is False for child in children.values())

    def test_register_only_subscribes_enabled_child_update_topics(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        config = make_facade_config(
            enable_fair_value_gap=False,
            enable_liquidity_levels=False,
        )

        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=config,
            children=children,
        )

        facade.register()

        assert [sub.pattern for sub in facade._subscriptions] == [
            "analytics.price_action.market_structure.updated",
            "analytics.price_action.support_resistance.updated",
            "analytics.price_action.trend.updated",
        ]

        assert children["market_structure"].register_calls == 1
        assert children["support_resistance"].register_calls == 1
        assert children["trend"].register_calls == 1
        assert children["fair_value_gap"].register_calls == 0
        assert children["liquidity_levels"].register_calls == 0


# ---------------------------------------------------------------------------
# Child update handling / EventBus publication vulnerabilities
# ---------------------------------------------------------------------------

class TestPriceActionFacadeChildUpdates:
    @pytest.mark.asyncio
    async def test_invalid_child_update_payload_is_ignored_without_emit_or_state_mutation(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before_counts = dict(facade._child_update_counts)
        before_payloads = dict(facade._last_child_payloads)
        before_version = facade._state_version
        before_state = facade.snapshot()["state"]

        await facade.on_market_structure_updated(
            event_factory(
                "analytics.price_action.market_structure.updated",
                ["not", "a", "mapping"],
                correlation_id="invalid-child",
            )
        )

        assert facade._child_update_counts == before_counts
        assert facade._last_child_payloads == before_payloads
        assert facade._state_version == before_version
        assert facade.snapshot()["state"] == before_state
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_unknown_child_module_update_is_rejected_instead_of_expanding_counters(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before_counts = dict(facade._child_update_counts)
        before_payloads = dict(facade._last_child_payloads)

        await facade._handle_child_update(
            "unknown_module",
            event_factory(
                "analytics.price_action.unknown.updated",
                scoped_payload(
                    module_name="unknown_module",
                    last_price=999.0,
                    symbol=symbol,
                    timeframe=timeframe,
                ),
                correlation_id="unknown-child",
            ),
        )

        assert facade._child_update_counts == before_counts
        assert facade._last_child_payloads == before_payloads
        assert "unknown_module" not in metadata(facade.snapshot())["child_update_counts"]
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_wrong_scope_child_update_is_ignored_without_emit_or_state_mutation(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before_counts = dict(facade._child_update_counts)
        before_payloads = dict(facade._last_child_payloads)
        before_version = facade._state_version
        before_state = facade.snapshot()["state"]

        await facade.on_trend_updated(
            event_factory(
                "analytics.price_action.trend.updated",
                scoped_payload(
                    module_name="trend",
                    exchange="bybit",
                    market_type="usdm_futures",
                    symbol=symbol,
                    exchange_symbol=symbol,
                    timeframe=timeframe,
                    last_price=999.0,
                ),
                correlation_id="wrong-scope-child",
            )
        )

        assert facade._child_update_counts == before_counts
        assert facade._last_child_payloads == before_payloads
        assert facade._state_version == before_version
        assert facade.snapshot()["state"] == before_state
        assert recorder.calls == []

    @pytest.mark.asyncio
    async def test_valid_child_update_updates_counts_payloads_and_publishes_composite_update(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(
                publish_on_module_update=True,
                publish_composite_snapshot_on_module_update=False,
                publish_snapshots=False,
            ),
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        before_version = facade._state_version

        await facade.on_trend_updated(
            event_factory(
                "analytics.price_action.trend.updated",
                scoped_payload(
                    module_name="trend",
                    symbol=symbol,
                    timeframe=timeframe,
                    last_price=111.0,
                    new_signals_count=2,
                ),
                correlation_id="trend-update-1",
            )
        )

        assert facade._child_update_counts["trend"] == 1
        assert facade._last_child_payloads["trend"]["new_signals_count"] == 2
        assert facade._state_version > before_version

        assert emitted_topics(recorder) == ["analytics.price_action.updated"]

        emitted = recorder.calls[0]
        assert emitted["correlation_id"] == "trend-update-1"
        assert emitted["payload"]["updated_module"] == "trend"
        assert emitted["payload"]["source_topic"] == "analytics.price_action.trend.updated"
        assert emitted["payload"]["state_version"] == facade._state_version
        assert_payload_scope(emitted["payload"])

    @pytest.mark.asyncio
    async def test_publish_flags_emit_composite_update_and_snapshot_from_one_child_update(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(
                publish_on_module_update=True,
                publish_composite_snapshot_on_module_update=True,
                publish_snapshots=True,
            ),
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await facade.on_fair_value_gap_updated(
            event_factory(
                "analytics.price_action.fair_value_gap.updated",
                scoped_payload(
                    module_name="fair_value_gap",
                    symbol=symbol,
                    timeframe=timeframe,
                    last_price=108.0,
                ),
                correlation_id="fvg-update",
            )
        )

        assert emitted_topics(recorder) == [
            "analytics.price_action.updated",
            "analytics.price_action.snapshot",
        ]
        assert all(call["correlation_id"] == "fvg-update" for call in recorder.calls)
        assert_payload_scope(recorder.calls[0]["payload"])
        assert_payload_scope(recorder.calls[1]["payload"])

    @pytest.mark.asyncio
    async def test_child_update_does_not_publish_when_publish_on_update_disabled(
        self,
        event_bus,
        monkeypatch,
        event_factory,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(
                publish_on_module_update=False,
                publish_composite_snapshot_on_module_update=False,
                publish_snapshots=False,
            ),
        )
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await facade.on_liquidity_levels_updated(
            event_factory(
                "analytics.price_action.liquidity_levels.updated",
                scoped_payload(
                    module_name="liquidity_levels",
                    symbol=symbol,
                    timeframe=timeframe,
                    last_price=105.0,
                ),
                correlation_id="liq-update",
            )
        )

        assert facade._child_update_counts["liquidity_levels"] == 1
        assert facade._last_child_payloads["liquidity_levels"]["symbol"] == symbol
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# Snapshot / state version vulnerabilities
# ---------------------------------------------------------------------------

class TestPriceActionFacadeSnapshotAndState:
    def test_snapshot_metadata_contains_scope_enabled_registered_counts_and_last_child_modules(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            children=children,
        )

        children["market_structure"]._registered = True
        children["trend"]._registered = True
        facade._child_update_counts["trend"] = 3
        facade._last_child_payloads["trend"] = scoped_payload(
            module_name="trend",
            symbol=symbol,
            timeframe=timeframe,
            last_price=104.0,
        )

        snapshot = facade.snapshot()
        meta = metadata(snapshot)

        assert_payload_scope(snapshot)
        assert_payload_scope(snapshot["state"])

        assert meta["enabled_modules"] == [
            "market_structure",
            "support_resistance",
            "fair_value_gap",
            "liquidity_levels",
            "trend",
        ]
        assert meta["registered_modules"] == ["market_structure", "trend"]
        assert meta["child_update_counts"]["trend"] == 3
        assert meta["last_child_update_modules"] == ["trend"]
        assert snapshot["state"]["last_price"] == 104.0

    def test_repeated_snapshot_calls_do_not_advance_state_version_without_new_data(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)

        first = facade.snapshot()
        second = facade.snapshot()
        third = facade.get_state()

        assert metadata(first)["state_version"] == metadata(second)["state_version"]
        assert third.metadata["state_version"] == metadata(second)["state_version"]
        assert facade._state_version == metadata(first)["state_version"]

    def test_child_state_with_wrong_scope_is_ignored_in_composite_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            children=children,
        )

        children["trend"]._state = FakeChildState(
            exchange="bybit",
            market_type="usdm_futures",
            symbol=symbol,
            exchange_symbol=symbol,
            timeframe=timeframe,
            last_price=999.0,
            last_update=datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc),
        )

        state = facade.get_state()

        assert state.trend is None
        assert state.last_price != 999.0

    @pytest.mark.asyncio
    async def test_publish_composite_update_returns_false_when_event_bus_rejects(
        self,
        event_bus,
        monkeypatch,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder(accepted=False)
        monkeypatch.setattr(event_bus, "emit", recorder)

        accepted = await facade.publish_composite_update(
            updated_module="trend",
            source_topic="analytics.price_action.trend.updated",
            correlation_id="reject-update",
        )

        assert accepted is False
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["topic"] == "analytics.price_action.updated"
        assert_payload_scope(recorder.calls[0]["payload"])

    @pytest.mark.asyncio
    async def test_publish_composite_update_returns_false_when_event_bus_raises(
        self,
        event_bus,
        monkeypatch,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder(fail=True)
        monkeypatch.setattr(event_bus, "emit", recorder)

        accepted = await facade.publish_composite_update(
            updated_module="trend",
            source_topic="analytics.price_action.trend.updated",
            correlation_id="raise-update",
        )

        assert accepted is False
        assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# Reset / shutdown vulnerabilities
# ---------------------------------------------------------------------------

class TestPriceActionFacadeResetAndShutdown:
    def test_reset_clears_facade_counters_and_resets_children_when_enabled(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(reset_child_modules=True),
            children=children,
        )

        facade._child_update_counts["trend"] = 7
        facade._last_child_payloads["trend"] = scoped_payload(
            module_name="trend",
            symbol=symbol,
            timeframe=timeframe,
            last_price=104.0,
        )
        before_version = facade._state_version

        facade.reset()

        assert facade._state_version > before_version
        assert facade._last_child_payloads == {}
        assert facade._child_update_counts == {
            "market_structure": 0,
            "support_resistance": 0,
            "fair_value_gap": 0,
            "liquidity_levels": 0,
            "trend": 0,
        }
        assert all(child.reset_calls == 1 for child in children.values())

    def test_reset_does_not_reset_children_when_disabled_but_clears_facade_state(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(reset_child_modules=False),
            children=children,
        )

        facade._child_update_counts["fair_value_gap"] = 4
        facade._last_child_payloads["fair_value_gap"] = scoped_payload(
            module_name="fair_value_gap",
            symbol=symbol,
            timeframe=timeframe,
            last_price=102.0,
        )

        facade.reset()

        assert all(child.reset_calls == 0 for child in children.values())
        assert facade._last_child_payloads == {}
        assert all(count == 0 for count in facade._child_update_counts.values())

    def test_reset_failure_in_child_does_not_partially_clear_facade_counters(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        children["support_resistance"].fail_reset = True

        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(reset_child_modules=True),
            children=children,
        )

        facade._child_update_counts["trend"] = 9
        facade._last_child_payloads["trend"] = scoped_payload(
            module_name="trend",
            symbol=symbol,
            timeframe=timeframe,
            last_price=104.0,
        )
        before_counts = dict(facade._child_update_counts)
        before_payloads = dict(facade._last_child_payloads)
        before_version = facade._state_version

        with pytest.raises(RuntimeError, match="support_resistance reset failure"):
            facade.reset()

        assert facade._child_update_counts == before_counts
        assert facade._last_child_payloads == before_payloads
        assert facade._state_version == before_version

    @pytest.mark.asyncio
    async def test_reset_and_publish_emits_reset_event_with_scope_and_snapshot(
        self,
        event_bus,
        monkeypatch,
        symbol: str,
        timeframe: str,
    ) -> None:
        facade = make_facade(event_bus=event_bus, symbol=symbol, timeframe=timeframe)
        recorder = EmitRecorder()
        monkeypatch.setattr(event_bus, "emit", recorder)

        await facade.reset_and_publish(correlation_id="reset-corr")

        assert emitted_topics(recorder) == ["analytics.price_action.reset"]
        assert recorder.calls[0]["correlation_id"] == "reset-corr"
        assert recorder.calls[0]["payload"]["state_version"] == facade._state_version
        assert_payload_scope(recorder.calls[0]["payload"])

    @pytest.mark.asyncio
    async def test_shutdown_shuts_down_children_when_enabled_and_blocks_reregister(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(shutdown_child_modules=True),
            children=children,
        )
        facade.register()

        await facade.shutdown()
        await facade.shutdown()

        assert facade._shutdown is True
        assert facade._registered is False
        assert facade._subscriptions == []
        assert all(child.shutdown_calls >= 1 for child in children.values())

        with pytest.raises(RuntimeError, match="already shut down"):
            facade.register()

    @pytest.mark.asyncio
    async def test_shutdown_does_not_shutdown_children_when_disabled(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(shutdown_child_modules=False),
            children=children,
        )

        await facade.shutdown()

        assert facade._shutdown is True
        assert all(child.shutdown_calls == 0 for child in children.values())

    @pytest.mark.asyncio
    async def test_shutdown_continues_when_child_shutdown_fails(
        self,
        event_bus,
        symbol: str,
        timeframe: str,
    ) -> None:
        children = make_child_set(event_bus, symbol, timeframe)
        children["fair_value_gap"].fail_shutdown = True

        facade = make_facade(
            event_bus=event_bus,
            symbol=symbol,
            timeframe=timeframe,
            config=make_facade_config(shutdown_child_modules=True),
            children=children,
        )

        await facade.shutdown()

        assert facade._shutdown is True
        assert children["fair_value_gap"].shutdown_calls == 1
        assert children["trend"].shutdown_calls == 1