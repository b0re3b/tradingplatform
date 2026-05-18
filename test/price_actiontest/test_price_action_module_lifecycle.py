# tests/analytics/price_action/test_price_action_module_lifecycle.py

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import pytest

from core.event_bus import EventPriority
from core.scheduler import Scheduler

from analytics.price_action import FairValueGapAnalyzer, FairValueGapConfig


pytestmark = pytest.mark.usefixtures("event_bus")


# ---------------------------------------------------------------------------
# Local test utilities
# ---------------------------------------------------------------------------

class ExplodingObject:
    def __str__(self) -> str:
        return "exploding-object-as-string"


class NonSerializableEnum(str, Enum):
    VALUE = "enum_value"


@dataclass(slots=True)
class NestedDataclassPayload:
    created_at: datetime
    priority: EventPriority
    enum_value: NonSerializableEnum
    values: set[int]
    nested: dict[str, Any]


class EmitRecorder:
    """
    Async EventBus.emit replacement.

    It records every attempted emit and can fail/reject selected topics. This
    lets lifecycle tests verify _emit_event and scoped handlers without starting
    the EventBus worker.
    """

    def __init__(
        self,
        *,
        accepted: bool = True,
        fail: bool = False,
        reject_topics: set[str] | None = None,
    ) -> None:
        self.accepted = accepted
        self.fail = fail
        self.reject_topics = reject_topics or set()
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
            raise RuntimeError("forced EventBus.emit failure")

        if topic in self.reject_topics:
            return False

        return self.accepted


def build_fvg(
    *,
    event_bus,
    scheduler: Scheduler | None = None,
    config: FairValueGapConfig | None = None,
    exchange: str = "  binance  ",
    market_type: str = " usdm_futures ",
    symbol: str = "  btcusdt  ",
    exchange_symbol: str | None = " BTCUSDT ",
    timeframe: str = " 1m ",
) -> FairValueGapAnalyzer:
    return FairValueGapAnalyzer(
        exchange=exchange,
        market_type=market_type,
        symbol=symbol,
        exchange_symbol=exchange_symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        scheduler=scheduler,
        config=config
        or FairValueGapConfig(
            max_candles=500,
            max_gaps_per_layer=100,
            max_events=200,
            min_gap_pct_internal=0.0,
            min_gap_pct_external=0.0,
            merge_distance_pct_internal=0.0,
            merge_distance_pct_external=0.0,
            min_impulse_body_ratio=0.0,
            respected_reaction_threshold_pct=0.0001,
            invalidation_close_buffer_pct=0.0,
            retest_window_bars=8,
            publish_snapshots=False,
            snapshot_interval_seconds=None,
            market_candle_topic="market.candle.closed",
            market_candles_topic="market.candles.updated",
            require_event_scope=True,
        ),
    )


def snapshot_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    metadata = snapshot.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


# ---------------------------------------------------------------------------
# Constructor / config hardening
# ---------------------------------------------------------------------------

def test_constructor_normalizes_full_futures_scope_and_topics(event_bus):
    config = FairValueGapConfig(
        max_candles=500,
        max_gaps_per_layer=100,
        max_events=200,
        event_namespace=" .analytics.price_action.fair_value_gap. ",
        market_candle_topic=" .market.candle.closed. ",
        market_candles_topic=" .market.candles.updated. ",
        require_event_scope=True,
    )

    analyzer = build_fvg(
        event_bus=event_bus,
        config=config,
        exchange="  BINANCE  ",
        market_type=" USDM_FUTURES ",
        symbol="  ethusdt  ",
        exchange_symbol=" ETHUSDT ",
        timeframe=" 5m ",
    )

    assert analyzer.exchange == "binance"
    assert analyzer.market_type == "usdm_futures"
    assert analyzer.symbol == "ETHUSDT"
    assert analyzer.exchange_symbol == "ETHUSDT"
    assert analyzer.timeframe == "5m"
    assert analyzer.key == ("binance", "usdm_futures", "ETHUSDT", "5m")
    assert analyzer.scope_payload == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "ETHUSDT",
        "exchange_symbol": "ETHUSDT",
        "timeframe": "5m",
        "key": ["binance", "usdm_futures", "ETHUSDT", "5m"],
    }
    assert analyzer.config.event_namespace == "analytics.price_action.fair_value_gap"
    assert analyzer.config.market_candle_topic == "market.candle.closed"
    assert analyzer.config.market_candles_topic == "market.candles.updated"
    assert analyzer.config.require_event_scope is True


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("", "symbol must not be empty"),
        ("   ", "symbol must not be empty"),
    ],
)
def test_constructor_rejects_empty_symbol(event_bus, symbol, expected):
    with pytest.raises(ValueError, match=expected):
        build_fvg(event_bus=event_bus, symbol=symbol)


def test_constructor_current_contract_defaults_empty_timeframe_to_1m(event_bus):
    """
    Current production contract: empty timeframe is normalized to DEFAULT_TIMEFRAME.

    If the project decides to make timeframe strict, replace this test with a
    ValueError expectation and update normalize_timeframe() accordingly.
    """
    analyzer = build_fvg(event_bus=event_bus, timeframe="   ")

    assert analyzer.timeframe == "1m"
    assert analyzer.key == ("binance", "usdm_futures", "BTCUSDT", "1m")


def test_constructor_rejects_non_core_event_bus():
    with pytest.raises(TypeError, match="event_bus must be an instance"):
        FairValueGapAnalyzer(
            symbol="BTCUSDT",
            timeframe="1m",
            event_bus=object(),  # type: ignore[arg-type]
        )


def test_constructor_rejects_non_core_scheduler(event_bus):
    with pytest.raises(TypeError, match="scheduler must be an instance"):
        FairValueGapAnalyzer(
            symbol="BTCUSDT",
            timeframe="1m",
            event_bus=event_bus,
            scheduler=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("event_namespace", " ... ", "event_namespace must not be empty"),
        ("snapshot_interval_seconds", 0.0, "snapshot_interval_seconds must be > 0"),
        ("snapshot_job_timeout_seconds", 0.0, "snapshot_job_timeout_seconds must be > 0"),
        ("snapshot_job_max_retries", -1, "snapshot_job_max_retries must be >= 0"),
        ("snapshot_job_retry_delay_seconds", -0.01, "snapshot_job_retry_delay_seconds must be >= 0"),
        ("market_candle_topic", " . ", "market_candle_topic must not be empty"),
        ("market_candles_topic", " . ", "market_candles_topic must not be empty"),
    ],
)
def test_base_config_rejects_dangerous_infrastructure_values(
    event_bus,
    fair_value_gap_config,
    field_name,
    value,
    expected,
):
    config = replace(fair_value_gap_config, **{field_name: value})

    with pytest.raises(ValueError, match=expected):
        build_fvg(event_bus=event_bus, config=config)


def test_base_config_coerces_raw_event_priority(event_bus, fair_value_gap_config):
    config = replace(
        fair_value_gap_config,
        event_priority=EventPriority.HIGH.value,  # type: ignore[arg-type]
    )

    analyzer = build_fvg(event_bus=event_bus, config=config)

    assert analyzer.config.event_priority is EventPriority.HIGH


@pytest.mark.xfail(
    reason=(
        "Known futures-only hardening target: normalize_market_type() currently "
        "accepts arbitrary strings. Make spot invalid in production, then remove xfail."
    )
)
def test_constructor_rejects_spot_market_type_for_futures_only_package(
    event_bus,
    fair_value_gap_config,
):
    with pytest.raises(ValueError, match="market_type"):
        build_fvg(
            event_bus=event_bus,
            config=fair_value_gap_config,
            market_type="spot",
        )


# ---------------------------------------------------------------------------
# Registration / unregistration / shutdown vulnerabilities
# ---------------------------------------------------------------------------

def test_register_is_idempotent_and_does_not_duplicate_data_layer_subscriptions(
    event_bus,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    analyzer.register()
    first_subscriptions = list(analyzer._subscriptions)

    analyzer.register()
    second_subscriptions = list(analyzer._subscriptions)

    assert analyzer._registered is True
    assert len(first_subscriptions) == 2
    assert second_subscriptions == first_subscriptions
    assert [sub.pattern for sub in second_subscriptions] == [
        "market.candle.closed",
        "market.candles.updated",
    ]


def test_register_uses_scoped_wrappers_for_data_layer_candle_topics(
    event_bus,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    analyzer.register()

    subscriptions = {sub.pattern: sub for sub in analyzer._subscriptions}

    assert subscriptions["market.candle.closed"].handler == analyzer._on_candle_event_scoped
    assert subscriptions["market.candles.updated"].handler == analyzer._on_candles_event_scoped


def test_register_with_market_subscription_disabled_does_not_subscribe(
    event_bus,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        subscribe_market_candles=False,
        publish_snapshots=False,
        snapshot_interval_seconds=None,
    )
    analyzer = build_fvg(event_bus=event_bus, config=config)

    analyzer.register()

    assert analyzer._registered is True
    assert analyzer._subscriptions == []
    assert analyzer._scheduled_job_ids == []


def test_unregister_clears_internal_state_even_when_event_bus_unsubscribe_fails(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    analyzer.register()

    assert len(analyzer._subscriptions) == 2

    def broken_unsubscribe(_subscription):
        raise RuntimeError("forced unsubscribe failure")

    monkeypatch.setattr(event_bus, "unsubscribe", broken_unsubscribe)

    analyzer.unregister()

    assert analyzer._registered is False
    assert analyzer._subscriptions == []
    assert analyzer._scheduled_job_ids == []


def test_unregister_is_safe_when_called_before_register(event_bus, fair_value_gap_config):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    analyzer.unregister()

    assert analyzer._registered is False
    assert analyzer._subscriptions == []
    assert analyzer._scheduled_job_ids == []


@pytest.mark.asyncio
async def test_shutdown_is_idempotent_and_blocks_reregister(event_bus, fair_value_gap_config):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    analyzer.register()

    await analyzer.shutdown()
    await analyzer.shutdown()

    assert analyzer._shutdown is True
    assert analyzer._registered is False
    assert analyzer._subscriptions == []

    with pytest.raises(RuntimeError, match="already shut down"):
        analyzer.register()


# ---------------------------------------------------------------------------
# Scheduler / snapshot job hardening
# ---------------------------------------------------------------------------

def test_snapshot_job_is_not_registered_without_scheduler_even_when_requested(
    event_bus,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        publish_snapshots=True,
        snapshot_interval_seconds=30.0,
    )
    analyzer = build_fvg(event_bus=event_bus, config=config, scheduler=None)

    analyzer.register()

    assert analyzer._registered is True
    assert analyzer._scheduled_job_ids == []


def test_snapshot_job_is_registered_once_and_not_duplicated_by_second_register(
    event_bus,
    scheduler,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        publish_snapshots=True,
        snapshot_interval_seconds=30.0,
        snapshot_job_timeout_seconds=3.0,
        snapshot_job_max_retries=2,
        snapshot_job_retry_delay_seconds=0.5,
    )
    analyzer = build_fvg(event_bus=event_bus, scheduler=scheduler, config=config)

    analyzer.register()
    first_job_ids = list(analyzer._scheduled_job_ids)

    analyzer.register()
    second_job_ids = list(analyzer._scheduled_job_ids)

    assert len(first_job_ids) == 1
    assert second_job_ids == first_job_ids
    assert scheduler.get_job(first_job_ids[0]) is not None

    job = scheduler.get_job(first_job_ids[0])
    assert job is not None
    assert "analytics.price_action.fair_value_gap.snapshot" in job.name
    assert "binance.usdm_futures.btcusdt.1m" in job.name


def test_unregister_clears_snapshot_job_ids_even_when_disable_job_fails(
    event_bus,
    scheduler,
    monkeypatch,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        publish_snapshots=True,
        snapshot_interval_seconds=30.0,
    )
    analyzer = build_fvg(event_bus=event_bus, scheduler=scheduler, config=config)
    analyzer.register()

    assert len(analyzer._scheduled_job_ids) == 1

    def broken_disable_job(_job_id: str):
        raise RuntimeError("forced disable failure")

    monkeypatch.setattr(scheduler, "disable_job", broken_disable_job)

    analyzer.unregister()

    assert analyzer._registered is False
    assert analyzer._scheduled_job_ids == []


# ---------------------------------------------------------------------------
# Candle payload extraction / parsing edge cases
# ---------------------------------------------------------------------------

def test_extract_candles_payload_accepts_all_supported_shapes(
    event_bus,
    event_factory,
    candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    candle_0 = candle_factory(0)
    candle_1 = candle_factory(1)

    shapes = [
        event_factory("market.candle.closed", candle_0),
        event_factory("market.candle.closed", {"candle": candle_0}),
        event_factory("market.candles.updated", [candle_0, candle_1]),
        event_factory(
            "market.candles.updated",
            candles_updated_payload([candle_0, candle_1]),
        ),
    ]

    extracted_lengths = [
        len(analyzer._extract_candles_payload(event))
        for event in shapes
    ]

    assert extracted_lengths == [1, 1, 2, 2]


def test_extract_candles_payload_merges_parent_scope_into_child_candles(
    event_bus,
    event_factory,
    candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    child_without_scope = candle_factory(0)
    for key in ("exchange", "market_type", "symbol", "exchange_symbol", "timeframe", "key"):
        child_without_scope.pop(key, None)

    payload = candles_updated_payload([child_without_scope])
    event = event_factory("market.candles.updated", payload)

    extracted = analyzer._extract_candles_payload(event)

    assert len(extracted) == 1
    assert extracted[0]["exchange"] == "binance"
    assert extracted[0]["market_type"] == "usdm_futures"
    assert extracted[0]["symbol"] == "BTCUSDT"
    assert extracted[0]["exchange_symbol"] == "BTCUSDT"
    assert extracted[0]["timeframe"] == "1m"

    parsed = analyzer._parse_candles(extracted)
    assert len(parsed) == 1
    assert parsed[0].key == analyzer.key


def test_extract_candles_payload_filters_mixed_or_malicious_sequences_without_crash(
    event_bus,
    event_factory,
    candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    valid_candle = candle_factory(0)
    valid_without_scope = candle_factory(1)
    for key in ("exchange", "market_type", "symbol", "exchange_symbol", "timeframe", "key"):
        valid_without_scope.pop(key, None)

    event = event_factory(
        "market.candles.updated",
        candles_updated_payload(
            [
                valid_candle,
                "not-a-candle",
                b"bytes",
                123,
                None,
                valid_without_scope,
            ]
        ),
    )

    extracted = analyzer._extract_candles_payload(event)

    assert len(extracted) == 2
    assert extracted[0] == valid_candle
    assert extracted[1]["open"] == valid_without_scope["open"]
    assert extracted[1]["exchange"] == "binance"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        "market-data",
        b"market-data",
        123,
        {"not": "a candle"},
        {"candles": "not-a-list"},
        {"candle": "not-a-mapping"},
    ],
)
def test_extract_candles_payload_returns_empty_for_invalid_shapes(
    event_bus,
    event_factory,
    fair_value_gap_config,
    payload,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    event = event_factory("market.candle.closed", payload)

    assert analyzer._extract_candles_payload(event) == []


@pytest.mark.parametrize(
    ("timestamp", "expected_tz"),
    [
        (datetime(2026, 1, 1, 12, 0), timezone.utc),
        (datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), timezone.utc),
        ("2026-01-01T12:00:00Z", timezone.utc),
        ("2026-01-01T12:00:00+00:00", timezone.utc),
        (1_767_264_000, timezone.utc),
        (1_767_264_000_000, timezone.utc),
    ],
)
def test_parse_candle_accepts_common_timestamp_formats(
    event_bus,
    candle_factory,
    fair_value_gap_config,
    timestamp,
    expected_tz,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    payload = candle_factory(0, timestamp=timestamp)

    parsed = analyzer._parse_candle(payload)

    assert parsed.timestamp.tzinfo is not None
    assert parsed.timestamp.utcoffset() == expected_tz.utcoffset(parsed.timestamp)
    assert parsed.exchange == "binance"
    assert parsed.market_type == "usdm_futures"
    assert parsed.symbol == "BTCUSDT"
    assert parsed.timeframe == "1m"
    assert parsed.key == analyzer.key


@pytest.mark.parametrize(
    "broken_payload",
    [
        {"timestamp": "2026-01-01T00:00:00Z", "high": 1, "low": 0, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "low": 0, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 1, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 1, "low": 0},
        {"timestamp": None, "open": 1, "high": 1, "low": 0, "close": 1},
        {"timestamp": "", "open": 1, "high": 1, "low": 0, "close": 1},
        {"timestamp": object(), "open": 1, "high": 1, "low": 0, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": -1, "high": 1, "low": 0, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 0.5, "low": 0, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 1.5, "close": 1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0, "close": 1, "volume": -1},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0, "close": 1, "index": -1},
    ],
)
def test_parse_candle_rejects_malformed_or_inconsistent_ohlcv(
    event_bus,
    fair_value_gap_config,
    broken_payload,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    with pytest.raises(ValueError, match="invalid candle payload|missing required candle field"):
        analyzer._parse_candle(broken_payload)


def test_parse_candle_rejects_wrong_scope_even_when_ohlcv_is_valid(
    event_bus,
    wrong_scope_candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    wrong_exchange = wrong_scope_candle_factory(0, wrong_exchange=True)
    wrong_market_type = wrong_scope_candle_factory(1, wrong_market_type=True)
    wrong_symbol = wrong_scope_candle_factory(2, wrong_symbol=True)

    for payload in (wrong_exchange, wrong_market_type, wrong_symbol):
        with pytest.raises(ValueError, match="candle scope does not match"):
            analyzer._parse_candle(payload)


def test_parse_candles_skips_wrong_scope_and_malformed_items_without_raising(
    event_bus,
    candle_factory,
    wrong_scope_candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    valid = candle_factory(0)
    wrong_scope = wrong_scope_candle_factory(1, wrong_exchange=True)
    malformed = candle_factory(2)
    malformed["low"] = malformed["high"] + 100.0

    parsed = analyzer._parse_candles(
        [
            valid,
            wrong_scope,
            malformed,
            candle_factory(3),
        ],
        start_index=50,
    )

    assert len(parsed) == 2
    assert [candle.index for candle in parsed] == [50, 53]
    assert all(candle.key == analyzer.key for candle in parsed)


def test_parse_candles_uses_start_index_to_prevent_duplicate_local_indexes(
    event_bus,
    candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    payloads = [
        candle_factory(0, open_=100.0),
        candle_factory(0, open_=101.0),
        candle_factory(0, open_=102.0),
    ]

    parsed = analyzer._parse_candles(payloads, start_index=50)

    assert [candle.index for candle in parsed] == [50, 51, 52]


# ---------------------------------------------------------------------------
# Scoped EventBus wrapper hardening
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scoped_single_candle_handler_processes_matching_closed_candle_and_emits_update(
    event_bus,
    monkeypatch,
    event_factory,
    candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    event = event_factory(
        "market.candle.closed",
        candle_factory(0, is_closed=True),
        source="CandlesCache",
        correlation_id="closed-candle-0",
    )

    await analyzer._on_candle_event_scoped(event)

    metadata = snapshot_metadata(analyzer.snapshot())
    assert metadata["total_candles"] == 1
    assert recorder.calls
    assert recorder.calls[-1]["topic"] == "analytics.price_action.fair_value_gap.updated"
    assert recorder.calls[-1]["correlation_id"] == "closed-candle-0"


@pytest.mark.asyncio
async def test_scoped_single_candle_handler_ignores_wrong_scope_without_emit_or_mutation(
    event_bus,
    monkeypatch,
    event_factory,
    wrong_scope_candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    before = analyzer.snapshot()

    event = event_factory(
        "market.candle.closed",
        wrong_scope_candle_factory(0, wrong_exchange=True),
        source="CandlesCache",
        correlation_id="wrong-exchange",
    )

    await analyzer._on_candle_event_scoped(event)

    assert analyzer.snapshot()["state"] == before["state"]
    assert snapshot_metadata(analyzer.snapshot())["total_candles"] == 0
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_scoped_batch_handler_processes_matching_candles_from_candles_updated_payload(
    event_bus,
    monkeypatch,
    event_factory,
    candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    candles = [candle_factory(0), candle_factory(1), candle_factory(2)]
    event = event_factory(
        "market.candles.updated",
        candles_updated_payload(candles),
        source="CandlesCache",
        correlation_id="batch-1",
    )

    await analyzer._on_candles_event_scoped(event)

    metadata = snapshot_metadata(analyzer.snapshot())
    assert metadata["total_candles"] == 3
    assert recorder.calls
    assert recorder.calls[-1]["topic"] == "analytics.price_action.fair_value_gap.updated"
    assert recorder.calls[-1]["correlation_id"] == "batch-1"


@pytest.mark.asyncio
async def test_scoped_batch_handler_filters_wrong_scope_children_inside_matching_parent_payload(
    event_bus,
    monkeypatch,
    event_factory,
    candle_factory,
    wrong_scope_candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    candles = [
        candle_factory(0),
        wrong_scope_candle_factory(1, wrong_exchange=True),
        wrong_scope_candle_factory(2, wrong_market_type=True),
        candle_factory(3),
    ]

    event = event_factory(
        "market.candles.updated",
        candles_updated_payload(candles),
        source="CandlesCache",
        correlation_id="mixed-scope-batch",
    )

    await analyzer._on_candles_event_scoped(event)

    metadata = snapshot_metadata(analyzer.snapshot())
    assert metadata["total_candles"] == 2
    assert metadata["global_candle_index"] == 2
    assert recorder.calls
    assert recorder.calls[-1]["topic"] == "analytics.price_action.fair_value_gap.updated"


@pytest.mark.asyncio
async def test_scoped_batch_handler_ignores_fully_wrong_scope_payload_without_emit_or_mutation(
    event_bus,
    monkeypatch,
    event_factory,
    wrong_scope_candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    wrong_candles = [
        wrong_scope_candle_factory(0, wrong_exchange=True),
        wrong_scope_candle_factory(1, wrong_exchange=True),
    ]

    payload = candles_updated_payload(
        wrong_candles,
        exchange="bybit",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        timeframe="1m",
    )

    event = event_factory(
        "market.candles.updated",
        payload,
        source="CandlesCache",
        correlation_id="wrong-parent-scope",
    )

    await analyzer._on_candles_event_scoped(event)

    metadata = snapshot_metadata(analyzer.snapshot())
    assert metadata["total_candles"] == 0
    assert recorder.calls == []


@pytest.mark.xfail(
    reason=(
        "Known idempotency hardening target: duplicate delivery of the same "
        "closed candle through market.candle.closed and market.candles.updated "
        "is not globally deduplicated yet."
    )
)
@pytest.mark.asyncio
async def test_duplicate_closed_candle_delivery_does_not_advance_state_twice(
    event_bus,
    monkeypatch,
    event_factory,
    candle_factory,
    candles_updated_payload,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    candle = candle_factory(10, is_closed=True)

    await analyzer._on_candle_event_scoped(
        event_factory(
            "market.candle.closed",
            candle,
            source="CandlesCache",
            correlation_id="dup-single",
        )
    )
    await analyzer._on_candles_event_scoped(
        event_factory(
            "market.candles.updated",
            candles_updated_payload([dict(candle)]),
            source="CandlesCache",
            correlation_id="dup-batch",
        )
    )

    metadata = snapshot_metadata(analyzer.snapshot())
    assert metadata["total_candles"] == 1
    assert metadata["global_candle_index"] == 1


# ---------------------------------------------------------------------------
# Event emitting / serialization vulnerabilities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_event_returns_false_and_does_not_call_bus_when_disabled(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    config = replace(fair_value_gap_config, emit_events=False)
    analyzer = build_fvg(event_bus=event_bus, config=config)
    recorder = EmitRecorder()

    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer._emit_event(
        ".analytics.price_action.test.",
        {"value": 1},
        correlation_id="corr-disabled",
    )

    assert accepted is False
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_emit_event_normalizes_topic_serializes_payload_and_passes_core_metadata(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    payload = {
        "dataclass": NestedDataclassPayload(
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            priority=EventPriority.HIGH,
            enum_value=NonSerializableEnum.VALUE,
            values={3, 1, 2},
            nested={"obj": ExplodingObject()},
        ),
        "datetime": datetime(2026, 1, 1, 13, 0),
        "enum": EventPriority.LOW,
        "set_value": {"a", "b"},
        "object": ExplodingObject(),
    }

    accepted = await analyzer._emit_event(
        ".analytics.price_action.fair_value_gap.custom.",
        payload,
        source="unit-test",
        priority=EventPriority.CRITICAL,
        correlation_id="corr-serialization",
        headers={"test": "yes"},
    )

    assert accepted is True
    assert len(recorder.calls) == 1

    call = recorder.calls[0]
    assert call["topic"] == "analytics.price_action.fair_value_gap.custom"
    assert call["priority"] is EventPriority.CRITICAL
    assert call["source"] == "unit-test"
    assert call["correlation_id"] == "corr-serialization"
    assert call["headers"] == {"test": "yes"}

    emitted_payload = call["payload"]
    assert emitted_payload["dataclass"]["priority"] == EventPriority.HIGH.value
    assert emitted_payload["dataclass"]["enum_value"] == NonSerializableEnum.VALUE.value
    assert sorted(emitted_payload["dataclass"]["values"]) == [1, 2, 3]
    assert emitted_payload["dataclass"]["nested"]["obj"] == "exploding-object-as-string"
    assert emitted_payload["datetime"].endswith("+00:00")
    assert emitted_payload["enum"] == EventPriority.LOW.value
    assert sorted(emitted_payload["set_value"]) == ["a", "b"]
    assert emitted_payload["object"] == "exploding-object-as-string"


@pytest.mark.asyncio
async def test_emit_event_wraps_non_mapping_safe_payload_before_emit(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer._emit_event(
        "analytics.price_action.fair_value_gap.scalar",
        ["unexpected", "list"],  # type: ignore[arg-type]
    )

    assert accepted is True
    assert recorder.calls[0]["payload"] == {"value": ["unexpected", "list"]}


@pytest.mark.asyncio
async def test_emit_event_returns_false_instead_of_leaking_event_bus_exception(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder(fail=True)
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer._emit_event(
        "analytics.price_action.fair_value_gap.explodes",
        {"value": 1},
    )

    assert accepted is False
    assert len(recorder.calls) == 1


@pytest.mark.asyncio
async def test_emit_event_returns_false_when_event_bus_rejects_topic(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder(
        reject_topics={"analytics.price_action.fair_value_gap.rejected"}
    )
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer._emit_event(
        "analytics.price_action.fair_value_gap.rejected",
        {"value": 1},
    )

    assert accepted is False
    assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# Snapshot envelope / topic construction invariants
# ---------------------------------------------------------------------------

def test_snapshot_envelope_never_exposes_raw_dataclasses_or_enums(
    event_bus,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    envelope = analyzer._snapshot_envelope(
        state=NestedDataclassPayload(
            created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
            priority=EventPriority.CRITICAL,
            enum_value=NonSerializableEnum.VALUE,
            values={2, 1},
            nested={"object": ExplodingObject()},
        ),
        metadata={
            "priority": EventPriority.LOW,
            "custom_object": ExplodingObject(),
        },
    )

    assert envelope["exchange"] == "binance"
    assert envelope["market_type"] == "usdm_futures"
    assert envelope["symbol"] == "BTCUSDT"
    assert envelope["exchange_symbol"] == "BTCUSDT"
    assert envelope["timeframe"] == "1m"
    assert envelope["key"] == ["binance", "usdm_futures", "BTCUSDT", "1m"]
    assert envelope["module"] == "FairValueGapAnalyzer"
    assert envelope["state"]["priority"] == EventPriority.CRITICAL.value
    assert envelope["state"]["enum_value"] == NonSerializableEnum.VALUE.value
    assert sorted(envelope["state"]["values"]) == [1, 2]
    assert envelope["state"]["nested"]["object"] == "exploding-object-as-string"
    assert envelope["metadata"]["priority"] == EventPriority.LOW.value
    assert envelope["metadata"]["custom_object"] == "exploding-object-as-string"


def test_public_snapshot_contains_full_scope_and_serialized_config(
    event_bus,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)

    snapshot = analyzer.snapshot()

    assert snapshot["exchange"] == "binance"
    assert snapshot["market_type"] == "usdm_futures"
    assert snapshot["symbol"] == "BTCUSDT"
    assert snapshot["exchange_symbol"] == "BTCUSDT"
    assert snapshot["timeframe"] == "1m"
    assert snapshot["key"] == ["binance", "usdm_futures", "BTCUSDT", "1m"]
    assert isinstance(snapshot["generated_at"], str)
    assert isinstance(snapshot["state"], dict)

    metadata = snapshot_metadata(snapshot)
    assert metadata["total_candles"] == 0
    assert metadata["global_candle_index"] == 0
    assert metadata["config"]["market_candle_topic"] == "market.candle.closed"
    assert metadata["config"]["market_candles_topic"] == "market.candles.updated"
    assert metadata["config"]["require_event_scope"] is True


def test_build_event_name_strips_duplicate_dots_and_allows_namespace_only_event(
    event_bus,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        event_namespace=".analytics.price_action.fair_value_gap.",
    )
    analyzer = build_fvg(event_bus=event_bus, config=config)

    assert (
        analyzer._build_event_name(".updated.")
        == "analytics.price_action.fair_value_gap.updated"
    )
    assert analyzer._build_event_name("...") == "analytics.price_action.fair_value_gap"


def test_subscribe_rejects_empty_pattern_before_touching_event_bus(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    calls: list[Any] = []

    def recording_subscribe(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("event_bus.subscribe must not be called for an empty pattern")

    monkeypatch.setattr(event_bus, "subscribe", recording_subscribe)

    with pytest.raises(ValueError, match="subscription pattern must not be empty"):
        analyzer._subscribe(" ... ", analyzer.on_candle_event)

    assert calls == []