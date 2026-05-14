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
from analytics.price_action.base import BasePriceActionConfig


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

    It records every attempted emit and can be configured to fail or reject
    selected topics. This lets lifecycle tests verify _emit_event behavior
    without depending on EventBus worker lifecycle.
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
    symbol: str = "  btcusdt  ",
    timeframe: str = " 1m ",
) -> FairValueGapAnalyzer:
    return FairValueGapAnalyzer(
        symbol=symbol,
        timeframe=timeframe,
        event_bus=event_bus,
        scheduler=scheduler,
        config=config or FairValueGapConfig(
            max_candles=500,
            max_gaps_per_layer=100,
            max_events=200,
            min_gap_pct_internal=0.0,
            min_gap_pct_external=0.0,
            merge_distance_pct_internal=0.0,
            merge_distance_pct_external=0.0,
            min_impulse_body_ratio=0.0,
            publish_snapshots=False,
            snapshot_interval_seconds=None,
        ),
    )


# ---------------------------------------------------------------------------
# Constructor / config hardening
# ---------------------------------------------------------------------------

def test_constructor_normalizes_symbol_timeframe_and_topics(event_bus):
    config = FairValueGapConfig(
        max_candles=500,
        max_gaps_per_layer=100,
        max_events=200,
        event_namespace=" .analytics.price_action.fair_value_gap. ",
        market_candle_topic=" .market.candle. ",
        market_candles_topic=" .market.candles. ",
    )

    analyzer = build_fvg(
        event_bus=event_bus,
        config=config,
        symbol="  ethusdt  ",
        timeframe=" 5m ",
    )

    assert analyzer.symbol == "ETHUSDT"
    assert analyzer.timeframe == "5m"
    assert analyzer.config.event_namespace == "analytics.price_action.fair_value_gap"
    assert analyzer.config.market_candle_topic == "market.candle"
    assert analyzer.config.market_candles_topic == "market.candles"


@pytest.mark.parametrize(
    ("symbol", "timeframe", "expected"),
    [
        ("", "1m", "symbol must not be empty"),
        ("   ", "1m", "symbol must not be empty"),
        ("BTCUSDT", "", "timeframe must not be empty"),
        ("BTCUSDT", "   ", "timeframe must not be empty"),
    ],
)
def test_constructor_rejects_empty_identity(event_bus, symbol, timeframe, expected):
    with pytest.raises(ValueError, match=expected):
        build_fvg(event_bus=event_bus, symbol=symbol, timeframe=timeframe)


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


# ---------------------------------------------------------------------------
# Registration / unregistration / shutdown vulnerabilities
# ---------------------------------------------------------------------------

def test_register_is_idempotent_and_does_not_duplicate_market_subscriptions(
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
        fair_value_gap_config.market_candle_topic,
        fair_value_gap_config.market_candles_topic,
    ]


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
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    candle_0 = candle_factory(0)
    candle_1 = candle_factory(1)

    shapes = [
        event_factory("market.candle", candle_0),
        event_factory("market.candle", {"candle": candle_0}),
        event_factory("market.candles", [candle_0, candle_1]),
        event_factory("market.candles", {"candles": [candle_0, candle_1]}),
    ]

    extracted_lengths = [
        len(analyzer._extract_candles_payload(event))
        for event in shapes
    ]

    assert extracted_lengths == [1, 1, 2, 2]


def test_extract_candles_payload_filters_mixed_or_malicious_sequences_without_crash(
    event_bus,
    event_factory,
    candle_factory,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    valid_candle = candle_factory(0)

    event = event_factory(
        "market.candles",
        {
            "candles": [
                valid_candle,
                "not-a-candle",
                b"bytes",
                123,
                None,
                {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
            ]
        },
    )

    extracted = analyzer._extract_candles_payload(event)

    assert len(extracted) == 2
    assert extracted[0] == valid_candle
    assert extracted[1]["open"] == 1.0


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

    event = event_factory("market.candle", payload)

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
async def test_emit_event_rejects_empty_event_name_before_hitting_event_bus(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    with pytest.raises(ValueError, match="event_name must not be empty"):
        await analyzer._emit_event(" ... ", {"value": 1})

    assert recorder.calls == []


@pytest.mark.asyncio
async def test_emit_many_counts_only_accepted_events_and_keeps_processing_after_rejection(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder(
        reject_topics={"analytics.price_action.fair_value_gap.rejected"}
    )
    monkeypatch.setattr(event_bus, "emit", recorder)

    emitted = await analyzer._emit_many(
        [
            ("analytics.price_action.fair_value_gap.accepted_1", {"n": 1}),
            ("analytics.price_action.fair_value_gap.rejected", {"n": 2}),
            ("analytics.price_action.fair_value_gap.accepted_2", {"n": 3}),
        ],
        correlation_id="corr-many",
    )

    assert emitted == 2
    assert [call["topic"] for call in recorder.calls] == [
        "analytics.price_action.fair_value_gap.accepted_1",
        "analytics.price_action.fair_value_gap.rejected",
        "analytics.price_action.fair_value_gap.accepted_2",
    ]


# ---------------------------------------------------------------------------
# Snapshot / reset publication hardening
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_publish_snapshot_returns_false_when_snapshot_publishing_disabled(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    config = replace(fair_value_gap_config, publish_snapshots=False)
    analyzer = build_fvg(event_bus=event_bus, config=config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer.publish_snapshot(correlation_id="corr-snapshot-disabled")

    assert accepted is False
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_publish_snapshot_emits_snapshot_envelope_when_enabled(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    config = replace(fair_value_gap_config, publish_snapshots=True)
    analyzer = build_fvg(event_bus=event_bus, config=config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer.publish_snapshot(
        snapshot_name=".custom_snapshot.",
        correlation_id="corr-snapshot",
    )

    assert accepted is True
    assert len(recorder.calls) == 1

    call = recorder.calls[0]
    assert call["topic"] == "analytics.price_action.fair_value_gap.custom_snapshot"
    assert call["correlation_id"] == "corr-snapshot"

    payload = call["payload"]
    assert payload["symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "1m"
    assert payload["module"] == "FairValueGapAnalyzer"
    assert "published_at" in payload
    assert "snapshot" in payload
    assert payload["snapshot"]["symbol"] == "BTCUSDT"
    assert payload["snapshot"]["timeframe"] == "1m"
    assert payload["snapshot"]["module"] == "FairValueGapAnalyzer"


@pytest.mark.asyncio
async def test_publish_reset_uses_event_namespace_and_core_emit_path(
    event_bus,
    monkeypatch,
    fair_value_gap_config,
):
    analyzer = build_fvg(event_bus=event_bus, config=fair_value_gap_config)
    recorder = EmitRecorder()
    monkeypatch.setattr(event_bus, "emit", recorder)

    accepted = await analyzer.publish_reset(correlation_id="corr-reset")

    assert accepted is True
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["topic"] == "analytics.price_action.fair_value_gap.reset"
    assert recorder.calls[0]["payload"]["symbol"] == "BTCUSDT"
    assert recorder.calls[0]["payload"]["timeframe"] == "1m"
    assert recorder.calls[0]["payload"]["module"] == "FairValueGapAnalyzer"
    assert recorder.calls[0]["correlation_id"] == "corr-reset"


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

    assert envelope["symbol"] == "BTCUSDT"
    assert envelope["timeframe"] == "1m"
    assert envelope["module"] == "FairValueGapAnalyzer"
    assert envelope["state"]["priority"] == EventPriority.CRITICAL.value
    assert envelope["state"]["enum_value"] == NonSerializableEnum.VALUE.value
    assert sorted(envelope["state"]["values"]) == [1, 2]
    assert envelope["state"]["nested"]["object"] == "exploding-object-as-string"
    assert envelope["metadata"]["priority"] == EventPriority.LOW.value
    assert envelope["metadata"]["custom_object"] == "exploding-object-as-string"


def test_build_event_name_strips_duplicate_dots_and_allows_namespace_only_event(
    event_bus,
    fair_value_gap_config,
):
    config = replace(
        fair_value_gap_config,
        event_namespace=".analytics.price_action.fair_value_gap.",
    )
    analyzer = build_fvg(event_bus=event_bus, config=config)

    assert analyzer._build_event_name(".updated.") == "analytics.price_action.fair_value_gap.updated"
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