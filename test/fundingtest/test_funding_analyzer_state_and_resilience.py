# tests/analytics/funding/test_funding_analyzer_state_and_resilience.py

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from analytics.funding.config import FundingAnalyzerConfig
from analytics.funding.enums import (
    FundingDataSource,
    FundingEventType,
    FundingTimeframe,
)
from analytics.funding.funding_analyzer import FundingAnalyzer, FundingMarketContext
from analytics.funding.models import (
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


# =============================================================================
# Local helpers
# =============================================================================

def _topics(event_bus: Any) -> list[str]:
    return [event.topic for event in event_bus.published]


def _payloads_for(event_bus: Any, topic: str) -> list[dict[str, Any]]:
    return [event.payload for event in event_bus.published if event.topic == topic]


def _last_payload_for(event_bus: Any, topic: str) -> dict[str, Any]:
    payloads = _payloads_for(event_bus, topic)
    assert payloads, f"No payloads for topic={topic!r}. Published topics={_topics(event_bus)}"
    return payloads[-1]


def _event_count(event_bus: Any, topic: str) -> int:
    return _topics(event_bus).count(topic)


async def _run_funding_update(
    analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    *,
    topic: str | None = None,
    correlation_id: str | None = "corr-state-resilience",
    **payload_overrides: Any,
) -> None:
    await analyzer.on_funding(
        make_event(
            make_funding_payload(**payload_overrides),
            topic=topic or analyzer.config.funding_event_name,
            correlation_id=correlation_id,
        )
    )


def _get_lock_for_default_key(analyzer: FundingAnalyzer) -> asyncio.Lock:
    key = analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    return analyzer._locks[key]


class FaultyDetector:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("detector failed intentionally")
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.exc


class FaultyAnalyzeDetector:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("pressure analyzer failed intentionally")
        self.calls: list[dict[str, Any]] = []

    def analyze(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.exc

    def is_high_pressure(self, *_: Any, **__: Any) -> bool:
        return False

    def is_squeeze_risk(self, *_: Any, **__: Any) -> bool:
        return False

    def build_summary(self, *_: Any, **__: Any) -> str:
        return "faulty pressure analyzer"


class ExplodingParquetStorage:
    def __init__(self) -> None:
        self.append_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []

    def append_records(self, *, dataset: str, records: list[dict[str, Any]]) -> None:
        self.append_calls.append({"dataset": dataset, "records": records})
        raise RuntimeError("parquet append failed intentionally")

    def read_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.read_calls.append(kwargs)
        return []


class RecordingParquetStorage:
    def __init__(self) -> None:
        self.append_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []

    def append_records(self, *, dataset: str, records: list[dict[str, Any]]) -> None:
        self.append_calls.append({"dataset": dataset, "records": list(records)})

    def read_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.read_calls.append(kwargs)
        return []


# =============================================================================
# Parsing / low-level utility behavior
# =============================================================================

def test_make_key_is_full_futures_scope_not_legacy_symbol_exchange_string(
    funding_analyzer: FundingAnalyzer,
) -> None:
    key = funding_analyzer._make_key(
        symbol=" btc/usdt ",
        exchange=" BINANCE ",
        market_type="USDM_FUTURES",
        timeframe="1h",
    )

    assert key == ("binance", "usdm_futures", "BTCUSDT", "1h")
    assert isinstance(key, tuple)
    assert len(key) == 4


def test_make_key_uses_config_defaults_for_market_type_and_timeframe(
    funding_analyzer: FundingAnalyzer,
) -> None:
    key = funding_analyzer._make_key(
        symbol="ETHUSDT",
        exchange="bybit",
    )

    assert key == (
        "bybit",
        funding_analyzer.config.default_market_type,
        "ETHUSDT",
        funding_analyzer.config.default_timeframe.value,
    )


def test_extract_payload_requires_mapping_payload(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
) -> None:
    assert funding_analyzer._extract_payload(make_event({"symbol": "BTCUSDT"})) == {
        "symbol": "BTCUSDT"
    }

    with pytest.raises(TypeError):
        funding_analyzer._extract_payload(make_event(["not", "dict"]))

    with pytest.raises(TypeError):
        funding_analyzer._extract_payload(make_event(None))


def test_first_present_does_not_drop_zero_or_empty_string(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._first_present({"a": 0, "b": 1}, "a", "b") == 0
    assert funding_analyzer._first_present({"a": 0.0, "b": 1.0}, "a", "b") == 0.0
    assert funding_analyzer._first_present({"a": "", "b": "fallback"}, "a", "b") == ""
    assert funding_analyzer._first_present({"a": None, "b": 2}, "a", "b") == 2
    assert funding_analyzer._first_present({}, "missing") is None


def test_parse_datetime_handles_naive_aware_seconds_milliseconds_and_zulu(
    funding_analyzer: FundingAnalyzer,
) -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    assert funding_analyzer._parse_datetime(naive) == aware
    assert funding_analyzer._parse_datetime(aware) == aware
    assert funding_analyzer._parse_datetime(1_767_268_800) == aware
    assert funding_analyzer._parse_datetime(1_767_268_800_000) == aware
    assert funding_analyzer._parse_datetime("2026-01-01T12:00:00Z") == aware

    with pytest.raises(TypeError):
        funding_analyzer._parse_datetime(object())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("0", 0.0),
        (0, 0.0),
        (0.0, 0.0),
        ("0.0001", 0.0001),
        ("not-a-number", None),
        (object(), None),
    ],
)
def test_to_optional_float_is_strict_but_accepts_zero(
    funding_analyzer: FundingAnalyzer,
    value: Any,
    expected: float | None,
) -> None:
    result = funding_analyzer._to_optional_float(value)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_parse_funding_snapshot_preserves_zero_values_and_full_scope(
    funding_analyzer: FundingAnalyzer,
    now_utc: datetime,
) -> None:
    snapshot = funding_analyzer._parse_funding_snapshot(
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "btc/usdt",
            "timeframe": "1h",
            "exchange_symbol": "BTC/USDT:USDT",
            "funding_rate": "0",
            "predicted_funding_rate": 0,
            "mark_price": 0,
            "index_price": 0,
            "open_interest": 0,
            "volume_24h": 0,
            "event_time": now_utc,
            "received_at": now_utc,
            "metadata": {"source": "hard-test"},
        }
    )

    assert snapshot.key == ("binance", "usdm_futures", "BTCUSDT", "1h")
    assert snapshot.exchange == FundingDataSource.BINANCE
    assert snapshot.market_type == "usdm_futures"
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.timeframe == FundingTimeframe.H1
    assert snapshot.exchange_symbol == "BTC/USDT:USDT"
    assert snapshot.funding_rate == pytest.approx(0.0)
    assert snapshot.predicted_funding_rate == pytest.approx(0.0)
    assert snapshot.mark_price == pytest.approx(0.0)
    assert snapshot.index_price == pytest.approx(0.0)
    assert snapshot.open_interest == pytest.approx(0.0)
    assert snapshot.volume_24h == pytest.approx(0.0)
    assert snapshot.metadata["scope"] == {
        "exchange": "binance",
        "market_type": "usdm_futures",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"exchange": "binance", "market_type": "usdm_futures", "funding_rate": 0.0001},
        {"exchange": "binance", "market_type": "usdm_futures", "symbol": "", "funding_rate": 0.0001},
        {"exchange": "binance", "market_type": "usdm_futures", "symbol": "BTCUSDT"},
        {"exchange": "binance", "market_type": "usdm_futures", "symbol": "BTCUSDT", "funding_rate": "bad"},
        {"exchange": "binance", "market_type": "usdm_futures", "symbol": "BTCUSDT", "funding_rate": 0.1, "event_time": object()},
    ],
)
def test_parse_funding_snapshot_rejects_malformed_payloads(
    funding_analyzer: FundingAnalyzer,
    payload: dict[str, Any],
) -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        funding_analyzer._parse_funding_snapshot(payload)


def test_key_from_payload_supports_nested_fallback_scope(
    funding_analyzer: FundingAnalyzer,
) -> None:
    key = funding_analyzer._key_from_payload(
        {"price": 50_000},
        fallback_payload={
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        },
    )

    assert key == ("binance", "usdm_futures", "BTCUSDT", "1h")


def test_key_from_payload_returns_none_when_symbol_missing(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._key_from_payload({"exchange": "binance"}) is None


# =============================================================================
# Statistics hard behavior
# =============================================================================

def test_build_statistics_rejects_empty_history(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
) -> None:
    with pytest.raises(ValueError, match="history must not be empty"):
        funding_analyzer._build_statistics(
            snapshot=make_snapshot(),
            history=deque(),
        )


def test_build_statistics_single_snapshot_has_zero_std_and_no_zscore(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    snapshot = make_snapshot(funding_rate=0.0001, event_time=now_utc)

    statistics = funding_analyzer._build_statistics(
        snapshot=snapshot,
        history=deque([snapshot]),
    )

    assert statistics.key == snapshot.key
    assert statistics.current_rate == pytest.approx(0.0001)
    assert statistics.mean_rate == pytest.approx(0.0001)
    assert statistics.median_rate == pytest.approx(0.0001)
    assert statistics.std_rate == pytest.approx(0.0)
    assert statistics.zscore is None
    assert statistics.percentile == pytest.approx(100.0)
    assert statistics.sample_size == 1
    assert statistics.window_start == now_utc
    assert statistics.window_end == now_utc


def test_build_statistics_uses_statistics_window_size_not_full_history(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        statistics_window_size=3,
        max_history_per_key=10,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    rates = [-0.01, -0.01, 0.0001, 0.0002, 0.0003]
    history = deque(
        [
            make_snapshot(
                funding_rate=rate,
                event_time=now_utc + timedelta(minutes=index),
            )
            for index, rate in enumerate(rates)
        ],
        maxlen=10,
    )

    statistics = analyzer._build_statistics(
        snapshot=history[-1],
        history=history,
    )

    assert statistics.sample_size == 3
    assert statistics.current_rate == pytest.approx(0.0003)
    assert statistics.mean_rate == pytest.approx((0.0001 + 0.0002 + 0.0003) / 3)
    assert statistics.min_rate == pytest.approx(0.0001)
    assert statistics.max_rate == pytest.approx(0.0003)
    assert statistics.window_start == history[-3].event_time
    assert statistics.window_end == history[-1].event_time


def test_percentile_helper_uses_less_or_equal_semantics(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._calc_percentile([], 1.0) is None
    assert funding_analyzer._calc_percentile([1.0], 1.0) == pytest.approx(100.0)
    assert funding_analyzer._calc_percentile([1.0, 1.0, 1.0], 1.0) == pytest.approx(100.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], -1.0) == pytest.approx(100.0 / 3.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], 0.0) == pytest.approx(200.0 / 3.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], 1.0) == pytest.approx(100.0)


def test_numeric_change_helpers_are_none_safe(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._calc_change_pct(None, 10.0) is None
    assert funding_analyzer._calc_change_pct(0.0, 10.0) is None
    assert funding_analyzer._calc_change_pct(100.0, 110.0) == pytest.approx(0.10)

    assert funding_analyzer._calc_price_change_pct(50_000.0, 50_100.0) == pytest.approx(0.002)
    assert funding_analyzer._calc_delta(None, 1.0) is None
    assert funding_analyzer._calc_delta(10.0, 25.0) == pytest.approx(15.0)


# =============================================================================
# State behavior / bounded history
# =============================================================================

@pytest.mark.asyncio
async def test_on_funding_keeps_bounded_history_per_full_futures_key(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        max_history_per_key=3,
        statistics_window_size=3,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    rates = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005]

    for rate in rates:
        await _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            funding_rate=rate,
        )

    key = analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert len(analyzer._history[key]) == 3
    assert [item.funding_rate for item in analyzer._history[key]] == pytest.approx(rates[-3:])

    latest = analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    statistics = analyzer.get_statistics(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert latest is analyzer._history[key][-1]
    assert latest.funding_rate == pytest.approx(0.00005)

    assert statistics is not None
    assert statistics.sample_size == 3
    assert statistics.current_rate == pytest.approx(0.00005)
    assert statistics.mean_rate == pytest.approx(sum(rates[-3:]) / 3)


@pytest.mark.asyncio
async def test_on_funding_state_isolated_by_exchange_market_type_symbol_and_timeframe(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    cases = [
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "funding_rate": 0.0001,
        },
        {
            "exchange": "binance",
            "market_type": "coinm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "funding_rate": 0.0002,
        },
        {
            "exchange": "bybit",
            "market_type": "linear",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "funding_rate": -0.0003,
        },
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "funding_rate": 0.0004,
        },
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "4h",
            "funding_rate": 0.0005,
        },
    ]

    for case in cases:
        await _run_funding_update(
            funding_analyzer,
            make_event,
            make_funding_payload,
            **case,
        )

    assert funding_analyzer.stats()["keys_tracked"] == 5
    assert funding_analyzer.stats()["latest_statistics"] == 5
    assert funding_analyzer.stats()["latest_regime_states"] == 5
    assert funding_analyzer.stats()["latest_pressure_states"] == 5

    for case in cases:
        snapshot = funding_analyzer.get_latest_snapshot(
            case["symbol"],
            case["exchange"],
            market_type=case["market_type"],
            timeframe=case["timeframe"],
        )
        assert snapshot is not None
        assert snapshot.funding_rate == pytest.approx(case["funding_rate"])


@pytest.mark.asyncio
async def test_max_tracked_keys_blocks_new_keys_but_does_not_block_existing_key_updates(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    fake_event_bus: Any,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        max_tracked_keys=1,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        symbol="BTCUSDT",
        funding_rate=0.0001,
        correlation_id="corr-max-key-1",
    )
    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        symbol="ETHUSDT",
        funding_rate=0.0002,
        correlation_id="corr-max-key-2",
    )
    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        symbol="BTCUSDT",
        funding_rate=0.0003,
        correlation_id="corr-max-key-3",
    )

    assert analyzer.stats()["keys_tracked"] == 1

    assert analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ).funding_rate == pytest.approx(0.0003)

    assert analyzer.get_latest_snapshot(
        "ETHUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ) is None

    assert _event_count(fake_event_bus, analyzer.config.analytics_event_name) == 2


@pytest.mark.asyncio
async def test_enrich_snapshot_uses_context_only_when_snapshot_fields_are_missing(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
) -> None:
    context = FundingMarketContext(
        latest_open_interest=2_000_000.0,
        latest_price=55_000.0,
    )

    missing = make_snapshot(open_interest=None, mark_price=None)
    funding_analyzer._enrich_snapshot(missing, context)

    assert missing.open_interest == pytest.approx(2_000_000.0)
    assert missing.mark_price == pytest.approx(55_000.0)

    explicit = make_snapshot(open_interest=1_000_000.0, mark_price=50_000.0)
    funding_analyzer._enrich_snapshot(explicit, context)

    assert explicit.open_interest == pytest.approx(1_000_000.0)
    assert explicit.mark_price == pytest.approx(50_000.0)


# =============================================================================
# Context state and raw-topic guards
# =============================================================================

@pytest.mark.asyncio
async def test_context_updates_keep_previous_latest_values_per_key(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_open_interest(
        make_event(make_context_payload(open_interest=1_000_000.0))
    )
    await funding_analyzer.on_open_interest(
        make_event(make_context_payload(open_interest=1_100_000.0))
    )

    await funding_analyzer.on_candle(
        make_event(make_context_payload(close=50_000.0, price=1.0))
    )
    await funding_analyzer.on_trade(
        make_event(make_context_payload(price=50_100.0))
    )

    await funding_analyzer.on_cvd_update(
        make_event(make_context_payload(cvd=10_000.0))
    )
    await funding_analyzer.on_cvd_update(
        make_event(make_context_payload(cvd=None, cumulative_volume_delta=25_000.0))
    )

    context = funding_analyzer.get_market_context(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert context is not None
    assert context.previous_open_interest == pytest.approx(1_000_000.0)
    assert context.latest_open_interest == pytest.approx(1_100_000.0)
    assert context.previous_price == pytest.approx(50_000.0)
    assert context.latest_price == pytest.approx(50_100.0)
    assert context.previous_cvd == pytest.approx(10_000.0)
    assert context.latest_cvd == pytest.approx(25_000.0)


@pytest.mark.asyncio
async def test_context_handlers_ignore_raw_topics_when_legacy_mode_disabled(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest=1_000_000.0),
            topic=funding_analyzer.config.raw_open_interest_event_name,
        )
    )
    await funding_analyzer.on_candle(
        make_event(
            make_context_payload(close=50_000.0),
            topic=funding_analyzer.config.raw_candle_event_name,
        )
    )
    await funding_analyzer.on_trade(
        make_event(
            make_context_payload(price=50_100.0),
            topic=funding_analyzer.config.raw_trade_event_name,
        )
    )
    await funding_analyzer.on_liquidation(
        make_event(
            make_context_payload(side="long", notional=100_000.0),
            topic=funding_analyzer.config.raw_liquidation_event_name,
        )
    )

    assert funding_analyzer.stats()["contexts_tracked"] == 0


@pytest.mark.asyncio
async def test_raw_funding_topic_is_blocked_even_with_direct_handler_call_by_default(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        topic=funding_analyzer.config.raw_funding_event_name,
        funding_rate=0.00035,
    )

    assert funding_analyzer.stats()["keys_tracked"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_raw_funding_topic_is_allowed_in_legacy_mode(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        allow_legacy_raw_topics=True,
        min_samples_for_statistics=1,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        topic=analyzer.config.raw_funding_event_name,
        funding_rate=0.00035,
    )

    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.config.analytics_event_name in _topics(fake_event_bus)


@pytest.mark.asyncio
async def test_malformed_context_payloads_do_not_create_contexts(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
) -> None:
    malformed = make_event({})

    await funding_analyzer.on_open_interest(malformed)
    await funding_analyzer.on_candle(malformed)
    await funding_analyzer.on_trade(malformed)
    await funding_analyzer.on_cvd_update(malformed)
    await funding_analyzer.on_liquidation(malformed)

    assert funding_analyzer.stats()["contexts_tracked"] == 0


# =============================================================================
# Failure / resilience
# =============================================================================

@pytest.mark.asyncio
async def test_malformed_funding_payload_does_not_mutate_state_or_publish(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
) -> None:
    malformed_payloads = [
        [],
        {},
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "funding_rate": 0.0001,
        },
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "funding_rate": "not-a-number",
        },
        {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "funding_rate": 0.0001,
            "event_time": object(),
        },
    ]

    for payload in malformed_payloads:
        await funding_analyzer.on_funding(make_event(payload))

    assert funding_analyzer.stats()["keys_tracked"] == 0
    assert funding_analyzer.stats()["latest_statistics"] == 0
    assert funding_analyzer.stats()["latest_regime_states"] == 0
    assert funding_analyzer.stats()["latest_pressure_states"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_detector_failure_releases_lock_and_does_not_publish_events(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    faulty_regime_detector = FaultyDetector(RuntimeError("regime detector exploded"))
    analyzer = make_funding_analyzer(regime_detector=faulty_regime_detector)

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.00035,
    )

    lock = _get_lock_for_default_key(analyzer)

    assert lock.locked() is False
    assert faulty_regime_detector.calls

    # Поточний FundingAnalyzer додає snapshot у history до detector pipeline.
    # Це не атомарно. Тест робить цей contract явним.
    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 0
    assert analyzer.stats()["latest_regime_states"] == 0
    assert analyzer.stats()["latest_pressure_states"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_pressure_analyzer_failure_releases_lock_and_keeps_partial_state_explicit(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
) -> None:
    faulty_pressure = FaultyAnalyzeDetector(RuntimeError("pressure analyzer exploded"))
    analyzer = make_funding_analyzer(
        regime_detector=stub_regime_detector,
        pressure_analyzer=faulty_pressure,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.00035,
    )

    lock = _get_lock_for_default_key(analyzer)

    assert lock.locked() is False
    assert faulty_pressure.calls

    # Regime detector уже відпрацював локально, але latest maps оновлюються нижче
    # після всього detector pipeline, тому вони мають залишитись порожніми.
    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 0
    assert analyzer.stats()["latest_regime_states"] == 0
    assert analyzer.stats()["latest_pressure_states"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_emit_failure_releases_lock_but_state_is_already_committed(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    failing_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    analyzer = make_funding_analyzer(
        event_bus=failing_event_bus,
        scheduler=fake_scheduler,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.00035,
    )

    lock = _get_lock_for_default_key(analyzer)

    assert lock.locked() is False
    assert failing_event_bus.emit_attempts

    # Publish failure occurs after detector/state commit.
    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 1
    assert analyzer.stats()["latest_regime_states"] == 1
    assert analyzer.stats()["latest_pressure_states"] == 1


@pytest.mark.asyncio
async def test_lock_timeout_skips_processing_without_mutating_state(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    key = funding_analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    lock = funding_analyzer._locks[key]

    await lock.acquire()

    try:
        await _run_funding_update(
            funding_analyzer,
            make_event,
            make_funding_payload,
            funding_rate=0.00035,
        )
    finally:
        lock.release()

    assert funding_analyzer.stats()["keys_tracked"] == 0
    assert funding_analyzer.stats()["latest_statistics"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_concurrent_updates_do_not_corrupt_bounded_history(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        max_history_per_key=10,
        statistics_window_size=10,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    rates = [0.00001 * index for index in range(1, 21)]

    await asyncio.gather(
        *[
            _run_funding_update(
                analyzer,
                make_event,
                make_funding_payload,
                funding_rate=rate,
                event_time=datetime(2026, 1, 1, 12, index, tzinfo=timezone.utc),
                correlation_id=f"corr-concurrent-{index}",
            )
            for index, rate in enumerate(rates)
        ]
    )

    key = analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert len(analyzer._history[key]) == 10
    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 1
    assert analyzer._locks[key].locked() is False


# =============================================================================
# Cleanup
# =============================================================================

@pytest.mark.asyncio
async def test_cleanup_removes_stale_context_without_history_and_removes_lock(
    funding_analyzer: FundingAnalyzer,
    make_key: Callable[..., tuple[str, str, str, str]],
    now_utc: datetime,
) -> None:
    key = make_key(symbol="BTCUSDT")
    stale_time = now_utc - timedelta(seconds=funding_analyzer.config.stale_state_ttl_sec + 10)

    funding_analyzer._market_context[key] = FundingMarketContext(
        latest_price=50_000.0,
        updated_at=stale_time,
    )
    _ = funding_analyzer._locks[key]

    assert key in funding_analyzer._market_context
    assert key in funding_analyzer._locks

    await funding_analyzer.cleanup_stale_state()

    assert key not in funding_analyzer._market_context
    assert key not in funding_analyzer._locks


@pytest.mark.asyncio
async def test_cleanup_does_not_remove_context_when_history_exists(
    funding_analyzer: FundingAnalyzer,
    make_key: Callable[..., tuple[str, str, str, str]],
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    key = make_key(symbol="BTCUSDT")
    stale_time = now_utc - timedelta(seconds=funding_analyzer.config.stale_state_ttl_sec + 10)

    funding_analyzer._market_context[key] = FundingMarketContext(
        latest_price=50_000.0,
        updated_at=stale_time,
    )
    funding_analyzer._history[key].append(make_snapshot())

    await funding_analyzer.cleanup_stale_state()

    assert key in funding_analyzer._market_context
    assert key in funding_analyzer._history


@pytest.mark.asyncio
async def test_cleanup_clears_stale_liquidations_but_keeps_context(
    funding_analyzer: FundingAnalyzer,
    make_key: Callable[..., tuple[str, str, str, str]],
    now_utc: datetime,
) -> None:
    key = make_key(symbol="BTCUSDT")
    stale_time = now_utc - timedelta(seconds=funding_analyzer.config.stale_state_ttl_sec + 10)

    funding_analyzer._market_context[key] = FundingMarketContext(
        latest_price=50_000.0,
        updated_at=now_utc,
        long_liquidations=100_000.0,
        short_liquidations=50_000.0,
        liquidation_updated_at=stale_time,
    )

    await funding_analyzer.cleanup_stale_state()

    context = funding_analyzer._market_context[key]
    assert context.latest_price == pytest.approx(50_000.0)
    assert context.long_liquidations is None
    assert context.short_liquidations is None
    assert context.liquidation_updated_at is None


# =============================================================================
# Emit cooldown and signal state
# =============================================================================

def test_should_skip_emit_is_scoped_by_event_name_and_full_key(
    funding_analyzer: FundingAnalyzer,
) -> None:
    key_a = funding_analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    key_b = funding_analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="coinm_futures",
        timeframe="1h",
    )

    assert funding_analyzer._should_skip_emit(
        event_name=funding_analyzer.config.signal_event_name,
        key=key_a,
    ) is False
    assert funding_analyzer._should_skip_emit(
        event_name=funding_analyzer.config.signal_event_name,
        key=key_a,
    ) is True

    assert funding_analyzer._should_skip_emit(
        event_name=funding_analyzer.config.regime_event_name,
        key=key_a,
    ) is False

    assert funding_analyzer._should_skip_emit(
        event_name=funding_analyzer.config.signal_event_name,
        key=key_b,
    ) is False


# =============================================================================
# Parquet resilience
# =============================================================================

@pytest.mark.asyncio
async def test_parquet_buffer_uses_configured_batch_size_and_dataset_name(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    storage = RecordingParquetStorage()
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        enable_parquet_history=True,
        load_history_from_parquet_on_start=False,
        parquet_dataset_name="custom_funding_dataset",
        parquet_flush_batch_size=2,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
    )
    analyzer = make_funding_analyzer(
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        config=config,
        parquet_storage=storage,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
        correlation_id="corr-parquet-1",
    )
    assert storage.append_calls == []
    assert analyzer.stats()["parquet_buffer_size"] == 1

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0002,
        correlation_id="corr-parquet-2",
    )

    assert len(storage.append_calls) == 1
    assert storage.append_calls[0]["dataset"] == "custom_funding_dataset"
    assert len(storage.append_calls[0]["records"]) == 2
    assert analyzer.stats()["parquet_buffer_size"] == 0


@pytest.mark.asyncio
async def test_parquet_flush_failure_restores_buffer(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    storage = ExplodingParquetStorage()
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        enable_parquet_history=True,
        load_history_from_parquet_on_start=False,
        parquet_dataset_name="custom_funding_dataset",
        parquet_flush_batch_size=1,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        emit_signals=False,
    )
    analyzer = make_funding_analyzer(
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        config=config,
        parquet_storage=storage,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
        correlation_id="corr-parquet-fail",
    )

    assert len(storage.append_calls) == 1
    assert analyzer.stats()["parquet_buffer_size"] == 1


def test_parquet_root_uses_config_not_hardcoded_path(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        enable_parquet_history=True,
        parquet_base_path="/tmp/custom-parquet-root",
        parquet_dataset_name="custom_dataset_name",
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    assert str(analyzer._parquet_root()) == "/tmp/custom-parquet-root/custom_dataset_name"


# =============================================================================
# History API
# =============================================================================

@pytest.mark.asyncio
async def test_get_history_returns_recent_in_memory_snapshots_in_order(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        max_history_per_key=10,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    rates = [0.0001, 0.0002, 0.0003, 0.0004]

    for index, rate in enumerate(rates):
        await _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            funding_rate=rate,
            event_time=datetime(2026, 1, 1, 12, index, tzinfo=timezone.utc),
        )

    history = await analyzer.get_history(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
        limit=2,
        include_parquet=False,
    )

    assert [item.funding_rate for item in history] == pytest.approx([0.0003, 0.0004])


@pytest.mark.asyncio
async def test_get_history_with_non_positive_limit_returns_empty(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert await funding_analyzer.get_history(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
        limit=0,
    ) == []

    assert await funding_analyzer.get_history(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
        limit=-10,
    ) == []


# =============================================================================
# Model scope copy
# =============================================================================

def test_copy_scope_forces_detector_result_scope_to_snapshot_scope(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    make_regime_state: Callable[..., Any],
) -> None:
    snapshot = make_snapshot(
        exchange=FundingDataSource.BINANCE,
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe=FundingTimeframe.H1,
        exchange_symbol="BTC/USDT:USDT",
    )
    regime_state = make_regime_state(
        exchange=FundingDataSource.BYBIT,
        market_type="linear",
        symbol="ETHUSDT",
        timeframe=FundingTimeframe.M5,
        exchange_symbol="ETHUSDT",
        metadata={},
    )

    result = funding_analyzer._copy_scope(regime_state, snapshot)

    assert result.exchange == snapshot.exchange
    assert result.market_type == snapshot.market_type
    assert result.symbol == snapshot.symbol
    assert result.timeframe == snapshot.timeframe
    assert result.exchange_symbol == snapshot.exchange_symbol
    assert result.metadata["scope"] == funding_key_to_dict(snapshot.key)
    assert result.metadata["exchange_symbol"] == snapshot.exchange_symbol