# tests/analytics/funding/test_funding_analyzer_state_and_resilience.py

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest

from analytics.funding.enums import (
    FundingDataSource,
    FundingTimeframe,
)
from analytics.funding.funding_analyzer import (
    FundingAnalyzer,
    FundingAnalyzerConfig,
    FundingMarketContext,
)
from analytics.funding.models import (
    FundingSnapshot,
    FundingStatistics,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_pressure_stub_signal_compatible(stub_pressure_analyzer: Any) -> None:
    """
    Backward-compatible helper for StubPressureAnalyzer from conftest.py.

    FundingAnalyzer._build_signals() may call:
    - is_high_pressure(...)
    - is_squeeze_risk(...)
    - build_summary(...)
    """
    if not hasattr(stub_pressure_analyzer, "is_squeeze_risk"):
        stub_pressure_analyzer.is_squeeze_risk = lambda pressure_state, threshold=0.65: True


def _published_topics(event_bus: Any) -> list[str]:
    return [event.topic for event in event_bus.published]


async def _run_funding_update(
    analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    funding_rate: float | str = 0.0001,
    mark_price: float | str | None = 50_000.0,
    open_interest: float | str | None = 1_000_000.0,
    correlation_id: str = "corr-resilience",
) -> None:
    await analyzer.on_funding(
        make_event(
            make_funding_payload(
                symbol=symbol,
                exchange=exchange,
                funding_rate=funding_rate,
                mark_price=mark_price,
                open_interest=open_interest,
            ),
            topic=analyzer.config.funding_event_name,
            correlation_id=correlation_id,
        )
    )


def _make_analyzer(
    *,
    fake_event_bus: Any,
    fake_scheduler: Any | None,
    config: FundingAnalyzerConfig,
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> FundingAnalyzer:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    return FundingAnalyzer(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
        config=config,
        regime_detector=stub_regime_detector,  # type: ignore[arg-type]
        pressure_analyzer=stub_pressure_analyzer,  # type: ignore[arg-type]
        flip_detector=stub_flip_detector,  # type: ignore[arg-type]
        extremes_detector=stub_extremes_detector,  # type: ignore[arg-type]
        divergence_detector=stub_divergence_detector,  # type: ignore[arg-type]
    )


class FaultyRegimeDetector:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("regime detector failed")
        self.calls: list[dict[str, Any]] = []

    def detect(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.exc


class FaultyEmitEventBus:
    """
    EventBus fake that fails during emit.

    Використовується для перевірки, що lock буде release-нутий навіть тоді,
    коли помилка сталася на стадії publish.
    """

    def __init__(self) -> None:
        self.subscriptions: list[Any] = []
        self.unsubscribed: list[Any] = []
        self.published_attempts: list[tuple[str, dict[str, Any] | None, dict[str, Any]]] = []

    def subscribe(self, topic: str, handler: Callable[..., Any], *, name: str | None = None) -> Any:
        subscription = type(
            "FakeSubscription",
            (),
            {
                "topic": topic,
                "handler": handler,
                "name": name,
            },
        )()
        self.subscriptions.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Any) -> None:
        self.unsubscribed.append(subscription)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.published_attempts.append((topic, payload, kwargs))
        raise RuntimeError("emit failed")


# ---------------------------------------------------------------------------
# Parsing and utility helpers
# ---------------------------------------------------------------------------


def test_make_key_normalizes_symbol_and_exchange(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._make_key(" btcusdt ", " BINANCE ") == "BTCUSDT::binance"
    assert funding_analyzer._make_key("EthUsdt", "ByBit") == "ETHUSDT::bybit"


def test_extract_payload_requires_dict(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
) -> None:
    valid_event = make_event({"symbol": "BTCUSDT"})
    assert funding_analyzer._extract_payload(valid_event) == {"symbol": "BTCUSDT"}

    invalid_event = make_event({"symbol": "BTCUSDT"})
    invalid_event.payload = ["not", "dict"]

    with pytest.raises(TypeError):
        funding_analyzer._extract_payload(invalid_event)


def test_to_optional_float_handles_valid_and_invalid_values(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._to_optional_float(None) is None
    assert funding_analyzer._to_optional_float("0.0001") == pytest.approx(0.0001)
    assert funding_analyzer._to_optional_float(123) == pytest.approx(123.0)
    assert funding_analyzer._to_optional_float("not-a-number") is None
    assert funding_analyzer._to_optional_float(object()) is None


def test_parse_exchange_falls_back_to_unknown(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._parse_exchange("binance") == FundingDataSource.BINANCE
    assert funding_analyzer._parse_exchange(" BINANCE ") == FundingDataSource.BINANCE
    assert funding_analyzer._parse_exchange("not-supported") == FundingDataSource.UNKNOWN


def test_parse_datetime_supports_datetime_seconds_milliseconds_and_iso_strings(
    funding_analyzer: FundingAnalyzer,
) -> None:
    naive_dt = datetime(2026, 1, 1, 12, 0, 0)
    aware_dt = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

    parsed_naive = funding_analyzer._parse_datetime(naive_dt)
    parsed_aware = funding_analyzer._parse_datetime(aware_dt)
    parsed_seconds = funding_analyzer._parse_datetime(1_767_268_800)
    parsed_millis = funding_analyzer._parse_datetime(1_767_268_800_000)
    parsed_iso_z = funding_analyzer._parse_datetime("2026-01-01T12:00:00Z")

    assert parsed_naive.tzinfo is not None
    assert parsed_naive == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert parsed_aware == aware_dt
    assert parsed_seconds == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert parsed_millis == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert parsed_iso_z == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_datetime_rejects_unsupported_value(
    funding_analyzer: FundingAnalyzer,
) -> None:
    with pytest.raises(TypeError):
        funding_analyzer._parse_datetime(object())


def test_parse_funding_snapshot_normalizes_and_converts_values(
    funding_analyzer: FundingAnalyzer,
) -> None:
    snapshot = funding_analyzer._parse_funding_snapshot(
        {
            "symbol": " btcusdt ",
            "exchange": "binance",
            "funding_rate": "0.0001",
            "predicted_funding_rate": "0.0002",
            "mark_price": "50000.5",
            "index_price": "49990.5",
            "open_interest": "1000000",
            "volume_24h": "250000000",
            "next_funding_time": "2026-01-01T16:00:00Z",
            "event_time": "2026-01-01T12:00:00Z",
            "received_at": "2026-01-01T12:00:01Z",
            "metadata": {"source": "unit-test"},
        }
    )

    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.exchange == FundingDataSource.BINANCE
    assert snapshot.funding_rate == pytest.approx(0.0001)
    assert snapshot.predicted_funding_rate == pytest.approx(0.0002)
    assert snapshot.mark_price == pytest.approx(50_000.5)
    assert snapshot.index_price == pytest.approx(49_990.5)
    assert snapshot.open_interest == pytest.approx(1_000_000.0)
    assert snapshot.volume_24h == pytest.approx(250_000_000.0)
    assert snapshot.next_funding_time == datetime(2026, 1, 1, 16, 0, 0, tzinfo=timezone.utc)
    assert snapshot.event_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert snapshot.received_at == datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
    assert snapshot.metadata == {"source": "unit-test"}


def test_parse_funding_snapshot_supports_ts_and_timestamp_fallbacks(
    funding_analyzer: FundingAnalyzer,
) -> None:
    snapshot_from_ts = funding_analyzer._parse_funding_snapshot(
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "funding_rate": 0.0001,
            "ts": 1_767_268_800_000,
        }
    )
    snapshot_from_timestamp = funding_analyzer._parse_funding_snapshot(
        {
            "symbol": "ETHUSDT",
            "exchange": "bybit",
            "funding_rate": 0.0002,
            "timestamp": 1_767_268_800,
        }
    )

    assert snapshot_from_ts.event_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert snapshot_from_timestamp.event_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_funding_snapshot_uses_safe_metadata_default(
    funding_analyzer: FundingAnalyzer,
) -> None:
    snapshot = funding_analyzer._parse_funding_snapshot(
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "funding_rate": 0.0001,
            "metadata": "not-a-dict",
        }
    )

    assert snapshot.metadata == {}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"exchange": "binance", "funding_rate": 0.0001},
        {"symbol": "BTCUSDT", "exchange": "binance", "funding_rate": "bad"},
        {"symbol": "BTCUSDT", "exchange": "binance", "funding_rate": 0.0001, "event_time": object()},
    ],
)
def test_parse_funding_snapshot_rejects_malformed_payloads(
    funding_analyzer: FundingAnalyzer,
    payload: dict[str, Any],
) -> None:
    with pytest.raises((KeyError, TypeError, ValueError)):
        funding_analyzer._parse_funding_snapshot(payload)


def test_numeric_change_helpers_are_safe(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._calc_change_pct(None, 10.0) is None
    assert funding_analyzer._calc_change_pct(0.0, 10.0) is None
    assert funding_analyzer._calc_change_pct(100.0, 110.0) == pytest.approx(0.10)

    assert funding_analyzer._calc_price_change_pct(50_000.0, 50_100.0) == pytest.approx(0.002)
    assert funding_analyzer._calc_delta(None, 10.0) is None
    assert funding_analyzer._calc_delta(10.0, 25.0) == pytest.approx(15.0)


def test_percentile_helper_handles_empty_values_duplicates_and_bounds(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer._calc_percentile([], 0.0) is None
    assert funding_analyzer._calc_percentile([1.0], 1.0) == pytest.approx(100.0)
    assert funding_analyzer._calc_percentile([1.0, 1.0, 1.0], 1.0) == pytest.approx(100.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], -1.0) == pytest.approx(100.0 / 3.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], 0.0) == pytest.approx(200.0 / 3.0)
    assert funding_analyzer._calc_percentile([-1.0, 0.0, 1.0], 1.0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Statistics builder edge cases
# ---------------------------------------------------------------------------


def test_build_statistics_rejects_empty_history(
    funding_analyzer: FundingAnalyzer,
) -> None:
    with pytest.raises(ValueError, match="history must not be empty"):
        funding_analyzer._build_statistics(
            symbol="BTCUSDT",
            exchange="binance",
            history=deque(),
            timeframe=FundingTimeframe.H1,
        )


def test_build_statistics_for_single_snapshot_has_zero_std_and_no_zscore(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    snapshot = make_snapshot(
        funding_rate=0.0001,
        event_time=now_utc,
    )
    history = deque([snapshot])

    statistics = funding_analyzer._build_statistics(
        symbol="btcusdt",
        exchange="binance",
        history=history,
        timeframe=FundingTimeframe.H1,
    )

    assert statistics.symbol == "BTCUSDT"
    assert statistics.exchange == FundingDataSource.BINANCE
    assert statistics.timeframe == FundingTimeframe.H1
    assert statistics.current_rate == pytest.approx(0.0001)
    assert statistics.mean_rate == pytest.approx(0.0001)
    assert statistics.median_rate == pytest.approx(0.0001)
    assert statistics.std_rate == pytest.approx(0.0)
    assert statistics.zscore is None
    assert statistics.percentile == pytest.approx(100.0)
    assert statistics.sample_size == 1
    assert statistics.window_start == now_utc
    assert statistics.window_end == now_utc


def test_build_statistics_for_equal_rates_has_zero_std_and_no_zscore(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    history = deque(
        [
            make_snapshot(funding_rate=0.0001, event_time=now_utc + timedelta(minutes=i))
            for i in range(5)
        ]
    )

    statistics = funding_analyzer._build_statistics(
        symbol="BTCUSDT",
        exchange="binance",
        history=history,
        timeframe=FundingTimeframe.H1,
    )

    assert statistics.current_rate == pytest.approx(0.0001)
    assert statistics.mean_rate == pytest.approx(0.0001)
    assert statistics.median_rate == pytest.approx(0.0001)
    assert statistics.std_rate == pytest.approx(0.0)
    assert statistics.zscore is None
    assert statistics.percentile == pytest.approx(100.0)
    assert statistics.sample_size == 5


def test_build_statistics_for_mixed_rates_calculates_window_and_distribution(
    funding_analyzer: FundingAnalyzer,
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
) -> None:
    rates = [-0.0002, -0.0001, 0.0, 0.0001, 0.0003]
    history = deque(
        [
            make_snapshot(
                funding_rate=rate,
                event_time=now_utc + timedelta(minutes=index),
            )
            for index, rate in enumerate(rates)
        ]
    )

    statistics = funding_analyzer._build_statistics(
        symbol="BTCUSDT",
        exchange="binance",
        history=history,
        timeframe=FundingTimeframe.H1,
    )

    assert statistics.current_rate == pytest.approx(0.0003)
    assert statistics.mean_rate == pytest.approx(0.00002)
    assert statistics.median_rate == pytest.approx(0.0)
    assert statistics.min_rate == pytest.approx(-0.0002)
    assert statistics.max_rate == pytest.approx(0.0003)
    assert statistics.std_rate > 0.0
    assert statistics.zscore is not None
    assert statistics.zscore > 0.0
    assert statistics.percentile == pytest.approx(100.0)
    assert statistics.sample_size == 5
    assert statistics.window_start == now_utc
    assert statistics.window_end == now_utc + timedelta(minutes=4)


# ---------------------------------------------------------------------------
# State/cache/history behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_funding_keeps_bounded_history_maxlen(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    analyzer = _make_analyzer(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=3,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=0.05,
        ),
        stub_regime_detector=stub_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    rates = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005]

    for rate in rates:
        await _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            funding_rate=rate,
        )

    key = analyzer._make_key("BTCUSDT", "binance")
    history = analyzer._history[key]
    statistics = analyzer.get_statistics("BTCUSDT", "binance")

    assert len(history) == 3
    assert [snapshot.funding_rate for snapshot in history] == pytest.approx(rates[-3:])

    assert analyzer.get_latest_snapshot("btcusdt", "BINANCE") is history[-1]
    assert analyzer.get_latest_snapshot("BTCUSDT", "binance").funding_rate == pytest.approx(0.00005)

    assert statistics is not None
    assert statistics.sample_size == 3
    assert statistics.current_rate == pytest.approx(0.00005)
    assert statistics.min_rate == pytest.approx(0.00003)
    assert statistics.max_rate == pytest.approx(0.00005)


@pytest.mark.asyncio
async def test_getters_are_key_normalized_and_return_latest_cached_objects(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        symbol="btcusdt",
        exchange="BINANCE",
        funding_rate=0.0001,
    )

    assert funding_analyzer.get_latest_snapshot(" BTCUSDT ", " binance ") is not None
    assert funding_analyzer.get_statistics("btcusdt", "BINANCE") is not None
    assert funding_analyzer.get_regime_state("btcusdt", "BINANCE") is not None
    assert funding_analyzer.get_pressure_state("btcusdt", "BINANCE") is not None

    stats = funding_analyzer.stats()

    assert stats["symbols_tracked"] == 1
    assert stats["latest_statistics"] == 1
    assert stats["latest_regime_states"] == 1
    assert stats["latest_pressure_states"] == 1


@pytest.mark.asyncio
async def test_symbol_exchange_state_isolated_between_markets(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        symbol="BTCUSDT",
        exchange="binance",
        funding_rate=0.0001,
    )
    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        symbol="BTCUSDT",
        exchange="bybit",
        funding_rate=-0.0002,
    )
    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        symbol="ETHUSDT",
        exchange="binance",
        funding_rate=0.0003,
    )

    assert funding_analyzer.stats()["symbols_tracked"] == 3

    btc_binance = funding_analyzer.get_latest_snapshot("BTCUSDT", "binance")
    btc_bybit = funding_analyzer.get_latest_snapshot("BTCUSDT", "bybit")
    eth_binance = funding_analyzer.get_latest_snapshot("ETHUSDT", "binance")

    assert btc_binance is not None
    assert btc_bybit is not None
    assert eth_binance is not None

    assert btc_binance.funding_rate == pytest.approx(0.0001)
    assert btc_bybit.funding_rate == pytest.approx(-0.0002)
    assert eth_binance.funding_rate == pytest.approx(0.0003)


@pytest.mark.asyncio
async def test_snapshot_is_enriched_from_context_when_payload_fields_are_missing(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    make_market_context: Callable[..., FundingMarketContext],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[key] = make_market_context(
        latest_open_interest=1_234_567.0,
        latest_price=51_000.0,
        previous_open_interest=1_000_000.0,
        previous_price=50_900.0,
    )

    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        mark_price=None,
        open_interest=None,
    )

    snapshot = funding_analyzer.get_latest_snapshot("BTCUSDT", "binance")

    assert snapshot is not None
    assert snapshot.open_interest == pytest.approx(1_234_567.0)
    assert snapshot.mark_price == pytest.approx(51_000.0)


# ---------------------------------------------------------------------------
# Cleanup stale state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_removes_stale_context_without_history(
    funding_analyzer: FundingAnalyzer,
    make_market_context: Callable[..., FundingMarketContext],
    now_utc: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(funding_analyzer, "_utc_now", lambda: now_utc)

    stale_key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[stale_key] = make_market_context(
        updated_at=now_utc - timedelta(seconds=funding_analyzer.config.stale_context_ttl_sec + 1),
        liquidation_updated_at=None,
    )

    await funding_analyzer.cleanup_stale_state()

    assert stale_key not in funding_analyzer._market_context
    assert funding_analyzer.get_market_context("BTCUSDT", "binance") is None


@pytest.mark.asyncio
async def test_cleanup_keeps_stale_context_when_history_exists(
    funding_analyzer: FundingAnalyzer,
    make_market_context: Callable[..., FundingMarketContext],
    make_snapshot: Callable[..., FundingSnapshot],
    now_utc: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(funding_analyzer, "_utc_now", lambda: now_utc)

    key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[key] = make_market_context(
        updated_at=now_utc - timedelta(seconds=funding_analyzer.config.stale_context_ttl_sec + 1),
        liquidation_updated_at=None,
    )
    funding_analyzer._history[key].append(make_snapshot(symbol="BTCUSDT"))

    await funding_analyzer.cleanup_stale_state()

    assert key in funding_analyzer._market_context
    assert funding_analyzer.get_market_context("BTCUSDT", "binance") is not None


@pytest.mark.asyncio
async def test_cleanup_clears_stale_liquidation_values_without_removing_context(
    funding_analyzer: FundingAnalyzer,
    make_market_context: Callable[..., FundingMarketContext],
    now_utc: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(funding_analyzer, "_utc_now", lambda: now_utc)

    key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[key] = make_market_context(
        updated_at=now_utc,
        long_liquidations=100_000.0,
        short_liquidations=200_000.0,
        liquidation_updated_at=now_utc
        - timedelta(seconds=funding_analyzer.config.stale_liquidation_ttl_sec + 1),
    )

    await funding_analyzer.cleanup_stale_state()

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.long_liquidations is None
    assert context.short_liquidations is None
    assert context.liquidation_updated_at is None
    assert key in funding_analyzer._market_context


@pytest.mark.asyncio
async def test_cleanup_does_not_modify_fresh_context_or_fresh_liquidations(
    funding_analyzer: FundingAnalyzer,
    make_market_context: Callable[..., FundingMarketContext],
    now_utc: datetime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(funding_analyzer, "_utc_now", lambda: now_utc)

    key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[key] = make_market_context(
        updated_at=now_utc - timedelta(seconds=1),
        long_liquidations=100_000.0,
        short_liquidations=200_000.0,
        liquidation_updated_at=now_utc - timedelta(seconds=1),
    )

    await funding_analyzer.cleanup_stale_state()

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.long_liquidations == pytest.approx(100_000.0)
    assert context.short_liquidations == pytest.approx(200_000.0)
    assert context.liquidation_updated_at is not None


@pytest.mark.asyncio
async def test_cleanup_handles_context_with_none_timestamps(
    funding_analyzer: FundingAnalyzer,
) -> None:
    key = funding_analyzer._make_key("BTCUSDT", "binance")
    funding_analyzer._market_context[key] = FundingMarketContext(
        latest_open_interest=1_000_000.0,
        latest_price=50_000.0,
        long_liquidations=100_000.0,
        short_liquidations=200_000.0,
        updated_at=None,
        liquidation_updated_at=None,
    )

    await funding_analyzer.cleanup_stale_state()

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.latest_open_interest == pytest.approx(1_000_000.0)
    assert context.latest_price == pytest.approx(50_000.0)
    assert context.long_liquidations == pytest.approx(100_000.0)
    assert context.short_liquidations == pytest.approx(200_000.0)


# ---------------------------------------------------------------------------
# Lock timeout / release / failure resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_funding_returns_without_state_mutation_when_lock_times_out(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    analyzer = _make_analyzer(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=10,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=0.01,
        ),
        stub_regime_detector=stub_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    key = analyzer._make_key("BTCUSDT", "binance")
    lock = analyzer._locks[key]

    await lock.acquire()

    try:
        await _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            funding_rate=0.0001,
        )
    finally:
        lock.release()

    assert analyzer.stats()["symbols_tracked"] == 0
    assert analyzer.stats()["latest_statistics"] == 0
    assert fake_event_bus.published == []
    assert len(stub_regime_detector.calls) == 0
    assert len(stub_pressure_analyzer.calls) == 0
    assert len(stub_flip_detector.calls) == 0
    assert len(stub_extremes_detector.calls) == 0
    assert len(stub_divergence_detector.calls) == 0


@pytest.mark.asyncio
async def test_on_funding_releases_lock_when_detector_raises(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    faulty_regime_detector = FaultyRegimeDetector()

    analyzer = _make_analyzer(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=10,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=0.05,
        ),
        stub_regime_detector=faulty_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
    )

    key = analyzer._make_key("BTCUSDT", "binance")
    lock = analyzer._locks[key]

    assert lock.locked() is False
    assert len(faulty_regime_detector.calls) == 1

    # Snapshot/history уже були оновлені до помилки detector-а.
    assert analyzer.stats()["symbols_tracked"] == 1

    # Але downstream latest-state не має кешуватись після failure.
    assert analyzer.stats()["latest_statistics"] == 0
    assert analyzer.stats()["latest_regime_states"] == 0
    assert analyzer.stats()["latest_pressure_states"] == 0
    assert fake_event_bus.published == []

    await asyncio.wait_for(lock.acquire(), timeout=0.05)
    lock.release()


@pytest.mark.asyncio
async def test_on_funding_releases_lock_when_event_emit_raises(
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    event_bus = FaultyEmitEventBus()

    analyzer = _make_analyzer(
        fake_event_bus=event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=10,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=0.05,
        ),
        stub_regime_detector=stub_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    await _run_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
    )

    key = analyzer._make_key("BTCUSDT", "binance")
    lock = analyzer._locks[key]

    assert lock.locked() is False
    assert event_bus.published_attempts

    # State був порахований до publish failure, тому latest caches вже є.
    assert analyzer.stats()["symbols_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 1
    assert analyzer.stats()["latest_regime_states"] == 1
    assert analyzer.stats()["latest_pressure_states"] == 1

    await asyncio.wait_for(lock.acquire(), timeout=0.05)
    lock.release()


@pytest.mark.asyncio
async def test_on_funding_ignores_parse_error_before_lock_creation_for_invalid_symbol(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
) -> None:
    event = make_event(
        {
            "exchange": "binance",
            "funding_rate": 0.0001,
        },
        topic=funding_analyzer.config.funding_event_name,
    )

    await funding_analyzer.on_funding(event)

    assert funding_analyzer.stats()["symbols_tracked"] == 0
    assert funding_analyzer.stats()["latest_statistics"] == 0
    assert len(funding_analyzer._locks) == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_on_funding_invalid_numeric_rate_does_not_pollute_state(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
) -> None:
    event = make_event(
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "funding_rate": "invalid-rate",
        },
        topic=funding_analyzer.config.funding_event_name,
    )

    await funding_analyzer.on_funding(event)

    assert funding_analyzer.get_latest_snapshot("BTCUSDT", "binance") is None
    assert funding_analyzer.get_statistics("BTCUSDT", "binance") is None
    assert funding_analyzer.stats()["symbols_tracked"] == 0
    assert fake_event_bus.published == []


# ---------------------------------------------------------------------------
# Concurrent updates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_funding_updates_for_same_symbol_are_serialized_by_lock(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    analyzer = _make_analyzer(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=20,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=1.0,
        ),
        stub_regime_detector=stub_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    rates = [index * 0.00001 for index in range(1, 11)]

    await asyncio.gather(
        *[
            _run_funding_update(
                analyzer,
                make_event,
                make_funding_payload,
                funding_rate=rate,
                correlation_id=f"corr-concurrent-{index}",
            )
            for index, rate in enumerate(rates)
        ]
    )

    key = analyzer._make_key("BTCUSDT", "binance")
    history = analyzer._history[key]
    statistics = analyzer.get_statistics("BTCUSDT", "binance")

    assert len(history) == len(rates)
    assert sorted(snapshot.funding_rate for snapshot in history) == pytest.approx(sorted(rates))

    assert statistics is not None
    assert statistics.sample_size == len(rates)

    assert len(stub_regime_detector.calls) == len(rates)
    assert len(stub_pressure_analyzer.calls) == len(rates)
    assert len(stub_flip_detector.calls) == len(rates)
    assert len(stub_extremes_detector.calls) == len(rates)
    assert len(stub_divergence_detector.calls) == len(rates)

    assert analyzer._locks[key].locked() is False


@pytest.mark.asyncio
async def test_concurrent_funding_updates_for_different_symbols_keep_separate_locks_and_state(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    analyzer = _make_analyzer(
        fake_event_bus=fake_event_bus,
        fake_scheduler=fake_scheduler,
        config=FundingAnalyzerConfig(
            history_size=20,
            publish_signal_event=False,
            enable_cleanup_job=False,
            state_lock_timeout_sec=1.0,
        ),
        stub_regime_detector=stub_regime_detector,
        stub_pressure_analyzer=stub_pressure_analyzer,
        stub_flip_detector=stub_flip_detector,
        stub_extremes_detector=stub_extremes_detector,
        stub_divergence_detector=stub_divergence_detector,
    )

    await asyncio.gather(
        _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            symbol="BTCUSDT",
            exchange="binance",
            funding_rate=0.0001,
            correlation_id="corr-btc-binance",
        ),
        _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            symbol="ETHUSDT",
            exchange="binance",
            funding_rate=0.0002,
            correlation_id="corr-eth-binance",
        ),
        _run_funding_update(
            analyzer,
            make_event,
            make_funding_payload,
            symbol="BTCUSDT",
            exchange="bybit",
            funding_rate=-0.0003,
            correlation_id="corr-btc-bybit",
        ),
    )

    btc_binance_key = analyzer._make_key("BTCUSDT", "binance")
    eth_binance_key = analyzer._make_key("ETHUSDT", "binance")
    btc_bybit_key = analyzer._make_key("BTCUSDT", "bybit")

    assert set(analyzer._history.keys()) == {
        btc_binance_key,
        eth_binance_key,
        btc_bybit_key,
    }

    assert len(analyzer._history[btc_binance_key]) == 1
    assert len(analyzer._history[eth_binance_key]) == 1
    assert len(analyzer._history[btc_bybit_key]) == 1

    assert analyzer._history[btc_binance_key][-1].funding_rate == pytest.approx(0.0001)
    assert analyzer._history[eth_binance_key][-1].funding_rate == pytest.approx(0.0002)
    assert analyzer._history[btc_bybit_key][-1].funding_rate == pytest.approx(-0.0003)

    assert analyzer.stats()["symbols_tracked"] == 3
    assert analyzer.stats()["latest_statistics"] == 3
    assert analyzer.stats()["latest_regime_states"] == 3
    assert analyzer.stats()["latest_pressure_states"] == 3

    assert analyzer._locks[btc_binance_key].locked() is False
    assert analyzer._locks[eth_binance_key].locked() is False
    assert analyzer._locks[btc_bybit_key].locked() is False


# ---------------------------------------------------------------------------
# End-to-end malformed/context resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_context_events_do_not_block_later_valid_funding_update(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    malformed_event = make_event({}, topic="malformed")

    await funding_analyzer.on_open_interest(malformed_event)
    await funding_analyzer.on_candle(malformed_event)
    await funding_analyzer.on_trade(malformed_event)
    await funding_analyzer.on_cvd_update(malformed_event)
    await funding_analyzer.on_liquidation(malformed_event)

    assert funding_analyzer.stats()["contexts_tracked"] == 0

    await _run_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
    )

    assert funding_analyzer.get_latest_snapshot("BTCUSDT", "binance") is not None
    assert funding_analyzer.get_statistics("BTCUSDT", "binance") is not None
    assert funding_analyzer.config.analytics_updated_event_name in _published_topics(fake_event_bus)


@pytest.mark.asyncio
async def test_invalid_context_values_are_ignored_but_valid_values_later_work(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest="bad"),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )
    await funding_analyzer.on_candle(
        make_event(
            make_context_payload(close="bad", price="also-bad"),
            topic=funding_analyzer.config.candle_event_name,
        )
    )
    await funding_analyzer.on_trade(
        make_event(
            make_context_payload(price="bad"),
            topic=funding_analyzer.config.trade_event_name,
        )
    )
    await funding_analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd="bad", cumulative_volume_delta="also-bad"),
            topic=funding_analyzer.config.cvd_event_name,
        )
    )

    assert funding_analyzer.stats()["contexts_tracked"] == 0

    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest="1000000"),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )
    await funding_analyzer.on_candle(
        make_event(
            make_context_payload(close="50000"),
            topic=funding_analyzer.config.candle_event_name,
        )
    )
    await funding_analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd="12345"),
            topic=funding_analyzer.config.cvd_event_name,
        )
    )

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.latest_open_interest == pytest.approx(1_000_000.0)
    assert context.latest_price == pytest.approx(50_000.0)
    assert context.latest_cvd == pytest.approx(12_345.0)


def test_stats_returns_stable_shape(
    funding_analyzer: FundingAnalyzer,
) -> None:
    assert funding_analyzer.stats() == {
        "registered": False,
        "subscriptions": 0,
        "cleanup_job_id": None,
        "symbols_tracked": 0,
        "contexts_tracked": 0,
        "latest_statistics": 0,
        "latest_regime_states": 0,
        "latest_pressure_states": 0,
        "latest_flip_events": 0,
        "latest_extreme_events": 0,
        "latest_divergence_events": 0,
    }