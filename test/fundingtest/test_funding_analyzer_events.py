# tests/analytics/funding/test_funding_analyzer_events.py

from __future__ import annotations

from typing import Any, Callable

import pytest

from analytics.funding.enums import (
    FundingBias,
    FundingEventType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
)
from analytics.funding.funding_analyzer import FundingAnalyzer, FundingAnalyzerConfig
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _make_pressure_stub_signal_compatible(stub_pressure_analyzer: Any) -> None:
    """
    Backward-compatible helper for the conftest StubPressureAnalyzer.

    FundingAnalyzer._build_signals() викликає:
    - pressure_analyzer.is_high_pressure(...)
    - pressure_analyzer.is_squeeze_risk(...)
    - pressure_analyzer.build_summary(...)

    Якщо у StubPressureAnalyzer з conftest.py ще немає is_squeeze_risk,
    додаємо його динамічно для цих orchestration-тестів.
    """
    if not hasattr(stub_pressure_analyzer, "is_squeeze_risk"):
        stub_pressure_analyzer.is_squeeze_risk = lambda pressure_state, threshold=0.65: True


def _published_payloads(event_bus: Any, topic: str) -> list[dict[str, Any]]:
    return [event.payload for event in event_bus.published if event.topic == topic]


def _published_signal_payloads(analyzer: FundingAnalyzer, event_bus: Any) -> list[dict[str, Any]]:
    return _published_payloads(event_bus, analyzer.config.analytics_signal_event_name)


def _signal_types(analyzer: FundingAnalyzer, event_bus: Any) -> list[str]:
    payloads = _published_signal_payloads(analyzer, event_bus)
    return [payload["payload"]["signal_type"] for payload in payloads]


async def _run_single_funding_update(
    analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    *,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    funding_rate: float | str = 0.0001,
    correlation_id: str = "corr-funding-1",
) -> None:
    event = make_event(
        make_funding_payload(
            symbol=symbol,
            exchange=exchange,
            funding_rate=funding_rate,
        ),
        topic=analyzer.config.funding_event_name,
        correlation_id=correlation_id,
    )
    await analyzer.on_funding(event)


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------


def test_register_subscribes_to_all_expected_topics_and_adds_cleanup_job(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()

    assert funding_analyzer.stats()["registered"] is True
    assert funding_analyzer.stats()["subscriptions"] == 6

    subscribed_topics = [subscription.topic for subscription in fake_event_bus.subscriptions]

    assert subscribed_topics == [
        funding_analyzer.config.funding_event_name,
        funding_analyzer.config.open_interest_event_name,
        funding_analyzer.config.candle_event_name,
        funding_analyzer.config.trade_event_name,
        funding_analyzer.config.cvd_event_name,
        funding_analyzer.config.liquidation_event_name,
    ]

    subscribed_names = [subscription.name for subscription in fake_event_bus.subscriptions]

    assert subscribed_names == [
        "funding_analyzer.on_funding",
        "funding_analyzer.on_open_interest",
        "funding_analyzer.on_candle",
        "funding_analyzer.on_trade",
        "funding_analyzer.on_cvd_update",
        "funding_analyzer.on_liquidation",
    ]

    cleanup_job_id = funding_analyzer.stats()["cleanup_job_id"]

    assert cleanup_job_id is not None
    assert cleanup_job_id in fake_scheduler.jobs

    cleanup_job = fake_scheduler.jobs[cleanup_job_id]

    assert cleanup_job.name == funding_analyzer.config.cleanup_job_name
    assert cleanup_job.func == funding_analyzer.cleanup_stale_state
    assert cleanup_job.interval == funding_analyzer.config.cleanup_interval_sec
    assert cleanup_job.timeout == funding_analyzer.config.cleanup_timeout_sec
    assert cleanup_job.max_retries == 1
    assert cleanup_job.retry_delay == 1.0
    assert cleanup_job.allow_overlap is False
    assert cleanup_job.run_immediately is False
    assert cleanup_job.enabled is True


def test_register_is_idempotent_and_does_not_duplicate_subscriptions(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()
    first_cleanup_job_id = funding_analyzer.stats()["cleanup_job_id"]

    funding_analyzer.register()

    assert funding_analyzer.stats()["registered"] is True
    assert funding_analyzer.stats()["subscriptions"] == 6
    assert len(fake_event_bus.subscriptions) == 6
    assert funding_analyzer.stats()["cleanup_job_id"] == first_cleanup_job_id
    assert len(fake_scheduler.jobs) == 1


def test_register_reuses_existing_cleanup_job_by_name(
    funding_analyzer: FundingAnalyzer,
    fake_scheduler: Any,
) -> None:
    existing_job_id = fake_scheduler.add_interval_job(
        name=funding_analyzer.config.cleanup_job_name,
        func=funding_analyzer.cleanup_stale_state,
        interval=123.0,
        timeout=9.0,
        max_retries=0,
        retry_delay=0.0,
        allow_overlap=False,
        run_immediately=False,
        enabled=True,
    )

    funding_analyzer.register()

    assert funding_analyzer.stats()["cleanup_job_id"] == existing_job_id
    assert len(fake_scheduler.jobs) == 1


def test_register_without_cleanup_config_does_not_add_scheduler_job(
    funding_analyzer_no_cleanup: FundingAnalyzer,
    fake_event_bus: Any,
) -> None:
    funding_analyzer_no_cleanup.register()

    assert funding_analyzer_no_cleanup.stats()["registered"] is True
    assert funding_analyzer_no_cleanup.stats()["subscriptions"] == 6
    assert funding_analyzer_no_cleanup.stats()["cleanup_job_id"] is None
    assert len(fake_event_bus.subscriptions) == 6


def test_unregister_removes_subscriptions_and_disables_cleanup_job(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()

    cleanup_job_id = funding_analyzer.stats()["cleanup_job_id"]
    assert cleanup_job_id is not None

    funding_analyzer.unregister()

    assert funding_analyzer.stats()["registered"] is False
    assert funding_analyzer.stats()["subscriptions"] == 0
    assert len(fake_event_bus.unsubscribed) == 6
    assert fake_scheduler.disabled_job_ids == [cleanup_job_id]
    assert fake_scheduler.jobs[cleanup_job_id].enabled is False


def test_unregister_is_safe_when_not_registered(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.unregister()

    assert funding_analyzer.stats()["registered"] is False
    assert fake_event_bus.unsubscribed == []
    assert fake_scheduler.disabled_job_ids == []


def test_unregister_tolerates_missing_cleanup_job(
    funding_analyzer: FundingAnalyzer,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()

    cleanup_job_id = funding_analyzer.stats()["cleanup_job_id"]
    assert cleanup_job_id is not None

    fake_scheduler.jobs.pop(cleanup_job_id)

    funding_analyzer.unregister()

    assert funding_analyzer.stats()["registered"] is False
    assert funding_analyzer.stats()["subscriptions"] == 0


# ---------------------------------------------------------------------------
# context handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_open_interest_updates_previous_and_latest_values(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    first_event = make_event(
        make_context_payload(open_interest="1000000"),
        topic=funding_analyzer.config.open_interest_event_name,
    )
    second_event = make_event(
        make_context_payload(open_interest=1_050_000.0),
        topic=funding_analyzer.config.open_interest_event_name,
    )

    await funding_analyzer.on_open_interest(first_event)
    await funding_analyzer.on_open_interest(second_event)

    context = funding_analyzer.get_market_context("btcusdt", "binance")

    assert context is not None
    assert context.previous_open_interest == pytest.approx(1_000_000.0)
    assert context.latest_open_interest == pytest.approx(1_050_000.0)
    assert context.updated_at is not None


@pytest.mark.asyncio
async def test_on_open_interest_ignores_invalid_open_interest(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    event = make_event(
        make_context_payload(open_interest="not-a-number"),
        topic=funding_analyzer.config.open_interest_event_name,
    )

    await funding_analyzer.on_open_interest(event)

    assert funding_analyzer.get_market_context("BTCUSDT", "binance") is None
    assert funding_analyzer.stats()["contexts_tracked"] == 0


@pytest.mark.asyncio
async def test_on_candle_prefers_close_and_updates_price_context(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    first_event = make_event(
        make_context_payload(close="50000", price="49900"),
        topic=funding_analyzer.config.candle_event_name,
    )
    second_event = make_event(
        make_context_payload(close=50_100.0, price=49_900.0),
        topic=funding_analyzer.config.candle_event_name,
    )

    await funding_analyzer.on_candle(first_event)
    await funding_analyzer.on_candle(second_event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.previous_price == pytest.approx(50_000.0)
    assert context.latest_price == pytest.approx(50_100.0)
    assert context.updated_at is not None


@pytest.mark.asyncio
async def test_on_candle_falls_back_to_price_when_close_missing(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    event = make_event(
        make_context_payload(price="50123.45"),
        topic=funding_analyzer.config.candle_event_name,
    )

    await funding_analyzer.on_candle(event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.latest_price == pytest.approx(50_123.45)


@pytest.mark.asyncio
async def test_on_trade_updates_price_context(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    first_event = make_event(
        make_context_payload(price=50_000.0),
        topic=funding_analyzer.config.trade_event_name,
    )
    second_event = make_event(
        make_context_payload(price="50100"),
        topic=funding_analyzer.config.trade_event_name,
    )

    await funding_analyzer.on_trade(first_event)
    await funding_analyzer.on_trade(second_event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.previous_price == pytest.approx(50_000.0)
    assert context.latest_price == pytest.approx(50_100.0)


@pytest.mark.asyncio
async def test_on_cvd_update_supports_direct_payload(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    first_event = make_event(
        make_context_payload(cvd="10000"),
        topic=funding_analyzer.config.cvd_event_name,
    )
    second_event = make_event(
        make_context_payload(cumulative_volume_delta=25_000.0),
        topic=funding_analyzer.config.cvd_event_name,
    )

    await funding_analyzer.on_cvd_update(first_event)
    await funding_analyzer.on_cvd_update(second_event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.previous_cvd == pytest.approx(10_000.0)
    assert context.latest_cvd == pytest.approx(25_000.0)


@pytest.mark.asyncio
async def test_on_cvd_update_supports_nested_payload(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
) -> None:
    event = make_event(
        {
            "payload": {
                "symbol": "btcusdt",
                "exchange": "binance",
                "cvd": "12345.67",
            }
        },
        topic=funding_analyzer.config.cvd_event_name,
    )

    await funding_analyzer.on_cvd_update(event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.latest_cvd == pytest.approx(12_345.67)


@pytest.mark.asyncio
async def test_on_liquidation_updates_long_and_short_values_from_side_notional(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    long_event = make_event(
        make_context_payload(side="long", notional="150000"),
        topic=funding_analyzer.config.liquidation_event_name,
    )
    short_event = make_event(
        make_context_payload(side="short", qty="2", price="50000"),
        topic=funding_analyzer.config.liquidation_event_name,
    )

    await funding_analyzer.on_liquidation(long_event)
    await funding_analyzer.on_liquidation(short_event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.long_liquidations == pytest.approx(150_000.0)
    assert context.short_liquidations == pytest.approx(100_000.0)
    assert context.updated_at is not None
    assert context.liquidation_updated_at == context.updated_at


@pytest.mark.asyncio
async def test_on_liquidation_supports_aggregated_payload_without_side(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    event = make_event(
        make_context_payload(
            side="",
            long_liquidations="111000",
            short_liquidations="222000",
        ),
        topic=funding_analyzer.config.liquidation_event_name,
    )

    await funding_analyzer.on_liquidation(event)

    context = funding_analyzer.get_market_context("BTCUSDT", "binance")

    assert context is not None
    assert context.long_liquidations == pytest.approx(111_000.0)
    assert context.short_liquidations == pytest.approx(222_000.0)


@pytest.mark.asyncio
async def test_context_handlers_swallow_malformed_payloads_without_polluting_state(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
) -> None:
    malformed_event = make_event({}, topic="malformed")

    await funding_analyzer.on_open_interest(malformed_event)
    await funding_analyzer.on_candle(malformed_event)
    await funding_analyzer.on_trade(malformed_event)
    await funding_analyzer.on_cvd_update(malformed_event)
    await funding_analyzer.on_liquidation(malformed_event)

    assert funding_analyzer.stats()["contexts_tracked"] == 0


# ---------------------------------------------------------------------------
# on_funding pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_funding_processes_full_pipeline_and_publishes_all_events(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    make_context_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest=1_000_000.0),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )
    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest=1_100_000.0),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )
    await funding_analyzer.on_candle(
        make_event(
            make_context_payload(close=50_000.0),
            topic=funding_analyzer.config.candle_event_name,
        )
    )
    await funding_analyzer.on_candle(
        make_event(
            make_context_payload(close=50_010.0),
            topic=funding_analyzer.config.candle_event_name,
        )
    )
    await funding_analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd=10_000.0),
            topic=funding_analyzer.config.cvd_event_name,
        )
    )
    await funding_analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd=25_000.0),
            topic=funding_analyzer.config.cvd_event_name,
        )
    )
    await funding_analyzer.on_liquidation(
        make_event(
            make_context_payload(side="long", notional=150_000.0),
            topic=funding_analyzer.config.liquidation_event_name,
        )
    )

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.0001,
        correlation_id="corr-full-pipeline",
    )

    assert funding_analyzer.get_latest_snapshot("BTCUSDT", "binance") is not None
    assert funding_analyzer.get_statistics("BTCUSDT", "binance") is not None
    assert funding_analyzer.get_regime_state("BTCUSDT", "binance") is not None
    assert funding_analyzer.get_pressure_state("BTCUSDT", "binance") is not None

    stats = funding_analyzer.stats()

    assert stats["symbols_tracked"] == 1
    assert stats["contexts_tracked"] == 1
    assert stats["latest_statistics"] == 1
    assert stats["latest_regime_states"] == 1
    assert stats["latest_pressure_states"] == 1
    assert stats["latest_flip_events"] == 1
    assert stats["latest_extreme_events"] == 1
    assert stats["latest_divergence_events"] == 1

    assert len(stub_regime_detector.calls) == 1
    assert len(stub_pressure_analyzer.calls) == 1
    assert len(stub_flip_detector.calls) == 1
    assert len(stub_extremes_detector.calls) == 1
    assert len(stub_divergence_detector.calls) == 1

    snapshot = funding_analyzer.get_latest_snapshot("BTCUSDT", "binance")
    assert snapshot is not None
    assert snapshot.open_interest == pytest.approx(1_100_000.0)
    assert snapshot.mark_price == pytest.approx(50_010.0)

    pressure_call = stub_pressure_analyzer.calls[0]
    assert pressure_call["previous_open_interest"] == pytest.approx(1_000_000.0)
    assert pressure_call["current_price"] == pytest.approx(50_010.0)
    assert pressure_call["previous_price"] == pytest.approx(50_000.0)

    divergence_call = stub_divergence_detector.calls[0]
    assert divergence_call["price_change_pct"] == pytest.approx(0.0002)
    assert divergence_call["oi_change_pct"] == pytest.approx(0.10)
    assert divergence_call["cvd_change"] == pytest.approx(15_000.0)
    assert divergence_call["long_liquidations"] == pytest.approx(150_000.0)

    topics = fake_event_bus.topics()

    assert funding_analyzer.config.analytics_updated_event_name in topics
    assert funding_analyzer.config.analytics_regime_event_name in topics
    assert funding_analyzer.config.analytics_pressure_event_name in topics
    assert funding_analyzer.config.analytics_flip_event_name in topics
    assert funding_analyzer.config.analytics_extreme_event_name in topics
    assert funding_analyzer.config.analytics_divergence_event_name in topics
    assert topics.count(funding_analyzer.config.analytics_signal_event_name) == 5

    for published_event in fake_event_bus.published:
        assert published_event.kwargs["source"] == "funding_analyzer"
        assert published_event.kwargs["correlation_id"] == "corr-full-pipeline"

    updated_payload = fake_event_bus.last_payload_for(
        funding_analyzer.config.analytics_updated_event_name
    )

    assert updated_payload is not None
    assert updated_payload["event_type"] == FundingEventType.SNAPSHOT.value
    assert updated_payload["symbol"] == "BTCUSDT"
    assert updated_payload["exchange"] == "binance"
    assert updated_payload["source"] == funding_analyzer.SOURCE
    assert updated_payload["payload"]["snapshot"]["funding_rate"] == pytest.approx(0.0001)
    assert updated_payload["payload"]["statistics"]["sample_size"] == 1
    assert updated_payload["payload"]["regime_state"] is not None
    assert updated_payload["payload"]["pressure_state"] is not None
    assert updated_payload["payload"]["flip_event"] is not None
    assert updated_payload["payload"]["extreme_event"] is not None
    assert updated_payload["payload"]["divergence_event"] is not None


@pytest.mark.asyncio
async def test_on_funding_uses_previous_snapshot_and_previous_regime_state_on_second_update(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        funding_rate=-0.00008,
        correlation_id="corr-1",
    )
    first_regime_state = funding_analyzer.get_regime_state("BTCUSDT", "binance")

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        funding_rate=0.00012,
        correlation_id="corr-2",
    )

    assert len(stub_regime_detector.calls) == 2
    assert len(stub_pressure_analyzer.calls) == 2
    assert len(stub_flip_detector.calls) == 2

    second_regime_call = stub_regime_detector.calls[1]
    second_pressure_call = stub_pressure_analyzer.calls[1]
    second_flip_call = stub_flip_detector.calls[1]

    assert second_regime_call["previous_state"] is first_regime_state
    assert second_pressure_call["previous_snapshot"] is not None
    assert second_pressure_call["previous_snapshot"].funding_rate == pytest.approx(-0.00008)
    assert second_flip_call["previous_snapshot"] is not None
    assert second_flip_call["previous_snapshot"].funding_rate == pytest.approx(-0.00008)

    statistics = funding_analyzer.get_statistics("BTCUSDT", "binance")
    assert statistics is not None
    assert statistics.sample_size == 2
    assert statistics.current_rate == pytest.approx(0.00012)
    assert statistics.min_rate == pytest.approx(-0.00008)
    assert statistics.max_rate == pytest.approx(0.00012)


@pytest.mark.asyncio
async def test_on_funding_works_with_nested_payload(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    event = make_event(
        {
            "payload": make_funding_payload(
                symbol="ethusdt",
                exchange="binance",
                funding_rate="0.0002",
            )
        },
        topic=funding_analyzer.config.funding_event_name,
        correlation_id="corr-nested",
    )

    await funding_analyzer.on_funding(event)

    snapshot = funding_analyzer.get_latest_snapshot("ETHUSDT", "binance")

    assert snapshot is not None
    assert snapshot.symbol == "ETHUSDT"
    assert snapshot.funding_rate == pytest.approx(0.0002)

    assert funding_analyzer.config.analytics_updated_event_name in fake_event_bus.topics()


@pytest.mark.asyncio
async def test_on_funding_ignores_invalid_payload_without_publish(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
) -> None:
    event = make_event(
        {
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "funding_rate": "not-a-number",
        },
        topic=funding_analyzer.config.funding_event_name,
    )

    await funding_analyzer.on_funding(event)

    assert funding_analyzer.get_latest_snapshot("BTCUSDT", "binance") is None
    assert funding_analyzer.stats()["symbols_tracked"] == 0
    assert fake_event_bus.published == []


@pytest.mark.asyncio
async def test_on_funding_ignores_missing_symbol_without_publish(
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
    assert fake_event_bus.published == []


# ---------------------------------------------------------------------------
# publishing flags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_updated_event_flag_disables_updated_event_only(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    funding_analyzer.config.publish_updated_event = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    topics = fake_event_bus.topics()

    assert funding_analyzer.config.analytics_updated_event_name not in topics
    assert funding_analyzer.config.analytics_regime_event_name in topics
    assert funding_analyzer.config.analytics_pressure_event_name in topics
    assert funding_analyzer.config.analytics_flip_event_name in topics
    assert funding_analyzer.config.analytics_extreme_event_name in topics
    assert funding_analyzer.config.analytics_divergence_event_name in topics


@pytest.mark.asyncio
async def test_publish_signal_event_flag_disables_all_signal_events(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    funding_analyzer.config.publish_signal_event = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    topics = fake_event_bus.topics()

    assert funding_analyzer.config.analytics_signal_event_name not in topics
    assert funding_analyzer.config.analytics_updated_event_name in topics
    assert funding_analyzer.config.analytics_regime_event_name in topics
    assert funding_analyzer.config.analytics_pressure_event_name in topics


@pytest.mark.asyncio
async def test_regime_event_is_not_published_when_unchanged_and_not_forced(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_regime_detector.state.changed = False
    funding_analyzer.config.publish_regime_event_on_every_update = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    assert funding_analyzer.config.analytics_regime_event_name not in fake_event_bus.topics()


@pytest.mark.asyncio
async def test_regime_event_is_published_when_forced_even_if_unchanged(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_regime_detector.state.changed = False
    funding_analyzer.config.publish_regime_event_on_every_update = True

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    assert funding_analyzer.config.analytics_regime_event_name in fake_event_bus.topics()


@pytest.mark.asyncio
async def test_pressure_event_is_not_published_when_low_pressure_and_not_forced(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_pressure_analyzer.high_pressure = False
    funding_analyzer.config.publish_pressure_event_on_every_update = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    assert funding_analyzer.config.analytics_pressure_event_name not in fake_event_bus.topics()


@pytest.mark.asyncio
async def test_pressure_event_is_published_when_forced_even_if_low_pressure(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_pressure_analyzer.high_pressure = False
    funding_analyzer.config.publish_pressure_event_on_every_update = True

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    assert funding_analyzer.config.analytics_pressure_event_name in fake_event_bus.topics()


@pytest.mark.asyncio
async def test_none_detector_events_are_not_published_or_cached(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_flip_detector.event = None
    stub_extremes_detector.event = None
    stub_divergence_detector.event = None

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    topics = fake_event_bus.topics()
    stats = funding_analyzer.stats()

    assert funding_analyzer.config.analytics_flip_event_name not in topics
    assert funding_analyzer.config.analytics_extreme_event_name not in topics
    assert funding_analyzer.config.analytics_divergence_event_name not in topics

    assert stats["latest_flip_events"] == 0
    assert stats["latest_extreme_events"] == 0
    assert stats["latest_divergence_events"] == 0

    updated_payload = fake_event_bus.last_payload_for(
        funding_analyzer.config.analytics_updated_event_name
    )

    assert updated_payload is not None
    assert updated_payload["payload"]["flip_event"] is None
    assert updated_payload["payload"]["extreme_event"] is None
    assert updated_payload["payload"]["divergence_event"] is None


# ---------------------------------------------------------------------------
# signal publishing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_events_are_built_for_all_enabled_signal_sources(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert signal_types == [
        FundingSignalType.REGIME_CHANGE.value,
        FundingSignalType.SQUEEZE_WARNING.value,
        FundingSignalType.FLIP_DETECTED.value,
        FundingSignalType.SQUEEZE_WARNING.value,
        FundingSignalType.DIVERGENCE_DETECTED.value,
    ]

    signal_payloads = _published_signal_payloads(funding_analyzer, fake_event_bus)

    assert len(signal_payloads) == 5

    for signal_event_payload in signal_payloads:
        assert signal_event_payload["event_type"] == FundingEventType.SIGNAL.value
        assert signal_event_payload["symbol"] == "BTCUSDT"
        assert signal_event_payload["exchange"] == "binance"
        assert signal_event_payload["source"] == funding_analyzer.SOURCE
        assert "payload" in signal_event_payload
        assert -1.0 <= signal_event_payload["payload"]["score"] <= 1.0
        assert 0.0 <= signal_event_payload["payload"]["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_signal_priority_is_high_for_signal_events(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_events = [
        event
        for event in fake_event_bus.published
        if event.topic == funding_analyzer.config.analytics_signal_event_name
    ]

    assert signal_events

    for event in signal_events:
        assert event.kwargs["priority"].name == "HIGH"


@pytest.mark.asyncio
async def test_signal_source_flags_disable_individual_signal_types(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    funding_analyzer.config.signal_on_regime_change = False
    funding_analyzer.config.signal_on_high_pressure = False
    funding_analyzer.config.signal_on_extreme = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert FundingSignalType.REGIME_CHANGE.value not in signal_types
    assert FundingSignalType.SQUEEZE_WARNING.value not in signal_types
    assert FundingSignalType.CROWDING_WARNING.value not in signal_types

    assert signal_types == [
        FundingSignalType.FLIP_DETECTED.value,
        FundingSignalType.DIVERGENCE_DETECTED.value,
    ]


@pytest.mark.asyncio
async def test_flip_signal_can_be_disabled_independently(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    funding_analyzer.config.signal_on_flip = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert FundingSignalType.FLIP_DETECTED.value not in signal_types


@pytest.mark.asyncio
async def test_divergence_signal_can_be_disabled_independently(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    funding_analyzer.config.signal_on_divergence = False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert FundingSignalType.DIVERGENCE_DETECTED.value not in signal_types


@pytest.mark.asyncio
async def test_pressure_signal_uses_crowding_warning_when_not_squeeze_risk(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    stub_pressure_analyzer.is_squeeze_risk = lambda pressure_state, threshold=0.65: False

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert FundingSignalType.CROWDING_WARNING.value in signal_types


@pytest.mark.asyncio
async def test_extreme_signal_uses_reversion_setup_when_not_squeeze_risk(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_extremes_detector: Any,
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    assert stub_extremes_detector.event is not None
    stub_extremes_detector.event.is_squeeze_risk = False
    stub_extremes_detector.event.is_reversal_risk = True

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    signal_types = _signal_types(funding_analyzer, fake_event_bus)

    assert FundingSignalType.REVERSION_SETUP.value in signal_types


# ---------------------------------------------------------------------------
# payload contract checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_pressure_flip_extreme_divergence_event_payload_contracts(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        correlation_id="corr-contracts",
    )

    expected_event_types_by_topic = {
        funding_analyzer.config.analytics_regime_event_name: FundingEventType.REGIME.value,
        funding_analyzer.config.analytics_pressure_event_name: FundingEventType.PRESSURE.value,
        funding_analyzer.config.analytics_flip_event_name: FundingEventType.FLIP.value,
        funding_analyzer.config.analytics_extreme_event_name: FundingEventType.EXTREME.value,
        funding_analyzer.config.analytics_divergence_event_name: FundingEventType.DIVERGENCE.value,
    }

    for topic, expected_event_type in expected_event_types_by_topic.items():
        payloads = _published_payloads(fake_event_bus, topic)

        assert len(payloads) == 1

        payload = payloads[0]

        assert payload["event_type"] == expected_event_type
        assert payload["symbol"] == "BTCUSDT"
        assert payload["exchange"] == "binance"
        assert payload["timeframe"] == funding_analyzer.config.default_timeframe.value
        assert payload["source"] == funding_analyzer.SOURCE
        assert payload["payload"] is not None
        assert payload["event_time"] is not None


# ---------------------------------------------------------------------------
# custom config topics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_config_topics_are_used_for_subscribe_and_publish(
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
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    config = FundingAnalyzerConfig(
        funding_event_name="custom.market.funding",
        open_interest_event_name="custom.market.open_interest",
        candle_event_name="custom.market.candle",
        trade_event_name="custom.market.trade",
        cvd_event_name="custom.analytics.cvd",
        liquidation_event_name="custom.market.liquidation",
        analytics_updated_event_name="custom.analytics.funding.updated",
        analytics_regime_event_name="custom.analytics.funding.regime",
        analytics_pressure_event_name="custom.analytics.funding.pressure",
        analytics_flip_event_name="custom.analytics.funding.flip",
        analytics_extreme_event_name="custom.analytics.funding.extreme",
        analytics_divergence_event_name="custom.analytics.funding.divergence",
        analytics_signal_event_name="custom.analytics.funding.signal",
        enable_cleanup_job=True,
    )

    analyzer = FundingAnalyzer(
        event_bus=fake_event_bus,  # type: ignore[arg-type]
        scheduler=fake_scheduler,  # type: ignore[arg-type]
        config=config,
        regime_detector=stub_regime_detector,  # type: ignore[arg-type]
        pressure_analyzer=stub_pressure_analyzer,  # type: ignore[arg-type]
        flip_detector=stub_flip_detector,  # type: ignore[arg-type]
        extremes_detector=stub_extremes_detector,  # type: ignore[arg-type]
        divergence_detector=stub_divergence_detector,  # type: ignore[arg-type]
    )

    analyzer.register()

    assert [subscription.topic for subscription in fake_event_bus.subscriptions] == [
        "custom.market.funding",
        "custom.market.open_interest",
        "custom.market.candle",
        "custom.market.trade",
        "custom.analytics.cvd",
        "custom.market.liquidation",
    ]

    await _run_single_funding_update(
        analyzer,
        make_event,
        make_funding_payload,
        correlation_id="corr-custom-topics",
    )

    topics = fake_event_bus.topics()

    assert "custom.analytics.funding.updated" in topics
    assert "custom.analytics.funding.regime" in topics
    assert "custom.analytics.funding.pressure" in topics
    assert "custom.analytics.funding.flip" in topics
    assert "custom.analytics.funding.extreme" in topics
    assert "custom.analytics.funding.divergence" in topics
    assert "custom.analytics.funding.signal" in topics


# ---------------------------------------------------------------------------
# explicit state variants through stub outputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_funding_publishes_updated_event_with_none_optional_events_when_stubs_return_none(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
    stub_pressure_analyzer: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_flip_detector.event = None
    stub_extremes_detector.event = None
    stub_divergence_detector.event = None

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
    )

    updated_payload = fake_event_bus.last_payload_for(
        funding_analyzer.config.analytics_updated_event_name
    )

    assert updated_payload is not None
    assert updated_payload["payload"]["snapshot"] is not None
    assert updated_payload["payload"]["statistics"] is not None
    assert updated_payload["payload"]["regime_state"] is not None
    assert updated_payload["payload"]["pressure_state"] is not None
    assert updated_payload["payload"]["flip_event"] is None
    assert updated_payload["payload"]["extreme_event"] is None
    assert updated_payload["payload"]["divergence_event"] is None


@pytest.mark.asyncio
async def test_on_funding_can_emit_short_pressure_positive_signal_score(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_pressure_analyzer: Any,
    stub_regime_detector: Any,
) -> None:
    _make_pressure_stub_signal_compatible(stub_pressure_analyzer)

    stub_regime_detector.state.bias = FundingBias.SQUEEZE_RISK_SHORTS
    stub_regime_detector.state.regime = FundingRegime.EXTREME_NEGATIVE
    stub_regime_detector.state.current_rate = -0.0002
    stub_regime_detector.state.changed = True

    stub_pressure_analyzer.state.direction = FundingPressureDirection.SHORT
    stub_pressure_analyzer.state.level = FundingPressureLevel.HIGH
    stub_pressure_analyzer.state.bias = FundingBias.SQUEEZE_RISK_SHORTS
    stub_pressure_analyzer.state.pressure_score = 0.8

    await _run_single_funding_update(
        funding_analyzer,
        make_event,
        make_funding_payload,
        funding_rate=-0.0002,
    )

    pressure_signal_payloads = [
        payload["payload"]
        for payload in _published_signal_payloads(funding_analyzer, fake_event_bus)
        if payload["payload"]["signal_type"]
        in {
            FundingSignalType.SQUEEZE_WARNING.value,
            FundingSignalType.CROWDING_WARNING.value,
        }
    ]

    assert pressure_signal_payloads

    pressure_signal = pressure_signal_payloads[0]

    assert pressure_signal["score"] == pytest.approx(0.8)
    assert pressure_signal["bias"] == FundingBias.SQUEEZE_RISK_SHORTS.value
    assert pressure_signal["regime"] == FundingRegime.EXTREME_NEGATIVE.value