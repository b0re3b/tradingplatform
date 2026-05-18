# tests/analytics/funding/test_funding_analyzer_events.py

from __future__ import annotations

from typing import Any, Callable

import pytest

from core.event_bus import EventPriority

from analytics.funding.config import FundingAnalyzerConfig
from analytics.funding.enums import (
    FundingBias,
    FundingDataSource,
    FundingEventType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)
from analytics.funding.funding_analyzer import FundingAnalyzer
from analytics.funding.models import (
    FundingDivergenceEvent,
    FundingExtremeEvent,
    FundingFlipEvent,
    FundingPressureState,
    FundingRegimeState,
)


# =============================================================================
# Local assertions / helpers
# =============================================================================

def _topics(event_bus: Any) -> list[str]:
    return [event.topic for event in event_bus.published]


def _payloads_for(event_bus: Any, topic: str) -> list[dict[str, Any]]:
    return [event.payload for event in event_bus.published if event.topic == topic]


def _last_payload_for(event_bus: Any, topic: str) -> dict[str, Any]:
    payloads = _payloads_for(event_bus, topic)
    assert payloads, f"No payloads were published for topic={topic!r}. Topics={_topics(event_bus)}"
    return payloads[-1]


def _events_for(event_bus: Any, topic: str) -> list[Any]:
    return [event for event in event_bus.published if event.topic == topic]


def _last_event_for(event_bus: Any, topic: str) -> Any:
    events = _events_for(event_bus, topic)
    assert events, f"No events were published for topic={topic!r}. Topics={_topics(event_bus)}"
    return events[-1]


def _analytics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("payload")
    assert isinstance(nested, dict), payload
    return nested


def _signal_payloads(analyzer: FundingAnalyzer, event_bus: Any) -> list[dict[str, Any]]:
    return [
        _analytics_payload(payload)
        for payload in _payloads_for(event_bus, analyzer.config.signal_event_name)
    ]


def _signal_types(analyzer: FundingAnalyzer, event_bus: Any) -> set[str]:
    return {
        payload["signal_type"]
        for payload in _signal_payloads(analyzer, event_bus)
    }


def _assert_event_bus_emit_contract(
    *,
    event: Any,
    expected_source: str,
    expected_correlation_id: str | None,
    expected_priority: EventPriority,
    expected_scope_fragment: str = "BTCUSDT",
) -> None:
    assert event.kwargs["source"] == expected_source
    assert event.kwargs["correlation_id"] == expected_correlation_id
    assert event.kwargs["priority"] == expected_priority

    headers = event.kwargs.get("headers")
    assert isinstance(headers, dict), event.kwargs
    assert "scope" in headers
    assert expected_scope_fragment in str(headers["scope"])


def _assert_scoped_analytics_event_payload(
    *,
    payload: dict[str, Any],
    expected_event_type: str,
    expected_symbol: str = "BTCUSDT",
    expected_exchange: str = "binance",
    expected_market_type: str = "usdm_futures",
    expected_timeframe: str = "1h",
    expected_exchange_symbol: str = "BTCUSDT",
    expected_source: str = FundingAnalyzer.SOURCE,
) -> None:
    assert payload["event_type"] == expected_event_type
    assert payload["symbol"] == expected_symbol
    assert payload["exchange"] == expected_exchange
    assert payload["market_type"] == expected_market_type
    assert payload["timeframe"] == expected_timeframe
    assert payload["exchange_symbol"] == expected_exchange_symbol
    assert payload["source"] == expected_source
    assert payload["event_time"] is not None

    nested = payload["payload"]
    assert isinstance(nested, dict)
    assert nested["symbol"] == expected_symbol
    assert nested["exchange"] == expected_exchange
    assert nested["market_type"] == expected_market_type
    assert nested["timeframe"] == expected_timeframe
    assert nested["exchange_symbol"] == expected_exchange_symbol

    scope = nested.get("scope")
    assert scope == {
        "exchange": expected_exchange,
        "market_type": expected_market_type,
        "symbol": expected_symbol,
        "timeframe": expected_timeframe,
    }


async def _apply_full_context(
    *,
    analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest=1_000_000.0),
            topic=analyzer.config.open_interest_event_name,
            correlation_id="corr-oi-1",
        )
    )
    await analyzer.on_open_interest(
        make_event(
            make_context_payload(open_interest=1_100_000.0),
            topic=analyzer.config.open_interest_event_name,
            correlation_id="corr-oi-2",
        )
    )

    await analyzer.on_candle(
        make_event(
            make_context_payload(close=50_000.0, price=49_000.0),
            topic=analyzer.config.candle_event_name,
            correlation_id="corr-candle-1",
        )
    )
    await analyzer.on_candle(
        make_event(
            make_context_payload(close=50_010.0, price=49_000.0),
            topic=analyzer.config.candle_event_name,
            correlation_id="corr-candle-2",
        )
    )

    await analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd=10_000.0),
            topic=analyzer.config.cvd_event_name,
            correlation_id="corr-cvd-1",
        )
    )
    await analyzer.on_cvd_update(
        make_event(
            make_context_payload(cvd=25_000.0),
            topic=analyzer.config.cvd_event_name,
            correlation_id="corr-cvd-2",
        )
    )

    await analyzer.on_liquidation(
        make_event(
            make_context_payload(side="long", notional=150_000.0),
            topic=analyzer.config.liquidation_event_name,
            correlation_id="corr-liq-long",
        )
    )
    await analyzer.on_liquidation(
        make_event(
            make_context_payload(side="short", qty=2, price=50_000.0, notional=None),
            topic=analyzer.config.liquidation_event_name,
            correlation_id="corr-liq-short",
        )
    )


# =============================================================================
# register / unregister / lifecycle
# =============================================================================

def test_register_subscribes_only_to_production_topics_by_default_and_adds_expected_jobs(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()

    assert funding_analyzer.stats()["registered"] is True
    assert funding_analyzer.stats()["subscriptions"] == 6

    assert [subscription.topic for subscription in fake_event_bus.subscriptions] == [
        funding_analyzer.config.funding_event_name,
        funding_analyzer.config.open_interest_event_name,
        funding_analyzer.config.candle_event_name,
        funding_analyzer.config.trade_event_name,
        funding_analyzer.config.cvd_event_name,
        funding_analyzer.config.liquidation_event_name,
    ]

    assert [subscription.name for subscription in fake_event_bus.subscriptions] == [
        "funding_analyzer.on_funding",
        "funding_analyzer.on_open_interest",
        "funding_analyzer.on_candle",
        "funding_analyzer.on_trade",
        "funding_analyzer.on_cvd_update",
        "funding_analyzer.on_liquidation",
    ]

    assert all(not subscription.topic.startswith("market.") or subscription.topic.endswith(".updated") or subscription.topic == "market.candle.closed"
               for subscription in fake_event_bus.subscriptions)

    stats = funding_analyzer.stats()

    assert stats["cleanup_job_id"] is not None
    assert stats["heartbeat_job_id"] is not None

    # У hard-test config parquet вимкнений, тому parquet job не має з'являтися.
    assert stats["parquet_flush_job_id"] is None

    cleanup_job = fake_scheduler.jobs[stats["cleanup_job_id"]]
    heartbeat_job = fake_scheduler.jobs[stats["heartbeat_job_id"]]

    # Ці assertions спеціально жорсткі: зараз FundingAnalyzer hardcode-ить job names.
    # Якщо config має cleanup_job_name/heartbeat_job_name, але analyzer їх ігнорує — тест це покаже.
    assert cleanup_job.name == funding_analyzer.config.cleanup_job_name
    assert cleanup_job.func == funding_analyzer.cleanup_stale_state
    assert cleanup_job.interval == funding_analyzer.config.cleanup_interval_sec
    assert cleanup_job.timeout == min(30.0, max(1.0, funding_analyzer.config.cleanup_interval_sec))
    assert cleanup_job.max_retries == 1
    assert cleanup_job.retry_delay == 1.0
    assert cleanup_job.allow_overlap is False
    assert cleanup_job.run_immediately is False
    assert cleanup_job.enabled is True

    assert heartbeat_job.name == funding_analyzer.config.heartbeat_job_name
    assert heartbeat_job.func == funding_analyzer.emit_heartbeat
    assert heartbeat_job.interval == funding_analyzer.config.heartbeat_interval_sec
    assert heartbeat_job.timeout == 5.0
    assert heartbeat_job.max_retries == 0
    assert heartbeat_job.allow_overlap is False
    assert heartbeat_job.enabled is True


def test_register_is_idempotent_and_does_not_duplicate_subscriptions_or_jobs(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()
    first_stats = funding_analyzer.stats()

    funding_analyzer.register()
    second_stats = funding_analyzer.stats()

    assert first_stats["registered"] is True
    assert second_stats["registered"] is True
    assert second_stats["subscriptions"] == 6
    assert len(fake_event_bus.subscriptions) == 6

    assert second_stats["cleanup_job_id"] == first_stats["cleanup_job_id"]
    assert second_stats["heartbeat_job_id"] == first_stats["heartbeat_job_id"]

    # cleanup + heartbeat only
    assert len(fake_scheduler.jobs) == 2


def test_register_respects_disabled_context_subscriptions(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        use_open_interest_context=False,
        use_price_context=False,
        use_trades_context=False,
        use_cvd_context=False,
        use_liquidation_context=False,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    analyzer.register()

    assert [subscription.topic for subscription in fake_event_bus.subscriptions] == [
        analyzer.config.funding_event_name,
    ]
    assert [subscription.name for subscription in fake_event_bus.subscriptions] == [
        "funding_analyzer.on_funding",
    ]
    assert analyzer.stats()["subscriptions"] == 1

    # Scheduler jobs не залежать від context subscriptions.
    assert analyzer.stats()["cleanup_job_id"] is not None
    assert analyzer.stats()["heartbeat_job_id"] is not None


def test_register_includes_legacy_raw_topics_only_when_explicitly_enabled(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        allow_legacy_raw_topics=True,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    analyzer.register()

    assert [subscription.topic for subscription in fake_event_bus.subscriptions] == [
        analyzer.config.funding_event_name,
        analyzer.config.open_interest_event_name,
        analyzer.config.candle_event_name,
        analyzer.config.trade_event_name,
        analyzer.config.cvd_event_name,
        analyzer.config.liquidation_event_name,
        analyzer.config.raw_funding_event_name,
        analyzer.config.raw_open_interest_event_name,
        analyzer.config.raw_candle_event_name,
        analyzer.config.raw_trade_event_name,
        analyzer.config.raw_liquidation_event_name,
    ]

    assert [subscription.name for subscription in fake_event_bus.subscriptions][-5:] == [
        "funding_analyzer.on_raw_funding",
        "funding_analyzer.on_raw_open_interest",
        "funding_analyzer.on_raw_candle",
        "funding_analyzer.on_raw_trade",
        "funding_analyzer.on_raw_liquidation",
    ]

    assert analyzer.stats()["allow_legacy_raw_topics"] is True
    assert analyzer.stats()["legacy_raw_input_topics"] == [
        "market.funding",
        "market.open_interest",
        "market.candle",
        "market.trade",
        "market.liquidation",
    ]


def test_register_skips_everything_when_disabled(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    config = FundingAnalyzerConfig(
        enabled=False,
        default_market_type="usdm_futures",
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    analyzer.register()

    assert analyzer.stats()["registered"] is False
    assert analyzer.stats()["subscriptions"] == 0
    assert fake_event_bus.subscriptions == []
    assert fake_scheduler.jobs == {}


def test_unregister_removes_all_subscriptions_and_disables_all_registered_jobs(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    funding_analyzer.register()

    cleanup_job_id = funding_analyzer.stats()["cleanup_job_id"]
    heartbeat_job_id = funding_analyzer.stats()["heartbeat_job_id"]

    assert cleanup_job_id is not None
    assert heartbeat_job_id is not None

    funding_analyzer.unregister()

    assert funding_analyzer.stats()["registered"] is False
    assert funding_analyzer.stats()["subscriptions"] == 0
    assert len(fake_event_bus.unsubscribed) == 6

    assert fake_scheduler.disabled_job_ids == [cleanup_job_id, heartbeat_job_id]
    assert fake_scheduler.jobs[cleanup_job_id].enabled is False
    assert fake_scheduler.jobs[heartbeat_job_id].enabled is False

    assert funding_analyzer.stats()["cleanup_job_id"] is None
    assert funding_analyzer.stats()["heartbeat_job_id"] is None


@pytest.mark.asyncio
async def test_start_and_stop_emit_lifecycle_events_and_do_not_double_register(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    fake_scheduler: Any,
) -> None:
    await funding_analyzer.start()
    await funding_analyzer.start()

    assert funding_analyzer.stats()["started"] is True
    assert funding_analyzer.stats()["registered"] is True
    assert len(fake_event_bus.subscriptions) == 6

    started_events = _events_for(fake_event_bus, funding_analyzer.config.analyzer_started_event_name)
    assert len(started_events) == 1
    assert started_events[0].kwargs["priority"] == EventPriority.LOW
    assert started_events[0].kwargs["source"] == funding_analyzer.SOURCE

    started_payload = started_events[0].payload
    assert started_payload["service_name"] == funding_analyzer.config.service_name
    assert started_payload["scope"] == "exchange:market_type:symbol:timeframe"
    assert started_payload["production_input_topics"] == list(funding_analyzer.config.production_input_topics)

    await funding_analyzer.stop()

    assert funding_analyzer.stats()["started"] is False
    assert funding_analyzer.stats()["registered"] is False
    assert len(fake_scheduler.disabled_job_ids) == 2

    stopped_events = _events_for(fake_event_bus, funding_analyzer.config.analyzer_stopped_event_name)
    assert len(stopped_events) == 1
    assert stopped_events[0].kwargs["priority"] == EventPriority.LOW
    assert stopped_events[0].kwargs["source"] == funding_analyzer.SOURCE
    assert stopped_events[0].payload["service_name"] == funding_analyzer.config.service_name
    assert stopped_events[0].payload["stats"]["started"] is False


# =============================================================================
# Context event handlers
# =============================================================================

@pytest.mark.asyncio
async def test_context_handlers_keep_futures_scope_separated_by_market_type(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(
                market_type="usdm_futures",
                open_interest=1_000_000.0,
            ),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )
    await funding_analyzer.on_open_interest(
        make_event(
            make_context_payload(
                market_type="coinm_futures",
                open_interest=2_000_000.0,
            ),
            topic=funding_analyzer.config.open_interest_event_name,
        )
    )

    usdm_context = funding_analyzer.get_market_context(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    coinm_context = funding_analyzer.get_market_context(
        "BTCUSDT",
        "binance",
        market_type="coinm_futures",
        timeframe="1h",
    )

    assert usdm_context is not None
    assert coinm_context is not None
    assert usdm_context is not coinm_context
    assert usdm_context.latest_open_interest == pytest.approx(1_000_000.0)
    assert coinm_context.latest_open_interest == pytest.approx(2_000_000.0)
    assert funding_analyzer.stats()["contexts_tracked"] == 2


@pytest.mark.asyncio
async def test_context_handlers_update_previous_latest_values_and_ignore_invalid_payloads(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_open_interest(
        make_event(make_context_payload(open_interest="1000000"))
    )
    await funding_analyzer.on_open_interest(
        make_event(make_context_payload(open_interest_value="1100000"))
    )
    await funding_analyzer.on_open_interest(
        make_event(make_context_payload(open_interest="not-a-number"))
    )

    await funding_analyzer.on_candle(
        make_event(make_context_payload(close="50000", price="1"))
    )
    await funding_analyzer.on_candle(
        make_event(make_context_payload(close=None, price="50010"))
    )
    await funding_analyzer.on_candle(
        make_event(make_context_payload(close=None, price="bad"))
    )

    await funding_analyzer.on_cvd_update(
        make_event(make_context_payload(cvd="10000"))
    )
    await funding_analyzer.on_cvd_update(
        make_event(make_context_payload(cvd=None, cumulative_volume_delta="25000"))
    )
    await funding_analyzer.on_cvd_update(
        make_event(make_context_payload(cvd="bad", cumulative_volume_delta=None))
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
    assert context.latest_price == pytest.approx(50_010.0)
    assert context.previous_cvd == pytest.approx(10_000.0)
    assert context.latest_cvd == pytest.approx(25_000.0)
    assert context.updated_at is not None


@pytest.mark.asyncio
async def test_trade_handler_supports_nested_trade_payload_and_keeps_outer_scope_as_fallback(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_nested_trade_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_trade(
        make_event(
            make_nested_trade_payload(price="49999.5"),
            topic=funding_analyzer.config.trade_event_name,
        )
    )
    await funding_analyzer.on_trade(
        make_event(
            make_nested_trade_payload(price="50001.5"),
            topic=funding_analyzer.config.trade_event_name,
        )
    )

    context = funding_analyzer.get_market_context(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert context is not None
    assert context.previous_price == pytest.approx(49_999.5)
    assert context.latest_price == pytest.approx(50_001.5)


@pytest.mark.asyncio
async def test_liquidation_handler_supports_side_notional_qty_price_and_aggregated_payloads(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
    make_nested_liquidation_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_liquidation(
        make_event(
            make_context_payload(side="long", notional="150000"),
            topic=funding_analyzer.config.liquidation_event_name,
        )
    )
    await funding_analyzer.on_liquidation(
        make_event(
            make_nested_liquidation_payload(side="short", qty="2", price="50000", notional=None),
            topic=funding_analyzer.config.liquidation_event_name,
        )
    )

    context = funding_analyzer.get_market_context(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert context is not None
    assert context.long_liquidations == pytest.approx(150_000.0)
    assert context.short_liquidations == pytest.approx(100_000.0)
    assert context.liquidation_updated_at is not None
    assert context.updated_at == context.liquidation_updated_at

    await funding_analyzer.on_liquidation(
        make_event(
            make_context_payload(
                side="",
                long_liquidations="111000",
                short_liquidations="222000",
            ),
            topic=funding_analyzer.config.liquidation_event_name,
        )
    )

    assert context.long_liquidations == pytest.approx(111_000.0)
    assert context.short_liquidations == pytest.approx(222_000.0)


@pytest.mark.asyncio
async def test_context_handlers_do_not_pollute_state_when_scope_is_missing_or_disallowed(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_context_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        allowed_symbols={"ETHUSDT"},
        min_samples_for_statistics=1,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_open_interest(
        make_event(
            make_context_payload(symbol="BTCUSDT", open_interest=1_000_000.0),
            topic=analyzer.config.open_interest_event_name,
        )
    )
    await analyzer.on_candle(
        make_event(
            {},
            topic=analyzer.config.candle_event_name,
        )
    )
    await analyzer.on_trade(
        make_event(
            {"trade": {"price": "50000"}},
            topic=analyzer.config.trade_event_name,
        )
    )

    assert analyzer.stats()["contexts_tracked"] == 0


# =============================================================================
# on_funding full pipeline
# =============================================================================

@pytest.mark.asyncio
async def test_on_funding_full_pipeline_publishes_all_analytics_events_with_strict_contract(
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
    await _apply_full_context(
        analyzer=funding_analyzer,
        make_event=make_event,
        make_context_payload=make_context_payload,
    )

    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(
                funding_rate=0.00035,
                mark_price=None,
                open_interest=None,
                metadata={"origin": "funding-cache", "test_case": "full-pipeline"},
            ),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-full-pipeline",
        )
    )

    topics = _topics(fake_event_bus)

    assert topics.count(funding_analyzer.config.analytics_event_name) == 1
    assert topics.count(funding_analyzer.config.snapshot_event_name) == 1
    assert topics.count(funding_analyzer.config.regime_event_name) == 1
    assert topics.count(funding_analyzer.config.pressure_event_name) == 1
    assert topics.count(funding_analyzer.config.flip_event_name) == 1
    assert topics.count(funding_analyzer.config.extreme_event_name) == 1
    assert topics.count(funding_analyzer.config.divergence_event_name) == 1
    assert topics.count(funding_analyzer.config.signal_event_name) >= 5

    assert topics[:7] == [
        funding_analyzer.config.analytics_event_name,
        funding_analyzer.config.snapshot_event_name,
        funding_analyzer.config.regime_event_name,
        funding_analyzer.config.pressure_event_name,
        funding_analyzer.config.flip_event_name,
        funding_analyzer.config.extreme_event_name,
        funding_analyzer.config.divergence_event_name,
    ]

    expected_topic_event_type = {
        funding_analyzer.config.snapshot_event_name: FundingEventType.SNAPSHOT.value,
        funding_analyzer.config.regime_event_name: FundingEventType.REGIME.value,
        funding_analyzer.config.pressure_event_name: FundingEventType.PRESSURE.value,
        funding_analyzer.config.flip_event_name: FundingEventType.FLIP.value,
        funding_analyzer.config.extreme_event_name: FundingEventType.EXTREME.value,
        funding_analyzer.config.divergence_event_name: FundingEventType.DIVERGENCE.value,
    }

    for topic, expected_event_type in expected_topic_event_type.items():
        event = _last_event_for(fake_event_bus, topic)
        payload = event.payload

        _assert_scoped_analytics_event_payload(
            payload=payload,
            expected_event_type=expected_event_type,
        )
        _assert_event_bus_emit_contract(
            event=event,
            expected_source=funding_analyzer.SOURCE,
            expected_correlation_id="corr-full-pipeline",
            expected_priority=EventPriority.NORMAL,
        )

    for event in _events_for(fake_event_bus, funding_analyzer.config.signal_event_name):
        _assert_event_bus_emit_contract(
            event=event,
            expected_source=funding_analyzer.SOURCE,
            expected_correlation_id="corr-full-pipeline",
            expected_priority=EventPriority.HIGH,
        )

    updated_payload = _last_payload_for(fake_event_bus, funding_analyzer.config.analytics_event_name)

    _assert_scoped_analytics_event_payload(
        payload=updated_payload,
        expected_event_type=FundingEventType.SNAPSHOT.value,
    )

    updated_nested = _analytics_payload(updated_payload)
    assert set(updated_nested) >= {
        "snapshot",
        "statistics",
        "regime_state",
        "pressure_state",
        "flip_event",
        "extreme_event",
        "divergence_event",
    }

    assert updated_nested["snapshot"]["funding_rate"] == pytest.approx(0.00035)
    assert updated_nested["snapshot"]["mark_price"] == pytest.approx(50_010.0)
    assert updated_nested["snapshot"]["open_interest"] == pytest.approx(1_100_000.0)

    assert updated_nested["statistics"]["sample_size"] == 1
    assert updated_nested["regime_state"]["regime"] == FundingRegime.EXTREME_POSITIVE.value
    assert updated_nested["pressure_state"]["level"] == FundingPressureLevel.EXTREME.value
    assert updated_nested["flip_event"]["flip_type"] == FundingFlipType.NEGATIVE_TO_POSITIVE.value
    assert updated_nested["extreme_event"]["severity"] == pytest.approx(0.9)
    assert updated_nested["divergence_event"]["confidence"] == pytest.approx(0.82)

    assert len(stub_regime_detector.calls) == 1
    assert len(stub_pressure_analyzer.calls) == 1
    assert len(stub_flip_detector.calls) == 1
    assert len(stub_extremes_detector.calls) == 1
    assert len(stub_divergence_detector.calls) == 1

    pressure_call = stub_pressure_analyzer.calls[0]
    assert pressure_call["snapshot"].mark_price == pytest.approx(50_010.0)
    assert pressure_call["snapshot"].open_interest == pytest.approx(1_100_000.0)
    assert pressure_call["previous_open_interest"] == pytest.approx(1_000_000.0)
    assert pressure_call["current_price"] == pytest.approx(50_010.0)
    assert pressure_call["previous_price"] == pytest.approx(50_000.0)
    assert pressure_call["previous_snapshot"] is None

    divergence_call = stub_divergence_detector.calls[0]
    assert divergence_call["price_change_pct"] == pytest.approx(0.0002)
    assert divergence_call["oi_change_pct"] == pytest.approx(0.10)
    assert divergence_call["cvd_change"] == pytest.approx(15_000.0)
    assert divergence_call["long_liquidations"] == pytest.approx(150_000.0)
    assert divergence_call["short_liquidations"] == pytest.approx(100_000.0)

    stats = funding_analyzer.stats()
    assert stats["keys_tracked"] == 1
    assert stats["contexts_tracked"] == 1
    assert stats["latest_statistics"] == 1
    assert stats["latest_regime_states"] == 1
    assert stats["latest_pressure_states"] == 1
    assert stats["latest_flip_events"] == 1
    assert stats["latest_extreme_events"] == 1
    assert stats["latest_divergence_events"] == 1


@pytest.mark.asyncio
async def test_on_funding_second_update_passes_previous_snapshot_and_previous_regime_state(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
) -> None:
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=-0.00010, mark_price=50_000.0),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-prev-1",
        )
    )
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035, mark_price=50_100.0),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-prev-2",
        )
    )

    assert len(stub_regime_detector.calls) == 2
    assert len(stub_pressure_analyzer.calls) == 2
    assert len(stub_flip_detector.calls) == 2

    first_regime_call = stub_regime_detector.calls[0]
    second_regime_call = stub_regime_detector.calls[1]
    first_pressure_call = stub_pressure_analyzer.calls[0]
    second_pressure_call = stub_pressure_analyzer.calls[1]
    first_flip_call = stub_flip_detector.calls[0]
    second_flip_call = stub_flip_detector.calls[1]

    assert first_regime_call["previous_state"] is None
    assert second_regime_call["previous_state"] is not None
    assert second_regime_call["previous_state"].key == second_regime_call["snapshot"].key

    assert first_pressure_call["previous_snapshot"] is None
    assert second_pressure_call["previous_snapshot"] is not None
    assert second_pressure_call["previous_snapshot"].funding_rate == pytest.approx(-0.00010)
    assert second_pressure_call["previous_snapshot"].mark_price == pytest.approx(50_000.0)

    assert first_flip_call["previous_snapshot"] is None
    assert second_flip_call["previous_snapshot"] is not None
    assert second_flip_call["previous_snapshot"].funding_rate == pytest.approx(-0.00010)

    assert funding_analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ).funding_rate == pytest.approx(0.00035)

    assert funding_analyzer.get_statistics(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ).sample_size == 2


@pytest.mark.asyncio
async def test_on_funding_does_not_cross_contaminate_same_symbol_on_different_futures_market_types(
    funding_analyzer: FundingAnalyzer,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(
                market_type="usdm_futures",
                funding_rate=0.00011,
                exchange_symbol="BTCUSDT",
            ),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-usdm",
        )
    )
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(
                market_type="coinm_futures",
                funding_rate=-0.00022,
                exchange_symbol="BTCUSD_PERP",
            ),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-coinm",
        )
    )

    usdm_snapshot = funding_analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    )
    coinm_snapshot = funding_analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="coinm_futures",
        timeframe="1h",
    )

    assert usdm_snapshot is not None
    assert coinm_snapshot is not None
    assert usdm_snapshot.key != coinm_snapshot.key
    assert usdm_snapshot.exchange_symbol == "BTCUSDT"
    assert coinm_snapshot.exchange_symbol == "BTCUSD_PERP"
    assert usdm_snapshot.funding_rate == pytest.approx(0.00011)
    assert coinm_snapshot.funding_rate == pytest.approx(-0.00022)

    assert funding_analyzer.stats()["keys_tracked"] == 2
    assert funding_analyzer.stats()["latest_statistics"] == 2
    assert funding_analyzer.stats()["latest_regime_states"] == 2


@pytest.mark.asyncio
async def test_on_funding_skips_disallowed_key_before_mutating_state_or_calling_detectors(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
    fake_event_bus: Any,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        allowed_symbols={"ETHUSDT"},
        min_samples_for_statistics=1,
        emit_signals=True,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(symbol="BTCUSDT", funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-disallowed-symbol",
        )
    )

    assert analyzer.stats()["keys_tracked"] == 0
    assert analyzer.stats()["contexts_tracked"] == 0
    assert analyzer.stats()["latest_statistics"] == 0
    assert fake_event_bus.published == []

    assert stub_regime_detector.calls == []
    assert stub_pressure_analyzer.calls == []
    assert stub_flip_detector.calls == []
    assert stub_extremes_detector.calls == []
    assert stub_divergence_detector.calls == []


# =============================================================================
# Optional detector events / emission switches
# =============================================================================

@pytest.mark.asyncio
async def test_on_funding_publishes_updated_event_with_none_optional_detector_events(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    stub_flip_detector.return_none = True
    stub_extremes_detector.return_none = True
    stub_divergence_detector.return_none = True

    analyzer = make_funding_analyzer(
        flip_detector=stub_flip_detector,
        extremes_detector=stub_extremes_detector,
        divergence_detector=stub_divergence_detector,
    )

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-none-optionals",
        )
    )

    assert analyzer.config.flip_event_name not in _topics(fake_event_bus)
    assert analyzer.config.extreme_event_name not in _topics(fake_event_bus)
    assert analyzer.config.divergence_event_name not in _topics(fake_event_bus)

    updated = _analytics_payload(_last_payload_for(fake_event_bus, analyzer.config.analytics_event_name))
    assert updated["flip_event"] is None
    assert updated["extreme_event"] is None
    assert updated["divergence_event"] is None

    assert analyzer.stats()["latest_flip_events"] == 0
    assert analyzer.stats()["latest_extreme_events"] == 0
    assert analyzer.stats()["latest_divergence_events"] == 0


@pytest.mark.asyncio
async def test_emit_switches_suppress_only_their_specific_topics_but_pipeline_state_is_still_updated(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        emit_snapshots=False,
        emit_regime_events=False,
        emit_pressure_events=False,
        emit_flip_events=False,
        emit_extreme_events=False,
        emit_divergence_events=False,
        emit_signals=False,
        emit_analytics_events=True,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-emit-switches",
        )
    )

    assert _topics(fake_event_bus) == [analyzer.config.analytics_event_name]

    assert analyzer.stats()["latest_statistics"] == 1
    assert analyzer.stats()["latest_regime_states"] == 1
    assert analyzer.stats()["latest_pressure_states"] == 1
    assert analyzer.stats()["latest_flip_events"] == 1
    assert analyzer.stats()["latest_extreme_events"] == 1
    assert analyzer.stats()["latest_divergence_events"] == 1


@pytest.mark.asyncio
async def test_emit_analytics_events_false_suppresses_updated_event_but_not_specific_events(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        emit_analytics_events=False,
        emit_snapshots=True,
        emit_regime_events=True,
        emit_pressure_events=True,
        emit_flip_events=True,
        emit_extreme_events=True,
        emit_divergence_events=True,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-no-updated",
        )
    )

    assert analyzer.config.analytics_event_name not in _topics(fake_event_bus)
    assert analyzer.config.snapshot_event_name in _topics(fake_event_bus)
    assert analyzer.config.regime_event_name in _topics(fake_event_bus)
    assert analyzer.config.pressure_event_name in _topics(fake_event_bus)
    assert analyzer.config.flip_event_name in _topics(fake_event_bus)
    assert analyzer.config.extreme_event_name in _topics(fake_event_bus)
    assert analyzer.config.divergence_event_name in _topics(fake_event_bus)


# =============================================================================
# Signal generation hard checks
# =============================================================================

@pytest.mark.asyncio
async def test_signal_generation_emits_all_enabled_signal_families_with_expected_scores(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-signals",
        )
    )

    signal_payloads = _signal_payloads(funding_analyzer, fake_event_bus)
    signal_types = {payload["signal_type"] for payload in signal_payloads}

    assert signal_types >= {
        FundingSignalType.REGIME_CHANGE.value,
        FundingSignalType.SQUEEZE_WARNING.value,
        FundingSignalType.FLIP_DETECTED.value,
        FundingSignalType.EXTREME_DETECTED.value,
        FundingSignalType.REVERSION_SETUP.value,
        FundingSignalType.DIVERGENCE_DETECTED.value,
    }

    by_type = {payload["signal_type"]: payload for payload in signal_payloads}

    assert by_type[FundingSignalType.REGIME_CHANGE.value]["score"] == pytest.approx(-0.95)
    assert by_type[FundingSignalType.SQUEEZE_WARNING.value]["score"] == pytest.approx(-0.92)
    assert by_type[FundingSignalType.FLIP_DETECTED.value]["score"] == pytest.approx(-0.8)
    assert by_type[FundingSignalType.EXTREME_DETECTED.value]["score"] == pytest.approx(-0.9)
    assert by_type[FundingSignalType.REVERSION_SETUP.value]["score"] == pytest.approx(-0.75)
    assert by_type[FundingSignalType.DIVERGENCE_DETECTED.value]["score"] == pytest.approx(-0.82)

    for payload in signal_payloads:
        assert payload["symbol"] == "BTCUSDT"
        assert payload["exchange"] == "binance"
        assert payload["market_type"] == "usdm_futures"
        assert payload["timeframe"] == "1h"
        assert payload["exchange_symbol"] == "BTCUSDT"
        assert payload["confidence"] >= funding_analyzer.config.signal_min_confidence
        assert payload["description"]
        assert isinstance(payload["supporting_factors"], list)
        assert payload["metadata"]["scope"] == {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        }


@pytest.mark.asyncio
async def test_signal_min_confidence_filters_weak_signals_but_not_base_events(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    stub_regime_detector.confidence = 0.10
    stub_pressure_analyzer.pressure_score = 0.10
    stub_pressure_analyzer.squeeze_probability = 0.10
    stub_pressure_analyzer.mean_reversion_probability = 0.10
    stub_flip_detector.confidence = 0.10
    stub_extremes_detector.severity = 0.10
    stub_divergence_detector.confidence = 0.10

    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        signal_min_confidence=0.90,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(
        config=config,
        regime_detector=stub_regime_detector,
        pressure_analyzer=stub_pressure_analyzer,
        flip_detector=stub_flip_detector,
        extremes_detector=stub_extremes_detector,
        divergence_detector=stub_divergence_detector,
    )

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-low-confidence",
        )
    )

    assert analyzer.config.signal_event_name not in _topics(fake_event_bus)

    assert analyzer.config.analytics_event_name in _topics(fake_event_bus)
    assert analyzer.config.snapshot_event_name in _topics(fake_event_bus)
    assert analyzer.config.regime_event_name in _topics(fake_event_bus)
    assert analyzer.config.pressure_event_name in _topics(fake_event_bus)
    assert analyzer.config.flip_event_name in _topics(fake_event_bus)
    assert analyzer.config.extreme_event_name in _topics(fake_event_bus)
    assert analyzer.config.divergence_event_name in _topics(fake_event_bus)


@pytest.mark.asyncio
async def test_signal_cooldown_is_scoped_by_event_name_and_full_futures_key(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        signal_cooldown_sec=9999.0,
        min_emit_interval_ms=0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(market_type="usdm_futures", funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-cooldown-1",
        )
    )
    first_signal_count = _topics(fake_event_bus).count(analyzer.config.signal_event_name)
    assert first_signal_count > 0

    await analyzer.on_funding(
        make_event(
            make_funding_payload(market_type="usdm_futures", funding_rate=0.00036),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-cooldown-2",
        )
    )
    second_signal_count = _topics(fake_event_bus).count(analyzer.config.signal_event_name)

    assert second_signal_count == first_signal_count

    await analyzer.on_funding(
        make_event(
            make_funding_payload(
                market_type="coinm_futures",
                exchange_symbol="BTCUSD_PERP",
                funding_rate=0.00037,
            ),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-cooldown-3",
        )
    )
    third_signal_count = _topics(fake_event_bus).count(analyzer.config.signal_event_name)

    assert third_signal_count > second_signal_count


@pytest.mark.asyncio
async def test_short_pressure_and_negative_funding_produce_positive_scores(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
    make_regime_state: Callable[..., FundingRegimeState],
    make_pressure_state: Callable[..., FundingPressureState],
    make_flip_event: Callable[..., FundingFlipEvent],
    make_extreme_event: Callable[..., FundingExtremeEvent],
    make_divergence_event: Callable[..., FundingDivergenceEvent],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    stub_regime_detector.regime = FundingRegime.EXTREME_NEGATIVE
    stub_regime_detector.bias = FundingBias.SQUEEZE_RISK_SHORTS
    stub_regime_detector.confidence = 0.91

    stub_pressure_analyzer.level = FundingPressureLevel.EXTREME
    stub_pressure_analyzer.direction = FundingPressureDirection.SHORT
    stub_pressure_analyzer.bias = FundingBias.SQUEEZE_RISK_SHORTS
    stub_pressure_analyzer.pressure_score = 0.93
    stub_pressure_analyzer.squeeze_probability = 0.87
    stub_pressure_analyzer.mean_reversion_probability = 0.77

    stub_flip_detector.flip_type = FundingFlipType.POSITIVE_TO_NEGATIVE
    stub_flip_detector.confidence = 0.81

    # Для short-side funding extremes позитивний score має означати bullish/short-squeeze risk.
    # Stub factory бере funding_rate зі snapshot через detector.
    stub_extremes_detector.severity = 0.89

    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        signal_min_confidence=0.0,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(
        config=config,
        regime_detector=stub_regime_detector,
        pressure_analyzer=stub_pressure_analyzer,
        flip_detector=stub_flip_detector,
        extremes_detector=stub_extremes_detector,
        divergence_detector=stub_divergence_detector,
    )

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=-0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-short-pressure",
        )
    )

    by_type = {payload["signal_type"]: payload for payload in _signal_payloads(analyzer, fake_event_bus)}

    assert by_type[FundingSignalType.REGIME_CHANGE.value]["score"] == pytest.approx(0.91)
    assert by_type[FundingSignalType.SQUEEZE_WARNING.value]["score"] == pytest.approx(0.93)
    assert by_type[FundingSignalType.FLIP_DETECTED.value]["score"] == pytest.approx(0.81)
    assert by_type[FundingSignalType.EXTREME_DETECTED.value]["score"] == pytest.approx(0.89)

    assert by_type[FundingSignalType.REGIME_CHANGE.value]["bias"] == FundingBias.SQUEEZE_RISK_SHORTS.value
    assert by_type[FundingSignalType.SQUEEZE_WARNING.value]["regime"] == FundingRegime.EXTREME_NEGATIVE.value


# =============================================================================
# Custom topic / config contract
# =============================================================================

@pytest.mark.asyncio
async def test_custom_topic_config_is_used_for_subscribe_publish_and_lifecycle(
    fake_event_bus: Any,
    fake_scheduler: Any,
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        funding_event_name="custom.market.funding.updated",
        funding_event_patterns=("custom.market.funding.updated", "custom.market.funding.v2"),
        open_interest_event_name="custom.market.oi.updated",
        open_interest_event_patterns=("custom.market.oi.updated",),
        candle_event_name="custom.market.candle.closed",
        candle_event_patterns=("custom.market.candle.closed",),
        trade_event_name="custom.market.trades.updated",
        trade_event_patterns=("custom.market.trades.updated",),
        cvd_event_name="custom.analytics.orderflow.updated",
        cvd_event_patterns=("custom.analytics.orderflow.updated",),
        liquidation_event_name="custom.market.liquidations.updated",
        liquidation_event_patterns=("custom.market.liquidations.updated",),
        snapshot_event_name="custom.analytics.funding.snapshot",
        regime_event_name="custom.analytics.funding.regime",
        pressure_event_name="custom.analytics.funding.pressure",
        flip_event_name="custom.analytics.funding.flip",
        extreme_event_name="custom.analytics.funding.extreme",
        divergence_event_name="custom.analytics.funding.divergence",
        signal_event_name="custom.analytics.funding.signal",
        analytics_event_name="custom.analytics.funding.updated",
        analyzer_started_event_name="custom.analytics.funding.started",
        analyzer_stopped_event_name="custom.analytics.funding.stopped",
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        signal_cooldown_sec=0.0,
        min_emit_interval_ms=0,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(
        event_bus=fake_event_bus,
        scheduler=fake_scheduler,
        config=config,
    )

    await analyzer.start()

    assert [subscription.topic for subscription in fake_event_bus.subscriptions] == [
        "custom.market.funding.updated",
        "custom.market.funding.v2",
        "custom.market.oi.updated",
        "custom.market.candle.closed",
        "custom.market.trades.updated",
        "custom.analytics.orderflow.updated",
        "custom.market.liquidations.updated",
    ]

    assert "custom.analytics.funding.started" in _topics(fake_event_bus)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic="custom.market.funding.v2",
            correlation_id="corr-custom-topic",
        )
    )

    assert "custom.analytics.funding.updated" in _topics(fake_event_bus)
    assert "custom.analytics.funding.snapshot" in _topics(fake_event_bus)
    assert "custom.analytics.funding.regime" in _topics(fake_event_bus)
    assert "custom.analytics.funding.pressure" in _topics(fake_event_bus)
    assert "custom.analytics.funding.flip" in _topics(fake_event_bus)
    assert "custom.analytics.funding.extreme" in _topics(fake_event_bus)
    assert "custom.analytics.funding.divergence" in _topics(fake_event_bus)
    assert "custom.analytics.funding.signal" in _topics(fake_event_bus)

    await analyzer.stop()

    assert "custom.analytics.funding.stopped" in _topics(fake_event_bus)


def test_custom_scheduler_job_names_from_config_are_not_silently_ignored(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_scheduler: Any,
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        cleanup_job_name="custom.analytics.funding.cleanup",
        heartbeat_job_name="custom.analytics.funding.heartbeat",
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config, scheduler=fake_scheduler)

    analyzer.register()

    cleanup_job = fake_scheduler.jobs[analyzer.stats()["cleanup_job_id"]]
    heartbeat_job = fake_scheduler.jobs[analyzer.stats()["heartbeat_job_id"]]

    # Цей тест може впасти на поточному класі, якщо _register_cleanup_job()
    # і _register_heartbeat_job() hardcode-ять names замість config.*_job_name.
    # Це саме той тип проблеми, яку hard-тести мають показувати.
    assert cleanup_job.name == "custom.analytics.funding.cleanup"
    assert heartbeat_job.name == "custom.analytics.funding.heartbeat"


# =============================================================================
# Bad payload / resilience at event layer
# =============================================================================

@pytest.mark.asyncio
async def test_on_funding_malformed_payload_does_not_publish_or_mutate_state(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    stub_regime_detector: Any,
    stub_pressure_analyzer: Any,
    stub_flip_detector: Any,
    stub_extremes_detector: Any,
    stub_divergence_detector: Any,
) -> None:
    malformed_events = [
        make_event([], topic=funding_analyzer.config.funding_event_name),
        make_event({}, topic=funding_analyzer.config.funding_event_name),
        make_event(
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "symbol": "BTCUSDT",
                "funding_rate": "not-a-number",
            },
            topic=funding_analyzer.config.funding_event_name,
        ),
        make_event(
            {
                "exchange": "binance",
                "market_type": "usdm_futures",
                "symbol": "",
                "funding_rate": 0.0001,
            },
            topic=funding_analyzer.config.funding_event_name,
        ),
    ]

    for event in malformed_events:
        await funding_analyzer.on_funding(event)

    assert fake_event_bus.published == []
    assert funding_analyzer.stats()["keys_tracked"] == 0
    assert funding_analyzer.stats()["latest_statistics"] == 0
    assert funding_analyzer.stats()["latest_regime_states"] == 0
    assert funding_analyzer.stats()["latest_pressure_states"] == 0

    assert stub_regime_detector.calls == []
    assert stub_pressure_analyzer.calls == []
    assert stub_flip_detector.calls == []
    assert stub_extremes_detector.calls == []
    assert stub_divergence_detector.calls == []


@pytest.mark.asyncio
async def test_emit_failure_does_not_leave_per_key_lock_locked(
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

    await analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-emit-failure",
        )
    )

    key = analyzer._make_key(
        symbol="BTCUSDT",
        exchange="binance",
        market_type="usdm_futures",
        timeframe="1h",
    )

    assert key in analyzer._locks
    assert analyzer._locks[key].locked() is False
    assert failing_event_bus.emit_attempts

    # Жорсткий момент: поточний pipeline оновлює state до publish.
    # Якщо бізнес-рішення буде "emit failure має rollback-ати state",
    # цей assertion потрібно змінити на протилежний.
    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.stats()["latest_statistics"] == 1


@pytest.mark.asyncio
async def test_event_emit_headers_include_full_scope_for_every_specific_analytics_event(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-headers",
        )
    )

    for topic in [
        funding_analyzer.config.analytics_event_name,
        funding_analyzer.config.snapshot_event_name,
        funding_analyzer.config.regime_event_name,
        funding_analyzer.config.pressure_event_name,
        funding_analyzer.config.flip_event_name,
        funding_analyzer.config.extreme_event_name,
        funding_analyzer.config.divergence_event_name,
    ]:
        event = _last_event_for(fake_event_bus, topic)

        headers = event.kwargs.get("headers")
        assert isinstance(headers, dict)
        assert headers["scope"] is not None
        assert "binance" in headers["scope"]
        assert "usdm_futures" in headers["scope"]
        assert "BTCUSDT" in headers["scope"]
        assert "1h" in headers["scope"]

    for event in _events_for(fake_event_bus, funding_analyzer.config.signal_event_name):
        headers = event.kwargs.get("headers")
        assert isinstance(headers, dict)
        assert headers["scope"] is not None
        assert "binance" in headers["scope"]
        assert "usdm_futures" in headers["scope"]
        assert "BTCUSDT" in headers["scope"]
        assert "1h" in headers["scope"]


@pytest.mark.asyncio
async def test_event_payload_scope_and_header_scope_do_not_disagree(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(
                exchange="binance",
                market_type="usdm_futures",
                symbol="btcusdt",
                timeframe="1h",
                exchange_symbol="BTC/USDT:USDT",
                funding_rate=0.00035,
            ),
            topic=funding_analyzer.config.funding_event_name,
            correlation_id="corr-scope-consistency",
        )
    )

    for event in fake_event_bus.published:
        if not event.topic.startswith("analytics.funding."):
            continue

        payload = event.payload
        if "payload" not in payload:
            continue

        nested = payload["payload"]
        if "scope" not in nested:
            continue

        assert nested["scope"] == {
            "exchange": "binance",
            "market_type": "usdm_futures",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        }
        assert nested["exchange_symbol"] == "BTC/USDT:USDT"

        header_scope = str(event.kwargs["headers"]["scope"])
        assert "binance" in header_scope
        assert "usdm_futures" in header_scope
        assert "BTCUSDT" in header_scope
        assert "1h" in header_scope


# =============================================================================
# max_tracked_keys / production guard
# =============================================================================

@pytest.mark.asyncio
async def test_max_tracked_keys_blocks_new_keys_but_allows_existing_key_updates(
    make_funding_analyzer: Callable[..., FundingAnalyzer],
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    config = FundingAnalyzerConfig(
        default_market_type="usdm_futures",
        min_samples_for_statistics=1,
        max_tracked_keys=1,
        emit_signals=False,
        enable_parquet_history=False,
        load_history_from_parquet_on_start=False,
    )
    analyzer = make_funding_analyzer(config=config)

    await analyzer.on_funding(
        make_event(
            make_funding_payload(symbol="BTCUSDT", funding_rate=0.00010),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-max-1",
        )
    )
    await analyzer.on_funding(
        make_event(
            make_funding_payload(symbol="ETHUSDT", funding_rate=0.00020),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-max-2",
        )
    )
    await analyzer.on_funding(
        make_event(
            make_funding_payload(symbol="BTCUSDT", funding_rate=0.00030),
            topic=analyzer.config.funding_event_name,
            correlation_id="corr-max-3",
        )
    )

    assert analyzer.stats()["keys_tracked"] == 1
    assert analyzer.get_latest_snapshot(
        "BTCUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ).funding_rate == pytest.approx(0.00030)

    assert analyzer.get_latest_snapshot(
        "ETHUSDT",
        "binance",
        market_type="usdm_futures",
        timeframe="1h",
    ) is None

    # 2 BTC updates emitted, ETH skipped.
    assert _topics(fake_event_bus).count(analyzer.config.analytics_event_name) == 2


@pytest.mark.asyncio
async def test_raw_funding_topic_is_not_registered_by_default_but_direct_handler_call_still_processes_payload(
    funding_analyzer: FundingAnalyzer,
    fake_event_bus: Any,
    make_event: Callable[..., Any],
    make_funding_payload: Callable[..., dict[str, Any]],
) -> None:
    funding_analyzer.register()

    assert funding_analyzer.config.raw_funding_event_name not in [
        subscription.topic for subscription in fake_event_bus.subscriptions
    ]

    # Це прямий виклик handler-а, не EventBus route.
    # Якщо потрібно жорстко заборонити raw topics навіть при direct call,
    # FundingAnalyzer має перевіряти event.topic у handler-і.
    await funding_analyzer.on_funding(
        make_event(
            make_funding_payload(funding_rate=0.00035),
            topic=funding_analyzer.config.raw_funding_event_name,
            correlation_id="corr-direct-raw",
        )
    )

    assert funding_analyzer.stats()["keys_tracked"] == 1
    assert funding_analyzer.config.analytics_event_name in _topics(fake_event_bus)