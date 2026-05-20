"""
Tests for simulated time, local historical data loading and market replay.

Covered modules:
- backtesting.backtest_time
- backtesting.data_loader
- backtesting.market_replay

These tests focus on deterministic replay behavior, historical data ingestion,
event ordering, period filtering, warmup handling and replay lifecycle controls.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from backtesting.backtest_time import BacktestClock
from backtesting.config import BacktestTimeConfig, DataLoaderConfig
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestDataType,
    BacktestEventType,
    BacktestStatus,
    DataGapPolicy,
    DataValidationLevel,
    ReplayEventPriority,
    ReplayMode,
    WarmupPolicy,
)
from backtesting.exceptions import (
    BacktestTimeTravelError,
    DataGapError,
    DataLoadError,
    DataValidationError,
    MarketReplayNotPreparedError,
    ReplayOrderingError,
    ReplaySeekError,
)
from backtesting.market_replay import (
    MarketReplay,
    build_dataset_from_records,
    build_replay_event_from_record,
    market_topic_for_data_type,
    replay_priority_for_data_type,
)
from backtesting.models import (
    BacktestDataset,
    BacktestEvent,
    BacktestPeriod,
    HistoricalCandle,
    timestamp_ms,
)


# =============================================================================
# BacktestClock
# =============================================================================


def test_clock_starts_and_reports_initial_state(backtest_period: BacktestPeriod) -> None:
    clock = BacktestClock(
        period=backtest_period,
        config=BacktestTimeConfig(),
    )

    assert not clock.started
    assert not clock.stopped

    clock.start(total_events=10)

    assert clock.started
    assert not clock.stopped
    assert clock.timestamp_ms() == backtest_period.start_ms

    stats = clock.stats()
    assert stats["started"] is True
    assert stats["total_events"] == 10
    assert stats["processed_events"] == 0


def test_clock_advances_deterministically(
    backtest_clock: BacktestClock,
    backtest_period: BacktestPeriod,
) -> None:
    target = backtest_period.start + timedelta(minutes=10)

    backtest_clock.advance_to(target)

    assert backtest_clock.now() == target
    assert backtest_clock.timestamp_ms() == timestamp_ms(target)


def test_clock_allows_equal_timestamp_when_configured(
    backtest_clock: BacktestClock,
    backtest_period: BacktestPeriod,
) -> None:
    backtest_clock.advance_to(backtest_period.start, allow_equal=True)

    assert backtest_clock.timestamp_ms() == backtest_period.start_ms


def test_clock_rejects_backward_time_travel(
    backtest_clock: BacktestClock,
    backtest_period: BacktestPeriod,
) -> None:
    backtest_clock.advance_to(backtest_period.start + timedelta(minutes=10))

    with pytest.raises(BacktestTimeTravelError):
        backtest_clock.advance_to(backtest_period.start + timedelta(minutes=5))


def test_clock_can_allow_backward_time_travel_when_configured(
    backtest_period: BacktestPeriod,
) -> None:
    clock = BacktestClock(
        period=backtest_period,
        config=BacktestTimeConfig(
            allow_time_travel_backwards=True,
        ),
    )
    clock.start(total_events=10)

    clock.advance_to(backtest_period.start + timedelta(minutes=10))
    clock.advance_to(backtest_period.start + timedelta(minutes=5))

    assert clock.now() == backtest_period.start + timedelta(minutes=5)


def test_clock_rejects_out_of_range_time(
    backtest_clock: BacktestClock,
    backtest_period: BacktestPeriod,
) -> None:
    with pytest.raises(Exception):
        backtest_clock.advance_to(backtest_period.end + timedelta(days=1))


def test_clock_progress_updates_with_processed_events(
    backtest_clock: BacktestClock,
) -> None:
    backtest_clock.mark_event_processed()
    backtest_clock.mark_event_processed()

    stats = backtest_clock.stats()

    assert stats["processed_events"] == 2
    assert stats["progress_events_pct"] == 2.0


@pytest.mark.asyncio
async def test_clock_interval_job_runs_when_due(
    backtest_period: BacktestPeriod,
) -> None:
    calls: list[int] = []

    async def job() -> None:
        calls.append(1)

    clock = BacktestClock(
        period=backtest_period,
        config=BacktestTimeConfig(),
    )
    clock.start(total_events=10)

    clock.add_interval_job(
        name="test_job",
        callback=job,
        interval=timedelta(minutes=1),
        run_immediately=False,
    )

    await clock.advance_to_async(
        backtest_period.start + timedelta(minutes=2),
        run_due_jobs=True,
    )

    assert calls


@pytest.mark.asyncio
async def test_clock_interval_job_does_not_run_before_due(
    backtest_period: BacktestPeriod,
) -> None:
    calls: list[int] = []

    async def job() -> None:
        calls.append(1)

    clock = BacktestClock(
        period=backtest_period,
        config=BacktestTimeConfig(),
    )
    clock.start(total_events=10)

    clock.add_interval_job(
        name="test_job",
        callback=job,
        interval=timedelta(minutes=10),
        run_immediately=False,
    )

    await clock.advance_to_async(
        backtest_period.start + timedelta(minutes=1),
        run_due_jobs=True,
    )

    assert calls == []


# =============================================================================
# DataLoader
# =============================================================================


def test_data_loader_discovers_expected_candle_file(
    temp_history_data_loader_config: DataLoaderConfig,
) -> None:
    loader = DataLoader(temp_history_data_loader_config)

    files = loader.discover_files()

    assert len(files) == 1
    assert files[0].data_type == BacktestDataType.CANDLES
    assert files[0].symbol == "BTCUSDT"
    assert files[0].timeframe == "1m"
    assert files[0].path.name == "BTCUSDT_1m.csv"


def test_data_loader_loads_csv_candles(
    temp_history_data_loader_config: DataLoaderConfig,
    backtest_period: BacktestPeriod,
) -> None:
    loader = DataLoader(temp_history_data_loader_config)

    bundle = loader.load_bundle(period=backtest_period)

    assert len(bundle.candles) == 10
    assert bundle.total_records == 10
    assert bundle.candles[0].symbol == "BTCUSDT"
    assert bundle.candles[0].market_type == "usdm_futures"
    assert bundle.candles[0].timeframe == "1m"


def test_data_loader_builds_sorted_dataset(
    temp_history_data_loader_config: DataLoaderConfig,
    backtest_period: BacktestPeriod,
) -> None:
    loader = DataLoader(temp_history_data_loader_config)

    dataset = loader.load_dataset(period=backtest_period, run_id="test_run")

    assert dataset.events
    assert dataset.info.total_events == len(dataset.events)
    assert dataset.events == sorted(dataset.events, key=lambda item: item.sort_key())
    assert all(event.topic == "market.candle" for event in dataset.events)
    assert all(event.run_id == "test_run" for event in dataset.events)


def test_data_loader_filters_records_by_period(
    temp_history_data_loader_config: DataLoaderConfig,
    start_time,
) -> None:
    period = BacktestPeriod(
        start=start_time + timedelta(minutes=3),
        end=start_time + timedelta(minutes=6),
    )

    loader = DataLoader(temp_history_data_loader_config)
    dataset = loader.load_dataset(period=period)

    assert dataset.events
    assert all(period.start_ms <= event.timestamp_ms <= period.end_ms for event in dataset.events)


def test_data_loader_marks_warmup_events(
    temp_history_data_loader_config: DataLoaderConfig,
    start_time,
) -> None:
    period = BacktestPeriod(
        start=start_time + timedelta(minutes=3),
        end=start_time + timedelta(minutes=6),
        warmup_start=start_time,
    )

    loader = DataLoader(temp_history_data_loader_config)
    dataset = loader.load_dataset(period=period)

    warmup_events = [event for event in dataset.events if event.is_warmup]
    trading_events = [event for event in dataset.events if not event.is_warmup]

    assert warmup_events
    assert trading_events
    assert all(event.timestamp_ms < period.start_ms for event in warmup_events)
    assert all(event.timestamp_ms >= period.start_ms for event in trading_events)


def test_data_loader_deduplicates_duplicate_candles(
    temp_history_data_loader_config: DataLoaderConfig,
    temp_history_dir: Path,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    candle_path = (
        temp_history_dir
        / "binance"
        / "usdm_futures"
        / "candles"
        / "BTCUSDT"
        / "1m"
        / "BTCUSDT_1m.csv"
    )

    first = sample_candles[0]

    with candle_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            f"{first.exchange},{first.symbol},{first.market_type},{first.timeframe},"
            f"{first.timestamp_ms},{first.received_at_ms},{first.open_time_ms},{first.close_time_ms},"
            f"{first.open},{first.high},{first.low},{first.close},"
            f"{first.volume},{first.quote_volume},{first.trades_count},{first.is_closed}"
        )

    loader = DataLoader(temp_history_data_loader_config)
    bundle = loader.load_bundle(period=backtest_period)

    assert len(bundle.candles) == len(sample_candles)


def test_data_loader_raises_when_required_candles_missing(
    data_loader_config: DataLoaderConfig,
) -> None:
    loader = DataLoader(data_loader_config)

    with pytest.raises(DataLoadError):
        loader.load_bundle()


def test_data_loader_allows_missing_optional_streams(
    temp_history_data_loader_config: DataLoaderConfig,
    backtest_period: BacktestPeriod,
) -> None:
    temp_history_data_loader_config.data_types = {
        BacktestDataType.CANDLES,
        BacktestDataType.TRADES,
    }
    temp_history_data_loader_config.require_trades = False
    temp_history_data_loader_config.allow_empty_optional_streams = True

    loader = DataLoader(temp_history_data_loader_config)
    bundle = loader.load_bundle(period=backtest_period)

    assert bundle.candles
    assert bundle.trades == []


def test_data_loader_detects_candle_gaps_when_policy_error(
    temp_history_data_loader_config: DataLoaderConfig,
    temp_history_dir: Path,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    candle_path = (
        temp_history_dir
        / "binance"
        / "usdm_futures"
        / "candles"
        / "BTCUSDT"
        / "1m"
        / "BTCUSDT_1m.csv"
    )

    # Перезаписуємо файл тільки двома свічками з великим розривом.
    first = sample_candles[0]
    late = sample_candles[-1]

    with candle_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "exchange,symbol,market_type,timeframe,timestamp_ms,received_at_ms,"
            "open_time_ms,close_time_ms,open,high,low,close,volume,quote_volume,trades_count,is_closed\n"
        )
        for candle in (first, late):
            handle.write(
                f"{candle.exchange},{candle.symbol},{candle.market_type},{candle.timeframe},"
                f"{candle.timestamp_ms},{candle.received_at_ms},{candle.open_time_ms},{candle.close_time_ms},"
                f"{candle.open},{candle.high},{candle.low},{candle.close},"
                f"{candle.volume},{candle.quote_volume},{candle.trades_count},{candle.is_closed}\n"
            )

    temp_history_data_loader_config.gap_policy = DataGapPolicy.ERROR
    temp_history_data_loader_config.max_allowed_gap_seconds = 60

    loader = DataLoader(temp_history_data_loader_config)

    with pytest.raises(DataGapError):
        loader.load_bundle(period=backtest_period)


def test_data_loader_warns_on_candle_gaps_when_policy_warn(
    temp_history_data_loader_config: DataLoaderConfig,
    temp_history_dir: Path,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    candle_path = (
        temp_history_dir
        / "binance"
        / "usdm_futures"
        / "candles"
        / "BTCUSDT"
        / "1m"
        / "BTCUSDT_1m.csv"
    )

    first = sample_candles[0]
    late = sample_candles[-1]

    with candle_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "exchange,symbol,market_type,timeframe,timestamp_ms,received_at_ms,"
            "open_time_ms,close_time_ms,open,high,low,close,volume,quote_volume,trades_count,is_closed\n"
        )
        for candle in (first, late):
            handle.write(
                f"{candle.exchange},{candle.symbol},{candle.market_type},{candle.timeframe},"
                f"{candle.timestamp_ms},{candle.received_at_ms},{candle.open_time_ms},{candle.close_time_ms},"
                f"{candle.open},{candle.high},{candle.low},{candle.close},"
                f"{candle.volume},{candle.quote_volume},{candle.trades_count},{candle.is_closed}\n"
            )

    temp_history_data_loader_config.gap_policy = DataGapPolicy.WARN
    temp_history_data_loader_config.max_allowed_gap_seconds = 60

    loader = DataLoader(temp_history_data_loader_config)
    bundle = loader.load_bundle(period=backtest_period)

    assert bundle.warnings


def test_data_loader_strict_validation_rejects_bad_symbol(
    temp_history_data_loader_config: DataLoaderConfig,
    temp_history_dir: Path,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    candle_path = (
        temp_history_dir
        / "binance"
        / "usdm_futures"
        / "candles"
        / "BTCUSDT"
        / "1m"
        / "BTCUSDT_1m.csv"
    )

    first = sample_candles[0]

    with candle_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "exchange,symbol,market_type,timeframe,timestamp_ms,received_at_ms,"
            "open_time_ms,close_time_ms,open,high,low,close,volume,quote_volume,trades_count,is_closed\n"
        )
        handle.write(
            f"{first.exchange},ETHUSDT,{first.market_type},{first.timeframe},"
            f"{first.timestamp_ms},{first.received_at_ms},{first.open_time_ms},{first.close_time_ms},"
            f"{first.open},{first.high},{first.low},{first.close},"
            f"{first.volume},{first.quote_volume},{first.trades_count},{first.is_closed}\n"
        )

    temp_history_data_loader_config.validation_level = DataValidationLevel.STRICT

    loader = DataLoader(temp_history_data_loader_config)

    with pytest.raises(DataValidationError):
        loader.load_bundle(period=backtest_period)


# =============================================================================
# Market replay helpers
# =============================================================================


def test_market_topic_for_data_type() -> None:
    assert market_topic_for_data_type(BacktestDataType.CANDLES) == "market.candle"
    assert market_topic_for_data_type(BacktestDataType.TRADES) == "market.trade"
    assert market_topic_for_data_type(BacktestDataType.FUNDING) == "market.funding"
    assert market_topic_for_data_type(BacktestDataType.OPEN_INTEREST) == "market.open_interest"


def test_replay_priority_for_data_type() -> None:
    assert replay_priority_for_data_type(BacktestDataType.CANDLES) == ReplayEventPriority.CANDLE
    assert replay_priority_for_data_type(BacktestDataType.TRADES) == ReplayEventPriority.TRADE
    assert replay_priority_for_data_type(BacktestDataType.FUNDING) == ReplayEventPriority.FUNDING


def test_build_replay_event_from_record(
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    event = build_replay_event_from_record(
        sample_candles[0],
        data_type=BacktestDataType.CANDLES,
        period=backtest_period,
        run_id="test_run",
        sequence=1,
    )

    assert event.run_id == "test_run"
    assert event.topic == "market.candle"
    assert event.event_type == BacktestEventType.MARKET
    assert event.sequence == 1
    assert event.payload["symbol"] == "BTCUSDT"


def test_build_dataset_from_records(
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    dataset = build_dataset_from_records(
        {BacktestDataType.CANDLES: sample_candles},
        period=backtest_period,
        run_id="test_run",
    )

    assert dataset.events
    assert len(dataset.events) == len(sample_candles)
    assert dataset.events == sorted(dataset.events, key=lambda item: item.sort_key())


# =============================================================================
# MarketReplay
# =============================================================================


def test_market_replay_prepare_rejects_empty_dataset(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )

    with pytest.raises(MarketReplayNotPreparedError):
        replay.prepare(BacktestDataset(events=[]))


@pytest.mark.asyncio
async def test_market_replay_emits_market_candle_events(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    received: list[dict] = []

    async def on_candle(payload: dict) -> None:
        received.append(payload)

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    await replay.replay()

    assert len(received) == len(sample_dataset.events)
    assert all(payload["symbol"] == "BTCUSDT" for payload in received)
    assert all(payload["metadata"]["backtest"] is True for payload in received)


@pytest.mark.asyncio
async def test_market_replay_advances_clock_to_event_timestamp(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    await replay.replay_step()

    first_event = sample_dataset.events[0]

    assert backtest_clock.timestamp_ms() == first_event.timestamp_ms


@pytest.mark.asyncio
async def test_market_replay_preserves_timestamp_order(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    seen_timestamps: list[int] = []

    async def on_candle(payload: dict) -> None:
        seen_timestamps.append(int(payload["timestamp_ms"]))

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    await replay.replay()

    assert seen_timestamps == sorted(seen_timestamps)


@pytest.mark.asyncio
async def test_market_replay_batches_events_with_same_timestamp(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_candles: list[HistoricalCandle],
    backtest_period: BacktestPeriod,
) -> None:
    first = sample_candles[0]
    second = sample_candles[1]

    second_same_time = HistoricalCandle(
        exchange=second.exchange,
        symbol=second.symbol,
        market_type=second.market_type,
        timeframe=second.timeframe,
        timestamp_ms=first.timestamp_ms,
        received_at_ms=first.received_at_ms,
        open_time_ms=first.open_time_ms,
        close_time_ms=first.close_time_ms,
        open=second.open,
        high=second.high,
        low=second.low,
        close=second.close,
        volume=second.volume,
        quote_volume=second.quote_volume,
        trades_count=second.trades_count,
        is_closed=True,
        source="test",
        metadata={"same_timestamp": True},
    )

    dataset = build_dataset_from_records(
        {BacktestDataType.CANDLES: [first, second_same_time]},
        period=backtest_period,
        run_id="test_run",
    )

    market_replay_config.batch_events_by_timestamp = True

    received: list[dict] = []

    async def on_candle(payload: dict) -> None:
        received.append(payload)

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(dataset)

    await replay.replay()

    assert len(received) == 2
    assert received[0]["timestamp_ms"] == received[1]["timestamp_ms"]


@pytest.mark.asyncio
async def test_market_replay_skips_warmup_when_policy_none(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    temp_history_data_loader_config: DataLoaderConfig,
    start_time,
) -> None:
    period = BacktestPeriod(
        start=start_time + timedelta(minutes=3),
        end=start_time + timedelta(minutes=6),
        warmup_start=start_time,
    )
    dataset = DataLoader(temp_history_data_loader_config).load_dataset(period=period)

    market_replay_config.warmup_policy = WarmupPolicy.NONE
    market_replay_config.emit_warmup_events = False

    received: list[dict] = []

    async def on_candle(payload: dict) -> None:
        received.append(payload)

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(dataset)

    await replay.replay()

    assert received
    assert all(payload["is_warmup"] is False for payload in received)


@pytest.mark.asyncio
async def test_market_replay_emits_warmup_when_enabled(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    temp_history_data_loader_config: DataLoaderConfig,
    start_time,
) -> None:
    period = BacktestPeriod(
        start=start_time + timedelta(minutes=3),
        end=start_time + timedelta(minutes=6),
        warmup_start=start_time,
    )
    dataset = DataLoader(temp_history_data_loader_config).load_dataset(period=period)

    market_replay_config.warmup_policy = WarmupPolicy.REPLAY
    market_replay_config.emit_warmup_events = True

    received: list[dict] = []

    async def on_candle(payload: dict) -> None:
        received.append(payload)

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(dataset)

    await replay.replay()

    assert any(payload["is_warmup"] is True for payload in received)
    assert any(payload["is_warmup"] is False for payload in received)


@pytest.mark.asyncio
async def test_market_replay_pause_resume(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    await replay.start()
    checkpoint = await replay.pause()

    assert checkpoint.index == 0
    assert replay.stats()["paused"] is True

    await replay.resume()

    assert replay.stats()["paused"] is False
    assert replay.stats()["running"] is True


@pytest.mark.asyncio
async def test_market_replay_step_mode_processes_one_event(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    received: list[dict] = []

    async def on_candle(payload: dict) -> None:
        received.append(payload)

    event_bus.subscribe("market.candle", on_candle)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    event = await replay.replay_step()

    assert event is not None
    assert len(received) == 1
    assert replay.stats()["processed_events"] == 1


@pytest.mark.asyncio
async def test_market_replay_seek_to_index(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    checkpoint = await replay.seek_to_index(2)

    assert checkpoint.index == 2
    assert replay.stats()["current_index"] == 2


@pytest.mark.asyncio
async def test_market_replay_seek_to_timestamp(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    target = sample_dataset.events[3].event_time
    checkpoint = await replay.seek_to_timestamp(target)

    assert checkpoint.index == 3


@pytest.mark.asyncio
async def test_market_replay_seek_to_timestamp_raises_when_not_found(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
    end_time,
) -> None:
    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    with pytest.raises(ReplaySeekError):
        await replay.seek_to_timestamp(end_time + timedelta(days=1))


def test_market_replay_fails_on_unsorted_dataset(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    events = list(reversed(sample_dataset.events))
    dataset = BacktestDataset(events=events)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )

    with pytest.raises(ReplayOrderingError):
        replay.prepare(dataset)


@pytest.mark.asyncio
async def test_market_replay_lifecycle_events_are_emitted(
    event_bus,
    backtest_clock: BacktestClock,
    market_replay_config,
    sample_dataset: BacktestDataset,
) -> None:
    lifecycle: list[tuple[str, dict]] = []

    async def on_started(payload: dict) -> None:
        lifecycle.append(("started", payload))

    async def on_finished(payload: dict) -> None:
        lifecycle.append(("finished", payload))

    event_bus.subscribe(market_replay_config.replay_started_topic, on_started)
    event_bus.subscribe(market_replay_config.replay_finished_topic, on_finished)

    replay = MarketReplay(
        config=market_replay_config,
        event_bus=event_bus,
        clock=backtest_clock,
    )
    replay.prepare(sample_dataset)

    await replay.replay()

    labels = [item[0] for item in lifecycle]

    assert "started" in labels
    assert "finished" in labels
    assert replay.stats()["status"] in {BacktestStatus.COMPLETED.value, BacktestStatus.COMPLETED}