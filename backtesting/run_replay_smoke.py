from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from backtesting.backtest_time import BacktestClock
from backtesting.config import BacktestTimeConfig, DataLoaderConfig, MarketReplayConfig
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestDataType,
    DataGapPolicy,
    DataValidationLevel,
    HistoricalDataFormat,
    ReplayMode,
)
from backtesting.market_replay import MarketReplay
from backtesting.models import BacktestPeriod
from backtesting.strategy_tester import InMemoryBacktestEventBus


async def main() -> None:
    period = BacktestPeriod(
        start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
    )

    loader = DataLoader(
        DataLoaderConfig(
            data_dir="data/history",
            input_format=HistoricalDataFormat.CSV,
            exchange="binance",
            market_type="usdm_futures",
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            data_types={
                BacktestDataType.CANDLES,
                BacktestDataType.FUNDING,
            },
            require_candles=True,
            require_funding=False,
            allow_empty_optional_streams=True,
            validation_level=DataValidationLevel.BASIC,
            gap_policy=DataGapPolicy.WARN,
            drop_duplicate_events=True,
        )
    )

    dataset = loader.load_dataset(
        period=period,
        run_id="btc_april_2026_replay_smoke",
    )

    event_bus = InMemoryBacktestEventBus()
    clock = BacktestClock(period=period, config=BacktestTimeConfig())
    clock.start(total_events=len(dataset.events))

    candles_seen = 0
    funding_seen = 0

    async def on_candle(payload: dict) -> None:
        nonlocal candles_seen
        candles_seen += 1

    async def on_funding(payload: dict) -> None:
        nonlocal funding_seen
        funding_seen += 1

    event_bus.subscribe("market.candle", on_candle)
    event_bus.subscribe("market.funding", on_funding)

    replay = MarketReplay(
        config=MarketReplayConfig(
            replay_mode=ReplayMode.FULL_RUN,
            batch_events_by_timestamp=True,
            emit_market_candles=True,
            emit_market_funding=True,
            fail_on_emit_error=True,
        ),
        event_bus=event_bus,
        clock=clock,
    )

    replay.prepare(dataset)
    await replay.replay()

    print("Replay completed")
    print("dataset events:", len(dataset.events))
    print("candles seen:", candles_seen)
    print("funding seen:", funding_seen)
    print("clock:", clock.now())


if __name__ == "__main__":
    asyncio.run(main())