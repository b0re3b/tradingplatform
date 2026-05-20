from __future__ import annotations

from datetime import datetime, timezone

from backtesting.config import DataLoaderConfig
from backtesting.data_loader import DataLoader
from backtesting.enums import (
    BacktestDataType,
    DataGapPolicy,
    DataValidationLevel,
    HistoricalDataFormat,
)
from backtesting.models import BacktestPeriod


def main() -> None:
    period = BacktestPeriod(
        start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
    )

    config = DataLoaderConfig(
        data_dir="data/history",
        input_format=HistoricalDataFormat.CSV,
        exchange="binance",
        market_type="usdm_futures",
        symbols=["BTCUSDT"],
        timeframes=["1m"],
        data_types={
            BacktestDataType.CANDLES,
            BacktestDataType.FUNDING,
            # BacktestDataType.OPEN_INTEREST,  # увімкни тільки якщо файл реально є
        },
        require_candles=True,
        require_funding=False,
        require_open_interest=False,
        require_trades=False,
        require_orderbook=False,
        allow_empty_optional_streams=True,
        validation_level=DataValidationLevel.BASIC,
        gap_policy=DataGapPolicy.WARN,
        drop_duplicate_events=True,
    )

    loader = DataLoader(config)

    bundle = loader.load_bundle(period=period)
    print("Bundle loaded:")
    print("candles:", len(bundle.candles))
    print("funding:", len(bundle.funding))
    print("open_interest:", len(bundle.open_interest))
    print("trades:", len(bundle.trades))
    print("warnings:", bundle.warnings)

    dataset = loader.load_dataset(
        period=period,
        run_id="btc_april_2026_check",
    )

    print("")
    print("Dataset loaded:")
    print("events:", len(dataset.events))
    print("ordering:", dataset.ordering)
    print("first:", dataset.events[0].event_time if dataset.events else None)
    print("last:", dataset.events[-1].event_time if dataset.events else None)

    if dataset.info:
        print("")
        print("Dataset info:")
        print(dataset.info.to_dict())


if __name__ == "__main__":
    main()