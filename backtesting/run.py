from __future__ import annotations

import argparse
import asyncio
import importlib
from decimal import Decimal
from typing import Any

from backtesting.config import BacktestConfig
from backtesting.engine import BacktestEngine


DEFAULT_FACTORY = "backtesting.factory:ProductionBacktestFactory"


def _load_factory(path: str, config: BacktestConfig) -> Any:
    module_name, _, attr_name = path.partition(":")
    if not module_name or not attr_name:
        raise ValueError("Factory path must use format 'module.path:FactoryClassOrObject'.")

    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)

    if isinstance(factory, type):
        try:
            return factory(config)
        except TypeError:
            return factory()

    return factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production-pipeline Binance USD-M Futures backtest."
    )

    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="usdm_futures")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "RIVERUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1m", "15m"])
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--initial-balance", default="1000")
    parser.add_argument("--report", choices=["html", "markdown"], default="html")

    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        help=f"Factory in 'module.path:ClassNameOrObject' format. Default: {DEFAULT_FACTORY}",
    )

    parser.add_argument("--disable-funding", action="store_true")
    parser.add_argument("--disable-open-interest", action="store_true")
    parser.add_argument("--disable-orderflow", action="store_true")
    parser.add_argument("--disable-mark-price", action="store_true")

    parser.add_argument(
        "--enable-agg-trades",
        action="store_true",
        help="Download Binance aggTrades history. Disabled by default to avoid 429 limits.",
    )
    parser.add_argument(
        "--enable-scheduler-loop",
        action="store_true",
        help="Run production Scheduler loop during backtest. Disabled by default for deterministic replay.",
    )
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--request-retry-delay", type=float, default=2.0)
    parser.add_argument("--replay-drain-every", type=int, default=5000)
    parser.add_argument("--replay-batch-drain-timeout", type=float, default=120.0)
    parser.add_argument("--replay-final-drain-timeout", type=float, default=600.0)
    parser.add_argument("--low-queue-size-threshold", type=int, default=100)
    parser.add_argument("--event-bus-queue-size", type=int, default=250_000)
    parser.add_argument("--event-bus-workers", type=int, default=12)

    return parser.parse_args()


async def amain() -> None:
    args = parse_args()

    config = BacktestConfig(
        exchange=args.exchange,
        market_type=args.market_type,
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        lookback_days=args.lookback_days,
        initial_balance_usd=Decimal(args.initial_balance),
        report_format=args.report,
        enable_funding=not args.disable_funding,
        enable_open_interest=not args.disable_open_interest,
        enable_orderflow=not args.disable_orderflow,
        enable_mark_price=not args.disable_mark_price,
        enable_agg_trades=args.enable_agg_trades,
        disable_scheduler_loop=not args.enable_scheduler_loop,
        request_delay_seconds=args.request_delay,
        request_retries=args.request_retries,
        request_retry_delay_seconds=args.request_retry_delay,
        replay_drain_every_events=args.replay_drain_every,
        replay_batch_drain_timeout_seconds=args.replay_batch_drain_timeout,
        replay_final_drain_timeout_seconds=args.replay_final_drain_timeout,
        low_queue_size_threshold=args.low_queue_size_threshold,
        event_bus_max_queue_size=args.event_bus_queue_size,
        event_bus_worker_count=args.event_bus_workers,
    )
    config.validate()

    factory = _load_factory(args.factory, config)

    result = await BacktestEngine(config=config, factory=factory).run()

    print()
    print("Backtest finished")
    print(f"Backtest ID: {result.backtest_id}")
    print(f"Report: {result.report_path}")
    print(f"Replayed events: {result.replayed_events}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
