from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backtesting.config import HistoryDownloaderConfig
from backtesting.enums import BacktestDataType, HistoricalDataFormat
from backtesting.exceptions import HistoricalDataDownloadError
from backtesting.history_downloader import HistoryDownloader


# =============================================================================
# Why this script exists
# =============================================================================
#
# Binance aggregate trades are very heavy through REST:
# - endpoint limit is small;
# - BTCUSDT can produce millions of aggregate trades for a week;
# - downloading trades sequentially can look like a hang.
#
# This runner downloads BTCUSDT, DOGEUSDT, SOLUSDT concurrently and keeps
# trades disabled by default. This is enough to verify most of the futures
# pipeline:
#
#   candles + funding + open_interest + liquidations + orderbook_snapshot
#
# Enable trades explicitly when needed:
#
#   INCLUDE_TRADES=1 HISTORY_DAYS=1 python backtesting/download_history_binance_futures_fast.py
#
# For full historical orderflow, prefer archived aggTrades files or a dedicated
# historical tick provider instead of public REST pagination.


# =============================================================================
# Helpers
# =============================================================================


def _utc_now_floor_minute() -> datetime:
    return datetime.now(timezone.utc).replace(second=0, microsecond=0)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_list(name: str, default: list[str], *, upper: bool = True) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default)

    values = [item.strip() for item in raw.split(",") if item.strip()]
    if upper:
        return [item.upper() for item in values]
    return values


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


# =============================================================================
# Runtime config
# =============================================================================


@dataclass(slots=True, frozen=True)
class FastBinanceHistoryConfig:
    """
    Fast concurrent downloader config.

    Defaults are tuned for a practical full-pipeline backtest:
    - BTCUSDT, DOGEUSDT, SOLUSDT
    - last 2 days
    - candles + funding + open_interest + liquidations + orderbook_snapshot
    - trades disabled by default because REST aggTrades is slow/heavy; orderbook snapshots enabled as current REST snapshots

    Enable trades only when you need orderflow strategies:
        INCLUDE_TRADES=1 HISTORY_DAYS=1
    """

    symbols: list[str]
    timeframes: list[str]
    output_dir: Path
    output_format: HistoricalDataFormat

    days: int = 2
    end_time: datetime | None = None

    exchange: str = "binance"
    market_type: str = "usdm_futures"

    include_candles: bool = True
    include_funding: bool = True
    include_open_interest: bool = True
    include_liquidations: bool = True
    include_trades: bool = False

    # Binance /fapi/v1/depth is current snapshot only, not historical.
    # It is enabled because orderbook context is important, but it should be
    # treated as a snapshot enrichment, not a true historical orderbook stream.
    include_orderbook_snapshot: bool = True

    overwrite_existing: bool = True
    skip_existing: bool = False
    validate_after_download: bool = True

    request_timeout_seconds: float = 20.0
    request_pause_seconds: float = 0.05

    max_retries: int = 3
    retry_delay_seconds: float = 0.5
    rate_limit_sleep_seconds: float = 0.05

    # Symbol-level concurrency. Keep modest to avoid rate limits.
    max_concurrent_symbols: int = 3

    # Stream-level concurrency per symbol. Keep 1 for safety with HistoryDownloader.
    max_concurrent_streams_per_symbol: int = 1

    fail_on_optional_stream_error: bool = False

    @classmethod
    def from_env(cls) -> "FastBinanceHistoryConfig":
        output_format_raw = os.getenv("HISTORICAL_DATA_FORMAT", "csv").strip().lower()

        try:
            output_format = HistoricalDataFormat(output_format_raw)
        except ValueError:
            raise ValueError(
                "HISTORICAL_DATA_FORMAT must be one of: "
                f"{', '.join(item.value for item in HistoricalDataFormat)}"
            ) from None

        return cls(
            symbols=_env_list("SYMBOLS", ["BTCUSDT", "DOGEUSDT", "SOLUSDT"]),
            timeframes=_env_list("TIMEFRAMES", ["1m"], upper=False),
            output_dir=_env_path("HISTORY_OUTPUT_DIR", "data/history"),
            output_format=output_format,
            days=_env_int("HISTORY_DAYS", 2),
            include_candles=_env_bool("INCLUDE_CANDLES", True),
            include_funding=_env_bool("INCLUDE_FUNDING", True),
            include_open_interest=_env_bool("INCLUDE_OPEN_INTEREST", True),
            include_liquidations=_env_bool("INCLUDE_LIQUIDATIONS", True),
            include_trades=_env_bool("INCLUDE_TRADES", False),
            include_orderbook_snapshot=_env_bool("INCLUDE_ORDERBOOK_SNAPSHOT", True),
            fail_on_optional_stream_error=_env_bool("FAIL_ON_OPTIONAL_STREAM_ERROR", False),
            request_timeout_seconds=_env_float("BINANCE_REQUEST_TIMEOUT_SECONDS", 20.0),
            request_pause_seconds=_env_float("BINANCE_REQUEST_PAUSE_SECONDS", 0.05),
            max_retries=_env_int("BINANCE_MAX_RETRIES", 3),
            retry_delay_seconds=_env_float("BINANCE_RETRY_DELAY_SECONDS", 0.5),
            rate_limit_sleep_seconds=_env_float("BINANCE_RATE_LIMIT_SLEEP_SECONDS", 0.05),
            max_concurrent_symbols=_env_int("MAX_CONCURRENT_SYMBOLS", 3),
            max_concurrent_streams_per_symbol=_env_int("MAX_CONCURRENT_STREAMS_PER_SYMBOL", 1),
        )

    @property
    def resolved_end_time(self) -> datetime:
        return self.end_time or _utc_now_floor_minute()

    @property
    def resolved_start_time(self) -> datetime:
        return self.resolved_end_time - timedelta(days=self.days)

    def primary_data_types(self) -> set[BacktestDataType]:
        data_types: set[BacktestDataType] = set()

        if self.include_candles:
            data_types.add(BacktestDataType.CANDLES)

        if self.include_funding:
            data_types.add(BacktestDataType.FUNDING)

        if self.include_open_interest:
            data_types.add(BacktestDataType.OPEN_INTEREST)

        return data_types

    def optional_data_types(self) -> set[BacktestDataType]:
        data_types: set[BacktestDataType] = set()

        if self.include_liquidations:
            data_types.add(BacktestDataType.LIQUIDATIONS)

        if self.include_trades:
            data_types.add(BacktestDataType.TRADES)

        if self.include_orderbook_snapshot:
            data_types.add(BacktestDataType.ORDERBOOK_SNAPSHOT)

        return data_types

    def all_requested_data_types(self) -> set[BacktestDataType]:
        return self.primary_data_types() | self.optional_data_types()


# =============================================================================
# Binance public REST client
# =============================================================================


class BinancePublicRestClientError(RuntimeError):
    pass


class BinanceUSDMFuturesPublicRestClient:
    """
    Dependency-free public Binance USD-M futures REST client.

    Exposes Binance-like methods used by HistoryDownloader.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        pause_seconds: float = 0.05,
        user_agent: str = "trading-system-backtesting/1.0",
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.pause_seconds = max(0.0, float(pause_seconds))
        self.user_agent = user_agent

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        clean_params = {
            key: self._normalize_param_value(value)
            for key, value in params.items()
            if value is not None
        }

        url = f"{self.BASE_URL}{path}?{urlencode(clean_params)}"

        def request() -> Any:
            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)

            req = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )

            try:
                with urlopen(req, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    if not body:
                        return None
                    return json.loads(body)

            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise BinancePublicRestClientError(
                    f"Binance HTTP {exc.code} for {path}. URL={url}. Response={body}"
                ) from exc

            except URLError as exc:
                raise BinancePublicRestClientError(
                    f"Binance request failed for {path}. URL={url}. Error={exc}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise BinancePublicRestClientError(
                    f"Binance returned non-JSON response for {path}. URL={url}."
                ) from exc

        return await asyncio.to_thread(request)

    @staticmethod
    def _normalize_param_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return int(value.astimezone(timezone.utc).timestamp() * 1000)
        return value

    @staticmethod
    def _start(
        *,
        start_time: int | None = None,
        startTime: int | None = None,
    ) -> int | None:
        return startTime if startTime is not None else start_time

    @staticmethod
    def _end(
        *,
        end_time: int | None = None,
        endTime: int | None = None,
    ) -> int | None:
        return endTime if endTime is not None else end_time

    async def get_klines(
        self,
        *,
        symbol: str,
        interval: str | None = None,
        timeframe: str | None = None,
        start_time: int | None = None,
        startTime: int | None = None,
        end_time: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
        market_type: str | None = None,
        **_: Any,
    ) -> list[Any]:
        return await self._get(
            "/fapi/v1/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval or timeframe or "1m",
                "startTime": self._start(start_time=start_time, startTime=startTime),
                "endTime": self._end(end_time=end_time, endTime=endTime),
                "limit": min(max(int(limit), 1), 1500),
            },
        )

    async def get_candles(self, **kwargs: Any) -> list[Any]:
        return await self.get_klines(**kwargs)

    async def fetch_klines(self, **kwargs: Any) -> list[Any]:
        return await self.get_klines(**kwargs)

    async def get_agg_trades(
        self,
        *,
        symbol: str,
        start_time: int | None = None,
        startTime: int | None = None,
        end_time: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
        market_type: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/fapi/v1/aggTrades",
            {
                "symbol": symbol.upper(),
                "startTime": self._start(start_time=start_time, startTime=startTime),
                "endTime": self._end(end_time=end_time, endTime=endTime),
                "limit": min(max(int(limit), 1), 1000),
            },
        )

    async def get_trades(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.get_agg_trades(**kwargs)

    async def fetch_trades(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.get_agg_trades(**kwargs)

    async def get_funding_rate_history(
        self,
        *,
        symbol: str,
        start_time: int | None = None,
        startTime: int | None = None,
        end_time: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
        market_type: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol.upper(),
                "startTime": self._start(start_time=start_time, startTime=startTime),
                "endTime": self._end(end_time=end_time, endTime=endTime),
                "limit": min(max(int(limit), 1), 1000),
            },
        )

    async def get_funding_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.get_funding_rate_history(**kwargs)

    async def get_open_interest_history(
        self,
        *,
        symbol: str,
        period: str = "5m",
        start_time: int | None = None,
        startTime: int | None = None,
        end_time: int | None = None,
        endTime: int | None = None,
        limit: int = 500,
        market_type: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/futures/data/openInterestHist",
            {
                "symbol": symbol.upper(),
                "period": period,
                "startTime": self._start(start_time=start_time, startTime=startTime),
                "endTime": self._end(end_time=end_time, endTime=endTime),
                "limit": min(max(int(limit), 1), 500),
            },
        )

    async def get_open_interest(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.get_open_interest_history(**kwargs)

    async def get_force_orders(
        self,
        *,
        symbol: str,
        start_time: int | None = None,
        startTime: int | None = None,
        end_time: int | None = None,
        endTime: int | None = None,
        limit: int = 1000,
        market_type: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        return await self._get(
            "/fapi/v1/allForceOrders",
            {
                "symbol": symbol.upper(),
                "startTime": self._start(start_time=start_time, startTime=startTime),
                "endTime": self._end(end_time=end_time, endTime=endTime),
                "limit": min(max(int(limit), 1), 1000),
            },
        )

    async def get_liquidations(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.get_force_orders(**kwargs)

    async def depth(
        self,
        *,
        symbol: str,
        limit: int = 100,
        depth: int | None = None,
        market_type: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return await self._get(
            "/fapi/v1/depth",
            {
                "symbol": symbol.upper(),
                "limit": min(max(int(depth or limit), 5), 1000),
            },
        )

    async def get_orderbook(self, **kwargs: Any) -> dict[str, Any]:
        return await self.depth(**kwargs)

    async def get_order_book(self, **kwargs: Any) -> dict[str, Any]:
        return await self.depth(**kwargs)


# =============================================================================
# Downloader orchestration
# =============================================================================


def build_history_downloader_config(
    runtime: FastBinanceHistoryConfig,
    *,
    symbol: str,
    timeframes: list[str],
    data_types: set[BacktestDataType],
) -> HistoryDownloaderConfig:
    return HistoryDownloaderConfig(
        enabled=True,
        exchange=runtime.exchange,
        market_type=runtime.market_type,
        symbols=[symbol],
        timeframes=timeframes,
        data_types=data_types,
        output_dir=runtime.output_dir,
        output_format=runtime.output_format,
        overwrite_existing=runtime.overwrite_existing,
        skip_existing=runtime.skip_existing,
        validate_after_download=runtime.validate_after_download,
        request_limit=1000,
        max_retries=runtime.max_retries,
        retry_delay_seconds=runtime.retry_delay_seconds,
        request_timeout_seconds=runtime.request_timeout_seconds,
        rate_limit_sleep_seconds=runtime.rate_limit_sleep_seconds,
        candle_limit_per_request=1000,
        trade_limit_per_request=1000,
        funding_limit_per_request=1000,
        open_interest_limit_per_request=500,
        include_mark_price=True,
        include_index_price=True,
        include_liquidations=BacktestDataType.LIQUIDATIONS in data_types,
        include_orderbook_snapshots=BacktestDataType.ORDERBOOK_SNAPSHOT in data_types,
        metadata={
            "script": "download_history_binance_futures_fast",
            "download_window": f"last_{runtime.days}_days",
            "start_time": runtime.resolved_start_time.isoformat(),
            "end_time": runtime.resolved_end_time.isoformat(),
            "note": (
                "Trades are disabled by default because REST aggTrades can be very slow. "
                "Use INCLUDE_TRADES=1 HISTORY_DAYS=1 for orderflow smoke tests."
            ),
        },
    )


async def download_symbol_batch(
    *,
    runtime: FastBinanceHistoryConfig,
    symbol: str,
    data_types: set[BacktestDataType],
    label: str,
) -> dict[BacktestDataType, dict[str, int]]:
    if not data_types:
        return {}

    rest_client = BinanceUSDMFuturesPublicRestClient(
        timeout_seconds=runtime.request_timeout_seconds,
        pause_seconds=runtime.request_pause_seconds,
    )

    config = build_history_downloader_config(
        runtime,
        symbol=symbol,
        timeframes=runtime.timeframes,
        data_types=data_types,
    )

    downloader = HistoryDownloader(
        config=config,
        rest_client=rest_client,
        event_bus=None,
    )

    started_at = time.monotonic()
    print(f"[{symbol}] starting {label}: {[item.value for item in sorted(data_types, key=lambda x: x.value)]}")

    await downloader.start()

    try:
        result = await downloader.download_all(
            start_time=runtime.resolved_start_time,
            end_time=runtime.resolved_end_time,
            symbols=[symbol],
            timeframes=runtime.timeframes,
            data_types=data_types,
        )

        elapsed = time.monotonic() - started_at
        total = sum(sum(counts.values()) for counts in result.values())
        print(f"[{symbol}] completed {label}: records={total}, elapsed={elapsed:.1f}s")
        return result

    finally:
        await downloader.stop()


def merge_results(
    target: dict[BacktestDataType, dict[str, int]],
    source: dict[BacktestDataType, dict[str, int]],
) -> dict[BacktestDataType, dict[str, int]]:
    for data_type, counts in source.items():
        target.setdefault(data_type, {})
        target[data_type].update(counts)
    return target


async def download_symbol(
    *,
    runtime: FastBinanceHistoryConfig,
    symbol: str,
) -> tuple[str, dict[BacktestDataType, dict[str, int]], list[str]]:
    results: dict[BacktestDataType, dict[str, int]] = {}
    warnings: list[str] = []

    primary = runtime.primary_data_types()
    optional = runtime.optional_data_types()

    merge_results(
        results,
        await download_symbol_batch(
            runtime=runtime,
            symbol=symbol,
            data_types=primary,
            label="primary",
        ),
    )

    if optional:
        try:
            merge_results(
                results,
                await download_symbol_batch(
                    runtime=runtime,
                    symbol=symbol,
                    data_types=optional,
                    label="optional",
                ),
            )
        except HistoricalDataDownloadError as exc:
            message = f"[{symbol}] optional stream failed: {exc}"
            if runtime.fail_on_optional_stream_error:
                raise
            warnings.append(message)
            print(message)
        except Exception as exc:
            message = f"[{symbol}] optional stream failed unexpectedly: {exc}"
            if runtime.fail_on_optional_stream_error:
                raise
            warnings.append(message)
            print(message)

    return symbol, results, warnings


async def bounded_download_symbol(
    *,
    semaphore: asyncio.Semaphore,
    runtime: FastBinanceHistoryConfig,
    symbol: str,
) -> tuple[str, dict[BacktestDataType, dict[str, int]], list[str]]:
    async with semaphore:
        return await download_symbol(runtime=runtime, symbol=symbol)


def print_final_results(
    *,
    runtime: FastBinanceHistoryConfig,
    results_by_symbol: dict[str, dict[BacktestDataType, dict[str, int]]],
    warnings: list[str],
) -> None:
    print("")
    print("Downloaded historical data:")
    for symbol in sorted(results_by_symbol):
        print(f"- {symbol}:")
        symbol_results = results_by_symbol[symbol]
        if not symbol_results:
            print("  no data")
            continue

        for data_type in sorted(symbol_results, key=lambda item: item.value):
            print(f"  {data_type.value}:")
            for key, count in sorted(symbol_results[data_type].items()):
                print(f"    {key}: {count}")

    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print("")
    print("Output directory:")
    print(runtime.output_dir.resolve())
    print("")


async def main() -> None:
    runtime = FastBinanceHistoryConfig.from_env()

    if runtime.days <= 0:
        raise ValueError("HISTORY_DAYS must be positive.")

    if runtime.include_trades and runtime.days > 1:
        print("")
        print("WARNING: INCLUDE_TRADES=1 with HISTORY_DAYS > 1 can be very slow via REST.")
        print("Recommended for first full-pipeline test:")
        print("  INCLUDE_TRADES=0 HISTORY_DAYS=2")
        print("or for orderflow smoke test:")
        print("  INCLUDE_TRADES=1 HISTORY_DAYS=1")
        print("")

    print("")
    print("Downloading Binance USD-M futures history")
    print(f"Exchange:      {runtime.exchange}")
    print(f"Market type:   {runtime.market_type}")
    print(f"Symbols:       {runtime.symbols}")
    print(f"Timeframes:    {runtime.timeframes}")
    print(f"Start:         {runtime.resolved_start_time.isoformat()}")
    print(f"End:           {runtime.resolved_end_time.isoformat()}")
    print(f"Output dir:    {runtime.output_dir}")
    print(f"Output format: {runtime.output_format.value}")
    print(f"Primary:       {[item.value for item in sorted(runtime.primary_data_types(), key=lambda x: x.value)]}")
    print(f"Optional:      {[item.value for item in sorted(runtime.optional_data_types(), key=lambda x: x.value)]}")
    print(f"Concurrency:   symbols={runtime.max_concurrent_symbols}")
    print("")

    if not runtime.all_requested_data_types():
        raise RuntimeError("No data types selected. Enable at least one INCLUDE_* flag.")

    semaphore = asyncio.Semaphore(max(1, runtime.max_concurrent_symbols))

    tasks = [
        bounded_download_symbol(
            semaphore=semaphore,
            runtime=runtime,
            symbol=symbol,
        )
        for symbol in runtime.symbols
    ]

    results_by_symbol: dict[str, dict[BacktestDataType, dict[str, int]]] = {}
    warnings: list[str] = []

    for task in asyncio.as_completed(tasks):
        symbol, results, symbol_warnings = await task
        results_by_symbol[symbol] = results
        warnings.extend(symbol_warnings)

    print_final_results(
        runtime=runtime,
        results_by_symbol=results_by_symbol,
        warnings=warnings,
    )


if __name__ == "__main__":
    asyncio.run(main())