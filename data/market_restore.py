from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.logger import get_logger
from storage.parquet_loader import ParquetLoaderConfig, ParquetMarketLoader


@dataclass(slots=True)
class MarketRestoreConfig:
    """
    Startup restore config for state-driven market data.

    This class belongs to the data layer because the target of restore is the
    shared MarketStateStore, but it delegates parquet reads to storage.
    """

    root_dir: str = "data/parquet"
    default_exchange: str = "binance"
    default_market_type: str = "usdm_futures"
    batch_size: int = 1_000
    evaluate_after_restore: bool = True

    restore_candles: bool = True
    restore_trades: bool = False
    restore_orderbook_snapshots: bool = True
    restore_funding: bool = True
    restore_open_interest: bool = True
    restore_liquidations: bool = True

    suppress_persistable_on_replay: bool = True


@dataclass(slots=True)
class MarketRestoreResult:
    exchange: str
    market_type: str
    symbols: list[str]
    timeframes: list[str]
    requested_start_ms: int | None = None
    requested_end_ms: int | None = None
    datasets: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def rows_loaded(self) -> int:
        total = 0
        for value in self.datasets.values():
            if isinstance(value, Mapping):
                total += int(value.get("rows_loaded") or 0)
        return total

    @property
    def ok(self) -> bool:
        return not self.errors and all(bool(value.get("ok", True)) for value in self.datasets.values() if isinstance(value, Mapping))

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "requested_start_ms": self.requested_start_ms,
            "requested_end_ms": self.requested_end_ms,
            "datasets": dict(self.datasets),
            "rows_loaded": self.rows_loaded,
            "elapsed_ms": self.elapsed_ms,
            "errors": list(self.errors),
            "ok": self.ok,
        }


class MarketStateRestorer:
    """
    Data-layer bootstrapper for parquet -> ingestion -> MarketStateStore restore.

    It intentionally hydrates state through MarketIngestionService instead of
    mutating caches directly. That keeps one normalization path for REST, WS,
    warmup and parquet replay, while suppressing persistable events to avoid
    duplicated parquet rows during restart.
    """

    def __init__(
        self,
        *,
        market_ingestion: Any,
        market_scheduler: Any | None = None,
        event_bus: Any | None = None,
        config: MarketRestoreConfig | None = None,
        loader: ParquetMarketLoader | None = None,
        service_name: str = "market_state_restorer",
    ) -> None:
        self.config = config or MarketRestoreConfig()
        self.market_ingestion = market_ingestion
        self.market_scheduler = market_scheduler
        self.event_bus = event_bus
        self._logger = get_logger(__name__, service=service_name, event_type="market_restore")
        self.loader = loader or ParquetMarketLoader(
            config=ParquetLoaderConfig(
                root_dir=self.config.root_dir,
                default_exchange=self.config.default_exchange,
                default_market_type=self.config.default_market_type,
                batch_size=self.config.batch_size,
                evaluate_after_load=self.config.evaluate_after_restore,
                load_candles=self.config.restore_candles,
                load_trades=self.config.restore_trades,
                load_orderbook_snapshots=self.config.restore_orderbook_snapshots,
                load_funding=self.config.restore_funding,
                load_open_interest=self.config.restore_open_interest,
                load_liquidations=self.config.restore_liquidations,
                suppress_persistable_on_replay=self.config.suppress_persistable_on_replay,
            ),
            root_dir=Path(self.config.root_dir),
            market_ingestion=market_ingestion,
            market_scheduler=market_scheduler,
            event_bus=event_bus,
        )

    async def restore(
        self,
        *,
        symbols: Sequence[str],
        timeframes: Sequence[str],
        exchange: str | None = None,
        market_type: str | None = None,
        start_ms: int | float | str | datetime | None = None,
        end_ms: int | float | str | datetime | None = None,
        restore_candles: bool | None = None,
        restore_trades: bool | None = None,
        restore_orderbook_snapshots: bool | None = None,
        restore_funding: bool | None = None,
        restore_open_interest: bool | None = None,
        restore_liquidations: bool | None = None,
        evaluate_after_restore: bool | None = None,
    ) -> MarketRestoreResult:
        started = time.time()
        exchange = str(exchange or self.config.default_exchange).lower()
        market_type = str(market_type or self.config.default_market_type).lower()
        symbol_list = [str(item).upper() for item in symbols]
        timeframe_list = [str(item) for item in timeframes]
        start = self.loader._normalize_timestamp_ms(start_ms)  # noqa: SLF001 - shared internal normalization helper.
        end = self.loader._normalize_timestamp_ms(end_ms)  # noqa: SLF001

        result = MarketRestoreResult(
            exchange=exchange,
            market_type=market_type,
            symbols=symbol_list,
            timeframes=timeframe_list,
            requested_start_ms=start,
            requested_end_ms=end,
        )

        try:
            datasets = await self.loader.load_market_data_to_ingestion(
                exchange=exchange,
                market_type=market_type,
                symbols=symbol_list,
                timeframes=timeframe_list,
                start_ms=start,
                end_ms=end,
                load_candles=self.config.restore_candles if restore_candles is None else restore_candles,
                load_trades=self.config.restore_trades if restore_trades is None else restore_trades,
                load_orderbook_snapshots=self.config.restore_orderbook_snapshots if restore_orderbook_snapshots is None else restore_orderbook_snapshots,
                load_funding=self.config.restore_funding if restore_funding is None else restore_funding,
                load_open_interest=self.config.restore_open_interest if restore_open_interest is None else restore_open_interest,
                load_liquidations=self.config.restore_liquidations if restore_liquidations is None else restore_liquidations,
                batch_size=self.config.batch_size,
                evaluate_after_load=self.config.evaluate_after_restore if evaluate_after_restore is None else evaluate_after_restore,
                source="parquet_restore",
            )
            result.datasets = datasets
        except Exception as exc:
            result.errors.append(str(exc))
            self._logger.exception("Market state parquet restore failed")

        result.elapsed_ms = int((time.time() - started) * 1000)
        self._logger.info(
            "Market state parquet restore completed | symbols=%s timeframes=%s rows_loaded=%s elapsed_ms=%s ok=%s",
            len(symbol_list),
            len(timeframe_list),
            result.rows_loaded,
            result.elapsed_ms,
            result.ok,
        )
        return result


__all__ = ["MarketRestoreConfig", "MarketRestoreResult", "MarketStateRestorer"]
