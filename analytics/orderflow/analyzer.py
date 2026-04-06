from __future__ import annotations

from typing import Any, Optional

from core.event_bus import EventBus

from .aggressive_trades import AggressiveTradesAnalyzer
from .config import OrderFlowConfig
from .cvd import CvdAnalyzer
from .orderbook_imbalance import OrderbookImbalanceAnalyzer
from .volume_delta import VolumeDeltaAnalyzer


class OrderFlowAnalyzer:
    """
    Фасад пакета analytics.orderflow.

    Відповідальність:
    - створює та тримає всі order flow analyzers
    - централізовано стартує/зупиняє їх
    - віддає агреговані stats
    - дає зручну точку доступу до підмодулів
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        trades_cache: Any,
        orderbook_cache: Any,
        config: Optional[OrderFlowConfig] = None,
        app_config: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        trades_topic_patterns: Optional[list[str]] = None,
        orderbook_topic_patterns: Optional[list[str]] = None,
    ) -> None:
        self._event_bus = event_bus
        self._trades_cache = trades_cache
        self._orderbook_cache = orderbook_cache
        self._scheduler = scheduler
        self._config = config or (
            OrderFlowConfig.from_app_config(app_config)
            if app_config is not None
            else OrderFlowConfig()
        )

        self._trades_topic_patterns = (
            list(trades_topic_patterns)
            if trades_topic_patterns is not None
            else list(self._config.source_topic_patterns_trades)
        )
        self._orderbook_topic_patterns = (
            list(orderbook_topic_patterns)
            if orderbook_topic_patterns is not None
            else list(self._config.source_topic_patterns_orderbook)
        )

        self.cvd = CvdAnalyzer(
            event_bus=self._event_bus,
            trades_cache=self._trades_cache,
            config=self._config.cvd,
            scheduler=self._scheduler,
            source_topic_patterns=self._trades_topic_patterns,
        )

        self.volume_delta = VolumeDeltaAnalyzer(
            event_bus=self._event_bus,
            trades_cache=self._trades_cache,
            config=self._config.volume_delta,
            scheduler=self._scheduler,
            source_topic_patterns=self._trades_topic_patterns,
        )

        self.aggressive_trades = AggressiveTradesAnalyzer(
            event_bus=self._event_bus,
            trades_cache=self._trades_cache,
            config=self._config.aggressive_trades,
            scheduler=self._scheduler,
            source_topic_patterns=self._trades_topic_patterns,
        )

        self.orderbook_imbalance = OrderbookImbalanceAnalyzer(
            event_bus=self._event_bus,
            orderbook_cache=self._orderbook_cache,
            config=self._config.orderbook_imbalance,
            scheduler=self._scheduler,
            source_topic_patterns=self._orderbook_topic_patterns,
        )

        self._modules = {
            "cvd": self.cvd,
            "volume_delta": self.volume_delta,
            "aggressive_trades": self.aggressive_trades,
            "orderbook_imbalance": self.orderbook_imbalance,
        }

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        if not self._config.enabled:
            return

        for module in self._modules.values():
            module.start()

        self._running = True

    def stop(self) -> None:
        if not self._running:
            return

        for module in self._modules.values():
            module.stop()

        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_module(self, name: str):
        return self._modules.get(name)

    def get_latest_stats(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = str(symbol).upper()
        return {
            "symbol": normalized_symbol,
            "cvd": self.cvd.get_latest_stats(normalized_symbol),
            "volume_delta": self.volume_delta.get_latest_stats(normalized_symbol),
            "aggressive_trades": self.aggressive_trades.get_latest_stats(normalized_symbol),
            "orderbook_imbalance": self.orderbook_imbalance.get_latest_stats(normalized_symbol),
        }

    async def process_symbol(self, symbol: str) -> dict[str, Any]:
        normalized_symbol = str(symbol).upper()

        cvd_stats = await self.cvd.process_symbol(normalized_symbol)
        volume_delta_stats = await self.volume_delta.process_symbol(normalized_symbol)
        aggressive_trades_stats = await self.aggressive_trades.process_symbol(normalized_symbol)
        orderbook_imbalance_stats = await self.orderbook_imbalance.process_symbol(
            normalized_symbol
        )

        return {
            "symbol": normalized_symbol,
            "cvd": cvd_stats,
            "volume_delta": volume_delta_stats,
            "aggressive_trades": aggressive_trades_stats,
            "orderbook_imbalance": orderbook_imbalance_stats,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "trades_topic_patterns": list(self._trades_topic_patterns),
            "orderbook_topic_patterns": list(self._orderbook_topic_patterns),
            "modules": {
                "cvd": self.cvd.stats(),
                "volume_delta": self.volume_delta.stats(),
                "aggressive_trades": self.aggressive_trades.stats(),
                "orderbook_imbalance": self.orderbook_imbalance.stats(),
            },
        }