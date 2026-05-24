# app/__init__.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_FACTORY_EXPORTS = {
    "build_market_state_store": ".factories",
    "build_market_ingestion_service": ".factories",
    "build_market_scheduler": ".factories",
    "build_rest_clients": ".factories",
    "build_exchange_ws_clients": ".factories",
    "build_data_caches": ".factories",
    "build_market_stream": ".factories",
    "build_parquet_storage": ".factories",
    "build_analytics_components": ".factories",
    "build_strategy_factories": ".factories",
    "build_strategy_engine": ".factories",
    "build_risk_manager": ".factories",
    "build_execution_components": ".factories",
    "build_telegram_service": ".factories",
    "build_news_service": ".factories",
    "send_telegram_startup_message": ".factories",
}

_RUNTIME_EXPORTS = {
    "RuntimeSettings": ".runtime",
    "env_bool": ".runtime",
    "env_int": ".runtime",
    "env_float": ".runtime",
    "env_list": ".runtime",
    "env_str": ".runtime",
    "maybe_await": ".runtime",
    "call_if_exists": ".runtime",
    "register_component": ".runtime",
    "start_component": ".runtime",
    "stop_component": ".runtime",
    "build_event_bus": ".runtime",
    "build_scheduler": ".runtime",
    "install_signal_handlers": ".runtime",
    "chunked": ".runtime",
}

_UNIVERSE_EXPORTS = {
    "ExchangeUniverse": ".universe",
    "okx_to_canonical": ".universe",
    "discover_binance_symbols": ".universe",
    "discover_bybit_symbols": ".universe",
    "discover_okx_symbols": ".universe",
    "discover_mexc_symbols": ".universe",
    "discover_exchange_universe": ".universe",
}

_MAIN_EXPORTS = {
    "TradingSystemRuntime": ".main",
    "amain": ".main",
    "main": ".main",
}

_EXPORTS = {
    **_FACTORY_EXPORTS,
    **_RUNTIME_EXPORTS,
    **_UNIVERSE_EXPORTS,
    **_MAIN_EXPORTS,
}

__all__ = [
    "__version__",
    *_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from .factories import (
        build_analytics_components,
        build_market_state_store,
        build_market_ingestion_service,
        build_market_scheduler,
        build_data_caches,
        build_exchange_ws_clients,
        build_execution_components,
        build_market_stream,
        build_parquet_storage,
        build_news_service,
        build_rest_clients,
        build_risk_manager,
        build_strategy_engine,
        build_strategy_factories,
        build_telegram_service,
        send_telegram_startup_message,
    )
    from .main import TradingSystemRuntime, amain, main
    from .runtime import (
        RuntimeSettings,
        build_event_bus,
        build_scheduler,
        call_if_exists,
        chunked,
        env_bool,
        env_float,
        env_int,
        env_list,
        env_str,
        install_signal_handlers,
        maybe_await,
        register_component,
        start_component,
        stop_component,
    )
    from .universe import (
        ExchangeUniverse,
        discover_binance_symbols,
        discover_bybit_symbols,
        discover_exchange_universe,
        discover_mexc_symbols,
        discover_okx_symbols,
        okx_to_canonical,
    )