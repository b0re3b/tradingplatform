from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.runtime import RuntimeSettings


@dataclass(slots=True)
class ExchangeUniverse:
    binance: list[str] = field(default_factory=list)
    bybit: list[str] = field(default_factory=list)
    okx: list[str] = field(default_factory=list)
    mexc: list[str] = field(default_factory=list)

    def all_canonical_symbols(self) -> list[str]:
        result: set[str] = set()

        for symbol in self.binance:
            result.add(to_canonical_symbol(symbol))

        for symbol in self.bybit:
            result.add(to_canonical_symbol(symbol))

        for inst_id in self.okx:
            result.add(okx_to_canonical(inst_id))

        for symbol in self.mexc:
            result.add(mexc_to_canonical(symbol))

        return sorted(result)


def to_canonical_symbol(symbol: str) -> str:
    return str(symbol).replace("-", "").replace("_", "").upper()


def okx_to_canonical(inst_id: str) -> str:
    value = str(inst_id).upper()
    if value.endswith("-SWAP"):
        value = value.removesuffix("-SWAP")
    return value.replace("-", "")


def canonical_to_okx_swap(symbol: str, quote_asset: str = "USDT") -> str:
    normalized = to_canonical_symbol(symbol)
    quote = quote_asset.upper()

    if not normalized.endswith(quote):
        return normalized

    base = normalized[: -len(quote)]
    return f"{base}-{quote}-SWAP"


def mexc_to_canonical(symbol: str) -> str:
    return str(symbol).replace("_", "").replace("-", "").upper()


def canonical_to_mexc_contract(symbol: str, quote_asset: str = "USDT") -> str:
    normalized = to_canonical_symbol(symbol)
    quote = quote_asset.upper()

    if not normalized.endswith(quote):
        return normalized

    base = normalized[: -len(quote)]
    return f"{base}_{quote}"


def _allowed(symbol: str, settings: RuntimeSettings) -> bool:
    normalized = to_canonical_symbol(symbol)

    if settings.symbol_allowlist and normalized not in settings.symbol_allowlist:
        return False

    if settings.symbol_blocklist and normalized in settings.symbol_blocklist:
        return False

    return True


def _first_str(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _truthy_active_state(value: Any) -> bool:
    """
    Tolerant active-state parser for exchange instrument payloads.

    Many exchanges use different values:
    - Binance: TRADING
    - Bybit: Trading
    - OKX: live
    - MEXC: 1 / enabled / online / trading
    """
    if value is None:
        return True

    normalized = str(value).strip().lower()

    if not normalized:
        return True

    inactive = {
        "0",
        "false",
        "offline",
        "disabled",
        "disable",
        "closed",
        "delisted",
        "suspend",
        "suspended",
        "settled",
        "preopen",
    }
    if normalized in inactive:
        return False

    active = {
        "1",
        "true",
        "online",
        "enabled",
        "enable",
        "live",
        "trading",
        "open",
        "normal",
    }
    if normalized in active:
        return True

    # Unknown status should not kill discovery unless clearly inactive.
    return True


async def discover_binance_symbols(rest: Any, settings: RuntimeSettings) -> list[str]:
    payload = await rest.get_exchange_info()
    items = payload.get("symbols", []) if isinstance(payload, dict) else []
    symbols: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        symbol = _first_str(item, "symbol").upper()
        if not symbol.endswith(settings.quote_asset):
            continue

        status = _first_str(item, "status")
        if status and status.upper() != "TRADING":
            continue

        contract_type = _first_str(item, "contractType", "contract_type")
        if contract_type and contract_type.upper() != "PERPETUAL":
            continue

        if _allowed(symbol, settings):
            symbols.append(symbol)

    return sorted(set(symbols))


async def discover_bybit_symbols(rest: Any, settings: RuntimeSettings) -> list[str]:
    symbols: list[str] = []
    cursor: str | None = None

    while True:
        kwargs: dict[str, Any] = {"category": "linear", "limit": 1000}
        if cursor:
            kwargs["cursor"] = cursor

        payload = await rest.get_instruments_info(**kwargs)
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        items = result.get("list", []) if isinstance(result, dict) else []

        for item in items:
            if not isinstance(item, dict):
                continue

            symbol = _first_str(item, "symbol").upper()
            quote = _first_str(item, "quoteCoin", "quoteCurrency", "quoteAsset").upper()
            status = _first_str(item, "status", "state")

            if quote and quote != settings.quote_asset:
                continue

            if not symbol.endswith(settings.quote_asset):
                continue

            # Keep perpetual/linear contracts only. This filters dated futures like BTCUSDT-05JUN26.
            contract_type = _first_str(item, "contractType", "contract_type").lower()
            delivery_time = _first_str(item, "deliveryTime", "delivery_time", "deliveryDate", "delivery_date")
            if "-" in symbol or delivery_time not in {"", "0"}:
                continue

            if contract_type and contract_type not in {"linearperpetual", "perpetual", "swap"}:
                continue

            if not _truthy_active_state(status):
                continue

            if _allowed(symbol, settings):
                symbols.append(symbol)

        cursor = str(result.get("nextPageCursor") or "") if isinstance(result, dict) else ""
        if not cursor:
            break

    return sorted(set(symbols))


async def discover_okx_symbols(rest: Any, settings: RuntimeSettings) -> list[str]:
    payload = await rest.get_instruments(inst_type="SWAP")
    items = payload.get("data", []) if isinstance(payload, dict) else []
    symbols: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        inst_id = _first_str(item, "instId", "inst_id").upper()
        if not inst_id:
            continue

        quote = _first_str(item, "quoteCcy", "quoteCurrency", "quoteAsset", "settleCcy").upper()
        state = _first_str(item, "state", "status")

        if quote and quote != settings.quote_asset:
            continue

        if not inst_id.endswith(f"-{settings.quote_asset}-SWAP"):
            continue

        if not _truthy_active_state(state):
            continue

        canonical = okx_to_canonical(inst_id)
        if _allowed(canonical, settings):
            symbols.append(inst_id)

    return sorted(set(symbols))


async def discover_mexc_symbols(rest: Any, settings: RuntimeSettings) -> list[str]:
    payload = await rest.get_contract_info()
    data = payload.get("data", []) if isinstance(payload, dict) else []

    if isinstance(data, dict):
        for key in ("symbols", "contracts", "list", "items", "result"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break

    if not isinstance(data, list):
        return []

    symbols: list[str] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        raw_symbol = _first_str(
            item,
            "symbol",
            "contractName",
            "contract_name",
            "displayName",
            "display_name",
        ).upper()

        if not raw_symbol:
            continue

        quote = _first_str(
            item,
            "quoteCoin",
            "quoteCurrency",
            "quoteAsset",
            "quote",
            "settleCoin",
            "settleCurrency",
        ).upper()

        canonical = mexc_to_canonical(raw_symbol)

        if quote and quote != settings.quote_asset:
            continue

        if not canonical.endswith(settings.quote_asset):
            continue

        # MEXC contract API: state=0 у payload означає активний/доступний контракт.
        # Тому НЕ можна трактувати "0" як inactive для MEXC.
        state = _first_str(item, "state", "status")
        if state:
            normalized_state = state.strip().lower()
            inactive_states = {
                "offline",
                "disabled",
                "disable",
                "closed",
                "delisted",
                "suspended",
                "suspend",
            }
            if normalized_state in inactive_states:
                continue

        hidden = item.get("isHidden", item.get("is_hidden", item.get("hidden", False)))
        if isinstance(hidden, bool):
            if hidden:
                continue
        elif str(hidden).strip().lower() in {"1", "true", "yes", "y", "on"}:
            continue

        api_allowed = item.get("apiAllowed", item.get("api_allowed", True))
        if isinstance(api_allowed, bool):
            if not api_allowed:
                continue
        elif str(api_allowed).strip().lower() in {"0", "false", "no", "n", "off"}:
            continue

        if _allowed(canonical, settings):
            symbols.append(canonical_to_mexc_contract(canonical, settings.quote_asset))

    return sorted(set(symbols))


async def discover_exchange_universe(rest_clients: dict[str, Any], settings: RuntimeSettings) -> ExchangeUniverse:
    if not settings.discover_all_symbols:
        manual = [to_canonical_symbol(symbol) for symbol in settings.symbol_allowlist]

        return ExchangeUniverse(
            binance=manual if "binance" in settings.market_data_exchanges else [],
            bybit=manual if "bybit" in settings.market_data_exchanges else [],
            okx=[
                canonical_to_okx_swap(symbol, settings.quote_asset)
                for symbol in manual
            ]
            if "okx" in settings.market_data_exchanges
            else [],
            mexc=[
                canonical_to_mexc_contract(symbol, settings.quote_asset)
                for symbol in manual
            ]
            if "mexc" in settings.market_data_exchanges
            else [],
        )

    universe = ExchangeUniverse()

    if "binance" in settings.market_data_exchanges and "binance" in rest_clients:
        universe.binance = await discover_binance_symbols(rest_clients["binance"], settings)

    if "bybit" in settings.market_data_exchanges and "bybit" in rest_clients:
        universe.bybit = await discover_bybit_symbols(rest_clients["bybit"], settings)

    if "okx" in settings.market_data_exchanges and "okx" in rest_clients:
        universe.okx = await discover_okx_symbols(rest_clients["okx"], settings)

    if "mexc" in settings.market_data_exchanges and "mexc" in rest_clients:
        universe.mexc = await discover_mexc_symbols(rest_clients["mexc"], settings)

    return universe