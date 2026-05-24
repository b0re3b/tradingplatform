from __future__ import annotations

"""State-driven market-data input adapters for analytics.

This module is the compatibility boundary between the new ``data`` state layer
and existing analytics modules.  Analytics should read market data through
snapshots/facades instead of subscribing to high-frequency raw EventBus topics.
The facades intentionally expose the old cache-like read methods used by the
current analyzers, but their source of truth is ``MarketStateStore.snapshot()``.
"""

import inspect
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _plain(value.to_dict())
        except Exception:
            pass
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _scope_kwargs(
    *,
    exchange: str | None = None,
    market_type: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    return {
        "exchange": (exchange or "binance").lower(),
        "market_type": market_type or "usdm_futures",
        "symbol": (symbol or "").upper(),
        "timeframe": timeframe,
    }


class MarketStateSnapshotSource:
    """Small async wrapper around ``MarketStateStore.snapshot``.

    It accepts both the new data layer and duck-typed stores used in tests.
    """

    def __init__(self, market_state_store: Any) -> None:
        self.market_state_store = market_state_store

    async def snapshot(
        self,
        *,
        exchange: str | None = None,
        market_type: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> Any | None:
        if self.market_state_store is None or not symbol:
            return None

        snapshot = getattr(self.market_state_store, "snapshot", None)
        if not callable(snapshot):
            return None

        kwargs = _scope_kwargs(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Preferred new API: snapshot(MarketScope) or snapshot(scope=MarketScope)
        try:
            from data.market_models import MarketScope  # type: ignore

            scope = MarketScope(
                exchange=kwargs["exchange"],
                market_type=kwargs["market_type"],
                symbol=kwargs["symbol"],
                timeframe=kwargs.get("timeframe"),
            )
            for args, kw in (((scope,), {}), ((), {"scope": scope})):
                try:
                    result = snapshot(*args, **kw)
                    if inspect.isawaitable(result):
                        result = await result
                    return result
                except TypeError:
                    continue
        except Exception:
            pass

        # Backward/duck-typed API.
        candidates = (
            ({"exchange": kwargs["exchange"], "market_type": kwargs["market_type"], "symbol": kwargs["symbol"], "timeframe": kwargs.get("timeframe")}),
            ({"exchange": kwargs["exchange"], "market_type": kwargs["market_type"], "symbol": kwargs["symbol"]}),
            ({"symbol": kwargs["symbol"], "timeframe": kwargs.get("timeframe")}),
            ({"symbol": kwargs["symbol"]}),
        )
        for kw in candidates:
            try:
                result = snapshot(**{k: v for k, v in kw.items() if v is not None})
                if inspect.isawaitable(result):
                    result = await result
                return result
            except TypeError:
                continue
        return None

    async def dirty_scopes(self, *, limit: int = 1000, sources: set[str] | None = None) -> list[Any]:
        registry = getattr(self.market_state_store, "dirty_registry", None) or getattr(self.market_state_store, "_dirty", None)
        if registry is None:
            return []
        for name in ("pop_dirty", "pop", "drain", "snapshot_dirty", "peek_dirty"):
            method = getattr(registry, name, None)
            if not callable(method):
                continue
            try:
                result = method(limit=limit, sources=sources)
            except TypeError:
                try:
                    result = method(limit=limit, source=sources)
                except TypeError:
                    try:
                        result = method(limit=limit, reasons=sources)
                    except TypeError:
                        try:
                            result = method(limit=limit)
                        except TypeError:
                            result = method()
            if inspect.isawaitable(result):
                result = await result
            return list(result or [])
        return []


def _snapshot_trades(snapshot: Any) -> list[Any]:
    trades = _get(snapshot, "trades")
    if trades is None:
        trades = _get(snapshot, "trades_window")
    if trades is None:
        return []
    if isinstance(trades, Mapping):
        trades = trades.get("trades") or trades.get("items") or trades.get("window") or []
    else:
        trades = _get(trades, "trades", trades)
    return list(_plain(trades) or [])


def _snapshot_window_items(container: Any, *, timeframe: str | None = None, item_keys: tuple[str, ...] = ("items", "window")) -> list[Any]:
    """Extract a bounded window from snapshot containers.

    The new MarketSnapshot stores candle windows as a mapping keyed by timeframe,
    e.g. ``snapshot.candles["1m"].candles``.  Older analytics code often
    expected ``snapshot.candles.candles`` or ``snapshot.candles["candles"]``.
    This helper supports all of those shapes and intentionally picks the
    requested timeframe first, then falls back to the first available window.
    """
    if container is None:
        return []

    selected = container
    if isinstance(container, Mapping):
        if timeframe and timeframe in container:
            selected = container.get(timeframe)
        elif timeframe and str(timeframe) in container:
            selected = container.get(str(timeframe))
        else:
            direct_keys = ("candles", "trades", "liquidations", *item_keys)
            selected = None
            for key in direct_keys:
                if key in container:
                    selected = container.get(key)
                    break
            if selected is None and container:
                selected = next(iter(container.values()))

    if isinstance(selected, Mapping):
        for key in ("candles", "trades", "liquidations", "items", "window"):
            if key in selected:
                selected = selected.get(key)
                break

    for attr in ("candles", "trades", "liquidations", "items", "window"):
        value = getattr(selected, attr, None)
        if value is not None:
            selected = value
            break

    return list(_plain(selected) or [])


def _snapshot_candles(snapshot: Any, timeframe: str | None = None) -> list[Any]:
    candles = _get(snapshot, "candles")
    if candles is None:
        candles = _get(snapshot, "candles_window")
    return _snapshot_window_items(candles, timeframe=timeframe, item_keys=("candles", "items", "window"))


def _snapshot_orderbook(snapshot: Any) -> dict[str, Any] | None:
    orderbook = _get(snapshot, "orderbook")
    if orderbook is None:
        return None
    orderbook = _plain(orderbook)
    if not isinstance(orderbook, Mapping):
        return None
    result = dict(orderbook)
    if "bids" not in result:
        result["bids"] = []
    if "asks" not in result:
        result["asks"] = []
    if "mid_price" not in result:
        bid = _as_float(result.get("best_bid"))
        ask = _as_float(result.get("best_ask"))
        if bid is not None and ask is not None:
            result["mid_price"] = (bid + ask) / 2.0
    return result


def _snapshot_funding(snapshot: Any) -> Any | None:
    return _plain(_get(snapshot, "funding"))


def _snapshot_open_interest(snapshot: Any) -> Any | None:
    return _plain(_get(snapshot, "open_interest"))


def _snapshot_liquidations(snapshot: Any, timeframe: str | None = None) -> list[Any]:
    items = _get(snapshot, "liquidations") or []
    return _snapshot_window_items(items, timeframe=timeframe, item_keys=("liquidations", "items", "window"))


def snapshot_candles(snapshot: Any, timeframe: str | None = None) -> list[Any]:
    """Public helper used by snapshot-driven analytics to read candle windows."""
    return _snapshot_candles(snapshot, timeframe=timeframe)


def snapshot_liquidations(snapshot: Any, timeframe: str | None = None) -> list[Any]:
    """Public helper used by snapshot-driven analytics to read liquidation windows."""
    return _snapshot_liquidations(snapshot, timeframe=timeframe)


class StateBackedTradesCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_recent_trades(self, **kwargs: Any) -> list[Any]:
        limit = int(kwargs.pop("limit", 0) or 0)
        snapshot = await self.source.snapshot(**kwargs)
        trades = _snapshot_trades(snapshot)
        return trades[-limit:] if limit > 0 else trades

    async def get_trades_since(self, *, since_ts: float | None = None, **kwargs: Any) -> list[Any]:
        trades = await self.get_recent_trades(**kwargs)
        if since_ts is None:
            return trades
        result = []
        for trade in trades:
            ts = _get(trade, "timestamp", None) or _get(trade, "timestamp_ms", None) or _get(trade, "event_time", None)
            try:
                if isinstance(ts, str):
                    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                else:
                    parsed = float(ts) / (1000.0 if float(ts) > 10_000_000_000 else 1.0)
            except Exception:
                parsed = time.time()
            if parsed >= since_ts:
                result.append(trade)
        return result

    async def get_trades(self, **kwargs: Any) -> list[Any]:
        return await self.get_recent_trades(**kwargs)

    async def get_last_trade(self, **kwargs: Any) -> Any | None:
        trades = await self.get_recent_trades(limit=1, **kwargs)
        return trades[-1] if trades else None


class StateBackedOrderBookCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_book(self, **kwargs: Any) -> dict[str, Any] | None:
        snapshot = await self.source.snapshot(**kwargs)
        return _snapshot_orderbook(snapshot)

    async def get_orderbook(self, **kwargs: Any) -> dict[str, Any] | None:
        return await self.get_book(**kwargs)

    async def get(self, **kwargs: Any) -> dict[str, Any] | None:
        return await self.get_book(**kwargs)

    async def get_top_of_book(self, **kwargs: Any) -> dict[str, Any] | None:
        book = await self.get_book(**kwargs)
        if not book:
            return None
        return {
            "best_bid": book.get("best_bid") or (book.get("bids") or [[None]])[0][0],
            "best_ask": book.get("best_ask") or (book.get("asks") or [[None]])[0][0],
            "mid_price": book.get("mid_price"),
        }


class StateBackedCandlesCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_recent_candles(self, **kwargs: Any) -> list[Any]:
        limit = int(kwargs.pop("limit", 0) or 0)
        timeframe = kwargs.get("timeframe")
        snapshot = await self.source.snapshot(**kwargs)
        candles = _snapshot_candles(snapshot, timeframe=timeframe)
        return candles[-limit:] if limit > 0 else candles

    async def get_candles(self, **kwargs: Any) -> list[Any]:
        return await self.get_recent_candles(**kwargs)

    async def get_last_candle(self, **kwargs: Any) -> Any | None:
        candles = await self.get_recent_candles(limit=1, **kwargs)
        return candles[-1] if candles else None


class StateBackedFundingCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_latest(self, **kwargs: Any) -> Any | None:
        return _snapshot_funding(await self.source.snapshot(**kwargs))

    async def get_history(self, **kwargs: Any) -> list[Any]:
        latest = await self.get_latest(**kwargs)
        if latest is None:
            return []
        if isinstance(latest, Mapping) and isinstance(latest.get("history"), list):
            return list(latest["history"])
        return [latest]


class StateBackedOpenInterestCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_latest(self, **kwargs: Any) -> Any | None:
        return _snapshot_open_interest(await self.source.snapshot(**kwargs))

    async def get_history(self, **kwargs: Any) -> list[Any]:
        latest = await self.get_latest(**kwargs)
        if latest is None:
            return []
        if isinstance(latest, Mapping) and isinstance(latest.get("history"), list):
            return list(latest["history"])
        return [latest]


class StateBackedLiquidationsCache:
    def __init__(self, source: MarketStateSnapshotSource) -> None:
        self.source = source

    async def get_recent_liquidations(self, **kwargs: Any) -> list[Any]:
        limit = int(kwargs.pop("limit", 0) or 0)
        timeframe = kwargs.get("timeframe")
        snapshot = await self.source.snapshot(**kwargs)
        items = _snapshot_liquidations(snapshot, timeframe=timeframe)
        return items[-limit:] if limit > 0 else items

    async def get_liquidations(self, **kwargs: Any) -> list[Any]:
        return await self.get_recent_liquidations(**kwargs)


class StateBackedCacheBundle:
    def __init__(self, market_state_store: Any) -> None:
        self.source = MarketStateSnapshotSource(market_state_store)
        self.trades = StateBackedTradesCache(self.source)
        self.orderbook = StateBackedOrderBookCache(self.source)
        self.candles = StateBackedCandlesCache(self.source)
        self.funding = StateBackedFundingCache(self.source)
        self.open_interest = StateBackedOpenInterestCache(self.source)
        self.liquidations = StateBackedLiquidationsCache(self.source)


def build_state_backed_cache_bundle(market_state_store: Any | None) -> StateBackedCacheBundle | None:
    if market_state_store is None:
        return None
    return StateBackedCacheBundle(market_state_store)
