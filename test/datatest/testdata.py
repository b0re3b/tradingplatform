from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

import pytest

# Варіант імпортів під стандартну структуру проєкту:
# tests/ лежить поруч із trading_system/ або запускається з project root.
try:
    from trading_system.data.candles_cache import CandlesCache
    from trading_system.data.trades_cache import TradesCache
    from trading_system.data.orderbook_cache import OrderBookCache
    from trading_system.data.funding_cache import FundingCache
    from trading_system.data.open_interest_cache import OpenInterestCache
except ImportError:  # fallback, якщо пакети імпортуються відносно project root
    from data.candles_cache import CandlesCache
    from data.trades_cache import TradesCache
    from data.orderbook_cache import OrderBookCache
    from data.funding_cache import FundingCache
    from data.open_interest_cache import OpenInterestCache


@dataclass(slots=True)
class FakeEvent:
    topic: str
    payload: dict[str, Any]


class FakeEventBus:
    """
    Мінімальна тестова EventBus-імітація.

    Потрібна тільки для data cache integration tests:
    - subscribe(topic, handler)
    - emit(topic, payload, **kwargs)
    - збереження всіх emitted events для assertions
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[FakeEvent], Any]]] = {}
        self.emitted: list[FakeEvent] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[[FakeEvent], Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.subscribers.setdefault(topic, []).append(handler)

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        event = FakeEvent(topic=topic, payload=payload or {})
        self.emitted.append(event)

        for handler in list(self.subscribers.get(topic, [])):
            result = handler(event)
            if inspect.isawaitable(result):
                await result

    def events(self, topic: str) -> list[FakeEvent]:
        return [event for event in self.emitted if event.topic == topic]


def now_ms() -> int:
    return int(time.time() * 1000)


@pytest.fixture()
def fake_config() -> object:
    """
    Data cache класи лише зберігають config і не читають exchange credentials.
    Тому для цих тестів достатньо lightweight object.
    """
    return object()


@pytest.fixture()
def event_bus() -> FakeEventBus:
    return FakeEventBus()


@pytest.fixture()
def caches(fake_config: object, event_bus: FakeEventBus) -> dict[str, Any]:
    """
    Створює всі data caches з великим retention, щоб інтеграційний тест
    не залежав від історичних timestamp-ів і не видаляв тестові записи одразу.
    """
    long_retention_ms = 3650 * 24 * 60 * 60 * 1000

    created = {
        "candles": CandlesCache(
            config=fake_config,
            event_bus=event_bus,
            scheduler=None,
            retention_ms=long_retention_ms,
        ),
        "trades": TradesCache(
            config=fake_config,
            event_bus=event_bus,
            scheduler=None,
            retention_ms=long_retention_ms,
        ),
        "orderbook": OrderBookCache(
            config=fake_config,
            event_bus=event_bus,
            scheduler=None,
        ),
        "funding": FundingCache(
            config=fake_config,
            event_bus=event_bus,
            scheduler=None,
            retention_ms=long_retention_ms,
        ),
        "open_interest": OpenInterestCache(
            config=fake_config,
            event_bus=event_bus,
            scheduler=None,
            retention_ms=long_retention_ms,
        ),
    }

    for cache in created.values():
        cache.register()

    return created


def assert_orderbook_cache_contract() -> None:
    """
    Production contract check.

    У твоєму traceback було видно реальний bug:
    OrderBookCache._normalize_inbound_payload() викликає self._now_ms(),
    але в класі може бути відсутній метод _now_ms().

    Цей тестовий файл не monkeypatch-ить production-код, щоб не приховувати bug.
    Якщо ця перевірка падає — додай у data/orderbook_cache.py:

        @staticmethod
        def _now_ms() -> int:
            return int(time.time() * 1000)
    """
    assert hasattr(OrderBookCache, "_now_ms"), (
        "OrderBookCache має production-баг: відсутній метод _now_ms(), "
        "але _normalize_inbound_payload() його викликає. "
        "Додай у data/orderbook_cache.py: "
        "@staticmethod def _now_ms() -> int: return int(time.time() * 1000)"
    )


def candle_event(
    *,
    exchange: str,
    market_type: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1m",
    open_time_ms: int,
    close: float,
    is_closed: bool = True,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "source": f"{exchange}_ws",
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": symbol,
        "timeframe": timeframe,
        "open_time_ms": open_time_ms,
        "close_time_ms": open_time_ms + 59_999,
        "timestamp_ms": open_time_ms + 59_999,
        "received_at_ms": open_time_ms + 60_050,
        "open": close - 10.0,
        "high": close + 20.0,
        "low": close - 20.0,
        "close": close,
        "volume": 100.0,
        "quote_volume": close * 100.0,
        "trades_count": 100,
        "is_closed": is_closed,
    }


def trade_event(
    *,
    exchange: str,
    market_type: str,
    symbol: str = "BTCUSDT",
    trade_id: str,
    price: float,
    timestamp_ms: int,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "source": f"{exchange}_ws",
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": symbol,
        "trade_id": trade_id,
        "price": price,
        "quantity": 1.5,
        "qty": 1.5,
        "quote_qty": price * 1.5,
        "side": "buy",
        "aggressor_side": "buy",
        "timestamp_ms": timestamp_ms,
        "received_at_ms": timestamp_ms + 25,
    }


def orderbook_snapshot_event(
    *,
    exchange: str,
    market_type: str,
    symbol: str = "BTCUSDT",
    bid: float,
    ask: float,
    sequence: int,
    timestamp_ms: int,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "source": f"{exchange}_rest",
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": symbol,
        "type": "snapshot",
        "bids": [[bid, 2.0]],
        "asks": [[ask, 3.0]],
        "sequence": sequence,
        "timestamp_ms": timestamp_ms,
        "received_at_ms": timestamp_ms + 50,
    }


def funding_event(
    *,
    exchange: str,
    market_type: str,
    symbol: str = "BTCUSDT",
    rate: float,
    timestamp_ms: int,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "source": f"{exchange}_rest",
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": symbol,
        "funding_rate": rate,
        "predicted_rate": rate * 1.1,
        "next_funding_time_ms": timestamp_ms + 8 * 60 * 60 * 1000,
        "timestamp_ms": timestamp_ms,
        "received_at_ms": timestamp_ms + 50,
    }


def open_interest_event(
    *,
    exchange: str,
    market_type: str,
    symbol: str = "BTCUSDT",
    open_interest: float,
    timestamp_ms: int,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "source": f"{exchange}_rest",
        "market_type": market_type,
        "symbol": symbol,
        "exchange_symbol": symbol,
        "open_interest": open_interest,
        "open_interest_value": open_interest * 50_000,
        "mark_price": 50_000.0,
        "timestamp_ms": timestamp_ms,
        "received_at_ms": timestamp_ms + 50,
    }


@pytest.mark.asyncio
async def test_all_market_data_is_separated_by_exchange_market_type_symbol_and_timeframe(
    caches: dict[str, Any],
    event_bus: FakeEventBus,
) -> None:
    """
    Головний інтеграційний тест.

    Перевіряє, що однаковий symbol BTCUSDT з різних бірж не змішується:
    - candles: exchange + market_type + symbol + timeframe
    - trades: exchange + market_type + symbol
    - orderbook: exchange + market_type + symbol
    - funding: exchange + market_type + symbol
    - open interest: exchange + market_type + symbol
    """
    assert_orderbook_cache_contract()

    venues = [
        ("binance", "usdm_futures", 50_000.0),
        ("bybit", "linear", 51_000.0),
        ("okx", "swap", 52_000.0),
        ("mexc", "usdm_futures", 53_000.0),
    ]

    base_ts = now_ms()

    for idx, (exchange, market_type, price) in enumerate(venues):
        ts = base_ts + idx * 60_000

        await event_bus.emit(
            "market.candle",
            candle_event(
                exchange=exchange,
                market_type=market_type,
                open_time_ms=ts,
                close=price,
            ),
        )
        await event_bus.emit(
            "market.trade",
            trade_event(
                exchange=exchange,
                market_type=market_type,
                trade_id=f"{exchange}-trade-1",
                price=price + 1.0,
                timestamp_ms=ts + 1_000,
            ),
        )
        await event_bus.emit(
            "market.orderbook.snapshot",
            orderbook_snapshot_event(
                exchange=exchange,
                market_type=market_type,
                bid=price - 1.0,
                ask=price + 1.0,
                sequence=100 + idx,
                timestamp_ms=ts + 2_000,
            ),
        )
        await event_bus.emit(
            "market.funding",
            funding_event(
                exchange=exchange,
                market_type=market_type,
                rate=0.0001 + idx * 0.00001,
                timestamp_ms=ts + 3_000,
            ),
        )
        await event_bus.emit(
            "market.open_interest",
            open_interest_event(
                exchange=exchange,
                market_type=market_type,
                open_interest=10_000.0 + idx * 1_000.0,
                timestamp_ms=ts + 4_000,
            ),
        )

    for idx, (exchange, market_type, price) in enumerate(venues):
        candles = await caches["candles"].get_recent_candles(
            exchange=exchange,
            market_type=market_type,
            symbol="BTCUSDT",
            timeframe="1m",
            limit=10,
        )
        assert len(candles) == 1
        assert candles[0]["exchange"] == exchange
        assert candles[0]["market_type"] == market_type
        assert candles[0]["symbol"] == "BTCUSDT"
        assert candles[0]["close"] == pytest.approx(price)

        last_trade = await caches["trades"].get_last_trade(
            exchange=exchange,
            market_type=market_type,
            symbol="BTCUSDT",
        )
        assert last_trade is not None
        assert last_trade["exchange"] == exchange
        assert last_trade["market_type"] == market_type
        assert last_trade["price"] == pytest.approx(price + 1.0)
        assert last_trade["trade_id"] == f"{exchange}-trade-1"

        book = await caches["orderbook"].get_book(
            exchange=exchange,
            market_type=market_type,
            symbol="BTCUSDT",
            depth=1,
        )
        assert book is not None
        assert book["exchange"] == exchange
        assert book["market_type"] == market_type
        assert book["best_bid"][0] == pytest.approx(price - 1.0)
        assert book["best_ask"][0] == pytest.approx(price + 1.0)

        funding = await caches["funding"].get_latest(
            exchange=exchange,
            market_type=market_type,
            symbol="BTCUSDT",
        )
        assert funding is not None
        assert funding["exchange"] == exchange
        assert funding["market_type"] == market_type
        assert funding["funding_rate"] == pytest.approx(0.0001 + idx * 0.00001)

        oi = await caches["open_interest"].get_latest(
            exchange=exchange,
            market_type=market_type,
            symbol="BTCUSDT",
        )
        assert oi is not None
        assert oi["exchange"] == exchange
        assert oi["market_type"] == market_type
        assert oi["open_interest"] == pytest.approx(10_000.0 + idx * 1_000.0)

    assert len(event_bus.events("market.candles.updated")) == 4
    assert len(event_bus.events("market.candle.closed")) == 4
    assert len(event_bus.events("market.trades.updated")) == 4
    assert len(event_bus.events("market.orderbook.updated")) == 4
    assert len(event_bus.events("market.funding.updated")) == 4
    assert len(event_bus.events("market.open_interest.updated")) == 4


@pytest.mark.asyncio
async def test_same_exchange_same_symbol_different_market_types_are_separated(
    caches: dict[str, Any],
    event_bus: FakeEventBus,
) -> None:
    """
    Перевіряє другий важливий кейс:
    одна біржа + один symbol, але різні market_type не мають змішуватись.
    """
    base_ts = now_ms()

    await event_bus.emit(
        "market.candle",
        candle_event(
            exchange="binance",
            market_type="spot",
            open_time_ms=base_ts,
            close=40_000.0,
        ),
    )
    await event_bus.emit(
        "market.candle",
        candle_event(
            exchange="binance",
            market_type="usdm_futures",
            open_time_ms=base_ts,
            close=41_000.0,
        ),
    )

    spot = await caches["candles"].get_last_candle(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    futures = await caches["candles"].get_last_candle(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
    )

    assert spot is not None
    assert futures is not None
    assert spot["close"] == pytest.approx(40_000.0)
    assert futures["close"] == pytest.approx(41_000.0)


@pytest.mark.asyncio
async def test_same_exchange_same_symbol_different_timeframes_are_separated(
    caches: dict[str, Any],
    event_bus: FakeEventBus,
) -> None:
    """
    Перевіряє, що candles з 1m і 5m не перезаписують одна одну.
    """
    base_ts = now_ms()

    await event_bus.emit(
        "market.candle",
        candle_event(
            exchange="okx",
            market_type="swap",
            timeframe="1m",
            open_time_ms=base_ts,
            close=50_000.0,
        ),
    )
    await event_bus.emit(
        "market.candle",
        candle_event(
            exchange="okx",
            market_type="swap",
            timeframe="5m",
            open_time_ms=base_ts,
            close=55_000.0,
        ),
    )

    candle_1m = await caches["candles"].get_last_candle(
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    candle_5m = await caches["candles"].get_last_candle(
        exchange="okx",
        market_type="swap",
        symbol="BTCUSDT",
        timeframe="5m",
    )

    assert candle_1m is not None
    assert candle_5m is not None
    assert candle_1m["close"] == pytest.approx(50_000.0)
    assert candle_5m["close"] == pytest.approx(55_000.0)


@pytest.mark.asyncio
async def test_rest_snapshots_and_ws_live_updates_go_to_same_exchange_scoped_cache(
    caches: dict[str, Any],
    event_bus: FakeEventBus,
) -> None:
    """
    Перевіряє, що REST snapshot і WS live event для однієї біржі/market_type/symbol
    потрапляють в один і той самий scoped cache, але не зачіпають інші біржі.
    """
    base_ts = now_ms()

    await event_bus.emit(
        "market.candles.snapshot",
        {
            "exchange": "bybit",
            "source": "bybit_rest",
            "market_type": "linear",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "candles": [
                candle_event(
                    exchange="bybit",
                    market_type="linear",
                    open_time_ms=base_ts,
                    close=45_000.0,
                ),
                candle_event(
                    exchange="bybit",
                    market_type="linear",
                    open_time_ms=base_ts + 60_000,
                    close=45_100.0,
                ),
            ],
        },
    )

    await event_bus.emit(
        "market.candle",
        candle_event(
            exchange="bybit",
            market_type="linear",
            open_time_ms=base_ts + 120_000,
            close=45_200.0,
        ),
    )

    bybit_candles = await caches["candles"].get_recent_candles(
        exchange="bybit",
        market_type="linear",
        symbol="BTCUSDT",
        timeframe="1m",
        limit=10,
    )
    binance_candles = await caches["candles"].get_recent_candles(
        exchange="binance",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
        limit=10,
    )

    assert [c["close"] for c in bybit_candles] == pytest.approx(
        [45_000.0, 45_100.0, 45_200.0]
    )
    assert binance_candles == []


@pytest.mark.asyncio
async def test_missing_market_type_falls_back_to_perpetual_and_can_hide_adapter_bugs(
    caches: dict[str, Any],
    event_bus: FakeEventBus,
) -> None:
    """
    Документує ризик: якщо exchange adapter не передає market_type,
    cache поставить default 'perpetual'.

    Це не змішає різні exchange, але може змішати spot/futures/swap
    всередині однієї біржі, якщо adapter-и не уніфіковані.
    """
    payload = candle_event(
        exchange="mexc",
        market_type="usdm_futures",
        open_time_ms=now_ms(),
        close=30_000.0,
    )
    payload.pop("market_type")

    await event_bus.emit("market.candle", payload)

    explicit_usdm = await caches["candles"].get_recent_candles(
        exchange="mexc",
        market_type="usdm_futures",
        symbol="BTCUSDT",
        timeframe="1m",
        limit=10,
    )
    default_perpetual = await caches["candles"].get_recent_candles(
        exchange="mexc",
        market_type="perpetual",
        symbol="BTCUSDT",
        timeframe="1m",
        limit=10,
    )

    assert explicit_usdm == []
    assert len(default_perpetual) == 1
    assert default_perpetual[0]["market_type"] == "perpetual"
