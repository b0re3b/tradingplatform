# test/spreadenginetest/factories.py

from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, TypeVar

from analytics.spreads.enums import (
    InstrumentType,
    OpportunityStatus,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from analytics.spreads.models import (
    ArbitrageOpportunity,
    SpreadSignal,
    SpreadSnapshot,
)


ModelT = TypeVar("ModelT")


# ============================================================
# Generic helpers
# ============================================================

def utcnow() -> datetime:
    """
    Strategy package currently works with naive UTC datetime.
    Keep tests aligned with that contract.
    """
    return datetime.utcnow()


def stale_time(*, seconds: int = 30) -> datetime:
    return utcnow() - timedelta(seconds=seconds)


def future_time(*, seconds: int = 30) -> datetime:
    return utcnow() + timedelta(seconds=seconds)


def d(value: Decimal | int | float | str | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _merge_kwargs(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Merge defaults with user overrides.

    This avoids errors like:
        TypeError: got multiple values for keyword argument 'zscore'
    when wrapper factories define defaults and tests override them.
    """
    payload = dict(defaults)
    payload.update(overrides)
    return payload


def _build_model(model_cls: type[ModelT], **kwargs: Any) -> ModelT:
    """
    Створює analytics model, передаючи тільки ті kwargs, які реально
    приймає constructor.

    Це робить factories менш крихкими, якщо в analytics.spreads.models
    додадуть або приберуть необов'язкові поля.
    """
    signature = inspect.signature(model_cls)

    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_var_kwargs:
        return model_cls(**kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return model_cls(**accepted_kwargs)


def _enum_member(enum_cls: type[Any], *names: str, fallback_value: str | None = None) -> Any:
    """
    Дістає enum member без жорсткої прив'язки до exact naming,
    наприклад VALID / NORMAL / CROSS_EXCHANGE.
    """
    for name in names:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)

    if fallback_value is not None:
        try:
            return enum_cls(fallback_value)
        except Exception:
            pass

    try:
        return next(iter(enum_cls))
    except Exception as exc:
        raise ValueError(f"Cannot resolve enum member for {enum_cls!r}") from exc


def _metadata(base: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    payload = dict(base or {})
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


# ============================================================
# Enum defaults
# ============================================================

def spot_type() -> InstrumentType:
    return _enum_member(InstrumentType, "SPOT", fallback_value="spot")


def futures_type() -> InstrumentType:
    return _enum_member(
        InstrumentType,
        "PERPETUAL",
        "FUTURES",
        "SWAP",
        fallback_value="perpetual",
    )


def active_status() -> OpportunityStatus:
    return _enum_member(OpportunityStatus, "ACTIVE", fallback_value="active")


def expired_status() -> OpportunityStatus:
    return _enum_member(
        OpportunityStatus,
        "EXPIRED",
        "INACTIVE",
        "CLOSED",
        fallback_value="expired",
    )


def normal_regime() -> SpreadRegime:
    return _enum_member(SpreadRegime, "NORMAL", fallback_value="normal")


def elevated_regime() -> SpreadRegime:
    return _enum_member(SpreadRegime, "ELEVATED", fallback_value="elevated")


def extreme_regime() -> SpreadRegime:
    return _enum_member(SpreadRegime, "EXTREME", fallback_value="extreme")


def dislocated_regime() -> SpreadRegime:
    return _enum_member(SpreadRegime, "DISLOCATED", fallback_value="dislocated")


def valid_quote_status() -> Any:
    try:
        from analytics.spreads.enums import QuoteValidity

        return _enum_member(QuoteValidity, "VALID", "FRESH", fallback_value="valid")
    except Exception:
        return SimpleNamespace(value="valid")


def spread_direction_long() -> Any:
    try:
        from analytics.spreads.enums import SpreadDirection

        return _enum_member(SpreadDirection, "LONG", "WIDENING", fallback_value="long")
    except Exception:
        return SimpleNamespace(value="long")


def spread_direction_short() -> Any:
    try:
        from analytics.spreads.enums import SpreadDirection

        return _enum_member(SpreadDirection, "SHORT", "NARROWING", fallback_value="short")
    except Exception:
        return SimpleNamespace(value="short")


def pricing_source_mid() -> Any:
    try:
        from analytics.spreads.enums import PricingSource

        return _enum_member(PricingSource, "MID", "MID_PRICE", fallback_value="mid")
    except Exception:
        return SimpleNamespace(value="mid")


# ============================================================
# Stats factory
# ============================================================

def make_spread_stats(
    *,
    zscore: Decimal | int | float | str | None = "2.5",
    mean: Decimal | int | float | str | None = "0",
    stddev: Decimal | int | float | str | None = "1",
    sample_size: int = 100,
    **overrides: Any,
) -> Any:
    """
    Strategy uses only snapshot.stats.zscore.
    If analytics.spreads.models exposes SpreadStats, use it.
    Otherwise SimpleNamespace is enough for strategy behavior tests.
    """
    payload = {
        "zscore": d(zscore),
        "mean": d(mean),
        "stddev": d(stddev),
        "std": d(stddev),
        "sample_size": sample_size,
        **overrides,
    }

    try:
        from analytics.spreads.models import SpreadStats

        return _build_model(SpreadStats, **payload)
    except Exception:
        return SimpleNamespace(**payload)


# ============================================================
# SpreadSignal factories
# ============================================================

def make_spread_signal(
    *,
    symbol: str = "BTCUSDT",
    spread_type: SpreadType = SpreadType.SPOT_FUTURES,
    signal_type: SpreadSignalType = SpreadSignalType.MEAN_REVERSION,
    exchange_a: str | None = "binance",
    exchange_b: str | None = "bybit",
    confidence: Decimal | int | float | str | None = "0.90",
    message: str = "test spread signal",
    timestamp: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> SpreadSignal:
    return _build_model(
        SpreadSignal,
        symbol=symbol,
        spread_type=spread_type,
        signal_type=signal_type,
        exchange_a=exchange_a,
        exchange_b=exchange_b,
        confidence=d(confidence),
        message=message,
        timestamp=timestamp or utcnow(),
        metadata=_metadata(metadata),
        **overrides,
    )


def make_mean_reversion_signal(
    *,
    symbol: str = "BTCUSDT",
    spot_exchange: str = "binance",
    futures_exchange: str = "bybit",
    **overrides: Any,
) -> SpreadSignal:
    return make_spread_signal(
        **_merge_kwargs(
            {
                "symbol": symbol,
                "spread_type": SpreadType.SPOT_FUTURES,
                "signal_type": SpreadSignalType.MEAN_REVERSION,
                "exchange_a": spot_exchange,
                "exchange_b": futures_exchange,
                "metadata": {
                    "spot_exchange": spot_exchange,
                    "futures_exchange": futures_exchange,
                },
            },
            overrides,
        )
    )


def make_regime_shift_signal(
    *,
    symbol: str = "BTCUSDT",
    spot_exchange: str = "binance",
    futures_exchange: str = "bybit",
    **overrides: Any,
) -> SpreadSignal:
    return make_spread_signal(
        **_merge_kwargs(
            {
                "symbol": symbol,
                "spread_type": SpreadType.SPOT_FUTURES,
                "signal_type": SpreadSignalType.REGIME_SHIFT,
                "exchange_a": spot_exchange,
                "exchange_b": futures_exchange,
                "metadata": {
                    "spot_exchange": spot_exchange,
                    "futures_exchange": futures_exchange,
                },
            },
            overrides,
        )
    )


def make_basis_data_quality_signal(
    *,
    symbol: str = "BTCUSDT",
    spot_exchange: str = "binance",
    futures_exchange: str = "bybit",
    signal_type: SpreadSignalType = SpreadSignalType.STALE_DATA,
    **overrides: Any,
) -> SpreadSignal:
    return make_spread_signal(
        **_merge_kwargs(
            {
                "symbol": symbol,
                "spread_type": SpreadType.SPOT_FUTURES,
                "signal_type": signal_type,
                "exchange_a": spot_exchange,
                "exchange_b": futures_exchange,
                "message": "basis data quality issue",
                "metadata": {
                    "spot_exchange": spot_exchange,
                    "futures_exchange": futures_exchange,
                },
            },
            overrides,
        )
    )


def make_arb_signal(
    *,
    symbol: str = "BTCUSDT",
    buy_exchange: str = "binance",
    sell_exchange: str = "bybit",
    instrument_type: InstrumentType | None = None,
    **overrides: Any,
) -> SpreadSignal:
    instrument_type = instrument_type or futures_type()

    return make_spread_signal(
        **_merge_kwargs(
            {
                "symbol": symbol,
                "spread_type": SpreadType.CROSS_EXCHANGE,
                "signal_type": SpreadSignalType.ARBITRAGE,
                "exchange_a": buy_exchange,
                "exchange_b": sell_exchange,
                "message": "cross exchange arbitrage confirmation",
                "metadata": {
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "instrument_type": instrument_type.value,
                    "buy_instrument_type": instrument_type.value,
                    "sell_instrument_type": instrument_type.value,
                },
            },
            overrides,
        )
    )


def make_arb_data_quality_signal(
    *,
    symbol: str = "BTCUSDT",
    buy_exchange: str = "binance",
    sell_exchange: str = "bybit",
    instrument_type: InstrumentType | None = None,
    signal_type: SpreadSignalType = SpreadSignalType.STALE_DATA,
    **overrides: Any,
) -> SpreadSignal:
    instrument_type = instrument_type or futures_type()

    return make_spread_signal(
        **_merge_kwargs(
            {
                "symbol": symbol,
                "spread_type": SpreadType.CROSS_EXCHANGE,
                "signal_type": signal_type,
                "exchange_a": buy_exchange,
                "exchange_b": sell_exchange,
                "message": "arb data quality issue",
                "metadata": {
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "instrument_type": instrument_type.value,
                    "buy_instrument_type": instrument_type.value,
                    "sell_instrument_type": instrument_type.value,
                },
            },
            overrides,
        )
    )


# ============================================================
# SpreadSnapshot factories
# ============================================================

def make_spot_futures_snapshot(
    *,
    symbol: str = "BTCUSDT",
    spot_exchange: str = "binance",
    futures_exchange: str = "bybit",
    timestamp: datetime | None = None,
    zscore: Decimal | int | float | str | None = "2.5",
    basis: Decimal | int | float | str | None = "100",
    spread_bps: Decimal | int | float | str | None = "25",
    spread_pct: Decimal | int | float | str | None = "0.25",
    raw_spread: Decimal | int | float | str | None = "100",
    net_spread: Decimal | int | float | str | None = "95",
    funding_adjusted_spread: Decimal | int | float | str | None = "80",
    regime: SpreadRegime | None = None,
    direction: Any | None = None,
    quote_validity: Any | None = None,
    pricing_source: Any | None = None,
    leg_a_mid: Decimal | int | float | str | None = "50000",
    leg_b_mid: Decimal | int | float | str | None = "50100",
    leg_a_bid: Decimal | int | float | str | None = "49999",
    leg_a_ask: Decimal | int | float | str | None = "50001",
    leg_b_bid: Decimal | int | float | str | None = "50099",
    leg_b_ask: Decimal | int | float | str | None = "50101",
    metadata: dict[str, Any] | None = None,
    stats: Any | None = None,
    **overrides: Any,
) -> SpreadSnapshot:
    return _build_model(
        SpreadSnapshot,
        symbol=symbol,
        spread_type=SpreadType.SPOT_FUTURES,
        leg_a_exchange=spot_exchange,
        leg_b_exchange=futures_exchange,
        leg_a_type=spot_type(),
        leg_b_type=futures_type(),
        leg_a_mid=d(leg_a_mid),
        leg_b_mid=d(leg_b_mid),
        leg_a_bid=d(leg_a_bid),
        leg_a_ask=d(leg_a_ask),
        leg_b_bid=d(leg_b_bid),
        leg_b_ask=d(leg_b_ask),
        basis=d(basis),
        raw_spread=d(raw_spread),
        spread_pct=d(spread_pct),
        spread_bps=d(spread_bps),
        net_spread=d(net_spread),
        funding_adjusted_spread=d(funding_adjusted_spread),
        estimated_fees=d("1"),
        estimated_slippage=d("1"),
        regime=regime or elevated_regime(),
        direction=direction or spread_direction_short(),
        quote_validity=quote_validity or valid_quote_status(),
        pricing_source=pricing_source or pricing_source_mid(),
        timestamp=timestamp or utcnow(),
        stats=stats if stats is not None else make_spread_stats(zscore=zscore),
        metadata=_metadata(
            metadata,
            spot_exchange=spot_exchange,
            futures_exchange=futures_exchange,
            basis=str(d(basis)) if basis is not None else None,
            zscore=str(d(zscore)) if zscore is not None else None,
        ),
        **overrides,
    )


def make_valid_basis_snapshot(**overrides: Any) -> SpreadSnapshot:
    return make_spot_futures_snapshot(
        **_merge_kwargs(
            {
                "zscore": "2.5",
                "basis": "100",
                "raw_spread": "100",
                "net_spread": "95",
                "funding_adjusted_spread": "80",
                "spread_bps": "25",
                "regime": elevated_regime(),
            },
            overrides,
        )
    )


def make_basis_close_snapshot(**overrides: Any) -> SpreadSnapshot:
    return make_spot_futures_snapshot(
        **_merge_kwargs(
            {
                "zscore": "0.25",
                "basis": "10",
                "raw_spread": "10",
                "net_spread": "8",
                "funding_adjusted_spread": "10",
                "spread_bps": "3",
                "regime": normal_regime(),
            },
            overrides,
        )
    )


def make_basis_stop_snapshot(**overrides: Any) -> SpreadSnapshot:
    return make_spot_futures_snapshot(
        **_merge_kwargs(
            {
                "zscore": "5.0",
                "basis": "180",
                "raw_spread": "180",
                "net_spread": "170",
                "funding_adjusted_spread": "120",
                "spread_bps": "45",
                "regime": dislocated_regime(),
            },
            overrides,
        )
    )


def make_stale_basis_snapshot(**overrides: Any) -> SpreadSnapshot:
    return make_spot_futures_snapshot(
        **_merge_kwargs(
            {
                "timestamp": stale_time(seconds=30),
                "zscore": "2.5",
                "basis": "100",
                "raw_spread": "100",
                "net_spread": "95",
                "funding_adjusted_spread": "80",
                "spread_bps": "25",
                "regime": elevated_regime(),
            },
            overrides,
        )
    )


def make_cross_exchange_snapshot(
    *,
    symbol: str = "BTCUSDT",
    buy_exchange: str = "binance",
    sell_exchange: str = "bybit",
    instrument_type: InstrumentType | None = None,
    timestamp: datetime | None = None,
    spread_bps: Decimal | int | float | str | None = "12",
    spread_pct: Decimal | int | float | str | None = "0.12",
    raw_spread: Decimal | int | float | str | None = "60",
    net_spread: Decimal | int | float | str | None = "55",
    opportunity_net_edge: Decimal | int | float | str | None = "55",
    opportunity_net_edge_bps: Decimal | int | float | str | None = "11",
    opportunity_confidence: Decimal | int | float | str | None = "0.90",
    opportunity_status: OpportunityStatus | str | None = None,
    regime: SpreadRegime | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> SpreadSnapshot:
    instrument_type = instrument_type or futures_type()
    status_value = (
        opportunity_status.value
        if hasattr(opportunity_status, "value")
        else opportunity_status
    )

    return _build_model(
        SpreadSnapshot,
        symbol=symbol,
        spread_type=SpreadType.CROSS_EXCHANGE,
        leg_a_exchange=buy_exchange,
        leg_b_exchange=sell_exchange,
        leg_a_type=instrument_type,
        leg_b_type=instrument_type,
        leg_a_mid=d("50000"),
        leg_b_mid=d("50060"),
        leg_a_bid=d("49999"),
        leg_a_ask=d("50001"),
        leg_b_bid=d("50059"),
        leg_b_ask=d("50061"),
        raw_spread=d(raw_spread),
        spread_pct=d(spread_pct),
        spread_bps=d(spread_bps),
        net_spread=d(net_spread),
        funding_adjusted_spread=None,
        estimated_fees=d("2"),
        estimated_slippage=d("3"),
        regime=regime or elevated_regime(),
        direction=spread_direction_long(),
        quote_validity=valid_quote_status(),
        pricing_source=pricing_source_mid(),
        timestamp=timestamp or utcnow(),
        stats=make_spread_stats(zscore="1.0"),
        metadata=_metadata(
            metadata,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            instrument_type=instrument_type.value,
            buy_instrument_type=instrument_type.value,
            sell_instrument_type=instrument_type.value,
            opportunity_net_edge=str(d(opportunity_net_edge)),
            opportunity_net_edge_bps=str(d(opportunity_net_edge_bps)),
            opportunity_confidence=str(d(opportunity_confidence)),
            opportunity_status=status_value or active_status().value,
        ),
        **overrides,
    )


def make_arb_snapshot_edge_lost(**overrides: Any) -> SpreadSnapshot:
    return make_cross_exchange_snapshot(
        **_merge_kwargs(
            {
                "net_spread": "0",
                "spread_bps": "0",
                "opportunity_net_edge": "0",
                "opportunity_net_edge_bps": "0",
            },
            overrides,
        )
    )


def make_arb_snapshot_inactive(**overrides: Any) -> SpreadSnapshot:
    return make_cross_exchange_snapshot(
        **_merge_kwargs(
            {
                "opportunity_status": expired_status(),
            },
            overrides,
        )
    )


def make_stale_arb_snapshot(**overrides: Any) -> SpreadSnapshot:
    return make_cross_exchange_snapshot(
        **_merge_kwargs(
            {
                "timestamp": stale_time(seconds=30),
            },
            overrides,
        )
    )


# ============================================================
# ArbitrageOpportunity factories
# ============================================================

def make_arbitrage_opportunity(
    *,
    symbol: str = "BTCUSDT",
    buy_exchange: str = "binance",
    sell_exchange: str = "bybit",
    buy_instrument_type: InstrumentType | None = None,
    sell_instrument_type: InstrumentType | None = None,
    buy_price: Decimal | int | float | str | None = "50000",
    sell_price: Decimal | int | float | str | None = "50080",
    gross_edge: Decimal | int | float | str | None = "80",
    estimated_fees: Decimal | int | float | str | None = "10",
    estimated_slippage: Decimal | int | float | str | None = "5",
    net_edge: Decimal | int | float | str | None = "65",
    spread_pct: Decimal | int | float | str | None = "0.16",
    spread_bps: Decimal | int | float | str | None = "16",
    confidence: Decimal | int | float | str | None = "0.90",
    status: OpportunityStatus | None = None,
    is_profitable: bool = True,
    timestamp: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ArbitrageOpportunity:
    buy_instrument_type = buy_instrument_type or futures_type()
    sell_instrument_type = sell_instrument_type or futures_type()

    return _build_model(
        ArbitrageOpportunity,
        symbol=symbol,
        buy_exchange=buy_exchange,
        sell_exchange=sell_exchange,
        buy_instrument_type=buy_instrument_type,
        sell_instrument_type=sell_instrument_type,
        buy_price=d(buy_price),
        sell_price=d(sell_price),
        gross_edge=d(gross_edge),
        estimated_fees=d(estimated_fees),
        estimated_slippage=d(estimated_slippage),
        net_edge=d(net_edge),
        spread_pct=d(spread_pct),
        spread_bps=d(spread_bps),
        confidence=d(confidence),
        status=status or active_status(),
        is_profitable=is_profitable,
        timestamp=timestamp or utcnow(),
        expires_at=expires_at or future_time(seconds=30),
        metadata=_metadata(
            metadata,
            net_edge_bps=str(d(spread_bps)),
            reference_buy_notional="50000",
            buy_instrument_type=buy_instrument_type.value,
            sell_instrument_type=sell_instrument_type.value,
        ),
        **overrides,
    )


def make_valid_arb_opportunity(**overrides: Any) -> ArbitrageOpportunity:
    return make_arbitrage_opportunity(
        **_merge_kwargs(
            {
                "net_edge": "65",
                "spread_bps": "16",
                "confidence": "0.90",
                "status": active_status(),
                "is_profitable": True,
                "expires_at": future_time(seconds=30),
            },
            overrides,
        )
    )


def make_unprofitable_arb_opportunity(**overrides: Any) -> ArbitrageOpportunity:
    return make_arbitrage_opportunity(
        **_merge_kwargs(
            {
                "net_edge": "0",
                "spread_bps": "0",
                "confidence": "0.90",
                "status": active_status(),
                "is_profitable": False,
            },
            overrides,
        )
    )


def make_expired_arb_opportunity(**overrides: Any) -> ArbitrageOpportunity:
    return make_arbitrage_opportunity(
        **_merge_kwargs(
            {
                "timestamp": stale_time(seconds=60),
                "expires_at": stale_time(seconds=30),
                "status": active_status(),
                "is_profitable": True,
            },
            overrides,
        )
    )


def make_inactive_arb_opportunity(**overrides: Any) -> ArbitrageOpportunity:
    return make_arbitrage_opportunity(
        **_merge_kwargs(
            {
                "status": expired_status(),
                "is_profitable": False,
            },
            overrides,
        )
    )