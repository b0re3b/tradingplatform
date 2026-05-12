from __future__ import annotations

from enum import Enum
from typing import Self


class StrEnumMixin(str, Enum):
    """
    Базовий mixin для string-based enum-контрактів.

    Призначення:
    - стабільні string values для EventBus payload / storage / dashboard;
    - безпечне створення enum з raw string;
    - єдиний API values()/has_value()/from_value();
    - відсутність залежності від EventBus/Scheduler/logger.

    Важливо:
    цей mixin не містить runtime-логіки й не має side effects.
    """

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(item.value for item in cls)

    @classmethod
    def has_value(cls, value: str | Self | None) -> bool:
        if value is None:
            return False

        if isinstance(value, cls):
            return True

        normalized = str(value).strip().lower()
        return normalized in cls.values()

    @classmethod
    def from_value(
        cls,
        value: str | Self | None,
        *,
        default: Self | None = None,
        strict: bool = False,
    ) -> Self:
        """
        Безпечно конвертує raw string у enum.

        Args:
            value:
                Raw string або вже готовий enum.
            default:
                Значення за замовчуванням, якщо value не знайдено.
            strict:
                Якщо True — кидати ValueError замість default.

        Returns:
            Enum value.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not None:
                return default
            if strict:
                raise ValueError(f"{cls.__name__} value must not be None")
            raise ValueError(f"Invalid {cls.__name__}: {value!r}")

        normalized = str(value).strip().lower()

        for item in cls:
            if item.value == normalized:
                return item

        if default is not None:
            return default

        if strict:
            raise ValueError(
                f"Invalid {cls.__name__}: {value!r}. "
                f"Allowed values: {', '.join(cls.values())}"
            )

        raise ValueError(
            f"Invalid {cls.__name__}: {value!r}. "
            f"Allowed values: {', '.join(cls.values())}"
        )

    def __str__(self) -> str:
        return self.value


class SpreadType(StrEnumMixin):
    """
    Тип spread-аналітики.
    """

    SPOT_FUTURES = "spot_futures"
    CROSS_EXCHANGE = "cross_exchange"

    @property
    def is_spot_futures(self) -> bool:
        return self is SpreadType.SPOT_FUTURES

    @property
    def is_cross_exchange(self) -> bool:
        return self is SpreadType.CROSS_EXCHANGE


class InstrumentType(StrEnumMixin):
    """
    Тип торгового інструмента.

    UNKNOWN дозволений для raw market-data normalization,
    але не має використовуватись у production spread calculations.
    """

    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURES = "futures"
    UNKNOWN = "unknown"

    @property
    def is_derivative(self) -> bool:
        return self in {
            InstrumentType.PERPETUAL,
            InstrumentType.FUTURES,
        }

    @property
    def is_tradeable(self) -> bool:
        return self is not InstrumentType.UNKNOWN

    @property
    def is_spot(self) -> bool:
        return self is InstrumentType.SPOT

    @classmethod
    def derivatives(cls) -> set["InstrumentType"]:
        return {
            cls.PERPETUAL,
            cls.FUTURES,
        }

    @classmethod
    def spread_supported(cls) -> set["InstrumentType"]:
        return {
            cls.SPOT,
            cls.PERPETUAL,
            cls.FUTURES,
        }


class SpreadSignalType(StrEnumMixin):
    """
    Тип spread-сигналу, який може бути опублікований у:
    analytics.spreads.signal.generated
    """

    WIDENING = "widening"
    NARROWING = "narrowing"
    ANOMALY = "anomaly"
    ARBITRAGE = "arbitrage"
    MEAN_REVERSION = "mean_reversion"
    REGIME_SHIFT = "regime_shift"
    STALE_DATA = "stale_data"
    INVALID_DATA = "invalid_data"

    @property
    def is_actionable(self) -> bool:
        """
        Чи може сигнал потенційно бути використаний strategy layer.
        """
        return self in {
            SpreadSignalType.WIDENING,
            SpreadSignalType.NARROWING,
            SpreadSignalType.ANOMALY,
            SpreadSignalType.ARBITRAGE,
            SpreadSignalType.MEAN_REVERSION,
            SpreadSignalType.REGIME_SHIFT,
        }

    @property
    def is_data_quality(self) -> bool:
        """
        Чи є сигнал службовим сигналом якості даних.
        """
        return self in {
            SpreadSignalType.STALE_DATA,
            SpreadSignalType.INVALID_DATA,
        }

    @property
    def is_arbitrage(self) -> bool:
        return self is SpreadSignalType.ARBITRAGE


class SpreadDirection(StrEnumMixin):
    """
    Напрям spread-значення.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        if self is SpreadDirection.POSITIVE:
            return 1
        if self is SpreadDirection.NEGATIVE:
            return -1
        return 0

    @property
    def is_directional(self) -> bool:
        return self is not SpreadDirection.FLAT


class SpreadRegime(StrEnumMixin):
    """
    Режим spread-стану за rolling stats / z-score.
    """

    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"
    COMPRESSED = "compressed"
    DISLOCATED = "dislocated"

    @property
    def rank(self) -> int:
        """
        Severity/rank для порівняння режимів.
        """
        return {
            SpreadRegime.COMPRESSED: 0,
            SpreadRegime.NORMAL: 1,
            SpreadRegime.ELEVATED: 2,
            SpreadRegime.EXTREME: 3,
            SpreadRegime.DISLOCATED: 4,
        }[self]

    @property
    def is_abnormal(self) -> bool:
        return self in {
            SpreadRegime.ELEVATED,
            SpreadRegime.EXTREME,
            SpreadRegime.DISLOCATED,
        }

    @property
    def is_high_risk(self) -> bool:
        return self in {
            SpreadRegime.EXTREME,
            SpreadRegime.DISLOCATED,
        }


class OpportunityStatus(StrEnumMixin):
    """
    Lifecycle status для ArbitrageOpportunity.
    """

    ACTIVE = "active"
    EXPIRED = "expired"
    REJECTED = "rejected"
    EXECUTED = "executed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OpportunityStatus.EXPIRED,
            OpportunityStatus.REJECTED,
            OpportunityStatus.EXECUTED,
        }

    @property
    def can_execute(self) -> bool:
        return self is OpportunityStatus.ACTIVE


class QuoteValidity(StrEnumMixin):
    """
    Результат validation quote snapshot.
    """

    VALID = "valid"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"

    @property
    def is_usable(self) -> bool:
        return self is QuoteValidity.VALID

    @property
    def is_rejectable(self) -> bool:
        return self in {
            QuoteValidity.STALE,
            QuoteValidity.INVALID,
        }

    @property
    def is_missing_data(self) -> bool:
        return self is QuoteValidity.INCOMPLETE


class PricingSource(StrEnumMixin):
    """
    Джерело ціни для spread calculation.
    """

    BID_ASK = "bid_ask"
    MID = "mid"
    LAST = "last"
    MARK = "mark"
    INDEX = "index"

    @property
    def is_orderbook_based(self) -> bool:
        return self in {
            PricingSource.BID_ASK,
            PricingSource.MID,
        }

    @property
    def is_reference_price(self) -> bool:
        return self in {
            PricingSource.MARK,
            PricingSource.INDEX,
        }


# ============================================================
# Public helpers
# ============================================================

def parse_instrument_type(value: str | InstrumentType | None) -> InstrumentType:
    """
    Безпечний parser для raw exchange payload.

    Unknown/empty значення повертають InstrumentType.UNKNOWN.
    """
    return InstrumentType.from_value(
        value,
        default=InstrumentType.UNKNOWN,
    )


def parse_spread_type(value: str | SpreadType) -> SpreadType:
    return SpreadType.from_value(value, strict=True)


def parse_pricing_source(value: str | PricingSource | None) -> PricingSource:
    return PricingSource.from_value(
        value,
        default=PricingSource.BID_ASK,
    )


__all__ = [
    "StrEnumMixin",
    "SpreadType",
    "InstrumentType",
    "SpreadSignalType",
    "SpreadDirection",
    "SpreadRegime",
    "OpportunityStatus",
    "QuoteValidity",
    "PricingSource",
    "parse_instrument_type",
    "parse_spread_type",
    "parse_pricing_source",
]