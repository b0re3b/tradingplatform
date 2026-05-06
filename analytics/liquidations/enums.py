from __future__ import annotations

from enum import Enum


class LiquidationSide(str, Enum):
    """
    Сторона ліквідації.

    LONG:
        Ліквідовано long-позиції. Зазвичай це супроводжується sell pressure
        і потенційним рухом вниз.

    SHORT:
        Ліквідовано short-позиції. Зазвичай це супроводжується buy pressure
        і потенційним рухом вгору.

    UNKNOWN:
        Сторону не вдалося визначити з payload-а біржі.
    """

    LONG = "long"
    SHORT = "short"
    UNKNOWN = "unknown"

    @property
    def opposite(self) -> "LiquidationSide":
        if self is LiquidationSide.LONG:
            return LiquidationSide.SHORT
        if self is LiquidationSide.SHORT:
            return LiquidationSide.LONG
        return LiquidationSide.UNKNOWN

    @property
    def is_known(self) -> bool:
        return self is not LiquidationSide.UNKNOWN

    @property
    def is_bearish_pressure(self) -> bool:
        """
        Long liquidations зазвичай створюють sell pressure.
        """
        return self is LiquidationSide.LONG

    @property
    def is_bullish_pressure(self) -> bool:
        """
        Short liquidations зазвичай створюють buy pressure.
        """
        return self is LiquidationSide.SHORT

    @classmethod
    def from_raw(cls, value: object) -> "LiquidationSide":
        """
        Нормалізує raw side/position side/order side з біржових payload-ів.

        Важливо:
        - liquidation of LONG position -> LONG liquidation;
        - liquidation of SHORT position -> SHORT liquidation;
        - SELL liquidation order часто означає ліквідацію long;
        - BUY liquidation order часто означає ліквідацію short.
        """
        if value is None:
            return cls.UNKNOWN

        raw = str(value).strip().lower()

        if raw in {
            "long",
            "longs",
            "buy_long",
            "position_long",
            "liquidated_long",
            "long_liquidation",
        }:
            return cls.LONG

        if raw in {
            "short",
            "shorts",
            "sell_short",
            "position_short",
            "liquidated_short",
            "short_liquidation",
        }:
            return cls.SHORT

        if raw in {"sell", "ask", "s"}:
            return cls.LONG

        if raw in {"buy", "bid", "b"}:
            return cls.SHORT

        return cls.UNKNOWN


class LiquidationEventType(str, Enum):
    """
    Тип події у liquidation pipeline.

    RAW:
        Сирий payload із біржі.

    NORMALIZED:
        Валідована й нормалізована LiquidationEvent.

    LARGE:
        Окрема подія для великої ліквідації.

    CLUSTER_CANDIDATE:
        Потенційний liquidation cluster.

    CASCADE:
        Підтверджений liquidation cascade.

    EXHAUSTION:
        Ознака потенційного exhaustion після каскаду.

    SNAPSHOT / HEALTH:
        Службові події для dashboard/storage/monitoring.
    """

    RAW = "raw"
    NORMALIZED = "normalized"
    LARGE = "large"
    CLUSTER_CANDIDATE = "cluster_candidate"
    CASCADE = "cascade"
    EXHAUSTION = "exhaustion"
    SNAPSHOT = "snapshot"
    HEALTH = "health"


class CascadeDirection(str, Enum):
    """
    Напрям каскаду.

    DOWN:
        Каскад long-liquidations, потенційний тиск вниз.

    UP:
        Каскад short-liquidations, потенційний тиск вгору.

    UNKNOWN:
        Недостатньо даних для визначення напряму.
    """

    DOWN = "down"
    UP = "up"
    UNKNOWN = "unknown"

    @property
    def is_known(self) -> bool:
        return self is not CascadeDirection.UNKNOWN

    @property
    def is_bearish(self) -> bool:
        return self is CascadeDirection.DOWN

    @property
    def is_bullish(self) -> bool:
        return self is CascadeDirection.UP

    @classmethod
    def from_side(cls, side: LiquidationSide) -> "CascadeDirection":
        if side is LiquidationSide.LONG:
            return cls.DOWN
        if side is LiquidationSide.SHORT:
            return cls.UP
        return cls.UNKNOWN


class CascadeSeverity(str, Enum):
    """
    Дискретна оцінка сили liquidation cascade.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        if self is CascadeSeverity.LOW:
            return 1
        if self is CascadeSeverity.MEDIUM:
            return 2
        if self is CascadeSeverity.HIGH:
            return 3
        if self is CascadeSeverity.EXTREME:
            return 4
        return 0

    @property
    def is_actionable(self) -> bool:
        """
        Умовний helper для strategy/dashboard layer.

        Це не торгове рішення, а лише технічна класифікація сили.
        Strategy/Risk все одно мають самі вирішувати, що робити із сигналом.
        """
        return self in {CascadeSeverity.HIGH, CascadeSeverity.EXTREME}

    @classmethod
    def from_score(
        cls,
        score: float,
        *,
        low_threshold: float = 0.30,
        medium_threshold: float = 0.55,
        high_threshold: float = 0.75,
        extreme_threshold: float = 0.90,
    ) -> "CascadeSeverity":
        if score >= extreme_threshold:
            return cls.EXTREME
        if score >= high_threshold:
            return cls.HIGH
        if score >= medium_threshold:
            return cls.MEDIUM
        if score >= low_threshold:
            return cls.LOW
        return cls.LOW


class LiquidationStatus(str, Enum):
    """
    Статус lifecycle для liquidation event / cluster / detector result.
    """

    NEW = "new"
    ACTIVE = "active"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    COOLDOWN = "cooldown"
    EXPIRED = "expired"
    FAILED = "failed"

    @property
    def is_final(self) -> bool:
        return self in {
            LiquidationStatus.CONFIRMED,
            LiquidationStatus.REJECTED,
            LiquidationStatus.EXPIRED,
            LiquidationStatus.FAILED,
        }

    @property
    def is_active(self) -> bool:
        return self in {
            LiquidationStatus.NEW,
            LiquidationStatus.ACTIVE,
            LiquidationStatus.CANDIDATE,
            LiquidationStatus.COOLDOWN,
        }