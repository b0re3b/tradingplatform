from __future__ import annotations

from enum import Enum


class LiquidationSide(str, Enum):
    """
    Сторона ліквідації.

    LONG  -> ліквідували long-позиції, зазвичай це супроводжується тиском вниз.
    SHORT -> ліквідували short-позиції, зазвичай це супроводжується тиском вгору.
    UNKNOWN -> не вдалося коректно визначити сторону.
    """

    LONG = "long"
    SHORT = "short"
    UNKNOWN = "unknown"

    @property
    def opposite(self) -> "LiquidationSide":
        if self == LiquidationSide.LONG:
            return LiquidationSide.SHORT
        if self == LiquidationSide.SHORT:
            return LiquidationSide.LONG
        return LiquidationSide.UNKNOWN


class LiquidationEventType(str, Enum):
    """
    Тип події у liquidation pipeline.
    """

    RAW = "raw"
    NORMALIZED = "normalized"
    LARGE = "large"
    CLUSTER_CANDIDATE = "cluster_candidate"
    CASCADE = "cascade"
    EXHAUSTION = "exhaustion"


class CascadeDirection(str, Enum):
    """
    Напрям каскаду.
    DOWN -> каскад long-liquidations
    UP   -> каскад short-liquidations
    UNKNOWN -> не вистачило сигналу
    """

    DOWN = "down"
    UP = "up"
    UNKNOWN = "unknown"

    @classmethod
    def from_side(cls, side: LiquidationSide) -> "CascadeDirection":
        if side == LiquidationSide.LONG:
            return cls.DOWN
        if side == LiquidationSide.SHORT:
            return cls.UP
        return cls.UNKNOWN


class CascadeSeverity(str, Enum):
    """
    Дискретна оцінка сили каскаду.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class LiquidationStatus(str, Enum):
    """
    Статус детекції або життєвого циклу кластеру/сигналу.
    """

    NEW = "new"
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    COOLDOWN = "cooldown"
    EXPIRED = "expired"