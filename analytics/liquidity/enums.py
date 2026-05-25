from __future__ import annotations
from core.logger import get_logger

from enum import Enum


class LiquiditySide(str, Enum):
    """
    Сторона ліквідності відносно ринку.

    BUY_SIDE:
        Ліквідність над ціною, де зазвичай знаходяться buy stops шортистів.
    SELL_SIDE:
        Ліквідність під ціною, де зазвичай знаходяться sell stops лонгістів.
    BOTH:
        Рівень або зона має ліквідність з обох сторін.
    UNKNOWN:
        Сторону неможливо надійно визначити.
    """

    BUY_SIDE = "buy_side"
    SELL_SIDE = "sell_side"
    BOTH = "both"
    UNKNOWN = "unknown"

    @property
    def is_buy_side(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_buy_side", _analytics_args)
        except Exception:
            pass
        return self == LiquiditySide.BUY_SIDE

    @property
    def is_sell_side(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_sell_side", _analytics_args)
        except Exception:
            pass
        return self == LiquiditySide.SELL_SIDE

    @property
    def is_directional(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_directional", _analytics_args)
        except Exception:
            pass
        return self in {LiquiditySide.BUY_SIDE, LiquiditySide.SELL_SIDE}


class LiquidityLevelType(str, Enum):
    """
    Тип liquidity-рівня або liquidity-зони.

    Ці значення використовуються у liquidity-модулі, strategy layer,
    dashboard, AI-поясненнях і event payload-ах.
    """

    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"

    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"

    STOP_CLUSTER = "stop_cluster"

    RANGE_HIGH = "range_high"
    RANGE_LOW = "range_low"

    ORDERBOOK_WALL = "orderbook_wall"
    LIQUIDATION_ZONE = "liquidation_zone"

    @property
    def is_buy_side_type(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_buy_side_type", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.RANGE_HIGH,
        }

    @property
    def is_sell_side_type(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_sell_side_type", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_LOW,
        }

    @property
    def is_structural(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_structural", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidityLevelType.EQUAL_HIGHS,
            LiquidityLevelType.EQUAL_LOWS,
            LiquidityLevelType.SWING_HIGH,
            LiquidityLevelType.SWING_LOW,
            LiquidityLevelType.RANGE_HIGH,
            LiquidityLevelType.RANGE_LOW,
        }

    @property
    def is_external_source(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_external_source", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidityLevelType.ORDERBOOK_WALL,
            LiquidityLevelType.LIQUIDATION_ZONE,
        }

    def infer_side(self) -> LiquiditySide:
        """
        Визначає сторону ліквідності за типом рівня.
        """
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "infer_side", _analytics_args)
        except Exception:
            pass
        if self.is_buy_side_type:
            return LiquiditySide.BUY_SIDE
        if self.is_sell_side_type:
            return LiquiditySide.SELL_SIDE
        return LiquiditySide.UNKNOWN


class LiquidityStatus(str, Enum):
    """
    Поточний lifecycle-статус liquidity-рівня.
    """

    ACTIVE = "active"
    SWEPT = "swept"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    WEAK = "weak"

    @property
    def is_active(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_active", _analytics_args)
        except Exception:
            pass
        return self == LiquidityStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_terminal", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidityStatus.SWEPT,
            LiquidityStatus.INVALIDATED,
            LiquidityStatus.EXPIRED,
        }


class SweepStatus(str, Enum):
    """
    Статус sweep-поведінки liquidity-рівня.
    """

    NOT_SWEPT = "not_swept"
    PARTIALLY_SWEPT = "partially_swept"
    SWEPT = "swept"
    UNKNOWN = "unknown"

    @property
    def is_swept(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_swept", _analytics_args)
        except Exception:
            pass
        return self == SweepStatus.SWEPT

    @property
    def is_partially_swept(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_partially_swept", _analytics_args)
        except Exception:
            pass
        return self == SweepStatus.PARTIALLY_SWEPT

    @property
    def has_any_sweep(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "has_any_sweep", _analytics_args)
        except Exception:
            pass
        return self in {
            SweepStatus.PARTIALLY_SWEPT,
            SweepStatus.SWEPT,
        }


class ClusterStrength(str, Enum):
    """
    Якісна оцінка сили stop/liquidity cluster.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

    @property
    def rank(self) -> int:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "rank", _analytics_args)
        except Exception:
            pass
        ranks = {
            ClusterStrength.LOW: 1,
            ClusterStrength.MEDIUM: 2,
            ClusterStrength.HIGH: 3,
            ClusterStrength.EXTREME: 4,
        }
        return ranks[self]

    def is_at_least(self, other: "ClusterStrength") -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_at_least", _analytics_args)
        except Exception:
            pass
        return self.rank >= other.rank


class LiquidityBias(str, Enum):
    """
    Агрегований liquidity-bias для strategy layer.
    """

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"

    @property
    def is_directional(self) -> bool:
        try:
            _analytics_logger = getattr(self, "logger", None) or getattr(self, "_logger", None)
            if _analytics_logger is None:
                _analytics_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
                self.logger = _analytics_logger
            _analytics_class_name = self.__class__.__name__
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in (locals().items() if _analytics_logger.isEnabledFor(10) else ())
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_directional", _analytics_args)
        except Exception:
            pass
        return self in {LiquidityBias.UP, LiquidityBias.DOWN}