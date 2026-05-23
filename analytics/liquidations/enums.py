from __future__ import annotations
from core.logger import get_logger

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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "opposite", _analytics_args)
        except Exception:
            pass
        if self is LiquidationSide.LONG:
            return LiquidationSide.SHORT
        if self is LiquidationSide.SHORT:
            return LiquidationSide.LONG
        return LiquidationSide.UNKNOWN

    @property
    def is_known(self) -> bool:
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_known", _analytics_args)
        except Exception:
            pass
        return self is not LiquidationSide.UNKNOWN

    @property
    def is_bearish_pressure(self) -> bool:
        """
        Long liquidations зазвичай створюють sell pressure.
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bearish_pressure", _analytics_args)
        except Exception:
            pass
        return self is LiquidationSide.LONG

    @property
    def is_bullish_pressure(self) -> bool:
        """
        Short liquidations зазвичай створюють buy pressure.
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bullish_pressure", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "LiquidationSide"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_raw", _analytics_args)
        except Exception:
            pass
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_known", _analytics_args)
        except Exception:
            pass
        return self is not CascadeDirection.UNKNOWN

    @property
    def is_bearish(self) -> bool:
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bearish", _analytics_args)
        except Exception:
            pass
        return self is CascadeDirection.DOWN

    @property
    def is_bullish(self) -> bool:
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_bullish", _analytics_args)
        except Exception:
            pass
        return self is CascadeDirection.UP

    @classmethod
    def from_side(cls, side: LiquidationSide) -> "CascadeDirection":
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "CascadeDirection"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_side", _analytics_args)
        except Exception:
            pass
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "rank", _analytics_args)
        except Exception:
            pass
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_actionable", _analytics_args)
        except Exception:
            pass
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
        try:
            _analytics_class_name = cls.__name__ if "cls" in locals() else "CascadeSeverity"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
            _analytics_args = {
                _k: (
                    {"type": "dict", "size": len(_v), "keys": [str(_key) for _key in list(_v.keys())[:20]]}
                    if isinstance(_v, dict)
                    else {"type": type(_v).__name__, "size": len(_v)}
                    if isinstance(_v, (list, tuple, set, frozenset))
                    else {"type": type(_v).__name__}
                )
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "from_score", _analytics_args)
        except Exception:
            pass
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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_final", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidationStatus.CONFIRMED,
            LiquidationStatus.REJECTED,
            LiquidationStatus.EXPIRED,
            LiquidationStatus.FAILED,
        }

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
                for _k, _v in locals().items()
                if _k not in {"self", "cls", "_analytics_logger", "_analytics_class_name", "_analytics_args"}
                and not _k.startswith("_analytics")
            }
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_active", _analytics_args)
        except Exception:
            pass
        return self in {
            LiquidationStatus.NEW,
            LiquidationStatus.ACTIVE,
            LiquidationStatus.CANDIDATE,
            LiquidationStatus.COOLDOWN,
        }