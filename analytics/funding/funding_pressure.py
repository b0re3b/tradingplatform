from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from analytics.funding.enums import (
    FundingBias,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingTimeframe,
)
from analytics.funding.models import (
    FundingPressureState,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
    funding_key_to_dict,
)


@dataclass(slots=True)
class FundingPressureConfig:
    """
    Конфігурація pure analyzer-а funding pressure.

    FundingPressureAnalyzer не є runtime-компонентом:
    - не слухає EventBus;
    - не має Scheduler jobs;
    - не читає exchange/data cache напряму;
    - не зберігає історію.

    Runtime orchestration виконує FundingAnalyzer.
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    # Абсолютні пороги funding magnitude.
    neutral_abs_threshold: float = 0.00001
    elevated_abs_threshold: float = 0.00008
    extreme_abs_threshold: float = 0.00030

    # Pressure score thresholds.
    moderate_pressure_score_threshold: float = 0.45
    high_pressure_score_threshold: float = 0.70
    extreme_pressure_score_threshold: float = 0.90

    # Distribution thresholds.
    crowded_percentile_threshold: float = 85.0
    squeeze_percentile_threshold: float = 95.0
    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Price/OI context thresholds.
    price_stall_threshold_pct: float = 0.0010
    oi_growth_threshold_pct: float = 0.005

    # Pressure score weights.
    weight_magnitude: float = 0.30
    weight_percentile: float = 0.25
    weight_zscore: float = 0.15
    weight_oi_confirmation: float = 0.15
    weight_price_stall: float = 0.15

    # Squeeze probability weights.
    squeeze_pressure_weight: float = 0.50
    squeeze_percentile_weight: float = 0.20
    squeeze_oi_weight: float = 0.15
    squeeze_stall_weight: float = 0.15

    # Mean-reversion probability weights.
    mean_reversion_pressure_weight: float = 0.45
    mean_reversion_percentile_weight: float = 0.20
    mean_reversion_zscore_weight: float = 0.20
    mean_reversion_stall_weight: float = 0.15

    service_name: str = "funding_pressure_analyzer"

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.validate()

    def validate(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "validate", _analytics_args)
        except Exception:
            pass
        if self.neutral_abs_threshold < 0:
            raise ValueError("neutral_abs_threshold must be >= 0")
        if self.elevated_abs_threshold < 0:
            raise ValueError("elevated_abs_threshold must be >= 0")
        if self.extreme_abs_threshold < 0:
            raise ValueError("extreme_abs_threshold must be >= 0")

        if self.neutral_abs_threshold > self.elevated_abs_threshold:
            raise ValueError("neutral_abs_threshold must be <= elevated_abs_threshold")
        if self.elevated_abs_threshold > self.extreme_abs_threshold:
            raise ValueError("elevated_abs_threshold must be <= extreme_abs_threshold")

        self._validate_ratio(
            "moderate_pressure_score_threshold",
            self.moderate_pressure_score_threshold,
        )
        self._validate_ratio(
            "high_pressure_score_threshold",
            self.high_pressure_score_threshold,
        )
        self._validate_ratio(
            "extreme_pressure_score_threshold",
            self.extreme_pressure_score_threshold,
        )

        if self.moderate_pressure_score_threshold > self.high_pressure_score_threshold:
            raise ValueError(
                "moderate_pressure_score_threshold must be <= high_pressure_score_threshold"
            )
        if self.high_pressure_score_threshold > self.extreme_pressure_score_threshold:
            raise ValueError(
                "high_pressure_score_threshold must be <= extreme_pressure_score_threshold"
            )

        self._validate_percentile(
            "crowded_percentile_threshold",
            self.crowded_percentile_threshold,
        )
        self._validate_percentile(
            "squeeze_percentile_threshold",
            self.squeeze_percentile_threshold,
        )

        if self.crowded_percentile_threshold > self.squeeze_percentile_threshold:
            raise ValueError(
                "crowded_percentile_threshold must be <= squeeze_percentile_threshold"
            )

        if self.elevated_zscore_threshold < 0:
            raise ValueError("elevated_zscore_threshold must be >= 0")
        if self.extreme_zscore_threshold < 0:
            raise ValueError("extreme_zscore_threshold must be >= 0")
        if self.elevated_zscore_threshold > self.extreme_zscore_threshold:
            raise ValueError("elevated_zscore_threshold must be <= extreme_zscore_threshold")

        if self.price_stall_threshold_pct < 0:
            raise ValueError("price_stall_threshold_pct must be >= 0")
        if self.oi_growth_threshold_pct < 0:
            raise ValueError("oi_growth_threshold_pct must be >= 0")

        self._validate_weight_group(
            {
                "weight_magnitude": self.weight_magnitude,
                "weight_percentile": self.weight_percentile,
                "weight_zscore": self.weight_zscore,
                "weight_oi_confirmation": self.weight_oi_confirmation,
                "weight_price_stall": self.weight_price_stall,
            },
            expected_sum=1.0,
        )

        self._validate_weight_group(
            {
                "squeeze_pressure_weight": self.squeeze_pressure_weight,
                "squeeze_percentile_weight": self.squeeze_percentile_weight,
                "squeeze_oi_weight": self.squeeze_oi_weight,
                "squeeze_stall_weight": self.squeeze_stall_weight,
            },
            expected_sum=1.0,
        )

        self._validate_weight_group(
            {
                "mean_reversion_pressure_weight": self.mean_reversion_pressure_weight,
                "mean_reversion_percentile_weight": self.mean_reversion_percentile_weight,
                "mean_reversion_zscore_weight": self.mean_reversion_zscore_weight,
                "mean_reversion_stall_weight": self.mean_reversion_stall_weight,
            },
            expected_sum=1.0,
        )

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        try:
            _analytics_class_name = "FundingPressureConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_ratio", _analytics_args)
        except Exception:
            pass
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _validate_percentile(name: str, value: float) -> None:
        try:
            _analytics_class_name = "FundingPressureConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_percentile", _analytics_args)
        except Exception:
            pass
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"{name} must be in [0, 100]")

    @staticmethod
    def _validate_weight_group(
        values: dict[str, float],
        *,
        expected_sum: float,
    ) -> None:
        try:
            _analytics_class_name = "FundingPressureConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_weight_group", _analytics_args)
        except Exception:
            pass
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        total = sum(values.values())
        if abs(total - expected_sum) > 1e-9:
            names = ", ".join(values.keys())
            raise ValueError(f"weights must sum to {expected_sum}: {names}")


class FundingPressureAnalyzer:
    """
    Pure analyzer для funding pressure.

    Відповідальність:
    - будує FundingPressureState на основі snapshot/statistics/regime/context;
    - оцінює OI confirmation;
    - оцінює price stall;
    - рахує pressure_score;
    - оцінює squeeze_probability;
    - оцінює mean_reversion_probability;
    - повертає FundingPressureState з повним futures scope.

    Correct architecture:
        FundingAnalyzer
            -> FundingPressureAnalyzer.analyze(...)
            -> FundingPressureState
            -> FundingAnalyzer публікує analytics.funding.*

    Важливо:
    - не слухає EventBus;
    - не публікує EventBus events;
    - не має Scheduler jobs;
    - не читає exchange/data caches напряму;
    - не зберігає історію самостійно.
    """

    def __init__(
        self,
        config: FundingPressureConfig | None = None,
    ) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        self.config = config or FundingPressureConfig()
        self.config.validate()

        self.logger = get_logger(
            __name__,
            service_name=self.config.service_name,
            event_type="funding_pressure_analyzer",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        previous_snapshot: FundingSnapshot | None = None,
        previous_open_interest: float | None = None,
        current_price: float | None = None,
        previous_price: float | None = None,
        timeframe: FundingTimeframe | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> FundingPressureState:
        """
        Побудова FundingPressureState для поточного funding snapshot.

        Очікується, що snapshot/statistics/regime_state належать одному scope:
            exchange + market_type + symbol + timeframe

        FundingAnalyzer відповідає за:
        - EventBus subscriptions;
        - context cache;
        - previous_snapshot;
        - previous_open_interest;
        - publish output events.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze", _analytics_args)
        except Exception:
            pass
        self._validate_input_scope(
            snapshot=snapshot,
            statistics=statistics,
            regime_state=regime_state,
            previous_snapshot=previous_snapshot,
        )

        tf = timeframe or statistics.timeframe or snapshot.timeframe or self.config.default_timeframe

        oi_confirmation, oi_change_pct = self._detect_oi_confirmation(
            current_open_interest=snapshot.open_interest,
            previous_open_interest=previous_open_interest,
        )

        price_stall_confirmation, price_change_pct = self._detect_price_stall(
            current_price=(
                current_price
                if current_price is not None
                else snapshot.mark_price
            ),
            previous_price=(
                previous_price
                if previous_price is not None
                else (
                    previous_snapshot.mark_price
                    if previous_snapshot is not None
                    else None
                )
            ),
        )

        magnitude_score = self._calc_magnitude_score(snapshot.funding_rate)
        percentile_score = self._calc_percentile_score(statistics.percentile)
        zscore_score = self._calc_zscore_score(statistics.zscore)

        pressure_score = self._calc_pressure_score(
            magnitude_score=magnitude_score,
            percentile_score=percentile_score,
            zscore_score=zscore_score,
            oi_confirmation=oi_confirmation,
            price_stall_confirmation=price_stall_confirmation,
        )

        direction = self._detect_pressure_direction(
            funding_rate=snapshot.funding_rate,
            bias=regime_state.bias,
        )

        level = self._detect_pressure_level(pressure_score)

        squeeze_probability = self._estimate_squeeze_probability(
            pressure_score=pressure_score,
            percentile=statistics.percentile,
            oi_confirmation=oi_confirmation,
            price_stall_confirmation=price_stall_confirmation,
        )

        mean_reversion_probability = self._estimate_mean_reversion_probability(
            pressure_score=pressure_score,
            percentile=statistics.percentile,
            zscore=statistics.zscore,
            price_stall_confirmation=price_stall_confirmation,
        )

        metadata: dict[str, Any] = {
            "scope": funding_key_to_dict(snapshot.key),
            "exchange_symbol": snapshot.exchange_symbol,
            "regime": regime_state.regime.value,
            "regime_confidence": regime_state.confidence,
            "regime_bias": regime_state.bias.value,
            "percentile": statistics.percentile,
            "zscore": statistics.zscore,
            "sample_size": statistics.sample_size,
            "oi_change_pct": oi_change_pct,
            "price_change_pct": price_change_pct,
            "magnitude_score": magnitude_score,
            "percentile_score": percentile_score,
            "zscore_score": zscore_score,
            "current_open_interest": snapshot.open_interest,
            "previous_open_interest": previous_open_interest,
            "current_price": (
                current_price
                if current_price is not None
                else snapshot.mark_price
            ),
            "previous_price": (
                previous_price
                if previous_price is not None
                else (
                    previous_snapshot.mark_price
                    if previous_snapshot is not None
                    else None
                )
            ),
        }

        if previous_snapshot is not None:
            metadata["previous_funding_rate"] = previous_snapshot.funding_rate
            metadata["funding_rate_delta"] = (
                snapshot.funding_rate - previous_snapshot.funding_rate
            )

        if extra_metadata:
            metadata.update(extra_metadata)

        state = FundingPressureState(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            market_type=snapshot.market_type,
            timeframe=tf,
            exchange_symbol=snapshot.exchange_symbol,
            direction=direction,
            level=level,
            bias=self._resolve_pressure_bias(regime_state, direction, level),
            funding_rate=snapshot.funding_rate,
            pressure_score=pressure_score,
            oi_confirmation=oi_confirmation,
            price_stall_confirmation=price_stall_confirmation,
            squeeze_probability=squeeze_probability,
            mean_reversion_probability=mean_reversion_probability,
            event_time=snapshot.event_time,
            metadata=metadata,
        )

        self.logger.debug(
            "Funding pressure analyzed | exchange=%s market_type=%s symbol=%s "
            "timeframe=%s level=%s direction=%s score=%.4f squeeze_prob=%.4f "
            "mean_reversion_prob=%.4f oi_confirmation=%s price_stall=%s",
            state.exchange.value,
            state.market_type,
            state.symbol,
            state.timeframe.value,
            state.level.value,
            state.direction.value,
            state.pressure_score,
            state.squeeze_probability if state.squeeze_probability is not None else 0.0,
            (
                state.mean_reversion_probability
                if state.mean_reversion_probability is not None
                else 0.0
            ),
            state.oi_confirmation,
            state.price_stall_confirmation,
            extra={
                "scope": funding_key_to_dict(state.key),
                "exchange_symbol": state.exchange_symbol,
            },
        )

        return state

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def _detect_pressure_direction(
        self,
        funding_rate: float,
        bias: FundingBias,
    ) -> FundingPressureDirection:
        """
        Визначає напрям перекосу позиціонування.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_pressure_direction", _analytics_args)
        except Exception:
            pass
        if bias in {
            FundingBias.LONG_BIAS,
            FundingBias.OVERCROWDED_LONGS,
            FundingBias.SQUEEZE_RISK_LONGS,
        }:
            return FundingPressureDirection.LONG

        if bias in {
            FundingBias.SHORT_BIAS,
            FundingBias.OVERCROWDED_SHORTS,
            FundingBias.SQUEEZE_RISK_SHORTS,
        }:
            return FundingPressureDirection.SHORT

        if funding_rate > 0:
            return FundingPressureDirection.LONG

        if funding_rate < 0:
            return FundingPressureDirection.SHORT

        return FundingPressureDirection.NEUTRAL

    def _detect_pressure_level(
        self,
        pressure_score: float,
    ) -> FundingPressureLevel:
        """
        Визначає рівень funding pressure.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_pressure_level", _analytics_args)
        except Exception:
            pass
        pressure_score = self._clamp_0_1(pressure_score)

        if pressure_score >= self.config.extreme_pressure_score_threshold:
            return FundingPressureLevel.EXTREME

        if pressure_score >= self.config.high_pressure_score_threshold:
            return FundingPressureLevel.HIGH

        if pressure_score >= self.config.moderate_pressure_score_threshold:
            return FundingPressureLevel.MODERATE

        return FundingPressureLevel.LOW

    def _resolve_pressure_bias(
        self,
        regime_state: FundingRegimeState,
        direction: FundingPressureDirection,
        level: FundingPressureLevel,
    ) -> FundingBias:
        """
        Остаточна інтерпретація bias для pressure state.

        Якщо regime вже дав сильний bias — зберігаємо його.
        Якщо regime neutral, але pressure direction є, підтягуємо bias із direction.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_pressure_bias", _analytics_args)
        except Exception:
            pass
        if regime_state.bias != FundingBias.NEUTRAL:
            return regime_state.bias

        if direction == FundingPressureDirection.LONG:
            if level in {FundingPressureLevel.HIGH, FundingPressureLevel.EXTREME}:
                return FundingBias.OVERCROWDED_LONGS
            return FundingBias.LONG_BIAS

        if direction == FundingPressureDirection.SHORT:
            if level in {FundingPressureLevel.HIGH, FundingPressureLevel.EXTREME}:
                return FundingBias.OVERCROWDED_SHORTS
            return FundingBias.SHORT_BIAS

        return FundingBias.NEUTRAL

    # ------------------------------------------------------------------
    # OI / Price context
    # ------------------------------------------------------------------

    def _detect_oi_confirmation(
        self,
        current_open_interest: float | None,
        previous_open_interest: float | None,
    ) -> tuple[bool, float | None]:
        """
        Funding pressure сильніший, якщо OI зростає разом із перекосом.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_oi_confirmation", _analytics_args)
        except Exception:
            pass
        if (
            current_open_interest is None
            or previous_open_interest is None
            or previous_open_interest <= 0
        ):
            return False, None

        oi_change_pct = (
            (float(current_open_interest) - float(previous_open_interest))
            / float(previous_open_interest)
        )

        return oi_change_pct >= self.config.oi_growth_threshold_pct, oi_change_pct

    def _detect_price_stall(
        self,
        current_price: float | None,
        previous_price: float | None,
    ) -> tuple[bool, float | None]:
        """
        Якщо funding/OI перегріваються, а ціна майже не рухається,
        це часто ознака накопичення pressure.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_price_stall", _analytics_args)
        except Exception:
            pass
        if current_price is None or previous_price is None or previous_price <= 0:
            return False, None

        price_change_pct = abs(
            (float(current_price) - float(previous_price))
            / float(previous_price)
        )
        is_stalled = price_change_pct <= self.config.price_stall_threshold_pct

        return is_stalled, price_change_pct

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    def _calc_pressure_score(
        self,
        magnitude_score: float,
        percentile_score: float,
        zscore_score: float,
        oi_confirmation: bool,
        price_stall_confirmation: bool,
    ) -> float:
        """
        Головний агрегований pressure score в діапазоні [0, 1].
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_pressure_score", _analytics_args)
        except Exception:
            pass
        score = (
            self.config.weight_magnitude * self._clamp_0_1(magnitude_score)
            + self.config.weight_percentile * self._clamp_0_1(percentile_score)
            + self.config.weight_zscore * self._clamp_0_1(zscore_score)
            + self.config.weight_oi_confirmation * float(oi_confirmation)
            + self.config.weight_price_stall * float(price_stall_confirmation)
        )

        return self._clamp_0_1(score)

    def _calc_magnitude_score(
        self,
        funding_rate: float,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_magnitude_score", _analytics_args)
        except Exception:
            pass
        abs_rate = abs(float(funding_rate))

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_abs_threshold - self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = (abs_rate - self.config.neutral_abs_threshold) / denominator
        return self._clamp_0_1(normalized)

    def _calc_percentile_score(
        self,
        percentile: float | None,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_percentile_score", _analytics_args)
        except Exception:
            pass
        if percentile is None:
            return 0.0

        normalized_percentile = self._clamp(float(percentile), 0.0, 100.0)
        distance_from_center = abs(normalized_percentile - 50.0) / 50.0
        return self._clamp_0_1(distance_from_center)

    def _calc_zscore_score(
        self,
        zscore: float | None,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calc_zscore_score", _analytics_args)
        except Exception:
            pass
        if zscore is None:
            return 0.0

        denominator = max(self.config.extreme_zscore_threshold, 1e-12)
        normalized = abs(float(zscore)) / denominator
        return self._clamp_0_1(normalized)

    # ------------------------------------------------------------------
    # Probability estimates
    # ------------------------------------------------------------------

    def _estimate_squeeze_probability(
        self,
        pressure_score: float,
        percentile: float | None,
        oi_confirmation: bool,
        price_stall_confirmation: bool,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_squeeze_probability", _analytics_args)
        except Exception:
            pass
        percentile_component = self._calc_percentile_score(percentile)

        probability = (
            self.config.squeeze_pressure_weight * self._clamp_0_1(pressure_score)
            + self.config.squeeze_percentile_weight * percentile_component
            + self.config.squeeze_oi_weight * float(oi_confirmation)
            + self.config.squeeze_stall_weight * float(price_stall_confirmation)
        )

        return self._clamp_0_1(probability)

    def _estimate_mean_reversion_probability(
        self,
        pressure_score: float,
        percentile: float | None,
        zscore: float | None,
        price_stall_confirmation: bool,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_mean_reversion_probability", _analytics_args)
        except Exception:
            pass
        percentile_component = self._calc_percentile_score(percentile)
        zscore_component = self._calc_zscore_score(zscore)

        probability = (
            self.config.mean_reversion_pressure_weight * self._clamp_0_1(pressure_score)
            + self.config.mean_reversion_percentile_weight * percentile_component
            + self.config.mean_reversion_zscore_weight * zscore_component
            + self.config.mean_reversion_stall_weight * float(price_stall_confirmation)
        )

        return self._clamp_0_1(probability)

    # ------------------------------------------------------------------
    # Optional helper methods for analyzer / strategies
    # ------------------------------------------------------------------

    def is_high_pressure(
        self,
        state: FundingPressureState,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_high_pressure", _analytics_args)
        except Exception:
            pass
        return state.level in {
            FundingPressureLevel.HIGH,
            FundingPressureLevel.EXTREME,
        }

    def is_long_crowded(
        self,
        state: FundingPressureState,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_long_crowded", _analytics_args)
        except Exception:
            pass
        return (
            state.direction == FundingPressureDirection.LONG
            and self.is_high_pressure(state)
        )

    def is_short_crowded(
        self,
        state: FundingPressureState,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_short_crowded", _analytics_args)
        except Exception:
            pass
        return (
            state.direction == FundingPressureDirection.SHORT
            and self.is_high_pressure(state)
        )

    def is_squeeze_risk(
        self,
        state: FundingPressureState,
        threshold: float = 0.65,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_squeeze_risk", _analytics_args)
        except Exception:
            pass
        if state.squeeze_probability is None:
            return False
        return state.squeeze_probability >= threshold

    def is_mean_reversion_risk(
        self,
        state: FundingPressureState,
        threshold: float = 0.60,
    ) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_mean_reversion_risk", _analytics_args)
        except Exception:
            pass
        if state.mean_reversion_probability is None:
            return False
        return state.mean_reversion_probability >= threshold

    def build_summary(
        self,
        state: FundingPressureState,
    ) -> str:
        """
        Короткий summary для signal layer / dashboard / logs.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_summary", _analytics_args)
        except Exception:
            pass
        squeeze_probability = (
            state.squeeze_probability
            if state.squeeze_probability is not None
            else 0.0
        )
        mean_reversion_probability = (
            state.mean_reversion_probability
            if state.mean_reversion_probability is not None
            else 0.0
        )

        return (
            f"Funding pressure for "
            f"{state.exchange.value}:{state.market_type}:{state.symbol}:{state.timeframe.value}: "
            f"level={state.level.value}, "
            f"direction={state.direction.value}, "
            f"score={state.pressure_score:.4f}, "
            f"squeeze_probability={squeeze_probability:.4f}, "
            f"mean_reversion_probability={mean_reversion_probability:.4f}"
        )

    # ------------------------------------------------------------------
    # Validation / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input_scope(
        *,
        snapshot: FundingSnapshot,
        statistics: FundingStatistics,
        regime_state: FundingRegimeState,
        previous_snapshot: FundingSnapshot | None,
    ) -> None:
        """
        Funding pressure не можна рахувати на змішаних scope-ах.

        Всі основні моделі мають належати одному:
            exchange + market_type + symbol + timeframe
        """
        try:
            _analytics_class_name = "FundingPressureAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_input_scope", _analytics_args)
        except Exception:
            pass
        if snapshot.key != statistics.key:
            raise ValueError(
                "FundingSnapshot and FundingStatistics scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"statistics={funding_key_to_dict(statistics.key)}"
            )

        if snapshot.key != regime_state.key:
            raise ValueError(
                "FundingSnapshot and FundingRegimeState scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"regime_state={funding_key_to_dict(regime_state.key)}"
            )

        if previous_snapshot is not None and snapshot.key != previous_snapshot.key:
            raise ValueError(
                "FundingSnapshot and previous_snapshot scope mismatch: "
                f"snapshot={funding_key_to_dict(snapshot.key)} "
                f"previous_snapshot={funding_key_to_dict(previous_snapshot.key)}"
            )

    @staticmethod
    def _clamp_0_1(value: float) -> float:
        try:
            _analytics_class_name = "FundingPressureAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_clamp_0_1", _analytics_args)
        except Exception:
            pass
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        try:
            _analytics_class_name = "FundingPressureAnalyzer"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_clamp", _analytics_args)
        except Exception:
            pass
        return max(lower, min(upper, float(value)))


__all__ = [
    "FundingPressureConfig",
    "FundingPressureAnalyzer",
]