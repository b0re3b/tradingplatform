from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable

from .config import BaseSpreadConfig, CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .enums import SpreadRegime, SpreadSignalType, SpreadType
from .models import ArbitrageOpportunity, SpreadSignal, SpreadSnapshot
from .spread_regime_detector import RegimeShiftResult, SpreadRegimeDetector


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")


# ============================================================
# Signal reason codes
# ============================================================

class SignalBuildReason(str, Enum):
    """
    Stable reason codes for signal engine decisions.

    Ці reason codes можна використовувати в metadata, dashboard,
    storage або analyzer metrics.
    """

    SIGNAL_BUILT = "signal_built"

    MISSING_SPREAD_BPS = "missing_spread_bps"
    BELOW_WIDENING_THRESHOLD = "below_widening_threshold"

    MISSING_STATS = "missing_stats"
    MISSING_ZSCORE = "missing_zscore"
    BELOW_ANOMALY_THRESHOLD = "below_anomaly_threshold"

    UNSUPPORTED_SPREAD_TYPE = "unsupported_spread_type"
    INCOMPATIBLE_CONFIG = "incompatible_config"
    MISSING_FUNDING_ADJUSTED_SPREAD = "missing_funding_adjusted_spread"
    BELOW_MEAN_REVERSION_THRESHOLD = "below_mean_reversion_threshold"

    MISSING_PREVIOUS_SNAPSHOT = "missing_previous_snapshot"
    NO_REGIME_SHIFT = "no_regime_shift"
    BELOW_REGIME_SHIFT_THRESHOLD = "below_regime_shift_threshold"

    MISSING_OPPORTUNITY = "missing_opportunity"
    OPPORTUNITY_NOT_PROFITABLE = "opportunity_not_profitable"
    BELOW_ARBITRAGE_THRESHOLD = "below_arbitrage_threshold"


# ============================================================
# Result models
# ============================================================

@dataclass(slots=True)
class SignalBuildResult:
    """
    Результат побудови одного сигналу.
    """

    signal: SpreadSignal | None
    reason: SignalBuildReason | str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def built(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "built", _analytics_args)
        except Exception:
            pass
        return self.signal is not None

    @property
    def reason_value(self) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reason_value", _analytics_args)
        except Exception:
            pass
        if isinstance(self.reason, SignalBuildReason):
            return self.reason.value
        return str(self.reason)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        signal_payload = None
        if self.signal is not None:
            to_payload = getattr(self.signal, "to_payload", None)
            signal_payload = to_payload() if callable(to_payload) else self.signal

        return {
            "built": self.built,
            "reason": self.reason_value,
            "signal": signal_payload,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SignalEngineResult:
    """
    Результат оцінки одного SpreadSnapshot.
    """

    signals: list[SpreadSignal]
    build_results: list[SignalBuildResult] = field(default_factory=list)

    @property
    def has_signals(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "has_signals", _analytics_args)
        except Exception:
            pass
        return bool(self.signals)

    @property
    def signal_count(self) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "signal_count", _analytics_args)
        except Exception:
            pass
        return len(self.signals)

    @property
    def reasons(self) -> list[str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reasons", _analytics_args)
        except Exception:
            pass
        return [result.reason_value for result in self.build_results]

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        return {
            "has_signals": self.has_signals,
            "signal_count": self.signal_count,
            "reasons": self.reasons,
            "signals": [
                signal.to_payload() if hasattr(signal, "to_payload") else signal
                for signal in self.signals
            ],
            "build_results": [
                result.to_payload()
                for result in self.build_results
            ],
        }


# ============================================================
# Engine
# ============================================================

class SpreadSignalEngine:
    """
    Pure domain engine для побудови spread-сигналів.

    Відповідальність:
    - widening signal;
    - anomaly signal;
    - mean reversion signal;
    - regime shift signal;
    - arbitrage signal;
    - stale/invalid data signal helpers.

    Не відповідає за:
    - EventBus publish;
    - Scheduler jobs;
    - logging;
    - storage;
    - cooldown/throttling;
    - lifecycle analyzer-а.
    """

    def __init__(
        self,
        config: BaseSpreadConfig,
        regime_detector: SpreadRegimeDetector | None = None,
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
        self._config = config
        self._regime_detector = regime_detector or SpreadRegimeDetector(config)

    # ------------------------------------------------------------------
    # Main evaluation API
    # ------------------------------------------------------------------

    def evaluate_snapshot(
        self,
        snapshot: SpreadSnapshot,
        previous_snapshot: SpreadSnapshot | None = None,
        opportunity: ArbitrageOpportunity | None = None,
    ) -> SignalEngineResult:
        """
        Оцінює snapshot і повертає всі побудовані сигнали.

        Порядок сигналів стабільний:
        1. widening
        2. anomaly
        3. mean reversion
        4. regime shift
        5. arbitrage
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "evaluate_snapshot", _analytics_args)
        except Exception:
            pass
        build_results = [
            self.build_widening_signal_result(snapshot),
            self.build_anomaly_signal_result(snapshot),
            self.build_mean_reversion_signal_result(snapshot),
            self.build_regime_shift_signal_result(
                previous_snapshot=previous_snapshot,
                current_snapshot=snapshot,
            ),
            self.build_arbitrage_signal_result(
                snapshot=snapshot,
                opportunity=opportunity,
            ),
        ]

        signals = [
            result.signal
            for result in build_results
            if result.signal is not None
        ]

        return SignalEngineResult(
            signals=signals,
            build_results=build_results,
        )

    # ------------------------------------------------------------------
    # Backward-compatible signal builders
    # ------------------------------------------------------------------

    def build_widening_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_widening_signal", _analytics_args)
        except Exception:
            pass
        return self.build_widening_signal_result(snapshot).signal

    def build_anomaly_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_anomaly_signal", _analytics_args)
        except Exception:
            pass
        return self.build_anomaly_signal_result(snapshot).signal

    def build_mean_reversion_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_mean_reversion_signal", _analytics_args)
        except Exception:
            pass
        return self.build_mean_reversion_signal_result(snapshot).signal

    def build_regime_shift_signal(
        self,
        previous_snapshot: SpreadSnapshot | None,
        current_snapshot: SpreadSnapshot,
    ) -> SpreadSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_regime_shift_signal", _analytics_args)
        except Exception:
            pass
        return self.build_regime_shift_signal_result(
            previous_snapshot=previous_snapshot,
            current_snapshot=current_snapshot,
        ).signal

    def build_arbitrage_signal(
        self,
        snapshot: SpreadSnapshot,
        opportunity: ArbitrageOpportunity | None,
    ) -> SpreadSignal | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_arbitrage_signal", _analytics_args)
        except Exception:
            pass
        return self.build_arbitrage_signal_result(
            snapshot=snapshot,
            opportunity=opportunity,
        ).signal

    # ------------------------------------------------------------------
    # Detailed signal builders
    # ------------------------------------------------------------------

    def build_widening_signal_result(
        self,
        snapshot: SpreadSnapshot,
    ) -> SignalBuildResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_widening_signal_result", _analytics_args)
        except Exception:
            pass
        if snapshot.spread_bps is None:
            return self._not_built(
                SignalBuildReason.MISSING_SPREAD_BPS,
                snapshot=snapshot,
            )

        threshold = self._config.widening_bps_threshold
        abs_spread_bps = abs(snapshot.spread_bps)

        if abs_spread_bps < threshold:
            return self._not_built(
                SignalBuildReason.BELOW_WIDENING_THRESHOLD,
                snapshot=snapshot,
                metadata={
                    "spread_bps": str(snapshot.spread_bps),
                    "abs_spread_bps": str(abs_spread_bps),
                    "threshold": str(threshold),
                },
            )

        signal = SpreadSignal(
            signal_type=SpreadSignalType.WIDENING,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Spread widened to {snapshot.spread_bps} bps "
                f"between {snapshot.leg_a_exchange} and {snapshot.leg_b_exchange}"
            ),
            value=snapshot.spread_bps,
            threshold=threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                **self._snapshot_metadata(snapshot),
                "reason": SignalBuildReason.SIGNAL_BUILT.value,
                "abs_spread_bps": str(abs_spread_bps),
            },
        )

        return self._built(signal)

    def build_anomaly_signal_result(
        self,
        snapshot: SpreadSnapshot,
    ) -> SignalBuildResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_anomaly_signal_result", _analytics_args)
        except Exception:
            pass
        if snapshot.stats is None:
            return self._not_built(
                SignalBuildReason.MISSING_STATS,
                snapshot=snapshot,
            )

        zscore = snapshot.stats.zscore
        if zscore is None:
            return self._not_built(
                SignalBuildReason.MISSING_ZSCORE,
                snapshot=snapshot,
            )

        threshold = self._config.anomaly_zscore_threshold
        abs_zscore = abs(zscore)

        if abs_zscore < threshold:
            return self._not_built(
                SignalBuildReason.BELOW_ANOMALY_THRESHOLD,
                snapshot=snapshot,
                metadata={
                    "zscore": str(zscore),
                    "abs_zscore": str(abs_zscore),
                    "threshold": str(threshold),
                },
            )

        signal = SpreadSignal(
            signal_type=SpreadSignalType.ANOMALY,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Spread anomaly detected for {snapshot.symbol}: "
                f"z-score={zscore} on "
                f"{snapshot.leg_a_exchange}/{snapshot.leg_b_exchange}"
            ),
            value=zscore,
            threshold=threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                **self._snapshot_metadata(snapshot),
                "reason": SignalBuildReason.SIGNAL_BUILT.value,
                "abs_zscore": str(abs_zscore),
            },
        )

        return self._built(signal)

    def build_mean_reversion_signal_result(
        self,
        snapshot: SpreadSnapshot,
    ) -> SignalBuildResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_mean_reversion_signal_result", _analytics_args)
        except Exception:
            pass
        if snapshot.spread_type != SpreadType.SPOT_FUTURES:
            return self._not_built(
                SignalBuildReason.UNSUPPORTED_SPREAD_TYPE,
                snapshot=snapshot,
                metadata={
                    "expected_spread_type": SpreadType.SPOT_FUTURES.value,
                    "actual_spread_type": snapshot.spread_type.value,
                },
            )

        if not isinstance(self._config, SpotFuturesSpreadConfig):
            return self._not_built(
                SignalBuildReason.INCOMPATIBLE_CONFIG,
                snapshot=snapshot,
                metadata={
                    "expected_config": "SpotFuturesSpreadConfig",
                    "actual_config": self._config.__class__.__name__,
                },
            )

        if snapshot.stats is None:
            return self._not_built(
                SignalBuildReason.MISSING_STATS,
                snapshot=snapshot,
            )

        zscore = snapshot.stats.zscore
        if zscore is None:
            return self._not_built(
                SignalBuildReason.MISSING_ZSCORE,
                snapshot=snapshot,
            )

        threshold = self._config.mean_reversion_zscore_threshold
        abs_zscore = abs(zscore)

        if abs_zscore < threshold:
            return self._not_built(
                SignalBuildReason.BELOW_MEAN_REVERSION_THRESHOLD,
                snapshot=snapshot,
                metadata={
                    "zscore": str(zscore),
                    "abs_zscore": str(abs_zscore),
                    "threshold": str(threshold),
                },
            )

        if snapshot.funding_adjusted_spread is None:
            return self._not_built(
                SignalBuildReason.MISSING_FUNDING_ADJUSTED_SPREAD,
                snapshot=snapshot,
            )

        signal = SpreadSignal(
            signal_type=SpreadSignalType.MEAN_REVERSION,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Mean reversion candidate detected for {snapshot.symbol}: "
                f"z-score={zscore}, "
                f"funding-adjusted spread={snapshot.funding_adjusted_spread}"
            ),
            value=zscore,
            threshold=threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                **self._snapshot_metadata(snapshot),
                "reason": SignalBuildReason.SIGNAL_BUILT.value,
                "funding_adjusted_spread": self._to_str(snapshot.funding_adjusted_spread),
                "basis": self._to_str(snapshot.basis),
                "abs_zscore": str(abs_zscore),
            },
        )

        return self._built(signal)

    def build_regime_shift_signal_result(
        self,
        previous_snapshot: SpreadSnapshot | None,
        current_snapshot: SpreadSnapshot,
    ) -> SignalBuildResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_regime_shift_signal_result", _analytics_args)
        except Exception:
            pass
        if previous_snapshot is None:
            return self._not_built(
                SignalBuildReason.MISSING_PREVIOUS_SNAPSHOT,
                snapshot=current_snapshot,
            )

        shift = self._regime_detector.detect_shift(
            previous_snapshot,
            current_snapshot,
        )
        if not shift.changed:
            return self._not_built(
                SignalBuildReason.NO_REGIME_SHIFT,
                snapshot=current_snapshot,
                metadata=self._shift_metadata(shift),
            )

        threshold: Decimal | None = None
        current_zscore = (
            current_snapshot.stats.zscore
            if current_snapshot.stats is not None
            else None
        )

        if isinstance(self._config, SpotFuturesSpreadConfig):
            threshold = self._config.regime_shift_zscore_threshold

            if current_zscore is None:
                return self._not_built(
                    SignalBuildReason.MISSING_ZSCORE,
                    snapshot=current_snapshot,
                    metadata=self._shift_metadata(shift),
                )

            if abs(current_zscore) < threshold:
                return self._not_built(
                    SignalBuildReason.BELOW_REGIME_SHIFT_THRESHOLD,
                    snapshot=current_snapshot,
                    metadata={
                        **self._shift_metadata(shift),
                        "current_zscore": str(current_zscore),
                        "abs_current_zscore": str(abs(current_zscore)),
                        "threshold": str(threshold),
                    },
                )

        previous_regime = shift.previous_regime
        current_regime = shift.current_regime

        previous_regime_value = previous_regime.value if previous_regime else "unknown"
        current_regime_value = current_regime.value if current_regime else "unknown"

        signal = SpreadSignal(
            signal_type=SpreadSignalType.REGIME_SHIFT,
            spread_type=current_snapshot.spread_type,
            symbol=current_snapshot.symbol,
            message=(
                f"Spread regime shifted from "
                f"{previous_regime_value} to {current_regime_value} "
                f"for {current_snapshot.symbol}"
            ),
            value=shift.current_zscore,
            threshold=threshold,
            confidence=self._confidence_from_regime_shift(shift),
            exchange_a=current_snapshot.leg_a_exchange,
            exchange_b=current_snapshot.leg_b_exchange,
            metadata={
                **self._snapshot_metadata(current_snapshot),
                **self._shift_metadata(shift),
                "reason": SignalBuildReason.SIGNAL_BUILT.value,
            },
        )

        return self._built(signal)

    def build_arbitrage_signal_result(
        self,
        snapshot: SpreadSnapshot,
        opportunity: ArbitrageOpportunity | None,
    ) -> SignalBuildResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_arbitrage_signal_result", _analytics_args)
        except Exception:
            pass
        if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
            return self._not_built(
                SignalBuildReason.UNSUPPORTED_SPREAD_TYPE,
                snapshot=snapshot,
                metadata={
                    "expected_spread_type": SpreadType.CROSS_EXCHANGE.value,
                    "actual_spread_type": snapshot.spread_type.value,
                },
            )

        if opportunity is None:
            return self._not_built(
                SignalBuildReason.MISSING_OPPORTUNITY,
                snapshot=snapshot,
            )

        if not isinstance(self._config, CrossExchangeSpreadConfig):
            return self._not_built(
                SignalBuildReason.INCOMPATIBLE_CONFIG,
                snapshot=snapshot,
                metadata={
                    "expected_config": "CrossExchangeSpreadConfig",
                    "actual_config": self._config.__class__.__name__,
                },
            )

        if not opportunity.is_profitable:
            return self._not_built(
                SignalBuildReason.OPPORTUNITY_NOT_PROFITABLE,
                snapshot=snapshot,
                metadata={
                    "net_edge": self._to_str(opportunity.net_edge),
                    "gross_edge": self._to_str(opportunity.gross_edge),
                },
            )

        threshold = self._config.arbitrage_min_bps
        opportunity_spread_bps = opportunity.spread_bps
        spread_bps_value = opportunity_spread_bps if opportunity_spread_bps is not None else snapshot.spread_bps

        if spread_bps_value is not None and spread_bps_value < threshold:
            return self._not_built(
                SignalBuildReason.BELOW_ARBITRAGE_THRESHOLD,
                snapshot=snapshot,
                metadata={
                    "spread_bps": str(spread_bps_value),
                    "threshold": str(threshold),
                },
            )

        signal = SpreadSignal(
            signal_type=SpreadSignalType.ARBITRAGE,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Arbitrage opportunity detected for {snapshot.symbol}: "
                f"buy on {opportunity.buy_exchange}, "
                f"sell on {opportunity.sell_exchange}, "
                f"net edge={opportunity.net_edge}"
            ),
            value=opportunity.net_edge,
            threshold=threshold,
            confidence=self._confidence_from_opportunity(opportunity),
            exchange_a=opportunity.buy_exchange,
            exchange_b=opportunity.sell_exchange,
            metadata={
                **self._snapshot_metadata(snapshot),
                **self._opportunity_metadata(opportunity),
                "reason": SignalBuildReason.SIGNAL_BUILT.value,
            },
        )

        return self._built(signal)

    # ------------------------------------------------------------------
    # Data-quality signal helpers
    # ------------------------------------------------------------------

    def build_stale_data_signal(
        self,
        spread_type: SpreadType,
        symbol: str,
        exchange_a: str | None,
        exchange_b: str | None,
        message: str = "Stale spread data detected",
        value: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpreadSignal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_stale_data_signal", _analytics_args)
        except Exception:
            pass
        return SpreadSignal(
            signal_type=SpreadSignalType.STALE_DATA,
            spread_type=spread_type,
            symbol=symbol,
            message=message,
            value=value,
            threshold=None,
            confidence=Decimal("0.30"),
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            metadata={
                "reason": "stale_data",
                **dict(metadata or {}),
            },
        )

    def build_invalid_data_signal(
        self,
        spread_type: SpreadType,
        symbol: str,
        exchange_a: str | None,
        exchange_b: str | None,
        message: str = "Invalid spread data detected",
        value: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpreadSignal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_invalid_data_signal", _analytics_args)
        except Exception:
            pass
        return SpreadSignal(
            signal_type=SpreadSignalType.INVALID_DATA,
            spread_type=spread_type,
            symbol=symbol,
            message=message,
            value=value,
            threshold=None,
            confidence=Decimal("0.20"),
            exchange_a=exchange_a,
            exchange_b=exchange_b,
            metadata={
                "reason": "invalid_data",
                **dict(metadata or {}),
            },
        )

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def filter_actionable_signals(
        self,
        signals: Iterable[SpreadSignal],
    ) -> list[SpreadSignal]:
        """
        Повертає тільки actionable spread-сигнали.

        Якщо enum має property is_actionable — використовує її.
        Інакше fallback через explicit set.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "filter_actionable_signals", _analytics_args)
        except Exception:
            pass
        actionable: list[SpreadSignal] = []

        for signal in signals:
            is_actionable = getattr(signal.signal_type, "is_actionable", None)
            if isinstance(is_actionable, bool):
                if is_actionable:
                    actionable.append(signal)
                continue

            if signal.signal_type in {
                SpreadSignalType.WIDENING,
                SpreadSignalType.NARROWING,
                SpreadSignalType.ANOMALY,
                SpreadSignalType.ARBITRAGE,
                SpreadSignalType.MEAN_REVERSION,
                SpreadSignalType.REGIME_SHIFT,
            }:
                actionable.append(signal)

        return actionable

    # ------------------------------------------------------------------
    # Confidence helpers
    # ------------------------------------------------------------------

    def _confidence_from_snapshot(self, snapshot: SpreadSnapshot) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_confidence_from_snapshot", _analytics_args)
        except Exception:
            pass
        zscore = snapshot.stats.zscore if snapshot.stats is not None else None
        if zscore is None:
            return Decimal("0.50")

        abs_zscore = abs(zscore)

        if abs_zscore >= Decimal("5"):
            return Decimal("0.95")
        if abs_zscore >= Decimal("4"):
            return Decimal("0.90")
        if abs_zscore >= Decimal("3"):
            return Decimal("0.82")
        if abs_zscore >= Decimal("2"):
            return Decimal("0.72")
        if abs_zscore >= Decimal("1"):
            return Decimal("0.60")

        return Decimal("0.50")

    def _confidence_from_regime_shift(
        self,
        shift: RegimeShiftResult,
    ) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_confidence_from_regime_shift", _analytics_args)
        except Exception:
            pass
        zscore = shift.current_zscore
        if zscore is None:
            return Decimal("0.60")

        abs_zscore = abs(zscore)

        if getattr(shift, "is_high_risk_shift", False):
            if abs_zscore >= Decimal("5"):
                return Decimal("0.96")
            if abs_zscore >= Decimal("4"):
                return Decimal("0.92")
            if abs_zscore >= Decimal("3"):
                return Decimal("0.86")
            return Decimal("0.78")

        if abs_zscore >= Decimal("5"):
            return Decimal("0.94")
        if abs_zscore >= Decimal("4"):
            return Decimal("0.88")
        if abs_zscore >= Decimal("3"):
            return Decimal("0.80")

        return Decimal("0.68")

    def _confidence_from_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_confidence_from_opportunity", _analytics_args)
        except Exception:
            pass
        if opportunity.confidence is not None:
            return _clamp_confidence(opportunity.confidence)

        if opportunity.gross_edge <= DECIMAL_ZERO:
            return Decimal("0.20")

        if opportunity.net_edge <= DECIMAL_ZERO:
            return Decimal("0.30")

        total_costs = opportunity.estimated_fees + opportunity.estimated_slippage

        if total_costs <= DECIMAL_ZERO:
            return Decimal("0.90")

        coverage_ratio = opportunity.net_edge / total_costs

        if coverage_ratio >= Decimal("3"):
            return Decimal("0.92")
        if coverage_ratio >= Decimal("2"):
            return Decimal("0.84")
        if coverage_ratio >= DECIMAL_ONE:
            return Decimal("0.74")

        return Decimal("0.58")

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _snapshot_metadata(self, snapshot: SpreadSnapshot) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_snapshot_metadata", _analytics_args)
        except Exception:
            pass
        stats = snapshot.stats

        return {
            "spread_type": snapshot.spread_type.value,
            "symbol": snapshot.symbol,
            "leg_a_exchange": snapshot.leg_a_exchange,
            "leg_b_exchange": snapshot.leg_b_exchange,
            "leg_a_type": snapshot.leg_a_type.value,
            "leg_b_type": snapshot.leg_b_type.value,
            "pricing_source": snapshot.pricing_source.value,
            "raw_spread": self._to_str(snapshot.raw_spread),
            "net_spread": self._to_str(snapshot.net_spread),
            "spread_pct": self._to_str(snapshot.spread_pct),
            "spread_bps": self._to_str(snapshot.spread_bps),
            "basis": self._to_str(snapshot.basis),
            "funding_adjusted_spread": self._to_str(snapshot.funding_adjusted_spread),
            "direction": snapshot.direction.value,
            "regime": snapshot.regime.value,
            "quote_validity": snapshot.quote_validity.value,
            "timestamp": snapshot.timestamp.isoformat(),
            "zscore": self._to_str(stats.zscore) if stats is not None else None,
            "stats_count": stats.count if stats is not None else None,
        }

    def _shift_metadata(
        self,
        shift: RegimeShiftResult,
    ) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_shift_metadata", _analytics_args)
        except Exception:
            pass
        return {
            "previous_regime": shift.previous_regime.value if shift.previous_regime else None,
            "current_regime": shift.current_regime.value if shift.current_regime else None,
            "previous_zscore": self._to_str(shift.previous_zscore),
            "current_zscore": self._to_str(shift.current_zscore),
            "zscore_delta": self._to_str(shift.zscore_delta),
            "previous_rank": getattr(shift, "previous_rank", None),
            "current_rank": getattr(shift, "current_rank", None),
            "rank_delta": getattr(shift, "rank_delta", None),
            "shift_direction": (
                "up"
                if shift.is_shift_up
                else "down"
                if shift.is_shift_down
                else "flat"
            ),
            "is_high_risk_shift": getattr(shift, "is_high_risk_shift", False),
            "shift_reason": shift.reason,
        }

    def _opportunity_metadata(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_opportunity_metadata", _analytics_args)
        except Exception:
            pass
        return {
            "buy_exchange": opportunity.buy_exchange,
            "sell_exchange": opportunity.sell_exchange,
            "buy_instrument_type": opportunity.buy_instrument_type.value,
            "sell_instrument_type": opportunity.sell_instrument_type.value,
            "buy_price": self._to_str(opportunity.buy_price),
            "sell_price": self._to_str(opportunity.sell_price),
            "gross_edge": self._to_str(opportunity.gross_edge),
            "estimated_fees": self._to_str(opportunity.estimated_fees),
            "estimated_slippage": self._to_str(opportunity.estimated_slippage),
            "net_edge": self._to_str(opportunity.net_edge),
            "spread_bps": self._to_str(opportunity.spread_bps),
            "spread_pct": self._to_str(opportunity.spread_pct),
            "confidence": self._to_str(opportunity.confidence),
            "status": opportunity.status.value,
            "timestamp": opportunity.timestamp.isoformat(),
            "expires_at": opportunity.expires_at.isoformat()
            if opportunity.expires_at
            else None,
            "opportunity_metadata": dict(opportunity.metadata),
        }

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _built(signal: SpreadSignal) -> SignalBuildResult:
        try:
            _analytics_class_name = "SpreadSignalEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_built", _analytics_args)
        except Exception:
            pass
        return SignalBuildResult(
            signal=signal,
            reason=SignalBuildReason.SIGNAL_BUILT,
        )

    @staticmethod
    def _not_built(
        reason: SignalBuildReason,
        *,
        snapshot: SpreadSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SignalBuildResult:
        try:
            _analytics_class_name = "SpreadSignalEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_not_built", _analytics_args)
        except Exception:
            pass
        base_metadata: dict[str, Any] = {}

        if snapshot is not None:
            base_metadata = {
                "spread_type": snapshot.spread_type.value,
                "symbol": snapshot.symbol,
                "leg_a_exchange": snapshot.leg_a_exchange,
                "leg_b_exchange": snapshot.leg_b_exchange,
                "timestamp": snapshot.timestamp.isoformat(),
            }

        return SignalBuildResult(
            signal=None,
            reason=reason,
            metadata={
                **base_metadata,
                **dict(metadata or {}),
            },
        )

    @staticmethod
    def _to_str(value: Decimal | None) -> str | None:
        try:
            _analytics_class_name = "SpreadSignalEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_to_str", _analytics_args)
        except Exception:
            pass
        return str(value) if value is not None else None


# ============================================================
# Module helpers
# ============================================================

def _clamp_confidence(value: Decimal) -> Decimal:
    if value < DECIMAL_ZERO:
        return DECIMAL_ZERO
    if value > DECIMAL_ONE:
        return DECIMAL_ONE
    return value


__all__ = [
    "SignalBuildReason",
    "SignalBuildResult",
    "SignalEngineResult",
    "SpreadSignalEngine",
]