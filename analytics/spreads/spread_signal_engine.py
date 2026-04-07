from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from .config import BaseSpreadConfig, CrossExchangeSpreadConfig, SpotFuturesSpreadConfig
from .enums import SpreadSignalType, SpreadType
from .models import ArbitrageOpportunity, SpreadSignal, SpreadSnapshot
from .spread_regime_detector import SpreadRegimeDetector


DECIMAL_ZERO = Decimal("0")
DECIMAL_ONE = Decimal("1")
DECIMAL_100 = Decimal("100")


@dataclass(slots=True)
class SignalEngineResult:
    signals: list[SpreadSignal]

    @property
    def has_signals(self) -> bool:
        return bool(self.signals)


class SpreadSignalEngine:
    """
    Доменний engine для побудови spread-сигналів.

    Відповідальність:
    - widening signal
    - anomaly signal
    - mean reversion signal
    - regime shift signal
    - arbitrage signal

    Не відповідає за:
    - EventBus publish
    - cooldown / throttling
    - storage
    - logging
    """

    def __init__(
        self,
        config: BaseSpreadConfig,
        regime_detector: SpreadRegimeDetector | None = None,
    ) -> None:
        self._config = config
        self._regime_detector = regime_detector or SpreadRegimeDetector(config)

    def evaluate_snapshot(
        self,
        snapshot: SpreadSnapshot,
        previous_snapshot: SpreadSnapshot | None = None,
        opportunity: ArbitrageOpportunity | None = None,
    ) -> SignalEngineResult:
        signals: list[SpreadSignal] = []

        widening = self.build_widening_signal(snapshot)
        if widening is not None:
            signals.append(widening)

        anomaly = self.build_anomaly_signal(snapshot)
        if anomaly is not None:
            signals.append(anomaly)

        mean_reversion = self.build_mean_reversion_signal(snapshot)
        if mean_reversion is not None:
            signals.append(mean_reversion)

        regime_shift = self.build_regime_shift_signal(
            previous_snapshot=previous_snapshot,
            current_snapshot=snapshot,
        )
        if regime_shift is not None:
            signals.append(regime_shift)

        arbitrage = self.build_arbitrage_signal(snapshot, opportunity)
        if arbitrage is not None:
            signals.append(arbitrage)

        return SignalEngineResult(signals=signals)

    def build_widening_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
        if snapshot.spread_bps is None:
            return None

        threshold = self._config.widening_bps_threshold
        if abs(snapshot.spread_bps) < threshold:
            return None

        return SpreadSignal(
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
                "regime": snapshot.regime.value,
                "raw_spread": self._to_str(snapshot.raw_spread),
                "net_spread": self._to_str(snapshot.net_spread),
                "leg_a_type": snapshot.leg_a_type.value,
                "leg_b_type": snapshot.leg_b_type.value,
            },
        )

    def build_anomaly_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
        zscore = snapshot.stats.zscore if snapshot.stats is not None else None
        if zscore is None:
            return None

        threshold = self._config.anomaly_zscore_threshold
        if abs(zscore) < threshold:
            return None

        return SpreadSignal(
            signal_type=SpreadSignalType.ANOMALY,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Spread anomaly detected for {snapshot.symbol}: "
                f"z-score={zscore} on {snapshot.leg_a_exchange}/{snapshot.leg_b_exchange}"
            ),
            value=zscore,
            threshold=threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "spread_bps": self._to_str(snapshot.spread_bps),
                "regime": snapshot.regime.value,
                "raw_spread": self._to_str(snapshot.raw_spread),
            },
        )

    def build_mean_reversion_signal(self, snapshot: SpreadSnapshot) -> SpreadSignal | None:
        if snapshot.spread_type != SpreadType.SPOT_FUTURES:
            return None

        if not isinstance(self._config, SpotFuturesSpreadConfig):
            return None

        stats = snapshot.stats
        if stats is None or stats.zscore is None:
            return None

        threshold = self._config.mean_reversion_zscore_threshold
        if abs(stats.zscore) < threshold:
            return None

        if snapshot.funding_adjusted_spread is None:
            return None

        return SpreadSignal(
            signal_type=SpreadSignalType.MEAN_REVERSION,
            spread_type=snapshot.spread_type,
            symbol=snapshot.symbol,
            message=(
                f"Mean reversion candidate detected for {snapshot.symbol}: "
                f"z-score={stats.zscore}, funding-adjusted spread={snapshot.funding_adjusted_spread}"
            ),
            value=stats.zscore,
            threshold=threshold,
            confidence=self._confidence_from_snapshot(snapshot),
            exchange_a=snapshot.leg_a_exchange,
            exchange_b=snapshot.leg_b_exchange,
            metadata={
                "funding_adjusted_spread": self._to_str(snapshot.funding_adjusted_spread),
                "basis": self._to_str(snapshot.basis),
                "regime": snapshot.regime.value,
            },
        )

    def build_regime_shift_signal(
        self,
        previous_snapshot: SpreadSnapshot | None,
        current_snapshot: SpreadSnapshot,
    ) -> SpreadSignal | None:
        if previous_snapshot is None:
            return None

        shift = self._regime_detector.detect_shift(previous_snapshot, current_snapshot)
        if not shift.changed:
            return None

        threshold = None
        if isinstance(self._config, SpotFuturesSpreadConfig):
            threshold = self._config.regime_shift_zscore_threshold

            current_z = (
                current_snapshot.stats.zscore
                if current_snapshot.stats is not None
                else None
            )
            if current_z is None or abs(current_z) < threshold:
                return None

        return SpreadSignal(
            signal_type=SpreadSignalType.REGIME_SHIFT,
            spread_type=current_snapshot.spread_type,
            symbol=current_snapshot.symbol,
            message=(
                f"Spread regime shifted from "
                f"{shift.previous_regime.value} to {shift.current_regime.value} "
                f"for {current_snapshot.symbol}"
            ),
            value=shift.current_zscore,
            threshold=threshold,
            confidence=self._confidence_from_regime_shift(shift.current_zscore),
            exchange_a=current_snapshot.leg_a_exchange,
            exchange_b=current_snapshot.leg_b_exchange,
            metadata={
                "previous_regime": shift.previous_regime.value if shift.previous_regime else None,
                "current_regime": shift.current_regime.value if shift.current_regime else None,
                "previous_zscore": self._to_str(shift.previous_zscore),
                "current_zscore": self._to_str(shift.current_zscore),
                "zscore_delta": self._to_str(shift.zscore_delta),
                "shift_direction": (
                    "up" if shift.is_shift_up else "down" if shift.is_shift_down else "flat"
                ),
                "reason": shift.reason,
            },
        )

    def build_arbitrage_signal(
        self,
        snapshot: SpreadSnapshot,
        opportunity: ArbitrageOpportunity | None,
    ) -> SpreadSignal | None:
        if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
            return None

        if opportunity is None:
            return None

        if not isinstance(self._config, CrossExchangeSpreadConfig):
            return None

        if not opportunity.is_profitable:
            return None

        threshold = self._config.arbitrage_min_bps
        spread_bps = opportunity.spread_bps or snapshot.spread_bps

        if spread_bps is not None and spread_bps < threshold:
            return None

        return SpreadSignal(
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
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "buy_price": self._to_str(opportunity.buy_price),
                "sell_price": self._to_str(opportunity.sell_price),
                "gross_edge": self._to_str(opportunity.gross_edge),
                "estimated_fees": self._to_str(opportunity.estimated_fees),
                "estimated_slippage": self._to_str(opportunity.estimated_slippage),
                "net_edge": self._to_str(opportunity.net_edge),
                "spread_bps": self._to_str(opportunity.spread_bps),
                "spread_pct": self._to_str(opportunity.spread_pct),
                "status": opportunity.status.value,
            },
        )

    def build_stale_data_signal(
        self,
        spread_type: SpreadType,
        symbol: str,
        exchange_a: str | None,
        exchange_b: str | None,
        message: str = "Stale spread data detected",
        value: Decimal | None = None,
    ) -> SpreadSignal:
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
            metadata={},
        )

    def build_invalid_data_signal(
        self,
        spread_type: SpreadType,
        symbol: str,
        exchange_a: str | None,
        exchange_b: str | None,
        message: str = "Invalid spread data detected",
        value: Decimal | None = None,
    ) -> SpreadSignal:
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
            metadata={},
        )

    def _confidence_from_snapshot(self, snapshot: SpreadSnapshot) -> Decimal:
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

    def _confidence_from_regime_shift(self, zscore: Decimal | None) -> Decimal:
        if zscore is None:
            return Decimal("0.60")

        abs_zscore = abs(zscore)

        if abs_zscore >= Decimal("5"):
            return Decimal("0.94")
        if abs_zscore >= Decimal("4"):
            return Decimal("0.88")
        if abs_zscore >= Decimal("3"):
            return Decimal("0.80")

        return Decimal("0.68")

    def _confidence_from_opportunity(self, opportunity: ArbitrageOpportunity) -> Decimal:
        if opportunity.confidence is not None:
            return opportunity.confidence

        if opportunity.gross_edge <= DECIMAL_ZERO:
            return Decimal("0.20")

        total_costs = opportunity.estimated_fees + opportunity.estimated_slippage
        if total_costs <= DECIMAL_ZERO:
            return Decimal("0.90")

        coverage_ratio = opportunity.net_edge / total_costs if total_costs > DECIMAL_ZERO else DECIMAL_ONE

        if coverage_ratio >= Decimal("3"):
            return Decimal("0.92")
        if coverage_ratio >= Decimal("2"):
            return Decimal("0.84")
        if coverage_ratio >= Decimal("1"):
            return Decimal("0.74")

        return Decimal("0.58")

    @staticmethod
    def _to_str(value: Decimal | None) -> str | None:
        return str(value) if value is not None else None