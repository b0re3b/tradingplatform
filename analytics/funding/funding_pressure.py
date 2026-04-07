from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logger import get_logger

from .enums import (
    FundingBias,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingTimeframe,
)
from .models import (
    FundingPressureState,
    FundingRegimeState,
    FundingSnapshot,
    FundingStatistics,
)


@dataclass(slots=True)
class FundingPressureConfig:
    """
    Конфігурація аналізатора funding pressure.

    Pressure тут трактується як міра перекосу та скупчення позиціонування,
    яке може призводити до:
    - squeeze
    - crowding
    - pain move
    - mean reversion
    """

    default_timeframe: FundingTimeframe = FundingTimeframe.H1

    # Абсолютні пороги funding magnitude
    neutral_abs_threshold: float = 0.00001
    elevated_abs_threshold: float = 0.00008
    extreme_abs_threshold: float = 0.00030

    # Порогові значення рівнів pressure score
    moderate_pressure_score_threshold: float = 0.45
    high_pressure_score_threshold: float = 0.70
    extreme_pressure_score_threshold: float = 0.90

    # Додаткові пороги
    crowded_percentile_threshold: float = 85.0
    squeeze_percentile_threshold: float = 95.0
    elevated_zscore_threshold: float = 1.5
    extreme_zscore_threshold: float = 2.5

    # Price stall detection
    price_stall_threshold_pct: float = 0.0010

    # OI growth detection
    oi_growth_threshold_pct: float = 0.005

    # Ваги скорингу
    weight_magnitude: float = 0.30
    weight_percentile: float = 0.25
    weight_zscore: float = 0.15
    weight_oi_confirmation: float = 0.15
    weight_price_stall: float = 0.15

    # Ваги ймовірностей
    squeeze_pressure_weight: float = 0.50
    squeeze_percentile_weight: float = 0.20
    squeeze_oi_weight: float = 0.15
    squeeze_stall_weight: float = 0.15

    mean_reversion_pressure_weight: float = 0.45
    mean_reversion_percentile_weight: float = 0.20
    mean_reversion_zscore_weight: float = 0.20
    mean_reversion_stall_weight: float = 0.15


class FundingPressureAnalyzer:
    """
    Аналізатор funding pressure.

    Клас відповідає за побудову FundingPressureState на основі:
    - snapshot
    - statistics
    - regime state
    - OI context
    - price context

    Він не працює з EventBus і не зберігає історію —
    це pure analytics/service layer для FundingAnalyzer.
    """

    def __init__(
        self,
        config: FundingPressureConfig | None = None,
    ) -> None:
        self.config = config or FundingPressureConfig()
        self.logger = get_logger(__name__)

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

        Parameters
        ----------
        snapshot:
            Поточний funding snapshot.
        statistics:
            Статистика funding на вікні історії.
        regime_state:
            Поточний regime state.
        previous_snapshot:
            Попередній funding snapshot для додаткового контексту.
        previous_open_interest:
            Попереднє значення OI.
        current_price:
            Поточна ціна.
        previous_price:
            Попередня ціна.
        timeframe:
            Таймфрейм аналізу.
        extra_metadata:
            Додаткові службові поля metadata.

        Returns
        -------
        FundingPressureState
        """
        tf = timeframe or statistics.timeframe or self.config.default_timeframe

        oi_confirmation, oi_change_pct = self._detect_oi_confirmation(
            current_open_interest=snapshot.open_interest,
            previous_open_interest=previous_open_interest,
        )

        price_stall_confirmation, price_change_pct = self._detect_price_stall(
            current_price=current_price if current_price is not None else snapshot.mark_price,
            previous_price=previous_price if previous_price is not None else (
                previous_snapshot.mark_price if previous_snapshot else None
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
            "regime": regime_state.regime.value,
            "regime_confidence": regime_state.confidence,
            "percentile": statistics.percentile,
            "zscore": statistics.zscore,
            "sample_size": statistics.sample_size,
            "oi_change_pct": oi_change_pct,
            "price_change_pct": price_change_pct,
            "magnitude_score": magnitude_score,
            "percentile_score": percentile_score,
            "zscore_score": zscore_score,
        }

        if previous_snapshot is not None:
            metadata["previous_funding_rate"] = previous_snapshot.funding_rate
            metadata["funding_rate_delta"] = snapshot.funding_rate - previous_snapshot.funding_rate

        if extra_metadata:
            metadata.update(extra_metadata)

        state = FundingPressureState(
            symbol=snapshot.symbol,
            exchange=snapshot.exchange,
            timeframe=tf,
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
            "Funding pressure analyzed: symbol=%s exchange=%s level=%s direction=%s "
            "score=%.4f squeeze_prob=%.4f mean_reversion_prob=%.4f "
            "oi_confirmation=%s price_stall=%s",
            state.symbol,
            state.exchange.value,
            state.level.value,
            state.direction.value,
            state.pressure_score,
            state.squeeze_probability if state.squeeze_probability is not None else 0.0,
            state.mean_reversion_probability if state.mean_reversion_probability is not None else 0.0,
            state.oi_confirmation,
            state.price_stall_confirmation,
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

    def _detect_pressure_level(self, pressure_score: float) -> FundingPressureLevel:
        """
        Визначає рівень funding pressure.
        """
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
        Якщо regime слабкий/neutral, але pressure direction є,
        то підтягуємо bias із direction.
        """
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
        if (
            current_open_interest is None
            or previous_open_interest is None
            or previous_open_interest <= 0
        ):
            return False, None

        oi_change_pct = (
            (current_open_interest - previous_open_interest) / previous_open_interest
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
        if (
            current_price is None
            or previous_price is None
            or previous_price <= 0
        ):
            return False, None

        price_change_pct = abs((current_price - previous_price) / previous_price)
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
        score = (
            self.config.weight_magnitude * magnitude_score
            + self.config.weight_percentile * percentile_score
            + self.config.weight_zscore * zscore_score
            + self.config.weight_oi_confirmation * float(oi_confirmation)
            + self.config.weight_price_stall * float(price_stall_confirmation)
        )

        return max(0.0, min(1.0, score))

    def _calc_magnitude_score(self, funding_rate: float) -> float:
        abs_rate = abs(funding_rate)

        if abs_rate <= self.config.neutral_abs_threshold:
            return 0.0

        denominator = max(
            self.config.extreme_abs_threshold - self.config.neutral_abs_threshold,
            1e-12,
        )
        normalized = (abs_rate - self.config.neutral_abs_threshold) / denominator
        return max(0.0, min(1.0, normalized))

    def _calc_percentile_score(self, percentile: float | None) -> float:
        if percentile is None:
            return 0.0

        distance_from_center = abs(percentile - 50.0) / 50.0
        return max(0.0, min(1.0, distance_from_center))

    def _calc_zscore_score(self, zscore: float | None) -> float:
        if zscore is None:
            return 0.0

        normalized = abs(zscore) / max(self.config.extreme_zscore_threshold, 1e-12)
        return max(0.0, min(1.0, normalized))

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
        percentile_component = self._calc_percentile_score(percentile)

        probability = (
            self.config.squeeze_pressure_weight * pressure_score
            + self.config.squeeze_percentile_weight * percentile_component
            + self.config.squeeze_oi_weight * float(oi_confirmation)
            + self.config.squeeze_stall_weight * float(price_stall_confirmation)
        )

        return max(0.0, min(1.0, probability))

    def _estimate_mean_reversion_probability(
        self,
        pressure_score: float,
        percentile: float | None,
        zscore: float | None,
        price_stall_confirmation: bool,
    ) -> float:
        percentile_component = self._calc_percentile_score(percentile)
        zscore_component = self._calc_zscore_score(zscore)

        probability = (
            self.config.mean_reversion_pressure_weight * pressure_score
            + self.config.mean_reversion_percentile_weight * percentile_component
            + self.config.mean_reversion_zscore_weight * zscore_component
            + self.config.mean_reversion_stall_weight * float(price_stall_confirmation)
        )

        return max(0.0, min(1.0, probability))

    # ------------------------------------------------------------------
    # Optional helper methods for analyzer / strategies
    # ------------------------------------------------------------------

    def is_high_pressure(self, state: FundingPressureState) -> bool:
        return state.level in {
            FundingPressureLevel.HIGH,
            FundingPressureLevel.EXTREME,
        }

    def is_long_crowded(self, state: FundingPressureState) -> bool:
        return state.direction == FundingPressureDirection.LONG and self.is_high_pressure(state)

    def is_short_crowded(self, state: FundingPressureState) -> bool:
        return state.direction == FundingPressureDirection.SHORT and self.is_high_pressure(state)

    def is_squeeze_risk(self, state: FundingPressureState, threshold: float = 0.65) -> bool:
        if state.squeeze_probability is None:
            return False
        return state.squeeze_probability >= threshold

    def is_mean_reversion_risk(
        self,
        state: FundingPressureState,
        threshold: float = 0.60,
    ) -> bool:
        if state.mean_reversion_probability is None:
            return False
        return state.mean_reversion_probability >= threshold

    def build_summary(self, state: FundingPressureState) -> str:
        """
        Короткий текстовий summary, корисний для signal layer / dashboard / logs.
        """
        return (
            f"Funding pressure for {state.symbol}: "
            f"level={state.level.value}, "
            f"direction={state.direction.value}, "
            f"score={state.pressure_score:.4f}, "
            f"squeeze_probability={state.squeeze_probability:.4f} "
            f"mean_reversion_probability={state.mean_reversion_probability:.4f}"
        )