from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .config import OIAnalyzerConfig
from .enums import OIDirection
from .models import OIFeatures, OIMarketContext, OISnapshot


@dataclass(slots=True)
class OISeriesInput:
    """
    Контейнер з історичними рядами для побудови OI features.
    Усі ряди очікуються в хронологічному порядку:
    найстаріше -> найновіше.
    """

    oi_values: Sequence[float]
    oi_timestamps: Sequence[float]

    price_values: Sequence[float] | None = None
    price_timestamps: Sequence[float] | None = None

    volume_values: Sequence[float] | None = None
    volume_timestamps: Sequence[float] | None = None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _std(values: Sequence[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if len(cleaned) < 2:
        return None
    try:
        return statistics.pstdev(cleaned)
    except statistics.StatisticsError:
        return None


def _last(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return None


def _take_last(values: Sequence[float] | None, window: int) -> list[float]:
    if not values or window <= 0:
        return []
    return [float(v) for v in values[-window:] if v is not None]


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100.0


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def _infer_direction(
    delta_value: float | None,
    flat_epsilon: float = 1e-12,
) -> OIDirection:
    if delta_value is None:
        return OIDirection.UNKNOWN
    if abs(delta_value) <= flat_epsilon:
        return OIDirection.FLAT
    return OIDirection.UP if delta_value > 0 else OIDirection.DOWN


def _moving_average(values: Sequence[float], window: int) -> float | None:
    subset = _take_last(values, window)
    return _mean(subset)


def _zscore(values: Sequence[float], window: int) -> float | None:
    subset = _take_last(values, window)
    if len(subset) < 2:
        return None

    current = subset[-1]
    mu = _mean(subset)
    sigma = _std(subset)

    if mu is None or sigma is None or sigma == 0:
        return None

    return (current - mu) / sigma


def _velocity(values: Sequence[float], timestamps: Sequence[float]) -> float | None:
    """
    Проста оцінка швидкості зміни:
    (last_value - prev_value) / dt
    """
    if len(values) < 2 or len(timestamps) < 2:
        return None

    v1 = float(values[-2])
    v2 = float(values[-1])
    t1 = float(timestamps[-2])
    t2 = float(timestamps[-1])

    dt = t2 - t1
    if dt <= 0:
        return None

    return (v2 - v1) / dt


def _acceleration(
    values: Sequence[float],
    timestamps: Sequence[float],
) -> float | None:
    """
    Оцінка прискорення через різницю двох останніх швидкостей.
    """
    if len(values) < 3 or len(timestamps) < 3:
        return None

    v0 = float(values[-3])
    v1 = float(values[-2])
    v2 = float(values[-1])

    t0 = float(timestamps[-3])
    t1 = float(timestamps[-2])
    t2 = float(timestamps[-1])

    dt1 = t1 - t0
    dt2 = t2 - t1

    if dt1 <= 0 or dt2 <= 0:
        return None

    vel1 = (v1 - v0) / dt1
    vel2 = (v2 - v1) / dt2

    dt_mid = (dt1 + dt2) / 2.0
    if dt_mid <= 0:
        return None

    return (vel2 - vel1) / dt_mid


def _liquidation_imbalance(
    long_liquidations: float | None,
    short_liquidations: float | None,
) -> float | None:
    if long_liquidations is None or short_liquidations is None:
        return None

    total = long_liquidations + short_liquidations
    if total <= 0:
        return 0.0

    return (short_liquidations - long_liquidations) / total


def _aggressive_flow_imbalance(
    aggressive_buy_volume: float | None,
    aggressive_sell_volume: float | None,
) -> float | None:
    if aggressive_buy_volume is None or aggressive_sell_volume is None:
        return None

    total = aggressive_buy_volume + aggressive_sell_volume
    if total <= 0:
        return 0.0

    return (aggressive_buy_volume - aggressive_sell_volume) / total


def _volume_ratio(current_volume: float | None, volume_ma: float | None) -> float | None:
    if current_volume is None or volume_ma is None or volume_ma <= 0:
        return None
    return current_volume / volume_ma


def _oi_change_per_volume(
    oi_delta: float | None,
    volume: float | None,
) -> float | None:
    if oi_delta is None or volume is None or volume <= 0:
        return None
    return oi_delta / volume


def _oi_price_efficiency(
    oi_delta_pct: float | None,
    price_delta_pct: float | None,
) -> float | None:
    """
    Наскільки зміна ціни "підкріплена" зміною OI.
    > 1: OI рухається сильніше за ціну
    < 1: ціна рухається швидше, ніж OI
    """
    if oi_delta_pct is None or price_delta_pct is None:
        return None
    if abs(price_delta_pct) < 1e-12:
        return None
    return oi_delta_pct / price_delta_pct


def _bounded(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_zscore(zscore: float | None, cap: float = 5.0) -> float:
    if zscore is None:
        return 0.0
    if cap <= 0:
        return 0.0
    return _bounded(zscore / cap, -1.0, 1.0)


def _normalize_pct(value_pct: float | None, cap_pct: float = 5.0) -> float:
    if value_pct is None:
        return 0.0
    if cap_pct <= 0:
        return 0.0
    return _bounded(value_pct / cap_pct, -1.0, 1.0)


def _normalize_ratio_distance(ratio: float | None, neutral: float = 1.0, cap: float = 3.0) -> float:
    if ratio is None:
        return 0.0
    if cap <= neutral:
        return 0.0
    shifted = ratio - neutral
    max_dist = cap - neutral
    return _bounded(shifted / max_dist, -1.0, 1.0)


def _pressure_score(
    oi_delta_pct: float | None,
    price_delta_pct: float | None,
    volume_ratio: float | None,
    funding_rate: float | None,
    liquidation_imbalance: float | None,
    aggressive_flow_imbalance: float | None,
    oi_zscore: float | None,
) -> float | None:
    """
    Зведений score тиску / crowding / directional conviction.
    Діапазон приблизно [-1, 1].

    Позитивний:
        bullish pressure / long pressure / short squeeze risk
    Негативний:
        bearish pressure / short build / long flush
    """

    components: list[tuple[float, float]] = []

    # Ціна + OI
    if price_delta_pct is not None:
        components.append((_normalize_pct(price_delta_pct, 2.5), 0.24))
    if oi_delta_pct is not None:
        components.append((_normalize_pct(oi_delta_pct, 2.5), 0.24))

    # Обсяг як підтвердження руху
    if volume_ratio is not None:
        components.append((_normalize_ratio_distance(volume_ratio, neutral=1.0, cap=3.0), 0.12))

    # Агресивний потік
    if aggressive_flow_imbalance is not None:
        components.append((_bounded(aggressive_flow_imbalance), 0.14))

    # Ліквідації
    if liquidation_imbalance is not None:
        components.append((_bounded(liquidation_imbalance), 0.12))

    # Funding як crowding-сигнал
    if funding_rate is not None:
        components.append((_bounded(funding_rate / 0.03), 0.08))

    # Екстремальність OI
    if oi_zscore is not None:
        components.append((_normalize_zscore(oi_zscore, cap=4.0), 0.06))

    if not components:
        return None

    weighted_sum = sum(value * weight for value, weight in components)
    total_weight = sum(weight for _, weight in components)

    if total_weight <= 0:
        return None

    return _bounded(weighted_sum / total_weight)


class OIFeatureBuilder:
    """
    Основний builder для OI features.

    Він:
    - приймає поточний OI snapshot
    - бере історичні ряди OI / price / volume
    - враховує market context
    - повертає OIFeatures
    """

    def __init__(self, config: OIAnalyzerConfig) -> None:
        self.config = config
        self.windows = config.windows

    def compute_oi_delta(
        self,
        current_oi: float | None,
        previous_oi: float | None,
    ) -> float:
        return _delta(current_oi, previous_oi) or 0.0

    def compute_oi_delta_pct(
        self,
        current_oi: float | None,
        previous_oi: float | None,
    ) -> float:
        return _pct_change(current_oi, previous_oi) or 0.0

    def compute_price_delta(
        self,
        current_price: float | None,
        previous_price: float | None,
    ) -> float | None:
        return _delta(current_price, previous_price)

    def compute_price_delta_pct(
        self,
        current_price: float | None,
        previous_price: float | None,
    ) -> float | None:
        return _pct_change(current_price, previous_price)

    def compute_moving_average(
        self,
        values: Sequence[float],
        window: int,
    ) -> float | None:
        return _moving_average(values, window)

    def compute_std(
        self,
        values: Sequence[float],
        window: int,
    ) -> float | None:
        subset = _take_last(values, window)
        return _std(subset)

    def compute_zscore(
        self,
        values: Sequence[float],
        window: int,
    ) -> float | None:
        return _zscore(values, window)

    def compute_velocity(
        self,
        values: Sequence[float],
        timestamps: Sequence[float],
    ) -> float | None:
        return _velocity(values, timestamps)

    def compute_acceleration(
        self,
        values: Sequence[float],
        timestamps: Sequence[float],
    ) -> float | None:
        return _acceleration(values, timestamps)

    def compute_volume_ratio(
        self,
        current_volume: float | None,
        volume_ma: float | None,
    ) -> float | None:
        return _volume_ratio(current_volume, volume_ma)

    def compute_liquidation_imbalance(
        self,
        long_liquidations: float | None,
        short_liquidations: float | None,
    ) -> float | None:
        return _liquidation_imbalance(long_liquidations, short_liquidations)

    def compute_aggressive_flow_imbalance(
        self,
        aggressive_buy_volume: float | None,
        aggressive_sell_volume: float | None,
    ) -> float | None:
        return _aggressive_flow_imbalance(
            aggressive_buy_volume,
            aggressive_sell_volume,
        )

    def build_features(
        self,
        snapshot: OISnapshot,
        context: OIMarketContext | None,
        series: OISeriesInput,
    ) -> OIFeatures:
        """
        Побудова повного набору OI features.
        Очікується, що latest snapshot уже присутній в кінці oi_values.
        """
        oi_values = [float(v) for v in series.oi_values if v is not None]
        oi_timestamps = [float(ts) for ts in series.oi_timestamps if ts is not None]

        if not oi_values:
            raise ValueError("OISeriesInput.oi_values must contain at least one value")
        if len(oi_values) != len(oi_timestamps):
            raise ValueError("oi_values and oi_timestamps must have identical length")

        current_oi = float(snapshot.oi)
        previous_oi = float(oi_values[-2]) if len(oi_values) >= 2 else None

        oi_delta = self.compute_oi_delta(current_oi, previous_oi)
        oi_delta_pct = self.compute_oi_delta_pct(current_oi, previous_oi)

        oi_ma_fast = self.compute_moving_average(oi_values, self.windows.fast_window)
        oi_ma_slow = self.compute_moving_average(oi_values, self.windows.slow_window)
        oi_std = self.compute_std(oi_values, self.windows.zscore_window)
        oi_zscore = self.compute_zscore(oi_values, self.windows.zscore_window)
        oi_velocity = self.compute_velocity(oi_values, oi_timestamps)
        oi_acceleration = self.compute_acceleration(oi_values, oi_timestamps)

        price_values = (
            [float(v) for v in series.price_values if v is not None]
            if series.price_values
            else []
        )
        price_timestamps = (
            [float(ts) for ts in series.price_timestamps if ts is not None]
            if series.price_timestamps
            else []
        )

        volume_values = (
            [float(v) for v in series.volume_values if v is not None]
            if series.volume_values
            else []
        )

        context_price = context.price if context else None
        inferred_price = context_price if context_price is not None else _last(price_values)

        previous_price = float(price_values[-2]) if len(price_values) >= 2 else None
        if previous_price is None and len(price_values) >= 1 and inferred_price is not None:
            # якщо price_values містить лише одне значення, але воно збігається з current,
            # попередню точку ми не знаємо
            previous_price = None

        price_delta = (
            context.price_delta
            if context and context.price_delta is not None
            else self.compute_price_delta(inferred_price, previous_price)
        )

        price_delta_pct = (
            context.price_delta_pct
            if context and context.price_delta_pct is not None
            else self.compute_price_delta_pct(inferred_price, previous_price)
        )

        current_volume = context.volume if context else None
        if current_volume is None:
            current_volume = _last(volume_values)

        volume_ma = self.compute_moving_average(volume_values, self.windows.volume_window)
        volume_ratio = (
            context.volume_ratio
            if context and context.volume_ratio is not None
            else self.compute_volume_ratio(current_volume, volume_ma)
        )

        funding_rate = context.funding_rate if context else None
        long_liquidations = context.long_liquidations if context else None
        short_liquidations = context.short_liquidations if context else None
        cvd_delta = context.cvd_delta if context else None
        aggressive_buy_volume = context.aggressive_buy_volume if context else None
        aggressive_sell_volume = context.aggressive_sell_volume if context else None

        liquidation_imbalance = (
            context.liquidation_imbalance
            if context and context.liquidation_imbalance is not None
            else self.compute_liquidation_imbalance(
                long_liquidations,
                short_liquidations,
            )
        )

        aggressive_flow_imbalance = (
            context.aggressive_flow_imbalance
            if context and context.aggressive_flow_imbalance is not None
            else self.compute_aggressive_flow_imbalance(
                aggressive_buy_volume,
                aggressive_sell_volume,
            )
        )

        oi_change_per_volume = _oi_change_per_volume(oi_delta, current_volume)
        oi_price_efficiency = _oi_price_efficiency(oi_delta_pct, price_delta_pct)

        oi_direction = _infer_direction(oi_delta)
        price_direction = _infer_direction(price_delta)

        oi_pressure_score = _pressure_score(
            oi_delta_pct=oi_delta_pct,
            price_delta_pct=price_delta_pct,
            volume_ratio=volume_ratio,
            funding_rate=funding_rate,
            liquidation_imbalance=liquidation_imbalance,
            aggressive_flow_imbalance=aggressive_flow_imbalance,
            oi_zscore=oi_zscore,
        )

        return OIFeatures(
            oi=current_oi,
            oi_delta=oi_delta,
            oi_delta_pct=oi_delta_pct,
            oi_ma_fast=oi_ma_fast,
            oi_ma_slow=oi_ma_slow,
            oi_std=oi_std,
            oi_zscore=oi_zscore,
            oi_velocity=oi_velocity,
            oi_acceleration=oi_acceleration,
            price=inferred_price,
            price_delta=price_delta,
            price_delta_pct=price_delta_pct,
            volume=current_volume,
            volume_ma=volume_ma,
            volume_ratio=volume_ratio,
            funding_rate=funding_rate,
            long_liquidations=long_liquidations,
            short_liquidations=short_liquidations,
            liquidation_imbalance=liquidation_imbalance,
            cvd_delta=cvd_delta,
            aggressive_buy_volume=aggressive_buy_volume,
            aggressive_sell_volume=aggressive_sell_volume,
            aggressive_flow_imbalance=aggressive_flow_imbalance,
            oi_change_per_volume=oi_change_per_volume,
            oi_price_efficiency=oi_price_efficiency,
            oi_pressure_score=oi_pressure_score,
            oi_direction=oi_direction,
            price_direction=price_direction,
        )

    def build_minimal_features(
        self,
        snapshot: OISnapshot,
        oi_values: Sequence[float],
        oi_timestamps: Sequence[float],
    ) -> OIFeatures:
        """
        Спрощений варіант для випадків, коли ще немає price/volume context.
        Корисно для bootstrap-стану або unit-тестів.
        """
        return self.build_features(
            snapshot=snapshot,
            context=None,
            series=OISeriesInput(
                oi_values=oi_values,
                oi_timestamps=oi_timestamps,
            ),
        )

    def build_from_raw_inputs(
        self,
        *,
        snapshot: OISnapshot,
        context: OIMarketContext | None,
        oi_values: Sequence[float],
        oi_timestamps: Sequence[float],
        price_values: Sequence[float] | None = None,
        price_timestamps: Sequence[float] | None = None,
        volume_values: Sequence[float] | None = None,
        volume_timestamps: Sequence[float] | None = None,
    ) -> OIFeatures:
        """
        Зручний helper для analyzer layer.
        """
        return self.build_features(
            snapshot=snapshot,
            context=context,
            series=OISeriesInput(
                oi_values=oi_values,
                oi_timestamps=oi_timestamps,
                price_values=price_values,
                price_timestamps=price_timestamps,
                volume_values=volume_values,
                volume_timestamps=volume_timestamps,
            ),
        )

    @staticmethod
    def is_price_confirmation_present(
        features: OIFeatures,
        min_price_change_pct: float,
    ) -> bool:
        return (
            features.price_delta_pct is not None
            and abs(features.price_delta_pct) >= float(min_price_change_pct)
        )

    @staticmethod
    def is_volume_confirmation_present(
        features: OIFeatures,
        min_volume_ratio: float,
    ) -> bool:
        return (
            features.volume_ratio is not None
            and features.volume_ratio >= float(min_volume_ratio)
        )

    @staticmethod
    def is_oi_expansion_present(
        features: OIFeatures,
        min_oi_change_pct: float,
    ) -> bool:
        return abs(features.oi_delta_pct) >= float(min_oi_change_pct)

    @staticmethod
    def is_oi_extreme(
        features: OIFeatures,
        zscore_threshold: float,
    ) -> bool:
        return (
            features.oi_zscore is not None
            and abs(features.oi_zscore) >= float(zscore_threshold)
        )

    @staticmethod
    def describe_features(features: OIFeatures) -> list[str]:
        """
        Корисно для debug/logging/reasons generation.
        """
        reasons: list[str] = []

        if features.oi_direction == OIDirection.UP:
            reasons.append("oi_up")
        elif features.oi_direction == OIDirection.DOWN:
            reasons.append("oi_down")
        elif features.oi_direction == OIDirection.FLAT:
            reasons.append("oi_flat")

        if features.price_direction == OIDirection.UP:
            reasons.append("price_up")
        elif features.price_direction == OIDirection.DOWN:
            reasons.append("price_down")
        elif features.price_direction == OIDirection.FLAT:
            reasons.append("price_flat")

        if features.volume_ratio is not None:
            if features.volume_ratio >= 1.5:
                reasons.append("high_volume_confirmation")
            elif features.volume_ratio >= 1.0:
                reasons.append("moderate_volume_confirmation")
            else:
                reasons.append("weak_volume")

        if features.oi_zscore is not None:
            if features.oi_zscore >= 3:
                reasons.append("extreme_positive_oi_zscore")
            elif features.oi_zscore <= -3:
                reasons.append("extreme_negative_oi_zscore")
            elif abs(features.oi_zscore) >= 2:
                reasons.append("elevated_oi_zscore")

        if features.funding_rate is not None:
            if features.funding_rate > 0:
                reasons.append("positive_funding")
            elif features.funding_rate < 0:
                reasons.append("negative_funding")

        if features.liquidation_imbalance is not None:
            if features.liquidation_imbalance > 0.2:
                reasons.append("short_liquidation_pressure")
            elif features.liquidation_imbalance < -0.2:
                reasons.append("long_liquidation_pressure")

        if features.aggressive_flow_imbalance is not None:
            if features.aggressive_flow_imbalance > 0.15:
                reasons.append("aggressive_buy_flow")
            elif features.aggressive_flow_imbalance < -0.15:
                reasons.append("aggressive_sell_flow")

        if features.oi_pressure_score is not None:
            if features.oi_pressure_score >= 0.6:
                reasons.append("strong_positive_pressure")
            elif features.oi_pressure_score <= -0.6:
                reasons.append("strong_negative_pressure")

        return reasons