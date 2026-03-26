from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityLevelType, LiquiditySide, LiquidityStatus, SweepStatus
from .models import EqualLevel
from .scoring import LiquidityScorer
from .utils import (
    calculate_atr_from_ohlc,
    clamp,
    is_pivot_high,
    is_pivot_low,
    midpoint,
    pct_distance,
)


@dataclass(slots=True)
class PivotPoint:
    index: int
    price: float
    timestamp: datetime | None = None
    volume: float | None = None


class EqualHighsLowsDetector:
    """
    Детектор equal highs / equal lows.

    Очікування до candle:
    - dict з ключами: high, low, close, volume?, open_time?/close_time?/timestamp?
    або
    - object із відповідними атрибутами
    """

    def __init__(
        self,
        config: LiquidityConfig,
        scorer: LiquidityScorer | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)
        self._event_bus = event_bus
        self._logger = get_logger(__name__, service_name="liquidity")

    def detect(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Any],
        current_price: float | None = None,
    ) -> list[EqualLevel]:
        """
        Повний batch-аналіз свічок.
        """
        if len(candles) < self._minimum_required_candles():
            self._logger.debug(
                "Not enough candles for equal highs/lows detection",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles_count": len(candles),
                },
            )
            return []

        highs = [self._get_candle_high(c) for c in candles]
        lows = [self._get_candle_low(c) for c in candles]
        closes = [self._get_candle_close(c) for c in candles]

        tolerance_pct = self._resolve_tolerance_pct(highs, lows, closes, current_price)

        pivot_highs = self._find_pivot_highs(candles)
        pivot_lows = self._find_pivot_lows(candles)

        equal_highs = self._build_equal_levels(
            symbol=symbol,
            timeframe=timeframe,
            pivots=pivot_highs,
            level_type=LiquidityLevelType.EQUAL_HIGHS,
            side=LiquiditySide.BUY_SIDE,
            tolerance_pct=tolerance_pct,
            candles=candles,
            current_price=current_price,
        )

        equal_lows = self._build_equal_levels(
            symbol=symbol,
            timeframe=timeframe,
            pivots=pivot_lows,
            level_type=LiquidityLevelType.EQUAL_LOWS,
            side=LiquiditySide.SELL_SIDE,
            tolerance_pct=tolerance_pct,
            candles=candles,
            current_price=current_price,
        )

        levels = equal_highs + equal_lows
        levels.sort(key=lambda level: level.price)

        self._logger.info(
            "Equal highs/lows detected",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "equal_highs": len(equal_highs),
                "equal_lows": len(equal_lows),
                "total_levels": len(levels),
                "tolerance_pct": tolerance_pct,
            },
        )

        return levels

    def update_incremental(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Any],
        current_price: float | None = None,
    ) -> list[EqualLevel]:
        """
        Поки що для старту просто викликає batch detect().
        Пізніше можна оптимізувати через вікно останніх N свічок і state.
        """
        return self.detect(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            current_price=current_price,
        )

    def _minimum_required_candles(self) -> int:
        return self._config.pivot_lookback + self._config.pivot_lookforward + 3

    def _find_pivot_highs(self, candles: Sequence[Any]) -> list[PivotPoint]:
        highs = [self._get_candle_high(c) for c in candles]
        pivots: list[PivotPoint] = []

        for idx in range(len(candles)):
            if is_pivot_high(
                highs=highs,
                index=idx,
                left=self._config.pivot_lookback,
                right=self._config.pivot_lookforward,
            ):
                pivots.append(
                    PivotPoint(
                        index=idx,
                        price=highs[idx],
                        timestamp=self._get_candle_timestamp(candles[idx]),
                        volume=self._get_candle_volume(candles[idx]),
                    )
                )

        return self._filter_nearby_pivots_by_swing_distance(
            pivots=pivots,
            is_highs=True,
        )

    def _find_pivot_lows(self, candles: Sequence[Any]) -> list[PivotPoint]:
        lows = [self._get_candle_low(c) for c in candles]
        pivots: list[PivotPoint] = []

        for idx in range(len(candles)):
            if is_pivot_low(
                lows=lows,
                index=idx,
                left=self._config.pivot_lookback,
                right=self._config.pivot_lookforward,
            ):
                pivots.append(
                    PivotPoint(
                        index=idx,
                        price=lows[idx],
                        timestamp=self._get_candle_timestamp(candles[idx]),
                        volume=self._get_candle_volume(candles[idx]),
                    )
                )

        return self._filter_nearby_pivots_by_swing_distance(
            pivots=pivots,
            is_highs=False,
        )

    def _filter_nearby_pivots_by_swing_distance(
        self,
        pivots: list[PivotPoint],
        is_highs: bool,
    ) -> list[PivotPoint]:
        """
        Прибирає занадто близькі екстремуми.
        """
        if not pivots:
            return []

        filtered: list[PivotPoint] = [pivots[0]]

        for pivot in pivots[1:]:
            prev = filtered[-1]
            distance = pct_distance(pivot.price, prev.price)

            if distance < self._config.min_swing_distance_pct:
                if is_highs and pivot.price > prev.price:
                    filtered[-1] = pivot
                elif not is_highs and pivot.price < prev.price:
                    filtered[-1] = pivot
                continue

            filtered.append(pivot)

        return filtered

    def _build_equal_levels(
        self,
        symbol: str,
        timeframe: str,
        pivots: list[PivotPoint],
        level_type: LiquidityLevelType,
        side: LiquiditySide,
        tolerance_pct: float,
        candles: Sequence[Any],
        current_price: float | None,
    ) -> list[EqualLevel]:
        if len(pivots) < self._config.min_equal_touches:
            return []

        clusters = self._cluster_pivots_by_price(
            pivots=pivots,
            tolerance_pct=tolerance_pct,
        )

        levels: list[EqualLevel] = []
        for cluster in clusters:
            if len(cluster) < self._config.min_equal_touches:
                continue

            level = self._create_equal_level(
                symbol=symbol,
                timeframe=timeframe,
                cluster=cluster,
                level_type=level_type,
                side=side,
                tolerance_pct=tolerance_pct,
                candles=candles,
            )
            level.confidence = self._scorer.score_equal_level(
                level=level,
                current_price=current_price,
            )

            if level.confidence < self._config.min_confidence:
                continue

            self._mark_swept_status(
                level=level,
                candles=candles,
                level_type=level_type,
            )

            levels.append(level)

        return self._deduplicate_levels(levels)

    def _cluster_pivots_by_price(
        self,
        pivots: list[PivotPoint],
        tolerance_pct: float,
    ) -> list[list[PivotPoint]]:
        """
        Кластеризація pivot-точок по близькості цін.
        """
        if not pivots:
            return []

        sorted_pivots = sorted(pivots, key=lambda p: p.price)
        clusters: list[list[PivotPoint]] = [[sorted_pivots[0]]]

        for pivot in sorted_pivots[1:]:
            current_cluster = clusters[-1]
            cluster_prices = [p.price for p in current_cluster]
            cluster_center = sum(cluster_prices) / len(cluster_prices)

            if pct_distance(pivot.price, cluster_center) <= tolerance_pct:
                current_cluster.append(pivot)
            else:
                clusters.append([pivot])

        return clusters

    def _create_equal_level(
        self,
        symbol: str,
        timeframe: str,
        cluster: list[PivotPoint],
        level_type: LiquidityLevelType,
        side: LiquiditySide,
        tolerance_pct: float,
        candles: Sequence[Any],
    ) -> EqualLevel:
        prices = [p.price for p in cluster]
        pivot_indexes = [p.index for p in cluster]

        first_seen_at = cluster[0].timestamp
        last_seen_at = cluster[-1].timestamp

        price = midpoint(min(prices), max(prices))
        touches_count = len(cluster)
        reaction_count = self._estimate_reaction_count(
            pivot_indexes=pivot_indexes,
            candles=candles,
            level_type=level_type,
        )

        return EqualLevel(
            symbol=symbol,
            timeframe=timeframe,
            level_type=level_type,
            side=side,
            price=price,
            status=LiquidityStatus.ACTIVE,
            sweep_status=SweepStatus.NOT_SWEPT,
            confidence=0.0,
            touches_count=touches_count,
            reaction_count=reaction_count,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            source="equal_highs_lows",
            tolerance_pct=tolerance_pct,
            cluster_low=min(prices),
            cluster_high=max(prices),
            level_prices=prices,
            pivot_indexes=pivot_indexes,
            metadata={
                "pivot_count": len(cluster),
                "pivot_timestamps": [
                    p.timestamp.isoformat() if p.timestamp else None for p in cluster
                ],
            },
        )

    def _estimate_reaction_count(
        self,
        pivot_indexes: list[int],
        candles: Sequence[Any],
        level_type: LiquidityLevelType,
    ) -> int:
        """
        Спрощено: реакцією вважаємо відкат після pivot.
        """
        reactions = 0
        n = len(candles)

        for idx in pivot_indexes:
            if idx >= n - 1:
                continue

            pivot_high = self._get_candle_high(candles[idx])
            pivot_low = self._get_candle_low(candles[idx])

            future_slice = candles[idx + 1 : min(idx + 4, n)]
            if not future_slice:
                continue

            if level_type == LiquidityLevelType.EQUAL_HIGHS:
                min_future_low = min(self._get_candle_low(c) for c in future_slice)
                if pivot_high > 0:
                    reaction_pct = (pivot_high - min_future_low) / pivot_high
                    if reaction_pct >= self._config.min_swing_distance_pct / 2.0:
                        reactions += 1

            elif level_type == LiquidityLevelType.EQUAL_LOWS:
                max_future_high = max(self._get_candle_high(c) for c in future_slice)
                if pivot_low > 0:
                    reaction_pct = (max_future_high - pivot_low) / pivot_low
                    if reaction_pct >= self._config.min_swing_distance_pct / 2.0:
                        reactions += 1

        return reactions

    def _mark_swept_status(
        self,
        level: EqualLevel,
        candles: Sequence[Any],
        level_type: LiquidityLevelType,
    ) -> None:
        """
        Якщо після останнього pivot ціна вже явно пробила рівень — вважаємо sweep.
        """
        if not level.pivot_indexes:
            return

        last_pivot_idx = max(level.pivot_indexes)
        future_candles = candles[last_pivot_idx + 1 :]
        if not future_candles:
            return

        if level_type == LiquidityLevelType.EQUAL_HIGHS:
            max_future_high = max(self._get_candle_high(c) for c in future_candles)
            if max_future_high > (level.cluster_high or level.price):
                level.sweep_status = SweepStatus.SWEPT
                level.status = LiquidityStatus.SWEPT
                level.swept_at = self._get_candle_timestamp(future_candles[-1])

        elif level_type == LiquidityLevelType.EQUAL_LOWS:
            min_future_low = min(self._get_candle_low(c) for c in future_candles)
            if min_future_low < (level.cluster_low or level.price):
                level.sweep_status = SweepStatus.SWEPT
                level.status = LiquidityStatus.SWEPT
                level.swept_at = self._get_candle_timestamp(future_candles[-1])

    def _deduplicate_levels(self, levels: list[EqualLevel]) -> list[EqualLevel]:
        """
        Якщо кілька кластерів майже накладаються — залишаємо сильніший.
        """
        if not levels:
            return []

        levels = sorted(levels, key=lambda x: x.price)
        deduplicated: list[EqualLevel] = [levels[0]]

        for level in levels[1:]:
            prev = deduplicated[-1]
            if pct_distance(level.price, prev.price) <= level.tolerance_pct:
                if level.confidence > prev.confidence:
                    deduplicated[-1] = level
            else:
                deduplicated.append(level)

        return deduplicated

    def _resolve_tolerance_pct(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        current_price: float | None,
    ) -> float:
        """
        Обирає tolerance:
        - базовий config.equal_level_tolerance_pct
        - або ATR-based адаптивний tolerance
        """
        base_tolerance = self._config.equal_level_tolerance_pct

        if not self._config.use_atr_tolerance:
            return base_tolerance

        atr = calculate_atr_from_ohlc(
            highs=highs,
            lows=lows,
            closes=closes,
            period=self._config.atr_period,
        )

        reference_price = current_price or (closes[-1] if closes else 0.0)
        if reference_price <= 0:
            return base_tolerance

        atr_pct = atr / reference_price
        adaptive = max(base_tolerance, atr_pct * 0.35)

        return clamp(adaptive, base_tolerance, self._config.max_equal_cluster_width_pct)

    def _get_candle_value(self, candle: Any, field: str, default: Any = None) -> Any:
        if isinstance(candle, dict):
            return candle.get(field, default)
        return getattr(candle, field, default)

    def _get_candle_high(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "high")
        if value is None:
            raise ValueError("Candle must contain 'high'")
        return float(value)

    def _get_candle_low(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "low")
        if value is None:
            raise ValueError("Candle must contain 'low'")
        return float(value)

    def _get_candle_close(self, candle: Any) -> float:
        value = self._get_candle_value(candle, "close")
        if value is None:
            raise ValueError("Candle must contain 'close'")
        return float(value)

    def _get_candle_volume(self, candle: Any) -> float | None:
        value = self._get_candle_value(candle, "volume")
        if value is None:
            return None
        return float(value)

    def _get_candle_timestamp(self, candle: Any) -> datetime | None:
        for field in ("close_time", "timestamp", "open_time", "time"):
            value = self._get_candle_value(candle, field)
            if value is None:
                continue

            if isinstance(value, datetime):
                return value

            if isinstance(value, (int, float)):
                try:
                    # підтримка timestamp у ms і s
                    return datetime.utcfromtimestamp(value / 1000 if value > 1e12 else value)
                except Exception:
                    return None

            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except Exception:
                    return None

        return None