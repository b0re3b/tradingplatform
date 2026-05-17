from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from core.logger import get_logger

from .config import LiquidityConfig
from .enums import LiquidityLevelType, LiquiditySide
from .models import DEFAULT_EXCHANGE, DEFAULT_MARKET_TYPE, EqualLevel
from .scoring import LiquidityScorer
from .utils import (
    calculate_atr_pct_from_ohlc,
    clamp,
    extract_hlc,
    get_candle_high,
    get_candle_low,
    get_candle_timestamp,
    get_candle_volume,
    is_pivot_high,
    is_pivot_low,
    midpoint,
    pct_distance,
    safe_float,
    safe_mean,
)


@dataclass(slots=True)
class PivotPoint:
    """
    Внутрішня модель pivot-точки для equal highs / equal lows detection.
    """

    index: int
    price: float
    timestamp: datetime | None = None
    volume: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "price": self.price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "volume": self.volume,
        }


class EqualHighsLowsDetector:
    """
    Production-ready detector equal highs / equal lows.

    Відповідальність:
    - знайти pivot highs / pivot lows;
    - згрупувати близькі pivot-и у equal-level clusters;
    - оцінити confidence через LiquidityScorer;
    - визначити swept / partially swept status;
    - повернути список EqualLevel.

    Архітектурні правила:
    - не приймає EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - не зберігає глобальний mutable state;
    - використовується як pure domain detector всередині LiquidityMap.

    Multi-exchange behavior:
    - detector не отримує дані напряму з бірж;
    - exchange/market_type передаються як scope metadata;
    - основне джерело candles — LiquidityMap/LiquidityService, які вже отримали
      нормалізовані market.candle.closed з data layer.
    """

    def __init__(
        self,
        config: LiquidityConfig,
        scorer: LiquidityScorer | None = None,
    ) -> None:
        self._config = config
        self._config.validate()

        self._scorer = scorer or LiquidityScorer(config=config)

        self._logger = get_logger(
            __name__,
            service_name="analytics_liquidity",
            event_type="equal_highs_lows_detector",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Any],
        current_price: float | None = None,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> list[EqualLevel]:
        """
        Повний batch-аналіз equal highs / equal lows.

        Parameters
        ----------
        exchange:
            Біржа, наприклад binance / bybit / okx / mexc.
        market_type:
            Тип ринку, наприклад usdm_futures / linear / swap / spot.
        symbol:
            Нормалізований торговий символ, наприклад BTCUSDT.
        timeframe:
            Таймфрейм, наприклад 1m / 5m / 1h.
        candles:
            Нормалізовані candles або dict/object payload-и з high/low/close.
        current_price:
            Поточна ціна для proximity scoring.

        Returns
        -------
        list[EqualLevel]
            Відфільтровані, scored і deduplicated equal levels.
        """
        exchange = self._normalize_scope_value(exchange, DEFAULT_EXCHANGE)
        market_type = self._normalize_scope_value(market_type, DEFAULT_MARKET_TYPE)
        symbol = self._normalize_symbol(symbol)
        timeframe = self._normalize_timeframe(timeframe)

        candles_list = list(candles)

        if not self._config.enabled:
            self._logger.debug(
                "Equal highs/lows detection skipped: liquidity module disabled",
                extra={
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                },
            )
            return []

        if not symbol or not timeframe:
            raise ValueError("symbol and timeframe are required")

        minimum_candles = self._minimum_required_candles()
        if len(candles_list) < minimum_candles:
            self._logger.debug(
                "Not enough candles for equal highs/lows detection",
                extra={
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles_count": len(candles_list),
                    "minimum_required": minimum_candles,
                },
            )
            return []

        try:
            highs, lows, closes = extract_hlc(candles_list)
            self._validate_ohlc(
                highs=highs,
                lows=lows,
                closes=closes,
            )
        except Exception:
            self._logger.exception(
                "Failed to extract OHLC for equal highs/lows detection",
                extra={
                    "exchange": exchange,
                    "market_type": market_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles_count": len(candles_list),
                },
            )
            raise

        resolved_current_price = self._resolve_current_price(
            current_price=current_price,
            closes=closes,
        )

        tolerance_pct = self._resolve_tolerance_pct(
            highs=highs,
            lows=lows,
            closes=closes,
            current_price=resolved_current_price,
        )

        pivot_highs = self._find_pivot_highs(
            candles=candles_list,
            highs=highs,
        )
        pivot_lows = self._find_pivot_lows(
            candles=candles_list,
            lows=lows,
        )

        equal_highs = self._build_equal_levels(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            pivots=pivot_highs,
            level_type=LiquidityLevelType.EQUAL_HIGHS,
            side=LiquiditySide.BUY_SIDE,
            tolerance_pct=tolerance_pct,
            candles=candles_list,
            current_price=resolved_current_price,
        )

        equal_lows = self._build_equal_levels(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            pivots=pivot_lows,
            level_type=LiquidityLevelType.EQUAL_LOWS,
            side=LiquiditySide.SELL_SIDE,
            tolerance_pct=tolerance_pct,
            candles=candles_list,
            current_price=resolved_current_price,
        )

        levels = self._deduplicate_levels(equal_highs + equal_lows)
        levels.sort(key=lambda level: (level.price, level.level_type.value))

        self._logger.info(
            "Equal highs/lows detected",
            extra={
                "exchange": exchange,
                "market_type": market_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles_count": len(candles_list),
                "pivot_highs": len(pivot_highs),
                "pivot_lows": len(pivot_lows),
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
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
    ) -> list[EqualLevel]:
        """
        Incremental entrypoint.

        Поки що використовує batch detect() для стабільності.
        Пізніше можна оптимізувати через sliding window/state,
        але state має жити у LiquidityService/LiquidityState, не тут.
        """
        return self.detect(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            current_price=current_price,
        )

    # ------------------------------------------------------------------
    # Pivot detection
    # ------------------------------------------------------------------

    def _minimum_required_candles(self) -> int:
        return self._config.pivot_lookback + self._config.pivot_lookforward + 3

    def _find_pivot_highs(
        self,
        candles: Sequence[Any],
        highs: Sequence[float],
    ) -> list[PivotPoint]:
        pivots: list[PivotPoint] = []

        for index, price in enumerate(highs):
            if is_pivot_high(
                highs=highs,
                index=index,
                left=self._config.pivot_lookback,
                right=self._config.pivot_lookforward,
            ):
                pivots.append(
                    PivotPoint(
                        index=index,
                        price=price,
                        timestamp=self._normalize_timestamp(
                            get_candle_timestamp(candles[index])
                        ),
                        volume=self._safe_optional_volume(candles[index]),
                    )
                )

        return self._filter_nearby_pivots_by_swing_distance(
            pivots=pivots,
            is_highs=True,
        )

    def _find_pivot_lows(
        self,
        candles: Sequence[Any],
        lows: Sequence[float],
    ) -> list[PivotPoint]:
        pivots: list[PivotPoint] = []

        for index, price in enumerate(lows):
            if is_pivot_low(
                lows=lows,
                index=index,
                left=self._config.pivot_lookback,
                right=self._config.pivot_lookforward,
            ):
                pivots.append(
                    PivotPoint(
                        index=index,
                        price=price,
                        timestamp=self._normalize_timestamp(
                            get_candle_timestamp(candles[index])
                        ),
                        volume=self._safe_optional_volume(candles[index]),
                    )
                )

        return self._filter_nearby_pivots_by_swing_distance(
            pivots=pivots,
            is_highs=False,
        )

    def _filter_nearby_pivots_by_swing_distance(
        self,
        pivots: list[PivotPoint],
        *,
        is_highs: bool,
    ) -> list[PivotPoint]:
        """
        Прибирає занадто близькі pivot-и, залишаючи сильніший екстремум.
        """
        if not pivots:
            return []

        filtered: list[PivotPoint] = [pivots[0]]

        for pivot in pivots[1:]:
            previous = filtered[-1]
            distance = pct_distance(pivot.price, previous.price)

            if distance < self._config.min_swing_distance_pct:
                should_replace = (
                    pivot.price > previous.price
                    if is_highs
                    else pivot.price < previous.price
                )

                if should_replace:
                    filtered[-1] = pivot

                continue

            filtered.append(pivot)

        return filtered

    # ------------------------------------------------------------------
    # Equal level construction
    # ------------------------------------------------------------------

    def _build_equal_levels(
        self,
        exchange: str,
        market_type: str,
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
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
                cluster=cluster,
                level_type=level_type,
                side=side,
                tolerance_pct=tolerance_pct,
                candles=candles,
            )

            self._mark_sweep_status(
                level=level,
                candles=candles,
                level_type=level_type,
            )

            level.confidence = self._scorer.score_equal_level(
                level=level,
                current_price=current_price,
            )

            if level.confidence < self._config.min_confidence:
                continue

            levels.append(level)

        return self._deduplicate_levels(levels)

    def _cluster_pivots_by_price(
        self,
        pivots: list[PivotPoint],
        tolerance_pct: float,
    ) -> list[list[PivotPoint]]:
        """
        Кластеризує pivot-и за близькістю ціни.
        """
        if not pivots:
            return []

        sorted_pivots = sorted(pivots, key=lambda pivot: pivot.price)
        clusters: list[list[PivotPoint]] = [[sorted_pivots[0]]]

        for pivot in sorted_pivots[1:]:
            current_cluster = clusters[-1]
            cluster_center = safe_mean([item.price for item in current_cluster])

            if pct_distance(pivot.price, cluster_center) <= tolerance_pct:
                current_cluster.append(pivot)
            else:
                clusters.append([pivot])

        return clusters

    def _create_equal_level(
        self,
        exchange: str,
        market_type: str,
        symbol: str,
        timeframe: str,
        cluster: list[PivotPoint],
        level_type: LiquidityLevelType,
        side: LiquiditySide,
        tolerance_pct: float,
        candles: Sequence[Any],
    ) -> EqualLevel:
        prices = [pivot.price for pivot in cluster]
        pivot_indexes = [pivot.index for pivot in cluster]
        volumes = [
            pivot.volume
            for pivot in cluster
            if pivot.volume is not None
        ]

        cluster_low = min(prices)
        cluster_high = max(prices)
        price = midpoint(cluster_low, cluster_high)

        first_seen_at = cluster[0].timestamp
        last_seen_at = cluster[-1].timestamp

        reaction_count = self._estimate_reaction_count(
            pivot_indexes=pivot_indexes,
            candles=candles,
            level_type=level_type,
        )

        return EqualLevel(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
            level_type=level_type,
            side=side,
            price=price,
            confidence=0.0,
            touches_count=len(cluster),
            reaction_count=reaction_count,
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            source="equal_highs_lows_detector",
            tolerance_pct=tolerance_pct,
            cluster_low=cluster_low,
            cluster_high=cluster_high,
            level_prices=prices,
            pivot_indexes=pivot_indexes,
            metadata={
                "detector": self.__class__.__name__,
                "exchange": exchange,
                "market_type": market_type,
                "pivot_count": len(cluster),
                "pivot_indexes": pivot_indexes,
                "pivot_prices": prices,
                "pivot_timestamps": [
                    pivot.timestamp.isoformat() if pivot.timestamp else None
                    for pivot in cluster
                ],
                "avg_pivot_volume": safe_mean(volumes) if volumes else 0.0,
                "cluster_width_pct": (
                    abs(cluster_high - cluster_low) / abs(price)
                    if price > 0
                    else 0.0
                ),
            },
        )

    def _estimate_reaction_count(
        self,
        pivot_indexes: list[int],
        candles: Sequence[Any],
        level_type: LiquidityLevelType,
    ) -> int:
        """
        Оцінює кількість реакцій після pivot.

        Для equal highs реакція — помітний відкат вниз після high.
        Для equal lows реакція — помітний відскок вгору після low.
        """
        reactions = 0
        candles_count = len(candles)
        min_reaction_pct = self._config.min_swing_distance_pct / 2.0

        for index in pivot_indexes:
            if index >= candles_count - 1:
                continue

            future_slice = candles[index + 1 : min(index + 4, candles_count)]
            if not future_slice:
                continue

            pivot_high = get_candle_high(candles[index])
            pivot_low = get_candle_low(candles[index])

            if level_type == LiquidityLevelType.EQUAL_HIGHS:
                min_future_low = min(get_candle_low(candle) for candle in future_slice)

                if pivot_high > 0:
                    reaction_pct = (pivot_high - min_future_low) / abs(pivot_high)
                    if reaction_pct >= min_reaction_pct:
                        reactions += 1

            elif level_type == LiquidityLevelType.EQUAL_LOWS:
                max_future_high = max(get_candle_high(candle) for candle in future_slice)

                if pivot_low > 0:
                    reaction_pct = (max_future_high - pivot_low) / abs(pivot_low)
                    if reaction_pct >= min_reaction_pct:
                        reactions += 1

        return reactions

    # ------------------------------------------------------------------
    # Sweep detection
    # ------------------------------------------------------------------

    def _mark_sweep_status(
        self,
        level: EqualLevel,
        candles: Sequence[Any],
        level_type: LiquidityLevelType,
    ) -> None:
        """
        Визначає swept / partially swept status після останнього pivot.

        Логіка:
        - full sweep: ціна явно пробила cluster boundary;
        - partial sweep: ціна зайшла в tolerance-зону, але не дала повного sweep.
        """
        if not level.pivot_indexes:
            return

        last_pivot_index = max(level.pivot_indexes)
        future_candles = candles[last_pivot_index + 1 :]

        if not future_candles:
            return

        swept_at = self._normalize_timestamp(
            get_candle_timestamp(future_candles[-1])
        )

        if level_type == LiquidityLevelType.EQUAL_HIGHS:
            self._mark_high_level_sweep_status(
                level=level,
                future_candles=future_candles,
                swept_at=swept_at,
            )
            return

        if level_type == LiquidityLevelType.EQUAL_LOWS:
            self._mark_low_level_sweep_status(
                level=level,
                future_candles=future_candles,
                swept_at=swept_at,
            )

    def _mark_high_level_sweep_status(
        self,
        level: EqualLevel,
        future_candles: Sequence[Any],
        swept_at: datetime | None,
    ) -> None:
        cluster_high = level.cluster_high or level.price
        tolerance_price = cluster_high * (1.0 - level.tolerance_pct)

        max_future_high = max(get_candle_high(candle) for candle in future_candles)

        if max_future_high > cluster_high:
            level.mark_swept(swept_at=swept_at)
            return

        if max_future_high >= tolerance_price:
            level.mark_partially_swept(swept_at=swept_at)

    def _mark_low_level_sweep_status(
        self,
        level: EqualLevel,
        future_candles: Sequence[Any],
        swept_at: datetime | None,
    ) -> None:
        cluster_low = level.cluster_low or level.price
        tolerance_price = cluster_low * (1.0 + level.tolerance_pct)

        min_future_low = min(get_candle_low(candle) for candle in future_candles)

        if min_future_low < cluster_low:
            level.mark_swept(swept_at=swept_at)
            return

        if min_future_low <= tolerance_price:
            level.mark_partially_swept(swept_at=swept_at)

    # ------------------------------------------------------------------
    # Deduplication / tolerance
    # ------------------------------------------------------------------

    def _deduplicate_levels(
        self,
        levels: list[EqualLevel],
    ) -> list[EqualLevel]:
        """
        Якщо кілька clusters майже накладаються — залишаємо сильніший.

        Scope-aware:
        рівні з різних exchange/market_type/symbol/timeframe не дедуплікуються
        між собою.
        """
        if not levels:
            return []

        sorted_levels = sorted(
            levels,
            key=lambda level: (
                level.exchange,
                level.market_type,
                level.symbol,
                level.timeframe,
                level.side.value,
                level.level_type.value,
                level.price,
            ),
        )

        deduplicated: list[EqualLevel] = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            previous = deduplicated[-1]

            same_scope = (
                level.exchange == previous.exchange
                and level.market_type == previous.market_type
                and level.symbol == previous.symbol
                and level.timeframe == previous.timeframe
            )
            same_side = level.side == previous.side
            same_type = level.level_type == previous.level_type
            tolerance = max(level.tolerance_pct, previous.tolerance_pct)

            is_duplicate = (
                same_scope
                and same_side
                and same_type
                and pct_distance(level.price, previous.price) <= tolerance
            )

            if is_duplicate:
                if self._is_better_level(level, previous):
                    deduplicated[-1] = level
                continue

            deduplicated.append(level)

        return deduplicated

    def _is_better_level(
        self,
        candidate: EqualLevel,
        current: EqualLevel,
    ) -> bool:
        """
        Визначає, який рівень краще залишити при deduplication.
        """
        candidate_score = (
            candidate.confidence,
            candidate.touches_count,
            candidate.reaction_count,
            -candidate.cluster_width,
        )
        current_score = (
            current.confidence,
            current.touches_count,
            current.reaction_count,
            -current.cluster_width,
        )

        return candidate_score > current_score

    def _resolve_tolerance_pct(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        current_price: float | None,
    ) -> float:
        """
        Обирає final tolerance:
        - базовий equal_level_tolerance_pct;
        - optional ATR-based adaptive tolerance.
        """
        base_tolerance = self._config.equal_level_tolerance_pct

        if not self._config.use_atr_tolerance:
            return base_tolerance

        if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
            return base_tolerance

        atr_pct = calculate_atr_pct_from_ohlc(
            highs=highs,
            lows=lows,
            closes=closes,
            period=self._config.atr_period,
        )

        reference_price = current_price or closes[-1]
        if reference_price is None or reference_price <= 0:
            return base_tolerance

        adaptive_tolerance = max(
            base_tolerance,
            atr_pct * self._config.atr_tolerance_multiplier,
        )

        return clamp(
            adaptive_tolerance,
            self._config.min_atr_tolerance_pct,
            self._config.max_atr_tolerance_pct,
        )

    # ------------------------------------------------------------------
    # Validation / parsing helpers
    # ------------------------------------------------------------------

    def _validate_ohlc(
        self,
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
    ) -> None:
        if not highs or not lows or not closes:
            raise ValueError("OHLC series must not be empty")

        if len(highs) != len(lows) or len(lows) != len(closes):
            raise ValueError("OHLC series must have the same length")

        for index, (high, low, close) in enumerate(zip(highs, lows, closes)):
            if high <= 0 or low <= 0 or close <= 0:
                raise ValueError(
                    f"OHLC values must be positive at index={index}: "
                    f"high={high}, low={low}, close={close}"
                )

            if high < low:
                raise ValueError(
                    f"Candle high must be >= low at index={index}: "
                    f"high={high}, low={low}"
                )

    def _resolve_current_price(
        self,
        current_price: float | None,
        closes: Sequence[float],
    ) -> float | None:
        if current_price is not None:
            resolved = safe_float(current_price)
            return resolved if resolved > 0 else None

        if not closes:
            return None

        last_close = safe_float(closes[-1])
        return last_close if last_close > 0 else None

    def _safe_optional_volume(self, candle: Any) -> float | None:
        volume = safe_float(get_candle_volume(candle), default=0.0)
        return volume if volume > 0 else None

    def _normalize_timestamp(
        self,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_scope_value(value: Any, default: str) -> str:
        normalized = str(value or default).strip()
        return normalized if normalized else default

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_timeframe(value: Any) -> str:
        return str(value or "").strip()