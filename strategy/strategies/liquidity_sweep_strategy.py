from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics.liquidity.enums import LiquidityBias, LiquiditySide, SweepStatus
from analytics.liquidity.liquidity_service import LiquidityTopics
from analytics.liquidity.models import LiquidityLevel, LiquidityMapSnapshot, StopCluster
from core.logger import get_logger


class LiquiditySweepStrategyTopics:
    SIGNAL_GENERATED = "signal.generated"


@dataclass(slots=True)
class LiquiditySweepStrategyConfig:
    enabled: bool = True

    # Які sweep-и беремо в роботу
    min_swept_level_confidence: float = 0.45
    allow_partial_sweeps: bool = True

    # Підтвердження reclaim після sweep
    reclaim_distance_pct: float = 0.0006
    stop_buffer_pct: float = 0.0012

    # Фільтри якості
    min_signal_confidence: float = 0.58
    min_cluster_confidence: float = 0.35
    min_opposite_liquidity_score: float = 0.20

    # Час життя pending sweep
    pending_sweep_ttl_seconds: int = 180

    # Захист від дублікатів
    cooldown_seconds: int = 45

    # Якщо True — шортимо після sweep buy-side liquidity
    # і лонгуємо після sweep sell-side liquidity
    contrarian_reversal_mode: bool = True

    # Якщо True — подія одразу емітиться в event bus
    emit_signals: bool = True
    signal_topic: str = LiquiditySweepStrategyTopics.SIGNAL_GENERATED

    def validate(self) -> None:
        if not 0.0 <= self.min_swept_level_confidence <= 1.0:
            raise ValueError("min_swept_level_confidence must be between 0 and 1")
        if not 0.0 <= self.min_signal_confidence <= 1.0:
            raise ValueError("min_signal_confidence must be between 0 and 1")
        if not 0.0 <= self.min_cluster_confidence <= 1.0:
            raise ValueError("min_cluster_confidence must be between 0 and 1")
        if not 0.0 <= self.min_opposite_liquidity_score <= 1.0:
            raise ValueError("min_opposite_liquidity_score must be between 0 and 1")
        if self.reclaim_distance_pct <= 0:
            raise ValueError("reclaim_distance_pct must be > 0")
        if self.stop_buffer_pct <= 0:
            raise ValueError("stop_buffer_pct must be > 0")
        if self.pending_sweep_ttl_seconds < 1:
            raise ValueError("pending_sweep_ttl_seconds must be >= 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")


@dataclass(slots=True)
class PendingSweep:
    symbol: str
    timeframe: str
    side: LiquiditySide
    price: float
    confidence: float
    sweep_status: SweepStatus
    level_type: str
    detected_at: datetime
    source_level: LiquidityLevel

    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.side.value}:{self.level_type}:{round(self.price, 8)}"

    def is_expired(self, now: datetime, ttl_seconds: int) -> bool:
        return (now - self.detected_at).total_seconds() > ttl_seconds


@dataclass(slots=True)
class LiquiditySweepSignal:
    strategy_name: str
    symbol: str
    timeframe: str
    side: str  # LONG | SHORT
    signal_type: str
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    confidence: float
    timestamp: datetime
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LiquiditySweepStrategy:
    """
    Стратегія для сценарію:
    1) liquidity level swept
    2) price reclaims level назад
    3) strategy forms reversal trade toward opposite liquidity

    Основна логіка:
    - buy-side sweep -> шукаємо reclaim вниз -> SHORT
    - sell-side sweep -> шукаємо reclaim вгору -> LONG
    """

    name = "liquidity_sweep_strategy"

    def __init__(
        self,
        event_bus: Any,
        config: LiquiditySweepStrategyConfig | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._config = config or LiquiditySweepStrategyConfig()
        self._config.validate()

        self._logger = get_logger(__name__, service_name="strategy.liquidity_sweep")

        self._pending_sweeps: dict[str, PendingSweep] = {}
        self._last_signal_at: dict[str, datetime] = {}
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            self._logger.warning("LiquiditySweepStrategy is already running")
            return

        if self._event_bus is None:
            raise ValueError("event_bus is required")

        await self._event_bus.subscribe(
            LiquidityTopics.LIQUIDITY_LEVEL_SWEPT,
            self._on_liquidity_level_swept,
        )
        await self._event_bus.subscribe(
            LiquidityTopics.LIQUIDITY_MAP_UPDATED,
            self._on_liquidity_map_updated,
        )

        self._running = True
        self._logger.info("LiquiditySweepStrategy started")

    async def stop(self) -> None:
        if not self._running:
            self._logger.warning("LiquiditySweepStrategy is not running")
            return

        unsubscribe = getattr(self._event_bus, "unsubscribe", None)
        if unsubscribe is not None:
            await unsubscribe(
                LiquidityTopics.LIQUIDITY_LEVEL_SWEPT,
                self._on_liquidity_level_swept,
            )
            await unsubscribe(
                LiquidityTopics.LIQUIDITY_MAP_UPDATED,
                self._on_liquidity_map_updated,
            )

        self._running = False
        self._pending_sweeps.clear()

        self._logger.info("LiquiditySweepStrategy stopped")

    async def _on_liquidity_level_swept(self, event: Any) -> None:
        payload = self._unwrap_payload(event)
        if not isinstance(payload, LiquidityLevel):
            self._logger.debug("Skip swept event: payload is not LiquidityLevel")
            return

        level = payload

        if not self._is_valid_swept_level(level):
            return

        now = self._extract_timestamp(event) or datetime.utcnow()

        pending = PendingSweep(
            symbol=level.symbol,
            timeframe=level.timeframe,
            side=level.side,
            price=float(level.price),
            confidence=float(level.confidence),
            sweep_status=level.sweep_status,
            level_type=level.level_type.value,
            detected_at=now,
            source_level=level,
        )

        async with self._lock:
            self._cleanup_expired(now)
            self._pending_sweeps[pending.key()] = pending

        self._logger.info(
            "Pending liquidity sweep registered",
            extra={
                "symbol": level.symbol,
                "timeframe": level.timeframe,
                "side": level.side.value,
                "price": level.price,
                "confidence": level.confidence,
                "sweep_status": level.sweep_status.value,
                "level_type": level.level_type.value,
            },
        )

    async def _on_liquidity_map_updated(self, event: Any) -> None:
        payload = self._unwrap_payload(event)
        if not isinstance(payload, LiquidityMapSnapshot):
            self._logger.debug("Skip map update: payload is not LiquidityMapSnapshot")
            return

        snapshot = payload
        now = self._extract_timestamp(event) or snapshot.timestamp or datetime.utcnow()

        async with self._lock:
            self._cleanup_expired(now)
            relevant = [
                sweep
                for sweep in self._pending_sweeps.values()
                if sweep.symbol == snapshot.symbol and sweep.timeframe == snapshot.timeframe
            ]

        if not relevant:
            return

        for sweep in relevant:
            signal = self._evaluate_pending_sweep(snapshot=snapshot, sweep=sweep, now=now)
            if signal is None:
                continue

            emitted = await self._emit_signal(signal)
            if emitted:
                async with self._lock:
                    self._pending_sweeps.pop(sweep.key(), None)

    def evaluate_snapshot(
        self,
        snapshot: LiquidityMapSnapshot,
        swept_level: LiquidityLevel,
        now: datetime | None = None,
    ) -> LiquiditySweepSignal | None:
        """
        Синхронний helper, якщо хочеш викликати strategy напряму без EventBus.
        """
        now = now or datetime.utcnow()

        if not self._is_valid_swept_level(swept_level):
            return None

        pending = PendingSweep(
            symbol=swept_level.symbol,
            timeframe=swept_level.timeframe,
            side=swept_level.side,
            price=float(swept_level.price),
            confidence=float(swept_level.confidence),
            sweep_status=swept_level.sweep_status,
            level_type=swept_level.level_type.value,
            detected_at=now,
            source_level=swept_level,
        )
        return self._evaluate_pending_sweep(snapshot=snapshot, sweep=pending, now=now)

    def _evaluate_pending_sweep(
        self,
        snapshot: LiquidityMapSnapshot,
        sweep: PendingSweep,
        now: datetime,
    ) -> LiquiditySweepSignal | None:
        if not self._config.enabled:
            return None

        if snapshot.current_price <= 0:
            return None

        if self._is_on_cooldown(snapshot.symbol, snapshot.timeframe, now):
            return None

        current_price = float(snapshot.current_price)

        if sweep.side == LiquiditySide.BUY_SIDE:
            if not self._config.contrarian_reversal_mode:
                return None
            return self._evaluate_short_after_buy_side_sweep(snapshot, sweep, current_price, now)

        if sweep.side == LiquiditySide.SELL_SIDE:
            if not self._config.contrarian_reversal_mode:
                return None
            return self._evaluate_long_after_sell_side_sweep(snapshot, sweep, current_price, now)

        return None

    def _evaluate_short_after_buy_side_sweep(
        self,
        snapshot: LiquidityMapSnapshot,
        sweep: PendingSweep,
        current_price: float,
        now: datetime,
    ) -> LiquiditySweepSignal | None:
        reclaim_threshold = sweep.price * (1.0 + self._config.reclaim_distance_pct)

        # Після sweep buy-side liquidity ціна повинна повернутися назад під рівень/reclaim threshold
        if current_price > reclaim_threshold:
            return None

        opposite_target = self._resolve_short_target(snapshot)
        opposite_score = max(snapshot.below_liquidity_score, snapshot.signal.magnet_score_down if snapshot.signal else 0.0)

        if opposite_score < self._config.min_opposite_liquidity_score:
            return None

        cluster_bonus = self._cluster_bonus(snapshot.strongest_cluster_below)
        bias_bonus = self._bias_bonus(snapshot.bias, expected=LiquidityBias.DOWN)

        confidence = self._clamp01(
            0.35 * sweep.confidence
            + 0.25 * opposite_score
            + 0.20 * cluster_bonus
            + 0.20 * bias_bonus
        )

        if confidence < self._config.min_signal_confidence:
            return None

        stop_loss = sweep.price * (1.0 + self._config.stop_buffer_pct)
        take_profit = opposite_target

        reason = (
            f"buy-side liquidity swept at {sweep.price:.8f}, "
            f"price reclaimed below level, expecting reversal toward sell-side liquidity"
        )

        return LiquiditySweepSignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side="SHORT",
            signal_type="liquidity_sweep_reversal",
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=now,
            reason=reason,
            metadata={
                "swept_side": sweep.side.value,
                "swept_price": sweep.price,
                "swept_confidence": sweep.confidence,
                "sweep_status": sweep.sweep_status.value,
                "snapshot_bias": snapshot.bias.value,
                "below_liquidity_score": snapshot.below_liquidity_score,
                "above_liquidity_score": snapshot.above_liquidity_score,
                "pressure_score": snapshot.liquidity_pressure_score,
                "magnet_score_down": snapshot.signal.magnet_score_down if snapshot.signal else None,
                "strongest_cluster_below_confidence": (
                    snapshot.strongest_cluster_below.confidence
                    if snapshot.strongest_cluster_below is not None
                    else None
                ),
            },
        )

    def _evaluate_long_after_sell_side_sweep(
        self,
        snapshot: LiquidityMapSnapshot,
        sweep: PendingSweep,
        current_price: float,
        now: datetime,
    ) -> LiquiditySweepSignal | None:
        reclaim_threshold = sweep.price * (1.0 - self._config.reclaim_distance_pct)

        # Після sweep sell-side liquidity ціна повинна повернутися назад над рівень/reclaim threshold
        if current_price < reclaim_threshold:
            return None

        opposite_target = self._resolve_long_target(snapshot)
        opposite_score = max(snapshot.above_liquidity_score, snapshot.signal.magnet_score_up if snapshot.signal else 0.0)

        if opposite_score < self._config.min_opposite_liquidity_score:
            return None

        cluster_bonus = self._cluster_bonus(snapshot.strongest_cluster_above)
        bias_bonus = self._bias_bonus(snapshot.bias, expected=LiquidityBias.UP)

        confidence = self._clamp01(
            0.35 * sweep.confidence
            + 0.25 * opposite_score
            + 0.20 * cluster_bonus
            + 0.20 * bias_bonus
        )

        if confidence < self._config.min_signal_confidence:
            return None

        stop_loss = sweep.price * (1.0 - self._config.stop_buffer_pct)
        take_profit = opposite_target

        reason = (
            f"sell-side liquidity swept at {sweep.price:.8f}, "
            f"price reclaimed above level, expecting reversal toward buy-side liquidity"
        )

        return LiquiditySweepSignal(
            strategy_name=self.name,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            side="LONG",
            signal_type="liquidity_sweep_reversal",
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=now,
            reason=reason,
            metadata={
                "swept_side": sweep.side.value,
                "swept_price": sweep.price,
                "swept_confidence": sweep.confidence,
                "sweep_status": sweep.sweep_status.value,
                "snapshot_bias": snapshot.bias.value,
                "below_liquidity_score": snapshot.below_liquidity_score,
                "above_liquidity_score": snapshot.above_liquidity_score,
                "pressure_score": snapshot.liquidity_pressure_score,
                "magnet_score_up": snapshot.signal.magnet_score_up if snapshot.signal else None,
                "strongest_cluster_above_confidence": (
                    snapshot.strongest_cluster_above.confidence
                    if snapshot.strongest_cluster_above is not None
                    else None
                ),
            },
        )

    async def _emit_signal(self, signal: LiquiditySweepSignal) -> bool:
        symbol_tf_key = f"{signal.symbol}:{signal.timeframe}"
        now = signal.timestamp

        if self._is_on_cooldown(signal.symbol, signal.timeframe, now):
            return False

        self._last_signal_at[symbol_tf_key] = now

        self._logger.info(
            "Liquidity sweep signal generated",
            extra={
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "side": signal.side,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "signal_type": signal.signal_type,
            },
        )

        if self._config.emit_signals and self._event_bus is not None:
            await self._event_bus.emit(self._config.signal_topic, signal)

        return True

    def _resolve_short_target(self, snapshot: LiquidityMapSnapshot) -> float | None:
        target = snapshot.nearest_below_level
        if target is not None:
            return self._extract_target_price(target)

        if snapshot.strongest_cluster_below is not None:
            return snapshot.strongest_cluster_below.center_price

        return None

    def _resolve_long_target(self, snapshot: LiquidityMapSnapshot) -> float | None:
        target = snapshot.nearest_above_level
        if target is not None:
            return self._extract_target_price(target)

        if snapshot.strongest_cluster_above is not None:
            return snapshot.strongest_cluster_above.center_price

        return None

    def _extract_target_price(self, obj: LiquidityLevel | StopCluster) -> float:
        if isinstance(obj, LiquidityLevel):
            return float(obj.price)
        return float(obj.center_price)

    def _cluster_bonus(self, cluster: StopCluster | None) -> float:
        if cluster is None:
            return 0.0
        if cluster.confidence < self._config.min_cluster_confidence:
            return 0.0
        return self._clamp01(cluster.confidence)

    def _bias_bonus(self, actual: LiquidityBias, expected: LiquidityBias) -> float:
        if actual == expected:
            return 1.0
        if actual == LiquidityBias.NEUTRAL:
            return 0.45
        return 0.0

    def _is_valid_swept_level(self, level: LiquidityLevel) -> bool:
        if level.confidence < self._config.min_swept_level_confidence:
            return False

        if level.sweep_status == SweepStatus.SWEPT:
            return True

        if level.sweep_status == SweepStatus.PARTIALLY_SWEPT and self._config.allow_partial_sweeps:
            return True

        return False

    def _is_on_cooldown(self, symbol: str, timeframe: str, now: datetime) -> bool:
        key = f"{symbol}:{timeframe}"
        last_ts = self._last_signal_at.get(key)
        if last_ts is None:
            return False
        return (now - last_ts).total_seconds() < self._config.cooldown_seconds

    def _cleanup_expired(self, now: datetime) -> None:
        expired_keys = [
            key
            for key, sweep in self._pending_sweeps.items()
            if sweep.is_expired(now, self._config.pending_sweep_ttl_seconds)
        ]
        for key in expired_keys:
            self._pending_sweeps.pop(key, None)

    def _unwrap_payload(self, event: Any) -> Any:
        if hasattr(event, "payload"):
            return event.payload
        return event

    def _extract_timestamp(self, event: Any) -> datetime | None:
        if hasattr(event, "timestamp"):
            ts = getattr(event, "timestamp")
            if isinstance(ts, datetime):
                return ts

        payload = self._unwrap_payload(event)
        if hasattr(payload, "timestamp"):
            ts = getattr(payload, "timestamp")
            if isinstance(ts, datetime):
                return ts

        if isinstance(payload, dict):
            ts = payload.get("timestamp")
            if isinstance(ts, datetime):
                return ts

        return None

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))