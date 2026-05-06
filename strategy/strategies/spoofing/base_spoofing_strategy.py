from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from core.event_bus import EventBus, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.spoofing import (
    SpoofingFeatures,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingSignal,
    SpoofingType,
)


class SetupStatus(str, Enum):
    """
    Стан setup-а в lifecycle strategy-рівня.
    """

    PENDING = "pending"
    CONFIRMED = "confirmed"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class StrategyDirection(str, Enum):
    """
    Напрямок торгової ідеї.
    """

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(slots=True)
class BaseSpoofingStrategyConfig:
    """
    Базовий конфіг для spoofing-стратегій.
    """

    enabled: bool = True

    # EventBus topics
    spoofing_detected_topic: str = "analytics.spoofing.detected"
    spoofing_updated_topic: str = "analytics.spoofing.updated"
    market_trade_topic: str = "market.trade"
    market_orderbook_topic: str = "market.orderbook"

    strategy_signal_topic: str = "signal.generated"
    strategy_setup_topic: str = "strategy.spoofing.setup_created"
    strategy_update_topic: str = "strategy.spoofing.setup_updated"
    strategy_invalidation_topic: str = "strategy.spoofing.setup_invalidated"
    strategy_expired_topic: str = "strategy.spoofing.setup_expired"

    # filtering
    min_score: float = 0.65
    min_confidence: float = 0.55
    allowed_severities: tuple[SpoofingSeverity, ...] = (
        SpoofingSeverity.MEDIUM,
        SpoofingSeverity.HIGH,
        SpoofingSeverity.CRITICAL,
    )

    # lifecycle
    setup_ttl_ms: int = 8_000
    cooldown_ms_same_symbol_pattern: int = 10_000
    max_active_setups_per_symbol: int = 8

    # confirmation
    require_confirmation: bool = True
    min_confirmation_move_bps: float = 1.0
    max_adverse_move_bps: float = 2.5

    # risk-ish defaults, not full RiskManager responsibility
    default_entry_offset_bps: float = 0.0
    default_stop_buffer_bps: float = 3.0
    default_take_profit_bps: float = 6.0
    rr_multiplier: float = 2.0

    # behavior
    process_updates_for_existing_signal: bool = True
    allow_multiple_setups_same_symbol: bool = True
    publish_setup_events: bool = True
    publish_debug_updates: bool = True

    # cleanup
    cleanup_interval_ms: int = 2_000
    cleanup_timeout_sec: float = 1.0
    cleanup_max_retries: int = 1
    cleanup_retry_delay_sec: float = 0.5


@dataclass(slots=True)
class SpoofingTradeSetup:
    """
    Внутрішня модель trade setup, який створюється на основі SpoofingSignal.
    """

    setup_id: str
    source_signal_id: str

    symbol: str
    exchange: str

    strategy_name: str
    direction: StrategyDirection

    spoofing_type: SpoofingType
    pattern: SpoofingPattern
    severity: SpoofingSeverity

    score: float
    confidence: float

    reference_price: float
    entry_price: float
    stop_price: float
    take_profit_price: float

    signal_side: SpoofingSide
    detected_at: datetime
    expires_at: datetime

    status: SetupStatus = SetupStatus.PENDING
    confirmed_at: datetime | None = None
    triggered_at: datetime | None = None
    invalidated_at: datetime | None = None
    expired_at: datetime | None = None

    confirmation_price: float | None = None
    last_price: float | None = None

    features: SpoofingFeatures | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in {SetupStatus.PENDING, SetupStatus.CONFIRMED}

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            SetupStatus.TRIGGERED,
            SetupStatus.EXPIRED,
            SetupStatus.INVALIDATED,
            SetupStatus.CANCELLED,
            SetupStatus.REJECTED,
        }


class BaseSpoofingStrategy(ABC):
    """
    Базовий клас для всіх strategy/strategies/spoofing/* стратегій.

    Призначення:
    - слухати spoofing analytics events;
    - створювати setup-и на основі SpoofingSignal;
    - трекати lifecycle setup-ів;
    - підтверджувати / інвалідовувати / експайрити setup-и;
    - публікувати normalized strategy signal у strategy layer.

    Важливо:
    - цей клас НЕ займається spoofing detection;
    - він працює поверх уже готових SpoofingSignal від analytics.spoofing;
    - конкретні rules для entry/filtering визначаються у subclass.
    """

    strategy_name: str = "base_spoofing_strategy"

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        config: BaseSpoofingStrategyConfig | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config or BaseSpoofingStrategyConfig()

        self.logger = get_logger(
            __name__,
            service_name=f"strategy.spoofing.{self.strategy_name}",
            event_type="strategy",
            strategy=self.strategy_name,
        )

        self._active_setups_by_id: dict[str, SpoofingTradeSetup] = {}
        self._setup_id_by_signal_id: dict[str, str] = {}
        self._setup_ids_by_symbol: dict[str, set[str]] = {}
        self._cooldowns: dict[tuple[str, str, str], datetime] = {}

        self._latest_price_by_symbol: dict[str, float] = {}
        self._latest_trade_ts_by_symbol: dict[str, datetime] = {}

        self._registered: bool = False
        self._subscriptions: list[Subscription] = []

        self._cleanup_job_id: str | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running: bool = False

        self._stats: dict[str, int] = {
            "signals_received": 0,
            "setups_created": 0,
            "setups_confirmed": 0,
            "setups_triggered": 0,
            "setups_invalidated": 0,
            "setups_expired": 0,
            "signals_emitted": 0,
            "rejected_by_filter": 0,
            "duplicate_signals": 0,
            "cooldown_blocks": 0,
            "errors": 0,
            "emit_failures": 0,
        }

    # -------------------------------------------------------------------------
    # lifecycle
    # -------------------------------------------------------------------------

    def register(self) -> None:
        """
        Підписка на core EventBus.

        Idempotent: повторний виклик не створює дублікати subscriptions.
        """
        if self._registered:
            self.log_debug("register skipped: already registered")
            return

        if self.event_bus is None:
            self.log_warning("register skipped: event_bus is None")
            return

        self._subscriptions.append(
            self.event_bus.subscribe(
                self.config.spoofing_detected_topic,
                self.on_spoofing_detected,
                name=f"{self.strategy_name}.on_spoofing_detected",
            )
        )
        self._subscriptions.append(
            self.event_bus.subscribe(
                self.config.spoofing_updated_topic,
                self.on_spoofing_updated,
                name=f"{self.strategy_name}.on_spoofing_updated",
            )
        )
        self._subscriptions.append(
            self.event_bus.subscribe(
                self.config.market_trade_topic,
                self.on_market_trade,
                name=f"{self.strategy_name}.on_market_trade",
            )
        )

        self._registered = True

        self.log_info(
            "Strategy registered",
            strategy_name=self.strategy_name,
            spoofing_detected_topic=self.config.spoofing_detected_topic,
            spoofing_updated_topic=self.config.spoofing_updated_topic,
            market_trade_topic=self.config.market_trade_topic,
        )

    def unregister(self) -> None:
        """
        Знімає subscriptions з core EventBus.
        """
        if not self._registered:
            return

        if self.event_bus is not None:
            for subscription in self._subscriptions:
                try:
                    self.event_bus.unsubscribe(subscription)
                except Exception:
                    self.log_exception(
                        "Failed to unsubscribe strategy handler",
                        strategy_name=self.strategy_name,
                    )

        self._subscriptions.clear()
        self._registered = False

        self.log_info("Strategy unregistered", strategy_name=self.strategy_name)

    async def start(self) -> None:
        """
        Async lifecycle.

        Subscriptions йдуть через core EventBus.
        Periodic cleanup запускається через core Scheduler, якщо scheduler переданий.
        Якщо Scheduler не передано, використовується fallback asyncio task.
        """
        if self._is_running:
            return

        if not self._registered:
            self.register()

        self._is_running = True

        if self.config.cleanup_interval_ms > 0:
            interval_sec = max(self.config.cleanup_interval_ms / 1000.0, 0.25)

            if self.scheduler is not None:
                self._cleanup_job_id = self.scheduler.add_interval_job(
                    name=f"{self.strategy_name}.cleanup",
                    func=self.run_cleanup_once,
                    interval=interval_sec,
                    run_immediately=False,
                    max_retries=self.config.cleanup_max_retries,
                    retry_delay=self.config.cleanup_retry_delay_sec,
                    timeout=self.config.cleanup_timeout_sec,
                    allow_overlap=False,
                    enabled=True,
                )
            else:
                self.log_warning(
                    "Scheduler is None; using fallback cleanup task",
                    strategy_name=self.strategy_name,
                )
                self._cleanup_task = asyncio.create_task(
                    self._cleanup_loop(),
                    name=f"{self.strategy_name}-cleanup-loop",
                )

        self.log_info("Strategy started", strategy_name=self.strategy_name)

    async def stop(self) -> None:
        """
        Акуратно зупиняє runtime lifecycle.
        """
        self._is_running = False

        if self.scheduler is not None and self._cleanup_job_id is not None:
            try:
                self.scheduler.remove_job(self._cleanup_job_id)
            except KeyError:
                pass
            except Exception:
                self.log_exception(
                    "Failed to remove cleanup scheduler job",
                    strategy_name=self.strategy_name,
                    cleanup_job_id=self._cleanup_job_id,
                )
            finally:
                self._cleanup_job_id = None

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            finally:
                self._cleanup_task = None

        self.unregister()

        self.log_info("Strategy stopped", strategy_name=self.strategy_name)

    # -------------------------------------------------------------------------
    # event handlers
    # -------------------------------------------------------------------------

    async def on_spoofing_detected(self, event: Any) -> None:
        """
        Головна точка входу для нових spoofing-сигналів.
        """
        self._stats["signals_received"] += 1

        signal = self._extract_spoofing_signal(event)
        if signal is None:
            self.log_warning("Received invalid spoofing detected event")
            return

        try:
            if not self.config.enabled:
                self.log_debug(
                    "Strategy disabled, detected signal ignored",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                )
                return

            if not self.accepts_signal(signal):
                self._stats["rejected_by_filter"] += 1
                self.log_debug(
                    "Signal rejected by accepts_signal",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    spoofing_type=signal.spoofing_type.value,
                    pattern=signal.pattern.value,
                    score=signal.score,
                    confidence=signal.confidence,
                )
                return

            if self._has_active_setup_for_signal(signal.signal_id):
                self._stats["duplicate_signals"] += 1
                self.log_debug(
                    "Duplicate signal ignored: setup already exists",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                )
                return

            if self._is_cooldown_active(signal):
                self._stats["cooldown_blocks"] += 1
                self.log_debug(
                    "Signal blocked by cooldown",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    pattern=signal.pattern.value,
                )
                return

            if not self._can_create_more_setups_for_symbol(signal.symbol):
                self.log_debug(
                    "Max active setups for symbol reached",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                )
                return

            setup = self.build_setup(signal)
            if setup is None:
                self._stats["rejected_by_filter"] += 1
                self.log_debug(
                    "build_setup returned None",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                )
                return

            self._store_setup(setup)
            self._stats["setups_created"] += 1
            self._set_cooldown(signal)

            self.log_info(
                "Spoofing setup created",
                strategy_name=self.strategy_name,
                setup_id=setup.setup_id,
                signal_id=setup.source_signal_id,
                symbol=setup.symbol,
                direction=setup.direction.value,
                pattern=setup.pattern.value,
                spoofing_type=setup.spoofing_type.value,
                score=setup.score,
                confidence=setup.confidence,
                entry_price=setup.entry_price,
                stop_price=setup.stop_price,
                take_profit_price=setup.take_profit_price,
            )

            if self.config.publish_setup_events:
                await self._publish_setup_created(setup)

            if not self.config.require_confirmation:
                confirmed = self.confirm_setup(
                    setup=setup,
                    current_price=setup.reference_price,
                    signal=signal,
                )
                if confirmed:
                    await self._emit_strategy_signal(setup)

        except Exception:
            self._stats["errors"] += 1
            self.log_exception(
                "Failed to handle spoofing detected event",
                signal_id=getattr(signal, "signal_id", None),
                symbol=getattr(signal, "symbol", None),
            )
            raise

    async def on_spoofing_updated(self, event: Any) -> None:
        """
        Обробка updated spoofing signal.
        """
        if not self.config.process_updates_for_existing_signal:
            return

        signal = self._extract_spoofing_signal(event)
        if signal is None:
            return

        setup = self.get_setup_by_signal_id(signal.signal_id)
        if setup is None or setup.is_terminal:
            return

        try:
            self.apply_signal_update(setup=setup, signal=signal)

            if self.should_invalidate_from_signal_update(setup=setup, signal=signal):
                await self.invalidate_setup(
                    setup,
                    reason="signal_update_invalidation",
                    metadata={
                        "signal_id": signal.signal_id,
                        "score": signal.score,
                        "confidence": signal.confidence,
                    },
                )
                return

            if setup.status == SetupStatus.PENDING:
                latest_price = self._latest_price_by_symbol.get(
                    setup.symbol,
                    setup.reference_price,
                )
                if self.confirm_setup(setup=setup, current_price=latest_price, signal=signal):
                    await self._emit_strategy_signal(setup)

            if self.config.publish_setup_events and self.config.publish_debug_updates:
                await self._publish_setup_updated(setup, reason="signal_update")

        except Exception:
            self._stats["errors"] += 1
            self.log_exception(
                "Failed to handle spoofing updated event",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
            )
            raise

    async def on_market_trade(self, event: Any) -> None:
        """
        Обробка market.trade для confirmation/invalidation setup-ів.

        Очікує payload dict:
        {
            "symbol": "...",
            "price": 123.45,
            ...
        }
        """
        payload = getattr(event, "payload", event)
        if not isinstance(payload, dict):
            return

        symbol = payload.get("symbol")
        price = self._safe_float(payload.get("price"))
        if not symbol or price <= 0:
            return

        now = self._extract_event_timestamp(event, fallback=self.now())

        self._latest_price_by_symbol[symbol] = price
        self._latest_trade_ts_by_symbol[symbol] = now

        active_setups = self.get_active_setups_for_symbol(symbol)
        if not active_setups:
            return

        for setup in list(active_setups):
            if setup.is_terminal:
                continue

            setup.last_price = price

            if self.is_setup_expired(setup, now=now):
                await self.expire_setup(setup, reason="ttl_expired_on_trade")
                continue

            if self.should_invalidate_on_price(setup=setup, current_price=price):
                await self.invalidate_setup(
                    setup,
                    reason="price_invalidation",
                    metadata={"current_price": price},
                )
                continue

            if setup.status == SetupStatus.PENDING:
                signal = self.rebuild_minimal_signal_from_setup(setup)
                if self.confirm_setup(setup=setup, current_price=price, signal=signal):
                    await self._emit_strategy_signal(setup)
                    continue

            if setup.status == SetupStatus.CONFIRMED and self.should_trigger_entry(
                setup=setup,
                current_price=price,
            ):
                await self._emit_strategy_signal(setup)
                continue

    # -------------------------------------------------------------------------
    # abstract / extension hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def supports_pattern(self, signal: SpoofingSignal) -> bool:
        raise NotImplementedError

    def accepts_signal(self, signal: SpoofingSignal) -> bool:
        if signal.score < self.config.min_score:
            return False
        if signal.confidence < self.config.min_confidence:
            return False
        if signal.severity not in self.config.allowed_severities:
            return False
        if not self.supports_pattern(signal):
            return False
        return True

    def build_setup(self, signal: SpoofingSignal) -> SpoofingTradeSetup | None:
        direction = self.resolve_direction(signal)
        if direction == StrategyDirection.FLAT:
            return None

        reference_price = self.resolve_reference_price(signal)
        if reference_price <= 0:
            return None

        entry_price = self.compute_entry_price(signal, direction, reference_price)
        stop_price = self.compute_stop_price(signal, direction, reference_price)
        take_profit_price = self.compute_take_profit_price(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            reference_price=reference_price,
        )

        now = self.now()
        expires_at = now + timedelta(milliseconds=self.config.setup_ttl_ms)

        setup = SpoofingTradeSetup(
            setup_id=self.build_setup_id(signal),
            source_signal_id=signal.signal_id,
            symbol=signal.symbol,
            exchange=signal.exchange,
            strategy_name=self.strategy_name,
            direction=direction,
            spoofing_type=signal.spoofing_type,
            pattern=signal.pattern,
            severity=signal.severity,
            score=signal.score,
            confidence=signal.confidence,
            reference_price=reference_price,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            signal_side=signal.side,
            detected_at=now,
            expires_at=expires_at,
            status=SetupStatus.PENDING,
            features=signal.features,
            metadata={
                "signal_id": signal.signal_id,
                "pattern": signal.pattern.value,
                "spoofing_type": signal.spoofing_type.value,
                "severity": signal.severity.value,
                "price_level": signal.price_level,
                "source_detected_at": self._dt_to_iso(signal.detected_at),
            },
        )

        self.enrich_setup(setup, signal)
        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        return None

    def apply_signal_update(self, *, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        setup.score = max(setup.score, signal.score)
        setup.confidence = max(setup.confidence, signal.confidence)
        setup.severity = self._max_severity(setup.severity, signal.severity)
        setup.features = signal.features
        setup.metadata["last_signal_update_at"] = self._dt_to_iso(self.now())

    def should_invalidate_from_signal_update(
        self,
        *,
        setup: SpoofingTradeSetup,
        signal: SpoofingSignal,
    ) -> bool:
        return False

    def confirm_setup(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
        signal: SpoofingSignal,
    ) -> bool:
        if setup.status != SetupStatus.PENDING:
            return False

        move_bps = self.signed_bps_move(
            current_price=current_price,
            reference_price=setup.reference_price,
        )

        if setup.direction == StrategyDirection.LONG:
            passed = move_bps >= self.config.min_confirmation_move_bps
        elif setup.direction == StrategyDirection.SHORT:
            passed = move_bps <= -self.config.min_confirmation_move_bps
        else:
            passed = False

        if not passed:
            return False

        setup.status = SetupStatus.CONFIRMED
        setup.confirmed_at = self.now()
        setup.confirmation_price = current_price
        self._stats["setups_confirmed"] += 1

        self.log_info(
            "Setup confirmed",
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction.value,
            current_price=current_price,
            reference_price=setup.reference_price,
            move_bps=move_bps,
        )
        return True

    def should_trigger_entry(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> bool:
        return setup.status == SetupStatus.CONFIRMED

    def should_invalidate_on_price(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> bool:
        adverse_bps = self._compute_adverse_move_bps(
            setup=setup,
            current_price=current_price,
        )
        return adverse_bps >= self.config.max_adverse_move_bps

    # -------------------------------------------------------------------------
    # pricing helpers
    # -------------------------------------------------------------------------

    def resolve_direction(self, signal: SpoofingSignal) -> StrategyDirection:
        if signal.side == SpoofingSide.ASK:
            return StrategyDirection.LONG
        if signal.side == SpoofingSide.BID:
            return StrategyDirection.SHORT
        return StrategyDirection.FLAT

    def resolve_reference_price(self, signal: SpoofingSignal) -> float:
        if signal.price_level and signal.price_level > 0:
            return float(signal.price_level)

        if signal.features is not None:
            if getattr(signal.features, "reference_price", None):
                return float(signal.features.reference_price)

        return 0.0

    def compute_entry_price(
        self,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        reference_price: float,
    ) -> float:
        offset = self.config.default_entry_offset_bps / 10_000.0
        if direction == StrategyDirection.LONG:
            return reference_price * (1.0 + offset)
        if direction == StrategyDirection.SHORT:
            return reference_price * (1.0 - offset)
        return reference_price

    def compute_stop_price(
        self,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        reference_price: float,
    ) -> float:
        buffer_ratio = self.config.default_stop_buffer_bps / 10_000.0
        if direction == StrategyDirection.LONG:
            return reference_price * (1.0 - buffer_ratio)
        if direction == StrategyDirection.SHORT:
            return reference_price * (1.0 + buffer_ratio)
        return reference_price

    def compute_take_profit_price(
        self,
        *,
        signal: SpoofingSignal,
        direction: StrategyDirection,
        entry_price: float,
        stop_price: float,
        reference_price: float,
    ) -> float:
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            fallback_ratio = self.config.default_take_profit_bps / 10_000.0
            if direction == StrategyDirection.LONG:
                return reference_price * (1.0 + fallback_ratio)
            if direction == StrategyDirection.SHORT:
                return reference_price * (1.0 - fallback_ratio)
            return reference_price

        reward = risk * self.config.rr_multiplier
        if direction == StrategyDirection.LONG:
            return entry_price + reward
        if direction == StrategyDirection.SHORT:
            return entry_price - reward
        return entry_price

    # -------------------------------------------------------------------------
    # setup state helpers
    # -------------------------------------------------------------------------

    def get_setup(self, setup_id: str) -> SpoofingTradeSetup | None:
        return self._active_setups_by_id.get(setup_id)

    def get_setup_by_signal_id(self, signal_id: str) -> SpoofingTradeSetup | None:
        setup_id = self._setup_id_by_signal_id.get(signal_id)
        if setup_id is None:
            return None
        return self._active_setups_by_id.get(setup_id)

    def get_active_setups_for_symbol(self, symbol: str) -> list[SpoofingTradeSetup]:
        setup_ids = self._setup_ids_by_symbol.get(symbol, set())
        setups: list[SpoofingTradeSetup] = []

        for setup_id in setup_ids:
            setup = self._active_setups_by_id.get(setup_id)
            if setup is None:
                continue
            if setup.is_active:
                setups.append(setup)

        setups.sort(key=lambda item: item.detected_at, reverse=True)
        return setups

    def _store_setup(self, setup: SpoofingTradeSetup) -> None:
        self._active_setups_by_id[setup.setup_id] = setup
        self._setup_id_by_signal_id[setup.source_signal_id] = setup.setup_id
        self._setup_ids_by_symbol.setdefault(setup.symbol, set()).add(setup.setup_id)

    def _remove_setup(self, setup: SpoofingTradeSetup) -> None:
        self._active_setups_by_id.pop(setup.setup_id, None)
        self._setup_id_by_signal_id.pop(setup.source_signal_id, None)

        if setup.symbol in self._setup_ids_by_symbol:
            self._setup_ids_by_symbol[setup.symbol].discard(setup.setup_id)
            if not self._setup_ids_by_symbol[setup.symbol]:
                self._setup_ids_by_symbol.pop(setup.symbol, None)

    def _has_active_setup_for_signal(self, signal_id: str) -> bool:
        setup = self.get_setup_by_signal_id(signal_id)
        return setup is not None and setup.is_active

    def _can_create_more_setups_for_symbol(self, symbol: str) -> bool:
        active = self.get_active_setups_for_symbol(symbol)

        if self.config.allow_multiple_setups_same_symbol:
            return len(active) < self.config.max_active_setups_per_symbol

        return len(active) == 0

    def build_setup_id(self, signal: SpoofingSignal) -> str:
        return f"{self.strategy_name}:{signal.signal_id}"

    def is_setup_expired(
        self,
        setup: SpoofingTradeSetup,
        *,
        now: datetime | None = None,
    ) -> bool:
        ts = now or self.now()
        return ts >= setup.expires_at

    async def expire_setup(
        self,
        setup: SpoofingTradeSetup,
        *,
        reason: str,
    ) -> None:
        if setup.is_terminal:
            return

        setup.status = SetupStatus.EXPIRED
        setup.expired_at = self.now()
        setup.metadata["expire_reason"] = reason

        self._stats["setups_expired"] += 1

        self.log_info(
            "Setup expired",
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            reason=reason,
        )

        if self.config.publish_setup_events:
            await self._publish_setup_expired(setup, reason=reason)

        self._remove_setup(setup)

    async def invalidate_setup(
        self,
        setup: SpoofingTradeSetup,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if setup.is_terminal:
            return

        setup.status = SetupStatus.INVALIDATED
        setup.invalidated_at = self.now()
        setup.metadata["invalidation_reason"] = reason

        if metadata:
            setup.metadata.update(metadata)

        self._stats["setups_invalidated"] += 1

        self.log_info(
            "Setup invalidated",
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            reason=reason,
        )

        if self.config.publish_setup_events:
            await self._publish_setup_invalidated(setup, reason=reason)

        self._remove_setup(setup)

    async def cancel_setup(
        self,
        setup: SpoofingTradeSetup,
        *,
        reason: str,
    ) -> None:
        if setup.is_terminal:
            return

        setup.status = SetupStatus.CANCELLED
        setup.metadata["cancel_reason"] = reason

        self.log_info(
            "Setup cancelled",
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            reason=reason,
        )

        self._remove_setup(setup)

    # -------------------------------------------------------------------------
    # signal emission
    # -------------------------------------------------------------------------

    async def _emit_strategy_signal(self, setup: SpoofingTradeSetup) -> None:
        if setup.is_terminal:
            return

        if setup.status in {SetupStatus.PENDING, SetupStatus.CONFIRMED}:
            setup.status = SetupStatus.TRIGGERED
            setup.triggered_at = self.now()
            self._stats["setups_triggered"] += 1

        payload = self.build_strategy_signal_payload(setup)

        emitted = await self.emit_event(
            self.config.strategy_signal_topic,
            payload,
        )

        if emitted:
            self._stats["signals_emitted"] += 1

        self.log_info(
            "Strategy signal emitted",
            strategy_name=self.strategy_name,
            setup_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction.value,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            take_profit_price=setup.take_profit_price,
            emitted=emitted,
        )

        self._remove_setup(setup)

    def build_strategy_signal_payload(self, setup: SpoofingTradeSetup) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "strategy_type": "spoofing",
            "setup_id": setup.setup_id,
            "source_signal_id": setup.source_signal_id,
            "symbol": setup.symbol,
            "exchange": setup.exchange,
            "side": "BUY" if setup.direction == StrategyDirection.LONG else "SELL",
            "direction": setup.direction.value,
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "take_profit_price": setup.take_profit_price,
            "reference_price": setup.reference_price,
            "confidence": setup.confidence,
            "score": setup.score,
            "severity": setup.severity.value,
            "spoofing_type": setup.spoofing_type.value,
            "pattern": setup.pattern.value,
            "signal_side": setup.signal_side.value,
            "detected_at": self._dt_to_iso(setup.detected_at),
            "confirmed_at": self._dt_to_iso(setup.confirmed_at),
            "triggered_at": self._dt_to_iso(setup.triggered_at),
            "features": self._serialize_features(setup.features),
            "metadata": dict(setup.metadata),
        }

    # -------------------------------------------------------------------------
    # publishing helpers
    # -------------------------------------------------------------------------

    async def _publish_setup_created(self, setup: SpoofingTradeSetup) -> bool:
        return await self.emit_event(
            self.config.strategy_setup_topic,
            {
                "strategy": self.strategy_name,
                "event": "setup_created",
                "setup": self.serialize_setup(setup),
            },
        )

    async def _publish_setup_updated(self, setup: SpoofingTradeSetup, *, reason: str) -> bool:
        return await self.emit_event(
            self.config.strategy_update_topic,
            {
                "strategy": self.strategy_name,
                "event": "setup_updated",
                "reason": reason,
                "setup": self.serialize_setup(setup),
            },
        )

    async def _publish_setup_invalidated(self, setup: SpoofingTradeSetup, *, reason: str) -> bool:
        return await self.emit_event(
            self.config.strategy_invalidation_topic,
            {
                "strategy": self.strategy_name,
                "event": "setup_invalidated",
                "reason": reason,
                "setup": self.serialize_setup(setup),
            },
        )

    async def _publish_setup_expired(self, setup: SpoofingTradeSetup, *, reason: str) -> bool:
        return await self.emit_event(
            self.config.strategy_expired_topic,
            {
                "strategy": self.strategy_name,
                "event": "setup_expired",
                "reason": reason,
                "setup": self.serialize_setup(setup),
            },
        )

    async def emit_event(self, topic: str, payload: dict[str, Any]) -> bool:
        """
        Уніфікований emit через core EventBus.
        """
        if self.event_bus is None:
            self._stats["emit_failures"] += 1
            self.log_warning("emit skipped: event_bus is None", topic=topic)
            return False

        try:
            accepted = await self.event_bus.emit(
                topic,
                payload,
                source=self.strategy_name,
            )

            if accepted is False:
                self._stats["emit_failures"] += 1
                self.log_warning(
                    "EventBus rejected event",
                    topic=topic,
                    strategy_name=self.strategy_name,
                )
                return False

            return True

        except Exception:
            self._stats["emit_failures"] += 1
            self.log_exception(
                "Failed to emit event",
                topic=topic,
                strategy_name=self.strategy_name,
            )
            raise

    # -------------------------------------------------------------------------
    # rebuild / extraction
    # -------------------------------------------------------------------------

    def rebuild_minimal_signal_from_setup(self, setup: SpoofingTradeSetup) -> SpoofingSignal:
        return SpoofingSignal(
            signal_id=setup.source_signal_id,
            symbol=setup.symbol,
            exchange=setup.exchange,
            side=setup.signal_side,
            spoofing_type=setup.spoofing_type,
            pattern=setup.pattern,
            status=None,  # type: ignore[arg-type]
            price_level=setup.reference_price,
            wall_id=setup.metadata.get("wall_id"),
            score=setup.score,
            confidence=setup.confidence,
            severity=setup.severity,
            first_seen_at=setup.detected_at,
            detected_at=setup.detected_at,
            features=setup.features or SpoofingFeatures(),
            detector_results=[],
            score_breakdown=None,  # type: ignore[arg-type]
            metadata=dict(setup.metadata),
        )

    def _extract_spoofing_signal(self, event: Any) -> SpoofingSignal | None:
        if isinstance(event, SpoofingSignal):
            return event

        payload = getattr(event, "payload", None)
        if isinstance(payload, SpoofingSignal):
            return payload

        if isinstance(payload, dict):
            signal = payload.get("signal")
            if isinstance(signal, SpoofingSignal):
                return signal

        return None

    # -------------------------------------------------------------------------
    # cooldown
    # -------------------------------------------------------------------------

    def _cooldown_key(self, signal: SpoofingSignal) -> tuple[str, str, str]:
        return (
            signal.symbol,
            signal.pattern.value,
            self.resolve_direction(signal).value,
        )

    def _is_cooldown_active(self, signal: SpoofingSignal) -> bool:
        key = self._cooldown_key(signal)
        expires_at = self._cooldowns.get(key)

        if expires_at is None:
            return False

        return self.now() < expires_at

    def _set_cooldown(self, signal: SpoofingSignal) -> None:
        key = self._cooldown_key(signal)
        self._cooldowns[key] = self.now() + timedelta(
            milliseconds=self.config.cooldown_ms_same_symbol_pattern,
        )

    def _cleanup_cooldowns(self) -> None:
        now = self.now()
        expired_keys = [key for key, dt in self._cooldowns.items() if dt <= now]

        for key in expired_keys:
            self._cooldowns.pop(key, None)

    # -------------------------------------------------------------------------
    # cleanup
    # -------------------------------------------------------------------------

    async def run_cleanup_once(self) -> None:
        """
        Один cleanup tick.

        Саме цей метод викликає core Scheduler.
        """
        try:
            now = self.now()
            self._cleanup_cooldowns()

            for setup in list(self._active_setups_by_id.values()):
                if setup.is_terminal:
                    self._remove_setup(setup)
                    continue

                if self.is_setup_expired(setup, now=now):
                    await self.expire_setup(setup, reason="ttl_expired_cleanup")

        except Exception:
            self._stats["errors"] += 1
            self.log_exception(
                "Cleanup tick failed",
                strategy_name=self.strategy_name,
            )
            raise

    async def _cleanup_loop(self) -> None:
        """
        Fallback cleanup-loop для standalone/test режиму.

        У production краще передавати core Scheduler.
        """
        interval_sec = max(self.config.cleanup_interval_ms / 1000.0, 0.25)

        while self._is_running:
            try:
                await self.run_cleanup_once()
                await asyncio.sleep(interval_sec)

            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(interval_sec)

    # -------------------------------------------------------------------------
    # math / utility
    # -------------------------------------------------------------------------

    def signed_bps_move(self, *, current_price: float, reference_price: float) -> float:
        if current_price <= 0 or reference_price <= 0:
            return 0.0
        return (current_price - reference_price) / reference_price * 10_000.0

    def _compute_adverse_move_bps(
        self,
        *,
        setup: SpoofingTradeSetup,
        current_price: float,
    ) -> float:
        if current_price <= 0 or setup.reference_price <= 0:
            return 0.0

        move_bps = self.signed_bps_move(
            current_price=current_price,
            reference_price=setup.reference_price,
        )

        if setup.direction == StrategyDirection.LONG:
            return max(0.0, -move_bps)

        if setup.direction == StrategyDirection.SHORT:
            return max(0.0, move_bps)

        return 0.0

    def serialize_setup(self, setup: SpoofingTradeSetup) -> dict[str, Any]:
        return {
            "setup_id": setup.setup_id,
            "source_signal_id": setup.source_signal_id,
            "symbol": setup.symbol,
            "exchange": setup.exchange,
            "strategy_name": setup.strategy_name,
            "direction": setup.direction.value,
            "spoofing_type": setup.spoofing_type.value,
            "pattern": setup.pattern.value,
            "severity": setup.severity.value,
            "score": setup.score,
            "confidence": setup.confidence,
            "reference_price": setup.reference_price,
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "take_profit_price": setup.take_profit_price,
            "signal_side": setup.signal_side.value,
            "detected_at": self._dt_to_iso(setup.detected_at),
            "expires_at": self._dt_to_iso(setup.expires_at),
            "status": setup.status.value,
            "confirmed_at": self._dt_to_iso(setup.confirmed_at),
            "triggered_at": self._dt_to_iso(setup.triggered_at),
            "invalidated_at": self._dt_to_iso(setup.invalidated_at),
            "expired_at": self._dt_to_iso(setup.expired_at),
            "confirmation_price": setup.confirmation_price,
            "last_price": setup.last_price,
            "features": self._serialize_features(setup.features),
            "metadata": dict(setup.metadata),
        }

    def _serialize_features(self, features: SpoofingFeatures | None) -> dict[str, Any] | None:
        if features is None:
            return None

        try:
            return asdict(features)
        except Exception:
            return {"repr": repr(features)}

    def _extract_event_timestamp(
        self,
        event: Any,
        *,
        fallback: datetime,
    ) -> datetime:
        payload = getattr(event, "payload", None)
        candidates: list[Any] = []

        if isinstance(payload, dict):
            candidates.extend(
                [
                    payload.get("timestamp"),
                    payload.get("ts"),
                    payload.get("event_time"),
                ]
            )

        candidates.append(getattr(event, "timestamp", None))

        for value in candidates:
            dt = self._coerce_datetime(value)
            if dt is not None:
                return dt

        return fallback

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return self.ensure_utc(value)

        if isinstance(value, (int, float)):
            if value > 10_000_000_000:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return self.ensure_utc(parsed)
            except Exception:
                return None

        return None

    def ensure_utc(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _dt_to_iso(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return self.ensure_utc(value).isoformat()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _max_severity(
        self,
        a: SpoofingSeverity,
        b: SpoofingSeverity,
    ) -> SpoofingSeverity:
        order = {
            SpoofingSeverity.LOW: 1,
            SpoofingSeverity.MEDIUM: 2,
            SpoofingSeverity.HIGH: 3,
            SpoofingSeverity.CRITICAL: 4,
        }
        return a if order[a] >= order[b] else b

    def stats(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "is_running": self._is_running,
            "registered": self._registered,
            "active_setups": len(self._active_setups_by_id),
            "cooldowns": len(self._cooldowns),
            "cleanup_job_id": self._cleanup_job_id,
            "has_fallback_cleanup_task": self._cleanup_task is not None,
            **self._stats,
        }

    # -------------------------------------------------------------------------
    # feature helpers
    # -------------------------------------------------------------------------

    def _feature_float(
        self,
        features: SpoofingFeatures | dict[str, Any] | None,
        name: str,
        default: float = 0.0,
    ) -> float:
        if features is None:
            return default

        if isinstance(features, dict):
            return self._safe_float(features.get(name), default)

        return self._safe_float(getattr(features, name, default), default)

    def _feature_bool(
        self,
        features: SpoofingFeatures | dict[str, Any] | None,
        name: str,
        default: bool = False,
    ) -> bool:
        if features is None:
            return default

        if isinstance(features, dict):
            value = features.get(name, default)
        else:
            value = getattr(features, name, default)

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        return default

    # -------------------------------------------------------------------------
    # logging helpers
    # -------------------------------------------------------------------------

    def log_debug(self, message: str, **kwargs: Any) -> None:
        self.logger.debug(message, extra=kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, extra=kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        self.logger.warning(message, extra=kwargs)

    def log_error(self, message: str, **kwargs: Any) -> None:
        self.logger.error(message, extra=kwargs)

    def log_exception(self, message: str, **kwargs: Any) -> None:
        self.logger.exception(message, extra=kwargs)