from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable

from core.event_bus import EventBus, Subscription
from core.logger import get_logger
from core.scheduler import Scheduler

from analytics.spoofing import (
    DetectorResult,
    ScoreContribution,
    SpoofingComponent,
    SpoofingFeatures,
    SpoofingPattern,
    SpoofingScore,
    SpoofingSeverity,
    SpoofingSide,
    SpoofingSignal,
    SpoofingStatus,
    SpoofingType,
)


DEFAULT_STRATEGY_MARKET_TYPE = "perpetual"
DEFAULT_STRATEGY_TIMEFRAME = "realtime"

StrategyScopeKey = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


class SetupStatus(str, Enum):
    """
    Стан setup-а на strategy-рівні.
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

    Цей клас поки зберігає старий EventBus-style lifecycle, але контракт
    уже узгоджено з актуальним analytics.spoofing:
    - full scope: exchange + market_type + symbol + timeframe;
    - detector_results;
    - score_breakdown;
    - analytics metadata.
    """

    enabled: bool = True

    # EventBus topics.
    # Поки залишено тут для поточного standalone режиму.
    spoofing_detected_topic: str = "analytics.spoofing.detected"
    spoofing_updated_topic: str = "analytics.spoofing.updated"

    # Новий data-flow має працювати через data cache update або StrategyContext.
    # Для поточного етапу підтримуємо topic конфігураційно.
    market_trade_topic: str = "market.trades.updated"
    market_orderbook_topic: str = "market.orderbook.updated"

    strategy_signal_topic: str = "signal.generated"
    strategy_setup_topic: str = "strategy.spoofing.setup_created"
    strategy_update_topic: str = "strategy.spoofing.setup_updated"
    strategy_invalidation_topic: str = "strategy.spoofing.setup_invalidated"
    strategy_expired_topic: str = "strategy.spoofing.setup_expired"

    # Base filtering.
    min_score: float = 0.65
    min_confidence: float = 0.55
    allowed_severities: tuple[SpoofingSeverity, ...] = (
        SpoofingSeverity.MEDIUM,
        SpoofingSeverity.HIGH,
        SpoofingSeverity.CRITICAL,
    )

    # Analytics score contract.
    require_score_passed: bool = False
    min_detector_count: int = 1
    min_agreement_ratio: float = 0.0
    min_average_confidence: float = 0.0

    # Lifecycle.
    setup_ttl_ms: int = 8_000
    cooldown_ms_same_symbol_pattern: int = 10_000
    max_active_setups_per_scope: int = 8

    # Backward-compatible old name.
    # New code should use max_active_setups_per_scope.
    max_active_setups_per_symbol: int = 8

    # Confirmation.
    require_confirmation: bool = True
    min_confirmation_move_bps: float = 1.0
    max_adverse_move_bps: float = 2.5

    # Risk-ish defaults, not full RiskManager responsibility.
    default_entry_offset_bps: float = 0.0
    default_stop_buffer_bps: float = 3.0
    default_take_profit_bps: float = 6.0
    rr_multiplier: float = 2.0

    # Behavior.
    process_updates_for_existing_signal: bool = True
    allow_multiple_setups_same_scope: bool = True

    # Backward-compatible old name.
    allow_multiple_setups_same_symbol: bool = True

    publish_setup_events: bool = True
    publish_debug_updates: bool = True

    # Cleanup.
    cleanup_interval_ms: int = 2_000
    cleanup_timeout_sec: float = 1.0
    cleanup_max_retries: int = 1
    cleanup_retry_delay_sec: float = 0.5

    def __post_init__(self) -> None:
        # Backward compatibility with old configs.
        if self.max_active_setups_per_scope == 8 and self.max_active_setups_per_symbol != 8:
            self.max_active_setups_per_scope = self.max_active_setups_per_symbol
        else:
            self.max_active_setups_per_symbol = self.max_active_setups_per_scope

        self.allow_multiple_setups_same_symbol = self.allow_multiple_setups_same_scope


@dataclass(slots=True)
class SpoofingTradeSetup:
    """
    Внутрішня модель trade setup, створена на основі analytics.spoofing.SpoofingSignal.

    Важливо:
    - setup тепер зберігає повний futures scope;
    - не втрачає detector_results / score_breakdown / analytics metadata;
    - downstream payload може передати risk/execution повний контекст.
    """

    setup_id: str
    source_signal_id: str

    exchange: str
    market_type: str
    symbol: str
    timeframe: str
    exchange_symbol: str

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
    detector_results: list[DetectorResult] = field(default_factory=list)
    score_breakdown: SpoofingScore | None = None

    analytics_metadata: dict[str, Any] = field(default_factory=dict)
    source_signal_snapshot: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_key(self) -> StrategyScopeKey:
        return (
            self.exchange,
            self.market_type,
            self.symbol,
            self.timeframe,
        )

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
    Базовий клас для strategy/strategies/spoofing/*.

    Поточна роль:
    - слухає analytics.spoofing.*;
    - створює setup-и на основі SpoofingSignal;
    - трекає setup lifecycle;
    - публікує strategy signal.

    Цей файл поки НЕ переводиться на новий StrategyEngine/BaseStrategy.
    Виправлення тут стосуються саме контракту з analytics.spoofing.
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

        # New canonical scoped index.
        self._setup_ids_by_scope: dict[StrategyScopeKey, set[str]] = {}

        # Backward-compatible helper index for legacy symbol-only reads.
        self._setup_ids_by_symbol: dict[str, set[str]] = {}

        self._cooldowns: dict[tuple[str, str, str, str, str, str], datetime] = {}

        self._latest_price_by_scope: dict[StrategyScopeKey, float] = {}
        self._latest_trade_ts_by_scope: dict[StrategyScopeKey, datetime] = {}

        # Backward-compatible legacy price index.
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
        Підписка на EventBus.

        Поки залишено в цьому класі для сумісності зі старим lifecycle.
        Пізніше це буде перенесено в StrategyEngine/StrategyEventHandler.
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
        Знімає subscriptions з EventBus.
        """
        if not self._registered:
            return

        if self.event_bus is not None:
            for subscription in self._subscriptions:
                try:
                    unsubscribe = getattr(subscription, "unsubscribe", None)
                    if callable(unsubscribe):
                        unsubscribe()
                        continue

                    close = getattr(subscription, "close", None)
                    if callable(close):
                        close()
                        continue

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

        Поки зберігаємо поточний механізм.
        У production бажано передавати core Scheduler.
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
        Головна точка входу для нових analytics.spoofing сигналів.
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
                    exchange=signal.exchange,
                    market_type=signal.market_type,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
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
                    exchange=signal.exchange,
                    market_type=signal.market_type,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
                    pattern=signal.pattern.value,
                )
                return

            if not self._can_create_more_setups_for_signal(signal):
                self.log_debug(
                    "Max active setups for scope reached",
                    signal_id=signal.signal_id,
                    exchange=signal.exchange,
                    market_type=signal.market_type,
                    symbol=signal.symbol,
                    timeframe=signal.timeframe,
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
                exchange=setup.exchange,
                market_type=setup.market_type,
                symbol=setup.symbol,
                timeframe=setup.timeframe,
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
        Обробка analytics.spoofing.updated.
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
                latest_price = self._latest_price_by_scope.get(
                    setup.scope_key,
                    self._latest_price_by_symbol.get(setup.symbol, setup.reference_price),
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
        Обробка price/trade update для confirmation/invalidation.

        Поки метод залишено для старого runtime.
        Новий flow надалі має передавати current price через StrategyContext
        або data cache update, а не напряму raw market.trade.

        Підтримувані payload форми:
        - {"exchange", "market_type", "symbol", "timeframe", "price", ...}
        - {"trade": {...}}
        - {"trades": [{...}, ...]}
        - market.trades.updated window payload, якщо містить last_trade/latest price.
        """
        payload = getattr(event, "payload", event)
        trade_payloads = self._extract_trade_payloads(payload)
        if not trade_payloads:
            return

        for trade in trade_payloads:
            await self._handle_trade_payload(event=event, payload=trade)

    async def _handle_trade_payload(self, *, event: Any, payload: dict[str, Any]) -> None:
        symbol = self.normalize_symbol(payload.get("symbol"))
        if not symbol:
            return

        exchange = self.normalize_exchange(payload.get("exchange"))
        market_type = self.normalize_market_type(payload.get("market_type"))
        timeframe = self.normalize_timeframe(payload.get("timeframe"))

        price = self._extract_trade_price(payload)
        if price <= 0:
            return

        now = self._extract_event_timestamp(event, payload=payload, fallback=self.now())

        scope_key = self.make_scope_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        self._latest_price_by_scope[scope_key] = price
        self._latest_trade_ts_by_scope[scope_key] = now

        self._latest_price_by_symbol[symbol] = price
        self._latest_trade_ts_by_symbol[symbol] = now

        active_setups = self.get_active_setups_for_scope_or_symbol(
            scope_key=scope_key,
            symbol=symbol,
            allow_symbol_fallback=exchange == "" or timeframe == DEFAULT_STRATEGY_TIMEFRAME,
        )
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
        """
        Base acceptance filter з урахуванням актуального analytics.spoofing contract.
        """
        if signal.score < self.config.min_score:
            return False

        if signal.confidence < self.config.min_confidence:
            return False

        if signal.severity not in self.config.allowed_severities:
            return False

        if not self._passes_score_contract(signal):
            return False

        if not self.supports_pattern(signal):
            return False

        return True

    def _passes_score_contract(self, signal: SpoofingSignal) -> bool:
        score = signal.score_breakdown

        if self.config.require_score_passed and score is not None:
            if not bool(getattr(score, "passed", False)):
                return False

        detector_count = self.analytics_int(signal, "detector_count", default=len(signal.detector_results))
        if detector_count < self.config.min_detector_count:
            return False

        agreement_ratio = self.analytics_float(signal, "agreement_ratio", default=0.0)
        if agreement_ratio < self.config.min_agreement_ratio:
            return False

        average_confidence = self.analytics_float(signal, "average_confidence", default=0.0)
        if average_confidence < self.config.min_average_confidence:
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

        analytics_metadata = self.extract_analytics_metadata(signal)
        source_snapshot = self.serialize_spoofing_signal(signal)

        setup = SpoofingTradeSetup(
            setup_id=self.build_setup_id(signal),
            source_signal_id=signal.signal_id,
            exchange=self.normalize_exchange(signal.exchange),
            market_type=self.normalize_market_type(signal.market_type),
            symbol=self.normalize_symbol(signal.symbol) or str(signal.symbol),
            timeframe=self.normalize_timeframe(signal.timeframe),
            exchange_symbol=self.normalize_exchange_symbol(
                getattr(signal, "exchange_symbol", None),
                fallback=self.normalize_symbol(signal.symbol) or str(signal.symbol),
            ),
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
            detector_results=list(signal.detector_results or []),
            score_breakdown=signal.score_breakdown,
            analytics_metadata=analytics_metadata,
            source_signal_snapshot=source_snapshot,
            metadata={
                "signal_id": signal.signal_id,
                "scope": {
                    "exchange": self.normalize_exchange(signal.exchange),
                    "market_type": self.normalize_market_type(signal.market_type),
                    "symbol": self.normalize_symbol(signal.symbol) or str(signal.symbol),
                    "timeframe": self.normalize_timeframe(signal.timeframe),
                    "exchange_symbol": self.normalize_exchange_symbol(
                        getattr(signal, "exchange_symbol", None),
                        fallback=self.normalize_symbol(signal.symbol) or str(signal.symbol),
                    ),
                },
                "pattern": signal.pattern.value,
                "spoofing_type": signal.spoofing_type.value,
                "severity": signal.severity.value,
                "price_level": signal.price_level,
                "wall_id": signal.wall_id,
                "source_detected_at": self._dt_to_iso(signal.detected_at),
                "source_first_seen_at": self._dt_to_iso(signal.first_seen_at),
                "analytics": analytics_metadata,
            },
        )

        self.enrich_setup(setup, signal)
        return setup

    def enrich_setup(self, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        return None

    def apply_signal_update(self, *, setup: SpoofingTradeSetup, signal: SpoofingSignal) -> None:
        """
        Оновлює setup, не втрачаючи актуальні analytics fields.
        """
        setup.score = max(setup.score, signal.score)
        setup.confidence = max(setup.confidence, signal.confidence)
        setup.severity = self._max_severity(setup.severity, signal.severity)

        setup.features = signal.features
        setup.detector_results = list(signal.detector_results or [])
        setup.score_breakdown = signal.score_breakdown
        setup.analytics_metadata = self.extract_analytics_metadata(signal)
        setup.source_signal_snapshot = self.serialize_spoofing_signal(signal)

        setup.metadata["last_signal_update_at"] = self._dt_to_iso(self.now())
        setup.metadata["analytics"] = dict(setup.analytics_metadata)
        setup.metadata["source_signal"] = dict(setup.source_signal_snapshot)

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
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
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
        """
        Base interpretation:
        - fake/pulled ASK pressure -> LONG;
        - fake/pulled BID support -> SHORT.

        Concrete strategy може перевизначити це для composite/layering cases.
        """
        if signal.side == SpoofingSide.ASK:
            return StrategyDirection.LONG
        if signal.side == SpoofingSide.BID:
            return StrategyDirection.SHORT
        return StrategyDirection.FLAT

    def resolve_reference_price(self, signal: SpoofingSignal) -> float:
        """
        Актуальний SpoofingFeatures не має reference_price.
        Правильний порядок:
        1. signal.price_level;
        2. signal.features.price;
        3. detector metadata price/reference_price fallback.
        """
        if signal.price_level and signal.price_level > 0:
            return float(signal.price_level)

        feature_price = self._feature_float(signal.features, "price")
        if feature_price > 0:
            return feature_price

        detector_price = self.first_detector_metadata_float(
            signal,
            names=("price", "price_level", "reference_price"),
            default=0.0,
        )
        if detector_price > 0:
            return detector_price

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

    def get_active_setups_for_scope(self, scope_key: StrategyScopeKey) -> list[SpoofingTradeSetup]:
        setup_ids = self._setup_ids_by_scope.get(scope_key, set())
        setups: list[SpoofingTradeSetup] = []

        for setup_id in setup_ids:
            setup = self._active_setups_by_id.get(setup_id)
            if setup is not None and setup.is_active:
                setups.append(setup)

        setups.sort(key=lambda item: item.detected_at, reverse=True)
        return setups

    def get_active_setups_for_symbol(self, symbol: str) -> list[SpoofingTradeSetup]:
        normalized_symbol = self.normalize_symbol(symbol)
        if not normalized_symbol:
            return []

        setup_ids = self._setup_ids_by_symbol.get(normalized_symbol, set())
        setups: list[SpoofingTradeSetup] = []

        for setup_id in setup_ids:
            setup = self._active_setups_by_id.get(setup_id)
            if setup is not None and setup.is_active:
                setups.append(setup)

        setups.sort(key=lambda item: item.detected_at, reverse=True)
        return setups

    def get_active_setups_for_scope_or_symbol(
        self,
        *,
        scope_key: StrategyScopeKey,
        symbol: str,
        allow_symbol_fallback: bool = True,
    ) -> list[SpoofingTradeSetup]:
        setups = self.get_active_setups_for_scope(scope_key)
        if setups or not allow_symbol_fallback:
            return setups

        return self.get_active_setups_for_symbol(symbol)

    def _store_setup(self, setup: SpoofingTradeSetup) -> None:
        self._active_setups_by_id[setup.setup_id] = setup
        self._setup_id_by_signal_id[setup.source_signal_id] = setup.setup_id
        self._setup_ids_by_scope.setdefault(setup.scope_key, set()).add(setup.setup_id)
        self._setup_ids_by_symbol.setdefault(setup.symbol, set()).add(setup.setup_id)

    def _remove_setup(self, setup: SpoofingTradeSetup) -> None:
        self._active_setups_by_id.pop(setup.setup_id, None)
        self._setup_id_by_signal_id.pop(setup.source_signal_id, None)

        scope_key = setup.scope_key
        if scope_key in self._setup_ids_by_scope:
            self._setup_ids_by_scope[scope_key].discard(setup.setup_id)
            if not self._setup_ids_by_scope[scope_key]:
                self._setup_ids_by_scope.pop(scope_key, None)

        if setup.symbol in self._setup_ids_by_symbol:
            self._setup_ids_by_symbol[setup.symbol].discard(setup.setup_id)
            if not self._setup_ids_by_symbol[setup.symbol]:
                self._setup_ids_by_symbol.pop(setup.symbol, None)

    def _has_active_setup_for_signal(self, signal_id: str) -> bool:
        setup = self.get_setup_by_signal_id(signal_id)
        return setup is not None and setup.is_active

    def _can_create_more_setups_for_signal(self, signal: SpoofingSignal) -> bool:
        scope_key = self.make_scope_key_from_signal(signal)
        active = self.get_active_setups_for_scope(scope_key)

        if self.config.allow_multiple_setups_same_scope:
            return len(active) < self.config.max_active_setups_per_scope

        return len(active) == 0

    # Backward-compatible old method name.
    def _can_create_more_setups_for_symbol(self, symbol: str) -> bool:
        active = self.get_active_setups_for_symbol(symbol)

        if self.config.allow_multiple_setups_same_symbol:
            return len(active) < self.config.max_active_setups_per_symbol

        return len(active) == 0

    def build_setup_id(self, signal: SpoofingSignal) -> str:
        return (
            f"{self.strategy_name}:"
            f"{signal.exchange}:{signal.market_type}:{signal.symbol}:{signal.timeframe}:"
            f"{signal.signal_id}"
        )

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
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
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
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
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
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
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
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            direction=setup.direction.value,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            take_profit_price=setup.take_profit_price,
            emitted=emitted,
        )

        self._remove_setup(setup)

    def build_strategy_signal_payload(self, setup: SpoofingTradeSetup) -> dict[str, Any]:
        """
        Payload не втрачає analytics.spoofing context.
        Надалі SignalProcessor зможе нормалізувати його у фінальну модель.
        """
        analytics_payload = {
            "source": "analytics.spoofing",
            "source_signal_id": setup.source_signal_id,
            "wall_id": setup.metadata.get("wall_id"),
            "score_breakdown": self.serialize_score_breakdown(setup.score_breakdown),
            "detector_results": self.serialize_detector_results(setup.detector_results),
            "metadata": dict(setup.analytics_metadata),
            "agreement_ratio": self._safe_float(
                setup.analytics_metadata.get("agreement_ratio"),
                0.0,
            ),
            "average_confidence": self._safe_float(
                setup.analytics_metadata.get("average_confidence"),
                0.0,
            ),
            "detector_count": int(
                self._safe_float(
                    setup.analytics_metadata.get("detector_count"),
                    len(setup.detector_results),
                )
            ),
            "threshold": self._safe_float(
                setup.analytics_metadata.get("threshold"),
                0.0,
            ),
            "passed": bool(setup.analytics_metadata.get("passed", False)),
        }

        return {
            "strategy": self.strategy_name,
            "strategy_type": "spoofing",
            "setup_id": setup.setup_id,
            "source_signal_id": setup.source_signal_id,

            # Full futures scope.
            "exchange": setup.exchange,
            "market_type": setup.market_type,
            "symbol": setup.symbol,
            "timeframe": setup.timeframe,
            "exchange_symbol": setup.exchange_symbol,

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
            "wall_id": setup.metadata.get("wall_id"),

            "detected_at": self._dt_to_iso(setup.detected_at),
            "confirmed_at": self._dt_to_iso(setup.confirmed_at),
            "triggered_at": self._dt_to_iso(setup.triggered_at),

            "features": self._serialize_features(setup.features),
            "analytics": analytics_payload,
            "source_signal": dict(setup.source_signal_snapshot),
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
        """
        Backward-compatible helper для старого confirmation flow.

        Важливо:
        - більше не передає status=None;
        - не втрачає market_type/timeframe/exchange_symbol;
        - повертає detector_results і score_breakdown із setup.
        """
        features = setup.features
        if features is None:
            features = SpoofingFeatures(
                exchange=setup.exchange,
                market_type=setup.market_type,
                symbol=setup.symbol,
                timeframe=setup.timeframe,
                exchange_symbol=setup.exchange_symbol,
                price=setup.reference_price,
            )

        return SpoofingSignal(
            signal_id=setup.source_signal_id,
            exchange=setup.exchange,
            market_type=setup.market_type,
            symbol=setup.symbol,
            timeframe=setup.timeframe,
            exchange_symbol=setup.exchange_symbol,
            side=setup.signal_side,
            spoofing_type=setup.spoofing_type,
            pattern=setup.pattern,
            status=SpoofingStatus.DETECTED,
            price_level=setup.reference_price,
            wall_id=setup.metadata.get("wall_id"),
            score=setup.score,
            confidence=setup.confidence,
            severity=setup.severity,
            first_seen_at=setup.detected_at,
            detected_at=setup.detected_at,
            features=features,
            detector_results=list(setup.detector_results),
            score_breakdown=setup.score_breakdown,
            metadata={
                **dict(setup.analytics_metadata),
                **dict(setup.metadata),
            },
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

    def _cooldown_key(self, signal: SpoofingSignal) -> tuple[str, str, str, str, str, str]:
        return (
            self.normalize_exchange(signal.exchange),
            self.normalize_market_type(signal.market_type),
            self.normalize_symbol(signal.symbol) or str(signal.symbol),
            self.normalize_timeframe(signal.timeframe),
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

        Пізніше треба прибрати при переході на StrategyLifecycleManager.
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
    # analytics helpers
    # -------------------------------------------------------------------------

    def extract_analytics_metadata(self, signal: SpoofingSignal) -> dict[str, Any]:
        """
        Витягує metadata, яку формує SpoofingScoreEngine / analytics.spoofing.
        """
        metadata = dict(signal.metadata or {})

        score = signal.score_breakdown
        if score is not None:
            metadata.setdefault("threshold", getattr(score, "threshold", None))
            metadata.setdefault("passed", getattr(score, "passed", None))
            metadata.setdefault("score_severity", getattr(getattr(score, "severity", None), "value", None))

        metadata.setdefault("detector_count", len(signal.detector_results or []))
        metadata.setdefault("score", signal.score)
        metadata.setdefault("confidence", signal.confidence)
        metadata.setdefault("severity", signal.severity.value)
        metadata.setdefault("pattern", signal.pattern.value)
        metadata.setdefault("spoofing_type", signal.spoofing_type.value)

        return self._serialize_plain(metadata)

    def analytics_float(self, signal: SpoofingSignal, name: str, default: float = 0.0) -> float:
        value = (signal.metadata or {}).get(name)
        if value is None and signal.score_breakdown is not None:
            value = getattr(signal.score_breakdown, name, None)
        return self._safe_float(value, default)

    def analytics_int(self, signal: SpoofingSignal, name: str, default: int = 0) -> int:
        return int(self.analytics_float(signal, name, float(default)))

    def analytics_bool(self, signal: SpoofingSignal, name: str, default: bool = False) -> bool:
        value = (signal.metadata or {}).get(name)
        if value is None and signal.score_breakdown is not None:
            value = getattr(signal.score_breakdown, name, None)
        return self._coerce_bool(value, default)

    def has_detector(
        self,
        signal: SpoofingSignal,
        detector: SpoofingComponent | str,
        *,
        positive_only: bool = True,
    ) -> bool:
        detector_value = detector.value if isinstance(detector, SpoofingComponent) else str(detector)

        for result in signal.detector_results or []:
            result_detector = getattr(result, "detector", None)
            result_detector_value = (
                result_detector.value
                if isinstance(result_detector, Enum)
                else str(result_detector)
            )

            if result_detector_value != detector_value:
                continue

            if positive_only:
                is_positive = getattr(result, "is_positive", None)
                if callable(is_positive):
                    if not is_positive():
                        continue

            return True

        return False

    def detector_results_for(
        self,
        signal: SpoofingSignal,
        detector: SpoofingComponent | str,
        *,
        positive_only: bool = True,
    ) -> list[DetectorResult]:
        detector_value = detector.value if isinstance(detector, SpoofingComponent) else str(detector)
        results: list[DetectorResult] = []

        for result in signal.detector_results or []:
            result_detector = getattr(result, "detector", None)
            result_detector_value = (
                result_detector.value
                if isinstance(result_detector, Enum)
                else str(result_detector)
            )
            if result_detector_value != detector_value:
                continue

            if positive_only:
                is_positive = getattr(result, "is_positive", None)
                if callable(is_positive) and not is_positive():
                    continue

            results.append(result)

        return results

    def detector_score(
        self,
        signal: SpoofingSignal,
        detector: SpoofingComponent | str,
        default: float = 0.0,
    ) -> float:
        results = self.detector_results_for(signal, detector)
        if not results:
            return default
        return max(self._safe_float(getattr(item, "score", None), default) for item in results)

    def detector_confidence(
        self,
        signal: SpoofingSignal,
        detector: SpoofingComponent | str,
        default: float = 0.0,
    ) -> float:
        results = self.detector_results_for(signal, detector)
        if not results:
            return default
        return max(self._safe_float(getattr(item, "confidence", None), default) for item in results)

    def detector_metadata(
        self,
        signal: SpoofingSignal,
        detector: SpoofingComponent | str,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}

        for result in self.detector_results_for(signal, detector):
            metadata = getattr(result, "metadata", None)
            if isinstance(metadata, dict):
                merged.update(metadata)

        return merged

    def first_detector_metadata_float(
        self,
        signal: SpoofingSignal,
        names: Iterable[str],
        *,
        detector: SpoofingComponent | str | None = None,
        default: float = 0.0,
    ) -> float:
        results = (
            self.detector_results_for(signal, detector)
            if detector is not None
            else list(signal.detector_results or [])
        )

        for result in results:
            metadata = getattr(result, "metadata", None)
            if not isinstance(metadata, dict):
                continue

            for name in names:
                value = self._safe_float(metadata.get(name), default)
                if value != default:
                    return value

        return default

    def score_contribution(
        self,
        signal: SpoofingSignal,
        component: Any,
        *,
        default: float = 0.0,
    ) -> float:
        score = signal.score_breakdown
        if score is None:
            return default

        component_value = component.value if isinstance(component, Enum) else str(component)

        for contribution in getattr(score, "contributions", []) or []:
            contribution_component = getattr(contribution, "component", None)
            contribution_component_value = (
                contribution_component.value
                if isinstance(contribution_component, Enum)
                else str(contribution_component)
            )

            if contribution_component_value == component_value:
                return self._safe_float(getattr(contribution, "value", None), default)

        return default

    # -------------------------------------------------------------------------
    # trade payload helpers
    # -------------------------------------------------------------------------

    def _extract_trade_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        candidates: list[Any] = []

        if isinstance(payload.get("trade"), dict):
            candidates.append(payload["trade"])

        if isinstance(payload.get("last_trade"), dict):
            candidates.append(payload["last_trade"])

        if isinstance(payload.get("latest_trade"), dict):
            candidates.append(payload["latest_trade"])

        trades = payload.get("trades")
        if isinstance(trades, list):
            candidates.extend(item for item in trades if isinstance(item, dict))

        # If no nested trade exists, treat payload itself as a trade-like payload.
        if not candidates:
            candidates.append(payload)

        result: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue

            merged = dict(payload)
            merged.update(item)

            # Avoid carrying full list into every item.
            merged.pop("trades", None)
            result.append(merged)

        return result

    def _extract_trade_price(self, payload: dict[str, Any]) -> float:
        for key in ("price", "last_price", "latest_price", "close", "mark_price"):
            price = self._safe_float(payload.get(key), 0.0)
            if price > 0:
                return price
        return 0.0

    # -------------------------------------------------------------------------
    # scope helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def normalize_exchange(exchange: object) -> str:
        return str(exchange or "").strip().lower()

    @staticmethod
    def normalize_market_type(market_type: object = DEFAULT_STRATEGY_MARKET_TYPE) -> str:
        normalized = str(market_type or DEFAULT_STRATEGY_MARKET_TYPE).strip().lower()
        return normalized or DEFAULT_STRATEGY_MARKET_TYPE

    @staticmethod
    def normalize_symbol(symbol: object) -> str:
        return str(symbol or "").strip().upper()

    @staticmethod
    def normalize_timeframe(timeframe: object = DEFAULT_STRATEGY_TIMEFRAME) -> str:
        normalized = str(timeframe or DEFAULT_STRATEGY_TIMEFRAME).strip()
        return normalized or DEFAULT_STRATEGY_TIMEFRAME

    @staticmethod
    def normalize_exchange_symbol(exchange_symbol: object, *, fallback: str) -> str:
        normalized = str(exchange_symbol or "").strip()
        return normalized or fallback

    def make_scope_key(
        self,
        *,
        exchange: object,
        market_type: object,
        symbol: object,
        timeframe: object,
    ) -> StrategyScopeKey:
        return (
            self.normalize_exchange(exchange),
            self.normalize_market_type(market_type),
            self.normalize_symbol(symbol),
            self.normalize_timeframe(timeframe),
        )

    def make_scope_key_from_signal(self, signal: SpoofingSignal) -> StrategyScopeKey:
        return self.make_scope_key(
            exchange=signal.exchange,
            market_type=signal.market_type,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
        )

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

            "exchange": setup.exchange,
            "market_type": setup.market_type,
            "symbol": setup.symbol,
            "timeframe": setup.timeframe,
            "exchange_symbol": setup.exchange_symbol,

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
            "detector_results": self.serialize_detector_results(setup.detector_results),
            "score_breakdown": self.serialize_score_breakdown(setup.score_breakdown),
            "analytics_metadata": dict(setup.analytics_metadata),
            "source_signal": dict(setup.source_signal_snapshot),
            "metadata": dict(setup.metadata),
        }

    def serialize_spoofing_signal(self, signal: SpoofingSignal) -> dict[str, Any]:
        return {
            "signal_id": signal.signal_id,
            "exchange": signal.exchange,
            "market_type": signal.market_type,
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "exchange_symbol": getattr(signal, "exchange_symbol", None),
            "side": signal.side.value,
            "spoofing_type": signal.spoofing_type.value,
            "pattern": signal.pattern.value,
            "status": signal.status.value,
            "price_level": signal.price_level,
            "wall_id": signal.wall_id,
            "score": signal.score,
            "confidence": signal.confidence,
            "severity": signal.severity.value,
            "first_seen_at": self._dt_to_iso(signal.first_seen_at),
            "detected_at": self._dt_to_iso(signal.detected_at),
            "features": self._serialize_features(signal.features),
            "detector_results": self.serialize_detector_results(signal.detector_results),
            "score_breakdown": self.serialize_score_breakdown(signal.score_breakdown),
            "metadata": self._serialize_plain(signal.metadata or {}),
        }

    def serialize_detector_results(
        self,
        detector_results: Iterable[DetectorResult] | None,
    ) -> list[dict[str, Any]]:
        if not detector_results:
            return []

        serialized: list[dict[str, Any]] = []

        for result in detector_results:
            try:
                data = {
                    "detector": self._enum_value(getattr(result, "detector", None)),
                    "decision": self._enum_value(getattr(result, "decision", None)),
                    "score": getattr(result, "score", None),
                    "confidence": getattr(result, "confidence", None),
                    "reason": getattr(result, "reason", None),
                    "wall_id": getattr(result, "wall_id", None),
                    "pattern": self._enum_value(getattr(result, "pattern", None)),
                    "features": self._serialize_features(getattr(result, "features", None)),
                    "metadata": self._serialize_plain(getattr(result, "metadata", {}) or {}),
                }
                serialized.append(data)
            except Exception:
                serialized.append({"repr": repr(result)})

        return serialized

    def serialize_score_breakdown(self, score: SpoofingScore | None) -> dict[str, Any] | None:
        if score is None:
            return None

        try:
            return {
                "total_score": getattr(score, "total_score", None),
                "confidence": getattr(score, "confidence", None),
                "severity": self._enum_value(getattr(score, "severity", None)),
                "threshold": getattr(score, "threshold", None),
                "passed": getattr(score, "passed", None),
                "contributions": [
                    self.serialize_score_contribution(item)
                    for item in (getattr(score, "contributions", []) or [])
                ],
                "metadata": self._serialize_plain(getattr(score, "metadata", {}) or {}),
            }
        except Exception:
            return {"repr": repr(score)}

    def serialize_score_contribution(self, contribution: ScoreContribution) -> dict[str, Any]:
        return {
            "component": self._enum_value(getattr(contribution, "component", None)),
            "value": getattr(contribution, "value", None),
            "weight": getattr(contribution, "weight", None),
            "raw_value": getattr(contribution, "raw_value", None),
            "reason": getattr(contribution, "reason", None),
            "metadata": self._serialize_plain(getattr(contribution, "metadata", {}) or {}),
        }

    def _serialize_features(self, features: SpoofingFeatures | dict[str, Any] | None) -> dict[str, Any] | None:
        if features is None:
            return None

        if isinstance(features, dict):
            return self._serialize_plain(features)

        try:
            return self._serialize_plain(asdict(features))
        except Exception:
            return {"repr": repr(features)}

    def _serialize_plain(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value

        if isinstance(value, datetime):
            return self._dt_to_iso(value)

        if is_dataclass(value):
            try:
                return self._serialize_plain(asdict(value))
            except Exception:
                return repr(value)

        if isinstance(value, dict):
            return {
                str(key): self._serialize_plain(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [self._serialize_plain(item) for item in value]

        if isinstance(value, tuple):
            return [self._serialize_plain(item) for item in value]

        if isinstance(value, set):
            return sorted(self._serialize_plain(item) for item in value)

        return value

    def _extract_event_timestamp(
        self,
        event: Any,
        *,
        fallback: datetime,
        payload: dict[str, Any] | None = None,
    ) -> datetime:
        event_payload = payload if payload is not None else getattr(event, "payload", None)
        candidates: list[Any] = []

        if isinstance(event_payload, dict):
            candidates.extend(
                [
                    event_payload.get("timestamp"),
                    event_payload.get("timestamp_ms"),
                    event_payload.get("ts"),
                    event_payload.get("event_time"),
                    event_payload.get("received_at"),
                    event_payload.get("received_at_ms"),
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

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        return default

    def _enum_value(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

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
            "active_scopes": len(self._setup_ids_by_scope),
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
            value = features.get(name)
            if value is None and isinstance(features.get("metadata"), dict):
                value = features["metadata"].get(name)
            return self._safe_float(value, default)

        value = getattr(features, name, None)
        if value is None:
            metadata = getattr(features, "metadata", None)
            if isinstance(metadata, dict):
                value = metadata.get(name)

        return self._safe_float(value, default)

    def _feature_bool(
        self,
        features: SpoofingFeatures | dict[str, Any] | None,
        name: str,
        default: bool = False,
    ) -> bool:
        if features is None:
            return default

        if isinstance(features, dict):
            value = features.get(name)
            if value is None and isinstance(features.get("metadata"), dict):
                value = features["metadata"].get(name)
            return self._coerce_bool(value, default)

        value = getattr(features, name, None)
        if value is None:
            metadata = getattr(features, "metadata", None)
            if isinstance(metadata, dict):
                value = metadata.get(name)

        return self._coerce_bool(value, default)

    def _feature_str(
        self,
        features: SpoofingFeatures | dict[str, Any] | None,
        name: str,
        default: str = "",
    ) -> str:
        if features is None:
            return default

        if isinstance(features, dict):
            value = features.get(name)
            if value is None and isinstance(features.get("metadata"), dict):
                value = features["metadata"].get(name)
            return str(value or default)

        value = getattr(features, name, None)
        if value is None:
            metadata = getattr(features, "metadata", None)
            if isinstance(metadata, dict):
                value = metadata.get(name)

        return str(value or default)

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