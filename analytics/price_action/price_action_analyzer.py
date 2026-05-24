from __future__ import annotations
from core.logger import get_logger

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler
from analytics.market_state_contract import MarketStateSnapshotSource, snapshot_candles

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.fair_value_gap import FairValueGapAnalyzer, FairValueGapConfig
from analytics.price_action.liquidity_levels import (
    LiquidityLevelsAnalyzer,
    LiquidityLevelsConfig,
)
from analytics.price_action.market_structure import (
    MarketStructureAnalyzer,
    MarketStructureConfig,
)
from analytics.price_action.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    FairValueGapState,
    LiquidityState,
    MarketStructureState,
    PriceActionCompositeState,
    SupportResistanceState,
    TrendState,
)
from analytics.price_action.support_resistance import (
    SupportResistanceAnalyzer,
    SupportResistanceConfig,
)
from analytics.price_action.trend import TrendAnalyzer, TrendConfig


@dataclass(slots=True)
class PriceActionAnalyzerConfig(BasePriceActionConfig):
    """
    Facade config for analytics.price_action.

    This config controls orchestration only. Domain-specific calculations remain
    inside individual analyzers and their own config models.

    Facade scope:
        exchange + market_type + symbol + timeframe
    """

    emit_events: bool = True
    event_namespace: str = "analytics.price_action"
    publish_snapshots: bool = False
    snapshot_interval_seconds: float | None = None

    # Facade itself should not consume market candles directly.
    # In the state-driven pipeline MarketScheduler feeds process_market_snapshot();
    # child analyzers are driven by this facade and must not subscribe to candle
    # events independently.
    subscribe_market_candles: bool = False

    # State-driven live mode. When market_scheduler is injected, the facade
    # registers itself as a bounded snapshot evaluator. evaluate_on_register also
    # schedules one managed initial pass, which covers the common startup order
    # where Parquet restore filled MarketStateStore before this analyzer was
    # registered and dirty scopes were already drained.
    register_market_scheduler_evaluator: bool = True
    evaluate_on_register: bool = True
    initial_evaluation_job_timeout_seconds: float = 30.0

    auto_register_modules: bool = True
    shutdown_child_modules: bool = True
    reset_child_modules: bool = True

    publish_on_module_update: bool = True
    publish_composite_snapshot_on_module_update: bool = False

    enable_market_structure: bool = True
    enable_support_resistance: bool = True
    enable_fair_value_gap: bool = True
    enable_liquidity_levels: bool = True
    enable_trend: bool = True

    market_structure_updated_topic: str = "analytics.price_action.market_structure.updated"
    support_resistance_updated_topic: str = "analytics.price_action.support_resistance.updated"
    fair_value_gap_updated_topic: str = "analytics.price_action.fair_value_gap.updated"
    liquidity_levels_updated_topic: str = "analytics.price_action.liquidity_levels.updated"
    trend_updated_topic: str = "analytics.price_action.trend.updated"

    market_structure_config: MarketStructureConfig | None = None
    support_resistance_config: SupportResistanceConfig | None = None
    fair_value_gap_config: FairValueGapConfig | None = None
    liquidity_levels_config: LiquidityLevelsConfig | None = None
    trend_config: TrendConfig | None = None

    def validate(self) -> None:
        """
        Validate facade infrastructure settings.

        Notes:
        - Uses explicit BasePriceActionConfig.validate(self), not zero-arg
          super(), because this config is a slotted dataclass.
        - Normalizes child update topics before checking emptiness.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "validate", _analytics_args)
        except Exception:
            pass
        BasePriceActionConfig.validate(self)

        self.market_structure_updated_topic = self._normalize_topic(
            self.market_structure_updated_topic
        )
        self.support_resistance_updated_topic = self._normalize_topic(
            self.support_resistance_updated_topic
        )
        self.fair_value_gap_updated_topic = self._normalize_topic(
            self.fair_value_gap_updated_topic
        )
        self.liquidity_levels_updated_topic = self._normalize_topic(
            self.liquidity_levels_updated_topic
        )
        self.trend_updated_topic = self._normalize_topic(self.trend_updated_topic)

        enabled_modules = (
            self.enable_market_structure,
            self.enable_support_resistance,
            self.enable_fair_value_gap,
            self.enable_liquidity_levels,
            self.enable_trend,
        )

        if self.initial_evaluation_job_timeout_seconds <= 0:
            raise ValueError("initial_evaluation_job_timeout_seconds must be > 0")

        self.register_market_scheduler_evaluator = bool(self.register_market_scheduler_evaluator)
        self.evaluate_on_register = bool(self.evaluate_on_register)

        if self.auto_register_modules and not any(enabled_modules):
            raise ValueError("at least one price action module must be enabled")

        if self.enable_market_structure and not self.market_structure_updated_topic:
            raise ValueError("market_structure_updated_topic must not be empty")

        if self.enable_support_resistance and not self.support_resistance_updated_topic:
            raise ValueError("support_resistance_updated_topic must not be empty")

        if self.enable_fair_value_gap and not self.fair_value_gap_updated_topic:
            raise ValueError("fair_value_gap_updated_topic must not be empty")

        if self.enable_liquidity_levels and not self.liquidity_levels_updated_topic:
            raise ValueError("liquidity_levels_updated_topic must not be empty")

        if self.enable_trend and not self.trend_updated_topic:
            raise ValueError("trend_updated_topic must not be empty")


class PriceActionAnalyzer(BasePriceActionModule[PriceActionCompositeState]):
    """
    Facade / orchestrator for analytics.price_action.

    Responsibilities:
    - own and register child price action analyzers;
    - listen only to enabled child *.updated events;
    - aggregate child states into PriceActionCompositeState;
    - publish analytics.price_action.updated / snapshot / reset;
    - expose one facade snapshot for strategy, dashboard and storage layers.

    This class must not duplicate domain calculations from child analyzers.

    Correct input flow:
        exchange adapters
            -> market.candle
            -> CandlesCache
            -> market.candle.closed / market.candles.updated
            -> child price_action analyzers
            -> analytics.price_action.<module>.updated
            -> PriceActionAnalyzer facade
            -> analytics.price_action.updated

    Scope:
        exchange + market_type + symbol + timeframe
    """

    _MODULE_ORDER: tuple[str, ...] = (
        "market_structure",
        "support_resistance",
        "fair_value_gap",
        "liquidity_levels",
        "trend",
    )

    def __init__(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        *,
        event_bus: EventBus,
        exchange: str = DEFAULT_EXCHANGE,
        market_type: str = DEFAULT_MARKET_TYPE,
        exchange_symbol: str | None = None,
        scheduler: Scheduler | None = None,
        config: PriceActionAnalyzerConfig | None = None,
        market_structure: MarketStructureAnalyzer | None = None,
        support_resistance: SupportResistanceAnalyzer | None = None,
        fair_value_gap: FairValueGapAnalyzer | None = None,
        liquidity_levels: LiquidityLevelsAnalyzer | None = None,
        trend: TrendAnalyzer | None = None,
        market_state_store: Any | None = None,
        market_scheduler: Any | None = None,
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
        resolved_config = config or PriceActionAnalyzerConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            market_type=market_type,
            exchange_symbol=exchange_symbol,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action",
        )

        self.config: PriceActionAnalyzerConfig = resolved_config
        self._market_state_store = market_state_store
        self._market_scheduler = market_scheduler
        self._market_scheduler_evaluator_name: str | None = None
        self._initial_evaluation_job_id: str | None = None
        self._initial_market_state_evaluation_completed = False
        self._initial_market_state_evaluation_running = False
        self._state_snapshot_source = (
            MarketStateSnapshotSource(market_state_store) if market_state_store is not None else None
        )

        # Enable flags are authoritative. Explicitly injected children must not
        # bypass disabled module configuration.
        self.market_structure = (
            market_structure
            if market_structure is not None
            else self._build_market_structure_analyzer()
        ) if self.config.enable_market_structure else None

        self.support_resistance = (
            support_resistance
            if support_resistance is not None
            else self._build_support_resistance_analyzer()
        ) if self.config.enable_support_resistance else None

        self.fair_value_gap = (
            fair_value_gap
            if fair_value_gap is not None
            else self._build_fair_value_gap_analyzer()
        ) if self.config.enable_fair_value_gap else None

        self.liquidity_levels = (
            liquidity_levels
            if liquidity_levels is not None
            else self._build_liquidity_levels_analyzer()
        ) if self.config.enable_liquidity_levels else None

        self.trend = (
            trend
            if trend is not None
            else self._build_trend_analyzer()
        ) if self.config.enable_trend else None

        self._validate_child_scopes()

        self._child_update_counts: dict[str, int] = {
            module_name: 0 for module_name in self._MODULE_ORDER
        }
        self._last_child_payloads: dict[str, dict[str, Any]] = {}
        self._state_version = 0
        # State-driven snapshots contain a rolling candle window.  The facade is
        # incremental, so feeding the full window on every MarketScheduler tick
        # duplicates candles inside child modules and breaks pivots/FVG/trend.
        # Keep one monotonic watermark per scoped analyzer and process only new
        # closed candles.  Initial evaluation still hydrates the full restored
        # closed window.
        self._last_processed_candle_open_time_ms: int | None = None

        self._state = self._new_composite_state()
        self._refresh_composite_state(advance_version=True)

        self.logger.info(
            "Initialized PriceActionAnalyzer facade",
            extra={
                **self._log_scope_extra(),
                "config": asdict(self.config),
                "enabled_modules": self._enabled_module_names(),
            },
        )

    # ------------------------------------------------------------------
    # State-driven child config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_driven_child_config(config: Any, factory: Any) -> Any:
        """Return a child config that is driven only by this facade.

        Child analyzers keep all domain thresholds/settings from the provided
        config, but candle EventBus subscriptions are disabled. In the
        high-load pipeline MarketStateStore/MarketScheduler supplies a coherent
        candle snapshot to PriceActionAnalyzer, and this facade calls
        child.update(candles=...) exactly once per evaluator invocation.
        """
        child_config = config if config is not None else factory()
        if hasattr(child_config, "subscribe_market_candles"):
            child_config.subscribe_market_candles = False
        return child_config

    # ------------------------------------------------------------------
    # Child analyzer construction
    # ------------------------------------------------------------------

    def _build_market_structure_analyzer(self) -> MarketStructureAnalyzer | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_market_structure_analyzer", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_market_structure:
            return None

        return MarketStructureAnalyzer(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self._state_driven_child_config(
                self.config.market_structure_config,
                MarketStructureConfig,
            ),
        )

    def _build_support_resistance_analyzer(self) -> SupportResistanceAnalyzer | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_support_resistance_analyzer", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_support_resistance:
            return None

        return SupportResistanceAnalyzer(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self._state_driven_child_config(
                self.config.support_resistance_config,
                SupportResistanceConfig,
            ),
        )

    def _build_fair_value_gap_analyzer(self) -> FairValueGapAnalyzer | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_fair_value_gap_analyzer", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_fair_value_gap:
            return None

        return FairValueGapAnalyzer(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self._state_driven_child_config(
                self.config.fair_value_gap_config,
                FairValueGapConfig,
            ),
        )

    def _build_liquidity_levels_analyzer(self) -> LiquidityLevelsAnalyzer | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_liquidity_levels_analyzer", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_liquidity_levels:
            return None

        return LiquidityLevelsAnalyzer(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self._state_driven_child_config(
                self.config.liquidity_levels_config,
                LiquidityLevelsConfig,
            ),
        )

    def _build_trend_analyzer(self) -> TrendAnalyzer | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_trend_analyzer", _analytics_args)
        except Exception:
            pass
        if not self.config.enable_trend:
            return None

        return TrendAnalyzer(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self._state_driven_child_config(
                self.config.trend_config,
                TrendConfig,
            ),
        )

    # ------------------------------------------------------------------
    # MarketScheduler integration
    # ------------------------------------------------------------------

    def _market_scheduler_evaluator_id(self) -> str:
        return (
            f"price_action:{self.symbol}:{self.timeframe}:"
            f"{self.exchange}:{self.market_type}"
        )

    def _register_market_scheduler_evaluator(self) -> None:
        if not self.config.register_market_scheduler_evaluator:
            return

        market_scheduler = getattr(self, "_market_scheduler", None)
        if market_scheduler is None:
            self.logger.warning(
                "PriceActionAnalyzer MarketScheduler is not configured; "
                "live state-driven candle snapshots will not update price_action",
                extra=self._log_scope_extra(),
            )
            return

        register_evaluator = getattr(market_scheduler, "register_evaluator", None)
        if not callable(register_evaluator):
            self.logger.warning(
                "Configured market_scheduler has no register_evaluator(); "
                "live state-driven candle snapshots will not update price_action",
                extra=self._log_scope_extra(),
            )
            return

        name = self._market_scheduler_evaluator_id()
        register_evaluator(
            name=name,
            callback=self.process_market_snapshot,
            enabled=True,
            metadata={
                "domain": "price_action",
                "exchange": self.exchange,
                "market_type": self.market_type,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "dirty_reasons": ("candle", "candle_closed", "warmup"),
            },
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
            dirty_reasons=("candle", "candle_closed", "warmup"),
        )
        self._market_scheduler_evaluator_name = name
        self.logger.info(
            "PriceActionAnalyzer registered MarketScheduler evaluator",
            extra={**self._log_scope_extra(), "evaluator": name},
        )

    def _unregister_market_scheduler_evaluator(self) -> None:
        market_scheduler = getattr(self, "_market_scheduler", None)
        name = getattr(self, "_market_scheduler_evaluator_name", None)
        if market_scheduler is None or not name:
            return

        unregister_evaluator = getattr(market_scheduler, "unregister_evaluator", None)
        if callable(unregister_evaluator):
            try:
                unregister_evaluator(name)
            except Exception:
                self.logger.exception(
                    "Failed to unregister PriceActionAnalyzer MarketScheduler evaluator",
                    extra={**self._log_scope_extra(), "evaluator": name},
                )
        self._market_scheduler_evaluator_name = None

    def _register_initial_market_state_evaluation_job(self) -> None:
        if not self.config.evaluate_on_register:
            return
        if self._state_snapshot_source is None:
            return
        if self.scheduler is None:
            self.logger.warning(
                "Initial price_action evaluation requested but Scheduler is not configured",
                extra=self._log_scope_extra(),
            )
            return
        if self._initial_market_state_evaluation_completed:
            return
        if self._initial_market_state_evaluation_running:
            return
        if self._initial_evaluation_job_id is not None:
            return

        job_name = (
            f"analytics.price_action.initial_evaluation."
            f"{self.exchange}.{self.market_type}.{self.symbol.lower()}.{self.timeframe}"
        )
        self._initial_evaluation_job_id = self.scheduler.add_interval_job(
            name=job_name,
            func=self._initial_market_state_evaluation_job,
            interval=3600.0,
            run_immediately=True,
            max_retries=0,
            retry_delay=1.0,
            timeout=self.config.initial_evaluation_job_timeout_seconds,
            allow_overlap=False,
            enabled=True,
        )
        self._scheduled_job_ids.append(self._initial_evaluation_job_id)
        self.logger.info(
            "PriceActionAnalyzer scheduled initial MarketState evaluation",
            extra={**self._log_scope_extra(), "job_id": self._initial_evaluation_job_id},
        )

    async def _initial_market_state_evaluation_job(self) -> None:
        job_id = self._initial_evaluation_job_id
        if self._initial_market_state_evaluation_completed:
            return
        if self._initial_market_state_evaluation_running:
            return

        self._initial_market_state_evaluation_running = True
        try:
            result = await self.process_market_state_snapshot()
            self._initial_market_state_evaluation_completed = True
            self.logger.info(
                "PriceActionAnalyzer initial MarketState evaluation completed",
                extra={**self._log_scope_extra(), "job_id": job_id, "result": result},
            )
        finally:
            self._initial_market_state_evaluation_running = False
            if job_id is not None and self.scheduler is not None:
                try:
                    # Do not remove a job from inside its own running callback: the
                    # core scheduler logs this as "Removing running job" and some
                    # implementations can reschedule the callback before removal is
                    # fully committed.  Disabling is enough for this one-shot warmup;
                    # unregister()/shutdown() will clean up module-owned jobs later.
                    self.scheduler.disable_job(job_id)
                except Exception:
                    self.logger.exception(
                        "Failed to disable initial price_action evaluation job",
                        extra={**self._log_scope_extra(), "job_id": job_id},
                    )

    # ------------------------------------------------------------------
    # Registration / lifecycle
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Register facade subscriptions and optionally child analyzers.

        The operation is rollback-safe: if any child module fails during
        auto-registration, facade subscriptions and already-registered children
        are cleaned up so the facade is not left half-registered.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "register", _analytics_args)
        except Exception:
            pass
        if self._registered:
            self.logger.warning(
                "PriceActionAnalyzer already registered",
                extra=self._log_scope_extra(),
            )
            return

        registered_children: list[BasePriceActionModule[Any]] = []

        try:
            super().register()

            for topic, handler, name in self._enabled_child_update_subscriptions():
                self._subscribe(topic, handler, name=name)

            if self.config.auto_register_modules:
                for module in self._iter_enabled_modules():
                    module.register()
                    registered_children.append(module)

            self._register_market_scheduler_evaluator()
            self._register_initial_market_state_evaluation_job()

        except Exception:
            for module in reversed(registered_children):
                try:
                    module.unregister()
                except Exception:
                    self.logger.exception(
                        "Failed to rollback registered child price action module",
                        extra={
                            **self._log_scope_extra(),
                            "child_module": getattr(
                                module,
                                "module_name",
                                module.__class__.__name__,
                            ),
                        },
                    )

            try:
                self.unregister()
            except Exception:
                self._subscriptions.clear()
                self._scheduled_job_ids.clear()
                self._registered = False

            raise

        self.logger.info(
            "PriceActionAnalyzer facade registered",
            extra={
                **self._log_scope_extra(),
                "enabled_modules": self._enabled_module_names(),
                "subscriptions": len(self._subscriptions),
            },
        )

    def unregister(self) -> None:
        """Unregister facade subscriptions, child scheduler jobs and MarketScheduler evaluator."""
        self._unregister_market_scheduler_evaluator()
        super().unregister()

    async def shutdown(self) -> None:
        """
        Shutdown facade and optionally child modules.

        EventBus and Scheduler lifecycles remain owned by application/core.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "shutdown", _analytics_args)
        except Exception:
            pass
        self._unregister_market_scheduler_evaluator()

        if self.config.shutdown_child_modules:
            for module in self._iter_enabled_modules():
                try:
                    await module.shutdown()
                except Exception:
                    self.logger.exception(
                        "Failed to shutdown child price action module",
                        extra={
                            **self._log_scope_extra(),
                            "child_module": module.module_name,
                        },
                    )

        await super().shutdown()


    # ------------------------------------------------------------------
    # Candle-window normalization / de-duplication
    # ------------------------------------------------------------------

    @staticmethod
    def _candle_mapping(raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        if hasattr(raw, "to_dict") and callable(raw.to_dict):
            try:
                value = raw.to_dict()
                if isinstance(value, Mapping):
                    return dict(value)
            except Exception:
                pass
        result: dict[str, Any] = {}
        for key in (
            "timeframe", "open_time_ms", "close_time_ms", "timestamp_ms",
            "open", "high", "low", "close", "volume", "is_closed", "metadata",
        ):
            value = getattr(raw, key, None)
            if value is not None:
                result[key] = value
        return result

    @staticmethod
    def _candle_int(raw: Mapping[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = raw.get(key)
            if value is None or value == "":
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _candle_is_closed(raw: Mapping[str, Any]) -> bool:
        value = raw.get("is_closed", raw.get("closed", raw.get("x", True)))
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "closed"}
        return bool(value)

    def _new_closed_candles_from_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        candles = [self._candle_mapping(item) for item in snapshot_candles(snapshot, timeframe=self.timeframe)]
        normalized: list[tuple[int, dict[str, Any]]] = []
        for candle in candles:
            if not candle:
                continue
            # Price-action modules should react to stable candle-close data.  Open
            # candle updates can revise OHLC and create false pivots/signals.
            if not self._candle_is_closed(candle):
                continue
            open_time_ms = self._candle_int(candle, "open_time_ms", "open_time", "start", "t", "timestamp_ms", "timestamp")
            if open_time_ms is None:
                continue
            if (
                self._last_processed_candle_open_time_ms is not None
                and open_time_ms <= self._last_processed_candle_open_time_ms
            ):
                continue
            candle.setdefault("exchange", self.exchange)
            candle.setdefault("market_type", self.market_type)
            candle.setdefault("symbol", self.symbol)
            candle.setdefault("exchange_symbol", self.exchange_symbol)
            candle.setdefault("timeframe", self.timeframe)
            normalized.append((open_time_ms, candle))

        normalized.sort(key=lambda item: item[0])
        # Deduplicate by open time inside one snapshot.  Keep the latest dict for
        # the candle, but preserve chronological processing order.
        deduped: dict[int, dict[str, Any]] = {}
        for open_time_ms, candle in normalized:
            deduped[open_time_ms] = candle
        return [deduped[key] for key in sorted(deduped)]

    def _mark_candles_processed(self, candles: list[Mapping[str, Any]]) -> None:
        if not candles:
            return
        latest = self._last_processed_candle_open_time_ms
        for candle in candles:
            open_time_ms = self._candle_int(candle, "open_time_ms", "open_time", "start", "t", "timestamp_ms", "timestamp")
            if open_time_ms is not None and (latest is None or open_time_ms > latest):
                latest = open_time_ms
        self._last_processed_candle_open_time_ms = latest

    async def process_market_state_snapshot(self) -> dict[str, Any]:
        """Evaluate this price-action scope from MarketStateStore candles.

        This is the current state-driven input contract.  It replaces
        direct consumption of high-frequency market.candles.updated events.
        Child analyzers keep their domain logic; this facade supplies them with
        a consistent candle snapshot and then publishes analytics.price_action.updated.
        """
        source = getattr(self, "_state_snapshot_source", None)
        if source is None:
            return {"processed": False, "reason": "market_state_store_not_configured"}

        snapshot = await source.snapshot(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        candles = self._new_closed_candles_from_snapshot(snapshot) if snapshot is not None else []
        if not candles:
            return {"processed": False, "reason": "no_new_closed_candles"}

        updated: list[str] = []
        for module_name, module in (
            ("market_structure", self.market_structure),
            ("support_resistance", self.support_resistance),
            ("fair_value_gap", self.fair_value_gap),
            ("liquidity_levels", self.liquidity_levels),
            ("trend", self.trend),
        ):
            if module is None:
                continue
            update = getattr(module, "update", None)
            if not callable(update):
                continue
            try:
                result = update(candles=candles)
            except TypeError:
                result = update(candles)
            self._child_update_counts[module_name] = self._child_update_counts.get(module_name, 0) + 1
            self._last_child_payloads[module_name] = result if isinstance(result, dict) else {"result": result}
            updated.append(module_name)

        await self.publish_composite_update(
            updated_module=updated[-1] if updated else None,
            source_topic="market_state.snapshot",
        )
        self._mark_candles_processed(candles)
        return {"processed": True, "updated_modules": updated, "candles": len(candles)}

    async def process_market_snapshot(self, snapshot: Any) -> dict[str, Any]:
        """MarketScheduler-compatible evaluator callback.

        Each PriceActionAnalyzer instance is scope-bound.  The MarketScheduler
        calls every evaluator for every dirty snapshot, so this method must
        ignore snapshots outside this analyzer's own scope and must never mutate
        ``self.exchange`` / ``self.symbol`` / ``self.timeframe``.  Mutating the
        facade scope makes child module state leak between symbols and causes
        PriceActionCompositeState scope validation errors.
        """
        if snapshot is None:
            return {"processed": False, "reason": "missing_snapshot"}

        scope = getattr(snapshot, "scope", None)
        snapshot_exchange = getattr(scope, "exchange", None) or getattr(snapshot, "exchange", None)
        snapshot_market_type = getattr(scope, "market_type", None) or getattr(snapshot, "market_type", None)
        snapshot_symbol = getattr(scope, "symbol", None) or getattr(snapshot, "symbol", None)
        snapshot_timeframe = getattr(scope, "timeframe", None) or getattr(snapshot, "timeframe", None)

        if (
            str(snapshot_exchange or "").lower() != str(self.exchange or "").lower()
            or str(snapshot_market_type or "").lower() != str(self.market_type or "").lower()
            or str(snapshot_symbol or "").upper() != str(self.symbol or "").upper()
            or str(snapshot_timeframe or "") != str(self.timeframe or "")
        ):
            return {
                "processed": False,
                "reason": "scope_mismatch",
                "snapshot_scope": {
                    "exchange": snapshot_exchange,
                    "market_type": snapshot_market_type,
                    "symbol": snapshot_symbol,
                    "timeframe": snapshot_timeframe,
                },
                "analyzer_scope": {
                    "exchange": self.exchange,
                    "market_type": self.market_type,
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                },
            }

        candles = self._new_closed_candles_from_snapshot(snapshot)
        if not candles:
            return {"processed": False, "reason": "no_new_closed_candles"}

        updated: list[str] = []
        for module_name, module in (
            ("market_structure", self.market_structure),
            ("support_resistance", self.support_resistance),
            ("fair_value_gap", self.fair_value_gap),
            ("liquidity_levels", self.liquidity_levels),
            ("trend", self.trend),
        ):
            if module is None:
                continue
            update = getattr(module, "update", None)
            if not callable(update):
                continue
            try:
                result = update(candles=candles)
            except TypeError:
                result = update(candles)
            self._child_update_counts[module_name] = self._child_update_counts.get(module_name, 0) + 1
            self._last_child_payloads[module_name] = result if isinstance(result, dict) else {"result": result}
            updated.append(module_name)

        await self.publish_composite_update(
            updated_module=updated[-1] if updated else None,
            source_topic="market_state.snapshot",
        )
        self._mark_candles_processed(candles)
        return {"processed": True, "updated_modules": updated, "candles": len(candles)}

    # ------------------------------------------------------------------
    # EventBus handlers for child module updates
    # ------------------------------------------------------------------

    async def on_market_structure_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_market_structure_updated", _analytics_args)
        except Exception:
            pass
        await self._handle_child_update("market_structure", event)

    async def on_support_resistance_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_support_resistance_updated", _analytics_args)
        except Exception:
            pass
        await self._handle_child_update("support_resistance", event)

    async def on_fair_value_gap_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_fair_value_gap_updated", _analytics_args)
        except Exception:
            pass
        await self._handle_child_update("fair_value_gap", event)

    async def on_liquidity_levels_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_liquidity_levels_updated", _analytics_args)
        except Exception:
            pass
        await self._handle_child_update("liquidity_levels", event)

    async def on_trend_updated(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_trend_updated", _analytics_args)
        except Exception:
            pass
        await self._handle_child_update("trend", event)

    async def _handle_child_update(self, module_name: str, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_handle_child_update", _analytics_args)
        except Exception:
            pass
        active_children = self.get_child_analyzers()
        if module_name not in active_children:
            self.logger.warning(
                "Ignoring update from disabled or unknown price action child module",
                extra={
                    **self._log_scope_extra(),
                    "child_module": module_name,
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        if not isinstance(event.payload, Mapping):
            self.logger.warning(
                "PriceActionAnalyzer received invalid child update payload",
                extra={
                    **self._log_scope_extra(),
                    "child_module": module_name,
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        if self.config.require_event_scope and not self._event_matches_module_scope(event):
            self.logger.debug(
                "PriceActionAnalyzer child update skipped because scope does not match",
                extra={
                    **self._log_scope_extra(),
                    "child_module": module_name,
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        self._child_update_counts[module_name] = (
            self._child_update_counts.get(module_name, 0) + 1
        )
        self._last_child_payloads[module_name] = dict(event.payload)

        self._refresh_composite_state(
            updated_module=module_name,
            source_topic=event.topic,
            advance_version=True,
        )

        if self.config.publish_on_module_update:
            await self.publish_composite_update(
                updated_module=module_name,
                source_topic=event.topic,
                correlation_id=event.correlation_id,
            )

        if self.config.publish_composite_snapshot_on_module_update:
            await self.publish_snapshot(correlation_id=event.correlation_id)

    # ------------------------------------------------------------------
    # Public facade API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        # Reset children first. If a child reset fails, facade counters/state are
        # intentionally left unchanged instead of becoming partially reset.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset", _analytics_args)
        except Exception:
            pass
        if self.config.reset_child_modules:
            for module in self._iter_enabled_modules():
                module.reset()

        self._child_update_counts = {
            module_name: 0 for module_name in self._MODULE_ORDER
        }
        self._last_child_payloads.clear()

        self._state = self._new_composite_state()
        self._refresh_composite_state(advance_version=True)

        self.logger.info(
            "PriceActionAnalyzer facade reset",
            extra={
                **self._log_scope_extra(),
                "state_version": self._state_version,
            },
        )

    async def reset_and_publish(self, *, correlation_id: str | None = None) -> None:
        """
        Async reset helper for EventBus-aware callers.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reset_and_publish", _analytics_args)
        except Exception:
            pass
        self.reset()

        await self._emit_event(
            self._build_event_name("reset"),
            {
                "state": self.snapshot(),
                "state_version": self._state_version,
                "reset_at": self._now_utc().isoformat(),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

    def get_state(self) -> PriceActionCompositeState:
        # Read-only access must not advance state_version.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_state", _analytics_args)
        except Exception:
            pass
        self._refresh_composite_state(advance_version=False)
        return self._state

    def snapshot(self) -> dict[str, Any]:
        # Read-only snapshots must not create fake version changes.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "snapshot", _analytics_args)
        except Exception:
            pass
        self._refresh_composite_state(advance_version=False)

        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "state_version": self._state_version,
                "enabled_modules": self._enabled_module_names(),
                "registered_modules": self._registered_module_names(),
                "child_update_counts": dict(self._child_update_counts),
                "last_child_update_modules": sorted(self._last_child_payloads.keys()),
                "config": self._serialize_config(),
            },
        )

    def get_child_analyzers(self) -> dict[str, BasePriceActionModule[Any]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_child_analyzers", _analytics_args)
        except Exception:
            pass
        modules: dict[str, BasePriceActionModule[Any] | None] = {
            "market_structure": self.market_structure,
            "support_resistance": self.support_resistance,
            "fair_value_gap": self.fair_value_gap,
            "liquidity_levels": self.liquidity_levels,
            "trend": self.trend,
        }

        return {
            module_name: module
            for module_name, module in modules.items()
            if module is not None and self._is_module_enabled(module_name)
        }

    def get_market_structure_state(self) -> MarketStructureState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_market_structure_state", _analytics_args)
        except Exception:
            pass
        return (
            self.market_structure.get_state()
            if self.market_structure is not None
            else None
        )

    def get_support_resistance_state(self) -> SupportResistanceState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_support_resistance_state", _analytics_args)
        except Exception:
            pass
        return (
            self.support_resistance.get_state()
            if self.support_resistance is not None
            else None
        )

    def get_fair_value_gap_state(self) -> FairValueGapState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_fair_value_gap_state", _analytics_args)
        except Exception:
            pass
        return (
            self.fair_value_gap.get_state()
            if self.fair_value_gap is not None
            else None
        )

    def get_liquidity_state(self) -> LiquidityState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_liquidity_state", _analytics_args)
        except Exception:
            pass
        return (
            self.liquidity_levels.get_state()
            if self.liquidity_levels is not None
            else None
        )

    def get_trend_state(self) -> TrendState | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_trend_state", _analytics_args)
        except Exception:
            pass
        return self.trend.get_state() if self.trend is not None else None

    async def publish_composite_update(
        self,
        *,
        updated_module: str | None = None,
        source_topic: str | None = None,
        correlation_id: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "publish_composite_update", _analytics_args)
        except Exception:
            pass
        self._refresh_composite_state(
            updated_module=updated_module,
            source_topic=source_topic,
            advance_version=True,
        )

        return await self._emit_event(
            self._build_event_name("updated"),
            {
                "state": self.snapshot(),
                "updated_module": updated_module,
                "source_topic": source_topic,
                "state_version": self._state_version,
                "updated_at": self._now_utc().isoformat(),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

    # ------------------------------------------------------------------
    # Composite state refresh
    # ------------------------------------------------------------------

    def _refresh_composite_state(
        self,
        *,
        updated_module: str | None = None,
        source_topic: str | None = None,
        advance_version: bool = False,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_composite_state", _analytics_args)
        except Exception:
            pass
        if advance_version:
            self._state_version += 1

        market_structure_state = self._state_or_none_if_wrong_scope(
            "market_structure",
            self.get_market_structure_state(),
        )
        support_resistance_state = self._state_or_none_if_wrong_scope(
            "support_resistance",
            self.get_support_resistance_state(),
        )
        fair_value_gap_state = self._state_or_none_if_wrong_scope(
            "fair_value_gap",
            self.get_fair_value_gap_state(),
        )
        liquidity_state = self._state_or_none_if_wrong_scope(
            "liquidity_levels",
            self.get_liquidity_state(),
        )
        trend_state = self._state_or_none_if_wrong_scope(
            "trend",
            self.get_trend_state(),
        )

        last_price, last_update = self._resolve_latest_price_and_update(
            market_structure_state=market_structure_state,
            support_resistance_state=support_resistance_state,
            fair_value_gap_state=fair_value_gap_state,
            liquidity_state=liquidity_state,
            trend_state=trend_state,
        )

        self._state = PriceActionCompositeState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            last_price=last_price,
            last_update=last_update,
            market_structure=market_structure_state,
            support_resistance=support_resistance_state,
            fair_value_gap=fair_value_gap_state,
            liquidity=liquidity_state,
            trend=trend_state,
            metadata={
                **self.scope_payload,
                "state_version": self._state_version,
                "updated_module": updated_module,
                "source_topic": source_topic,
                "enabled_modules": self._enabled_module_names(),
                "child_update_counts": dict(self._child_update_counts),
                "last_refreshed_at": self._now_utc().isoformat(),
            },
        )

    def _resolve_latest_price_and_update(
        self,
        *,
        market_structure_state: MarketStructureState | None,
        support_resistance_state: SupportResistanceState | None,
        fair_value_gap_state: FairValueGapState | None,
        liquidity_state: LiquidityState | None,
        trend_state: TrendState | None,
    ) -> tuple[float | None, datetime | None]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_latest_price_and_update", _analytics_args)
        except Exception:
            pass
        candidates: list[tuple[datetime, float]] = []

        for state in (
            market_structure_state,
            support_resistance_state,
            fair_value_gap_state,
            liquidity_state,
            trend_state,
        ):
            if state is None:
                continue

            last_update = getattr(state, "last_update", None)
            last_price = getattr(state, "last_price", None)

            if isinstance(last_update, datetime) and last_price is not None:
                candidates.append((last_update, float(last_price)))

        if candidates:
            latest_update, latest_price = max(candidates, key=lambda item: item[0])
            return latest_price, latest_update

        for state in (
            trend_state,
            fair_value_gap_state,
            liquidity_state,
            support_resistance_state,
            market_structure_state,
        ):
            if state is None:
                continue

            last_price = getattr(state, "last_price", None)
            if last_price is not None:
                return float(last_price), getattr(state, "last_update", None)

        return None, None

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _new_composite_state(self) -> PriceActionCompositeState:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_new_composite_state", _analytics_args)
        except Exception:
            pass
        return PriceActionCompositeState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
        )

    def _validate_child_scopes(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_child_scopes", _analytics_args)
        except Exception:
            pass
        for module_name, module in self.get_child_analyzers().items():
            self._validate_child_scope(module_name, module)

    def _validate_child_scope(
        self,
        module_name: str,
        module: BasePriceActionModule[Any],
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_child_scope", _analytics_args)
        except Exception:
            pass
        child_key = getattr(module, "key", None)
        if child_key != self.key:
            raise ValueError(
                f"Child price action module scope mismatch: "
                f"module={module_name}, child_key={child_key}, facade_key={self.key}"
            )

    def _state_or_none_if_wrong_scope(
        self,
        module_name: str,
        state: Any,
    ) -> Any | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_state_or_none_if_wrong_scope", _analytics_args)
        except Exception:
            pass
        if state is None:
            return None

        state_key = getattr(state, "key", None)
        if state_key == self.key:
            return state

        self.logger.warning(
            "Ignoring child state because scope does not match facade",
            extra={
                **self._log_scope_extra(),
                "child_module": module_name,
                "child_key": list(state_key) if state_key else None,
            },
        )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enabled_child_update_subscriptions(
        self,
    ) -> tuple[tuple[str, Any, str], ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_enabled_child_update_subscriptions", _analytics_args)
        except Exception:
            pass
        subscriptions: list[tuple[str, Any, str]] = []

        if self.market_structure is not None:
            subscriptions.append(
                (
                    self.config.market_structure_updated_topic,
                    self.on_market_structure_updated,
                    f"{self.module_name}.on_market_structure_updated",
                )
            )
        if self.support_resistance is not None:
            subscriptions.append(
                (
                    self.config.support_resistance_updated_topic,
                    self.on_support_resistance_updated,
                    f"{self.module_name}.on_support_resistance_updated",
                )
            )
        if self.fair_value_gap is not None:
            subscriptions.append(
                (
                    self.config.fair_value_gap_updated_topic,
                    self.on_fair_value_gap_updated,
                    f"{self.module_name}.on_fair_value_gap_updated",
                )
            )
        if self.liquidity_levels is not None:
            subscriptions.append(
                (
                    self.config.liquidity_levels_updated_topic,
                    self.on_liquidity_levels_updated,
                    f"{self.module_name}.on_liquidity_levels_updated",
                )
            )
        if self.trend is not None:
            subscriptions.append(
                (
                    self.config.trend_updated_topic,
                    self.on_trend_updated,
                    f"{self.module_name}.on_trend_updated",
                )
            )

        return tuple(subscriptions)

    def _is_module_enabled(self, module_name: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_module_enabled", _analytics_args)
        except Exception:
            pass
        return {
            "market_structure": self.config.enable_market_structure,
            "support_resistance": self.config.enable_support_resistance,
            "fair_value_gap": self.config.enable_fair_value_gap,
            "liquidity_levels": self.config.enable_liquidity_levels,
            "trend": self.config.enable_trend,
        }.get(module_name, False)

    def _iter_enabled_modules(self) -> tuple[BasePriceActionModule[Any], ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_iter_enabled_modules", _analytics_args)
        except Exception:
            pass
        return tuple(self.get_child_analyzers().values())

    def _enabled_module_names(self) -> list[str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_enabled_module_names", _analytics_args)
        except Exception:
            pass
        return list(self.get_child_analyzers().keys())

    def _registered_module_names(self) -> list[str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_registered_module_names", _analytics_args)
        except Exception:
            pass
        registered: list[str] = []

        for module_name, module in self.get_child_analyzers().items():
            if getattr(module, "_registered", False):
                registered.append(module_name)

        return registered


__all__ = [
    "PriceActionAnalyzerConfig",
    "PriceActionAnalyzer",
]