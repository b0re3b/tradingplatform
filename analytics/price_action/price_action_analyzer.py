from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.fair_value_gap import FairValueGapAnalyzer, FairValueGapConfig
from analytics.price_action.liquidity_levels import LiquidityLevelsAnalyzer, LiquidityLevelsConfig
from analytics.price_action.market_structure import MarketStructureAnalyzer, MarketStructureConfig
from analytics.price_action.models import (
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

    This config controls orchestration only. Domain-specific logic remains
    inside individual analyzers and their own config models.
    """

    emit_events: bool = True
    event_namespace: str = "analytics.price_action"
    publish_snapshots: bool = False
    snapshot_interval_seconds: float | None = None

    # Facade itself should not consume market candles directly.
    # Child analyzers own candle subscriptions.
    subscribe_market_candles: bool = False

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
    Facade / orchestrator for the analytics.price_action package.

    Responsibilities:
    - own and register child price action analyzers;
    - listen only to enabled child *.updated events;
    - aggregate child states into PriceActionCompositeState;
    - publish analytics.price_action.updated / snapshot / reset;
    - expose one facade snapshot for strategy, dashboard and storage layers.

    It must not duplicate domain calculations from child analyzers.
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
        timeframe: str,
        *,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        config: PriceActionAnalyzerConfig | None = None,
        market_structure: MarketStructureAnalyzer | None = None,
        support_resistance: SupportResistanceAnalyzer | None = None,
        fair_value_gap: FairValueGapAnalyzer | None = None,
        liquidity_levels: LiquidityLevelsAnalyzer | None = None,
        trend: TrendAnalyzer | None = None,
    ) -> None:
        resolved_config = config or PriceActionAnalyzerConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action",
        )

        self.config: PriceActionAnalyzerConfig = resolved_config

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

        self._child_update_counts: dict[str, int] = {
            module_name: 0 for module_name in self._MODULE_ORDER
        }
        self._last_child_payloads: dict[str, dict[str, Any]] = {}
        self._state_version = 0

        self._state = PriceActionCompositeState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        self._refresh_composite_state(advance_version=True)

        self.logger.info(
            "Initialized PriceActionAnalyzer facade",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "config": asdict(self.config),
                "enabled_modules": self._enabled_module_names(),
            },
        )

    # ------------------------------------------------------------------
    # Child analyzer construction
    # ------------------------------------------------------------------

    def _build_market_structure_analyzer(self) -> MarketStructureAnalyzer | None:
        if not self.config.enable_market_structure:
            return None

        return MarketStructureAnalyzer(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config.market_structure_config,
        )

    def _build_support_resistance_analyzer(self) -> SupportResistanceAnalyzer | None:
        if not self.config.enable_support_resistance:
            return None

        return SupportResistanceAnalyzer(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config.support_resistance_config,
        )

    def _build_fair_value_gap_analyzer(self) -> FairValueGapAnalyzer | None:
        if not self.config.enable_fair_value_gap:
            return None

        return FairValueGapAnalyzer(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config.fair_value_gap_config,
        )

    def _build_liquidity_levels_analyzer(self) -> LiquidityLevelsAnalyzer | None:
        if not self.config.enable_liquidity_levels:
            return None

        return LiquidityLevelsAnalyzer(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config.liquidity_levels_config,
        )

    def _build_trend_analyzer(self) -> TrendAnalyzer | None:
        if not self.config.enable_trend:
            return None

        return TrendAnalyzer(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=self.config.trend_config,
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
        if self._registered:
            self.logger.warning(
                "PriceActionAnalyzer already registered",
                extra={"symbol": self.symbol, "timeframe": self.timeframe},
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

        except Exception:
            for module in reversed(registered_children):
                try:
                    module.unregister()
                except Exception:
                    self.logger.exception(
                        "Failed to rollback registered child price action module",
                        extra={
                            "symbol": self.symbol,
                            "timeframe": self.timeframe,
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
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "enabled_modules": self._enabled_module_names(),
                "subscriptions": len(self._subscriptions),
            },
        )

    async def shutdown(self) -> None:
        """
        Shutdown facade and optionally child modules.

        EventBus and Scheduler lifecycles remain owned by the application/core layer.
        """
        if self.config.shutdown_child_modules:
            for module in self._iter_enabled_modules():
                try:
                    await module.shutdown()
                except Exception:
                    self.logger.exception(
                        "Failed to shutdown child price action module",
                        extra={
                            "symbol": self.symbol,
                            "timeframe": self.timeframe,
                            "child_module": module.module_name,
                        },
                    )

        await super().shutdown()

    # ------------------------------------------------------------------
    # EventBus handlers for child module updates
    # ------------------------------------------------------------------

    async def on_market_structure_updated(self, event: Event) -> None:
        await self._handle_child_update("market_structure", event)

    async def on_support_resistance_updated(self, event: Event) -> None:
        await self._handle_child_update("support_resistance", event)

    async def on_fair_value_gap_updated(self, event: Event) -> None:
        await self._handle_child_update("fair_value_gap", event)

    async def on_liquidity_levels_updated(self, event: Event) -> None:
        await self._handle_child_update("liquidity_levels", event)

    async def on_trend_updated(self, event: Event) -> None:
        await self._handle_child_update("trend", event)

    async def _handle_child_update(self, module_name: str, event: Event) -> None:
        active_children = self.get_child_analyzers()
        if module_name not in active_children:
            self.logger.warning(
                "Ignoring update from disabled or unknown price action child module",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
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
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "child_module": module_name,
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        self._child_update_counts[module_name] = self._child_update_counts.get(module_name, 0) + 1
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
        if self.config.reset_child_modules:
            for module in self._iter_enabled_modules():
                module.reset()

        self._child_update_counts = {
            module_name: 0 for module_name in self._MODULE_ORDER
        }
        self._last_child_payloads.clear()

        self._state = PriceActionCompositeState(
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        self._refresh_composite_state(advance_version=True)

        self.logger.info(
            "PriceActionAnalyzer facade reset",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "state_version": self._state_version,
            },
        )

    async def reset_and_publish(self, *, correlation_id: str | None = None) -> None:
        """
        Async reset helper for EventBus-aware callers.
        """
        self.reset()

        await self._emit_event(
            self._build_event_name("reset"),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "state": self.snapshot(),
                "state_version": self._state_version,
                "reset_at": self._now_utc().isoformat(),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

    def get_state(self) -> PriceActionCompositeState:
        # Read-only access must not advance state_version.
        self._refresh_composite_state(advance_version=False)
        return self._state

    def snapshot(self) -> dict[str, Any]:
        # Read-only snapshots must not create fake version changes.
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
        return self.market_structure.get_state() if self.market_structure is not None else None

    def get_support_resistance_state(self) -> SupportResistanceState | None:
        return self.support_resistance.get_state() if self.support_resistance is not None else None

    def get_fair_value_gap_state(self) -> FairValueGapState | None:
        return self.fair_value_gap.get_state() if self.fair_value_gap is not None else None

    def get_liquidity_state(self) -> LiquidityState | None:
        return self.liquidity_levels.get_state() if self.liquidity_levels is not None else None

    def get_trend_state(self) -> TrendState | None:
        return self.trend.get_state() if self.trend is not None else None

    async def publish_composite_update(
        self,
        *,
        updated_module: str | None = None,
        source_topic: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        self._refresh_composite_state(
            updated_module=updated_module,
            source_topic=source_topic,
            advance_version=False,
        )

        return await self._emit_event(
            self._build_event_name("updated"),
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
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
        if advance_version:
            self._state_version += 1

        market_structure_state = self.get_market_structure_state()
        support_resistance_state = self.get_support_resistance_state()
        fair_value_gap_state = self.get_fair_value_gap_state()
        liquidity_state = self.get_liquidity_state()
        trend_state = self.get_trend_state()

        last_price, last_update = self._resolve_latest_price_and_update(
            market_structure_state=market_structure_state,
            support_resistance_state=support_resistance_state,
            fair_value_gap_state=fair_value_gap_state,
            liquidity_state=liquidity_state,
            trend_state=trend_state,
        )

        self._state = PriceActionCompositeState(
            symbol=self.symbol,
            timeframe=self.timeframe,
            last_price=last_price,
            last_update=last_update,
            market_structure=market_structure_state,
            support_resistance=support_resistance_state,
            fair_value_gap=fair_value_gap_state,
            liquidity=liquidity_state,
            trend=trend_state,
            metadata={
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
    # Helpers
    # ------------------------------------------------------------------

    def _enabled_child_update_subscriptions(
        self,
    ) -> tuple[tuple[str, Any, str], ...]:
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
        return {
            "market_structure": self.config.enable_market_structure,
            "support_resistance": self.config.enable_support_resistance,
            "fair_value_gap": self.config.enable_fair_value_gap,
            "liquidity_levels": self.config.enable_liquidity_levels,
            "trend": self.config.enable_trend,
        }.get(module_name, False)

    def _iter_enabled_modules(self) -> tuple[BasePriceActionModule[Any], ...]:
        return tuple(self.get_child_analyzers().values())

    def _enabled_module_names(self) -> list[str]:
        return list(self.get_child_analyzers().keys())

    def _registered_module_names(self) -> list[str]:
        registered: list[str] = []

        for module_name, module in self.get_child_analyzers().items():
            if getattr(module, "_registered", False):
                registered.append(module_name)

        return registered


__all__ = [
    "PriceActionAnalyzerConfig",
    "PriceActionAnalyzer",
]