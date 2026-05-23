from __future__ import annotations
from core.logger import get_logger

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Mapping, Sequence
from uuid import uuid4

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import (
    MarketBias,
    StructureEventType,
    StructureLayer,
    SwingType,
)
from analytics.price_action.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    Candle,
    MarketStructureState,
    PriceActionKey,
    StructureEvent,
    StructureLayerState,
    SwingPoint,
    clamp_unit,
    make_price_action_key,
    price_action_key_to_dict,
)


@dataclass(slots=True)
class MarketStructureConfig(BasePriceActionConfig):
    pivot_left: int = 3
    pivot_right: int = 3

    internal_min_swing_distance_pct: float = 0.0008
    external_min_swing_distance_pct: float = 0.0020

    structure_break_threshold_pct: float = 0.0005
    require_close_break: bool = True

    max_candles: int = 4000
    max_internal_swings: int = 800
    max_external_swings: int = 400
    max_events: int = 1000

    alignment_window: int = 5
    external_strength_multiplier: float = 1.35
    min_external_strength: float = 0.30

    emit_events: bool = True
    event_namespace: str = "analytics.price_action.market_structure"
    publish_snapshots: bool = False

    subscribe_higher_timeframe_context: bool = True
    higher_timeframe_context_topic: str = (
        "analytics.price_action.higher_timeframe_context.updated"
    )

    def validate(self) -> None:
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

        if self.pivot_left < 1 or self.pivot_right < 1:
            raise ValueError("pivot_left and pivot_right must be >= 1")

        if self.internal_min_swing_distance_pct < 0:
            raise ValueError("internal_min_swing_distance_pct must be >= 0")

        if self.external_min_swing_distance_pct < 0:
            raise ValueError("external_min_swing_distance_pct must be >= 0")

        if self.structure_break_threshold_pct < 0:
            raise ValueError("structure_break_threshold_pct must be >= 0")

        if self.max_candles < (self.pivot_left + self.pivot_right + 10):
            raise ValueError("max_candles is too small for selected pivot settings")

        if self.max_internal_swings < 10:
            raise ValueError("max_internal_swings must be >= 10")

        if self.max_external_swings < 10:
            raise ValueError("max_external_swings must be >= 10")

        if self.max_events < 10:
            raise ValueError("max_events must be >= 10")

        if self.alignment_window < 1:
            raise ValueError("alignment_window must be >= 1")

        if self.external_strength_multiplier <= 0:
            raise ValueError("external_strength_multiplier must be > 0")

        if not 0.0 <= self.min_external_strength <= 1.0:
            raise ValueError("min_external_strength must be in [0.0, 1.0]")

        self.higher_timeframe_context_topic = self._normalize_topic(
            self.higher_timeframe_context_topic
        )

        if (
            self.subscribe_higher_timeframe_context
            and not self.higher_timeframe_context_topic
        ):
            raise ValueError("higher_timeframe_context_topic must not be empty")


class MarketStructureAnalyzer(BasePriceActionModule[MarketStructureState]):
    """
    Event-driven futures market structure analyzer.

    Correct input flow:
        exchange adapters
            -> market.candle
            -> CandlesCache
            -> market.candle.closed / market.candles.updated
            -> MarketStructureAnalyzer
            -> analytics.price_action.market_structure.*

    Scope:
        exchange + market_type + symbol + timeframe

    Responsibilities:
    - detect internal/external swing highs and lows;
    - classify HH / HL / LH / LL;
    - detect BOS / CHOCH / MSS-style structure breaks;
    - publish scoped analytics.price_action.market_structure.* events;
    - expose state and snapshots for strategy/dashboard/storage layers.
    """

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
        config: MarketStructureConfig | None = None,
        higher_timeframe: str | None = None,
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
        resolved_config = config or MarketStructureConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            market_type=market_type,
            exchange_symbol=exchange_symbol,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action.market_structure",
        )

        self.config: MarketStructureConfig = resolved_config
        self.higher_timeframe = str(higher_timeframe).strip() if higher_timeframe else None

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_swings: Deque[SwingPoint] = deque(
            maxlen=self.config.max_internal_swings
        )
        self._external_swings: Deque[SwingPoint] = deque(
            maxlen=self.config.max_external_swings
        )
        self._events: Deque[StructureEvent] = deque(maxlen=self.config.max_events)

        self._processed_pivots: set[tuple[int, SwingType]] = set()
        self._processed_structure_labels: set[
            tuple[str, StructureEventType, StructureLayer]
        ] = set()
        self._processed_breaks: set[tuple[StructureLayer, str, str, str]] = set()

        self._global_candle_index = 0
        self._last_processed_pivot_center_index = -1

        self._state = self._new_state()

        self.logger.info(
            "Initialized MarketStructureAnalyzer",
            extra={
                **self._log_scope_extra(),
                "higher_timeframe": self.higher_timeframe,
                "config": asdict(self.config),
            },
        )

    # -------------------------------------------------------------------------
    # Registration / EventBus handlers
    # -------------------------------------------------------------------------

    def register(self) -> None:
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
        super().register()

        if self.config.subscribe_higher_timeframe_context:
            self._subscribe(
                self.config.higher_timeframe_context_topic,
                self.on_higher_timeframe_context_event,
                name=f"{self.module_name}.on_higher_timeframe_context_event",
            )

    async def on_candle_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_candle_event", _analytics_args)
        except Exception:
            pass
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "MarketStructureAnalyzer received empty candle payload",
                extra={
                    **self._log_scope_extra(),
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        result = self.add_candles(candles)
        await self._publish_update_result(
            result,
            correlation_id=event.correlation_id,
        )

    async def on_candles_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_candles_event", _analytics_args)
        except Exception:
            pass
        candles = self._extract_candles_payload(event)
        if not candles:
            self.logger.warning(
                "MarketStructureAnalyzer received empty candles payload",
                extra={
                    **self._log_scope_extra(),
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        result = self.add_candles(candles)
        await self._publish_update_result(
            result,
            correlation_id=event.correlation_id,
        )

    async def on_higher_timeframe_context_event(self, event: Event) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "on_higher_timeframe_context_event", _analytics_args)
        except Exception:
            pass
        if not isinstance(event.payload, Mapping):
            self.logger.warning(
                "Invalid higher timeframe context payload",
                extra={
                    **self._log_scope_extra(),
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        if not self._higher_timeframe_context_matches_scope(event.payload):
            self.logger.debug(
                "Higher timeframe context skipped because scope does not match",
                extra={
                    **self._log_scope_extra(),
                    "topic": event.topic,
                    "event_id": event.event_id,
                },
            )
            return

        self.update_higher_timeframe_context(event.payload)
        self._refresh_state()

        await self._emit_event(
            self._build_event_name("mtf_alignment_updated"),
            {
                "state": self.snapshot(),
            },
            source=self.module_name,
            correlation_id=event.correlation_id,
        )

        if self.config.publish_snapshots:
            await self.publish_snapshot(correlation_id=event.correlation_id)

    async def _publish_update_result(
        self,
        result: Mapping[str, Any],
        *,
        correlation_id: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_publish_update_result", _analytics_args)
        except Exception:
            pass
        for swing in result.get("new_swings", []):
            if not isinstance(swing, Mapping):
                continue

            event_type = swing.get("swing_type")
            suffix = (
                StructureEventType.SWING_HIGH.value
                if event_type == SwingType.HIGH.value
                else StructureEventType.SWING_LOW.value
            )

            await self._emit_event(
                self._build_event_name(suffix),
                swing,
                source=self.module_name,
                correlation_id=correlation_id,
            )

        for event_payload in result.get("new_events", []):
            if not isinstance(event_payload, Mapping):
                continue

            event_type = event_payload.get("event_type")
            if not event_type:
                continue

            await self._emit_event(
                self._build_event_name(str(event_type)),
                event_payload,
                source=self.module_name,
                correlation_id=correlation_id,
            )

        await self._emit_event(
            self._build_event_name("updated"),
            {
                "state": result.get("state"),
                "new_swings_count": len(result.get("new_swings", [])),
                "new_events_count": len(result.get("new_events", [])),
            },
            source=self.module_name,
            correlation_id=correlation_id,
        )

        if self.config.publish_snapshots:
            await self.publish_snapshot(correlation_id=correlation_id)

    # -------------------------------------------------------------------------
    # Public sync domain API
    # -------------------------------------------------------------------------

    def reset(self) -> None:
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
        self._candles.clear()
        self._internal_swings.clear()
        self._external_swings.clear()
        self._events.clear()

        self._processed_pivots.clear()
        self._processed_structure_labels.clear()
        self._processed_breaks.clear()

        self._global_candle_index = 0
        self._last_processed_pivot_center_index = -1

        self._state = self._new_state()

        self.logger.info(
            "MarketStructureAnalyzer reset",
            extra=self._log_scope_extra(),
        )

    def get_state(self) -> MarketStructureState:
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
        return self._state

    def get_swings(self, layer: StructureLayer | None = None) -> list[SwingPoint]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_swings", _analytics_args)
        except Exception:
            pass
        if layer == StructureLayer.INTERNAL:
            return list(self._internal_swings)
        if layer == StructureLayer.EXTERNAL:
            return list(self._external_swings)
        return [*self._internal_swings, *self._external_swings]

    def get_events(self) -> list[StructureEvent]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_events", _analytics_args)
        except Exception:
            pass
        return list(self._events)

    def update(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        higher_timeframe_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "update", _analytics_args)
        except Exception:
            pass
        return self.add_candles(
            candles,
            higher_timeframe_context=higher_timeframe_context,
        )

    def add_candle(
        self,
        candle: Mapping[str, Any],
        *,
        higher_timeframe_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "add_candle", _analytics_args)
        except Exception:
            pass
        return self.add_candles(
            [candle],
            higher_timeframe_context=higher_timeframe_context,
        )

    def add_candles(
        self,
        candles: Sequence[Mapping[str, Any]],
        *,
        higher_timeframe_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "add_candles", _analytics_args)
        except Exception:
            pass
        if not candles:
            if higher_timeframe_context is not None:
                self.update_higher_timeframe_context(higher_timeframe_context)

            self._refresh_state()

            return {
                **self.scope_payload,
                "state": self.snapshot(),
                "new_swings": [],
                "new_events": [],
            }

        new_swings: list[SwingPoint] = []
        new_events: list[StructureEvent] = []

        for raw in candles:
            candle = self._parse_candle(raw, index=self._global_candle_index)
            self._global_candle_index += 1

            self._candles.append(candle)
            self._state.last_price = candle.close
            self._state.last_update = candle.timestamp

            swings_from_increment = self._process_incremental_pivot_detection()
            if swings_from_increment:
                new_swings.extend(swings_from_increment)

        if new_swings:
            label_events = self._classify_structure_labels(new_swings)
            if label_events:
                new_events.extend(label_events)

        break_events = self._detect_break_events()
        if break_events:
            new_events.extend(break_events)

        if higher_timeframe_context is not None:
            self.update_higher_timeframe_context(higher_timeframe_context)

        self._refresh_state()

        self.logger.debug(
            "Market structure incrementally updated",
            extra={
                **self._log_scope_extra(),
                "added_candles": len(candles),
                "new_swings": len(new_swings),
                "new_events": len(new_events),
                "internal_bias": self._state.internal.bias.value,
                "external_bias": self._state.external.bias.value,
                "alignment_score": self._state.mtf_alignment.alignment_score,
            },
        )

        return {
            **self.scope_payload,
            "state": self.snapshot(),
            "new_swings": [self._swing_to_dict(swing) for swing in new_swings],
            "new_events": [self._event_to_dict(event) for event in new_events],
        }

    def update_higher_timeframe_context(self, context: Mapping[str, Any]) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "update_higher_timeframe_context", _analytics_args)
        except Exception:
            pass
        mtf = self._state.mtf_alignment

        tf = (
            context.get("timeframe")
            or context.get("higher_timeframe")
            or self.higher_timeframe
        )
        bias_raw = (
            context.get("bias")
            or context.get("higher_timeframe_bias")
            or MarketBias.UNKNOWN
        )
        confidence = float(
            context.get(
                "confidence",
                context.get("higher_timeframe_confidence", 0.0),
            )
        )

        try:
            higher_bias = (
                bias_raw
                if isinstance(bias_raw, MarketBias)
                else MarketBias(str(bias_raw))
            )
        except ValueError:
            higher_bias = MarketBias.UNKNOWN

        mtf.higher_timeframe = str(tf) if tf is not None else None
        mtf.higher_timeframe_bias = higher_bias
        mtf.higher_timeframe_confidence = clamp_unit(confidence)
        mtf.last_updated = self._state.last_update
        mtf.metadata = {
            **self.scope_payload,
            **dict(context.get("metadata", {})),
        }

        self._refresh_alignment_state()

    def snapshot(self) -> dict[str, Any]:
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
        return self._snapshot_envelope(
            state=self._state,
            metadata={
                "total_candles": len(self._candles),
                "internal_swings": len(self._internal_swings),
                "external_swings": len(self._external_swings),
                "events": len(self._events),
                "higher_timeframe": self.higher_timeframe,
                "last_processed_pivot_center_index": (
                    self._last_processed_pivot_center_index
                ),
                "global_candle_index": self._global_candle_index,
                "config": self._serialize_config(),
            },
        )

    # -------------------------------------------------------------------------
    # Incremental pivot detection
    # -------------------------------------------------------------------------

    def _process_incremental_pivot_detection(self) -> list[SwingPoint]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_process_incremental_pivot_detection", _analytics_args)
        except Exception:
            pass
        candles = list(self._candles)
        needed = self.config.pivot_left + self.config.pivot_right + 1
        if len(candles) < needed:
            return []

        center_pos = len(candles) - self.config.pivot_right - 1
        center_candle = candles[center_pos]

        if center_candle.index <= self._last_processed_pivot_center_index:
            return []

        left_slice = candles[center_pos - self.config.pivot_left : center_pos]
        right_slice = candles[
            center_pos + 1 : center_pos + 1 + self.config.pivot_right
        ]

        is_swing_high = all(center_candle.high > x.high for x in left_slice) and all(
            center_candle.high >= x.high for x in right_slice
        )
        is_swing_low = all(center_candle.low < x.low for x in left_slice) and all(
            center_candle.low <= x.low for x in right_slice
        )

        self._last_processed_pivot_center_index = center_candle.index

        created_swings: list[SwingPoint] = []

        if is_swing_high:
            created_swings.extend(
                self._register_pivot(
                    center_candle=center_candle,
                    swing_type=SwingType.HIGH,
                )
            )

        if is_swing_low:
            created_swings.extend(
                self._register_pivot(
                    center_candle=center_candle,
                    swing_type=SwingType.LOW,
                )
            )

        return created_swings

    def _register_pivot(
        self,
        *,
        center_candle: Candle,
        swing_type: SwingType,
    ) -> list[SwingPoint]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_register_pivot", _analytics_args)
        except Exception:
            pass
        pivot_key = (center_candle.index, swing_type)
        if pivot_key in self._processed_pivots:
            return []

        self._processed_pivots.add(pivot_key)

        created: list[SwingPoint] = []

        internal_swing = self._maybe_create_swing(
            center_candle=center_candle,
            swing_type=swing_type,
            layer=StructureLayer.INTERNAL,
        )
        if internal_swing is not None:
            self._internal_swings.append(internal_swing)
            created.append(internal_swing)
            self._create_structure_event_for_swing(internal_swing)

        external_swing = self._maybe_create_swing(
            center_candle=center_candle,
            swing_type=swing_type,
            layer=StructureLayer.EXTERNAL,
        )
        if external_swing is not None:
            self._external_swings.append(external_swing)
            created.append(external_swing)
            self._create_structure_event_for_swing(external_swing)

        return created

    def _maybe_create_swing(
        self,
        *,
        center_candle: Candle,
        swing_type: SwingType,
        layer: StructureLayer,
    ) -> SwingPoint | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_maybe_create_swing", _analytics_args)
        except Exception:
            pass
        price = center_candle.high if swing_type == SwingType.HIGH else center_candle.low
        min_distance_pct = self._layer_min_distance_pct(layer)
        strength = self._calculate_swing_strength(center_candle, swing_type, layer)

        if layer == StructureLayer.EXTERNAL and strength < self.config.min_external_strength:
            return None

        existing_swings = self._swings_for_layer(layer)
        previous_same_type = self._last_swing_of_type(existing_swings, swing_type)

        if previous_same_type is not None:
            if previous_same_type.price <= 0:
                return None

            distance_pct = abs(price - previous_same_type.price) / previous_same_type.price
            if distance_pct < min_distance_pct:
                return None

        return SwingPoint(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            swing_id=uuid4().hex,
            timestamp=center_candle.timestamp,
            price=price,
            swing_type=swing_type,
            layer=layer,
            index=center_candle.index,
            candle_open=center_candle.open,
            candle_high=center_candle.high,
            candle_low=center_candle.low,
            candle_close=center_candle.close,
            strength=clamp_unit(strength),
            is_confirmed=True,
            metadata={
                "body_ratio": center_candle.body_ratio,
                "range_size": center_candle.range_size,
                "source_candle_key": list(center_candle.key),
            },
        )

    def _calculate_swing_strength(
        self,
        center_candle: Candle,
        swing_type: SwingType,
        layer: StructureLayer,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_swing_strength", _analytics_args)
        except Exception:
            pass
        candles = list(self._candles)
        center_pos = None

        for idx, candle in enumerate(candles):
            if candle.index == center_candle.index:
                center_pos = idx
                break

        if center_pos is None:
            return 0.0

        left_slice = candles[max(0, center_pos - self.config.pivot_left) : center_pos]
        right_slice = candles[
            center_pos + 1 : center_pos + 1 + self.config.pivot_right
        ]
        neighbors = [*left_slice, *right_slice]

        if not neighbors:
            return 0.0

        if swing_type == SwingType.HIGH:
            pivot_distance = mean_safe(
                [max(center_candle.high - x.high, 0.0) for x in neighbors]
            )
            normalizer = center_candle.high if center_candle.high > 0 else 1.0
        else:
            pivot_distance = mean_safe(
                [max(x.low - center_candle.low, 0.0) for x in neighbors]
            )
            normalizer = center_candle.low if center_candle.low > 0 else 1.0

        distance_score = pivot_distance / normalizer
        candle_quality = min(1.0, center_candle.body_ratio + 0.25)
        range_score = min(1.0, center_candle.range_size / max(center_candle.close, 1e-9))

        score = (distance_score * 8.0 + candle_quality + range_score) / 3.0

        if layer == StructureLayer.EXTERNAL:
            score *= self.config.external_strength_multiplier

        return clamp_unit(score)

    # -------------------------------------------------------------------------
    # Structure label classification
    # -------------------------------------------------------------------------

    def _classify_structure_labels(
        self,
        swings: Sequence[SwingPoint],
    ) -> list[StructureEvent]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_classify_structure_labels", _analytics_args)
        except Exception:
            pass
        created_events: list[StructureEvent] = []

        grouped: dict[StructureLayer, list[SwingPoint]] = {
            StructureLayer.INTERNAL: [],
            StructureLayer.EXTERNAL: [],
        }

        for swing in swings:
            grouped[swing.layer].append(swing)

        for layer, layer_swings in grouped.items():
            if not layer_swings:
                continue

            all_swings = self._sorted_swings_for_layer(layer)
            for swing in layer_swings:
                event = self._classify_single_swing(all_swings, swing)
                if event is not None:
                    created_events.append(event)

        return created_events

    def _classify_single_swing(
        self,
        all_swings: Sequence[SwingPoint],
        swing: SwingPoint,
    ) -> StructureEvent | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_classify_single_swing", _analytics_args)
        except Exception:
            pass
        same_type_swings = [
            x
            for x in all_swings
            if x.swing_type == swing.swing_type and x.index < swing.index
        ]

        if not same_type_swings:
            return None

        previous = same_type_swings[-1]

        if swing.swing_type == SwingType.HIGH:
            event_type = (
                StructureEventType.HH
                if swing.price > previous.price
                else StructureEventType.LH
            )
            direction = (
                MarketBias.BULLISH
                if event_type == StructureEventType.HH
                else MarketBias.BEARISH
            )
        else:
            event_type = (
                StructureEventType.HL
                if swing.price > previous.price
                else StructureEventType.LL
            )
            direction = (
                MarketBias.BULLISH
                if event_type == StructureEventType.HL
                else MarketBias.BEARISH
            )

        dedup_key = (swing.swing_id, event_type, swing.layer)
        if dedup_key in self._processed_structure_labels:
            return None

        self._processed_structure_labels.add(dedup_key)

        event = StructureEvent(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=direction,
            swing_id=swing.swing_id,
            reference_price=previous.price,
            reference_swing_id=previous.swing_id,
            confidence=self._label_confidence(swing, previous),
            metadata={
                "previous_price": previous.price,
                "previous_index": previous.index,
                "swing_strength": swing.strength,
                "previous_swing_id": previous.swing_id,
            },
        )

        self._events.append(event)
        return event

    def _label_confidence(self, current: SwingPoint, previous: SwingPoint) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_label_confidence", _analytics_args)
        except Exception:
            pass
        if previous.price <= 0:
            return clamp_unit(current.strength)

        move_pct = abs(current.price - previous.price) / previous.price
        raw = (current.strength + min(1.0, move_pct * 100.0)) / 2.0
        return clamp_unit(raw)

    # -------------------------------------------------------------------------
    # Break detection
    # -------------------------------------------------------------------------

    def _detect_break_events(self) -> list[StructureEvent]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_break_events", _analytics_args)
        except Exception:
            pass
        if not self._candles:
            return []

        current_candle = self._candles[-1]
        created_events: list[StructureEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            swings = self._sorted_swings_for_layer(layer)
            last_high = self._last_swing_of_type(swings, SwingType.HIGH)
            last_low = self._last_swing_of_type(swings, SwingType.LOW)

            if last_high is not None:
                high_break = self._maybe_break_event(
                    layer=layer,
                    current_candle=current_candle,
                    swing=last_high,
                    broken_side="high",
                )
                if high_break is not None:
                    created_events.append(high_break)

            if last_low is not None:
                low_break = self._maybe_break_event(
                    layer=layer,
                    current_candle=current_candle,
                    swing=last_low,
                    broken_side="low",
                )
                if low_break is not None:
                    created_events.append(low_break)

        return created_events

    def _maybe_break_event(
        self,
        *,
        layer: StructureLayer,
        current_candle: Candle,
        swing: SwingPoint,
        broken_side: str,
    ) -> StructureEvent | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_maybe_break_event", _analytics_args)
        except Exception:
            pass
        threshold = self.config.structure_break_threshold_pct
        reference_price = swing.price

        if broken_side == "high":
            required_price = reference_price * (1.0 + threshold)
            broken = (
                current_candle.close > required_price
                if self.config.require_close_break
                else current_candle.high > required_price
            )
            direction = MarketBias.BULLISH
        else:
            required_price = reference_price * (1.0 - threshold)
            broken = (
                current_candle.close < required_price
                if self.config.require_close_break
                else current_candle.low < required_price
            )
            direction = MarketBias.BEARISH

        if not broken:
            return None

        prev_bias = self._layer_state(layer).bias
        event_type = self._resolve_break_event_type(
            prev_bias=prev_bias,
            direction=direction,
        )

        dedup_key = (layer, swing.swing_id, str(current_candle.index), event_type.value)
        if dedup_key in self._processed_breaks:
            return None

        self._processed_breaks.add(dedup_key)

        event = StructureEvent(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=current_candle.timestamp,
            price=current_candle.close,
            layer=layer,
            direction=direction,
            swing_id=None,
            reference_price=swing.price,
            reference_swing_id=swing.swing_id,
            confidence=self._break_confidence(
                current_candle=current_candle,
                swing=swing,
                direction=direction,
            ),
            metadata={
                "broken_side": broken_side,
                "trigger_candle_index": current_candle.index,
                "trigger_close": current_candle.close,
                "trigger_high": current_candle.high,
                "trigger_low": current_candle.low,
                "threshold_pct": threshold,
                "source_candle_key": list(current_candle.key),
            },
        )

        self._events.append(event)
        return event

    def _resolve_break_event_type(
        self,
        *,
        prev_bias: MarketBias,
        direction: MarketBias,
    ) -> StructureEventType:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_break_event_type", _analytics_args)
        except Exception:
            pass
        if prev_bias in {MarketBias.UNKNOWN, MarketBias.RANGING}:
            return StructureEventType.BOS

        if prev_bias == direction:
            return StructureEventType.BOS

        return StructureEventType.CHOCH

    def _break_confidence(
        self,
        *,
        current_candle: Candle,
        swing: SwingPoint,
        direction: MarketBias,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_break_confidence", _analytics_args)
        except Exception:
            pass
        reference = swing.price if swing.price > 0 else 1.0

        if direction == MarketBias.BULLISH:
            move_pct = max(current_candle.close - swing.price, 0.0) / reference
        else:
            move_pct = max(swing.price - current_candle.close, 0.0) / reference

        raw = (
            swing.strength
            + current_candle.body_ratio
            + min(1.0, move_pct * 100.0)
        ) / 3.0
        return clamp_unit(raw)

    # -------------------------------------------------------------------------
    # State refresh
    # -------------------------------------------------------------------------

    def _refresh_state(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_state", _analytics_args)
        except Exception:
            pass
        self._refresh_layer_state(StructureLayer.INTERNAL)
        self._refresh_layer_state(StructureLayer.EXTERNAL)
        self._refresh_alignment_state()

    def _refresh_layer_state(self, layer: StructureLayer) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_layer_state", _analytics_args)
        except Exception:
            pass
        state = self._layer_state(layer)
        swings = self._sorted_swings_for_layer(layer)
        events = [x for x in self._events if x.layer == layer]

        highs = [x for x in swings if x.swing_type == SwingType.HIGH]
        lows = [x for x in swings if x.swing_type == SwingType.LOW]

        state.last_swing_high = highs[-1] if highs else None
        state.previous_swing_high = highs[-2] if len(highs) >= 2 else None
        state.last_swing_low = lows[-1] if lows else None
        state.previous_swing_low = lows[-2] if len(lows) >= 2 else None

        state.last_hh = self._last_event_of_type(events, StructureEventType.HH)
        state.last_hl = self._last_event_of_type(events, StructureEventType.HL)
        state.last_lh = self._last_event_of_type(events, StructureEventType.LH)
        state.last_ll = self._last_event_of_type(events, StructureEventType.LL)
        state.last_bos = self._last_event_of_type(events, StructureEventType.BOS)
        state.last_choch = self._last_event_of_type(events, StructureEventType.CHOCH)
        state.last_mss = self._last_event_of_type(events, StructureEventType.MSS)

        state.swing_count = len(swings)
        state.event_count = len(events)
        state.sequence = [x.event_type.value for x in events[-10:]]

        state.bias = self._infer_bias(layer)
        state.trend_strength = self._infer_trend_strength(layer)
        state.confidence = self._infer_layer_confidence(layer)
        state.in_breakout = bool(
            state.last_bos is not None
            and state.last_bos.timestamp == self._state.last_update
        )

    def _refresh_alignment_state(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_refresh_alignment_state", _analytics_args)
        except Exception:
            pass
        mtf = self._state.mtf_alignment
        internal_bias = self._state.internal.bias
        external_bias = self._state.external.bias

        mtf.internal_with_external_aligned = (
            internal_bias == external_bias
            and internal_bias not in {MarketBias.UNKNOWN, MarketBias.RANGING}
        )

        if mtf.higher_timeframe_bias not in {MarketBias.UNKNOWN, MarketBias.RANGING}:
            mtf.internal_bias_aligned = internal_bias == mtf.higher_timeframe_bias
            mtf.external_bias_aligned = external_bias == mtf.higher_timeframe_bias
        else:
            mtf.internal_bias_aligned = False
            mtf.external_bias_aligned = False

        score = 0.0
        if mtf.internal_with_external_aligned:
            score += 0.4
        if mtf.internal_bias_aligned:
            score += 0.3
        if mtf.external_bias_aligned:
            score += 0.3

        mtf.alignment_score = clamp_unit(score)

    def _infer_bias(self, layer: StructureLayer) -> MarketBias:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_infer_bias", _analytics_args)
        except Exception:
            pass
        state = self._layer_state(layer)

        latest_bullish_ts = None
        if state.last_hh and state.last_hl:
            latest_bullish_ts = max(state.last_hh.timestamp, state.last_hl.timestamp)

        latest_bearish_ts = None
        if state.last_lh and state.last_ll:
            latest_bearish_ts = max(state.last_lh.timestamp, state.last_ll.timestamp)

        if state.last_bos and state.last_choch:
            last_break = max(
                [state.last_bos, state.last_choch],
                key=lambda x: x.timestamp,
            )
        else:
            last_break = state.last_bos or state.last_choch

        if last_break is not None and last_break.direction is not None:
            return last_break.direction

        if latest_bullish_ts and latest_bearish_ts:
            if latest_bullish_ts > latest_bearish_ts:
                return MarketBias.BULLISH
            if latest_bearish_ts > latest_bullish_ts:
                return MarketBias.BEARISH

        if latest_bullish_ts:
            return MarketBias.BULLISH
        if latest_bearish_ts:
            return MarketBias.BEARISH

        return MarketBias.UNKNOWN

    def _infer_trend_strength(self, layer: StructureLayer) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_infer_trend_strength", _analytics_args)
        except Exception:
            pass
        swings = self._sorted_swings_for_layer(layer)
        if len(swings) < 2:
            return 0.0

        recent = swings[-self.config.alignment_window :]
        avg_strength = mean_safe([x.strength for x in recent])

        prices = [x.price for x in recent if x.price > 0]
        if len(prices) >= 2:
            price_dispersion = abs(prices[-1] - prices[0]) / prices[0]
        else:
            price_dispersion = 0.0

        raw = (avg_strength + min(1.0, price_dispersion * 100.0)) / 2.0
        return clamp_unit(raw)

    def _infer_layer_confidence(self, layer: StructureLayer) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_infer_layer_confidence", _analytics_args)
        except Exception:
            pass
        state = self._layer_state(layer)

        components: list[float] = [state.trend_strength]

        if state.last_bos:
            components.append(state.last_bos.confidence)
        if state.last_choch:
            components.append(state.last_choch.confidence)

        if state.last_swing_high:
            components.append(state.last_swing_high.strength)
        if state.last_swing_low:
            components.append(state.last_swing_low.strength)

        if not components:
            return 0.0

        return clamp_unit(sum(components) / len(components))

    # -------------------------------------------------------------------------
    # Event creation helpers
    # -------------------------------------------------------------------------

    def _create_structure_event_for_swing(self, swing: SwingPoint) -> StructureEvent:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_structure_event_for_swing", _analytics_args)
        except Exception:
            pass
        event_type = (
            StructureEventType.SWING_HIGH
            if swing.swing_type == SwingType.HIGH
            else StructureEventType.SWING_LOW
        )

        event = StructureEvent(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=swing.timestamp,
            price=swing.price,
            layer=swing.layer,
            direction=None,
            swing_id=swing.swing_id,
            reference_price=None,
            reference_swing_id=None,
            confidence=swing.strength,
            metadata={
                "index": swing.index,
                "strength": swing.strength,
                "swing_key": list(swing.key),
            },
        )

        self._events.append(event)
        return event

    # -------------------------------------------------------------------------
    # Scope helpers
    # -------------------------------------------------------------------------

    def _new_state(self) -> MarketStructureState:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_new_state", _analytics_args)
        except Exception:
            pass
        return MarketStructureState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
        )

    def _scope_kwargs(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_scope_kwargs", _analytics_args)
        except Exception:
            pass
        return {
            "exchange": self.exchange,
            "market_type": self.market_type,
            "symbol": self.symbol,
            "exchange_symbol": self.exchange_symbol,
            "timeframe": self.timeframe,
        }

    def _higher_timeframe_context_matches_scope(
        self,
        payload: Mapping[str, Any],
    ) -> bool:
        """
        Higher-timeframe context must belong to the same exchange/market/symbol.

        Its timeframe may differ from this module timeframe.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_higher_timeframe_context_matches_scope", _analytics_args)
        except Exception:
            pass
        if not self.config.require_event_scope:
            return True

        exchange = payload.get("exchange") or payload.get("venue")
        market_type = (
            payload.get("market_type")
            or payload.get("category")
            or payload.get("inst_type")
            or payload.get("instrument_type")
        )
        symbol = payload.get("symbol") or payload.get("s") or payload.get("instrument")
        timeframe = (
            payload.get("timeframe")
            or payload.get("higher_timeframe")
            or self.higher_timeframe
            or self.timeframe
        )

        if not exchange or not market_type or not symbol or not timeframe:
            return False

        try:
            key = make_price_action_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        except ValueError:
            return False

        return (
            key[0] == self.key[0]
            and key[1] == self.key[1]
            and key[2] == self.key[2]
        )

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    def _layer_min_distance_pct(self, layer: StructureLayer) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_layer_min_distance_pct", _analytics_args)
        except Exception:
            pass
        return (
            self.config.internal_min_swing_distance_pct
            if layer == StructureLayer.INTERNAL
            else self.config.external_min_swing_distance_pct
        )

    def _swings_for_layer(self, layer: StructureLayer) -> Deque[SwingPoint]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_swings_for_layer", _analytics_args)
        except Exception:
            pass
        return (
            self._internal_swings
            if layer == StructureLayer.INTERNAL
            else self._external_swings
        )

    def _sorted_swings_for_layer(self, layer: StructureLayer) -> list[SwingPoint]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_sorted_swings_for_layer", _analytics_args)
        except Exception:
            pass
        return sorted(self._swings_for_layer(layer), key=lambda x: x.index)

    def _layer_state(self, layer: StructureLayer) -> StructureLayerState:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_layer_state", _analytics_args)
        except Exception:
            pass
        return (
            self._state.internal
            if layer == StructureLayer.INTERNAL
            else self._state.external
        )

    def _last_swing_of_type(
        self,
        swings: Sequence[SwingPoint],
        swing_type: SwingType,
    ) -> SwingPoint | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_last_swing_of_type", _analytics_args)
        except Exception:
            pass
        filtered = [x for x in swings if x.swing_type == swing_type]
        return filtered[-1] if filtered else None

    def _last_event_of_type(
        self,
        events: Sequence[StructureEvent],
        event_type: StructureEventType,
    ) -> StructureEvent | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_last_event_of_type", _analytics_args)
        except Exception:
            pass
        filtered = [x for x in events if x.event_type == event_type]
        return filtered[-1] if filtered else None

    def _swing_to_dict(self, swing: SwingPoint) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_swing_to_dict", _analytics_args)
        except Exception:
            pass
        payload = self._safe_serialize(swing)
        if isinstance(payload, dict):
            payload["key"] = list(swing.key)
        return payload

    def _event_to_dict(self, event: StructureEvent) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_event_to_dict", _analytics_args)
        except Exception:
            pass
        payload = self._safe_serialize(event)
        if isinstance(payload, dict):
            payload["key"] = list(event.key)
        return payload


def mean_safe(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = [
    "MarketStructureConfig",
    "MarketStructureAnalyzer",
    "mean_safe",
]