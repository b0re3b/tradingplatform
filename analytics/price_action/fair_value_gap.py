from __future__ import annotations
from core.logger import get_logger

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Deque, Mapping, Sequence
from uuid import uuid4

from core.event_bus import Event, EventBus
from core.scheduler import Scheduler

from analytics.price_action.base import BasePriceActionConfig, BasePriceActionModule
from analytics.price_action.enums import (
    FVGDirection,
    FVGEventType,
    FVGStatus,
    StructureLayer,
)
from analytics.price_action.models import (
    DEFAULT_EXCHANGE,
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    Candle,
    FVGEvent,
    FairValueGap,
    FairValueGapState,
    LayerFVGState,
)


@dataclass(slots=True)
class FairValueGapConfig(BasePriceActionConfig):
    max_candles: int = 3000
    max_gaps_per_layer: int = 500
    max_events: int = 1000

    min_gap_pct_internal: float = 0.00035
    min_gap_pct_external: float = 0.00080
    merge_distance_pct_internal: float = 0.00025
    merge_distance_pct_external: float = 0.00050

    min_impulse_body_ratio: float = 0.45
    respected_reaction_threshold_pct: float = 0.0012
    invalidation_close_buffer_pct: float = 0.0002
    retest_window_bars: int = 20

    emit_events: bool = True
    event_namespace: str = "analytics.price_action.fair_value_gap"
    publish_snapshots: bool = False

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

        if self.max_candles < 100:
            raise ValueError("max_candles must be >= 100")
        if self.max_gaps_per_layer < 20:
            raise ValueError("max_gaps_per_layer must be >= 20")
        if self.max_events < 50:
            raise ValueError("max_events must be >= 50")

        if self.min_gap_pct_internal < 0:
            raise ValueError("min_gap_pct_internal must be >= 0")
        if self.min_gap_pct_external < 0:
            raise ValueError("min_gap_pct_external must be >= 0")
        if self.merge_distance_pct_internal < 0:
            raise ValueError("merge_distance_pct_internal must be >= 0")
        if self.merge_distance_pct_external < 0:
            raise ValueError("merge_distance_pct_external must be >= 0")
        if self.min_impulse_body_ratio < 0:
            raise ValueError("min_impulse_body_ratio must be >= 0")
        if self.respected_reaction_threshold_pct < 0:
            raise ValueError("respected_reaction_threshold_pct must be >= 0")
        if self.invalidation_close_buffer_pct < 0:
            raise ValueError("invalidation_close_buffer_pct must be >= 0")
        if self.retest_window_bars < 1:
            raise ValueError("retest_window_bars must be >= 1")


class FairValueGapAnalyzer(BasePriceActionModule[FairValueGapState]):
    """
    Event-driven futures Fair Value Gap analyzer.

    Correct input flow:
        exchange adapters
            -> market.candle
            -> CandlesCache
            -> market.candle.closed / market.candles.updated
            -> FairValueGapAnalyzer
            -> analytics.price_action.fair_value_gap.*

    Scope:
        exchange + market_type + symbol + timeframe

    Responsibilities:
    - detect bullish / bearish FVG using 3-candle logic;
    - maintain internal / external FVG layers;
    - track fills, partial fills, respected gaps, invalidations and retests;
    - publish scoped analytics.price_action.fair_value_gap.* events through EventBus;
    - expose state snapshots for strategy/dashboard/storage layers.
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
        config: FairValueGapConfig | None = None,
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
        resolved_config = config or FairValueGapConfig()

        super().__init__(
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            market_type=market_type,
            exchange_symbol=exchange_symbol,
            event_bus=event_bus,
            scheduler=scheduler,
            config=resolved_config,
            service_name="analytics.price_action.fair_value_gap",
        )

        self.config: FairValueGapConfig = resolved_config

        self._candles: Deque[Candle] = deque(maxlen=self.config.max_candles)
        self._internal_gaps: Deque[FairValueGap] = deque(
            maxlen=self.config.max_gaps_per_layer
        )
        self._external_gaps: Deque[FairValueGap] = deque(
            maxlen=self.config.max_gaps_per_layer
        )
        self._events: Deque[FVGEvent] = deque(maxlen=self.config.max_events)

        self._global_candle_index = 0
        self._last_processed_triplet_end_index = -1

        self._processed_fill_keys: set[tuple[str, int]] = set()
        self._processed_respect_keys: set[tuple[str, int]] = set()
        self._processed_invalidation_keys: set[tuple[str, int]] = set()
        self._processed_retest_keys: set[tuple[str, int]] = set()

        self._state = self._new_state()

        self.logger.info(
            "Initialized FairValueGapAnalyzer",
            extra={
                **self._log_scope_extra(),
                "config": asdict(self.config),
            },
        )

    # -------------------------------------------------------------------------
    # EventBus handlers
    # -------------------------------------------------------------------------

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
                "FairValueGapAnalyzer received empty candle payload",
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
                "FairValueGapAnalyzer received empty candles payload",
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
                "updated_gaps_count": len(result.get("updated_gaps", [])),
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
        self._internal_gaps.clear()
        self._external_gaps.clear()
        self._events.clear()

        self._global_candle_index = 0
        self._last_processed_triplet_end_index = -1

        self._processed_fill_keys.clear()
        self._processed_respect_keys.clear()
        self._processed_invalidation_keys.clear()
        self._processed_retest_keys.clear()

        self._state = self._new_state()

        self.logger.info(
            "FairValueGapAnalyzer reset",
            extra=self._log_scope_extra(),
        )

    def get_state(self) -> FairValueGapState:
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

    def get_gaps(self, layer: StructureLayer | None = None) -> list[FairValueGap]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_gaps", _analytics_args)
        except Exception:
            pass
        if layer == StructureLayer.INTERNAL:
            return list(self._internal_gaps)
        if layer == StructureLayer.EXTERNAL:
            return list(self._external_gaps)
        return [*self._internal_gaps, *self._external_gaps]

    def get_events(self) -> list[FVGEvent]:
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
        *,
        candles: Sequence[Mapping[str, Any]] | None = None,
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
        return self.add_candles(candles or [])

    def add_candle(self, candle: Mapping[str, Any]) -> dict[str, Any]:
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
        return self.add_candles([candle])

    def add_candles(self, candles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """
        Atomically ingest candle batches for Fair Value Gap analysis.

        If any candle in the batch is invalid, the analyzer rolls back all state
        mutations performed during this batch.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "add_candles", _analytics_args)
        except Exception:
            pass
        candles_batch = list(candles or [])

        if not candles_batch:
            self._refresh_state()
            return {
                **self.scope_payload,
                "state": self.snapshot(),
                "updated_gaps": [],
                "new_events": [],
            }

        rollback_candles = deepcopy(self._candles)
        rollback_internal_gaps = deepcopy(self._internal_gaps)
        rollback_external_gaps = deepcopy(self._external_gaps)
        rollback_events = deepcopy(self._events)

        rollback_global_candle_index = self._global_candle_index
        rollback_last_processed_triplet_end_index = (
            self._last_processed_triplet_end_index
        )

        rollback_processed_fill_keys = set(self._processed_fill_keys)
        rollback_processed_respect_keys = set(self._processed_respect_keys)
        rollback_processed_invalidation_keys = set(self._processed_invalidation_keys)
        rollback_processed_retest_keys = set(self._processed_retest_keys)

        rollback_state = deepcopy(self._state)

        updated_gaps: list[FairValueGap] = []
        new_events: list[FVGEvent] = []

        try:
            for raw in candles_batch:
                candle = self._parse_candle(raw, index=self._global_candle_index)
                self._global_candle_index += 1

                self._candles.append(candle)
                self._state.last_price = candle.close
                self._state.last_update = candle.timestamp

                created_gaps, creation_events = (
                    self._process_incremental_gap_detection()
                )
                updated_gaps.extend(created_gaps)
                new_events.extend(creation_events)

                lifecycle_events = self._process_gap_lifecycle(candle)
                new_events.extend(lifecycle_events)

            self._refresh_state()

        except Exception:
            self._candles = rollback_candles
            self._internal_gaps = rollback_internal_gaps
            self._external_gaps = rollback_external_gaps
            self._events = rollback_events

            self._global_candle_index = rollback_global_candle_index
            self._last_processed_triplet_end_index = (
                rollback_last_processed_triplet_end_index
            )

            self._processed_fill_keys = rollback_processed_fill_keys
            self._processed_respect_keys = rollback_processed_respect_keys
            self._processed_invalidation_keys = rollback_processed_invalidation_keys
            self._processed_retest_keys = rollback_processed_retest_keys

            self._state = rollback_state

            self.logger.exception(
                "Fair value gap batch ingestion failed and was rolled back",
                extra={
                    **self._log_scope_extra(),
                    "candles_count": len(candles_batch),
                    "rollback_global_candle_index": rollback_global_candle_index,
                    "rollback_last_processed_triplet_end_index": (
                        rollback_last_processed_triplet_end_index
                    ),
                },
            )
            raise

        self.logger.debug(
            "Fair value gap analyzer updated",
            extra={
                **self._log_scope_extra(),
                "updated_gaps": len(updated_gaps),
                "new_events": len(new_events),
                "last_price": self._state.last_price,
            },
        )

        return {
            **self.scope_payload,
            "state": self.snapshot(),
            "updated_gaps": [self._gap_to_dict(gap) for gap in updated_gaps],
            "new_events": [self._event_to_dict(event) for event in new_events],
        }

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
                "internal_gaps": len(self._internal_gaps),
                "external_gaps": len(self._external_gaps),
                "events": len(self._events),
                "global_candle_index": self._global_candle_index,
                "last_processed_triplet_end_index": (
                    self._last_processed_triplet_end_index
                ),
                "config": self._serialize_config(),
            },
        )

    # -------------------------------------------------------------------------
    # Incremental gap detection
    # -------------------------------------------------------------------------

    def _process_incremental_gap_detection(
        self,
    ) -> tuple[list[FairValueGap], list[FVGEvent]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_process_incremental_gap_detection", _analytics_args)
        except Exception:
            pass
        candles = list(self._candles)
        if len(candles) < 3:
            return [], []

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if c3.index <= self._last_processed_triplet_end_index:
            return [], []

        self._last_processed_triplet_end_index = c3.index

        updated_gaps: list[FairValueGap] = []
        new_events: list[FVGEvent] = []

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            created_gap, events = self._detect_gap_for_layer(
                c1=c1,
                c2=c2,
                c3=c3,
                layer=layer,
            )
            if created_gap is not None:
                updated_gaps.append(created_gap)
            new_events.extend(events)

        return updated_gaps, new_events

    def _detect_gap_for_layer(
        self,
        *,
        c1: Candle,
        c2: Candle,
        c3: Candle,
        layer: StructureLayer,
    ) -> tuple[FairValueGap | None, list[FVGEvent]]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_gap_for_layer", _analytics_args)
        except Exception:
            pass
        events: list[FVGEvent] = []

        self._assert_candle_scope(c1)
        self._assert_candle_scope(c2)
        self._assert_candle_scope(c3)

        if c2.body_ratio < self.config.min_impulse_body_ratio:
            return None, events

        if c1.high < c3.low:
            direction = FVGDirection.BULLISH
            lower_bound = c1.high
            upper_bound = c3.low
        elif c1.low > c3.high:
            direction = FVGDirection.BEARISH
            lower_bound = c3.high
            upper_bound = c1.low
        else:
            return None, events

        if lower_bound >= upper_bound:
            return None, events

        mid_price = (upper_bound + lower_bound) / 2.0
        size = upper_bound - lower_bound
        size_pct = size / max(mid_price, 1e-9)

        if size_pct < self._min_gap_pct(layer):
            return None, events

        strength = self._gap_strength(
            c1=c1,
            c2=c2,
            c3=c3,
            size_pct=size_pct,
        )

        merge_candidate = self._find_merge_candidate(
            layer=layer,
            direction=direction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        if merge_candidate is not None:
            self._merge_gap(
                merge_candidate,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                source_indices=[c1.index, c2.index, c3.index],
                timestamp=c3.timestamp,
                strength=strength,
            )

            events.append(
                self._create_event(
                    event_type=FVGEventType.FVG_MERGED,
                    gap=merge_candidate,
                    timestamp=c3.timestamp,
                    confidence=min(1.0, merge_candidate.strength),
                    reference_price=c3.close,
                    metadata={
                        "triplet_end_index": c3.index,
                        "merged_with_direction": direction.value,
                        "source_candle_keys": [
                            list(c1.key),
                            list(c2.key),
                            list(c3.key),
                        ],
                    },
                )
            )
            return merge_candidate, events

        gap = FairValueGap(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            gap_id=uuid4().hex,
            layer=layer,
            direction=direction,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            mid_price=mid_price,
            size=size,
            size_pct=size_pct,
            strength=strength,
            status=FVGStatus.ACTIVE,
            fill_percentage=0.0,
            touch_count=0,
            retest_count=0,
            created_at=c3.timestamp,
            updated_at=c3.timestamp,
            created_index=c3.index,
            source_candle_indices=[c1.index, c2.index, c3.index],
            metadata={
                "impulse_body_ratio": c2.body_ratio,
                "middle_candle_bullish": c2.is_bullish,
                "middle_candle_bearish": c2.is_bearish,
                "source_candle_keys": [
                    list(c1.key),
                    list(c2.key),
                    list(c3.key),
                ],
            },
        )

        self._gaps_for_layer(layer).append(gap)

        events.append(
            self._create_event(
                event_type=FVGEventType.FVG_CREATED,
                gap=gap,
                timestamp=c3.timestamp,
                confidence=strength,
                reference_price=c3.close,
                metadata={
                    "triplet_end_index": c3.index,
                    "size_pct": size_pct,
                    "source_candle_key": list(c3.key),
                },
            )
        )

        return gap, events

    # -------------------------------------------------------------------------
    # Gap lifecycle
    # -------------------------------------------------------------------------

    def _process_gap_lifecycle(self, candle: Candle) -> list[FVGEvent]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_process_gap_lifecycle", _analytics_args)
        except Exception:
            pass
        events: list[FVGEvent] = []
        self._assert_candle_scope(candle)

        for layer in (StructureLayer.INTERNAL, StructureLayer.EXTERNAL):
            for gap in list(self._gaps_for_layer(layer)):
                self._assert_gap_scope(gap)

                if gap.status in {FVGStatus.FILLED, FVGStatus.INVALIDATED}:
                    continue

                touched = self._is_gap_touched(gap, candle)
                if touched:
                    gap.touch_count += 1
                    gap.updated_at = candle.timestamp
                    if gap.first_touch_at is None:
                        gap.first_touch_at = candle.timestamp

                fill_pct = self._calculate_fill_percentage(gap, candle)
                previous_fill_pct = gap.fill_percentage

                if fill_pct > previous_fill_pct:
                    gap.fill_percentage = fill_pct
                    gap.last_fill_index = candle.index
                    gap.updated_at = candle.timestamp

                    fill_key = (gap.gap_id, candle.index)
                    if fill_key not in self._processed_fill_keys:
                        self._processed_fill_keys.add(fill_key)

                        if fill_pct >= 1.0:
                            event_type = FVGEventType.FVG_FILLED
                            gap.status = FVGStatus.FILLED
                            gap.filled_at = candle.timestamp
                        elif previous_fill_pct == 0.0:
                            event_type = FVGEventType.FVG_FILL_STARTED
                            gap.status = FVGStatus.PARTIALLY_FILLED
                        else:
                            event_type = FVGEventType.FVG_PARTIALLY_FILLED
                            gap.status = FVGStatus.PARTIALLY_FILLED

                        events.append(
                            self._create_event(
                                event_type=event_type,
                                gap=gap,
                                timestamp=candle.timestamp,
                                confidence=self._fill_confidence(
                                    gap,
                                    candle,
                                    fill_pct,
                                ),
                                reference_price=candle.close,
                                metadata={
                                    "candle_index": candle.index,
                                    "previous_fill_percentage": previous_fill_pct,
                                    "new_fill_percentage": fill_pct,
                                    "source_candle_key": list(candle.key),
                                },
                            )
                        )

                if gap.status in {FVGStatus.ACTIVE, FVGStatus.PARTIALLY_FILLED}:
                    respected = self._is_gap_respected(gap, candle)
                    if respected:
                        respect_key = (gap.gap_id, candle.index)
                        if respect_key not in self._processed_respect_keys:
                            self._processed_respect_keys.add(respect_key)

                            gap.status = FVGStatus.RESPECTED
                            gap.respected_at = candle.timestamp
                            gap.updated_at = candle.timestamp

                            events.append(
                                self._create_event(
                                    event_type=FVGEventType.FVG_RESPECTED,
                                    gap=gap,
                                    timestamp=candle.timestamp,
                                    confidence=self._respect_confidence(gap, candle),
                                    reference_price=candle.close,
                                    metadata={
                                        "candle_index": candle.index,
                                        "source_candle_key": list(candle.key),
                                    },
                                )
                            )

                if gap.status in {
                    FVGStatus.ACTIVE,
                    FVGStatus.PARTIALLY_FILLED,
                    FVGStatus.RESPECTED,
                }:
                    invalidated = self._is_gap_invalidated(gap, candle)
                    if invalidated:
                        invalidation_key = (gap.gap_id, candle.index)
                        if invalidation_key not in self._processed_invalidation_keys:
                            self._processed_invalidation_keys.add(invalidation_key)

                            gap.status = FVGStatus.INVALIDATED
                            gap.invalidated_at = candle.timestamp
                            gap.updated_at = candle.timestamp

                            events.append(
                                self._create_event(
                                    event_type=FVGEventType.FVG_INVALIDATED,
                                    gap=gap,
                                    timestamp=candle.timestamp,
                                    confidence=self._invalidation_confidence(
                                        gap,
                                        candle,
                                    ),
                                    reference_price=candle.close,
                                    metadata={
                                        "candle_index": candle.index,
                                        "source_candle_key": list(candle.key),
                                    },
                                )
                            )

                if gap.status in {
                    FVGStatus.ACTIVE,
                    FVGStatus.PARTIALLY_FILLED,
                    FVGStatus.RESPECTED,
                }:
                    retested = self._is_gap_retested(gap, candle)
                    if retested:
                        retest_key = (gap.gap_id, candle.index)
                        if retest_key not in self._processed_retest_keys:
                            self._processed_retest_keys.add(retest_key)

                            gap.retest_count += 1
                            gap.updated_at = candle.timestamp

                            events.append(
                                self._create_event(
                                    event_type=FVGEventType.FVG_RETESTED,
                                    gap=gap,
                                    timestamp=candle.timestamp,
                                    confidence=min(1.0, gap.strength),
                                    reference_price=candle.close,
                                    metadata={
                                        "candle_index": candle.index,
                                        "source_candle_key": list(candle.key),
                                    },
                                )
                            )

        return events

    # -------------------------------------------------------------------------
    # Detection rules
    # -------------------------------------------------------------------------

    def _is_gap_touched(self, gap: FairValueGap, candle: Candle) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_gap_touched", _analytics_args)
        except Exception:
            pass
        return candle.high >= gap.lower_bound and candle.low <= gap.upper_bound

    def _calculate_fill_percentage(self, gap: FairValueGap, candle: Candle) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_calculate_fill_percentage", _analytics_args)
        except Exception:
            pass
        if gap.size <= 0:
            return 0.0

        if gap.direction == FVGDirection.BULLISH:
            penetration = max(gap.upper_bound - candle.low, 0.0)
        else:
            penetration = max(candle.high - gap.lower_bound, 0.0)

        fill_pct = penetration / gap.size
        return max(0.0, min(1.0, fill_pct))

    def _is_gap_respected(self, gap: FairValueGap, candle: Candle) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_gap_respected", _analytics_args)
        except Exception:
            pass
        if not self._is_gap_touched(gap, candle):
            return False
        if gap.fill_percentage >= 1.0:
            return False

        threshold_pct = self.config.respected_reaction_threshold_pct

        if gap.direction == FVGDirection.BULLISH:
            reaction = max(candle.close - gap.upper_bound, 0.0) / max(
                gap.mid_price,
                1e-9,
            )
            return candle.close > gap.mid_price and reaction >= threshold_pct

        reaction = max(gap.lower_bound - candle.close, 0.0) / max(
            gap.mid_price,
            1e-9,
        )
        return candle.close < gap.mid_price and reaction >= threshold_pct

    def _is_gap_invalidated(self, gap: FairValueGap, candle: Candle) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_gap_invalidated", _analytics_args)
        except Exception:
            pass
        buffer_pct = self.config.invalidation_close_buffer_pct

        if gap.direction == FVGDirection.BULLISH:
            invalidation_price = gap.lower_bound * (1.0 - buffer_pct)
            return candle.close < invalidation_price

        invalidation_price = gap.upper_bound * (1.0 + buffer_pct)
        return candle.close > invalidation_price

    def _is_gap_retested(self, gap: FairValueGap, candle: Candle) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_gap_retested", _analytics_args)
        except Exception:
            pass
        if gap.created_index is None:
            return False

        bars_since_creation = candle.index - gap.created_index
        if bars_since_creation < 1 or bars_since_creation > self.config.retest_window_bars:
            return False

        return self._is_gap_touched(gap, candle)

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
        gaps = [gap for gap in list(self._gaps_for_layer(layer)) if gap.key == self.key]

        active_gaps = [gap for gap in gaps if gap.status == FVGStatus.ACTIVE]
        partially_filled_gaps = [
            gap for gap in gaps if gap.status == FVGStatus.PARTIALLY_FILLED
        ]
        filled_gaps = [gap for gap in gaps if gap.status == FVGStatus.FILLED]
        respected_gaps = [gap for gap in gaps if gap.status == FVGStatus.RESPECTED]
        invalidated_gaps = [
            gap for gap in gaps if gap.status == FVGStatus.INVALIDATED
        ]

        state.total_gaps = len(gaps)
        state.active_gaps = len(active_gaps)
        state.partially_filled_gaps = len(partially_filled_gaps)
        state.filled_gaps = len(filled_gaps)
        state.respected_gaps = len(respected_gaps)
        state.invalidated_gaps = len(invalidated_gaps)

        current_price = self._state.last_price

        state.nearest_bullish_gap = self._nearest_gap(
            gaps,
            current_price=current_price,
            direction=FVGDirection.BULLISH,
        )
        state.nearest_bearish_gap = self._nearest_gap(
            gaps,
            current_price=current_price,
            direction=FVGDirection.BEARISH,
        )
        state.strongest_bullish_gap = self._strongest_gap(
            gaps,
            direction=FVGDirection.BULLISH,
        )
        state.strongest_bearish_gap = self._strongest_gap(
            gaps,
            direction=FVGDirection.BEARISH,
        )

        state.recent_fill_activity = self._recent_fill_activity(gaps)

        layer_events = [
            event
            for event in self._events
            if event.layer == layer and event.key == self.key
        ]
        state.last_event = layer_events[-1] if layer_events else None
        state.metadata = {
            "open_gaps": len(
                [
                    gap
                    for gap in gaps
                    if gap.status
                    in {
                        FVGStatus.ACTIVE,
                        FVGStatus.PARTIALLY_FILLED,
                        FVGStatus.RESPECTED,
                    }
                ]
            ),
            "scope": self.scope_payload,
        }

    # -------------------------------------------------------------------------
    # Scope helpers
    # -------------------------------------------------------------------------

    def _new_state(self) -> FairValueGapState:
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
        return FairValueGapState(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
        )

    def _assert_candle_scope(self, candle: Candle) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_assert_candle_scope", _analytics_args)
        except Exception:
            pass
        if candle.key != self.key:
            raise ValueError(
                "candle scope does not match fair value gap module scope: "
                f"candle={candle.key}, module={self.key}"
            )

    def _assert_gap_scope(self, gap: FairValueGap) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_assert_gap_scope", _analytics_args)
        except Exception:
            pass
        if gap.key != self.key:
            raise ValueError(
                "gap scope does not match fair value gap module scope: "
                f"gap={gap.key}, module={self.key}"
            )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _gaps_for_layer(self, layer: StructureLayer) -> Deque[FairValueGap]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_gaps_for_layer", _analytics_args)
        except Exception:
            pass
        return (
            self._internal_gaps
            if layer == StructureLayer.INTERNAL
            else self._external_gaps
        )

    def _layer_state(self, layer: StructureLayer) -> LayerFVGState:
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
        return self._state.internal if layer == StructureLayer.INTERNAL else self._state.external

    def _min_gap_pct(self, layer: StructureLayer) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_min_gap_pct", _analytics_args)
        except Exception:
            pass
        return (
            self.config.min_gap_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.min_gap_pct_external
        )

    def _merge_distance_pct(self, layer: StructureLayer) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_merge_distance_pct", _analytics_args)
        except Exception:
            pass
        return (
            self.config.merge_distance_pct_internal
            if layer == StructureLayer.INTERNAL
            else self.config.merge_distance_pct_external
        )

    def _find_merge_candidate(
        self,
        *,
        layer: StructureLayer,
        direction: FVGDirection,
        lower_bound: float,
        upper_bound: float,
    ) -> FairValueGap | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_find_merge_candidate", _analytics_args)
        except Exception:
            pass
        candidates = [
            gap
            for gap in self._gaps_for_layer(layer)
            if gap.key == self.key
            and gap.direction == direction
            and gap.status != FVGStatus.INVALIDATED
        ]
        if not candidates:
            return None

        mid_price = (lower_bound + upper_bound) / 2.0
        threshold_pct = self._merge_distance_pct(layer)

        best: FairValueGap | None = None
        best_distance = float("inf")

        for gap in candidates:
            distance_pct = abs(mid_price - gap.mid_price) / max(gap.mid_price, 1e-9)
            overlaps = not (
                upper_bound < gap.lower_bound or lower_bound > gap.upper_bound
            )

            if overlaps or distance_pct <= threshold_pct:
                if distance_pct < best_distance:
                    best = gap
                    best_distance = distance_pct

        return best

    def _merge_gap(
        self,
        gap: FairValueGap,
        *,
        lower_bound: float,
        upper_bound: float,
        source_indices: list[int],
        timestamp: Any,
        strength: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_merge_gap", _analytics_args)
        except Exception:
            pass
        self._assert_gap_scope(gap)

        gap.lower_bound = min(gap.lower_bound, lower_bound)
        gap.upper_bound = max(gap.upper_bound, upper_bound)
        gap.mid_price = (gap.upper_bound + gap.lower_bound) / 2.0
        gap.size = gap.upper_bound - gap.lower_bound
        gap.size_pct = gap.size / max(gap.mid_price, 1e-9)
        gap.strength = max(gap.strength, strength)
        gap.updated_at = timestamp

        existing_indices = set(gap.source_candle_indices)
        for idx in source_indices:
            if idx not in existing_indices:
                gap.source_candle_indices.append(idx)

    def _gap_strength(
        self,
        *,
        c1: Candle,
        c2: Candle,
        c3: Candle,
        size_pct: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_gap_strength", _analytics_args)
        except Exception:
            pass
        impulse_score = min(1.0, c2.body_ratio)
        gap_score = min(1.0, size_pct * 200.0)
        range_score = min(1.0, c2.range_size / max(c2.close, 1e-9))
        reaction_score = min(
            1.0,
            abs(c3.close - c1.close) / max(c2.close, 1e-9) * 50.0,
        )

        raw = (impulse_score + gap_score + range_score + reaction_score) / 4.0
        return max(0.0, min(1.0, raw))

    def _nearest_gap(
        self,
        gaps: Sequence[FairValueGap],
        *,
        current_price: float | None,
        direction: FVGDirection,
    ) -> FairValueGap | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_nearest_gap", _analytics_args)
        except Exception:
            pass
        if current_price is None:
            return None

        candidates = [
            gap
            for gap in gaps
            if gap.key == self.key
            and gap.direction == direction
            and gap.status != FVGStatus.INVALIDATED
        ]
        if not candidates:
            return None

        return min(candidates, key=lambda gap: abs(gap.mid_price - current_price))

    def _strongest_gap(
        self,
        gaps: Sequence[FairValueGap],
        *,
        direction: FVGDirection,
    ) -> FairValueGap | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_strongest_gap", _analytics_args)
        except Exception:
            pass
        candidates = [
            gap
            for gap in gaps
            if gap.key == self.key
            and gap.direction == direction
            and gap.status != FVGStatus.INVALIDATED
        ]
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda gap: (
                gap.strength,
                gap.size_pct,
                gap.touch_count,
                gap.retest_count,
            ),
        )

    def _recent_fill_activity(self, gaps: Sequence[FairValueGap]) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_recent_fill_activity", _analytics_args)
        except Exception:
            pass
        open_gaps = [
            gap
            for gap in gaps
            if gap.key == self.key
            and gap.status
            in {
                FVGStatus.ACTIVE,
                FVGStatus.PARTIALLY_FILLED,
                FVGStatus.RESPECTED,
            }
        ]
        if not open_gaps:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                sum(gap.fill_percentage for gap in open_gaps) / len(open_gaps),
            ),
        )

    def _fill_confidence(
        self,
        gap: FairValueGap,
        candle: Candle,
        fill_pct: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_fill_confidence", _analytics_args)
        except Exception:
            pass
        raw = (gap.strength + candle.body_ratio + fill_pct) / 3.0
        return max(0.0, min(1.0, raw))

    def _respect_confidence(self, gap: FairValueGap, candle: Candle) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_respect_confidence", _analytics_args)
        except Exception:
            pass
        move_pct = abs(candle.close - gap.mid_price) / max(gap.mid_price, 1e-9)
        raw = (
            gap.strength
            + candle.body_ratio
            + min(1.0, move_pct * 100.0)
        ) / 3.0
        return max(0.0, min(1.0, raw))

    def _invalidation_confidence(self, gap: FairValueGap, candle: Candle) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_invalidation_confidence", _analytics_args)
        except Exception:
            pass
        if gap.direction == FVGDirection.BULLISH:
            move_pct = max(gap.lower_bound - candle.close, 0.0) / max(
                gap.mid_price,
                1e-9,
            )
        else:
            move_pct = max(candle.close - gap.upper_bound, 0.0) / max(
                gap.mid_price,
                1e-9,
            )

        raw = (
            gap.strength
            + candle.body_ratio
            + min(1.0, move_pct * 100.0)
        ) / 3.0
        return max(0.0, min(1.0, raw))

    def _create_event(
        self,
        *,
        event_type: FVGEventType,
        gap: FairValueGap,
        timestamp: Any,
        confidence: float,
        reference_price: float | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FVGEvent:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_create_event", _analytics_args)
        except Exception:
            pass
        self._assert_gap_scope(gap)

        event = FVGEvent(
            exchange=self.exchange,
            market_type=self.market_type,
            symbol=self.symbol,
            exchange_symbol=self.exchange_symbol,
            timeframe=self.timeframe,
            event_id=uuid4().hex,
            event_type=event_type,
            timestamp=timestamp,
            layer=gap.layer,
            gap_id=gap.gap_id,
            direction=gap.direction,
            upper_bound=gap.upper_bound,
            lower_bound=gap.lower_bound,
            fill_percentage=gap.fill_percentage,
            confidence=max(0.0, min(1.0, confidence)),
            reference_price=reference_price,
            metadata={
                **dict(metadata or {}),
                "gap_key": list(gap.key),
            },
        )
        self._events.append(event)
        return event

    def _gap_to_dict(self, gap: FairValueGap) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_gap_to_dict", _analytics_args)
        except Exception:
            pass
        serialized = self._safe_serialize(gap)
        if isinstance(serialized, dict):
            serialized["key"] = list(gap.key)
            return serialized
        return {"value": serialized, **self.scope_payload}

    def _event_to_dict(self, event: FVGEvent) -> dict[str, Any]:
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
        serialized = self._safe_serialize(event)
        if isinstance(serialized, dict):
            serialized["key"] = list(event.key)
            return serialized
        return {"value": serialized, **self.scope_payload}


__all__ = [
    "FairValueGapConfig",
    "FairValueGapAnalyzer",
]