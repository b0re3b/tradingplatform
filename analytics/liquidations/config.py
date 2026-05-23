from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from analytics.liquidations.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    LiquidationKey,
    liquidation_key_to_dict,
    make_liquidation_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)


# =============================================================================
# Canonical topics
# =============================================================================

# Raw/data-ingestion topic.
# Це topic, який слухає LiquidationStream як data-layer ingest.
DEFAULT_RAW_LIQUIDATION_TOPIC = "market.liquidation"

# Data/cache layer outputs.
DEFAULT_LIQUIDATION_RAW_TOPIC = "market.liquidation.raw"
DEFAULT_LIQUIDATION_NORMALIZED_TOPIC = "market.liquidation.normalized"
DEFAULT_LIQUIDATION_LARGE_TOPIC = "market.liquidation.large"
DEFAULT_LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"

# Stream diagnostics.
DEFAULT_STREAM_HEALTH_TOPIC = "system.analytics.liquidations.stream.health"
DEFAULT_STREAM_SNAPSHOT_TOPIC = "analytics.liquidations.stream.snapshot"

# Analytics detector topics.
DEFAULT_CASCADE_INPUT_TOPIC = DEFAULT_LIQUIDATION_NORMALIZED_TOPIC
DEFAULT_CASCADE_DETECTED_TOPIC = "analytics.liquidations.cascade_detected"
DEFAULT_EXHAUSTION_DETECTED_TOPIC = "analytics.liquidations.exhaustion_detected"
DEFAULT_CASCADE_SNAPSHOT_TOPIC = "analytics.liquidations.detector.snapshot"
DEFAULT_CASCADE_HEALTH_TOPIC = "system.analytics.liquidations.detector.health"

RAW_LIQUIDATION_MARKET_TOPICS = {
    DEFAULT_RAW_LIQUIDATION_TOPIC,
}

PRODUCTION_LIQUIDATION_DATA_TOPICS = {
    DEFAULT_LIQUIDATION_NORMALIZED_TOPIC,
    DEFAULT_LIQUIDATIONS_UPDATED_TOPIC,
    DEFAULT_LIQUIDATION_LARGE_TOPIC,
}


# =============================================================================
# Validation / normalization helpers
# =============================================================================

def _validate_topic(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in value:
        raise ValueError(f"{field_name} must not contain spaces")


def _validate_job_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in value:
        raise ValueError(f"{field_name} must not contain spaces")


def _validate_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_positive_float(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative_float(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_ratio(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_positive_decimal(value: Decimal, field_name: str) -> None:
    if Decimal(str(value)) <= Decimal("0"):
        raise ValueError(f"{field_name} must be > 0")


def _normalize_exchange_set(values: tuple[str, ...] | set[str] | list[str]) -> set[str]:
    return {
        normalize_exchange(value)
        for value in values
        if str(value).strip()
    }


def _normalize_symbol_set(values: tuple[str, ...] | set[str] | list[str]) -> set[str]:
    return {
        normalize_symbol(value)
        for value in values
        if str(value).strip()
    }


def _normalize_market_type_set(values: tuple[str, ...] | set[str] | list[str]) -> set[str]:
    return {
        normalize_market_type(value)
        for value in values
        if str(value).strip()
    }


def _normalize_timeframe_set(values: tuple[str, ...] | set[str] | list[str]) -> set[str]:
    return {
        normalize_timeframe(value)
        for value in values
        if str(value).strip()
    }


def _scope_key_to_string(key: LiquidationKey) -> str:
    scope = liquidation_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


# =============================================================================
# Stream config
# =============================================================================

@dataclass(slots=True)
class LiquidationStreamConfig:
    """
    Конфігурація ingestion/stream-рівня liquidation events.

    Роль LiquidationStream:
        exchange adapters
            -> EventBus.emit("market.liquidation")
            -> LiquidationStream
            -> LiquidationState / LiquidationMetrics
            -> market.liquidation.normalized
            -> market.liquidations.updated

    Це data/cache layer, тому йому дозволено слухати raw `market.liquidation`.
    Analytics detector-и raw topic слухати не повинні.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    enabled: bool = True

    # ------------------------------------------------------------------
    # Scoped defaults / filters
    # ------------------------------------------------------------------

    default_market_type: str = DEFAULT_MARKET_TYPE
    default_timeframe: str = DEFAULT_TIMEFRAME

    exchanges: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    market_types: tuple[str, ...] = ()
    timeframes: tuple[str, ...] = ()

    allowed_market_types: set[str] = field(
        default_factory=lambda: {
            "perpetual",
            "futures",
            "linear",
            "inverse",
            "swap",
            "usdm_futures",
            "coinm_futures",
        }
    )

    # ------------------------------------------------------------------
    # Input / output topics
    # ------------------------------------------------------------------

    input_topic_raw: str = DEFAULT_RAW_LIQUIDATION_TOPIC
    input_topics_raw: tuple[str, ...] = (DEFAULT_RAW_LIQUIDATION_TOPIC,)

    allow_raw_input_topics: bool = True

    max_buffer_size_per_symbol: int = 5000

    emit_raw_events: bool = False
    emit_large_events: bool = True
    emit_updated_events: bool = True

    large_liquidation_threshold_usd: Decimal = Decimal("100000")
    stale_event_threshold_seconds: int = 15

    publish_topic_raw: str = DEFAULT_LIQUIDATION_RAW_TOPIC
    publish_topic_normalized: str = DEFAULT_LIQUIDATION_NORMALIZED_TOPIC
    publish_topic_large: str = DEFAULT_LIQUIDATION_LARGE_TOPIC
    publish_topic_updated: str = DEFAULT_LIQUIDATIONS_UPDATED_TOPIC
    publish_topic_health: str = DEFAULT_STREAM_HEALTH_TOPIC
    publish_topic_snapshot: str = DEFAULT_STREAM_SNAPSHOT_TOPIC

    # ------------------------------------------------------------------
    # Scheduler jobs
    # ------------------------------------------------------------------

    healthcheck_interval_seconds: float = 10.0
    snapshot_interval_seconds: float = 30.0
    cleanup_interval_seconds: float = 60.0

    healthcheck_job_name: str = "analytics.liquidations.stream.healthcheck"
    snapshot_job_name: str = "analytics.liquidations.stream.snapshot"
    cleanup_job_name: str = "analytics.liquidations.stream.cleanup"

    scheduler_job_timeout_seconds: float = 5.0
    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_seconds: float = 1.0

    reconnect_on_health_degraded: bool = True
    reconnect_cooldown_seconds: float = 10.0

    consumer_idle_sleep_seconds: float = 0.01
    consumer_error_sleep_seconds: float = 1.0

    deduplication_enabled: bool = True
    recent_payload_fingerprints_size: int = 10_000
    recent_large_events_size: int = 500

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.default_market_type = normalize_market_type(self.default_market_type)
        self.default_timeframe = normalize_timeframe(self.default_timeframe)

        self.exchanges = tuple(sorted(_normalize_exchange_set(self.exchanges)))
        self.symbols = tuple(sorted(_normalize_symbol_set(self.symbols)))
        self.market_types = tuple(sorted(_normalize_market_type_set(self.market_types)))
        self.timeframes = tuple(sorted(_normalize_timeframe_set(self.timeframes)))

        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)

        self.input_topics_raw = tuple(
            topic.strip()
            for topic in self.input_topics_raw
            if str(topic).strip()
        ) or (self.input_topic_raw,)
        self.input_topic_raw = self.input_topics_raw[0]

        self.large_liquidation_threshold_usd = Decimal(str(self.large_liquidation_threshold_usd))
        self.metadata = dict(self.metadata or {})

        self.validate()

    @property
    def input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "input_topics", _analytics_args)
        except Exception:
            pass
        return self.input_topics_raw

    @property
    def output_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "output_topics", _analytics_args)
        except Exception:
            pass
        topics = [
            self.publish_topic_normalized,
            self.publish_topic_health,
            self.publish_topic_snapshot,
        ]

        if self.emit_raw_events:
            topics.append(self.publish_topic_raw)

        if self.emit_large_events:
            topics.append(self.publish_topic_large)

        if self.emit_updated_events:
            topics.append(self.publish_topic_updated)

        return tuple(topics)

    @property
    def scheduler_job_names(self) -> tuple[str, str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scheduler_job_names", _analytics_args)
        except Exception:
            pass
        return (
            self.healthcheck_job_name,
            self.snapshot_job_name,
            self.cleanup_job_name,
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
        _validate_positive_int(self.max_buffer_size_per_symbol, "stream.max_buffer_size_per_symbol")
        _validate_positive_decimal(self.large_liquidation_threshold_usd, "stream.large_liquidation_threshold_usd")
        _validate_positive_int(self.stale_event_threshold_seconds, "stream.stale_event_threshold_seconds")

        _validate_positive_float(self.healthcheck_interval_seconds, "stream.healthcheck_interval_seconds")
        _validate_positive_float(self.snapshot_interval_seconds, "stream.snapshot_interval_seconds")
        _validate_positive_float(self.cleanup_interval_seconds, "stream.cleanup_interval_seconds")

        _validate_positive_float(self.scheduler_job_timeout_seconds, "stream.scheduler_job_timeout_seconds")
        _validate_non_negative_int(self.scheduler_job_max_retries, "stream.scheduler_job_max_retries")
        _validate_non_negative_float(
            self.scheduler_job_retry_delay_seconds,
            "stream.scheduler_job_retry_delay_seconds",
        )

        _validate_non_negative_float(self.reconnect_cooldown_seconds, "stream.reconnect_cooldown_seconds")
        _validate_non_negative_float(self.consumer_idle_sleep_seconds, "stream.consumer_idle_sleep_seconds")
        _validate_non_negative_float(self.consumer_error_sleep_seconds, "stream.consumer_error_sleep_seconds")

        _validate_positive_int(
            self.recent_payload_fingerprints_size,
            "stream.recent_payload_fingerprints_size",
        )
        _validate_positive_int(self.recent_large_events_size, "stream.recent_large_events_size")

        for topic in self.input_topics_raw:
            _validate_topic(topic, "stream.input_topics_raw item")

        for topic in self.output_topics:
            _validate_topic(topic, "stream.output_topics item")

        for job_name in self.scheduler_job_names:
            _validate_job_name(job_name, "stream.scheduler_job_names item")

        if not self.allow_raw_input_topics:
            raw_used = set(self.input_topics_raw).intersection(RAW_LIQUIDATION_MARKET_TOPICS)
            if raw_used:
                raise ValueError(
                    "LiquidationStream is configured with raw input topics, "
                    f"but allow_raw_input_topics=False: {sorted(raw_used)}"
                )

    def make_key(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> LiquidationKey:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_key", _analytics_args)
        except Exception:
            pass
        return make_liquidation_key(
            exchange=exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_key(self, key: LiquidationKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_key", _analytics_args)
        except Exception:
            pass
        scope = liquidation_key_to_dict(key)

        if self.exchanges and scope["exchange"] not in self.exchanges:
            return False

        if self.symbols and scope["symbol"] not in self.symbols:
            return False

        if self.market_types and scope["market_type"] not in self.market_types:
            return False

        if self.allowed_market_types and scope["market_type"] not in self.allowed_market_types:
            return False

        if self.timeframes and scope["timeframe"] not in self.timeframes:
            return False

        return True

    def should_process_scope(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_process_scope", _analytics_args)
        except Exception:
            pass
        return self.should_process_key(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def is_raw_input_topic(self, topic: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_raw_input_topic", _analytics_args)
        except Exception:
            pass
        return topic in RAW_LIQUIDATION_MARKET_TOPICS

    def assert_input_topic_allowed(self, topic: str) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "assert_input_topic_allowed", _analytics_args)
        except Exception:
            pass
        _validate_topic(topic, "stream input topic")

        if self.is_raw_input_topic(topic) and not self.allow_raw_input_topics:
            raise ValueError(
                f"Raw liquidation topic {topic!r} is disabled for LiquidationStream"
            )

    def scoped_mapping_key(self, key: LiquidationKey) -> str:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scoped_mapping_key", _analytics_args)
        except Exception:
            pass
        return _scope_key_to_string(key)

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "enabled": self.enabled,
            "scope": "exchange:market_type:symbol:timeframe",
            "default_market_type": self.default_market_type,
            "default_timeframe": self.default_timeframe,
            "exchanges": list(self.exchanges),
            "symbols": list(self.symbols),
            "market_types": list(self.market_types),
            "timeframes": list(self.timeframes),
            "allowed_market_types": sorted(self.allowed_market_types),
            "input_topics_raw": list(self.input_topics_raw),
            "allow_raw_input_topics": self.allow_raw_input_topics,
            "output_topics": list(self.output_topics),
            "max_buffer_size_per_symbol": self.max_buffer_size_per_symbol,
            "emit_raw_events": self.emit_raw_events,
            "emit_large_events": self.emit_large_events,
            "emit_updated_events": self.emit_updated_events,
            "large_liquidation_threshold_usd": str(self.large_liquidation_threshold_usd),
            "stale_event_threshold_seconds": self.stale_event_threshold_seconds,
            "healthcheck_interval_seconds": self.healthcheck_interval_seconds,
            "snapshot_interval_seconds": self.snapshot_interval_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "scheduler_job_names": list(self.scheduler_job_names),
            "scheduler_job_timeout_seconds": self.scheduler_job_timeout_seconds,
            "scheduler_job_max_retries": self.scheduler_job_max_retries,
            "scheduler_job_retry_delay_seconds": self.scheduler_job_retry_delay_seconds,
            "reconnect_on_health_degraded": self.reconnect_on_health_degraded,
            "reconnect_cooldown_seconds": self.reconnect_cooldown_seconds,
            "consumer_idle_sleep_seconds": self.consumer_idle_sleep_seconds,
            "consumer_error_sleep_seconds": self.consumer_error_sleep_seconds,
            "deduplication_enabled": self.deduplication_enabled,
            "recent_payload_fingerprints_size": self.recent_payload_fingerprints_size,
            "recent_large_events_size": self.recent_large_events_size,
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Cascade detector config
# =============================================================================

@dataclass(slots=True)
class CascadeDetectorConfig:
    """
    Конфігурація detector-а liquidation cascades.

    Detector є analytics layer:
        LiquidationStream
            -> market.liquidation.normalized
            -> CascadeDetector
            -> analytics.liquidations.*

    CascadeDetector не має слухати raw `market.liquidation`.
    Він має працювати тільки з normalized/data-layer topic.
    """

    enabled: bool = True

    input_topic: str = DEFAULT_CASCADE_INPUT_TOPIC
    allow_raw_input_topics: bool = False

    window_seconds: int = 10
    min_events: int = 5
    min_total_notional_usd: Decimal = Decimal("250000")
    min_side_imbalance_ratio: float = 0.75

    cooldown_seconds: int = 15

    acceleration_enabled: bool = True
    min_acceleration_ratio: float = 1.20

    price_compaction_enabled: bool = True
    max_price_range_pct: float = 0.75

    continuation_score_weight: float = 0.40
    imbalance_score_weight: float = 0.25
    notional_score_weight: float = 0.20
    acceleration_score_weight: float = 0.15

    low_severity_threshold: float = 0.30
    medium_severity_threshold: float = 0.55
    high_severity_threshold: float = 0.75
    extreme_severity_threshold: float = 0.90

    publish_topic_detected: str = DEFAULT_CASCADE_DETECTED_TOPIC
    publish_topic_exhaustion: str = DEFAULT_EXHAUSTION_DETECTED_TOPIC
    publish_topic_snapshot: str = DEFAULT_CASCADE_SNAPSHOT_TOPIC
    publish_topic_health: str = DEFAULT_CASCADE_HEALTH_TOPIC

    snapshot_interval_seconds: float = 30.0
    healthcheck_interval_seconds: float = 15.0
    cleanup_interval_seconds: float = 60.0

    snapshot_job_name: str = "analytics.liquidations.cascade_detector.snapshot"
    healthcheck_job_name: str = "analytics.liquidations.cascade_detector.healthcheck"
    cleanup_job_name: str = "analytics.liquidations.cascade_detector.cleanup"

    scheduler_job_timeout_seconds: float = 5.0
    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_seconds: float = 1.0

    recent_signals_limit: int = 200

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.min_total_notional_usd = Decimal(str(self.min_total_notional_usd))
        self.metadata = dict(self.metadata or {})
        self.validate()

    @property
    def output_topics(self) -> tuple[str, str, str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "output_topics", _analytics_args)
        except Exception:
            pass
        return (
            self.publish_topic_detected,
            self.publish_topic_exhaustion,
            self.publish_topic_snapshot,
            self.publish_topic_health,
        )

    @property
    def scheduler_job_names(self) -> tuple[str, str, str]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scheduler_job_names", _analytics_args)
        except Exception:
            pass
        return (
            self.snapshot_job_name,
            self.healthcheck_job_name,
            self.cleanup_job_name,
        )

    @property
    def score_weights_sum(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "score_weights_sum", _analytics_args)
        except Exception:
            pass
        return (
            self.continuation_score_weight
            + self.imbalance_score_weight
            + self.notional_score_weight
            + self.acceleration_score_weight
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
        _validate_topic(self.input_topic, "cascade.input_topic")

        if self.input_topic in RAW_LIQUIDATION_MARKET_TOPICS and not self.allow_raw_input_topics:
            raise ValueError(
                "CascadeDetector must not subscribe to raw market.liquidation. "
                "Use market.liquidation.normalized instead."
            )

        _validate_positive_int(self.window_seconds, "cascade.window_seconds")
        _validate_positive_int(self.min_events, "cascade.min_events")
        _validate_positive_decimal(self.min_total_notional_usd, "cascade.min_total_notional_usd")
        _validate_ratio(self.min_side_imbalance_ratio, "cascade.min_side_imbalance_ratio")

        _validate_non_negative_int(self.cooldown_seconds, "cascade.cooldown_seconds")

        _validate_non_negative_float(self.min_acceleration_ratio, "cascade.min_acceleration_ratio")
        _validate_non_negative_float(self.max_price_range_pct, "cascade.max_price_range_pct")

        for field_name, value in (
            ("cascade.continuation_score_weight", self.continuation_score_weight),
            ("cascade.imbalance_score_weight", self.imbalance_score_weight),
            ("cascade.notional_score_weight", self.notional_score_weight),
            ("cascade.acceleration_score_weight", self.acceleration_score_weight),
        ):
            _validate_non_negative_float(value, field_name)

        if self.score_weights_sum <= 0:
            raise ValueError("cascade score weights sum must be > 0")

        thresholds = (
            self.low_severity_threshold,
            self.medium_severity_threshold,
            self.high_severity_threshold,
            self.extreme_severity_threshold,
        )

        for index, value in enumerate(thresholds):
            _validate_ratio(value, f"cascade severity threshold #{index}")

        if list(thresholds) != sorted(thresholds):
            raise ValueError("cascade severity thresholds must be sorted ascending")

        _validate_positive_float(self.snapshot_interval_seconds, "cascade.snapshot_interval_seconds")
        _validate_positive_float(self.healthcheck_interval_seconds, "cascade.healthcheck_interval_seconds")
        _validate_positive_float(self.cleanup_interval_seconds, "cascade.cleanup_interval_seconds")

        _validate_positive_float(self.scheduler_job_timeout_seconds, "cascade.scheduler_job_timeout_seconds")
        _validate_non_negative_int(self.scheduler_job_max_retries, "cascade.scheduler_job_max_retries")
        _validate_non_negative_float(
            self.scheduler_job_retry_delay_seconds,
            "cascade.scheduler_job_retry_delay_seconds",
        )

        _validate_positive_int(self.recent_signals_limit, "cascade.recent_signals_limit")

        for topic in self.output_topics:
            _validate_topic(topic, "cascade.output_topics item")

        for job_name in self.scheduler_job_names:
            _validate_job_name(job_name, "cascade.scheduler_job_names item")

    def normalized_score_weights(self) -> dict[str, float]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "normalized_score_weights", _analytics_args)
        except Exception:
            pass
        total = self.score_weights_sum
        if total <= 0:
            return {
                "continuation": 0.0,
                "imbalance": 0.0,
                "notional": 0.0,
                "acceleration": 0.0,
            }

        return {
            "continuation": self.continuation_score_weight / total,
            "imbalance": self.imbalance_score_weight / total,
            "notional": self.notional_score_weight / total,
            "acceleration": self.acceleration_score_weight / total,
        }

    def is_raw_input_topic(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_raw_input_topic", _analytics_args)
        except Exception:
            pass
        return self.input_topic in RAW_LIQUIDATION_MARKET_TOPICS

    def assert_input_topic_allowed(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "assert_input_topic_allowed", _analytics_args)
        except Exception:
            pass
        if self.is_raw_input_topic() and not self.allow_raw_input_topics:
            raise ValueError(
                "CascadeDetector raw input topic is not allowed. "
                "Use market.liquidation.normalized."
            )

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "enabled": self.enabled,
            "input_topic": self.input_topic,
            "allow_raw_input_topics": self.allow_raw_input_topics,
            "window_seconds": self.window_seconds,
            "min_events": self.min_events,
            "min_total_notional_usd": str(self.min_total_notional_usd),
            "min_side_imbalance_ratio": self.min_side_imbalance_ratio,
            "cooldown_seconds": self.cooldown_seconds,
            "acceleration_enabled": self.acceleration_enabled,
            "min_acceleration_ratio": self.min_acceleration_ratio,
            "price_compaction_enabled": self.price_compaction_enabled,
            "max_price_range_pct": self.max_price_range_pct,
            "score_weights": self.normalized_score_weights(),
            "severity_thresholds": {
                "low": self.low_severity_threshold,
                "medium": self.medium_severity_threshold,
                "high": self.high_severity_threshold,
                "extreme": self.extreme_severity_threshold,
            },
            "output_topics": list(self.output_topics),
            "snapshot_interval_seconds": self.snapshot_interval_seconds,
            "healthcheck_interval_seconds": self.healthcheck_interval_seconds,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "scheduler_job_names": list(self.scheduler_job_names),
            "scheduler_job_timeout_seconds": self.scheduler_job_timeout_seconds,
            "scheduler_job_max_retries": self.scheduler_job_max_retries,
            "scheduler_job_retry_delay_seconds": self.scheduler_job_retry_delay_seconds,
            "recent_signals_limit": self.recent_signals_limit,
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Metrics config
# =============================================================================

@dataclass(slots=True)
class LiquidationMetricsConfig:
    """
    Конфігурація runtime metrics liquidation-модуля.

    Metrics-клас має бути pure accumulator.
    Публікація snapshots має виконуватись runtime-класами через EventBus/Scheduler.
    """

    enabled: bool = True

    keep_symbol_level_counters: bool = True
    keep_exchange_level_counters: bool = True
    keep_market_type_level_counters: bool = True
    keep_scope_level_counters: bool = True

    latency_buckets_ms: tuple[int, ...] = (
        1,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2500,
        5000,
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.latency_buckets_ms = tuple(int(bucket) for bucket in self.latency_buckets_ms)
        self.metadata = dict(self.metadata or {})
        self.validate()

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
        if not self.latency_buckets_ms:
            raise ValueError("metrics.latency_buckets_ms must not be empty")

        if any(bucket <= 0 for bucket in self.latency_buckets_ms):
            raise ValueError("metrics.latency_buckets_ms values must be > 0")

        if tuple(sorted(self.latency_buckets_ms)) != self.latency_buckets_ms:
            raise ValueError("metrics.latency_buckets_ms must be sorted ascending")

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "enabled": self.enabled,
            "keep_symbol_level_counters": self.keep_symbol_level_counters,
            "keep_exchange_level_counters": self.keep_exchange_level_counters,
            "keep_market_type_level_counters": self.keep_market_type_level_counters,
            "keep_scope_level_counters": self.keep_scope_level_counters,
            "latency_buckets_ms": list(self.latency_buckets_ms),
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Root config
# =============================================================================

@dataclass(slots=True)
class LiquidationsConfig:
    """
    Кореневий конфіг liquidation-модуля.

    Цей config передається через dependency injection у bootstrap/container.
    Сам config не створює EventBus, Scheduler, Stream або Detector.
    """

    stream: LiquidationStreamConfig = field(default_factory=LiquidationStreamConfig)
    cascade: CascadeDetectorConfig = field(default_factory=CascadeDetectorConfig)
    metrics: LiquidationMetricsConfig = field(default_factory=LiquidationMetricsConfig)

    def __post_init__(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__post_init__", _analytics_args)
        except Exception:
            pass
        self.validate()

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
        self.stream.validate()
        self.cascade.validate()
        self.metrics.validate()

    @property
    def production_input_topics(self) -> tuple[str, ...]:
        """
        Topics, які analytics/detectors мають слухати в production.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "production_input_topics", _analytics_args)
        except Exception:
            pass
        return (self.cascade.input_topic,)

    @property
    def raw_ingestion_topics(self) -> tuple[str, ...]:
        """
        Raw topics, які дозволені тільки ingestion/data-layer stream-у.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "raw_ingestion_topics", _analytics_args)
        except Exception:
            pass
        return self.stream.input_topics

    @property
    def output_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "output_topics", _analytics_args)
        except Exception:
            pass
        return (
            *self.stream.output_topics,
            *self.cascade.output_topics,
        )

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scheduler_job_names", _analytics_args)
        except Exception:
            pass
        return (
            *self.stream.scheduler_job_names,
            *self.cascade.scheduler_job_names,
        )

    def to_dict(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_dict", _analytics_args)
        except Exception:
            pass
        return {
            "scope": "exchange:market_type:symbol:timeframe",
            "stream": self.stream.to_dict(),
            "cascade": self.cascade.to_dict(),
            "metrics": self.metrics.to_dict(),
            "production_input_topics": list(self.production_input_topics),
            "raw_ingestion_topics": list(self.raw_ingestion_topics),
            "output_topics": list(self.output_topics),
            "scheduler_job_names": list(self.scheduler_job_names),
        }


__all__ = [
    # topics
    "DEFAULT_RAW_LIQUIDATION_TOPIC",
    "DEFAULT_LIQUIDATION_RAW_TOPIC",
    "DEFAULT_LIQUIDATION_NORMALIZED_TOPIC",
    "DEFAULT_LIQUIDATION_LARGE_TOPIC",
    "DEFAULT_LIQUIDATIONS_UPDATED_TOPIC",
    "DEFAULT_STREAM_HEALTH_TOPIC",
    "DEFAULT_STREAM_SNAPSHOT_TOPIC",
    "DEFAULT_CASCADE_INPUT_TOPIC",
    "DEFAULT_CASCADE_DETECTED_TOPIC",
    "DEFAULT_EXHAUSTION_DETECTED_TOPIC",
    "DEFAULT_CASCADE_SNAPSHOT_TOPIC",
    "DEFAULT_CASCADE_HEALTH_TOPIC",
    "RAW_LIQUIDATION_MARKET_TOPICS",
    "PRODUCTION_LIQUIDATION_DATA_TOPICS",

    # configs
    "LiquidationStreamConfig",
    "CascadeDetectorConfig",
    "LiquidationMetricsConfig",
    "LiquidationsConfig",
]