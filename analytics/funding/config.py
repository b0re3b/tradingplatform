from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field
from typing import Any

from analytics.funding.enums import FundingTimeframe
from analytics.funding.models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    FundingKey,
    funding_key_to_dict,
    make_funding_key,
    normalize_exchange,
    normalize_market_type,
    normalize_symbol,
    normalize_timeframe,
)


# =============================================================================
# Canonical production topics
# =============================================================================

# Production input topics.
# Важливо: це data/cache/analytics-layer events, а не raw exchange events.
DEFAULT_FUNDING_UPDATED_TOPIC = "market.funding.updated"
DEFAULT_OPEN_INTEREST_UPDATED_TOPIC = "market.open_interest.updated"
DEFAULT_CANDLE_CLOSED_TOPIC = "market.candle.closed"
DEFAULT_TRADES_UPDATED_TOPIC = "market.trades.updated"
DEFAULT_CVD_UPDATED_TOPIC = "analytics.orderflow.updated"
DEFAULT_LIQUIDATIONS_UPDATED_TOPIC = "market.liquidations.updated"

# Legacy/raw topics.
# Не використовувати в production, якщо allow_legacy_raw_topics=False.
DEFAULT_RAW_FUNDING_TOPIC = "market.funding"
DEFAULT_RAW_OPEN_INTEREST_TOPIC = "market.open_interest"
DEFAULT_RAW_CANDLE_TOPIC = "market.candle"
DEFAULT_RAW_TRADE_TOPIC = "market.trade"
DEFAULT_RAW_LIQUIDATION_TOPIC = "market.liquidation"

# Analytics output topics.
DEFAULT_FUNDING_SNAPSHOT_TOPIC = "analytics.funding.snapshot"
DEFAULT_FUNDING_REGIME_TOPIC = "analytics.funding.regime"
DEFAULT_FUNDING_EXTREME_TOPIC = "analytics.funding.extreme"
DEFAULT_FUNDING_FLIP_TOPIC = "analytics.funding.flip"
DEFAULT_FUNDING_DIVERGENCE_TOPIC = "analytics.funding.divergence"
DEFAULT_FUNDING_PRESSURE_TOPIC = "analytics.funding.pressure"
DEFAULT_FUNDING_SIGNAL_TOPIC = "analytics.funding.signal"
DEFAULT_FUNDING_ANALYTICS_TOPIC = "analytics.funding.updated"

# Lifecycle / diagnostics topics.
DEFAULT_FUNDING_ANALYZER_STARTED_TOPIC = "analytics.funding.analyzer.started"
DEFAULT_FUNDING_ANALYZER_STOPPED_TOPIC = "analytics.funding.analyzer.stopped"
DEFAULT_FUNDING_ANALYZER_HEARTBEAT_TOPIC = "analytics.funding.analyzer.heartbeat"


RAW_FUNDING_MARKET_TOPICS = {
    DEFAULT_RAW_FUNDING_TOPIC,
    DEFAULT_RAW_OPEN_INTEREST_TOPIC,
    DEFAULT_RAW_CANDLE_TOPIC,
    DEFAULT_RAW_TRADE_TOPIC,
    DEFAULT_RAW_LIQUIDATION_TOPIC,
}


# =============================================================================
# Validation / normalization helpers
# =============================================================================

def _validate_non_empty_topic(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty topic string")


def _validate_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in range [0, 1]")


def _normalize_topic_patterns(
    values: tuple[str, ...] | list[str] | set[str] | None,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if not values:
        return fallback

    normalized = tuple(str(item).strip() for item in values if str(item).strip())
    return normalized or fallback


def _normalize_exchange_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    normalized: set[str] = set()

    for item in values:
        exchange = normalize_exchange(item)
        normalized.add(exchange.value)

    return normalized


def _normalize_market_type_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalize_market_type(item)
        for item in values
        if str(item).strip()
    }


def _normalize_symbol_set(
    values: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalize_symbol(item)
        for item in values
        if str(item).strip()
    }


def _normalize_timeframe_set(
    values: set[FundingTimeframe | str] | list[FundingTimeframe | str] | tuple[FundingTimeframe | str, ...] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalize_timeframe(item).value
        for item in values
        if str(item).strip()
    }


def _normalize_scoped_float_mapping(
    values: dict[str, float] | None,
) -> dict[str, float]:
    """
    Підтримує два формати ключів:
    - symbol-only: "BTCUSDT"
    - scoped: "binance:usdm_futures:BTCUSDT:1h"
    """
    if not values:
        return {}

    normalized: dict[str, float] = {}

    for key, value in values.items():
        normalized[_normalize_mapping_key(key)] = float(value)

    return normalized


def _normalize_mapping_key(key: str) -> str:
    raw = str(key).strip()
    if not raw:
        raise ValueError("mapping key must not be empty")

    parts = raw.split(":")
    if len(parts) == 4:
        funding_key = make_funding_key(
            exchange=parts[0],
            market_type=parts[1],
            symbol=parts[2],
            timeframe=parts[3],
        )
        scope = funding_key_to_dict(funding_key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )

    return normalize_symbol(raw)


def _validate_non_negative_mapping(name: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        if not key:
            raise ValueError(f"{name} contains empty key")
        _validate_non_negative_float(f"{name}[{key!r}]", value)


# =============================================================================
# Funding analyzer config
# =============================================================================

@dataclass(slots=True)
class FundingAnalyzerConfig:
    """
    Runtime config для analytics.funding.FundingAnalyzer.

    Відповідальність:
    - production input topics;
    - legacy/raw topic guard;
    - analytics output topics;
    - scoped market filters;
    - rolling history/statistics settings;
    - context switches;
    - signal/event emission toggles;
    - parquet-backed history settings;
    - Scheduler intervals;
    - cache/state safety limits.

    Correct production input flow:
        exchange adapters
            -> market.funding
            -> FundingCache
            -> market.funding.updated
            -> FundingAnalyzer

    Context flow:
        OpenInterestCache -> market.open_interest.updated -> FundingAnalyzer
        CandlesCache -> market.candle.closed -> FundingAnalyzer
        TradesCache -> market.trades.updated -> FundingAnalyzer
        Orderflow analytics -> analytics.orderflow.updated -> FundingAnalyzer
        Liquidation cache/analytics -> market.liquidations.updated або analytics.liquidations.*
            -> FundingAnalyzer

    Raw market topics:
        market.funding
        market.open_interest
        market.candle
        market.trade
        market.liquidation

    не мають використовуватися в production, якщо allow_legacy_raw_topics=False.
    """

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    enabled: bool = True
    service_name: str = "funding_analyzer"

    # ------------------------------------------------------------------
    # Production input topics
    # ------------------------------------------------------------------

    funding_event_name: str = DEFAULT_FUNDING_UPDATED_TOPIC
    funding_event_patterns: tuple[str, ...] = (DEFAULT_FUNDING_UPDATED_TOPIC,)

    open_interest_event_name: str = DEFAULT_OPEN_INTEREST_UPDATED_TOPIC
    open_interest_event_patterns: tuple[str, ...] = (DEFAULT_OPEN_INTEREST_UPDATED_TOPIC,)

    candle_event_name: str = DEFAULT_CANDLE_CLOSED_TOPIC
    candle_event_patterns: tuple[str, ...] = (DEFAULT_CANDLE_CLOSED_TOPIC,)

    trade_event_name: str = DEFAULT_TRADES_UPDATED_TOPIC
    trade_event_patterns: tuple[str, ...] = (DEFAULT_TRADES_UPDATED_TOPIC,)

    cvd_event_name: str = DEFAULT_CVD_UPDATED_TOPIC
    cvd_event_patterns: tuple[str, ...] = (DEFAULT_CVD_UPDATED_TOPIC,)

    liquidation_event_name: str = DEFAULT_LIQUIDATIONS_UPDATED_TOPIC
    liquidation_event_patterns: tuple[str, ...] = (DEFAULT_LIQUIDATIONS_UPDATED_TOPIC,)

    # ------------------------------------------------------------------
    # Legacy/raw topics
    # ------------------------------------------------------------------

    raw_funding_event_name: str = DEFAULT_RAW_FUNDING_TOPIC
    raw_open_interest_event_name: str = DEFAULT_RAW_OPEN_INTEREST_TOPIC
    raw_candle_event_name: str = DEFAULT_RAW_CANDLE_TOPIC
    raw_trade_event_name: str = DEFAULT_RAW_TRADE_TOPIC
    raw_liquidation_event_name: str = DEFAULT_RAW_LIQUIDATION_TOPIC

    allow_legacy_raw_topics: bool = False

    # ------------------------------------------------------------------
    # Output topics
    # ------------------------------------------------------------------

    snapshot_event_name: str = DEFAULT_FUNDING_SNAPSHOT_TOPIC
    regime_event_name: str = DEFAULT_FUNDING_REGIME_TOPIC
    extreme_event_name: str = DEFAULT_FUNDING_EXTREME_TOPIC
    flip_event_name: str = DEFAULT_FUNDING_FLIP_TOPIC
    divergence_event_name: str = DEFAULT_FUNDING_DIVERGENCE_TOPIC
    pressure_event_name: str = DEFAULT_FUNDING_PRESSURE_TOPIC
    signal_event_name: str = DEFAULT_FUNDING_SIGNAL_TOPIC
    analytics_event_name: str = DEFAULT_FUNDING_ANALYTICS_TOPIC

    analyzer_started_event_name: str = DEFAULT_FUNDING_ANALYZER_STARTED_TOPIC
    analyzer_stopped_event_name: str = DEFAULT_FUNDING_ANALYZER_STOPPED_TOPIC
    analyzer_heartbeat_event_name: str = DEFAULT_FUNDING_ANALYZER_HEARTBEAT_TOPIC

    # ------------------------------------------------------------------
    # Scoped defaults / filters
    # ------------------------------------------------------------------

    default_market_type: str = DEFAULT_MARKET_TYPE
    default_timeframe: FundingTimeframe = DEFAULT_TIMEFRAME

    allowed_exchanges: set[str] = field(default_factory=set)
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
    allowed_symbols: set[str] = field(default_factory=set)
    allowed_timeframes: set[FundingTimeframe | str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # History/statistics
    # ------------------------------------------------------------------

    history_window_size: int = 500
    min_samples_for_statistics: int = 20
    statistics_window_size: int = 300

    max_history_per_key: int = 1_000
    max_context_age_ms: int = 60_000
    max_snapshot_age_ms: int = 30_000

    # ------------------------------------------------------------------
    # Context switches
    # ------------------------------------------------------------------

    use_open_interest_context: bool = True
    use_price_context: bool = True
    use_trades_context: bool = True
    use_cvd_context: bool = True
    use_liquidation_context: bool = True

    # ------------------------------------------------------------------
    # Event emission switches
    # ------------------------------------------------------------------

    emit_snapshots: bool = True
    emit_regime_events: bool = True
    emit_extreme_events: bool = True
    emit_flip_events: bool = True
    emit_divergence_events: bool = True
    emit_pressure_events: bool = True
    emit_signals: bool = True
    emit_analytics_events: bool = True

    # ------------------------------------------------------------------
    # Signal construction switches
    # ------------------------------------------------------------------

    signal_on_regime_change: bool = True
    signal_on_high_pressure: bool = True
    signal_on_flip: bool = True
    signal_on_extreme: bool = True
    signal_on_divergence: bool = True

    signal_min_confidence: float = 0.20
    signal_cooldown_sec: float = 10.0
    scoped_signal_cooldown_sec: dict[str, float] = field(default_factory=dict)

    min_emit_interval_ms: int = 250

    # ------------------------------------------------------------------
    # Scheduler / maintenance
    # ------------------------------------------------------------------

    cleanup_interval_sec: float = 60.0
    heartbeat_interval_sec: float = 60.0
    stale_state_ttl_sec: float = 60.0 * 60.0

    cleanup_job_name: str = "analytics.funding.cleanup"
    heartbeat_job_name: str = "analytics.funding.heartbeat"

    # ------------------------------------------------------------------
    # Historical storage / parquet
    # ------------------------------------------------------------------

    enable_parquet_history: bool = True
    parquet_base_path: str = "data/parquet"
    parquet_dataset_name: str = "analytics_funding"

    parquet_flush_interval_sec: float = 30.0
    parquet_flush_timeout_sec: float = 10.0
    parquet_flush_batch_size: int = 250
    parquet_flush_job_name: str = "analytics.funding.parquet_flush"

    load_history_from_parquet_on_start: bool = True
    parquet_max_load_records_per_key: int = 500

    # ------------------------------------------------------------------
    # Safety limits
    # ------------------------------------------------------------------

    max_tracked_keys: int = 50_000
    max_cached_contexts: int = 50_000
    max_cached_statistics: int = 50_000
    max_cached_signals: int = 25_000

    # ------------------------------------------------------------------
    # Extensibility
    # ------------------------------------------------------------------

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

        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)

        self.scoped_signal_cooldown_sec = _normalize_scoped_float_mapping(
            self.scoped_signal_cooldown_sec
        )

        self.funding_event_patterns = _normalize_topic_patterns(
            self.funding_event_patterns,
            fallback=(self.funding_event_name,),
        )
        self.open_interest_event_patterns = _normalize_topic_patterns(
            self.open_interest_event_patterns,
            fallback=(self.open_interest_event_name,),
        )
        self.candle_event_patterns = _normalize_topic_patterns(
            self.candle_event_patterns,
            fallback=(self.candle_event_name,),
        )
        self.trade_event_patterns = _normalize_topic_patterns(
            self.trade_event_patterns,
            fallback=(self.trade_event_name,),
        )
        self.cvd_event_patterns = _normalize_topic_patterns(
            self.cvd_event_patterns,
            fallback=(self.cvd_event_name,),
        )
        self.liquidation_event_patterns = _normalize_topic_patterns(
            self.liquidation_event_patterns,
            fallback=(self.liquidation_event_name,),
        )

        self.funding_event_name = self.funding_event_patterns[0]
        self.open_interest_event_name = self.open_interest_event_patterns[0]
        self.candle_event_name = self.candle_event_patterns[0]
        self.trade_event_name = self.trade_event_patterns[0]
        self.cvd_event_name = self.cvd_event_patterns[0]
        self.liquidation_event_name = self.liquidation_event_patterns[0]

        self.parquet_base_path = str(self.parquet_base_path).strip() or "data/parquet"
        self.parquet_dataset_name = str(self.parquet_dataset_name).strip() or "analytics_funding"
        self.parquet_flush_job_name = (
            str(self.parquet_flush_job_name).strip()
            or "analytics.funding.parquet_flush"
        )
        self.cleanup_job_name = (
            str(self.cleanup_job_name).strip()
            or "analytics.funding.cleanup"
        )
        self.heartbeat_job_name = (
            str(self.heartbeat_job_name).strip()
            or "analytics.funding.heartbeat"
        )

        self.metadata = dict(self.metadata or {})

        self.validate()

    # ------------------------------------------------------------------
    # Topic groups
    # ------------------------------------------------------------------

    @property
    def production_input_topics(self) -> tuple[str, ...]:
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
        return (
            *self.funding_event_patterns,
            *self.open_interest_event_patterns,
            *self.candle_event_patterns,
            *self.trade_event_patterns,
            *self.cvd_event_patterns,
            *self.liquidation_event_patterns,
        )

    @property
    def funding_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "funding_input_topics", _analytics_args)
        except Exception:
            pass
        return self.funding_event_patterns

    @property
    def context_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "context_input_topics", _analytics_args)
        except Exception:
            pass
        topics: list[str] = []

        if self.use_open_interest_context:
            topics.extend(self.open_interest_event_patterns)

        if self.use_price_context:
            topics.extend(self.candle_event_patterns)

        if self.use_trades_context:
            topics.extend(self.trade_event_patterns)

        if self.use_cvd_context:
            topics.extend(self.cvd_event_patterns)

        if self.use_liquidation_context:
            topics.extend(self.liquidation_event_patterns)

        return tuple(topics)

    @property
    def legacy_raw_input_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "legacy_raw_input_topics", _analytics_args)
        except Exception:
            pass
        return (
            self.raw_funding_event_name,
            self.raw_open_interest_event_name,
            self.raw_candle_event_name,
            self.raw_trade_event_name,
            self.raw_liquidation_event_name,
        )

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
            self.snapshot_event_name,
            self.regime_event_name,
            self.extreme_event_name,
            self.flip_event_name,
            self.divergence_event_name,
            self.pressure_event_name,
            self.signal_event_name,
            self.analytics_event_name,
        )

    @property
    def lifecycle_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "lifecycle_topics", _analytics_args)
        except Exception:
            pass
        return (
            self.analyzer_started_event_name,
            self.analyzer_stopped_event_name,
            self.analyzer_heartbeat_event_name,
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
        names = [
            self.cleanup_job_name,
            self.heartbeat_job_name,
        ]

        if self.enable_parquet_history:
            names.append(self.parquet_flush_job_name)

        return tuple(names)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
        _validate_positive_int("history_window_size", self.history_window_size)
        _validate_positive_int("min_samples_for_statistics", self.min_samples_for_statistics)
        _validate_positive_int("statistics_window_size", self.statistics_window_size)

        if self.min_samples_for_statistics > self.statistics_window_size:
            raise ValueError(
                "min_samples_for_statistics must be <= statistics_window_size"
            )

        _validate_positive_int("max_history_per_key", self.max_history_per_key)
        _validate_positive_int("max_context_age_ms", self.max_context_age_ms)
        _validate_positive_int("max_snapshot_age_ms", self.max_snapshot_age_ms)

        _validate_ratio("signal_min_confidence", self.signal_min_confidence)
        _validate_non_negative_float("signal_cooldown_sec", self.signal_cooldown_sec)
        _validate_non_negative_int("min_emit_interval_ms", self.min_emit_interval_ms)

        _validate_non_negative_mapping(
            "scoped_signal_cooldown_sec",
            self.scoped_signal_cooldown_sec,
        )

        _validate_positive_float("cleanup_interval_sec", self.cleanup_interval_sec)
        _validate_positive_float("heartbeat_interval_sec", self.heartbeat_interval_sec)
        _validate_positive_float("stale_state_ttl_sec", self.stale_state_ttl_sec)

        _validate_non_empty_string("cleanup_job_name", self.cleanup_job_name)
        _validate_non_empty_string("heartbeat_job_name", self.heartbeat_job_name)

        _validate_non_empty_string("parquet_base_path", self.parquet_base_path)
        _validate_non_empty_string("parquet_dataset_name", self.parquet_dataset_name)
        _validate_non_empty_string("parquet_flush_job_name", self.parquet_flush_job_name)

        _validate_positive_float("parquet_flush_interval_sec", self.parquet_flush_interval_sec)
        _validate_positive_float("parquet_flush_timeout_sec", self.parquet_flush_timeout_sec)
        _validate_positive_int("parquet_flush_batch_size", self.parquet_flush_batch_size)
        _validate_positive_int(
            "parquet_max_load_records_per_key",
            self.parquet_max_load_records_per_key,
        )

        _validate_positive_int("max_tracked_keys", self.max_tracked_keys)
        _validate_positive_int("max_cached_contexts", self.max_cached_contexts)
        _validate_positive_int("max_cached_statistics", self.max_cached_statistics)
        _validate_positive_int("max_cached_signals", self.max_cached_signals)

        self._validate_all_topics()

        if not self.allow_legacy_raw_topics:
            used_raw_topics = set(self.production_input_topics).intersection(
                set(self.legacy_raw_input_topics)
            )
            if used_raw_topics:
                raise ValueError(
                    "FundingAnalyzer production input topics must use data-layer "
                    f"updated topics, not raw topics: {sorted(used_raw_topics)}"
                )

    def _validate_all_topics(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_all_topics", _analytics_args)
        except Exception:
            pass
        for topic in self.production_input_topics:
            _validate_non_empty_topic("production_input_topics item", topic)

        for topic in self.legacy_raw_input_topics:
            _validate_non_empty_topic("legacy_raw_input_topics item", topic)

        for topic in self.output_topics:
            _validate_non_empty_topic("output_topics item", topic)

        for topic in self.lifecycle_topics:
            _validate_non_empty_topic("lifecycle_topics item", topic)

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def make_key(
        self,
        *,
        exchange: str,
        market_type: str | None,
        symbol: str,
        timeframe: FundingTimeframe | str | None = None,
    ) -> FundingKey:
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
        return make_funding_key(
            exchange=exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_key(self, key: FundingKey) -> bool:
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
        scope = funding_key_to_dict(key)

        if self.allowed_exchanges and scope["exchange"] not in self.allowed_exchanges:
            return False

        if self.allowed_market_types and scope["market_type"] not in self.allowed_market_types:
            return False

        if self.allowed_symbols and scope["symbol"] not in self.allowed_symbols:
            return False

        if self.allowed_timeframes and scope["timeframe"] not in self.allowed_timeframes:
            return False

        return True

    def should_process_scope(
        self,
        *,
        exchange: str,
        market_type: str | None,
        symbol: str,
        timeframe: FundingTimeframe | str | None = None,
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

    @staticmethod
    def scoped_mapping_key(key: FundingKey) -> str:
        try:
            _analytics_class_name = "FundingAnalyzerConfig"
            _analytics_logger = get_logger(f"{__name__}.{_analytics_class_name}")
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
        scope = funding_key_to_dict(key)
        return (
            f"{scope['exchange']}:"
            f"{scope['market_type']}:"
            f"{scope['symbol']}:"
            f"{scope['timeframe']}"
        )

    def get_signal_cooldown(self, key: FundingKey) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "get_signal_cooldown", _analytics_args)
        except Exception:
            pass
        scoped_key = self.scoped_mapping_key(key)
        return self.scoped_signal_cooldown_sec.get(
            scoped_key,
            self.signal_cooldown_sec,
        )

    # ------------------------------------------------------------------
    # Topic guards
    # ------------------------------------------------------------------

    def is_raw_topic(self, topic: str) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_raw_topic", _analytics_args)
        except Exception:
            pass
        return topic in RAW_FUNDING_MARKET_TOPICS

    def assert_topic_allowed(
        self,
        topic: str,
        *,
        allow_raw: bool = False,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "assert_topic_allowed", _analytics_args)
        except Exception:
            pass
        _validate_non_empty_topic("topic", topic)

        if self.is_raw_topic(topic) and not allow_raw:
            raise ValueError(
                f"Raw topic {topic!r} is not allowed for FundingAnalyzer production "
                "subscriptions. Use data-layer updated topics instead."
            )

        if self.is_raw_topic(topic) and allow_raw and not self.allow_legacy_raw_topics:
            raise ValueError(
                f"Raw topic {topic!r} requested, but allow_legacy_raw_topics=False."
            )

    def assert_production_topics_allowed(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "assert_production_topics_allowed", _analytics_args)
        except Exception:
            pass
        for topic in self.production_input_topics:
            self.assert_topic_allowed(topic, allow_raw=False)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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
            "service_name": self.service_name,
            "scope": "exchange:market_type:symbol:timeframe",
            "default_market_type": self.default_market_type,
            "default_timeframe": self.default_timeframe.value,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "allowed_market_types": sorted(self.allowed_market_types),
            "allowed_symbols": sorted(self.allowed_symbols),
            "allowed_timeframes": sorted(self.allowed_timeframes),
            "production_input_topics": list(self.production_input_topics),
            "funding_input_topics": list(self.funding_input_topics),
            "context_input_topics": list(self.context_input_topics),
            "legacy_raw_input_topics": list(self.legacy_raw_input_topics),
            "allow_legacy_raw_topics": self.allow_legacy_raw_topics,
            "output_topics": list(self.output_topics),
            "lifecycle_topics": list(self.lifecycle_topics),
            "scheduler_job_names": list(self.scheduler_job_names),
            "history_window_size": self.history_window_size,
            "min_samples_for_statistics": self.min_samples_for_statistics,
            "statistics_window_size": self.statistics_window_size,
            "max_history_per_key": self.max_history_per_key,
            "max_context_age_ms": self.max_context_age_ms,
            "max_snapshot_age_ms": self.max_snapshot_age_ms,
            "use_open_interest_context": self.use_open_interest_context,
            "use_price_context": self.use_price_context,
            "use_trades_context": self.use_trades_context,
            "use_cvd_context": self.use_cvd_context,
            "use_liquidation_context": self.use_liquidation_context,
            "emit_snapshots": self.emit_snapshots,
            "emit_regime_events": self.emit_regime_events,
            "emit_extreme_events": self.emit_extreme_events,
            "emit_flip_events": self.emit_flip_events,
            "emit_divergence_events": self.emit_divergence_events,
            "emit_pressure_events": self.emit_pressure_events,
            "emit_signals": self.emit_signals,
            "emit_analytics_events": self.emit_analytics_events,
            "signal_on_regime_change": self.signal_on_regime_change,
            "signal_on_high_pressure": self.signal_on_high_pressure,
            "signal_on_flip": self.signal_on_flip,
            "signal_on_extreme": self.signal_on_extreme,
            "signal_on_divergence": self.signal_on_divergence,
            "signal_min_confidence": self.signal_min_confidence,
            "signal_cooldown_sec": self.signal_cooldown_sec,
            "scoped_signal_cooldown_sec": dict(self.scoped_signal_cooldown_sec),
            "min_emit_interval_ms": self.min_emit_interval_ms,
            "cleanup_interval_sec": self.cleanup_interval_sec,
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "stale_state_ttl_sec": self.stale_state_ttl_sec,
            "cleanup_job_name": self.cleanup_job_name,
            "heartbeat_job_name": self.heartbeat_job_name,
            "enable_parquet_history": self.enable_parquet_history,
            "parquet_base_path": self.parquet_base_path,
            "parquet_dataset_name": self.parquet_dataset_name,
            "parquet_flush_interval_sec": self.parquet_flush_interval_sec,
            "parquet_flush_timeout_sec": self.parquet_flush_timeout_sec,
            "parquet_flush_batch_size": self.parquet_flush_batch_size,
            "parquet_flush_job_name": self.parquet_flush_job_name,
            "load_history_from_parquet_on_start": self.load_history_from_parquet_on_start,
            "parquet_max_load_records_per_key": self.parquet_max_load_records_per_key,
            "max_tracked_keys": self.max_tracked_keys,
            "max_cached_contexts": self.max_cached_contexts,
            "max_cached_statistics": self.max_cached_statistics,
            "max_cached_signals": self.max_cached_signals,
            "metadata": dict(self.metadata),
        }


__all__ = [
    # production topics
    "DEFAULT_FUNDING_UPDATED_TOPIC",
    "DEFAULT_OPEN_INTEREST_UPDATED_TOPIC",
    "DEFAULT_CANDLE_CLOSED_TOPIC",
    "DEFAULT_TRADES_UPDATED_TOPIC",
    "DEFAULT_CVD_UPDATED_TOPIC",
    "DEFAULT_LIQUIDATIONS_UPDATED_TOPIC",

    # raw topics
    "DEFAULT_RAW_FUNDING_TOPIC",
    "DEFAULT_RAW_OPEN_INTEREST_TOPIC",
    "DEFAULT_RAW_CANDLE_TOPIC",
    "DEFAULT_RAW_TRADE_TOPIC",
    "DEFAULT_RAW_LIQUIDATION_TOPIC",
    "RAW_FUNDING_MARKET_TOPICS",

    # output topics
    "DEFAULT_FUNDING_SNAPSHOT_TOPIC",
    "DEFAULT_FUNDING_REGIME_TOPIC",
    "DEFAULT_FUNDING_EXTREME_TOPIC",
    "DEFAULT_FUNDING_FLIP_TOPIC",
    "DEFAULT_FUNDING_DIVERGENCE_TOPIC",
    "DEFAULT_FUNDING_PRESSURE_TOPIC",
    "DEFAULT_FUNDING_SIGNAL_TOPIC",
    "DEFAULT_FUNDING_ANALYTICS_TOPIC",

    # lifecycle topics
    "DEFAULT_FUNDING_ANALYZER_STARTED_TOPIC",
    "DEFAULT_FUNDING_ANALYZER_STOPPED_TOPIC",
    "DEFAULT_FUNDING_ANALYZER_HEARTBEAT_TOPIC",

    # config
    "FundingAnalyzerConfig",
]