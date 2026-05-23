from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass, field

from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    SpoofingKey,
    make_spoofing_key,
    spoofing_key_to_dict,
)
from ..liquidity.config import DEFAULT_EXCHANGE

# =============================================================================
# Project scope defaults
# =============================================================================

# Market-data exchanges used by the project.
# Bitget is intentionally excluded.
PROJECT_EXCHANGES: tuple[str, ...] = (
    "binance",
    "bybit",
    "okx",
    "mexc",
)

# Futures/perpetual market types used across supported exchanges.
# Binance USD-M Futures remains the execution-first default.
PROJECT_MARKET_TYPES: tuple[str, ...] = (
    "usdm_futures",
    "linear",
    "swap",
)

# Timeframes currently used by the project pipeline/backtesting flow.
PROJECT_TIMEFRAMES: tuple[str, ...] = (
    "1m",
    "15m",
)


def _default_exchange_allowlist() -> set[str]:
    return set(PROJECT_EXCHANGES)


def _default_market_type_allowlist() -> set[str]:
    return set(PROJECT_MARKET_TYPES)


def _default_timeframe_allowlist() -> set[str]:
    return set(PROJECT_TIMEFRAMES)


# =============================================================================
# Helpers
# =============================================================================


def _normalize_exchange(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    return normalized or None


def _normalize_market_type(value: str | None) -> str:
    normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized or DEFAULT_MARKET_TYPE


def _normalize_symbol(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip().upper()
    return normalized or None


def _normalize_timeframe(value: str | None) -> str:
    normalized = str(value or DEFAULT_TIMEFRAME).strip()
    return normalized or DEFAULT_TIMEFRAME


def _normalize_symbol_allowlist(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not values:
        return None

    normalized = {
        symbol
        for item in values
        if (symbol := _normalize_symbol(str(item))) is not None
    }

    return normalized or None


def _normalize_exchange_allowlist(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not values:
        return None

    normalized = {
        exchange
        for item in values
        if (exchange := _normalize_exchange(str(item))) is not None
    }

    return normalized or None


def _normalize_market_type_allowlist(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not values:
        return None

    normalized = {
        _normalize_market_type(str(item))
        for item in values
        if str(item).strip()
    }

    return normalized or None


def _normalize_timeframe_allowlist(values: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not values:
        return None

    normalized = {
        _normalize_timeframe(str(item))
        for item in values
        if str(item).strip()
    }

    return normalized or None


def _normalize_topic_patterns(values: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    normalized = tuple(
        str(item).strip()
        for item in values
        if str(item).strip()
    )

    if not normalized:
        raise ValueError("topic patterns must not be empty")

    return normalized


# =============================================================================
# Detector configs
# =============================================================================


@dataclass(slots=True)
class WallDetectionConfig:
    """
    Налаштування виявлення великих стін у стакані.

    Використовується OrderbookWallDetector.
    """

    enabled: bool = True

    min_wall_size_abs: float = 100_000.0
    min_wall_size_ratio: float = 3.0
    max_distance_from_mid_bps: float = 20.0
    near_best_quote_bps: float = 5.0

    min_levels_to_scan: int = 10
    max_levels_to_scan: int = 50


@dataclass(slots=True)
class PersistenceTrackerConfig:
    """
    Налаштування життєвого циклу tracked walls.

    PersistenceTracker не запускає власні loops. Періодичний cleanup має
    реєструвати SpoofingAnalyzer через core.scheduler.Scheduler.add_interval_job().
    """

    enabled: bool = True

    wall_ttl_ms: int = 15_000
    min_tracking_lifetime_ms: int = 50
    cleanup_interval_ms: int = 2_000

    # New key-first limit.
    max_walls_per_key: int = 500

    # Backward-compatible legacy name. Не використовувати в новому коді.
    max_walls_per_symbol: int = 500

    max_history_events_per_level: int = 200

    size_update_epsilon: float = 1e-9
    price_rounding_decimals: int = 8

    estimate_fill_from_trade_flow: bool = False
    estimate_fill_on_touch_only: bool = True

    def __post_init__(self) -> None:
        # Якщо старий config передає тільки max_walls_per_symbol, не ламаємо його.
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
        if self.max_walls_per_key == 500 and self.max_walls_per_symbol != 500:
            self.max_walls_per_key = self.max_walls_per_symbol
        else:
            self.max_walls_per_symbol = self.max_walls_per_key


@dataclass(slots=True)
class PullDetectionConfig:
    """
    Налаштування виявлення швидкого зняття ліквідності.

    Використовується OrderPullDetector.
    """

    enabled: bool = True

    max_pull_lifetime_ms: int = 2_500
    min_pull_ratio: float = 0.60
    max_fill_ratio_for_pull: float = 0.25
    min_removed_notional: float = 50_000.0

    fast_pull_lifetime_ms: int = 750
    strong_pull_ratio: float = 0.85


@dataclass(slots=True)
class FakeLiquidityConfig:
    """
    Налаштування виявлення фейкової ліквідності.

    Використовується FakeLiquidityDetector.
    """

    enabled: bool = True

    max_fill_ratio: float = 0.20
    min_pull_ratio: float = 0.70
    max_lifetime_ms: int = 4_000
    min_price_reaction_bps: float = 2.0


@dataclass(slots=True)
class LayeringConfig:
    """
    Налаштування multi-level layering detection.

    Використовується LayeringDetector.
    """

    enabled: bool = True

    min_layers: int = 3
    max_price_gap_bps_between_layers: float = 5.0
    min_total_layer_notional: float = 150_000.0
    synchronized_pull_window_ms: int = 1_000


@dataclass(slots=True)
class FlipPressureConfig:
    """
    Налаштування pressure flip / pressure bluff detection.

    Використовується FlipPressureDetector.
    """

    enabled: bool = True

    min_price_reaction_bps: float = 3.0
    reaction_window_ms: int = 3_000
    min_pressure_flip_strength: float = 0.60


@dataclass(slots=True)
class SpoofingScoreConfig:
    """
    Ваги та пороги фінального spoofing score.

    Використовується SpoofingScoreEngine.
    """

    enabled: bool = True

    detection_threshold: float = 0.65
    high_severity_threshold: float = 0.80
    critical_severity_threshold: float = 0.92

    weight_wall_size: float = 0.18
    weight_wall_distance: float = 0.10
    weight_persistence: float = 0.10
    weight_pull_speed: float = 0.18
    weight_fill_ratio: float = 0.14
    weight_price_reaction: float = 0.14
    weight_repetition: float = 0.08
    weight_layering: float = 0.08

    min_confidence: float = 0.30
    confidence_boost_on_detector_agreement: float = 0.10
    max_confidence: float = 0.99


@dataclass(slots=True)
class CandidateTrackingConfig:
    """
    Legacy-compatible candidate tracking config.

    У новій архітектурі основний state веде PersistenceTracker, але ці
    параметри можуть бути корисні для міграції старої candidate-based логіки,
    cooldown-ів або trade-flow confirmation.
    """

    enabled: bool = False

    candidate_ttl_ms: int = 12_000
    cooldown_ms_same_level: int = 15_000

    max_candidates_per_key: int = 200
    max_trade_events_per_key: int = 2_000

    # Backward-compatible legacy names. Не використовувати в новому коді.
    max_candidates_per_symbol: int = 200
    max_trade_events_per_symbol: int = 2_000

    similar_candidate_tolerance_bps: float = 1.0

    trade_pressure_window_ms: int = 3_000
    min_opposite_pressure_ratio: float = 1.35

    price_move_confirmation_bps: float = 4.0
    logical_invalidation_distance_multiplier: float = 3.0

    emit_raw_candidate_events: bool = False

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
        if self.max_candidates_per_key == 200 and self.max_candidates_per_symbol != 200:
            self.max_candidates_per_key = self.max_candidates_per_symbol
        else:
            self.max_candidates_per_symbol = self.max_candidates_per_key

        if self.max_trade_events_per_key == 2_000 and self.max_trade_events_per_symbol != 2_000:
            self.max_trade_events_per_key = self.max_trade_events_per_symbol
        else:
            self.max_trade_events_per_symbol = self.max_trade_events_per_key


# =============================================================================
# Analyzer config
# =============================================================================


@dataclass(slots=True)
class SpoofingAnalyzerConfig:
    """
    Загальний конфіг SpoofingAnalyzer.

    Analyzer є єдиним integration point пакета:
    - підписується на data-layer topics через EventBus;
    - читає OrderBookCache / TradesCache через scoped key;
    - запускає cleanup через Scheduler;
    - публікує тільки analytics.spoofing.* події.

    Production input flow:
        exchange adapters
            -> market.orderbook / market.trade
            -> OrderBookCache / TradesCache
            -> market.orderbook.updated / market.trades.updated
            -> analytics.spoofing
    """

    enabled: bool = True

    publish_updates: bool = True
    publish_detected_only: bool = False
    publish_lifecycle_events: bool = False
    publish_score_updates: bool = True
    publish_errors: bool = True

    max_tracked_walls_per_key: int = 500
    max_detector_results_per_cycle: int = 50

    # Backward-compatible legacy name. Не використовувати в новому коді.
    max_tracked_walls_per_symbol: int = 500

    # Production source topics: only data-layer updated events.
    event_topic_orderbook: str = "market.orderbook.updated"
    event_topic_trade: str = "market.trades.updated"

    source_topic_patterns_orderbook: tuple[str, ...] = ("market.orderbook.updated",)
    source_topic_patterns_trade: tuple[str, ...] = ("market.trades.updated",)

    # Legacy/raw topics are intentionally separated and disabled by default.
    # They may be used only in migration tests/manual tools, not production runtime.
    legacy_raw_orderbook_topic: str = "market.orderbook"
    legacy_raw_trade_topic: str = "market.trade"
    allow_legacy_raw_topics: bool = False

    event_topic_lifecycle: str = "analytics.spoofing.lifecycle"
    event_topic_updated: str = "analytics.spoofing.updated"
    event_topic_detected: str = "analytics.spoofing.detected"
    event_topic_score_updated: str = "analytics.spoofing.score_updated"
    event_topic_error: str = "analytics.spoofing.error"

    scheduler_cleanup_job_name: str = "analytics.spoofing.persistence_cleanup"
    scheduler_cleanup_enabled: bool = True

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
        if self.max_tracked_walls_per_key == 500 and self.max_tracked_walls_per_symbol != 500:
            self.max_tracked_walls_per_key = self.max_tracked_walls_per_symbol
        else:
            self.max_tracked_walls_per_symbol = self.max_tracked_walls_per_key

        self.source_topic_patterns_orderbook = _normalize_topic_patterns(
            self.source_topic_patterns_orderbook
        )
        self.source_topic_patterns_trade = _normalize_topic_patterns(
            self.source_topic_patterns_trade
        )

        # Keep single-topic fields aligned with production topic patterns.
        self.event_topic_orderbook = self.source_topic_patterns_orderbook[0]
        self.event_topic_trade = self.source_topic_patterns_trade[0]

    @property
    def source_topic_patterns(self) -> tuple[str, ...]:
        """
        Усі production source topic patterns для SpoofingAnalyzer.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "source_topic_patterns", _analytics_args)
        except Exception:
            pass
        return (
            *self.source_topic_patterns_orderbook,
            *self.source_topic_patterns_trade,
        )

    @property
    def legacy_raw_topic_patterns(self) -> tuple[str, ...]:
        """
        Legacy/raw topics для міграції або ручних тестів.
        Не підключати в production, якщо allow_legacy_raw_topics=False.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "legacy_raw_topic_patterns", _analytics_args)
        except Exception:
            pass
        return (
            self.legacy_raw_orderbook_topic,
            self.legacy_raw_trade_topic,
        )


# =============================================================================
# Root config
# =============================================================================


@dataclass(slots=True)
class SpoofingConfig:
    """
    Кореневий конфіг пакета analytics.spoofing.

    Цей config не дублює core.config.Config. Він є доменною typed config-моделлю
    і має передаватися в SpoofingAnalyzer / detector-и через constructor
    dependency injection.

    Correct scope:
        exchange + market_type + symbol + timeframe

    New code should prefer:
        default_exchange/default_market_type/default_timeframe
        exchange_allowlist/symbol_allowlist/market_type_allowlist/timeframe_allowlist

    Legacy fields:
        exchange
        symbols
    """

    enabled: bool = True

    # Scoped defaults. Symbol-only processing is unsafe without default_exchange.
    default_exchange: str = DEFAULT_EXCHANGE
    default_market_type: str = DEFAULT_MARKET_TYPE
    default_timeframe: str = DEFAULT_TIMEFRAME

    # Scoped allowlists. Defaults are populated with the exchanges, futures
    # market types and timeframes used by the project.
    exchange_allowlist: set[str] | None = field(default_factory=_default_exchange_allowlist)
    market_type_allowlist: set[str] | None = field(default_factory=_default_market_type_allowlist)
    symbol_allowlist: set[str] | None = None
    timeframe_allowlist: set[str] | None = field(default_factory=_default_timeframe_allowlist)

    # Legacy-compatible fields.
    exchange: str | None = None
    symbols: list[str] = field(default_factory=list)

    wall_detection: WallDetectionConfig = field(default_factory=WallDetectionConfig)
    persistence: PersistenceTrackerConfig = field(default_factory=PersistenceTrackerConfig)
    pull_detection: PullDetectionConfig = field(default_factory=PullDetectionConfig)
    fake_liquidity: FakeLiquidityConfig = field(default_factory=FakeLiquidityConfig)
    layering: LayeringConfig = field(default_factory=LayeringConfig)
    flip_pressure: FlipPressureConfig = field(default_factory=FlipPressureConfig)
    scoring: SpoofingScoreConfig = field(default_factory=SpoofingScoreConfig)
    candidate_tracking: CandidateTrackingConfig = field(default_factory=CandidateTrackingConfig)
    analyzer: SpoofingAnalyzerConfig = field(default_factory=SpoofingAnalyzerConfig)

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
        legacy_exchange = _normalize_exchange(self.exchange)
        normalized_default_exchange = _normalize_exchange(self.default_exchange)

        # Binance USD-M Futures is the project default for execution-capable
        # futures scope. If imported DEFAULT_EXCHANGE is empty, fall back to
        # Binance instead of leaving scope empty and breaking startup.
        self.default_exchange = normalized_default_exchange or legacy_exchange or "binance"
        self.exchange = self.default_exchange

        self.default_market_type = _normalize_market_type(self.default_market_type)
        self.default_timeframe = _normalize_timeframe(self.default_timeframe)

        self.exchange_allowlist = (
            _normalize_exchange_allowlist(self.exchange_allowlist)
            or _default_exchange_allowlist()
        )
        self.market_type_allowlist = (
            _normalize_market_type_allowlist(self.market_type_allowlist)
            or _default_market_type_allowlist()
        )
        self.symbol_allowlist = _normalize_symbol_allowlist(self.symbol_allowlist)
        self.timeframe_allowlist = (
            _normalize_timeframe_allowlist(self.timeframe_allowlist)
            or _default_timeframe_allowlist()
        )

        legacy_symbols = _normalize_symbol_allowlist(self.symbols)
        if self.symbol_allowlist is None and legacy_symbols is not None:
            self.symbol_allowlist = legacy_symbols

        self.symbols = sorted(self.symbol_allowlist or [])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Перевіряє config на логічну коректність.

        Валідація виконується явно на етапі bootstrap або перед передачею
        config у SpoofingAnalyzer.
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
        self._validate_scope_defaults()
        self._validate_wall_detection()
        self._validate_persistence()
        self._validate_pull_detection()
        self._validate_fake_liquidity()
        self._validate_layering()
        self._validate_flip_pressure()
        self._validate_scoring()
        self._validate_candidate_tracking()
        self._validate_analyzer()

    def _validate_scope_defaults(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_scope_defaults", _analytics_args)
        except Exception:
            pass
        if self.default_exchange is not None:
            self._validate_non_empty_string("default_exchange", self.default_exchange)

        self._validate_non_empty_string("default_market_type", self.default_market_type)
        self._validate_non_empty_string("default_timeframe", self.default_timeframe)

        for exchange in self.exchange_allowlist or ():
            self._validate_non_empty_string("exchange_allowlist item", exchange)

        for market_type in self.market_type_allowlist or ():
            self._validate_non_empty_string("market_type_allowlist item", market_type)

        for symbol in self.symbol_allowlist or ():
            self._validate_non_empty_string("symbol_allowlist item", symbol)

        for timeframe in self.timeframe_allowlist or ():
            self._validate_non_empty_string("timeframe_allowlist item", timeframe)

    def _validate_wall_detection(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_wall_detection", _analytics_args)
        except Exception:
            pass
        self._validate_non_negative(
            "wall_detection.min_wall_size_abs",
            self.wall_detection.min_wall_size_abs,
        )
        self._validate_non_negative(
            "wall_detection.min_wall_size_ratio",
            self.wall_detection.min_wall_size_ratio,
        )
        self._validate_non_negative(
            "wall_detection.max_distance_from_mid_bps",
            self.wall_detection.max_distance_from_mid_bps,
        )
        self._validate_non_negative(
            "wall_detection.near_best_quote_bps",
            self.wall_detection.near_best_quote_bps,
        )
        self._validate_positive_int(
            "wall_detection.min_levels_to_scan",
            self.wall_detection.min_levels_to_scan,
        )
        self._validate_positive_int(
            "wall_detection.max_levels_to_scan",
            self.wall_detection.max_levels_to_scan,
        )

        if self.wall_detection.min_levels_to_scan > self.wall_detection.max_levels_to_scan:
            raise ValueError(
                "wall_detection.min_levels_to_scan must be <= "
                "wall_detection.max_levels_to_scan"
            )

    def _validate_persistence(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_persistence", _analytics_args)
        except Exception:
            pass
        self._validate_positive_int(
            "persistence.wall_ttl_ms",
            self.persistence.wall_ttl_ms,
        )
        self._validate_non_negative_int(
            "persistence.min_tracking_lifetime_ms",
            self.persistence.min_tracking_lifetime_ms,
        )
        self._validate_positive_int(
            "persistence.cleanup_interval_ms",
            self.persistence.cleanup_interval_ms,
        )
        self._validate_positive_int(
            "persistence.max_walls_per_key",
            self.persistence.max_walls_per_key,
        )
        self._validate_positive_int(
            "persistence.max_history_events_per_level",
            self.persistence.max_history_events_per_level,
        )
        self._validate_non_negative(
            "persistence.size_update_epsilon",
            self.persistence.size_update_epsilon,
        )
        self._validate_non_negative_int(
            "persistence.price_rounding_decimals",
            self.persistence.price_rounding_decimals,
        )

    def _validate_pull_detection(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_pull_detection", _analytics_args)
        except Exception:
            pass
        self._validate_positive_int(
            "pull_detection.max_pull_lifetime_ms",
            self.pull_detection.max_pull_lifetime_ms,
        )
        self._validate_ratio(
            "pull_detection.min_pull_ratio",
            self.pull_detection.min_pull_ratio,
        )
        self._validate_ratio(
            "pull_detection.max_fill_ratio_for_pull",
            self.pull_detection.max_fill_ratio_for_pull,
        )
        self._validate_non_negative(
            "pull_detection.min_removed_notional",
            self.pull_detection.min_removed_notional,
        )
        self._validate_positive_int(
            "pull_detection.fast_pull_lifetime_ms",
            self.pull_detection.fast_pull_lifetime_ms,
        )
        self._validate_ratio(
            "pull_detection.strong_pull_ratio",
            self.pull_detection.strong_pull_ratio,
        )

        if self.pull_detection.fast_pull_lifetime_ms > self.pull_detection.max_pull_lifetime_ms:
            raise ValueError(
                "pull_detection.fast_pull_lifetime_ms must be <= "
                "pull_detection.max_pull_lifetime_ms"
            )

    def _validate_fake_liquidity(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_fake_liquidity", _analytics_args)
        except Exception:
            pass
        self._validate_ratio(
            "fake_liquidity.max_fill_ratio",
            self.fake_liquidity.max_fill_ratio,
        )
        self._validate_ratio(
            "fake_liquidity.min_pull_ratio",
            self.fake_liquidity.min_pull_ratio,
        )
        self._validate_positive_int(
            "fake_liquidity.max_lifetime_ms",
            self.fake_liquidity.max_lifetime_ms,
        )
        self._validate_non_negative(
            "fake_liquidity.min_price_reaction_bps",
            self.fake_liquidity.min_price_reaction_bps,
        )

    def _validate_layering(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_layering", _analytics_args)
        except Exception:
            pass
        self._validate_positive_int(
            "layering.min_layers",
            self.layering.min_layers,
        )
        self._validate_non_negative(
            "layering.max_price_gap_bps_between_layers",
            self.layering.max_price_gap_bps_between_layers,
        )
        self._validate_non_negative(
            "layering.min_total_layer_notional",
            self.layering.min_total_layer_notional,
        )
        self._validate_positive_int(
            "layering.synchronized_pull_window_ms",
            self.layering.synchronized_pull_window_ms,
        )

    def _validate_flip_pressure(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_flip_pressure", _analytics_args)
        except Exception:
            pass
        self._validate_non_negative(
            "flip_pressure.min_price_reaction_bps",
            self.flip_pressure.min_price_reaction_bps,
        )
        self._validate_positive_int(
            "flip_pressure.reaction_window_ms",
            self.flip_pressure.reaction_window_ms,
        )
        self._validate_ratio(
            "flip_pressure.min_pressure_flip_strength",
            self.flip_pressure.min_pressure_flip_strength,
        )

    def _validate_scoring(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_scoring", _analytics_args)
        except Exception:
            pass
        self._validate_ratio(
            "scoring.detection_threshold",
            self.scoring.detection_threshold,
        )
        self._validate_ratio(
            "scoring.high_severity_threshold",
            self.scoring.high_severity_threshold,
        )
        self._validate_ratio(
            "scoring.critical_severity_threshold",
            self.scoring.critical_severity_threshold,
        )
        self._validate_ratio(
            "scoring.min_confidence",
            self.scoring.min_confidence,
        )
        self._validate_ratio(
            "scoring.confidence_boost_on_detector_agreement",
            self.scoring.confidence_boost_on_detector_agreement,
        )
        self._validate_ratio(
            "scoring.max_confidence",
            self.scoring.max_confidence,
        )

        if self.scoring.high_severity_threshold > self.scoring.critical_severity_threshold:
            raise ValueError(
                "scoring.high_severity_threshold must be <= "
                "scoring.critical_severity_threshold"
            )

        if self.scoring.min_confidence > self.scoring.max_confidence:
            raise ValueError(
                "scoring.min_confidence must be <= scoring.max_confidence"
            )

        weights = self.scoring_weights()
        if any(weight < 0 for weight in weights):
            raise ValueError("all scoring weights must be >= 0")

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("sum of scoring weights must be > 0")

    def _validate_candidate_tracking(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_candidate_tracking", _analytics_args)
        except Exception:
            pass
        self._validate_positive_int(
            "candidate_tracking.candidate_ttl_ms",
            self.candidate_tracking.candidate_ttl_ms,
        )
        self._validate_non_negative_int(
            "candidate_tracking.cooldown_ms_same_level",
            self.candidate_tracking.cooldown_ms_same_level,
        )
        self._validate_positive_int(
            "candidate_tracking.max_candidates_per_key",
            self.candidate_tracking.max_candidates_per_key,
        )
        self._validate_positive_int(
            "candidate_tracking.max_trade_events_per_key",
            self.candidate_tracking.max_trade_events_per_key,
        )
        self._validate_non_negative(
            "candidate_tracking.similar_candidate_tolerance_bps",
            self.candidate_tracking.similar_candidate_tolerance_bps,
        )
        self._validate_positive_int(
            "candidate_tracking.trade_pressure_window_ms",
            self.candidate_tracking.trade_pressure_window_ms,
        )
        self._validate_positive(
            "candidate_tracking.min_opposite_pressure_ratio",
            self.candidate_tracking.min_opposite_pressure_ratio,
        )
        self._validate_non_negative(
            "candidate_tracking.price_move_confirmation_bps",
            self.candidate_tracking.price_move_confirmation_bps,
        )
        self._validate_positive(
            "candidate_tracking.logical_invalidation_distance_multiplier",
            self.candidate_tracking.logical_invalidation_distance_multiplier,
        )

    def _validate_analyzer(self) -> None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_analyzer", _analytics_args)
        except Exception:
            pass
        self._validate_positive_int(
            "analyzer.max_tracked_walls_per_key",
            self.analyzer.max_tracked_walls_per_key,
        )
        self._validate_positive_int(
            "analyzer.max_detector_results_per_cycle",
            self.analyzer.max_detector_results_per_cycle,
        )

        if self.analyzer.scheduler_cleanup_enabled:
            if not self.analyzer.scheduler_cleanup_job_name.strip():
                raise ValueError("analyzer.scheduler_cleanup_job_name must not be empty")

        for topic in self.analyzer.source_topic_patterns_orderbook:
            self._validate_topic("analyzer.source_topic_patterns_orderbook item", topic)

        for topic in self.analyzer.source_topic_patterns_trade:
            self._validate_topic("analyzer.source_topic_patterns_trade item", topic)

        if not self.analyzer.allow_legacy_raw_topics:
            forbidden = {
                self.analyzer.legacy_raw_orderbook_topic,
                self.analyzer.legacy_raw_trade_topic,
            }

            production_topics = set(self.analyzer.source_topic_patterns)
            raw_topics_used = production_topics.intersection(forbidden)

            if raw_topics_used:
                raise ValueError(
                    "SpoofingAnalyzer production source topics must use data-layer "
                    f"updated topics, not raw exchange topics: {sorted(raw_topics_used)}"
                )

        self._validate_topic("analyzer.event_topic_lifecycle", self.analyzer.event_topic_lifecycle)
        self._validate_topic("analyzer.event_topic_updated", self.analyzer.event_topic_updated)
        self._validate_topic("analyzer.event_topic_detected", self.analyzer.event_topic_detected)
        self._validate_topic(
            "analyzer.event_topic_score_updated",
            self.analyzer.event_topic_score_updated,
        )
        self._validate_topic("analyzer.event_topic_error", self.analyzer.event_topic_error)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def scoring_weights(self) -> list[float]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "scoring_weights", _analytics_args)
        except Exception:
            pass
        return [
            self.scoring.weight_wall_size,
            self.scoring.weight_wall_distance,
            self.scoring.weight_persistence,
            self.scoring.weight_pull_speed,
            self.scoring.weight_fill_ratio,
            self.scoring.weight_price_reaction,
            self.scoring.weight_repetition,
            self.scoring.weight_layering,
        ]

    @property
    def cleanup_interval_seconds(self) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "cleanup_interval_seconds", _analytics_args)
        except Exception:
            pass
        return self.persistence.cleanup_interval_ms / 1000.0

    def make_default_key(self, *, symbol: str, timeframe: str | None = None) -> SpoofingKey:
        """
        Backward-compatible helper для symbol-based викликів.

        У multi-exchange режимі symbol-only небезпечний, тому default_exchange
        обов'язковий.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "make_default_key", _analytics_args)
        except Exception:
            pass
        if not self.default_exchange:
            raise ValueError(
                "make_default_key(symbol) requires default_exchange. "
                "Use make_key(exchange=..., market_type=..., symbol=..., timeframe=...) instead."
            )

        return self.make_key(
            exchange=self.default_exchange,
            market_type=self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def make_key(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> SpoofingKey:
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
        return make_spoofing_key(
            exchange=exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def is_key_allowed(self, key: SpoofingKey) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_key_allowed", _analytics_args)
        except Exception:
            pass
        scope = spoofing_key_to_dict(key)

        return self.is_scope_allowed(
            exchange=scope["exchange"],
            market_type=scope["market_type"],
            symbol=scope["symbol"],
            timeframe=scope["timeframe"],
        )

    def is_scope_allowed(
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_scope_allowed", _analytics_args)
        except Exception:
            pass
        normalized_exchange = _normalize_exchange(exchange)
        normalized_market_type = _normalize_market_type(market_type or self.default_market_type)
        normalized_symbol = _normalize_symbol(symbol)
        normalized_timeframe = _normalize_timeframe(timeframe or self.default_timeframe)

        if normalized_exchange is None or normalized_symbol is None:
            return False

        if self.exchange_allowlist is not None and normalized_exchange not in self.exchange_allowlist:
            return False

        if self.market_type_allowlist is not None and normalized_market_type not in self.market_type_allowlist:
            return False

        if self.symbol_allowlist is not None and normalized_symbol not in self.symbol_allowlist:
            return False

        if self.timeframe_allowlist is not None and normalized_timeframe not in self.timeframe_allowlist:
            return False

        return True

    def is_symbol_allowed(self, symbol: str) -> bool:
        """
        Legacy-compatible helper.

        New code should use is_key_allowed() або is_scope_allowed().
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_symbol_allowed", _analytics_args)
        except Exception:
            pass
        normalized_symbol = _normalize_symbol(symbol)
        if normalized_symbol is None:
            return False

        if self.symbol_allowlist is None:
            return True

        return normalized_symbol in self.symbol_allowlist

    @property
    def production_source_topics(self) -> tuple[str, ...]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "production_source_topics", _analytics_args)
        except Exception:
            pass
        return self.analyzer.source_topic_patterns

    # ------------------------------------------------------------------
    # Primitive validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_ratio", _analytics_args)
        except Exception:
            pass
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_positive", _analytics_args)
        except Exception:
            pass
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_non_negative", _analytics_args)
        except Exception:
            pass
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_positive_int", _analytics_args)
        except Exception:
            pass
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_non_negative_int", _analytics_args)
        except Exception:
            pass
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    @staticmethod
    def _validate_topic(name: str, value: str) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_topic", _analytics_args)
        except Exception:
            pass
        if not value or not value.strip():
            raise ValueError(f"{name} must not be empty")

    @staticmethod
    def _validate_non_empty_string(name: str, value: str) -> None:
        try:
            _analytics_class_name = "SpoofingConfig"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_validate_non_empty_string", _analytics_args)
        except Exception:
            pass
        if not value or not value.strip():
            raise ValueError(f"{name} must not be empty")


__all__ = [
    "PROJECT_EXCHANGES",
    "PROJECT_MARKET_TYPES",
    "PROJECT_TIMEFRAMES",
    "WallDetectionConfig",
    "PersistenceTrackerConfig",
    "PullDetectionConfig",
    "FakeLiquidityConfig",
    "LayeringConfig",
    "FlipPressureConfig",
    "SpoofingScoreConfig",
    "CandidateTrackingConfig",
    "SpoofingAnalyzerConfig",
    "SpoofingConfig",
]