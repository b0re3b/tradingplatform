from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias


# =============================================================================
# Scope defaults
# =============================================================================

DEFAULT_EXCHANGE = "unknown"
DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "1m"

OIKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Canonical input topics
# =============================================================================

# Data/cache-layer topics. OIAnalyzer may listen to these.
DEFAULT_OPEN_INTEREST_UPDATED_TOPIC = "market.open_interest.updated"
DEFAULT_CANDLE_CLOSED_TOPIC = "market.candle.closed"
DEFAULT_CANDLES_UPDATED_TOPIC = "market.candles.updated"
DEFAULT_TRADES_UPDATED_TOPIC = "market.trades.updated"
DEFAULT_FUNDING_UPDATED_TOPIC = "market.funding.updated"

# Analytics-layer context topics.
DEFAULT_ORDERFLOW_UPDATED_TOPIC = "analytics.orderflow.updated"
DEFAULT_LIQUIDATIONS_UPDATED_TOPIC = "analytics.liquidations.updated"

# Raw market topics. OIAnalyzer must not subscribe to these in production.
DEFAULT_RAW_OPEN_INTEREST_TOPIC = "market.open_interest"
DEFAULT_RAW_CANDLE_TOPIC = "market.candle"
DEFAULT_RAW_TRADE_TOPIC = "market.trade"
DEFAULT_RAW_ORDERBOOK_TOPIC = "market.orderbook"
DEFAULT_RAW_FUNDING_TOPIC = "market.funding"

RAW_OI_MARKET_TOPICS = {
    DEFAULT_RAW_OPEN_INTEREST_TOPIC,
    DEFAULT_RAW_CANDLE_TOPIC,
    DEFAULT_RAW_TRADE_TOPIC,
    DEFAULT_RAW_ORDERBOOK_TOPIC,
    DEFAULT_RAW_FUNDING_TOPIC,
}


# =============================================================================
# Canonical output topics
# =============================================================================

DEFAULT_OI_UPDATED_TOPIC = "analytics.oi.updated"
DEFAULT_OI_REGIME_CHANGED_TOPIC = "analytics.oi.regime.changed"
DEFAULT_OI_DIVERGENCE_TOPIC = "analytics.oi.divergence"
DEFAULT_OI_ANOMALY_TOPIC = "analytics.oi.anomaly"
DEFAULT_OI_SQUEEZE_SETUP_TOPIC = "analytics.oi.squeeze_setup"
DEFAULT_OI_CAPITULATION_TOPIC = "analytics.oi.capitulation"
DEFAULT_OI_METRICS_TOPIC = "analytics.oi.metrics"


# =============================================================================
# Normalization / validation helpers
# =============================================================================

def normalize_exchange(value: object | None) -> str:
    normalized = str(value or DEFAULT_EXCHANGE).strip().lower()
    return normalized or DEFAULT_EXCHANGE


def normalize_market_type(value: object | None) -> str:
    normalized = str(value or DEFAULT_MARKET_TYPE).strip().lower()
    return normalized or DEFAULT_MARKET_TYPE


def normalize_symbol(value: object | None) -> str:
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def normalize_timeframe(value: object | None) -> str:
    normalized = str(value or DEFAULT_TIMEFRAME).strip()
    return normalized or DEFAULT_TIMEFRAME


def make_oi_key(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None,
) -> OIKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def oi_key_to_dict(key: OIKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def oi_key_to_string(key: OIKey) -> str:
    scope = oi_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def _normalize_topics(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalize_exchange_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_exchange(value) for value in values if str(value).strip()}


def _normalize_market_type_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_market_type(value) for value in values if str(value).strip()}


def _normalize_symbol_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_symbol(value) for value in values if str(value).strip()}


def _normalize_timeframe_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_timeframe(value) for value in values if str(value).strip()}


def _validate_topic(topic: str, field_name: str) -> None:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in topic:
        raise ValueError(f"{field_name} must not contain spaces")


def _validate_job_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")

    if " " in value:
        raise ValueError(f"{field_name} must not contain spaces")


def _validate_positive_float(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative_float(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


# =============================================================================
# Thresholds
# =============================================================================

@dataclass(slots=True)
class OIThresholds:
    """
    Порогові значення для класифікації режимів, дивергенцій та аномалій.

    Значення є стартовими дефолтами і мають калібруватися під:
    - біржу;
    - symbol;
    - timeframe;
    - futures market_type.
    """

    min_oi_change_pct: float = 0.25
    min_price_change_pct: float = 0.20

    volume_confirmation_ratio: float = 1.15
    aggressive_flow_confirmation: float = 0.10

    funding_extreme_positive: float = 0.01
    funding_extreme_negative: float = -0.01

    divergence_min_price_move_pct: float = 0.35
    divergence_max_oi_response_pct: float = 0.10
    divergence_min_confidence: float = 0.55

    anomaly_zscore_threshold: float = 2.5
    extreme_anomaly_zscore_threshold: float = 3.5
    overheated_zscore_threshold: float = 2.8

    capitulation_price_move_pct: float = 1.25
    capitulation_oi_drop_pct: float = 1.00
    deleveraging_oi_drop_pct: float = 1.50

    squeeze_funding_abs_threshold: float = 0.015
    squeeze_oi_build_pct: float = 0.75

    pressure_score_trend_threshold: float = 0.35
    pressure_score_exhaustion_threshold: float = 0.75

    def validate(self) -> None:
        if self.min_oi_change_pct < 0:
            raise ValueError("min_oi_change_pct must be >= 0")

        if self.min_price_change_pct < 0:
            raise ValueError("min_price_change_pct must be >= 0")

        if self.volume_confirmation_ratio <= 0:
            raise ValueError("volume_confirmation_ratio must be > 0")

        if self.aggressive_flow_confirmation < 0:
            raise ValueError("aggressive_flow_confirmation must be >= 0")

        if self.funding_extreme_positive < 0:
            raise ValueError("funding_extreme_positive must be >= 0")

        if self.funding_extreme_negative > 0:
            raise ValueError("funding_extreme_negative must be <= 0")

        if self.divergence_min_price_move_pct < 0:
            raise ValueError("divergence_min_price_move_pct must be >= 0")

        if self.divergence_max_oi_response_pct < 0:
            raise ValueError("divergence_max_oi_response_pct must be >= 0")

        if not 0 <= self.divergence_min_confidence <= 1:
            raise ValueError("divergence_min_confidence must be in [0, 1]")

        if self.anomaly_zscore_threshold <= 0:
            raise ValueError("anomaly_zscore_threshold must be > 0")

        if self.extreme_anomaly_zscore_threshold < self.anomaly_zscore_threshold:
            raise ValueError(
                "extreme_anomaly_zscore_threshold must be >= anomaly_zscore_threshold"
            )

        if self.overheated_zscore_threshold <= 0:
            raise ValueError("overheated_zscore_threshold must be > 0")

        if self.capitulation_price_move_pct < 0:
            raise ValueError("capitulation_price_move_pct must be >= 0")

        if self.capitulation_oi_drop_pct < 0:
            raise ValueError("capitulation_oi_drop_pct must be >= 0")

        if self.deleveraging_oi_drop_pct < 0:
            raise ValueError("deleveraging_oi_drop_pct must be >= 0")

        if self.squeeze_funding_abs_threshold < 0:
            raise ValueError("squeeze_funding_abs_threshold must be >= 0")

        if self.squeeze_oi_build_pct < 0:
            raise ValueError("squeeze_oi_build_pct must be >= 0")

        if self.pressure_score_trend_threshold < 0:
            raise ValueError("pressure_score_trend_threshold must be >= 0")

        if self.pressure_score_exhaustion_threshold < 0:
            raise ValueError("pressure_score_exhaustion_threshold must be >= 0")

        if self.pressure_score_exhaustion_threshold < self.pressure_score_trend_threshold:
            raise ValueError(
                "pressure_score_exhaustion_threshold must be >= "
                "pressure_score_trend_threshold"
            )


# =============================================================================
# Windows
# =============================================================================

@dataclass(slots=True)
class OIWindows:
    """
    Вікна історії для rolling statistics.
    """

    history_size: int = 300
    fast_window: int = 10
    slow_window: int = 30
    zscore_window: int = 50
    divergence_window: int = 20
    pressure_window: int = 12
    volume_window: int = 20

    def validate(self) -> None:
        if self.history_size < 20:
            raise ValueError("history_size must be >= 20")

        if self.fast_window < 2:
            raise ValueError("fast_window must be >= 2")

        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be > fast_window")

        if self.zscore_window < self.fast_window:
            raise ValueError("zscore_window must be >= fast_window")

        if self.divergence_window < 5:
            raise ValueError("divergence_window must be >= 5")

        if self.pressure_window < 3:
            raise ValueError("pressure_window must be >= 3")

        if self.volume_window < 2:
            raise ValueError("volume_window must be >= 2")

        if self.history_size < self.slow_window:
            raise ValueError("history_size must be >= slow_window")

        if self.history_size < self.zscore_window:
            raise ValueError("history_size must be >= zscore_window")

        if self.history_size < self.divergence_window:
            raise ValueError("history_size must be >= divergence_window")


# =============================================================================
# Cooldowns
# =============================================================================

@dataclass(slots=True)
class OICooldowns:
    """
    Антиспам / дедуплікація high-level OI events.
    """

    regime_change_cooldown_sec: float = 10.0
    divergence_event_cooldown_sec: float = 15.0
    anomaly_event_cooldown_sec: float = 15.0
    squeeze_event_cooldown_sec: float = 20.0
    capitulation_event_cooldown_sec: float = 20.0

    def validate(self) -> None:
        for name in (
            "regime_change_cooldown_sec",
            "divergence_event_cooldown_sec",
            "anomaly_event_cooldown_sec",
            "squeeze_event_cooldown_sec",
            "capitulation_event_cooldown_sec",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0")


# =============================================================================
# Maintenance
# =============================================================================

@dataclass(slots=True)
class OIMaintenanceConfig:
    """
    Scheduler-related налаштування для OIAnalyzer.

    Цей конфіг не запускає Scheduler напряму.
    Він лише описує, які periodic jobs має зареєструвати OIAnalyzer
    через core.scheduler.Scheduler.add_interval_job().
    """

    enable_periodic_cleanup: bool = True
    cleanup_interval_sec: float = 60.0

    enable_metrics_emit: bool = True
    metrics_interval_sec: float = 30.0

    cleanup_job_name: str = "analytics.open_interest.cleanup_stale_state"
    metrics_job_name: str = "analytics.open_interest.emit_metrics"

    cleanup_job_timeout_sec: float | None = 10.0
    metrics_job_timeout_sec: float | None = 5.0

    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_sec: float = 1.0

    def validate(self) -> None:
        _validate_positive_float(self.cleanup_interval_sec, "cleanup_interval_sec")
        _validate_positive_float(self.metrics_interval_sec, "metrics_interval_sec")

        _validate_job_name(self.cleanup_job_name, "cleanup_job_name")
        _validate_job_name(self.metrics_job_name, "metrics_job_name")

        if self.cleanup_job_timeout_sec is not None:
            _validate_positive_float(
                self.cleanup_job_timeout_sec,
                "cleanup_job_timeout_sec",
            )

        if self.metrics_job_timeout_sec is not None:
            _validate_positive_float(
                self.metrics_job_timeout_sec,
                "metrics_job_timeout_sec",
            )

        if self.scheduler_job_max_retries < 0:
            raise ValueError("scheduler_job_max_retries must be >= 0")

        _validate_non_negative_float(
            self.scheduler_job_retry_delay_sec,
            "scheduler_job_retry_delay_sec",
        )

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
        names: list[str] = []

        if self.enable_periodic_cleanup:
            names.append(self.cleanup_job_name)

        if self.enable_metrics_emit:
            names.append(self.metrics_job_name)

        return tuple(dict.fromkeys(names))


# =============================================================================
# Root config
# =============================================================================

@dataclass(slots=True)
class OIAnalyzerConfig:
    """
    Головний конфіг Open Interest analytics-модуля.

    Runtime-залежності не зберігаються тут:
    - EventBus передається в OIAnalyzer через constructor dependency injection;
    - Scheduler передається в OIAnalyzer через constructor dependency injection;
    - Logger створюється в OIAnalyzer через core.logger.get_logger().

    OIAnalyzer має слухати тільки data/cache-layer або analytics-layer topics.
    Raw exchange/market topics заборонені, якщо allow_raw_market_topics=False.

    Canonical scope:
        exchange + market_type + symbol + timeframe
    """

    enabled: bool = True

    source_name: str = "oi_analyzer"

    # ------------------------------------------------------------------
    # Scope defaults / filters
    # ------------------------------------------------------------------

    default_exchange: str = DEFAULT_EXCHANGE
    default_market_type: str = DEFAULT_MARKET_TYPE
    default_timeframe: str = DEFAULT_TIMEFRAME

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
    allowed_timeframes: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Input topics
    # ------------------------------------------------------------------

    open_interest_input_topics: tuple[str, ...] = (
        DEFAULT_OPEN_INTEREST_UPDATED_TOPIC,
    )
    candle_input_topics: tuple[str, ...] = (
        DEFAULT_CANDLE_CLOSED_TOPIC,
    )
    candles_updated_input_topics: tuple[str, ...] = (
        DEFAULT_CANDLES_UPDATED_TOPIC,
    )
    trades_input_topics: tuple[str, ...] = (
        DEFAULT_TRADES_UPDATED_TOPIC,
    )
    funding_input_topics: tuple[str, ...] = (
        DEFAULT_FUNDING_UPDATED_TOPIC,
    )
    orderflow_input_topics: tuple[str, ...] = (
        DEFAULT_ORDERFLOW_UPDATED_TOPIC,
    )
    liquidations_input_topics: tuple[str, ...] = (
        DEFAULT_LIQUIDATIONS_UPDATED_TOPIC,
    )

    allow_raw_market_topics: bool = False

    # ------------------------------------------------------------------
    # Output topics
    # ------------------------------------------------------------------

    update_topic: str = DEFAULT_OI_UPDATED_TOPIC
    regime_change_topic: str = DEFAULT_OI_REGIME_CHANGED_TOPIC
    divergence_topic: str = DEFAULT_OI_DIVERGENCE_TOPIC
    anomaly_topic: str = DEFAULT_OI_ANOMALY_TOPIC
    squeeze_setup_topic: str = DEFAULT_OI_SQUEEZE_SETUP_TOPIC
    capitulation_topic: str = DEFAULT_OI_CAPITULATION_TOPIC
    metrics_topic: str = DEFAULT_OI_METRICS_TOPIC

    # ------------------------------------------------------------------
    # Emit flags
    # ------------------------------------------------------------------

    emit_updates: bool = True
    emit_regime_changes: bool = True
    emit_divergences: bool = True
    emit_anomalies: bool = True
    emit_squeeze_events: bool = True
    emit_capitulation_events: bool = True
    emit_metrics: bool = True

    # ------------------------------------------------------------------
    # Analysis requirements
    # ------------------------------------------------------------------

    require_price_context: bool = False
    require_volume_confirmation: bool = True
    require_funding_for_squeeze: bool = False

    normalize_symbol: bool = True
    store_full_analysis: bool = True

    stale_context_after_sec: float = 30.0
    stale_state_cleanup_after_sec: float = 3600.0

    thresholds: OIThresholds = field(default_factory=OIThresholds)
    windows: OIWindows = field(default_factory=OIWindows)
    cooldowns: OICooldowns = field(default_factory=OICooldowns)
    maintenance: OIMaintenanceConfig = field(default_factory=OIMaintenanceConfig)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_exchange = normalize_exchange(self.default_exchange)
        self.default_market_type = normalize_market_type(self.default_market_type)
        self.default_timeframe = normalize_timeframe(self.default_timeframe)

        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)

        self.open_interest_input_topics = _normalize_topics(self.open_interest_input_topics)
        self.candle_input_topics = _normalize_topics(self.candle_input_topics)
        self.candles_updated_input_topics = _normalize_topics(
            self.candles_updated_input_topics
        )
        self.trades_input_topics = _normalize_topics(self.trades_input_topics)
        self.funding_input_topics = _normalize_topics(self.funding_input_topics)
        self.orderflow_input_topics = _normalize_topics(self.orderflow_input_topics)
        self.liquidations_input_topics = _normalize_topics(
            self.liquidations_input_topics
        )

        self.update_topic = self.update_topic.strip()
        self.regime_change_topic = self.regime_change_topic.strip()
        self.divergence_topic = self.divergence_topic.strip()
        self.anomaly_topic = self.anomaly_topic.strip()
        self.squeeze_setup_topic = self.squeeze_setup_topic.strip()
        self.capitulation_topic = self.capitulation_topic.strip()
        self.metrics_topic = self.metrics_topic.strip()

        self.metadata = dict(self.metadata or {})

        self.validate()

    # ------------------------------------------------------------------
    # Topic groups
    # ------------------------------------------------------------------

    @property
    def open_interest_topics(self) -> tuple[str, ...]:
        return self.open_interest_input_topics

    @property
    def candle_topics(self) -> tuple[str, ...]:
        return self.candle_input_topics

    @property
    def candles_updated_topics(self) -> tuple[str, ...]:
        return self.candles_updated_input_topics

    @property
    def trades_topics(self) -> tuple[str, ...]:
        return self.trades_input_topics

    @property
    def funding_topics(self) -> tuple[str, ...]:
        return self.funding_input_topics

    @property
    def orderflow_topics(self) -> tuple[str, ...]:
        return self.orderflow_input_topics

    @property
    def liquidations_topics(self) -> tuple[str, ...]:
        return self.liquidations_input_topics

    @property
    def production_input_topics(self) -> tuple[str, ...]:
        topics: list[str] = []

        topics.extend(self.open_interest_input_topics)
        topics.extend(self.candle_input_topics)
        topics.extend(self.candles_updated_input_topics)
        topics.extend(self.trades_input_topics)
        topics.extend(self.funding_input_topics)
        topics.extend(self.orderflow_input_topics)
        topics.extend(self.liquidations_input_topics)

        return tuple(dict.fromkeys(topics))

    @property
    def output_topics(self) -> tuple[str, ...]:
        topics: list[str] = []

        if self.emit_updates:
            topics.append(self.update_topic)

        if self.emit_regime_changes:
            topics.append(self.regime_change_topic)

        if self.emit_divergences:
            topics.append(self.divergence_topic)

        if self.emit_anomalies:
            topics.append(self.anomaly_topic)

        if self.emit_squeeze_events:
            topics.append(self.squeeze_setup_topic)

        if self.emit_capitulation_events:
            topics.append(self.capitulation_topic)

        if self.emit_metrics and self.maintenance.enable_metrics_emit:
            topics.append(self.metrics_topic)

        return tuple(dict.fromkeys(topics))

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
        return self.maintenance.scheduler_job_names

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def make_key(
        self,
        *,
        symbol: str,
        timeframe: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
    ) -> OIKey:
        return make_oi_key(
            exchange=exchange or self.default_exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_key(self, key: OIKey) -> bool:
        scope = oi_key_to_dict(key)

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
        symbol: str,
        timeframe: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
    ) -> bool:
        return self.should_process_key(
            self.make_key(
                exchange=exchange,
                market_type=market_type,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    def scoped_mapping_key(self, key: OIKey) -> str:
        return oi_key_to_string(key)

    # ------------------------------------------------------------------
    # Topic guards
    # ------------------------------------------------------------------

    def is_raw_market_topic(self, topic: str) -> bool:
        return topic in RAW_OI_MARKET_TOPICS

    def assert_input_topic_allowed(self, topic: str) -> None:
        _validate_topic(topic, "open interest input topic")

        if self.is_raw_market_topic(topic) and not self.allow_raw_market_topics:
            raise ValueError(
                f"Raw market topic {topic!r} is not allowed for OIAnalyzer. "
                "Use data/cache-layer topics such as market.open_interest.updated, "
                "market.candle.closed, market.candles.updated, market.trades.updated "
                "or analytics-layer context topics."
            )

    def assert_production_topics_allowed(self) -> None:
        for topic in self.production_input_topics:
            self.assert_input_topic_allowed(topic)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []

        if not self.source_name.strip():
            errors.append("source_name must not be empty")

        if not self.default_exchange:
            errors.append("default_exchange must not be empty")

        if not self.default_market_type:
            errors.append("default_market_type must not be empty")

        if not self.default_timeframe:
            errors.append("default_timeframe must not be empty")

        if not self.allowed_market_types:
            errors.append("allowed_market_types must not be empty")

        if not self.open_interest_input_topics:
            errors.append("open_interest_input_topics must not be empty")

        if not self.candle_input_topics and self.require_price_context:
            errors.append(
                "candle_input_topics must not be empty when require_price_context=True"
            )

        if self.stale_context_after_sec <= 0:
            errors.append("stale_context_after_sec must be > 0")

        if self.stale_state_cleanup_after_sec <= 0:
            errors.append("stale_state_cleanup_after_sec must be > 0")

        if self.stale_state_cleanup_after_sec < self.stale_context_after_sec:
            errors.append(
                "stale_state_cleanup_after_sec must be >= stale_context_after_sec"
            )

        try:
            for topic in self.production_input_topics:
                self.assert_input_topic_allowed(topic)

            for topic in self.output_topics:
                _validate_topic(topic, "open interest output topic")

        except ValueError as exc:
            errors.append(str(exc))

        try:
            self.thresholds.validate()
        except ValueError as exc:
            errors.append(str(exc))

        try:
            self.windows.validate()
        except ValueError as exc:
            errors.append(str(exc))

        try:
            self.cooldowns.validate()
        except ValueError as exc:
            errors.append(str(exc))

        try:
            self.maintenance.validate()
        except ValueError as exc:
            errors.append(str(exc))

        if errors:
            raise ValueError("Invalid OIAnalyzerConfig: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # Factories / diagnostics
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OIAnalyzerConfig":
        """
        Зручний factory для інтеграції з AppConfig / YAML / JSON / env-based config.

        Підтримує вкладені:
            thresholds
            windows
            cooldowns
            maintenance
        """
        raw = dict(data or {})

        thresholds = OIThresholds(**dict(raw.pop("thresholds", {}) or {}))
        windows = OIWindows(**dict(raw.pop("windows", {}) or {}))
        cooldowns = OICooldowns(**dict(raw.pop("cooldowns", {}) or {}))
        maintenance = OIMaintenanceConfig(**dict(raw.pop("maintenance", {}) or {}))

        return cls(
            thresholds=thresholds,
            windows=windows,
            cooldowns=cooldowns,
            maintenance=maintenance,
            **raw,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_name": self.source_name,
            "scope": "exchange:market_type:symbol:timeframe",
            "default_exchange": self.default_exchange,
            "default_market_type": self.default_market_type,
            "default_timeframe": self.default_timeframe,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "allowed_market_types": sorted(self.allowed_market_types),
            "allowed_symbols": sorted(self.allowed_symbols),
            "allowed_timeframes": sorted(self.allowed_timeframes),
            "production_input_topics": list(self.production_input_topics),
            "open_interest_input_topics": list(self.open_interest_input_topics),
            "candle_input_topics": list(self.candle_input_topics),
            "candles_updated_input_topics": list(self.candles_updated_input_topics),
            "trades_input_topics": list(self.trades_input_topics),
            "funding_input_topics": list(self.funding_input_topics),
            "orderflow_input_topics": list(self.orderflow_input_topics),
            "liquidations_input_topics": list(self.liquidations_input_topics),
            "allow_raw_market_topics": self.allow_raw_market_topics,
            "output_topics": list(self.output_topics),
            "update_topic": self.update_topic,
            "regime_change_topic": self.regime_change_topic,
            "divergence_topic": self.divergence_topic,
            "anomaly_topic": self.anomaly_topic,
            "squeeze_setup_topic": self.squeeze_setup_topic,
            "capitulation_topic": self.capitulation_topic,
            "metrics_topic": self.metrics_topic,
            "emit_updates": self.emit_updates,
            "emit_regime_changes": self.emit_regime_changes,
            "emit_divergences": self.emit_divergences,
            "emit_anomalies": self.emit_anomalies,
            "emit_squeeze_events": self.emit_squeeze_events,
            "emit_capitulation_events": self.emit_capitulation_events,
            "emit_metrics": self.emit_metrics,
            "require_price_context": self.require_price_context,
            "require_volume_confirmation": self.require_volume_confirmation,
            "require_funding_for_squeeze": self.require_funding_for_squeeze,
            "normalize_symbol": self.normalize_symbol,
            "store_full_analysis": self.store_full_analysis,
            "stale_context_after_sec": self.stale_context_after_sec,
            "stale_state_cleanup_after_sec": self.stale_state_cleanup_after_sec,
            "thresholds": asdict(self.thresholds),
            "windows": asdict(self.windows),
            "cooldowns": asdict(self.cooldowns),
            "maintenance": asdict(self.maintenance),
            "scheduler_job_names": list(self.scheduler_job_names),
            "metadata": dict(self.metadata),
        }


__all__ = [
    # scope
    "DEFAULT_EXCHANGE",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "OIKey",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_symbol",
    "normalize_timeframe",
    "make_oi_key",
    "oi_key_to_dict",
    "oi_key_to_string",

    # input topics
    "DEFAULT_OPEN_INTEREST_UPDATED_TOPIC",
    "DEFAULT_CANDLE_CLOSED_TOPIC",
    "DEFAULT_CANDLES_UPDATED_TOPIC",
    "DEFAULT_TRADES_UPDATED_TOPIC",
    "DEFAULT_FUNDING_UPDATED_TOPIC",
    "DEFAULT_ORDERFLOW_UPDATED_TOPIC",
    "DEFAULT_LIQUIDATIONS_UPDATED_TOPIC",
    "DEFAULT_RAW_OPEN_INTEREST_TOPIC",
    "DEFAULT_RAW_CANDLE_TOPIC",
    "DEFAULT_RAW_TRADE_TOPIC",
    "DEFAULT_RAW_ORDERBOOK_TOPIC",
    "DEFAULT_RAW_FUNDING_TOPIC",
    "RAW_OI_MARKET_TOPICS",

    # output topics
    "DEFAULT_OI_UPDATED_TOPIC",
    "DEFAULT_OI_REGIME_CHANGED_TOPIC",
    "DEFAULT_OI_DIVERGENCE_TOPIC",
    "DEFAULT_OI_ANOMALY_TOPIC",
    "DEFAULT_OI_SQUEEZE_SETUP_TOPIC",
    "DEFAULT_OI_CAPITULATION_TOPIC",
    "DEFAULT_OI_METRICS_TOPIC",

    # configs
    "OIThresholds",
    "OIWindows",
    "OICooldowns",
    "OIMaintenanceConfig",
    "OIAnalyzerConfig",
]