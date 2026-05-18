from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


DEFAULT_EXCHANGE = "unknown"
DEFAULT_MARKET_TYPE = "perpetual"
DEFAULT_TIMEFRAME = "1m"

LiquidityKey: TypeAlias = tuple[str, str, str, str]
# exchange, market_type, symbol, timeframe


# =============================================================================
# Canonical topics
# =============================================================================

# Production input topics.
# Liquidity analytics має слухати data/cache-layer events, не raw exchange events.
DEFAULT_CANDLE_CLOSED_TOPIC = "market.candle.closed"
DEFAULT_CANDLES_UPDATED_TOPIC = "market.candles.updated"
DEFAULT_ORDERBOOK_UPDATED_TOPIC = "market.orderbook.updated"

# Optional non-canonical price topic.
# За замовчуванням вимкнено, бо canonical price source для liquidity —
# candle close / candles cache.
DEFAULT_PRICE_UPDATED_TOPIC = "market.price.updated"

# Raw topics, які LiquidityService не має слухати напряму в production.
DEFAULT_RAW_CANDLE_TOPIC = "market.candle"
DEFAULT_RAW_ORDERBOOK_TOPIC = "market.orderbook"
DEFAULT_RAW_TRADE_TOPIC = "market.trade"

RAW_LIQUIDITY_MARKET_TOPICS = {
    DEFAULT_RAW_CANDLE_TOPIC,
    DEFAULT_RAW_ORDERBOOK_TOPIC,
    DEFAULT_RAW_TRADE_TOPIC,
}

# Analytics output topics.
DEFAULT_LIQUIDITY_MAP_UPDATED_TOPIC = "analytics.liquidity.map.updated"
DEFAULT_LIQUIDITY_LEVEL_DETECTED_TOPIC = "analytics.liquidity.level.detected"
DEFAULT_LIQUIDITY_LEVEL_SWEPT_TOPIC = "analytics.liquidity.level.swept"
DEFAULT_LIQUIDITY_STOP_CLUSTER_DETECTED_TOPIC = "analytics.liquidity.stop_cluster.detected"
DEFAULT_LIQUIDITY_SIGNAL_UPDATED_TOPIC = "analytics.liquidity.signal.updated"
DEFAULT_LIQUIDITY_STATE_METRICS_TOPIC = "analytics.liquidity.state.metrics"
DEFAULT_LIQUIDITY_HEALTHCHECK_TOPIC = "analytics.liquidity.healthcheck"


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


def make_liquidity_key(
    *,
    exchange: object | None,
    market_type: object | None,
    symbol: object,
    timeframe: object | None,
) -> LiquidityKey:
    return (
        normalize_exchange(exchange),
        normalize_market_type(market_type),
        normalize_symbol(symbol),
        normalize_timeframe(timeframe),
    )


def liquidity_key_to_dict(key: LiquidityKey) -> dict[str, str]:
    exchange, market_type, symbol, timeframe = key
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbol": symbol,
        "timeframe": timeframe,
    }


def liquidity_key_to_string(key: LiquidityKey) -> str:
    scope = liquidity_key_to_dict(key)
    return (
        f"{scope['exchange']}:"
        f"{scope['market_type']}:"
        f"{scope['symbol']}:"
        f"{scope['timeframe']}"
    )


def _normalize_string_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def _normalize_exchange_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_exchange(value) for value in values if str(value).strip()}


def _normalize_market_type_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_market_type(value) for value in values if str(value).strip()}


def _normalize_symbol_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_symbol(value) for value in values if str(value).strip()}


def _normalize_timeframe_set(values: set[str] | tuple[str, ...] | list[str]) -> set[str]:
    return {normalize_timeframe(value) for value in values if str(value).strip()}


def _normalize_topics(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return tuple(str(topic).strip() for topic in values if str(topic).strip())


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
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_open_ratio(value: float, field_name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(slots=True)
class LiquidityConfig:
    """
    Конфігурація analytics/liquidity модуля.

    Використовується всіма liquidity-компонентами:
    - EqualHighsLowsDetector
    - StopClustersDetector
    - LiquidityScorer
    - LiquidityMap
    - LiquidityService

    Архітектурно:
    - не імпортує EventBus / Scheduler / logger;
    - є чистою dataclass-конфігурацією;
    - runtime-залежності передаються через constructor dependency injection;
    - production input topics мають бути data/cache-layer topics;
    - raw market topics заборонені для LiquidityService, якщо allow_raw_market_topics=False.

    Scope:
        exchange + market_type + symbol + timeframe
    """

    # ------------------------------------------------------------------
    # Module switch
    # ------------------------------------------------------------------

    enabled: bool = True

    # ------------------------------------------------------------------
    # Scoped defaults / filters
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
    # Production input topics
    # ------------------------------------------------------------------

    candle_input_topics: tuple[str, ...] = (DEFAULT_CANDLE_CLOSED_TOPIC,)
    candles_updated_input_topics: tuple[str, ...] = ()
    orderbook_input_topics: tuple[str, ...] = (DEFAULT_ORDERBOOK_UPDATED_TOPIC,)

    # Optional. Disabled by default, because canonical price source is candle data.
    price_input_topics: tuple[str, ...] = ()
    allow_price_input_topics: bool = False

    # Raw topic guard.
    allow_raw_market_topics: bool = False

    # ------------------------------------------------------------------
    # Output topics
    # ------------------------------------------------------------------

    map_updated_topic: str = DEFAULT_LIQUIDITY_MAP_UPDATED_TOPIC
    level_detected_topic: str = DEFAULT_LIQUIDITY_LEVEL_DETECTED_TOPIC
    level_swept_topic: str = DEFAULT_LIQUIDITY_LEVEL_SWEPT_TOPIC
    stop_cluster_detected_topic: str = DEFAULT_LIQUIDITY_STOP_CLUSTER_DETECTED_TOPIC
    signal_updated_topic: str = DEFAULT_LIQUIDITY_SIGNAL_UPDATED_TOPIC
    state_metrics_topic: str = DEFAULT_LIQUIDITY_STATE_METRICS_TOPIC
    healthcheck_topic: str = DEFAULT_LIQUIDITY_HEALTHCHECK_TOPIC

    # ------------------------------------------------------------------
    # Pivot / swing detection
    # ------------------------------------------------------------------

    pivot_lookback: int = 3
    pivot_lookforward: int = 3
    min_swing_distance_pct: float = 0.0020

    # ------------------------------------------------------------------
    # Equal highs / lows
    # ------------------------------------------------------------------

    equal_level_tolerance_pct: float = 0.0008
    min_equal_touches: int = 2
    max_equal_cluster_width_pct: float = 0.0012

    # ------------------------------------------------------------------
    # Stop clusters
    # ------------------------------------------------------------------

    stop_cluster_padding_pct: float = 0.0015
    cluster_merge_distance_pct: float = 0.0007

    # ------------------------------------------------------------------
    # Filtering / retention
    # ------------------------------------------------------------------

    max_active_levels: int = 200
    max_active_clusters: int = 100
    level_expiry_bars: int = 300
    min_confidence: float = 0.35

    # ------------------------------------------------------------------
    # ATR / adaptive tolerance
    # ------------------------------------------------------------------

    use_atr_tolerance: bool = True
    atr_period: int = 14
    atr_tolerance_multiplier: float = 0.15
    min_atr_tolerance_pct: float = 0.0003
    max_atr_tolerance_pct: float = 0.0030

    # ------------------------------------------------------------------
    # Scoring behavior
    # ------------------------------------------------------------------

    use_volume_in_scoring: bool = True
    use_reaction_strength_in_scoring: bool = True
    use_orderbook_in_stop_clusters: bool = True
    use_time_decay: bool = True
    use_partial_sweep_penalty: bool = True

    # ------------------------------------------------------------------
    # Service context
    # ------------------------------------------------------------------

    max_candles_per_context: int = 500
    min_candles_for_snapshot: int = 30
    max_contexts: int = 1000

    snapshot_rebuild_min_interval_seconds: float = 1.0
    rebuild_on_orderbook_updates: bool = True
    rebuild_on_price_updates: bool = False

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    publish_events: bool = True

    emit_map_updates: bool = True
    emit_level_events: bool = True
    emit_cluster_events: bool = True
    emit_sweep_events: bool = True
    emit_signal_events: bool = True
    emit_state_metrics: bool = True

    # ------------------------------------------------------------------
    # Scheduler / maintenance
    # ------------------------------------------------------------------

    cleanup_enabled: bool = True
    cleanup_interval_seconds: float = 60.0
    state_metrics_interval_seconds: float = 30.0
    healthcheck_interval_seconds: float = 30.0

    scheduler_job_timeout_seconds: float = 5.0
    scheduler_job_max_retries: int = 1
    scheduler_job_retry_delay_seconds: float = 1.0

    cleanup_job_name: str = "analytics.liquidity.cleanup"
    state_metrics_job_name: str = "analytics.liquidity.emit_state_metrics"
    healthcheck_job_name: str = "analytics.liquidity.healthcheck"

    # ------------------------------------------------------------------
    # Incremental mode
    # ------------------------------------------------------------------

    incremental_mode: bool = True

    # ------------------------------------------------------------------
    # Extensibility
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_exchange = normalize_exchange(self.default_exchange)
        self.default_market_type = normalize_market_type(self.default_market_type)
        self.default_timeframe = normalize_timeframe(self.default_timeframe)

        self.allowed_exchanges = _normalize_exchange_set(self.allowed_exchanges)
        self.allowed_market_types = _normalize_market_type_set(self.allowed_market_types)
        self.allowed_symbols = _normalize_symbol_set(self.allowed_symbols)
        self.allowed_timeframes = _normalize_timeframe_set(self.allowed_timeframes)

        self.candle_input_topics = _normalize_topics(self.candle_input_topics)
        self.candles_updated_input_topics = _normalize_topics(
            self.candles_updated_input_topics
        )
        self.orderbook_input_topics = _normalize_topics(self.orderbook_input_topics)
        self.price_input_topics = _normalize_topics(self.price_input_topics)

        self.map_updated_topic = self.map_updated_topic.strip()
        self.level_detected_topic = self.level_detected_topic.strip()
        self.level_swept_topic = self.level_swept_topic.strip()
        self.stop_cluster_detected_topic = self.stop_cluster_detected_topic.strip()
        self.signal_updated_topic = self.signal_updated_topic.strip()
        self.state_metrics_topic = self.state_metrics_topic.strip()
        self.healthcheck_topic = self.healthcheck_topic.strip()

        self.cleanup_job_name = self.cleanup_job_name.strip()
        self.state_metrics_job_name = self.state_metrics_job_name.strip()
        self.healthcheck_job_name = self.healthcheck_job_name.strip()

        self.metadata = dict(self.metadata or {})

        self.validate()

    # ------------------------------------------------------------------
    # Topic groups
    # ------------------------------------------------------------------

    @property
    def production_input_topics(self) -> tuple[str, ...]:
        topics: list[str] = []

        topics.extend(self.candle_input_topics)
        topics.extend(self.candles_updated_input_topics)
        topics.extend(self.orderbook_input_topics)

        if self.allow_price_input_topics:
            topics.extend(self.price_input_topics)

        # Preserve order while removing duplicates.
        return tuple(dict.fromkeys(topics))

    @property
    def candle_topics(self) -> tuple[str, ...]:
        return self.candle_input_topics

    @property
    def candles_updated_topics(self) -> tuple[str, ...]:
        return self.candles_updated_input_topics

    @property
    def orderbook_topics(self) -> tuple[str, ...]:
        return self.orderbook_input_topics

    @property
    def price_topics(self) -> tuple[str, ...]:
        return self.price_input_topics if self.allow_price_input_topics else ()

    @property
    def output_topics(self) -> tuple[str, ...]:
        topics = [
            self.map_updated_topic,
            self.healthcheck_topic,
        ]

        if self.emit_level_events:
            topics.extend(
                [
                    self.level_detected_topic,
                    self.level_swept_topic,
                ]
            )

        if self.emit_cluster_events:
            topics.append(self.stop_cluster_detected_topic)

        if self.emit_signal_events:
            topics.append(self.signal_updated_topic)

        if self.emit_state_metrics:
            topics.append(self.state_metrics_topic)

        return tuple(dict.fromkeys(topics))

    @property
    def scheduler_job_names(self) -> tuple[str, ...]:
        names: list[str] = []

        if self.cleanup_enabled:
            names.append(self.cleanup_job_name)

        if self.emit_state_metrics:
            names.append(self.state_metrics_job_name)

        names.append(self.healthcheck_job_name)

        return tuple(dict.fromkeys(names))

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
    ) -> LiquidityKey:
        return make_liquidity_key(
            exchange=exchange or self.default_exchange,
            market_type=market_type or self.default_market_type,
            symbol=symbol,
            timeframe=timeframe or self.default_timeframe,
        )

    def should_process_key(self, key: LiquidityKey) -> bool:
        scope = liquidity_key_to_dict(key)

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

    def scoped_mapping_key(self, key: LiquidityKey) -> str:
        return liquidity_key_to_string(key)

    # ------------------------------------------------------------------
    # Topic guards
    # ------------------------------------------------------------------

    def is_raw_market_topic(self, topic: str) -> bool:
        return topic in RAW_LIQUIDITY_MARKET_TOPICS

    def is_price_topic(self, topic: str) -> bool:
        return topic in {DEFAULT_PRICE_UPDATED_TOPIC, *self.price_input_topics}

    def assert_input_topic_allowed(self, topic: str) -> None:
        _validate_topic(topic, "liquidity input topic")

        if self.is_raw_market_topic(topic) and not self.allow_raw_market_topics:
            raise ValueError(
                f"Raw market topic {topic!r} is not allowed for LiquidityService. "
                "Use data/cache-layer topics such as market.candle.closed, "
                "market.candles.updated or market.orderbook.updated."
            )

        if self.is_price_topic(topic) and not self.allow_price_input_topics:
            raise ValueError(
                f"Price topic {topic!r} is disabled. "
                "Use candle close / candles cache as canonical price source, "
                "or set allow_price_input_topics=True explicitly."
            )

    def assert_production_topics_allowed(self) -> None:
        for topic in self.production_input_topics:
            self.assert_input_topic_allowed(topic)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []

        # ------------------------------------------------------------------
        # Scope
        # ------------------------------------------------------------------

        if not self.default_exchange:
            errors.append("default_exchange must not be empty")

        if not self.default_market_type:
            errors.append("default_market_type must not be empty")

        if not self.default_timeframe:
            errors.append("default_timeframe must not be empty")

        if not self.allowed_market_types:
            errors.append("allowed_market_types must not be empty")

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------

        try:
            if not self.candle_input_topics and not self.candles_updated_input_topics:
                errors.append(
                    "at least one candle input topic must be configured "
                    "(candle_input_topics or candles_updated_input_topics)"
                )

            if not self.orderbook_input_topics and self.rebuild_on_orderbook_updates:
                errors.append(
                    "orderbook_input_topics must not be empty when "
                    "rebuild_on_orderbook_updates=True"
                )

            for topic in self.production_input_topics:
                self.assert_input_topic_allowed(topic)

            for topic in self.output_topics:
                _validate_topic(topic, "liquidity output topic")

            for job_name in self.scheduler_job_names:
                _validate_job_name(job_name, "liquidity scheduler job name")

        except ValueError as exc:
            errors.append(str(exc))

        # ------------------------------------------------------------------
        # Pivot / swing detection
        # ------------------------------------------------------------------

        if self.pivot_lookback < 1:
            errors.append("pivot_lookback must be >= 1")

        if self.pivot_lookforward < 1:
            errors.append("pivot_lookforward must be >= 1")

        if not 0 < self.min_swing_distance_pct < 1:
            errors.append("min_swing_distance_pct must be between 0 and 1")

        # ------------------------------------------------------------------
        # Equal highs / lows
        # ------------------------------------------------------------------

        if not 0 < self.equal_level_tolerance_pct < 1:
            errors.append("equal_level_tolerance_pct must be between 0 and 1")

        if self.min_equal_touches < 2:
            errors.append("min_equal_touches must be >= 2")

        if not 0 < self.max_equal_cluster_width_pct < 1:
            errors.append("max_equal_cluster_width_pct must be between 0 and 1")

        # ------------------------------------------------------------------
        # Stop clusters
        # ------------------------------------------------------------------

        if not 0 < self.stop_cluster_padding_pct < 1:
            errors.append("stop_cluster_padding_pct must be between 0 and 1")

        if not 0 <= self.cluster_merge_distance_pct < 1:
            errors.append("cluster_merge_distance_pct must be between 0 and 1")

        # ------------------------------------------------------------------
        # Filtering / retention
        # ------------------------------------------------------------------

        if self.max_active_levels < 1:
            errors.append("max_active_levels must be >= 1")

        if self.max_active_clusters < 1:
            errors.append("max_active_clusters must be >= 1")

        if self.level_expiry_bars < 1:
            errors.append("level_expiry_bars must be >= 1")

        if not 0 <= self.min_confidence <= 1:
            errors.append("min_confidence must be between 0 and 1")

        # ------------------------------------------------------------------
        # ATR / adaptive tolerance
        # ------------------------------------------------------------------

        if self.atr_period < 1:
            errors.append("atr_period must be >= 1")

        if self.atr_tolerance_multiplier < 0:
            errors.append("atr_tolerance_multiplier must be >= 0")

        if not 0 <= self.min_atr_tolerance_pct <= 1:
            errors.append("min_atr_tolerance_pct must be between 0 and 1")

        if not 0 <= self.max_atr_tolerance_pct <= 1:
            errors.append("max_atr_tolerance_pct must be between 0 and 1")

        if self.min_atr_tolerance_pct > self.max_atr_tolerance_pct:
            errors.append("min_atr_tolerance_pct must be <= max_atr_tolerance_pct")

        # ------------------------------------------------------------------
        # Service context
        # ------------------------------------------------------------------

        if self.max_candles_per_context < 1:
            errors.append("max_candles_per_context must be >= 1")

        if self.min_candles_for_snapshot < 1:
            errors.append("min_candles_for_snapshot must be >= 1")

        if self.min_candles_for_snapshot > self.max_candles_per_context:
            errors.append("min_candles_for_snapshot must be <= max_candles_per_context")

        if self.max_contexts < 1:
            errors.append("max_contexts must be >= 1")

        if self.snapshot_rebuild_min_interval_seconds < 0:
            errors.append("snapshot_rebuild_min_interval_seconds must be >= 0")

        # ------------------------------------------------------------------
        # Scheduler / maintenance
        # ------------------------------------------------------------------

        if self.cleanup_interval_seconds <= 0:
            errors.append("cleanup_interval_seconds must be > 0")

        if self.state_metrics_interval_seconds <= 0:
            errors.append("state_metrics_interval_seconds must be > 0")

        if self.healthcheck_interval_seconds <= 0:
            errors.append("healthcheck_interval_seconds must be > 0")

        if self.scheduler_job_timeout_seconds <= 0:
            errors.append("scheduler_job_timeout_seconds must be > 0")

        if self.scheduler_job_max_retries < 0:
            errors.append("scheduler_job_max_retries must be >= 0")

        if self.scheduler_job_retry_delay_seconds < 0:
            errors.append("scheduler_job_retry_delay_seconds must be >= 0")

        if errors:
            raise ValueError("Invalid LiquidityConfig: " + "; ".join(errors))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scope": "exchange:market_type:symbol:timeframe",
            "default_exchange": self.default_exchange,
            "default_market_type": self.default_market_type,
            "default_timeframe": self.default_timeframe,
            "allowed_exchanges": sorted(self.allowed_exchanges),
            "allowed_market_types": sorted(self.allowed_market_types),
            "allowed_symbols": sorted(self.allowed_symbols),
            "allowed_timeframes": sorted(self.allowed_timeframes),
            "production_input_topics": list(self.production_input_topics),
            "candle_input_topics": list(self.candle_input_topics),
            "candles_updated_input_topics": list(self.candles_updated_input_topics),
            "orderbook_input_topics": list(self.orderbook_input_topics),
            "price_input_topics": list(self.price_topics),
            "allow_price_input_topics": self.allow_price_input_topics,
            "allow_raw_market_topics": self.allow_raw_market_topics,
            "output_topics": list(self.output_topics),
            "map_updated_topic": self.map_updated_topic,
            "level_detected_topic": self.level_detected_topic,
            "level_swept_topic": self.level_swept_topic,
            "stop_cluster_detected_topic": self.stop_cluster_detected_topic,
            "signal_updated_topic": self.signal_updated_topic,
            "state_metrics_topic": self.state_metrics_topic,
            "healthcheck_topic": self.healthcheck_topic,
            "pivot_lookback": self.pivot_lookback,
            "pivot_lookforward": self.pivot_lookforward,
            "min_swing_distance_pct": self.min_swing_distance_pct,
            "equal_level_tolerance_pct": self.equal_level_tolerance_pct,
            "min_equal_touches": self.min_equal_touches,
            "max_equal_cluster_width_pct": self.max_equal_cluster_width_pct,
            "stop_cluster_padding_pct": self.stop_cluster_padding_pct,
            "cluster_merge_distance_pct": self.cluster_merge_distance_pct,
            "max_active_levels": self.max_active_levels,
            "max_active_clusters": self.max_active_clusters,
            "level_expiry_bars": self.level_expiry_bars,
            "min_confidence": self.min_confidence,
            "use_atr_tolerance": self.use_atr_tolerance,
            "atr_period": self.atr_period,
            "atr_tolerance_multiplier": self.atr_tolerance_multiplier,
            "min_atr_tolerance_pct": self.min_atr_tolerance_pct,
            "max_atr_tolerance_pct": self.max_atr_tolerance_pct,
            "use_volume_in_scoring": self.use_volume_in_scoring,
            "use_reaction_strength_in_scoring": self.use_reaction_strength_in_scoring,
            "use_orderbook_in_stop_clusters": self.use_orderbook_in_stop_clusters,
            "use_time_decay": self.use_time_decay,
            "use_partial_sweep_penalty": self.use_partial_sweep_penalty,
            "max_candles_per_context": self.max_candles_per_context,
            "min_candles_for_snapshot": self.min_candles_for_snapshot,
            "max_contexts": self.max_contexts,
            "snapshot_rebuild_min_interval_seconds": self.snapshot_rebuild_min_interval_seconds,
            "rebuild_on_orderbook_updates": self.rebuild_on_orderbook_updates,
            "rebuild_on_price_updates": self.rebuild_on_price_updates,
            "publish_events": self.publish_events,
            "emit_map_updates": self.emit_map_updates,
            "emit_level_events": self.emit_level_events,
            "emit_cluster_events": self.emit_cluster_events,
            "emit_sweep_events": self.emit_sweep_events,
            "emit_signal_events": self.emit_signal_events,
            "emit_state_metrics": self.emit_state_metrics,
            "cleanup_enabled": self.cleanup_enabled,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "state_metrics_interval_seconds": self.state_metrics_interval_seconds,
            "healthcheck_interval_seconds": self.healthcheck_interval_seconds,
            "scheduler_job_timeout_seconds": self.scheduler_job_timeout_seconds,
            "scheduler_job_max_retries": self.scheduler_job_max_retries,
            "scheduler_job_retry_delay_seconds": self.scheduler_job_retry_delay_seconds,
            "cleanup_job_name": self.cleanup_job_name,
            "state_metrics_job_name": self.state_metrics_job_name,
            "healthcheck_job_name": self.healthcheck_job_name,
            "scheduler_job_names": list(self.scheduler_job_names),
            "incremental_mode": self.incremental_mode,
            "metadata": dict(self.metadata),
        }


__all__ = [
    # defaults / key
    "DEFAULT_EXCHANGE",
    "DEFAULT_MARKET_TYPE",
    "DEFAULT_TIMEFRAME",
    "LiquidityKey",
    "normalize_exchange",
    "normalize_market_type",
    "normalize_symbol",
    "normalize_timeframe",
    "make_liquidity_key",
    "liquidity_key_to_dict",
    "liquidity_key_to_string",

    # input topics
    "DEFAULT_CANDLE_CLOSED_TOPIC",
    "DEFAULT_CANDLES_UPDATED_TOPIC",
    "DEFAULT_ORDERBOOK_UPDATED_TOPIC",
    "DEFAULT_PRICE_UPDATED_TOPIC",
    "DEFAULT_RAW_CANDLE_TOPIC",
    "DEFAULT_RAW_ORDERBOOK_TOPIC",
    "DEFAULT_RAW_TRADE_TOPIC",
    "RAW_LIQUIDITY_MARKET_TOPICS",

    # output topics
    "DEFAULT_LIQUIDITY_MAP_UPDATED_TOPIC",
    "DEFAULT_LIQUIDITY_LEVEL_DETECTED_TOPIC",
    "DEFAULT_LIQUIDITY_LEVEL_SWEPT_TOPIC",
    "DEFAULT_LIQUIDITY_STOP_CLUSTER_DETECTED_TOPIC",
    "DEFAULT_LIQUIDITY_SIGNAL_UPDATED_TOPIC",
    "DEFAULT_LIQUIDITY_STATE_METRICS_TOPIC",
    "DEFAULT_LIQUIDITY_HEALTHCHECK_TOPIC",

    # config
    "LiquidityConfig",
]