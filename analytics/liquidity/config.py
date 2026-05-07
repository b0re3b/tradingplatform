from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LiquidityConfig:
    """
    Конфігурація analytics/liquidity модуля.

    Цей config використовується всіма liquidity-компонентами:
    - EqualHighsLowsDetector
    - StopClustersDetector
    - LiquidityScorer
    - LiquidityMap
    - LiquidityService

    Важливо:
    - файл не імпортує core.EventBus / core.Scheduler / core.logger;
    - це чиста dataclass-конфігурація;
    - runtime-залежності передаються через constructor dependency injection
      у відповідних сервісах.
    """

    # ------------------------------------------------------------------
    # Module switch
    # ------------------------------------------------------------------

    enabled: bool = True

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

    # ------------------------------------------------------------------
    # Incremental mode
    # ------------------------------------------------------------------

    incremental_mode: bool = True

    def validate(self) -> None:
        """
        Validate config values early during module construction.

        Raises
        ------
        ValueError
            If one or more config values are invalid.
        """
        errors: list[str] = []

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