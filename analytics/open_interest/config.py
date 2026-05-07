from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OIThresholds:
    """
    Порогові значення для класифікації режимів, дивергенцій та аномалій.

    Значення є стартовими дефолтами і мають калібруватися під:
    - біржу
    - symbol
    - таймфрейм
    - тип ринку
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

    def validate(self) -> None:
        if self.cleanup_interval_sec <= 0:
            raise ValueError("cleanup_interval_sec must be > 0")

        if self.metrics_interval_sec <= 0:
            raise ValueError("metrics_interval_sec must be > 0")

        if not self.cleanup_job_name.strip():
            raise ValueError("cleanup_job_name must not be empty")

        if not self.metrics_job_name.strip():
            raise ValueError("metrics_job_name must not be empty")

        if self.cleanup_job_timeout_sec is not None and self.cleanup_job_timeout_sec <= 0:
            raise ValueError("cleanup_job_timeout_sec must be > 0 when provided")

        if self.metrics_job_timeout_sec is not None and self.metrics_job_timeout_sec <= 0:
            raise ValueError("metrics_job_timeout_sec must be > 0 when provided")


@dataclass(slots=True)
class OIAnalyzerConfig:
    """
    Головний конфіг Open Interest analytics-модуля.

    Runtime-залежності не зберігаються тут:
    - EventBus передається в OIAnalyzer через constructor dependency injection.
    - Scheduler передається в OIAnalyzer через constructor dependency injection.
    - Logger створюється в OIAnalyzer через core.logger.get_logger().
    """

    enabled: bool = True

    source_name: str = "oi_analyzer"

    emit_updates: bool = True
    emit_regime_changes: bool = True
    emit_divergences: bool = True
    emit_anomalies: bool = True
    emit_squeeze_events: bool = True
    emit_capitulation_events: bool = True

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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")

        if self.stale_context_after_sec <= 0:
            raise ValueError("stale_context_after_sec must be > 0")

        if self.stale_state_cleanup_after_sec <= 0:
            raise ValueError("stale_state_cleanup_after_sec must be > 0")

        if self.stale_state_cleanup_after_sec < self.stale_context_after_sec:
            raise ValueError(
                "stale_state_cleanup_after_sec must be >= stale_context_after_sec"
            )

        self.thresholds.validate()
        self.windows.validate()
        self.cooldowns.validate()
        self.maintenance.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OIAnalyzerConfig":
        """
        Зручний factory для інтеграції з AppConfig / YAML / JSON / env-based config.

        Очікуваний формат:

        {
            "enabled": true,
            "source_name": "oi_analyzer",
            "thresholds": {...},
            "windows": {...},
            "cooldowns": {...},
            "maintenance": {...}
        }
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
        return asdict(self)