from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OIThresholds:
    """
    Порогові значення для класифікації режимів, дивергенцій та аномалій.
    Всі значення підібрані як дефолтні стартові і можуть бути
    відкалібровані під конкретну біржу / таймфрейм / інструмент.
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
        if self.divergence_min_confidence < 0 or self.divergence_min_confidence > 1:
            raise ValueError("divergence_min_confidence must be in [0, 1]")
        if self.anomaly_zscore_threshold <= 0:
            raise ValueError("anomaly_zscore_threshold must be > 0")
        if self.extreme_anomaly_zscore_threshold < self.anomaly_zscore_threshold:
            raise ValueError(
                "extreme_anomaly_zscore_threshold must be >= anomaly_zscore_threshold"
            )
        if self.overheated_zscore_threshold <= 0:
            raise ValueError("overheated_zscore_threshold must be > 0")


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


@dataclass(slots=True)
class OICooldowns:
    """
    Антиспам/дедуплікація подій.
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
class OIAnalyzerConfig:
    """
    Головний конфіг OI-модуля.
    """

    enabled: bool = True
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
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

    @classmethod
    def from_dict(cls, data: dict) -> "OIAnalyzerConfig":
        """
        Зручно для інтеграції з AppConfig / YAML / env-based config.
        """
        data = dict(data or {})

        thresholds = OIThresholds(**data.pop("thresholds", {}))
        windows = OIWindows(**data.pop("windows", {}))
        cooldowns = OICooldowns(**data.pop("cooldowns", {}))

        return cls(
            thresholds=thresholds,
            windows=windows,
            cooldowns=cooldowns,
            **data,
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "emit_updates": self.emit_updates,
            "emit_regime_changes": self.emit_regime_changes,
            "emit_divergences": self.emit_divergences,
            "emit_anomalies": self.emit_anomalies,
            "emit_squeeze_events": self.emit_squeeze_events,
            "emit_capitulation_events": self.emit_capitulation_events,
            "require_price_context": self.require_price_context,
            "require_volume_confirmation": self.require_volume_confirmation,
            "require_funding_for_squeeze": self.require_funding_for_squeeze,
            "normalize_symbol": self.normalize_symbol,
            "store_full_analysis": self.store_full_analysis,
            "stale_context_after_sec": self.stale_context_after_sec,
            "stale_state_cleanup_after_sec": self.stale_state_cleanup_after_sec,
            "thresholds": self.thresholds.__dict__,
            "windows": self.windows.__dict__,
            "cooldowns": self.cooldowns.__dict__,
        }