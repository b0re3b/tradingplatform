from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WallDetectionConfig:
    """
    Налаштування виявлення великих стін у стакані.
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
    Налаштування життєвого циклу стінок.
    """
    enabled: bool = True

    wall_ttl_ms: int = 15_000
    min_tracking_lifetime_ms: int = 50
    cleanup_interval_ms: int = 2_000

    size_update_epsilon: float = 1e-9
    price_rounding_decimals: int = 8

    estimate_fill_from_trade_flow: bool = False
    estimate_fill_on_touch_only: bool = True


@dataclass(slots=True)
class PullDetectionConfig:
    """
    Налаштування виявлення швидкого зняття ліквідності.
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
    """
    enabled: bool = True

    max_fill_ratio: float = 0.20
    min_pull_ratio: float = 0.70
    max_lifetime_ms: int = 4_000
    min_price_reaction_bps: float = 2.0


@dataclass(slots=True)
class LayeringConfig:
    """
    Налаштування детекції layering.
    """
    enabled: bool = True

    min_layers: int = 3
    max_price_gap_bps_between_layers: float = 5.0
    min_total_layer_notional: float = 150_000.0
    synchronized_pull_window_ms: int = 1_000


@dataclass(slots=True)
class FlipPressureConfig:
    """
    Налаштування патерну pull-and-reversal / pressure flip.
    """
    enabled: bool = True

    min_price_reaction_bps: float = 3.0
    reaction_window_ms: int = 3_000
    min_pressure_flip_strength: float = 0.60


@dataclass(slots=True)
class SpoofingScoreConfig:
    """
    Ваги та пороги фінального spoofing score.
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
class SpoofingAnalyzerConfig:
    """
    Загальний конфіг spoofing analyzer.
    """
    enabled: bool = True

    publish_updates: bool = True
    publish_detected_only: bool = False

    max_tracked_walls_per_symbol: int = 500
    max_detector_results_per_cycle: int = 50

    event_topic_orderbook: str = "market.orderbook"
    event_topic_trade: str = "market.trade"
    event_topic_updated: str = "analytics.spoofing.updated"
    event_topic_detected: str = "analytics.spoofing.detected"
    event_topic_score_updated: str = "analytics.spoofing.score_updated"


@dataclass(slots=True)
class SpoofingConfig:
    """
    Кореневий конфіг пакета spoofing.
    """
    enabled: bool = True
    exchange: str | None = None
    symbols: list[str] = field(default_factory=list)

    wall_detection: WallDetectionConfig = field(default_factory=WallDetectionConfig)
    persistence: PersistenceTrackerConfig = field(default_factory=PersistenceTrackerConfig)
    pull_detection: PullDetectionConfig = field(default_factory=PullDetectionConfig)
    fake_liquidity: FakeLiquidityConfig = field(default_factory=FakeLiquidityConfig)
    layering: LayeringConfig = field(default_factory=LayeringConfig)
    flip_pressure: FlipPressureConfig = field(default_factory=FlipPressureConfig)
    scoring: SpoofingScoreConfig = field(default_factory=SpoofingScoreConfig)
    analyzer: SpoofingAnalyzerConfig = field(default_factory=SpoofingAnalyzerConfig)

    def validate(self) -> None:
        if self.wall_detection.min_wall_size_abs < 0:
            raise ValueError("wall_detection.min_wall_size_abs must be >= 0")

        if self.wall_detection.min_wall_size_ratio < 0:
            raise ValueError("wall_detection.min_wall_size_ratio must be >= 0")

        if self.pull_detection.min_pull_ratio < 0 or self.pull_detection.min_pull_ratio > 1:
            raise ValueError("pull_detection.min_pull_ratio must be in [0, 1]")

        if self.pull_detection.max_fill_ratio_for_pull < 0 or self.pull_detection.max_fill_ratio_for_pull > 1:
            raise ValueError("pull_detection.max_fill_ratio_for_pull must be in [0, 1]")

        if self.fake_liquidity.max_fill_ratio < 0 or self.fake_liquidity.max_fill_ratio > 1:
            raise ValueError("fake_liquidity.max_fill_ratio must be in [0, 1]")

        if self.fake_liquidity.min_pull_ratio < 0 or self.fake_liquidity.min_pull_ratio > 1:
            raise ValueError("fake_liquidity.min_pull_ratio must be in [0, 1]")

        if self.scoring.detection_threshold < 0 or self.scoring.detection_threshold > 1:
            raise ValueError("scoring.detection_threshold must be in [0, 1]")

        if self.scoring.high_severity_threshold < 0 or self.scoring.high_severity_threshold > 1:
            raise ValueError("scoring.high_severity_threshold must be in [0, 1]")

        if self.scoring.critical_severity_threshold < 0 or self.scoring.critical_severity_threshold > 1:
            raise ValueError("scoring.critical_severity_threshold must be in [0, 1]")

        if self.scoring.high_severity_threshold > self.scoring.critical_severity_threshold:
            raise ValueError(
                "scoring.high_severity_threshold must be <= scoring.critical_severity_threshold"
            )

        weights = [
            self.scoring.weight_wall_size,
            self.scoring.weight_wall_distance,
            self.scoring.weight_persistence,
            self.scoring.weight_pull_speed,
            self.scoring.weight_fill_ratio,
            self.scoring.weight_price_reaction,
            self.scoring.weight_repetition,
            self.scoring.weight_layering,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("all scoring weights must be >= 0")

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("sum of scoring weights must be > 0")