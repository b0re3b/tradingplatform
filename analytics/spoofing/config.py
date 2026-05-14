from __future__ import annotations

from dataclasses import dataclass, field


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

    max_walls_per_symbol: int = 500
    max_history_events_per_level: int = 200

    size_update_epsilon: float = 1e-9
    price_rounding_decimals: int = 8

    estimate_fill_from_trade_flow: bool = False
    estimate_fill_on_touch_only: bool = True


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

    Цей блок замінює параметри зі старого монолітного spoofing_detector.py.
    У новій архітектурі основний state веде PersistenceTracker, але ці
    параметри можуть бути корисні для міграції старої candidate-based логіки,
    cooldown-ів або trade-flow confirmation.
    """

    enabled: bool = False

    candidate_ttl_ms: int = 12_000
    cooldown_ms_same_level: int = 15_000

    max_candidates_per_symbol: int = 200
    max_trade_events_per_symbol: int = 2_000

    similar_candidate_tolerance_bps: float = 1.0

    trade_pressure_window_ms: int = 3_000
    min_opposite_pressure_ratio: float = 1.35

    price_move_confirmation_bps: float = 4.0
    logical_invalidation_distance_multiplier: float = 3.0

    emit_raw_candidate_events: bool = False


@dataclass(slots=True)
class SpoofingAnalyzerConfig:
    """
    Загальний конфіг SpoofingAnalyzer.

    Analyzer є єдиним integration point пакета:
    - підписується на market.* через EventBus;
    - запускає cleanup через Scheduler;
    - публікує analytics.spoofing.* події.
    """

    enabled: bool = True

    publish_updates: bool = True
    publish_detected_only: bool = False
    publish_lifecycle_events: bool = False
    publish_score_updates: bool = True
    publish_errors: bool = True

    max_tracked_walls_per_symbol: int = 500
    max_detector_results_per_cycle: int = 50

    event_topic_orderbook: str = "market.orderbook"
    event_topic_trade: str = "market.trade"

    event_topic_lifecycle: str = "analytics.spoofing.lifecycle"
    event_topic_updated: str = "analytics.spoofing.updated"
    event_topic_detected: str = "analytics.spoofing.detected"
    event_topic_score_updated: str = "analytics.spoofing.score_updated"
    event_topic_error: str = "analytics.spoofing.error"

    scheduler_cleanup_job_name: str = "analytics.spoofing.persistence_cleanup"
    scheduler_cleanup_enabled: bool = True


@dataclass(slots=True)
class SpoofingConfig:
    """
    Кореневий конфіг пакета analytics.spoofing.

    Цей config не дублює core.config.Config. Він є доменною typed config-моделлю
    і має передаватися в SpoofingAnalyzer / detector-и через constructor
    dependency injection.
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
    candidate_tracking: CandidateTrackingConfig = field(default_factory=CandidateTrackingConfig)
    analyzer: SpoofingAnalyzerConfig = field(default_factory=SpoofingAnalyzerConfig)

    def validate(self) -> None:
        """
        Перевіряє config на логічну коректність.

        Валідація виконується явно на етапі bootstrap або перед передачею
        config у SpoofingAnalyzer.
        """

        self._validate_wall_detection()
        self._validate_persistence()
        self._validate_pull_detection()
        self._validate_fake_liquidity()
        self._validate_layering()
        self._validate_flip_pressure()
        self._validate_scoring()
        self._validate_candidate_tracking()
        self._validate_analyzer()

    # ------------------------------------------------------------------
    # Section validators
    # ------------------------------------------------------------------

    def _validate_wall_detection(self) -> None:
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
            "persistence.max_walls_per_symbol",
            self.persistence.max_walls_per_symbol,
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
        self._validate_positive_int(
            "candidate_tracking.candidate_ttl_ms",
            self.candidate_tracking.candidate_ttl_ms,
        )
        self._validate_non_negative_int(
            "candidate_tracking.cooldown_ms_same_level",
            self.candidate_tracking.cooldown_ms_same_level,
        )
        self._validate_positive_int(
            "candidate_tracking.max_candidates_per_symbol",
            self.candidate_tracking.max_candidates_per_symbol,
        )
        self._validate_positive_int(
            "candidate_tracking.max_trade_events_per_symbol",
            self.candidate_tracking.max_trade_events_per_symbol,
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
        self._validate_positive_int(
            "analyzer.max_tracked_walls_per_symbol",
            self.analyzer.max_tracked_walls_per_symbol,
        )
        self._validate_positive_int(
            "analyzer.max_detector_results_per_cycle",
            self.analyzer.max_detector_results_per_cycle,
        )

        if self.analyzer.scheduler_cleanup_enabled:
            if not self.analyzer.scheduler_cleanup_job_name.strip():
                raise ValueError("analyzer.scheduler_cleanup_job_name must not be empty")

        self._validate_topic("analyzer.event_topic_orderbook", self.analyzer.event_topic_orderbook)
        self._validate_topic("analyzer.event_topic_trade", self.analyzer.event_topic_trade)
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
        return self.persistence.cleanup_interval_ms / 1000.0

    def is_symbol_allowed(self, symbol: str) -> bool:
        if not self.symbols:
            return True
        return symbol in self.symbols

    # ------------------------------------------------------------------
    # Primitive validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ratio(name: str, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0")

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")

    @staticmethod
    def _validate_topic(name: str, value: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{name} must not be empty")


__all__ = [
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