from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LiquidityConfig:
    """
    Конфігурація liquidity-модуля.

    Рекомендація:
    - цей клас має бути вкладений у глобальний AppConfig
    - усі пороги краще робити явними та легко тюнити
    """

    enabled: bool = True

    # Pivot / swing detection
    pivot_lookback: int = 3
    pivot_lookforward: int = 3
    min_swing_distance_pct: float = 0.0020

    # Equal highs / lows
    equal_level_tolerance_pct: float = 0.0008
    min_equal_touches: int = 2
    max_equal_cluster_width_pct: float = 0.0012

    # Stop clusters
    stop_cluster_padding_pct: float = 0.0015
    cluster_merge_distance_pct: float = 0.0007

    # Filtering / retention
    max_active_levels: int = 200
    max_active_clusters: int = 100
    level_expiry_bars: int = 300
    min_confidence: float = 0.35

    # Optional behavior
    use_atr_tolerance: bool = True
    atr_period: int = 14
    use_volume_in_scoring: bool = True
    use_reaction_strength_in_scoring: bool = True
    publish_events: bool = True
    incremental_mode: bool = True

    def validate(self) -> None:
        if self.pivot_lookback < 1:
            raise ValueError("pivot_lookback must be >= 1")

        if self.pivot_lookforward < 1:
            raise ValueError("pivot_lookforward must be >= 1")

        if not 0 < self.equal_level_tolerance_pct < 1:
            raise ValueError("equal_level_tolerance_pct must be between 0 and 1")

        if self.min_equal_touches < 2:
            raise ValueError("min_equal_touches must be >= 2")

        if not 0 < self.stop_cluster_padding_pct < 1:
            raise ValueError("stop_cluster_padding_pct must be between 0 and 1")

        if self.max_active_levels < 1:
            raise ValueError("max_active_levels must be >= 1")

        if self.max_active_clusters < 1:
            raise ValueError("max_active_clusters must be >= 1")

        if self.level_expiry_bars < 1:
            raise ValueError("level_expiry_bars must be >= 1")

        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")