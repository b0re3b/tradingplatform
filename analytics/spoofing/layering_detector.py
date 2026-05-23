from __future__ import annotations
from core.logger import get_logger

from statistics import mean
from typing import Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingDetector
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    OrderbookWallState,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSide,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DetectorResult,
    LayeringCandidateContext,
    LayeringCluster,
    SpoofingFeatures,
    SpoofingKey,
    TrackedWall,
    spoofing_key_to_dict,
)
from .persistence_tracker import PersistenceTracker


class LayeringDetector(BaseSpoofingDetector):
    """
    Detector multi-level layering.

    Основна ідея:
    - на одній стороні книги присутні кілька аномально великих рівнів;
    - рівні знаходяться близько по ціні;
    - сумарна ліквідність велика;
    - значна частина рівнів знімається/слабшає в близькому часовому вікні.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct production input flow:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> PersistenceTracker
            -> LayeringDetector

    Важливо:
    - працює поверх PersistenceTracker state;
    - не аналізує raw orderbook напряму;
    - не читає exchange adapters напряму;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - повертає тільки DetectorResult або None.
    """

    component = SpoofingComponent.LAYERING_DETECTOR

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
        persistence_tracker: PersistenceTracker,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "__init__", _analytics_args)
        except Exception:
            pass
        super().__init__(
            event_bus=event_bus,
            scheduler=scheduler,
            config=config,
        )
        self.persistence_tracker = persistence_tracker

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> DetectorResult | None:
        """
        Аналіз одного wall не є природним для layering,
        але підтримується через пошук кластера навколо цього рівня.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze", _analytics_args)
        except Exception:
            pass
        cluster = self._find_cluster_around_wall(
            wall=wall,
            current_mid_price=current_mid_price,
        )
        if cluster is None:
            return None

        candidate = self._evaluate_cluster(
            cluster=cluster,
            current_mid_price=current_mid_price,
        )
        if candidate is None:
            return None

        return self._build_result(candidate)

    def analyze_key(
        self,
        *,
        key: SpoofingKey,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Key-first API для scoped futures market.

        key:
            exchange + market_type + symbol + timeframe
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze_key", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.layering.enabled:
            return []

        walls = self.persistence_tracker.get_walls_for_key(key)

        return self.analyze_many(
            walls=walls,
            key=key,
            current_mid_price=current_mid_price,
        )

    def analyze_many(
        self,
        walls: Iterable[TrackedWall],
        *,
        key: SpoofingKey | None = None,
        exchange: str | None = None,
        symbol: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує набір tracked walls і повертає позитивні layering candidates.

        New code should pass key=SpoofingKey.
        Legacy filters exchange/symbol/market_type/timeframe залишені для міграції.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze_many", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.layering.enabled:
            return []

        filtered = self._filter_walls(
            walls=walls,
            key=key,
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )
        if not filtered:
            return []

        clusters = self._build_clusters(
            walls=filtered,
            current_mid_price=current_mid_price,
        )

        results: list[DetectorResult] = []
        seen_cluster_keys: set[str] = set()

        for cluster in clusters:
            cluster_key = self._cluster_key(cluster)
            if cluster_key in seen_cluster_keys:
                continue
            seen_cluster_keys.add(cluster_key)

            candidate = self._evaluate_cluster(
                cluster=cluster,
                current_mid_price=current_mid_price,
            )
            if candidate is None:
                continue

            result = self._build_result(candidate)
            if result.is_positive():
                results.append(result)

        results.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        return results

    def analyze_scope(
        self,
        *,
        exchange: str,
        symbol: str,
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        current_mid_price: float | None = None,
    ) -> list[DetectorResult]:
        """
        Аналізує всі tracked walls одного scoped futures market.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze_scope", _analytics_args)
        except Exception:
            pass
        key = self.make_key(
            exchange=exchange,
            market_type=market_type,
            symbol=symbol,
            timeframe=timeframe,
        )

        return self.analyze_key(
            key=key,
            current_mid_price=current_mid_price,
        )

    def analyze_symbol(
        self,
        *,
        exchange: str,
        symbol: str,
        current_mid_price: float | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> list[DetectorResult]:
        """
        Backward-compatible helper.

        New code should use analyze_key() або analyze_scope().
        Якщо market_type/timeframe не передані, аналізує всі scope-и для
        exchange + symbol.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "analyze_symbol", _analytics_args)
        except Exception:
            pass
        walls = self.persistence_tracker.get_walls_for_symbol(
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            timeframe=timeframe,
        )

        return self.analyze_many(
            walls=walls,
            exchange=exchange,
            symbol=symbol,
            market_type=market_type,
            timeframe=timeframe,
            current_mid_price=current_mid_price,
        )

    def is_layering_candidate(
        self,
        wall: TrackedWall,
        *,
        current_mid_price: float | None = None,
    ) -> bool:
        """
        Boolean helper для швидкої перевірки tracked wall.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_layering_candidate", _analytics_args)
        except Exception:
            pass
        return self.analyze(
            wall,
            current_mid_price=current_mid_price,
        ) is not None

    # -------------------------------------------------------------------------
    # Cluster construction
    # -------------------------------------------------------------------------

    def _build_clusters(
        self,
        *,
        walls: list[TrackedWall],
        current_mid_price: float | None = None,
    ) -> list[LayeringCluster]:
        """
        Будує потенційні layering-кластери для кожного scoped market + side окремо.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_clusters", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.layering.enabled:
            return []

        relevant = [
            wall
            for wall in walls
            if self._is_relevant_wall(
                wall=wall,
                current_mid_price=current_mid_price,
            )
        ]
        if not relevant:
            return []

        clusters: list[LayeringCluster] = []
        groups = self._group_walls_by_market_side(relevant)

        for key, side_walls in groups.items():
            exchange, market_type, symbol, timeframe, side = key
            ordered = self._sort_walls_for_side(side_walls, side)
            current_group: list[TrackedWall] = []

            for wall in ordered:
                if not current_group:
                    current_group.append(wall)
                    continue

                previous = current_group[-1]
                gap_bps = self.bps_distance(wall.price, previous.price)

                if gap_bps <= self.config.layering.max_price_gap_bps_between_layers:
                    current_group.append(wall)
                    continue

                cluster = self._make_cluster(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    walls=current_group,
                    current_mid_price=current_mid_price,
                )
                if cluster is not None:
                    clusters.append(cluster)

                current_group = [wall]

            if current_group:
                cluster = self._make_cluster(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side,
                    walls=current_group,
                    current_mid_price=current_mid_price,
                )
                if cluster is not None:
                    clusters.append(cluster)

        return clusters

    def _find_cluster_around_wall(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None = None,
    ) -> LayeringCluster | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_find_cluster_around_wall", _analytics_args)
        except Exception:
            pass
        if not self.should_process_key(wall.key):
            return None

        symbol_walls = self.persistence_tracker.get_walls_for_key(
            wall.key,
            side=wall.side,
        )
        clusters = self._build_clusters(
            walls=symbol_walls,
            current_mid_price=current_mid_price,
        )

        for cluster in clusters:
            if any(item.wall_id == wall.wall_id for item in cluster.walls):
                return cluster

        return None

    def _make_cluster(
        self,
        *,
        exchange: str,
        symbol: str,
        side: SpoofingSide,
        walls: list[TrackedWall],
        market_type: str = DEFAULT_MARKET_TYPE,
        timeframe: str = DEFAULT_TIMEFRAME,
        current_mid_price: float | None = None,
    ) -> LayeringCluster | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_make_cluster", _analytics_args)
        except Exception:
            pass
        if len(walls) < self.config.layering.min_layers:
            return None

        if not walls:
            return None

        first_wall = walls[0]
        key = first_wall.key

        if not self.should_process_key(key):
            return None

        # Safety: кластер не має змішувати різні exchange/market_type/symbol/timeframe/side.
        scoped_walls = [
            wall
            for wall in walls
            if wall.key == key and wall.side == side
        ]
        if len(scoped_walls) < self.config.layering.min_layers:
            return None

        total_notional = sum(wall.price * wall.max_size for wall in scoped_walls)
        if total_notional < self.config.layering.min_total_layer_notional:
            return None

        average_pull_ratio = self._mean_or_zero(wall.pull_ratio for wall in scoped_walls)
        average_fill_ratio = self._mean_or_zero(wall.fill_ratio for wall in scoped_walls)
        average_lifetime_ms = self._mean_or_zero(wall.lifetime_ms for wall in scoped_walls)
        synchronized_pull_ratio = self._estimate_synchronized_pull_ratio(scoped_walls)
        price_span_bps = self._estimate_price_span_bps(scoped_walls)

        layering_score = self._estimate_layering_score(
            walls=scoped_walls,
            total_notional=total_notional,
            synchronized_pull_ratio=synchronized_pull_ratio,
            price_span_bps=price_span_bps,
            current_mid_price=current_mid_price,
        )

        cluster_price = self._mean_or_zero(wall.price for wall in scoped_walls)
        cluster_wall_id = self._resolve_cluster_wall_id(scoped_walls)

        return LayeringCluster(
            exchange=first_wall.exchange,
            market_type=first_wall.market_type,
            symbol=first_wall.symbol,
            timeframe=first_wall.timeframe,
            exchange_symbol=first_wall.exchange_symbol,
            side=side,
            walls=scoped_walls,
            total_notional=total_notional,
            average_pull_ratio=average_pull_ratio,
            average_fill_ratio=average_fill_ratio,
            average_lifetime_ms=average_lifetime_ms,
            synchronized_pull_ratio=synchronized_pull_ratio,
            price_span_bps=price_span_bps,
            layering_score=layering_score,
            cluster_price=cluster_price,
            cluster_wall_id=cluster_wall_id,
        )

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_cluster(
        self,
        *,
        cluster: LayeringCluster,
        current_mid_price: float | None = None,
    ) -> LayeringCandidateContext | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_evaluate_cluster", _analytics_args)
        except Exception:
            pass
        if not self.should_process_key(cluster.key):
            return None

        if not self._passes_cluster_filters(cluster):
            return None

        price_reaction_bps = self._estimate_cluster_price_reaction_bps(
            cluster=cluster,
            current_mid_price=current_mid_price,
        )

        score = self._compute_score(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )
        confidence = self._compute_confidence(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )
        reason = self._build_reason(
            cluster=cluster,
            price_reaction_bps=price_reaction_bps,
        )

        return LayeringCandidateContext(
            cluster=cluster,
            confidence=confidence,
            score=score,
            reason=reason,
            price_reaction_bps=price_reaction_bps,
        )

    def _passes_cluster_filters(self, cluster: LayeringCluster) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_passes_cluster_filters", _analytics_args)
        except Exception:
            pass
        if len(cluster.walls) < self.config.layering.min_layers:
            return False

        if cluster.total_notional < self.config.layering.min_total_layer_notional:
            return False

        if cluster.synchronized_pull_ratio <= 0.0:
            return False

        if cluster.synchronized_pull_ratio < 0.5:
            return False

        max_allowed_fill = max(
            self.config.fake_liquidity.max_fill_ratio,
            self.config.pull_detection.max_fill_ratio_for_pull,
        )
        if cluster.average_fill_ratio > max_allowed_fill:
            return False

        if cluster.average_pull_ratio < self.config.pull_detection.min_pull_ratio:
            return False

        return True

    def _build_result(
        self,
        candidate: LayeringCandidateContext,
    ) -> DetectorResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_result", _analytics_args)
        except Exception:
            pass
        features = self._build_features(candidate)
        cluster = candidate.cluster

        return DetectorResult(
            detector=self.component,
            decision=DetectorDecision.POSITIVE,
            score=candidate.score,
            confidence=candidate.confidence,
            reason=candidate.reason,
            features=features,
            wall_id=cluster.cluster_wall_id,
            pattern=SpoofingPattern.MULTI_LEVEL_LAYERING,
            metadata={
                "scope": spoofing_key_to_dict(cluster.key),
                "exchange_symbol": cluster.exchange_symbol,
                "layers": len(cluster.walls),
                "total_notional": cluster.total_notional,
                "average_pull_ratio": cluster.average_pull_ratio,
                "average_fill_ratio": cluster.average_fill_ratio,
                "average_lifetime_ms": cluster.average_lifetime_ms,
                "synchronized_pull_ratio": cluster.synchronized_pull_ratio,
                "price_span_bps": cluster.price_span_bps,
                "price_reaction_bps": candidate.price_reaction_bps,
                "layering_score": cluster.layering_score,
                "wall_ids": [wall.wall_id for wall in cluster.walls],
            },
        )

    def _build_features(
        self,
        candidate: LayeringCandidateContext,
    ) -> SpoofingFeatures:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_features", _analytics_args)
        except Exception:
            pass
        cluster = candidate.cluster
        repetition_count = self._estimate_cluster_repetition_count(cluster)

        cancel_to_fill_ratio = self._compute_cancel_to_fill_ratio(
            pull_ratio=cluster.average_pull_ratio,
            fill_ratio=cluster.average_fill_ratio,
        )

        reference_mid = self._resolve_cluster_reference_mid(cluster)
        distance_from_mid_bps = (
            self.bps_distance(cluster.cluster_price, reference_mid)
            if reference_mid is not None and reference_mid > 0
            else 0.0
        )

        return SpoofingFeatures(
            symbol=cluster.symbol,
            exchange=cluster.exchange,
            market_type=cluster.market_type,
            timeframe=cluster.timeframe,
            exchange_symbol=cluster.exchange_symbol,
            side=cluster.side,
            price=cluster.cluster_price,
            wall_size=sum(wall.max_size for wall in cluster.walls),
            wall_size_ratio=self._estimate_cluster_wall_size_ratio(cluster),
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=cluster.average_lifetime_ms,
            updates_count=sum(wall.updates_count for wall in cluster.walls),
            repetition_count=repetition_count,
            fill_ratio=cluster.average_fill_ratio,
            pull_ratio=cluster.average_pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=candidate.price_reaction_bps,
            pressure_flip_strength=0.0,
            layering_score=cluster.layering_score,
            is_near_best_quote=self._cluster_near_best_quote(cluster),
            is_fast_pull=self._cluster_fast_pull(cluster),
            is_fake_liquidity=False,
            is_layering=True,
            metadata={
                "scope": spoofing_key_to_dict(cluster.key),
                "layers": len(cluster.walls),
                "total_notional": cluster.total_notional,
                "synchronized_pull_ratio": cluster.synchronized_pull_ratio,
                "price_span_bps": cluster.price_span_bps,
                "wall_ids": [wall.wall_id for wall in cluster.walls],
                "detector": self.component.value,
            },
        )

    # -------------------------------------------------------------------------
    # Scoring / confidence
    # -------------------------------------------------------------------------

    def _compute_score(
        self,
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_score", _analytics_args)
        except Exception:
            pass
        notional_component = self._normalize_total_notional(cluster.total_notional)
        pull_component = self.clamp(cluster.average_pull_ratio, 0.0, 1.0)
        fill_component = 1.0 - self.clamp(cluster.average_fill_ratio, 0.0, 1.0)
        sync_component = self.clamp(cluster.synchronized_pull_ratio, 0.0, 1.0)

        max_span = self._cluster_max_span_bps(cluster)
        compactness_component = 1.0 - self.clamp(
            cluster.price_span_bps / max(max_span, 1e-12),
            0.0,
            1.0,
        )

        layering_component = self.clamp(cluster.layering_score, 0.0, 1.0)
        reaction_component = self.clamp(
            price_reaction_bps
            / max(self.config.flip_pressure.min_price_reaction_bps * 3.0, 1e-12),
            0.0,
            1.0,
        )

        raw_score = (
            0.18 * notional_component
            + 0.17 * pull_component
            + 0.12 * fill_component
            + 0.20 * sync_component
            + 0.12 * compactness_component
            + 0.16 * layering_component
            + 0.05 * reaction_component
        )
        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_confidence(
        self,
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_confidence", _analytics_args)
        except Exception:
            pass
        confidence = 0.42

        if len(cluster.walls) >= self.config.layering.min_layers + 1:
            confidence += 0.10

        if cluster.synchronized_pull_ratio >= 0.75:
            confidence += 0.14

        if cluster.average_pull_ratio >= self.config.pull_detection.strong_pull_ratio:
            confidence += 0.10

        if cluster.average_fill_ratio <= self.config.fake_liquidity.max_fill_ratio * 0.75:
            confidence += 0.08

        if cluster.total_notional >= self.config.layering.min_total_layer_notional * 2.0:
            confidence += 0.08

        if price_reaction_bps >= self.config.flip_pressure.min_price_reaction_bps:
            confidence += 0.04

        if cluster.layering_score >= 0.75:
            confidence += 0.03

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Cluster metrics
    # -------------------------------------------------------------------------

    def _estimate_synchronized_pull_ratio(
        self,
        walls: list[TrackedWall],
    ) -> float:
        """
        Частка рівнів, які ослабли/зникли в близькому часовому вікні.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_synchronized_pull_ratio", _analytics_args)
        except Exception:
            pass
        if not walls:
            return 0.0

        relevant_states = self._removed_or_weakened_states()
        reference_times = [
            wall.last_seen_at
            for wall in walls
            if wall.state in relevant_states
        ]

        if len(reference_times) < self.config.layering.min_layers:
            return 0.0

        reference_times.sort()
        window_ms = self.config.layering.synchronized_pull_window_ms

        best_count = 0
        for base_time in reference_times:
            count = sum(
                1
                for ts in reference_times
                if abs((ts - base_time).total_seconds() * 1000.0) <= window_ms
            )
            best_count = max(best_count, count)

        return self.normalize_ratio(best_count, len(walls))

    def _estimate_price_span_bps(
        self,
        walls: list[TrackedWall],
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_price_span_bps", _analytics_args)
        except Exception:
            pass
        if len(walls) < 2:
            return 0.0

        prices = [wall.price for wall in walls if wall.price > 0]
        if len(prices) < 2:
            return 0.0

        return self.bps_distance(max(prices), min(prices))

    def _estimate_layering_score(
        self,
        *,
        walls: list[TrackedWall],
        total_notional: float,
        synchronized_pull_ratio: float,
        price_span_bps: float,
        current_mid_price: float | None = None,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_layering_score", _analytics_args)
        except Exception:
            pass
        layers_component = self.clamp(
            len(walls) / max(self.config.layering.min_layers + 2, 1),
            0.0,
            1.0,
        )
        notional_component = self._normalize_total_notional(total_notional)
        sync_component = self.clamp(synchronized_pull_ratio, 0.0, 1.0)

        max_span = self.config.layering.max_price_gap_bps_between_layers * max(
            len(walls) - 1,
            1,
        )
        compactness_component = 1.0 - self.clamp(
            price_span_bps / max(max_span, 1e-12),
            0.0,
            1.0,
        )

        proximity_component = 0.5
        if current_mid_price is not None and current_mid_price > 0:
            cluster_price = self._mean_or_zero(wall.price for wall in walls)
            distance = self.bps_distance(cluster_price, current_mid_price)
            max_distance = max(
                self.config.wall_detection.max_distance_from_mid_bps,
                1e-12,
            )
            proximity_component = 1.0 - self.clamp(
                distance / max_distance,
                0.0,
                1.0,
            )

        value = (
            0.20 * layers_component
            + 0.25 * notional_component
            + 0.30 * sync_component
            + 0.15 * compactness_component
            + 0.10 * proximity_component
        )
        return self.clamp(value, 0.0, 1.0)

    def _estimate_cluster_price_reaction_bps(
        self,
        *,
        cluster: LayeringCluster,
        current_mid_price: float | None,
    ) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_cluster_price_reaction_bps", _analytics_args)
        except Exception:
            pass
        if current_mid_price is None or current_mid_price <= 0:
            return 0.0

        reference_mid = self._resolve_cluster_reference_mid(cluster)
        if reference_mid is None or reference_mid <= 0:
            return 0.0

        signed_move = self.signed_bps_move(current_mid_price, reference_mid)

        if cluster.side == SpoofingSide.ASK:
            return max(0.0, signed_move)

        if cluster.side == SpoofingSide.BID:
            return max(0.0, -signed_move)

        return 0.0

    def _estimate_cluster_wall_size_ratio(
        self,
        cluster: LayeringCluster,
    ) -> float:
        """
        Груба оцінка cluster-vs-baseline ratio.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_cluster_wall_size_ratio", _analytics_args)
        except Exception:
            pass
        base_notional = max(self.config.wall_detection.min_wall_size_abs, 1e-12)
        return self.clamp(cluster.total_notional / base_notional, 0.0, 1000.0)

    # -------------------------------------------------------------------------
    # Filters / helpers
    # -------------------------------------------------------------------------

    def _filter_walls(
        self,
        *,
        walls: Iterable[TrackedWall],
        key: SpoofingKey | None = None,
        exchange: str | None,
        symbol: str | None,
        market_type: str | None,
        timeframe: str | None,
    ) -> list[TrackedWall]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_filter_walls", _analytics_args)
        except Exception:
            pass
        normalized_exchange = (
            self.normalize_exchange(exchange)
            if exchange is not None
            else None
        )
        normalized_symbol = (
            self.normalize_symbol(symbol)
            if symbol is not None
            else None
        )
        normalized_market_type = (
            self.normalize_market_type(market_type)
            if market_type is not None
            else None
        )
        normalized_timeframe = (
            self.normalize_timeframe(timeframe)
            if timeframe is not None
            else None
        )

        filtered: list[TrackedWall] = []

        for wall in walls:
            if key is not None and wall.key != key:
                continue
            if normalized_exchange is not None and wall.exchange != normalized_exchange:
                continue
            if normalized_market_type is not None and wall.market_type != normalized_market_type:
                continue
            if normalized_symbol is not None and wall.symbol != normalized_symbol:
                continue
            if normalized_timeframe is not None and wall.timeframe != normalized_timeframe:
                continue
            if not self.should_process_key(wall.key):
                continue

            filtered.append(wall)

        return filtered

    def _is_relevant_wall(
        self,
        *,
        wall: TrackedWall,
        current_mid_price: float | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_relevant_wall", _analytics_args)
        except Exception:
            pass
        if not self.should_process_key(wall.key):
            return False

        if wall.max_size <= 0 or wall.price <= 0:
            return False

        if wall.price * wall.max_size < self.config.wall_detection.min_wall_size_abs:
            return False

        if wall.state not in self._candidate_wall_states():
            return False

        if current_mid_price is not None and current_mid_price > 0:
            distance = self.bps_distance(wall.price, current_mid_price)
            if distance > self.config.wall_detection.max_distance_from_mid_bps * 2.0:
                return False

        if wall.side not in {SpoofingSide.BID, SpoofingSide.ASK}:
            return False

        return True

    @staticmethod
    def _group_walls_by_market_side(
        walls: Iterable[TrackedWall],
    ) -> dict[tuple[str, str, str, str, SpoofingSide], list[TrackedWall]]:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_group_walls_by_market_side", _analytics_args)
        except Exception:
            pass
        groups: dict[tuple[str, str, str, str, SpoofingSide], list[TrackedWall]] = {}

        for wall in walls:
            key = (
                wall.exchange,
                wall.market_type,
                wall.symbol,
                wall.timeframe,
                wall.side,
            )
            groups.setdefault(key, []).append(wall)

        return groups

    @staticmethod
    def _sort_walls_for_side(
        walls: list[TrackedWall],
        side: SpoofingSide,
    ) -> list[TrackedWall]:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_sort_walls_for_side", _analytics_args)
        except Exception:
            pass
        if side == SpoofingSide.BID:
            return sorted(walls, key=lambda item: item.price, reverse=True)
        return sorted(walls, key=lambda item: item.price)

    @staticmethod
    def _resolve_cluster_wall_id(
        walls: list[TrackedWall],
    ) -> str | None:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_cluster_wall_id", _analytics_args)
        except Exception:
            pass
        if not walls:
            return None

        strongest = max(walls, key=lambda item: item.price * item.max_size)
        return strongest.wall_id

    @staticmethod
    def _resolve_cluster_reference_mid(
        cluster: LayeringCluster,
    ) -> float | None:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_cluster_reference_mid", _analytics_args)
        except Exception:
            pass
        mids = [
            wall.mid_price_at_creation
            for wall in cluster.walls
            if wall.mid_price_at_creation is not None and wall.mid_price_at_creation > 0
        ]
        if not mids:
            return None
        return mean(mids)

    def _normalize_total_notional(self, total_notional: float) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_total_notional", _analytics_args)
        except Exception:
            pass
        base = max(self.config.layering.min_total_layer_notional, 1e-12)
        value = (total_notional - base) / max(base * 2.0, 1e-12)
        return self.clamp(value, 0.0, 1.0)

    def _cluster_near_best_quote(
        self,
        cluster: LayeringCluster,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_cluster_near_best_quote", _analytics_args)
        except Exception:
            pass
        return any(
            wall.touch_count > 0 or wall.near_touch_count > 0
            for wall in cluster.walls
        )

    def _cluster_fast_pull(
        self,
        cluster: LayeringCluster,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_cluster_fast_pull", _analytics_args)
        except Exception:
            pass
        return cluster.average_lifetime_ms <= self.config.pull_detection.fast_pull_lifetime_ms

    def _estimate_cluster_repetition_count(
        self,
        cluster: LayeringCluster,
    ) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_cluster_repetition_count", _analytics_args)
        except Exception:
            pass
        if not cluster.walls:
            return 0

        counts = []
        for wall in cluster.walls:
            history = self.persistence_tracker.get_recent_history(
                exchange=wall.exchange,
                market_type=wall.market_type,
                symbol=wall.symbol,
                timeframe=wall.timeframe,
                side=wall.side,
                price=wall.price,
                limit=50,
            )
            counts.append(len(history))

        return int(mean(counts)) if counts else 0

    @staticmethod
    def _cluster_key(cluster: LayeringCluster) -> str:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_cluster_key", _analytics_args)
        except Exception:
            pass
        wall_ids = sorted(wall.wall_id for wall in cluster.walls)
        scope = spoofing_key_to_dict(cluster.key)
        return (
            f"{scope['exchange']}:{scope['market_type']}:{scope['symbol']}:"
            f"{scope['timeframe']}:{cluster.side.value}:{'|'.join(wall_ids)}"
        )

    @staticmethod
    def _candidate_wall_states() -> set[OrderbookWallState]:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_candidate_wall_states", _analytics_args)
        except Exception:
            pass
        return {
            OrderbookWallState.ACTIVE,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.PULLED,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }

    @staticmethod
    def _removed_or_weakened_states() -> set[OrderbookWallState]:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_removed_or_weakened_states", _analytics_args)
        except Exception:
            pass
        return {
            OrderbookWallState.WEAKENING,
            OrderbookWallState.PULLED,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }

    @staticmethod
    def _compute_cancel_to_fill_ratio(
        *,
        pull_ratio: float,
        fill_ratio: float,
    ) -> float:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_cancel_to_fill_ratio", _analytics_args)
        except Exception:
            pass
        if fill_ratio > 0:
            return pull_ratio / fill_ratio
        if pull_ratio > 0:
            return pull_ratio
        return 0.0

    def _cluster_max_span_bps(self, cluster: LayeringCluster) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_cluster_max_span_bps", _analytics_args)
        except Exception:
            pass
        return self.config.layering.max_price_gap_bps_between_layers * max(
            len(cluster.walls) - 1,
            1,
        )

    @staticmethod
    def _mean_or_zero(values: Iterable[float]) -> float:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_mean_or_zero", _analytics_args)
        except Exception:
            pass
        items = [max(value, 0.0) for value in values]
        return mean(items) if items else 0.0

    @staticmethod
    def _build_reason(
        *,
        cluster: LayeringCluster,
        price_reaction_bps: float,
    ) -> str:
        try:
            _analytics_class_name = "LayeringDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_reason", _analytics_args)
        except Exception:
            pass
        return (
            f"layering candidate detected for {cluster.side.value.upper()} side, "
            f"exchange={cluster.exchange}, "
            f"market_type={cluster.market_type}, "
            f"symbol={cluster.symbol}, "
            f"timeframe={cluster.timeframe}, "
            f"layers={len(cluster.walls)}, "
            f"total_notional={cluster.total_notional:.2f}, "
            f"avg_pull_ratio={cluster.average_pull_ratio:.4f}, "
            f"avg_fill_ratio={cluster.average_fill_ratio:.4f}, "
            f"sync_pull_ratio={cluster.synchronized_pull_ratio:.4f}, "
            f"price_span_bps={cluster.price_span_bps:.4f}, "
            f"layering_score={cluster.layering_score:.4f}, "
            f"price_reaction_bps={price_reaction_bps:.4f}"
        )


__all__ = ["LayeringDetector"]