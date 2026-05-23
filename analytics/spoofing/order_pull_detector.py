from __future__ import annotations
from core.logger import get_logger

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
    SpoofingType,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    DetectorResult,
    PullCandidateContext,
    SpoofingFeatures,
    SpoofingKey,
    TrackedWall,
    spoofing_key_to_dict,
)
from .persistence_tracker import PersistenceTracker


class OrderPullDetector(BaseSpoofingDetector):
    """
    Detector швидкого зняття ліквідності.

    Призначення:
    - знайти tracked walls, які були достатньо великими;
    - існували недовго;
    - були значно або повністю зняті;
    - майже не були виконані.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Correct production input flow:
        exchange adapters
            -> market.orderbook
            -> OrderBookCache
            -> market.orderbook.updated
            -> SpoofingAnalyzer
            -> PersistenceTracker
            -> OrderPullDetector

    Важливо:
    - працює поверх PersistenceTracker state;
    - не аналізує raw orderbook напряму;
    - не читає exchange adapters напряму;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - повертає тільки DetectorResult або None.
    """

    component = SpoofingComponent.ORDER_PULL_DETECTOR

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
        repetition_count: int | None = None,
    ) -> DetectorResult | None:
        """
        Аналізує один tracked wall на предмет підозрілого pull.
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
        candidate = self._evaluate_pull_candidate(wall=wall)
        if candidate is None:
            return None

        features = self._build_features(
            candidate=candidate,
            current_mid_price=current_mid_price,
            repetition_count=repetition_count,
        )

        return DetectorResult(
            detector=self.component,
            decision=DetectorDecision.POSITIVE,
            score=candidate.score,
            confidence=candidate.confidence,
            reason=candidate.reason,
            features=features,
            wall_id=wall.wall_id,
            pattern=SpoofingPattern.PULL_AND_REVERSAL,
            metadata={
                "scope": spoofing_key_to_dict(wall.key),
                "exchange_symbol": wall.exchange_symbol,
                "spoofing_type": SpoofingType.ORDER_PULL.value,
                "pulled_notional": candidate.pulled_notional,
                "pulled_size_ratio": candidate.pulled_size_ratio,
                "pull_ratio": candidate.pull_ratio,
                "fill_ratio": candidate.fill_ratio,
                "lifetime_ms": candidate.lifetime_ms,
                "is_fast_pull": candidate.is_fast_pull,
                "is_strong_pull": candidate.is_strong_pull,
                "wall_state": wall.state.value,
            },
        )

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
        if not self.config.enabled or not self.config.pull_detection.enabled:
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
        Аналізує набір tracked walls і повертає позитивні pull candidates.

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
        if not self.config.enabled or not self.config.pull_detection.enabled:
            return []

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

        results: list[DetectorResult] = []

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

            repetition_count = self._estimate_repetition_count(wall)
            result = self.analyze(
                wall,
                current_mid_price=current_mid_price,
                repetition_count=repetition_count,
            )
            if result is not None and result.is_positive():
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

    def is_pull_candidate(self, wall: TrackedWall) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_pull_candidate", _analytics_args)
        except Exception:
            pass
        return self._evaluate_pull_candidate(wall=wall) is not None

    # -------------------------------------------------------------------------
    # Core detection logic
    # -------------------------------------------------------------------------

    def _evaluate_pull_candidate(
        self,
        *,
        wall: TrackedWall,
    ) -> PullCandidateContext | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_evaluate_pull_candidate", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.pull_detection.enabled:
            return None

        if not self.should_process_key(wall.key):
            return None

        if wall.max_size <= 0.0 or wall.price <= 0.0:
            return None

        wall_notional = wall.price * wall.max_size
        if wall_notional < self.config.wall_detection.min_wall_size_abs:
            return None

        lifetime_ms = wall.lifetime_ms
        if lifetime_ms <= 0:
            return None

        pull_ratio = wall.pull_ratio
        fill_ratio = wall.fill_ratio
        pulled_notional = wall.price * wall.estimated_pulled_size
        pulled_size_ratio = self.normalize_ratio(wall.estimated_pulled_size, wall.max_size)

        if not self._passes_basic_filters(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
        ):
            return None

        is_fast_pull = lifetime_ms <= self.config.pull_detection.fast_pull_lifetime_ms
        is_strong_pull = pull_ratio >= self.config.pull_detection.strong_pull_ratio

        score = self._compute_pull_score(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        confidence = self._compute_pull_confidence(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        reason = self._build_reason(
            wall=wall,
            lifetime_ms=lifetime_ms,
            pull_ratio=pull_ratio,
            fill_ratio=fill_ratio,
            pulled_notional=pulled_notional,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
        )

        return PullCandidateContext(
            wall=wall,
            pulled_notional=pulled_notional,
            pulled_size_ratio=pulled_size_ratio,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            lifetime_ms=lifetime_ms,
            is_fast_pull=is_fast_pull,
            is_strong_pull=is_strong_pull,
            confidence=confidence,
            score=score,
            reason=reason,
        )

    def _passes_basic_filters(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_passes_basic_filters", _analytics_args)
        except Exception:
            pass
        cfg = self.config.pull_detection

        if pulled_notional < cfg.min_removed_notional:
            return False

        if pull_ratio < cfg.min_pull_ratio:
            return False

        if fill_ratio > cfg.max_fill_ratio_for_pull:
            return False

        if lifetime_ms > cfg.max_pull_lifetime_ms:
            return False

        if wall.state not in {
            OrderbookWallState.PULLED,
            OrderbookWallState.WEAKENING,
            OrderbookWallState.EXPIRED,
            OrderbookWallState.FILLED,
        }:
            return False

        if wall.state == OrderbookWallState.FILLED and pull_ratio < cfg.strong_pull_ratio:
            return False

        return True

    def _build_features(
        self,
        *,
        candidate: PullCandidateContext,
        current_mid_price: float | None = None,
        repetition_count: int | None = None,
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
        wall = candidate.wall

        distance_from_mid_bps = self._resolve_distance_from_mid_bps(
            wall=wall,
            current_mid_price=current_mid_price,
        )

        cancel_to_fill_ratio = self._compute_cancel_to_fill_ratio(
            pull_ratio=candidate.pull_ratio,
            fill_ratio=candidate.fill_ratio,
        )

        repetition = (
            repetition_count
            if repetition_count is not None
            else self._estimate_repetition_count(wall)
        )

        is_near_best_quote = wall.near_touch_count > 0 or wall.touch_count > 0

        return SpoofingFeatures(
            symbol=wall.symbol,
            exchange=wall.exchange,
            market_type=wall.market_type,
            timeframe=wall.timeframe,
            exchange_symbol=wall.exchange_symbol,
            side=wall.side,
            price=wall.price,
            wall_size=wall.max_size,
            wall_size_ratio=self.normalize_ratio(wall.max_size, max(wall.initial_size, 1e-12)),
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=candidate.lifetime_ms,
            updates_count=wall.updates_count,
            repetition_count=repetition,
            fill_ratio=candidate.fill_ratio,
            pull_ratio=candidate.pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=0.0,
            pressure_flip_strength=0.0,
            layering_score=0.0,
            is_near_best_quote=is_near_best_quote,
            is_fast_pull=candidate.is_fast_pull,
            is_fake_liquidity=False,
            is_layering=False,
            metadata={
                "scope": spoofing_key_to_dict(wall.key),
                "wall_id": wall.wall_id,
                "pulled_notional": candidate.pulled_notional,
                "pulled_size_ratio": candidate.pulled_size_ratio,
                "estimated_pulled_size": wall.estimated_pulled_size,
                "estimated_filled_size": wall.estimated_filled_size,
                "current_to_max_ratio": wall.current_to_max_ratio,
                "wall_state": wall.state.value,
                "detector": self.component.value,
            },
        )

    # -------------------------------------------------------------------------
    # Score / confidence
    # -------------------------------------------------------------------------

    def _compute_pull_score(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> float:
        """
        Первинний score pull candidate в [0, 1].
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_pull_score", _analytics_args)
        except Exception:
            pass
        cfg = self.config.pull_detection

        max_lifetime = max(float(cfg.max_pull_lifetime_ms), 1.0)
        lifetime_component = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)

        min_pull_ratio = max(cfg.min_pull_ratio, 1e-12)
        pull_component = (pull_ratio - min_pull_ratio) / max(1.0 - min_pull_ratio, 1e-12)
        pull_component = self.clamp(pull_component, 0.0, 1.0)

        max_fill = max(cfg.max_fill_ratio_for_pull, 1e-12)
        fill_component = 1.0 - self.clamp(fill_ratio / max_fill, 0.0, 1.0)

        min_removed_notional = max(cfg.min_removed_notional, 1e-12)
        notional_component = (pulled_notional - min_removed_notional) / max(
            min_removed_notional * 2.0,
            1e-12,
        )
        notional_component = self.clamp(notional_component, 0.0, 1.0)

        behavior_bonus = 0.0
        if is_fast_pull:
            behavior_bonus += 0.08
        if is_strong_pull:
            behavior_bonus += 0.08
        if wall.state == OrderbookWallState.PULLED:
            behavior_bonus += 0.06

        raw_score = (
            0.30 * lifetime_component
            + 0.28 * pull_component
            + 0.18 * fill_component
            + 0.16 * notional_component
            + behavior_bonus
        )

        return self.clamp(raw_score, 0.0, 1.0)

    def _compute_pull_confidence(
        self,
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> float:
        """
        Confidence для order pull detection.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_pull_confidence", _analytics_args)
        except Exception:
            pass
        cfg = self.config.pull_detection
        confidence = 0.40

        if is_fast_pull:
            confidence += 0.16

        if is_strong_pull:
            confidence += 0.16

        if fill_ratio <= cfg.max_fill_ratio_for_pull * 0.5:
            confidence += 0.10

        if pulled_notional >= cfg.min_removed_notional * 2.0:
            confidence += 0.08

        if wall.state == OrderbookWallState.PULLED:
            confidence += 0.05

        if wall.near_touch_count == 0 and wall.touch_count == 0:
            confidence += 0.04

        if lifetime_ms <= cfg.fast_pull_lifetime_ms * 0.5:
            confidence += 0.04

        return self.clamp(confidence, 0.0, 0.99)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _estimate_repetition_count(self, wall: TrackedWall) -> int:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_estimate_repetition_count", _analytics_args)
        except Exception:
            pass
        history = self.persistence_tracker.get_recent_history(
            exchange=wall.exchange,
            market_type=wall.market_type,
            symbol=wall.symbol,
            timeframe=wall.timeframe,
            side=wall.side,
            price=wall.price,
            limit=100,
        )
        return len(history)

    def _resolve_distance_from_mid_bps(
        self,
        *,
        wall: TrackedWall,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_distance_from_mid_bps", _analytics_args)
        except Exception:
            pass
        reference_mid = current_mid_price or wall.mid_price_at_creation
        if reference_mid is None or reference_mid <= 0:
            return 0.0
        return self.bps_distance(wall.price, reference_mid)

    @staticmethod
    def _compute_cancel_to_fill_ratio(
        *,
        pull_ratio: float,
        fill_ratio: float,
    ) -> float:
        try:
            _analytics_class_name = "OrderPullDetector"
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

    @staticmethod
    def _build_reason(
        *,
        wall: TrackedWall,
        lifetime_ms: float,
        pull_ratio: float,
        fill_ratio: float,
        pulled_notional: float,
        is_fast_pull: bool,
        is_strong_pull: bool,
    ) -> str:
        try:
            _analytics_class_name = "OrderPullDetector"
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
        parts = [
            f"pull candidate detected for {wall.side.value.upper()} wall",
            f"exchange={wall.exchange}",
            f"market_type={wall.market_type}",
            f"symbol={wall.symbol}",
            f"timeframe={wall.timeframe}",
            f"pulled_notional={pulled_notional:.2f}",
            f"pull_ratio={pull_ratio:.4f}",
            f"fill_ratio={fill_ratio:.4f}",
            f"lifetime_ms={lifetime_ms:.2f}",
            f"state={wall.state.value}",
        ]

        if is_fast_pull:
            parts.append("fast_pull=true")
        if is_strong_pull:
            parts.append("strong_pull=true")

        return ", ".join(parts)


__all__ = ["OrderPullDetector"]