from __future__ import annotations
from core.logger import get_logger

from datetime import datetime, timedelta
from hashlib import sha1
from typing import Iterable

from core.event_bus import EventBus
from core.scheduler import Scheduler

from .base import BaseSpoofingModule
from .config import SpoofingConfig
from .enums import (
    DetectorDecision,
    ScoreComponent,
    SpoofingComponent,
    SpoofingPattern,
    SpoofingSeverity,
    SpoofingStatus,
    SpoofingType,
)
from .models import (
    DEFAULT_MARKET_TYPE,
    DEFAULT_TIMEFRAME,
    AggregationContext,
    DetectorResult,
    ScoreContribution,
    SpoofingFeatures,
    SpoofingKey,
    SpoofingScore,
    SpoofingSignal,
    make_spoofing_key,
    spoofing_key_to_dict,
)


class SpoofingScoreEngine(BaseSpoofingModule):
    """
    Aggregation/scoring engine для analytics.spoofing.

    Відповідає за:
    - фільтрацію позитивних detector results;
    - збирання detector results у єдиний AggregationContext;
    - merge SpoofingFeatures;
    - побудову contribution breakdown;
    - розрахунок фінального spoofing score;
    - визначення severity;
    - побудову фінального SpoofingSignal.

    Correct scope:
        exchange + market_type + symbol + timeframe

    Важливо:
    - не виявляє spoofing самостійно;
    - не підписується на EventBus;
    - не публікує події;
    - не запускає Scheduler jobs;
    - не читає exchange/data cache напряму;
    - працює як pure aggregation/scoring component.
    """

    component = SpoofingComponent.SPOOFING_SCORE

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        scheduler: Scheduler | None,
        config: SpoofingConfig,
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

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def score(
        self,
        detector_results: Iterable[DetectorResult],
        *,
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> SpoofingScore | None:
        """
        Рахує фінальний SpoofingScore на основі detector results.

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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "score", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.scoring.enabled:
            return None

        context = self._build_aggregation_context(
            detector_results=detector_results,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
        )
        if context is None:
            return None

        contributions = self._build_contributions(context)
        total_score = self._compute_total_score(contributions)
        confidence = self._compute_final_confidence(
            context=context,
            total_score=total_score,
        )
        severity = self._resolve_severity(total_score)
        threshold = self.config.scoring.detection_threshold
        passed = total_score >= threshold

        return SpoofingScore(
            total_score=total_score,
            confidence=confidence,
            severity=severity,
            contributions=contributions,
            threshold=threshold,
            passed=passed,
            metadata={
                "scope": spoofing_key_to_dict(context.key),
                "exchange": context.exchange,
                "market_type": context.market_type,
                "symbol": context.symbol,
                "timeframe": context.timeframe,
                "exchange_symbol": context.exchange_symbol,
                "agreement_ratio": context.agreement_ratio,
                "average_confidence": context.average_confidence,
                "primary_pattern": context.primary_pattern.value,
                "spoofing_type": context.spoofing_type.value,
                "detector_count": len(context.detector_results),
                "wall_id": context.wall_id,
            },
        )

    def build_signal(
        self,
        detector_results: Iterable[DetectorResult],
        *,
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
        status: SpoofingStatus = SpoofingStatus.DETECTED,
    ) -> SpoofingSignal | None:
        """
        Будує фінальний SpoofingSignal із detector results.

        New code should pass key=SpoofingKey.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "build_signal", _analytics_args)
        except Exception:
            pass
        if not self.config.enabled or not self.config.scoring.enabled:
            return None

        context = self._build_aggregation_context(
            detector_results=detector_results,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
        )
        if context is None:
            return None

        score = self.score(
            detector_results=context.detector_results,
            key=context.key,
        )
        if score is None:
            return None

        signal_id = self._build_signal_id(
            key=context.key,
            wall_id=context.wall_id,
            spoofing_type=context.spoofing_type,
            pattern=context.primary_pattern,
            price=context.price,
        )

        detected_at = self.now()
        first_seen_at = self._resolve_first_seen_at(
            detected_at=detected_at,
            features=context.features,
        )

        return SpoofingSignal(
            signal_id=signal_id,
            symbol=context.symbol,
            exchange=context.exchange,
            market_type=context.market_type,
            timeframe=context.timeframe,
            exchange_symbol=context.exchange_symbol,
            side=context.features.side,
            spoofing_type=context.spoofing_type,
            pattern=context.primary_pattern,
            status=status,
            price_level=context.price,
            wall_id=context.wall_id,
            score=score.total_score,
            confidence=score.confidence,
            severity=score.severity,
            first_seen_at=first_seen_at,
            detected_at=detected_at,
            features=context.features,
            detector_results=context.detector_results,
            score_breakdown=score,
            metadata={
                "scope": spoofing_key_to_dict(context.key),
                "agreement_ratio": context.agreement_ratio,
                "average_confidence": context.average_confidence,
                "detector_count": len(context.detector_results),
                "passed": score.passed,
                "threshold": score.threshold,
            },
        )

    def should_emit_detection(
        self,
        detector_results: Iterable[DetectorResult],
        *,
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> bool:
        """
        Перевіряє, чи достатній score для detection event.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "should_emit_detection", _analytics_args)
        except Exception:
            pass
        score = self.score(
            detector_results=detector_results,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
        )
        return score.passed if score is not None else False

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def _build_aggregation_context(
        self,
        *,
        detector_results: Iterable[DetectorResult],
        key: SpoofingKey | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
        market_type: str | None = None,
        timeframe: str | None = None,
    ) -> AggregationContext | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_aggregation_context", _analytics_args)
        except Exception:
            pass
        results = self._filter_positive_results(
            detector_results=detector_results,
            key=key,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
        )
        if not results:
            return None

        features = self._merge_features(results)
        if features is None:
            return None

        if not self.should_process_key(features.key):
            return None

        agreement_ratio = self._compute_agreement_ratio(results)
        average_confidence = self._compute_average_confidence(results)
        primary_pattern = self._resolve_primary_pattern(results)
        spoofing_type = self._resolve_spoofing_type(
            results=results,
            primary_pattern=primary_pattern,
        )
        wall_id = self._resolve_wall_id(results)

        return AggregationContext(
            symbol=features.symbol,
            exchange=features.exchange,
            market_type=features.market_type,
            timeframe=features.timeframe,
            exchange_symbol=features.exchange_symbol,
            price=features.price,
            features=features,
            detector_results=results,
            agreement_ratio=agreement_ratio,
            average_confidence=average_confidence,
            primary_pattern=primary_pattern,
            spoofing_type=spoofing_type,
            wall_id=wall_id,
        )

    def _filter_positive_results(
        self,
        *,
        detector_results: Iterable[DetectorResult],
        key: SpoofingKey | None,
        symbol: str | None,
        exchange: str | None,
        market_type: str | None,
        timeframe: str | None,
    ) -> list[DetectorResult]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_filter_positive_results", _analytics_args)
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

        results: list[DetectorResult] = []

        for result in detector_results:
            if result is None:
                continue
            if result.decision != DetectorDecision.POSITIVE:
                continue
            if result.features is None:
                continue

            feature_key = result.features.key

            if key is not None and feature_key != key:
                continue
            if normalized_exchange is not None and result.features.exchange != normalized_exchange:
                continue
            if normalized_market_type is not None and result.features.market_type != normalized_market_type:
                continue
            if normalized_symbol is not None and result.features.symbol != normalized_symbol:
                continue
            if normalized_timeframe is not None and result.features.timeframe != normalized_timeframe:
                continue
            if not self.should_process_key(feature_key):
                continue

            results.append(result)

        return results

    def _merge_features(
        self,
        results: list[DetectorResult],
    ) -> SpoofingFeatures | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_merge_features", _analytics_args)
        except Exception:
            pass
        feature_list = [
            result.features
            for result in results
            if result.features is not None
        ]
        if not feature_list:
            return None

        base = feature_list[0]
        base_key = base.key

        # Safety: scoring має агрегувати тільки один scoped futures market.
        scoped_features = [
            item
            for item in feature_list
            if item.key == base_key
        ]
        if not scoped_features:
            return None

        wall_size = max((item.wall_size for item in scoped_features), default=0.0)
        wall_size_ratio = max((item.wall_size_ratio for item in scoped_features), default=0.0)

        distance_values = [
            item.distance_from_mid_bps
            for item in scoped_features
            if item.distance_from_mid_bps >= 0
        ]
        distance_from_mid_bps = min(distance_values) if distance_values else 0.0

        lifetime_ms = max((item.lifetime_ms for item in scoped_features), default=0.0)
        updates_count = max((item.updates_count for item in scoped_features), default=0)
        repetition_count = max((item.repetition_count for item in scoped_features), default=0)

        fill_ratio = max((item.fill_ratio for item in scoped_features), default=0.0)
        pull_ratio = max((item.pull_ratio for item in scoped_features), default=0.0)
        cancel_to_fill_ratio = max(
            (item.cancel_to_fill_ratio for item in scoped_features),
            default=0.0,
        )

        price_reaction_bps = max(
            (item.price_reaction_bps for item in scoped_features),
            default=0.0,
        )
        pressure_flip_strength = max(
            (item.pressure_flip_strength for item in scoped_features),
            default=0.0,
        )
        layering_score = max(
            (item.layering_score for item in scoped_features),
            default=0.0,
        )

        merged_metadata: dict[str, object] = {}
        detector_names: list[str] = []

        for result in results:
            if result.features is None or result.features.key != base_key:
                continue
            detector_names.append(result.detector.value)

        for item in scoped_features:
            if item.metadata:
                merged_metadata.update(item.metadata)

        merged_metadata["scope"] = spoofing_key_to_dict(base_key)
        merged_metadata["detectors"] = detector_names

        return SpoofingFeatures(
            symbol=base.symbol,
            exchange=base.exchange,
            market_type=base.market_type,
            timeframe=base.timeframe,
            exchange_symbol=base.exchange_symbol,
            side=base.side,
            price=base.price,
            wall_size=wall_size,
            wall_size_ratio=wall_size_ratio,
            distance_from_mid_bps=distance_from_mid_bps,
            lifetime_ms=lifetime_ms,
            updates_count=updates_count,
            repetition_count=repetition_count,
            fill_ratio=fill_ratio,
            pull_ratio=pull_ratio,
            cancel_to_fill_ratio=cancel_to_fill_ratio,
            price_reaction_bps=price_reaction_bps,
            pressure_flip_strength=pressure_flip_strength,
            layering_score=layering_score,
            is_near_best_quote=any(item.is_near_best_quote for item in scoped_features),
            is_fast_pull=any(item.is_fast_pull for item in scoped_features),
            is_fake_liquidity=any(item.is_fake_liquidity for item in scoped_features),
            is_layering=any(item.is_layering for item in scoped_features),
            metadata=merged_metadata,
        )

    def _compute_agreement_ratio(self, results: list[DetectorResult]) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_agreement_ratio", _analytics_args)
        except Exception:
            pass
        if not results:
            return 0.0

        detector_count = len({result.detector for result in results})
        return self.clamp(detector_count / max(len(SpoofingComponent), 1), 0.0, 1.0)

    def _compute_average_confidence(self, results: list[DetectorResult]) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_average_confidence", _analytics_args)
        except Exception:
            pass
        if not results:
            return 0.0

        value = sum(result.confidence for result in results) / len(results)
        return self.clamp(value, 0.0, 1.0)

    @staticmethod
    def _resolve_primary_pattern(results: list[DetectorResult]) -> SpoofingPattern:
        try:
            _analytics_class_name = "SpoofingScoreEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_primary_pattern", _analytics_args)
        except Exception:
            pass
        if not results:
            return SpoofingPattern.UNKNOWN

        weights: dict[SpoofingPattern, float] = {}
        for result in results:
            weight = max(result.score, 0.0) * max(result.confidence, 0.0)
            weights[result.pattern] = weights.get(result.pattern, 0.0) + weight

        if not weights:
            return SpoofingPattern.UNKNOWN

        return max(weights.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _resolve_spoofing_type(
        *,
        results: list[DetectorResult],
        primary_pattern: SpoofingPattern,
    ) -> SpoofingType:
        try:
            _analytics_class_name = "SpoofingScoreEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_spoofing_type", _analytics_args)
        except Exception:
            pass
        detector_names = {result.detector for result in results}

        if len(detector_names) >= 2:
            return SpoofingType.COMPOSITE

        if primary_pattern == SpoofingPattern.PULL_AND_REVERSAL:
            return SpoofingType.ORDER_PULL
        if primary_pattern == SpoofingPattern.MULTI_LEVEL_LAYERING:
            return SpoofingType.LAYERING
        if primary_pattern == SpoofingPattern.PRESSURE_BLUFF:
            return SpoofingType.FLIP_PRESSURE
        if primary_pattern == SpoofingPattern.FAKE_ABSORPTION:
            return SpoofingType.FAKE_LIQUIDITY
        if primary_pattern == SpoofingPattern.SINGLE_LEVEL_SPOOF:
            return SpoofingType.FAKE_WALL

        return SpoofingType.UNKNOWN

    @staticmethod
    def _resolve_wall_id(results: list[DetectorResult]) -> str | None:
        try:
            _analytics_class_name = "SpoofingScoreEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_wall_id", _analytics_args)
        except Exception:
            pass
        wall_ids = [result.wall_id for result in results if result.wall_id]
        if not wall_ids:
            return None

        counts: dict[str, int] = {}
        for wall_id in wall_ids:
            counts[wall_id] = counts.get(wall_id, 0) + 1

        return max(counts.items(), key=lambda item: item[1])[0]

    # -------------------------------------------------------------------------
    # Contributions
    # -------------------------------------------------------------------------

    def _build_contributions(
        self,
        context: AggregationContext,
    ) -> list[ScoreContribution]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_contributions", _analytics_args)
        except Exception:
            pass
        features = context.features
        cfg = self.config.scoring

        return [
            self._make_contribution(
                component=ScoreComponent.WALL_SIZE,
                raw_value=features.wall_size_ratio,
                normalized_value=self._normalize_wall_size(features.wall_size_ratio),
                weight=cfg.weight_wall_size,
                description="Relative wall size vs local baseline",
            ),
            self._make_contribution(
                component=ScoreComponent.WALL_DISTANCE,
                raw_value=features.distance_from_mid_bps,
                normalized_value=self._normalize_wall_distance(features.distance_from_mid_bps),
                weight=cfg.weight_wall_distance,
                description="Wall proximity to current market",
            ),
            self._make_contribution(
                component=ScoreComponent.PERSISTENCE,
                raw_value=features.lifetime_ms,
                normalized_value=self._normalize_persistence(features.lifetime_ms),
                weight=cfg.weight_persistence,
                description="Wall lifetime profile",
            ),
            self._make_contribution(
                component=ScoreComponent.PULL_SPEED,
                raw_value=features.pull_ratio,
                normalized_value=self._normalize_pull_speed(
                    pull_ratio=features.pull_ratio,
                    lifetime_ms=features.lifetime_ms,
                    is_fast_pull=features.is_fast_pull,
                ),
                weight=cfg.weight_pull_speed,
                description="Fast liquidity removal behavior",
            ),
            self._make_contribution(
                component=ScoreComponent.FILL_RATIO,
                raw_value=features.fill_ratio,
                normalized_value=self._normalize_fill_ratio(
                    fill_ratio=features.fill_ratio,
                    cancel_to_fill_ratio=features.cancel_to_fill_ratio,
                ),
                weight=cfg.weight_fill_ratio,
                description="Low execution / high cancel profile",
            ),
            self._make_contribution(
                component=ScoreComponent.PRICE_REACTION,
                raw_value=max(
                    features.price_reaction_bps,
                    features.pressure_flip_strength,
                ),
                normalized_value=self._normalize_price_reaction(
                    price_reaction_bps=features.price_reaction_bps,
                    pressure_flip_strength=features.pressure_flip_strength,
                ),
                weight=cfg.weight_price_reaction,
                description="Market reaction after liquidity behavior",
            ),
            self._make_contribution(
                component=ScoreComponent.REPETITION,
                raw_value=float(features.repetition_count),
                normalized_value=self._normalize_repetition(features.repetition_count),
                weight=cfg.weight_repetition,
                description="Repeated appearance / manipulation pattern recurrence",
            ),
            self._make_contribution(
                component=ScoreComponent.LAYERING,
                raw_value=features.layering_score,
                normalized_value=self._normalize_layering(
                    layering_score=features.layering_score,
                    is_layering=features.is_layering,
                ),
                weight=cfg.weight_layering,
                description="Multi-level spoofing structure",
            ),
        ]

    def _make_contribution(
        self,
        *,
        component: ScoreComponent,
        raw_value: float,
        normalized_value: float,
        weight: float,
        description: str,
    ) -> ScoreContribution:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_make_contribution", _analytics_args)
        except Exception:
            pass
        normalized = self.clamp(normalized_value, 0.0, 1.0)
        safe_weight = max(weight, 0.0)
        contribution = normalized * safe_weight

        return ScoreContribution(
            component=component,
            raw_value=raw_value,
            normalized_value=normalized,
            weight=safe_weight,
            contribution=contribution,
            description=description,
        )

    def _compute_total_score(
        self,
        contributions: list[ScoreContribution],
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_total_score", _analytics_args)
        except Exception:
            pass
        total_weight = sum(item.weight for item in contributions)
        if total_weight <= 0:
            return 0.0

        weighted_sum = sum(item.contribution for item in contributions)
        return self.clamp(weighted_sum / total_weight, 0.0, 1.0)

    def _compute_final_confidence(
        self,
        *,
        context: AggregationContext,
        total_score: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_compute_final_confidence", _analytics_args)
        except Exception:
            pass
        cfg = self.config.scoring

        confidence = max(context.average_confidence, cfg.min_confidence)

        if context.agreement_ratio >= 0.75:
            confidence += cfg.confidence_boost_on_detector_agreement

        if total_score >= cfg.high_severity_threshold:
            confidence += 0.05

        if total_score >= cfg.critical_severity_threshold:
            confidence += 0.03

        return self.clamp(confidence, 0.0, cfg.max_confidence)

    def _resolve_severity(self, total_score: float) -> SpoofingSeverity:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_severity", _analytics_args)
        except Exception:
            pass
        cfg = self.config.scoring

        if total_score >= cfg.critical_severity_threshold:
            return SpoofingSeverity.CRITICAL
        if total_score >= cfg.high_severity_threshold:
            return SpoofingSeverity.HIGH
        if total_score >= cfg.detection_threshold:
            return SpoofingSeverity.MEDIUM
        return SpoofingSeverity.LOW

    # -------------------------------------------------------------------------
    # Normalization helpers
    # -------------------------------------------------------------------------

    def _normalize_wall_size(self, wall_size_ratio: float) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_wall_size", _analytics_args)
        except Exception:
            pass
        min_ratio = max(self.config.wall_detection.min_wall_size_ratio, 1e-12)
        value = (wall_size_ratio - min_ratio) / max(min_ratio * 2.0, 1e-12)
        return self.clamp(value, 0.0, 1.0)

    def _normalize_wall_distance(self, distance_from_mid_bps: float) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_wall_distance", _analytics_args)
        except Exception:
            pass
        max_distance = max(self.config.wall_detection.max_distance_from_mid_bps, 1e-12)
        value = 1.0 - self.clamp(distance_from_mid_bps / max_distance, 0.0, 1.0)
        return self.clamp(value, 0.0, 1.0)

    def _normalize_persistence(self, lifetime_ms: float) -> float:
        """
        Для spoofing короткий lifetime зазвичай підозріліший.
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_persistence", _analytics_args)
        except Exception:
            pass
        max_lifetime = max(float(self.config.pull_detection.max_pull_lifetime_ms), 1.0)
        value = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)
        return self.clamp(value, 0.0, 1.0)

    def _normalize_pull_speed(
        self,
        *,
        pull_ratio: float,
        lifetime_ms: float,
        is_fast_pull: bool,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_pull_speed", _analytics_args)
        except Exception:
            pass
        ratio_part = self.clamp(pull_ratio, 0.0, 1.0)

        max_lifetime = max(float(self.config.pull_detection.max_pull_lifetime_ms), 1.0)
        lifetime_part = 1.0 - self.clamp(lifetime_ms / max_lifetime, 0.0, 1.0)

        bonus = 0.10 if is_fast_pull else 0.0
        value = 0.65 * ratio_part + 0.35 * lifetime_part + bonus
        return self.clamp(value, 0.0, 1.0)

    def _normalize_fill_ratio(
        self,
        *,
        fill_ratio: float,
        cancel_to_fill_ratio: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_fill_ratio", _analytics_args)
        except Exception:
            pass
        fill_component = 1.0 - self.clamp(fill_ratio, 0.0, 1.0)
        ratio_component = self.clamp(cancel_to_fill_ratio / 3.0, 0.0, 1.0)
        value = 0.60 * fill_component + 0.40 * ratio_component
        return self.clamp(value, 0.0, 1.0)

    def _normalize_price_reaction(
        self,
        *,
        price_reaction_bps: float,
        pressure_flip_strength: float,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_price_reaction", _analytics_args)
        except Exception:
            pass
        reaction_part = self.clamp(price_reaction_bps / 10.0, 0.0, 1.0)
        pressure_part = self.clamp(pressure_flip_strength, 0.0, 1.0)
        return self.clamp(max(reaction_part, pressure_part), 0.0, 1.0)

    def _normalize_repetition(self, repetition_count: int) -> float:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_repetition", _analytics_args)
        except Exception:
            pass
        if repetition_count <= 0:
            return 0.0

        return self.clamp(repetition_count / 10.0, 0.0, 1.0)

    def _normalize_layering(
        self,
        *,
        layering_score: float,
        is_layering: bool,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_normalize_layering", _analytics_args)
        except Exception:
            pass
        base = self.clamp(layering_score, 0.0, 1.0)

        if is_layering and base < 0.5:
            base = 0.5

        return self.clamp(base, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Signal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_first_seen_at(
        *,
        detected_at: datetime,
        features: SpoofingFeatures,
    ) -> datetime:
        try:
            _analytics_class_name = "SpoofingScoreEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_first_seen_at", _analytics_args)
        except Exception:
            pass
        if features.lifetime_ms <= 0:
            return detected_at

        try:
            return detected_at - timedelta(milliseconds=features.lifetime_ms)
        except Exception:
            return detected_at

    @staticmethod
    def _build_signal_id(
        *,
        key: SpoofingKey,
        wall_id: str | None,
        spoofing_type: SpoofingType,
        pattern: SpoofingPattern,
        price: float,
    ) -> str:
        try:
            _analytics_class_name = "SpoofingScoreEngine"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_signal_id", _analytics_args)
        except Exception:
            pass
        scope = spoofing_key_to_dict(key)
        raw = (
            f"{scope['exchange']}|{scope['market_type']}|"
            f"{scope['symbol']}|{scope['timeframe']}|"
            f"{wall_id or 'none'}|"
            f"{spoofing_type.value}|{pattern.value}|{price:.12f}"
        )
        digest = sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"spoof-{digest}"


__all__ = ["SpoofingScoreEngine"]