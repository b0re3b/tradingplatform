from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from analytics.open_interest.enums import (
    OIAnomalyType,
    OIDivergenceType,
    OIRegime,
)
from analytics.open_interest.models import (
    OIAnalysisResult,
    OIAnomalyResult,
    OIDivergenceResult,
    OIFeatures,
    OIMarketContext,
    OIRegimeResult,
    OISnapshot,
)
from core.logger import get_logger
from strategy.base import ContextAwareComponent, NamedEntityMixin, PrioritizedMixin
from strategy.config import StrategyConfig
from strategy.enums import (
    ConfidenceGrade,
    MarketRegime,
    SignalPriority,
    SignalStrength,
    StrategyCategory,
)
from strategy.models import SignalContext, StrategySignal


@dataclass(slots=True)
class OIBasePayload:
    """
    Універсальний контейнер для open-interest strategy extraction.

    Це не replacement для domain-specific payload-ів окремих стратегій.
    Він потрібен як спільний normalized view над analytics.open_interest payload.
    """

    analysis: OIAnalysisResult | None = None
    snapshot: OISnapshot | None = None
    market_context: OIMarketContext | None = None
    features: OIFeatures | None = None
    regime: OIRegimeResult | None = None
    divergence: OIDivergenceResult | None = None
    anomaly: OIAnomalyResult | None = None

    scope: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class OpenInterestStrategyBase(
    ContextAwareComponent,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Base class для strategy/strategies/open_interest.

    Роль цього класу:
    - читати canonical payload від analytics.open_interest;
    - підтримувати specialized OI events як fallback;
    - централізувати extraction для analysis/features/regime/divergence/anomaly;
    - не рахувати аналітику повторно;
    - не створювати EventBus/Scheduler напряму;
    - залишити backward-compatible fallback на старі feature-map keys.

    Canonical input:
        context.open_interest = OIAnalysisResult.to_dict()

    Supported fallback inputs:
        - analytics.oi.regime_changed payload;
        - analytics.oi.divergence payload;
        - analytics.oi.anomaly payload;
        - analytics.oi.squeeze_setup payload;
        - analytics.oi.capitulation payload;
        - old context feature keys: oi.* / open_interest.*.
    """

    STRATEGY_NAME: str = "open_interest_base_strategy"
    DEFAULT_PRIORITY: int = 80
    REQUIRED_FEATURES: set[str] = set()

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: Any | None = None,
    ) -> None:
        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=get_logger(
                __name__,
                service_name="strategy",
                event_type=self.STRATEGY_NAME,
            ),
        )
        self.validate_config()

    @property
    def priority(self) -> int:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is not None:
            return strategy_cfg.priority
        return self.DEFAULT_PRIORITY

    # ------------------------------------------------------------------
    # Common validation helpers
    # ------------------------------------------------------------------

    def is_strategy_runtime_allowed(
        self,
        context: SignalContext,
    ) -> tuple[bool, str | None]:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is None:
            return False, "strategy_config_not_found"

        if not strategy_cfg.runtime.enabled:
            return False, "strategy_disabled"

        if strategy_cfg.runtime.symbols and context.symbol not in strategy_cfg.runtime.symbols:
            return False, "symbol_not_enabled_for_strategy"

        if strategy_cfg.runtime.timeframes and context.timeframe not in strategy_cfg.runtime.timeframes:
            return False, "timeframe_not_enabled_for_strategy"

        return True, None

    def is_market_regime_allowed(
        self,
        context: SignalContext,
    ) -> tuple[bool, str | None]:
        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        if strategy_cfg is None:
            return False, "strategy_config_not_found"

        regime = self.get_market_regime(context)
        if strategy_cfg.runtime.allowed_regimes and regime not in strategy_cfg.runtime.allowed_regimes:
            return False, "regime_not_allowed"

        return True, None

    def context_has_any_oi_data(
        self,
        context: SignalContext,
        keys: tuple[str, ...] = (),
    ) -> bool:
        if self.extract_oi_domain(context):
            return True

        if self.extract_oi_analysis(context) is not None:
            return True

        if self.extract_oi_features(context) is not None:
            return True

        return any(context.has_feature(name) for name in keys)

    def has_stale_required_features(self, context: SignalContext) -> bool:
        """
        Backward-compatible stale check.

        Для нового canonical context.open_interest payload freshness має
        контролювати StrategyContextBuilder / SignalContext creation layer.
        Тут лишаємо перевірку старих feature-map keys.
        """
        for name in self.REQUIRED_FEATURES:
            if context.has_feature(name) and context.feature_is_stale(name):
                return True
        return False

    # ------------------------------------------------------------------
    # Canonical OI domain extraction
    # ------------------------------------------------------------------

    def extract_oi_domain(self, context: SignalContext) -> dict[str, Any]:
        """
        Повертає raw open_interest domain payload із SignalContext.

        Expected canonical shape:
            OIAnalysisResult.to_dict()

        Але метод також приймає будь-який dict-подібний payload від
        specialized analytics.oi.* events.
        """
        domain = getattr(context, "open_interest", None)

        if isinstance(domain, OIAnalysisResult):
            return domain.to_dict()

        if isinstance(domain, Mapping):
            return dict(domain)

        return {}

    def extract_base_payload(self, context: SignalContext) -> OIBasePayload:
        """
        Unified normalized view для дочірніх OI-стратегій.

        Окремі стратегії можуть використовувати цей метод як стартову точку,
        а потім будувати власний domain-specific payload.
        """
        raw = self.extract_oi_domain(context)
        analysis = self.extract_oi_analysis(context)
        features = self.extract_oi_features(context)
        regime = self.extract_oi_regime_result(context)
        divergence = self.extract_oi_divergence_result(context)
        anomaly = self.extract_oi_anomaly_result(context)
        snapshot = self.extract_oi_snapshot(context)
        market_context = self.extract_oi_market_context(context)

        reasons: list[str] = []
        if regime is not None:
            reasons.extend(regime.reasons)
        if divergence is not None:
            reasons.extend(divergence.reasons)
        if anomaly is not None:
            reasons.extend(anomaly.reasons)

        return OIBasePayload(
            analysis=analysis,
            snapshot=snapshot,
            market_context=market_context,
            features=features,
            regime=regime,
            divergence=divergence,
            anomaly=anomaly,
            scope=self.extract_oi_scope(context),
            reasons=list(dict.fromkeys(reasons)),
            raw=raw,
        )

    def extract_oi_analysis(self, context: SignalContext) -> OIAnalysisResult | None:
        """
        Extract full OIAnalysisResult.

        Працює тільки коли context.open_interest містить повний
        analytics.oi.updated payload.
        """
        domain = getattr(context, "open_interest", None)

        if isinstance(domain, OIAnalysisResult):
            return domain

        if not isinstance(domain, Mapping):
            return None

        data = dict(domain)
        required = {"symbol", "timestamp", "snapshot", "context", "features", "regime"}
        if not required.issubset(data.keys()):
            return None

        try:
            return OIAnalysisResult.from_dict(data)
        except Exception as exc:
            self.logger.warning(
                "Failed to parse OIAnalysisResult",
                extra={
                    "strategy": self.STRATEGY_NAME,
                    "symbol": getattr(context, "symbol", None),
                    "error": repr(exc),
                },
            )
            return None

    def extract_oi_snapshot(self, context: SignalContext) -> OISnapshot | None:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.snapshot

        raw = self.extract_oi_domain(context).get("snapshot")
        if isinstance(raw, OISnapshot):
            return raw

        if isinstance(raw, Mapping):
            try:
                return OISnapshot.from_dict(dict(raw))
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse OISnapshot",
                    extra={
                        "strategy": self.STRATEGY_NAME,
                        "symbol": getattr(context, "symbol", None),
                        "error": repr(exc),
                    },
                )

        return None

    def extract_oi_market_context(
        self,
        context: SignalContext,
    ) -> OIMarketContext | None:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.context

        raw = self.extract_oi_domain(context).get("context")
        if isinstance(raw, OIMarketContext):
            return raw

        if isinstance(raw, Mapping):
            try:
                return OIMarketContext.from_dict(dict(raw))
            except Exception as exc:
                self.logger.debug(
                    "Failed to parse OIMarketContext",
                    extra={
                        "strategy": self.STRATEGY_NAME,
                        "symbol": getattr(context, "symbol", None),
                        "error": repr(exc),
                    },
                )

        return None

    def extract_oi_features(self, context: SignalContext) -> OIFeatures | None:
        """
        Extract OIFeatures from:
        1. full OIAnalysisResult;
        2. context.open_interest["features"];
        3. legacy context feature-map values.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.features

        domain = self.extract_oi_domain(context)
        raw = domain.get("features")

        if isinstance(raw, OIFeatures):
            return raw

        if isinstance(raw, Mapping):
            data = dict(raw)
            try:
                return OIFeatures.from_dict(data)
            except Exception:
                try:
                    return OIFeatures(**data)
                except Exception as exc:
                    self.logger.warning(
                        "Failed to parse OIFeatures",
                        extra={
                            "strategy": self.STRATEGY_NAME,
                            "symbol": getattr(context, "symbol", None),
                            "error": repr(exc),
                        },
                    )

        legacy = self._extract_legacy_feature_payload(context)
        if legacy:
            try:
                return OIFeatures.from_dict(legacy)
            except Exception:
                try:
                    return OIFeatures(**legacy)
                except Exception:
                    return None

        return None

    def extract_oi_regime_result(
        self,
        context: SignalContext,
    ) -> OIRegimeResult | None:
        """
        Extract OIRegimeResult from:
        - analysis.regime;
        - context.open_interest["regime"];
        - regime_changed specialized payload;
        - old feature keys.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.regime

        domain = self.extract_oi_domain(context)
        raw = domain.get("regime")

        if isinstance(raw, OIRegimeResult):
            return raw

        if isinstance(raw, Mapping):
            data = dict(raw)
            regime_value = (
                data.get("regime")
                or data.get("type")
                or data.get("regime_type")
                or data.get("new_regime")
            )
            if regime_value is not None:
                try:
                    return OIRegimeResult.from_dict(
                        {
                            "regime": regime_value,
                            "confidence": data.get("confidence", 0.0),
                            "score": data.get("score"),
                            "reasons": list(data.get("reasons") or []),
                        }
                    )
                except Exception:
                    pass

        flat_regime = (
            domain.get("new_regime")
            or domain.get("regime_type")
            or domain.get("oi_regime_type")
            or domain.get("oi_regime")
        )
        if flat_regime is not None:
            try:
                return OIRegimeResult.from_dict(
                    {
                        "regime": flat_regime,
                        "confidence": (
                            domain.get("regime_confidence")
                            if domain.get("regime_confidence") is not None
                            else domain.get("confidence", 0.0)
                        ),
                        "score": domain.get("regime_score", domain.get("score")),
                        "reasons": list(domain.get("reasons") or domain.get("regime_reasons") or []),
                    }
                )
            except Exception:
                pass

        legacy_regime = (
            self.get_feature_value(context, "oi.regime.type")
            or self.get_feature_value(context, "open_interest.regime.type")
        )
        if legacy_regime is not None:
            try:
                return OIRegimeResult.from_dict(
                    {
                        "regime": legacy_regime,
                        "confidence": self.get_feature_value(
                            context,
                            "oi.regime.confidence",
                            default=0.0,
                        ),
                        "score": self.get_feature_value(
                            context,
                            "oi.regime.score",
                            default=None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.get_feature_value(
                                context,
                                "oi.regime.reasons",
                                default=[],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def extract_oi_divergence_result(
        self,
        context: SignalContext,
    ) -> OIDivergenceResult | None:
        """
        Extract OIDivergenceResult from:
        - analysis.divergence;
        - context.open_interest["divergence"];
        - divergence specialized payload;
        - old feature keys.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.divergence

        domain = self.extract_oi_domain(context)
        raw = domain.get("divergence")

        if isinstance(raw, OIDivergenceResult):
            return raw

        if isinstance(raw, Mapping):
            data = self._normalize_divergence_payload(dict(raw))
            try:
                return OIDivergenceResult.from_dict(data)
            except Exception:
                pass

        flat_type = (
            domain.get("divergence_type")
            or domain.get("oi_divergence_type")
            or domain.get("open_interest_divergence_type")
            or domain.get("type")
        )
        if flat_type is not None:
            data = self._normalize_divergence_payload(domain)
            try:
                return OIDivergenceResult.from_dict(data)
            except Exception:
                pass

        legacy_type = (
            self.get_feature_value(context, "oi.divergence.type")
            or self.get_feature_value(context, "open_interest.divergence.type")
        )
        if legacy_type is not None:
            try:
                return OIDivergenceResult.from_dict(
                    {
                        "detected": self.safe_bool(
                            self.get_feature_value(
                                context,
                                "oi.divergence.detected",
                                default=legacy_type != OIDivergenceType.NONE.value,
                            )
                        ),
                        "divergence_type": legacy_type,
                        "confidence": self.get_feature_value(
                            context,
                            "oi.divergence.confidence",
                            default=0.0,
                        ),
                        "score": self.get_feature_value(
                            context,
                            "oi.divergence.score",
                            default=None,
                        ),
                        "window_size": self.get_feature_value(
                            context,
                            "oi.divergence.window_size",
                            default=None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.get_feature_value(
                                context,
                                "oi.divergence.reasons",
                                default=[],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def extract_oi_anomaly_result(
        self,
        context: SignalContext,
    ) -> OIAnomalyResult | None:
        """
        Extract OIAnomalyResult from:
        - analysis.anomaly;
        - context.open_interest["anomaly"];
        - anomaly/capitulation specialized payload;
        - old feature keys.
        """
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.anomaly

        domain = self.extract_oi_domain(context)
        raw = domain.get("anomaly")

        if isinstance(raw, OIAnomalyResult):
            return raw

        if isinstance(raw, Mapping):
            data = self._normalize_anomaly_payload(dict(raw))
            try:
                return OIAnomalyResult.from_dict(data)
            except Exception:
                pass

        flat_type = (
            domain.get("anomaly_type")
            or domain.get("oi_anomaly_type")
            or domain.get("open_interest_anomaly_type")
        )
        if flat_type is not None:
            data = self._normalize_anomaly_payload(domain)
            try:
                return OIAnomalyResult.from_dict(data)
            except Exception:
                pass

        legacy_type = (
            self.get_feature_value(context, "oi.anomaly.type")
            or self.get_feature_value(context, "open_interest.anomaly.type")
        )
        if legacy_type is not None:
            try:
                return OIAnomalyResult.from_dict(
                    {
                        "detected": self.safe_bool(
                            self.get_feature_value(
                                context,
                                "oi.anomaly.detected",
                                default=legacy_type != OIAnomalyType.NONE.value,
                            )
                        ),
                        "anomaly_type": legacy_type,
                        "strength": self.get_feature_value(
                            context,
                            "oi.anomaly.strength",
                            default=None,
                        ),
                        "confidence": self.get_feature_value(
                            context,
                            "oi.anomaly.confidence",
                            default=0.0,
                        ),
                        "score": self.get_feature_value(
                            context,
                            "oi.anomaly.score",
                            default=None,
                        ),
                        "reasons": self.normalize_reasons(
                            self.get_feature_value(
                                context,
                                "oi.anomaly.reasons",
                                default=[],
                            )
                        ),
                    }
                )
            except Exception:
                return None

        return None

    def extract_oi_scope(self, context: SignalContext) -> dict[str, Any]:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.scope_payload()

        domain = self.extract_oi_domain(context)
        scope = domain.get("scope")
        result: dict[str, Any] = dict(scope) if isinstance(scope, Mapping) else {}

        for key in (
            "exchange",
            "market_type",
            "symbol",
            "timeframe",
            "exchange_symbol",
            "scope_key",
            "oi_key",
            "key",
        ):
            if key in domain and key not in result:
                result[key] = domain[key]

        result.setdefault("symbol", getattr(context, "symbol", None))
        result.setdefault("timeframe", getattr(context, "timeframe", None))

        return {key: value for key, value in result.items() if value is not None}

    # ------------------------------------------------------------------
    # Specialized OI context predicates
    # ------------------------------------------------------------------

    def has_oi_analysis(self, context: SignalContext) -> bool:
        return self.extract_oi_analysis(context) is not None

    def has_oi_divergence(self, context: SignalContext) -> bool:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.has_divergence

        divergence = self.extract_oi_divergence_result(context)
        return divergence is not None and divergence.detected

    def has_oi_anomaly(self, context: SignalContext) -> bool:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return analysis.has_anomaly

        anomaly = self.extract_oi_anomaly_result(context)
        return anomaly is not None and anomaly.detected

    def has_risk_anomaly(self, context: SignalContext) -> bool:
        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is None or not anomaly.detected:
            return False
        return anomaly.anomaly_type.is_risk_anomaly

    def is_squeeze_context(self, context: SignalContext) -> bool:
        regime = self.extract_oi_regime_result(context)
        return regime is not None and regime.regime is OIRegime.SQUEEZE_SETUP

    def is_capitulation_context(self, context: SignalContext) -> bool:
        regime = self.extract_oi_regime_result(context)
        if regime is not None and regime.regime is OIRegime.CAPITULATION:
            return True

        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is None or not anomaly.detected:
            return False

        return anomaly.anomaly_type in {
            OIAnomalyType.LIQUIDATION_DRIVEN_OI_DROP,
            OIAnomalyType.SUDDEN_DELEVERAGING,
            OIAnomalyType.OI_COLLAPSE,
        }

    def oi_analysis_confidence(self, context: SignalContext) -> float:
        analysis = self.extract_oi_analysis(context)
        if analysis is not None:
            return self.clamp(analysis.confidence)

        values: list[float] = []

        regime = self.extract_oi_regime_result(context)
        if regime is not None:
            values.append(regime.confidence)

        divergence = self.extract_oi_divergence_result(context)
        if divergence is not None and divergence.detected:
            values.append(divergence.confidence)

        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is not None and anomaly.detected:
            values.append(anomaly.confidence)

        if not values:
            return 0.0

        return self.clamp(sum(values) / len(values))

    # ------------------------------------------------------------------
    # Generic section / alias helpers
    # ------------------------------------------------------------------

    def extract_domain_section(
        self,
        context: SignalContext,
        section: str,
        *,
        prefix_nested_keys: bool = False,
    ) -> dict[str, Any]:
        domain = self.extract_oi_domain(context)
        raw: dict[str, Any] = {}

        nested = domain.get(section)
        if isinstance(nested, Mapping):
            if prefix_nested_keys:
                raw.update({f"{section}_{key}": value for key, value in nested.items()})
            else:
                raw.update(dict(nested))

        return raw

    def merge_domain_keys(
        self,
        *,
        raw: dict[str, Any],
        domain: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> None:
        for key in keys:
            if key in domain and key not in raw:
                raw[key] = domain[key]

    def merge_aliases(
        self,
        *,
        raw: dict[str, Any],
        domain: Mapping[str, Any],
        aliases: dict[str, str],
    ) -> None:
        for src_key, dst_key in aliases.items():
            if src_key in domain and dst_key not in raw:
                raw[dst_key] = domain[src_key]

    def merge_feature_aliases(
        self,
        *,
        raw: dict[str, Any],
        context: SignalContext,
        feature_aliases: dict[str, str],
    ) -> None:
        for feature_name, dst_key in feature_aliases.items():
            if dst_key not in raw and context.has_feature(feature_name):
                raw[dst_key] = context.get_feature(feature_name)

    # ------------------------------------------------------------------
    # Market regime helper
    # ------------------------------------------------------------------

    def get_market_regime(self, context: SignalContext) -> MarketRegime:
        return context.regime.regime if context.regime is not None else MarketRegime.UNKNOWN

    # ------------------------------------------------------------------
    # Enum parsing helpers
    # ------------------------------------------------------------------

    @classmethod
    def parse_oi_regime(cls, value: Any) -> OIRegime:
        if isinstance(value, OIRegime):
            return value

        text = cls.safe_str(value).upper()
        if not text:
            return OIRegime.NEUTRAL

        for item in OIRegime:
            if item.value.upper() == text or item.name.upper() == text:
                return item

        return OIRegime.NEUTRAL

    @classmethod
    def parse_divergence_type(cls, value: Any) -> OIDivergenceType:
        if isinstance(value, OIDivergenceType):
            return value

        text = cls.safe_str(value).upper()
        if not text:
            return OIDivergenceType.NONE

        for item in OIDivergenceType:
            if item.value.upper() == text or item.name.upper() == text:
                return item

        return OIDivergenceType.NONE

    @classmethod
    def parse_anomaly_type(cls, value: Any) -> OIAnomalyType:
        if isinstance(value, OIAnomalyType):
            return value

        text = cls.safe_str(value).upper()
        if not text:
            return OIAnomalyType.NONE

        for item in OIAnomalyType:
            if item.value.upper() == text or item.name.upper() == text:
                return item

        return OIAnomalyType.NONE

    @staticmethod
    def divergence_to_side_hint(divergence_type: OIDivergenceType) -> str:
        """
        Centralized semantic hint for OI divergence strategies.

        Окремі стратегії можуть мапити це на SignalSide, але не повинні
        дублювати semantic interpretation enum-ів у кожному класі.
        """
        if divergence_type.is_bullish_context:
            return "bullish"
        if divergence_type.is_bearish_context:
            return "bearish"
        return "neutral"

    # ------------------------------------------------------------------
    # Common parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        if low > high:
            raise ValueError("low must be <= high")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return low

        if not math.isfinite(number):
            return low

        return max(low, min(high, number))

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default

        if not math.isfinite(number):
            return default

        return number

    @staticmethod
    def safe_int(value: Any, default: int | None = None) -> int | None:
        try:
            if value is None:
                return default
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default

        if not math.isfinite(number):
            return default

        try:
            return int(number)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def safe_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return default

        if isinstance(value, str):
            return value.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
                "detected",
            }

        return bool(value)

    @staticmethod
    def safe_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    @classmethod
    def normalize_reasons(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]

        if isinstance(value, tuple | set):
            return [str(v).strip() for v in value if str(v).strip()]

        text = str(value).strip()
        return [text] if text else []

    def get_feature_value(
        self,
        context: SignalContext,
        name: str,
        *,
        default: Any = None,
    ) -> Any:
        if context.has_feature(name):
            return context.get_feature(name)
        return default

    # ------------------------------------------------------------------
    # Common scoring helpers
    # ------------------------------------------------------------------

    def category_weight(self) -> float:
        if hasattr(self.config, "get_category_weight"):
            return self.config.get_category_weight(StrategyCategory.OPEN_INTEREST)
        return self.config.weighting.category_weights.get(StrategyCategory.OPEN_INTEREST, 1.0)

    def strategy_weight(self) -> float:
        if hasattr(self.config, "get_strategy_weight"):
            return self.config.get_strategy_weight(self.STRATEGY_NAME, default=1.0)

        strategy_cfg = self.config.get_strategy(self.STRATEGY_NAME)
        return strategy_cfg.weight if strategy_cfg is not None else 1.0

    def regime_adjustment(self, regime: MarketRegime) -> float:
        if hasattr(self.config, "get_regime_adjustment"):
            return self.config.get_regime_adjustment(regime)
        return self.config.weighting.regime_adjustments.get(regime, 1.0)

    def weighted_score(self, base_score: float, regime: MarketRegime) -> float:
        return self.clamp(
            base_score
            * self.category_weight()
            * self.strategy_weight()
            * self.regime_adjustment(regime)
        )

    def confidence_grade(self, confidence: float) -> ConfidenceGrade:
        cfg = self.config.confidence
        if confidence >= cfg.high_threshold:
            return ConfidenceGrade.VERY_HIGH
        if confidence >= cfg.medium_threshold:
            return ConfidenceGrade.HIGH
        if confidence >= cfg.low_threshold:
            return ConfidenceGrade.MEDIUM
        if confidence >= cfg.very_low_threshold:
            return ConfidenceGrade.LOW
        return ConfidenceGrade.VERY_LOW

    @staticmethod
    def strength_from_score(score: float) -> SignalStrength:
        if score >= 0.90:
            return SignalStrength.EXTREME
        if score >= 0.75:
            return SignalStrength.STRONG
        if score >= 0.55:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK

    @staticmethod
    def priority_from_score(score: float) -> SignalPriority:
        if score >= 0.90:
            return SignalPriority.CRITICAL
        if score >= 0.75:
            return SignalPriority.HIGH
        if score >= 0.50:
            return SignalPriority.MEDIUM
        return SignalPriority.LOW

    # ------------------------------------------------------------------
    # Common signal enrichment helpers
    # ------------------------------------------------------------------

    def add_required_source_features(
        self,
        signal: StrategySignal,
        context: SignalContext,
    ) -> None:
        for feature_name in self.REQUIRED_FEATURES:
            if context.has_feature(feature_name):
                signal.add_source_feature(feature_name)

    def add_oi_source_features(
        self,
        signal: StrategySignal,
        context: SignalContext,
    ) -> None:
        """
        Додає source features для canonical OI payload.

        Для нового pipeline source-фічі можуть не бути розсипані в feature_map,
        тому додаємо logical names, якщо відповідні секції реально присутні.
        """
        if self.extract_oi_features(context) is not None:
            signal.add_source_feature("analytics.open_interest.features")

        if self.extract_oi_regime_result(context) is not None:
            signal.add_source_feature("analytics.open_interest.regime")

        divergence = self.extract_oi_divergence_result(context)
        if divergence is not None:
            signal.add_source_feature("analytics.open_interest.divergence")

        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is not None:
            signal.add_source_feature("analytics.open_interest.anomaly")

        self.add_required_source_features(signal, context)

    def append_oi_analysis_metadata(
        self,
        signal: StrategySignal,
        context: SignalContext,
    ) -> None:
        base = self.extract_base_payload(context)

        if base.scope:
            signal.metadata["oi_scope"] = dict(base.scope)

        if base.analysis is not None:
            signal.metadata.update(
                {
                    "oi_analysis_confidence": base.analysis.confidence,
                    "oi_analysis_confidence_band": base.analysis.confidence_band.value,
                    "oi_has_divergence": base.analysis.has_divergence,
                    "oi_has_anomaly": base.analysis.has_anomaly,
                    "oi_analysis_metadata": dict(base.analysis.metadata),
                }
            )

        if base.snapshot is not None:
            signal.metadata["oi_snapshot"] = base.snapshot.to_dict()

        if base.market_context is not None:
            signal.metadata["oi_market_context"] = base.market_context.to_dict()

        if base.regime is not None:
            signal.metadata["oi_regime_result"] = base.regime.to_dict()
            signal.metadata["oi_regime"] = base.regime.regime.value
            signal.metadata["oi_regime_confidence"] = base.regime.confidence
            signal.metadata["oi_regime_score"] = base.regime.score
            signal.metadata["oi_regime_confidence_band"] = base.regime.confidence_band.value

        if base.divergence is not None:
            signal.metadata["oi_divergence_result"] = base.divergence.to_dict()
            signal.metadata["oi_divergence_detected"] = base.divergence.detected
            signal.metadata["oi_divergence_type"] = base.divergence.divergence_type.value
            signal.metadata["oi_divergence_confidence"] = base.divergence.confidence
            signal.metadata["oi_divergence_score"] = base.divergence.score
            signal.metadata["oi_divergence_confidence_band"] = base.divergence.confidence_band.value

        if base.anomaly is not None:
            signal.metadata["oi_anomaly_result"] = base.anomaly.to_dict()
            signal.metadata["oi_anomaly_detected"] = base.anomaly.detected
            signal.metadata["oi_anomaly_type"] = base.anomaly.anomaly_type.value
            signal.metadata["oi_anomaly_strength"] = base.anomaly.strength.value
            signal.metadata["oi_anomaly_confidence"] = base.anomaly.confidence
            signal.metadata["oi_anomaly_score"] = base.anomaly.score
            signal.metadata["oi_anomaly_confidence_band"] = base.anomaly.confidence_band.value
            signal.metadata["oi_is_risk_anomaly"] = base.anomaly.anomaly_type.is_risk_anomaly

        if base.features is not None:
            self.append_oi_feature_metadata(signal, base.features)

    def append_oi_analysis_reasons(
        self,
        signal: StrategySignal,
        context: SignalContext,
    ) -> None:
        regime = self.extract_oi_regime_result(context)
        if regime is not None:
            signal.add_reason(f"oi_regime:{regime.regime.value}")
            for reason in regime.reasons:
                signal.add_reason(f"oi_regime_reason:{reason}")

        divergence = self.extract_oi_divergence_result(context)
        if divergence is not None:
            signal.add_reason(f"oi_divergence:{divergence.divergence_type.value}")
            for reason in divergence.reasons:
                signal.add_reason(f"oi_divergence_reason:{reason}")

        anomaly = self.extract_oi_anomaly_result(context)
        if anomaly is not None and anomaly.detected:
            signal.add_reason(f"oi_anomaly:{anomaly.anomaly_type.value}")
            for reason in anomaly.reasons:
                signal.add_reason(f"oi_anomaly_reason:{reason}")

    def append_oi_feature_reasons(
        self,
        signal: StrategySignal,
        features: OIFeatures,
    ) -> None:
        if features.oi_delta_pct is not None:
            signal.add_reason(f"oi_delta_pct:{features.oi_delta_pct:.4f}")

        if features.price_delta_pct is not None:
            signal.add_reason(f"price_delta_pct:{features.price_delta_pct:.4f}")

        if features.volume_ratio is not None:
            signal.add_reason(f"volume_ratio:{features.volume_ratio:.4f}")

        if features.oi_zscore is not None:
            signal.add_reason(f"oi_zscore:{features.oi_zscore:.4f}")

        if features.funding_rate is not None:
            signal.add_reason(f"funding_rate:{features.funding_rate:.6f}")

        if features.liquidation_imbalance is not None:
            signal.add_reason(f"liquidation_imbalance:{features.liquidation_imbalance:.4f}")

        if features.aggressive_flow_imbalance is not None:
            signal.add_reason(
                f"aggressive_flow_imbalance:{features.aggressive_flow_imbalance:.4f}"
            )

        if features.oi_pressure_score is not None:
            signal.add_reason(f"oi_pressure_score:{features.oi_pressure_score:.4f}")

        if features.oi_price_efficiency is not None:
            signal.add_reason(f"oi_price_efficiency:{features.oi_price_efficiency:.4f}")

    def append_oi_feature_metadata(
        self,
        signal: StrategySignal,
        features: OIFeatures,
    ) -> None:
        """
        Додає повний OIFeatures payload.

        Старі flat metadata keys залишені для сумісності.
        Новий canonical формат також доступний як signal.metadata["oi_features"].
        """
        features_payload = features.to_dict()
        signal.metadata["oi_features"] = features_payload

        signal.metadata.update(
            {
                "oi": features.oi,
                "open_interest_value": features.open_interest_value,
                "oi_delta": features.oi_delta,
                "oi_delta_pct": features.oi_delta_pct,
                "oi_ma_fast": features.oi_ma_fast,
                "oi_ma_slow": features.oi_ma_slow,
                "oi_std": features.oi_std,
                "oi_zscore": features.oi_zscore,
                "oi_velocity": features.oi_velocity,
                "oi_acceleration": features.oi_acceleration,
                "price": features.price,
                "price_delta": features.price_delta,
                "price_delta_pct": features.price_delta_pct,
                "volume": features.volume,
                "quote_volume": features.quote_volume,
                "volume_ma": features.volume_ma,
                "volume_ratio": features.volume_ratio,
                "funding_rate": features.funding_rate,
                "predicted_funding_rate": features.predicted_funding_rate,
                "long_liquidations": features.long_liquidations,
                "short_liquidations": features.short_liquidations,
                "liquidation_imbalance": features.liquidation_imbalance,
                "cvd_delta": features.cvd_delta,
                "aggressive_buy_volume": features.aggressive_buy_volume,
                "aggressive_sell_volume": features.aggressive_sell_volume,
                "aggressive_flow_imbalance": features.aggressive_flow_imbalance,
                "oi_change_per_volume": features.oi_change_per_volume,
                "oi_price_efficiency": features.oi_price_efficiency,
                "oi_pressure_score": features.oi_pressure_score,
                "oi_direction": features.oi_direction.value,
                "price_direction": features.price_direction.value,
                "oi_feature_metadata": dict(features.metadata),
            }
        )

    # ------------------------------------------------------------------
    # Internal normalization helpers
    # ------------------------------------------------------------------

    def _extract_legacy_feature_payload(
        self,
        context: SignalContext,
    ) -> dict[str, Any]:
        """
        Best-effort fallback для старого StrategyContextBuilder, який клав
        OI values у feature_map, а не в context.open_interest.
        """
        names = {
            "exchange": "oi.features.exchange",
            "market_type": "oi.features.market_type",
            "symbol": "oi.features.symbol",
            "timeframe": "oi.features.timeframe",
            "timestamp": "oi.features.timestamp",
            "oi": "oi.features.oi",
            "open_interest_value": "oi.features.open_interest_value",
            "oi_delta": "oi.features.oi_delta",
            "oi_delta_pct": "oi.features.oi_delta_pct",
            "oi_ma_fast": "oi.features.oi_ma_fast",
            "oi_ma_slow": "oi.features.oi_ma_slow",
            "oi_std": "oi.features.oi_std",
            "oi_zscore": "oi.features.oi_zscore",
            "oi_velocity": "oi.features.oi_velocity",
            "oi_acceleration": "oi.features.oi_acceleration",
            "price": "oi.features.price",
            "price_delta": "oi.features.price_delta",
            "price_delta_pct": "oi.features.price_delta_pct",
            "volume": "oi.features.volume",
            "quote_volume": "oi.features.quote_volume",
            "volume_ma": "oi.features.volume_ma",
            "volume_ratio": "oi.features.volume_ratio",
            "funding_rate": "oi.features.funding_rate",
            "predicted_funding_rate": "oi.features.predicted_funding_rate",
            "long_liquidations": "oi.features.long_liquidations",
            "short_liquidations": "oi.features.short_liquidations",
            "liquidation_imbalance": "oi.features.liquidation_imbalance",
            "cvd_delta": "oi.features.cvd_delta",
            "aggressive_buy_volume": "oi.features.aggressive_buy_volume",
            "aggressive_sell_volume": "oi.features.aggressive_sell_volume",
            "aggressive_flow_imbalance": "oi.features.aggressive_flow_imbalance",
            "oi_change_per_volume": "oi.features.oi_change_per_volume",
            "oi_price_efficiency": "oi.features.oi_price_efficiency",
            "oi_pressure_score": "oi.features.oi_pressure_score",
            "oi_direction": "oi.features.oi_direction",
            "price_direction": "oi.features.price_direction",
        }

        payload: dict[str, Any] = {}
        for dst, feature_name in names.items():
            if context.has_feature(feature_name):
                payload[dst] = context.get_feature(feature_name)

        payload.setdefault("symbol", getattr(context, "symbol", None))
        payload.setdefault("timeframe", getattr(context, "timeframe", None))

        return {key: value for key, value in payload.items() if value is not None}

    def _normalize_divergence_payload(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        divergence_type = (
            data.get("divergence_type")
            or data.get("type")
            or data.get("oi_divergence_type")
            or data.get("open_interest_divergence_type")
            or OIDivergenceType.NONE.value
        )

        return {
            "detected": self.safe_bool(
                data.get("detected", data.get("is_detected")),
                default=self.parse_divergence_type(divergence_type) is not OIDivergenceType.NONE,
            ),
            "divergence_type": divergence_type,
            "confidence": data.get(
                "confidence",
                data.get("divergence_confidence", data.get("oi_divergence_confidence", 0.0)),
            ),
            "score": data.get(
                "score",
                data.get("divergence_score", data.get("oi_divergence_score")),
            ),
            "window_size": data.get("window_size"),
            "reasons": list(data.get("reasons") or data.get("divergence_reasons") or []),
        }

    def _normalize_anomaly_payload(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        anomaly_type = (
            data.get("anomaly_type")
            or data.get("type")
            or data.get("oi_anomaly_type")
            or data.get("open_interest_anomaly_type")
            or OIAnomalyType.NONE.value
        )

        return {
            "detected": self.safe_bool(
                data.get("detected", data.get("is_detected")),
                default=self.parse_anomaly_type(anomaly_type) is not OIAnomalyType.NONE,
            ),
            "anomaly_type": anomaly_type,
            "strength": data.get("strength", data.get("anomaly_strength")),
            "confidence": data.get(
                "confidence",
                data.get("anomaly_confidence", data.get("oi_anomaly_confidence", 0.0)),
            ),
            "score": data.get(
                "score",
                data.get("anomaly_score", data.get("oi_anomaly_score")),
            ),
            "reasons": list(data.get("reasons") or data.get("anomaly_reasons") or []),
        }