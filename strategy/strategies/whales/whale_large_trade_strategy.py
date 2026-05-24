# trading_system/strategy/strategies/whales/whale_large_trade_strategy.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.event_bus import EventBus
from core.scheduler import Scheduler

from ...config import StrategyConfig, StrategyDefinitionConfig
from ...enums import (
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from ...exceptions import StrategyConfigError
from ...models import StrategyContext, StrategyMetadata, StrategySignal
from .base import (
    WHALES_FEATURES,
    WhaleCompositeSnapshot,
    WhalesStrategyConfig,
    WhalesTradingStrategy,
)
from .utils import (
    ScoreBreakdown,
    extract_event_time,
    extract_large_trade_notional,
    extract_large_trade_payload,
    extract_large_trade_zscore,
    extract_reference_price,
    extract_total_notional,
    extract_whale_side,
    freshness_score,
    is_directional_side,
    is_stale,
    large_trade_score,
    serialize_for_metadata,
    side_label_to_signal_side,
    unit_score,
    whale_large_trade_source_features,
    weighted_score,
)


@dataclass(slots=True)
class WhaleLargeTradePayload:
    """
    Strategy-level payload для plain analytics.whales.large_trade.

    Це окремий path для одиничної великої whale-угоди. Він не вимагає
    whales.pressure або whales.liquidation_context і тому не конфліктує з
    absorption/liquidation strategies.
    """

    _logger = logging.getLogger(__name__ + ".WhaleLargeTradePayload")

    snapshot: WhaleCompositeSnapshot
    side: SignalSide
    whale_side: str
    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WhaleLargeTradeStrategyConfig(WhalesStrategyConfig):
    """Config для standalone large trade whale strategy."""

    _logger = logging.getLogger(__name__ + ".WhaleLargeTradeStrategyConfig")

    min_large_trade_score: float = 0.56
    min_large_trade_confidence: float = 0.52

    min_large_trade_notional: float = 150_000.0
    min_large_trade_zscore: float = 1.25

    require_directional_side: bool = True
    require_reference_price: bool = False

    score_notional_weight: float = 0.58
    score_zscore_weight: float = 0.24
    score_freshness_weight: float = 0.10
    score_price_weight: float = 0.08

    confidence_large_trade_weight: float = 0.62
    confidence_freshness_weight: float = 0.18
    confidence_price_weight: float = 0.10
    confidence_side_weight: float = 0.10

    high_priority_score: float = 0.80
    critical_priority_score: float = 0.92

    default_priority: SignalPriority = SignalPriority.MEDIUM
    default_setup_type: SetupType = SetupType.CONTINUATION

    tag_whale_large_trade: str = "whale_large_trade"
    tag_large_trade: str = "large_trade"
    tag_buy_large_trade: str = "buy_large_trade"
    tag_sell_large_trade: str = "sell_large_trade"

    required_whales_features: tuple[str, ...] = (
        WHALES_FEATURES.LARGE_TRADE,
    )

    def validate(self) -> None:
        WhalesStrategyConfig.validate(self)

        unit_fields = {
            "min_large_trade_score": self.min_large_trade_score,
            "min_large_trade_confidence": self.min_large_trade_confidence,
            "high_priority_score": self.high_priority_score,
            "critical_priority_score": self.critical_priority_score,
            "score_notional_weight": self.score_notional_weight,
            "score_zscore_weight": self.score_zscore_weight,
            "score_freshness_weight": self.score_freshness_weight,
            "score_price_weight": self.score_price_weight,
            "confidence_large_trade_weight": self.confidence_large_trade_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
            "confidence_price_weight": self.confidence_price_weight,
            "confidence_side_weight": self.confidence_side_weight,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        non_negative_fields = {
            "min_large_trade_notional": self.min_large_trade_notional,
            "min_large_trade_zscore": self.min_large_trade_zscore,
        }
        for field_name, value in non_negative_fields.items():
            if float(value) < 0.0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if (
            self.score_notional_weight
            + self.score_zscore_weight
            + self.score_freshness_weight
            + self.score_price_weight
        ) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if (
            self.confidence_large_trade_weight
            + self.confidence_freshness_weight
            + self.confidence_price_weight
            + self.confidence_side_weight
        ) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        for attr in (
            "tag_whale_large_trade",
            "tag_large_trade",
            "tag_buy_large_trade",
            "tag_sell_large_trade",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_whales_features:
            raise StrategyConfigError("required_whales_features cannot be empty")


class WhaleLargeTradeStrategy(WhalesTradingStrategy):
    """
    Standalone strategy for analytics.whales.large_trade.

    Input:
        StrategyContext with FeatureSource.WHALES and whales.large_trade.

    Output:
        StrategySignal | None.

    The strategy does not emit signal.generated directly; SignalProcessor owns
    routing, confluence, filtering and risk-ready conversion.
    """

    _logger = logging.getLogger(__name__ + ".WhaleLargeTradeStrategy")

    component_namespace = "strategy.whales.large_trade"
    category: StrategyCategory = StrategyCategory.WHALES
    default_setup_type: SetupType = SetupType.CONTINUATION

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        whales_config: WhaleLargeTradeStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_whales_config = whales_config or WhaleLargeTradeStrategyConfig()
        resolved_whales_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            whales_config=resolved_whales_config,
            service_name=service_name,
        )

        self.large_trade_config: WhaleLargeTradeStrategyConfig = resolved_whales_config

    @property
    def strategy_name(self) -> str:
        return "whale_large_trade"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.WHALES,
            timeframe=Timeframe.M1,
            tags=[
                self.large_trade_config.tag_whales,
                self.large_trade_config.tag_whale_large_trade,
                self.large_trade_config.tag_large_trade,
                "analytics_whales",
            ],
            version="2.0.0",
            description=(
                "Interprets standalone whale large-trade events from "
                "normalized StrategyContext and returns internal StrategySignal."
            ),
            required_features=set(self.required_features()),
            supported_regimes={
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.BREAKOUT,
                MarketRegime.SQUEEZE,
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.UNKNOWN,
            },
            metadata={
                "source": "analytics.whales",
                "strategy_type": "whale_large_trade",
                "base_class": "WhalesTradingStrategy",
                "canonical_payload": "WhaleCompositeSnapshot",
                "uses_large_trade": True,
                "requires_pressure": False,
                "requires_liquidation_context": False,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.large_trade_config.required_whales_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_whales_data(
            context,
            tuple(self.large_trade_config.required_whales_features),
        ):
            self.remember_no_signal(
                "missing_whale_large_trade_data",
                required_features=sorted(self.required_features()),
            )
            return None

        if self.has_stale_whales_features(
            context,
            tuple(self.large_trade_config.required_whales_features),
        ):
            self.remember_no_signal(
                "stale_whale_large_trade_feature",
                required_features=sorted(self.large_trade_config.required_whales_features),
            )
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.large_trade_config.stale_feature_max_age_seconds,
        ):
            self.remember_no_signal(
                "stale_whale_large_trade_payload",
                event_time=payload.event_time.isoformat() if payload.event_time else None,
            )
            return None

        if not self.accepts_whale_snapshot(
            payload.snapshot,
            require_futures_market_type=self.large_trade_config.require_futures_market_type,
            min_confidence=0.0,
        ):
            self.remember_no_signal(
                "invalid_whale_large_trade_snapshot",
                snapshot=serialize_for_metadata(payload.snapshot.to_dict()),
            )
            return None

        breakdown = self._build_score_breakdown(context=context, payload=payload)

        if breakdown.score < self.large_trade_config.min_large_trade_score:
            self.remember_no_signal(
                "whale_large_trade_score_below_threshold",
                score=breakdown.score,
                threshold=self.large_trade_config.min_large_trade_score,
                breakdown=breakdown.to_dict(),
            )
            return None

        if breakdown.confidence < self.large_trade_config.min_large_trade_confidence:
            self.remember_no_signal(
                "whale_large_trade_confidence_below_threshold",
                confidence=breakdown.confidence,
                threshold=self.large_trade_config.min_large_trade_confidence,
                breakdown=breakdown.to_dict(),
            )
            return None

        priority = self._priority(breakdown.score)
        tags = self._tags(payload)
        source_features = self._source_features(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "whale_large_trade_signal",
                    f"side:{payload.side.value}",
                    f"whale_side:{payload.whale_side}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        metadata = {
            "whales_setup_family": "whale_large_trade",
            "whales_strategy_version": "2.0.0",
            "contract": "whales",
            "contract_version": "strategy-domain-v1",
            "primary_section": "large_trade",
            "strategy_contract_role": "decision_module",
            "risk_ready_payload_owner": "SignalProcessor",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "mapped_side": payload.side.value,
            "whale_side": payload.whale_side,
            "large_trade_notional": payload.snapshot.large_trade_notional,
            "large_trade_zscore": payload.snapshot.large_trade_zscore,
            "reference_price": payload.snapshot.reference_price,
        }

        return self.build_whale_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.large_trade_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=priority,
        )

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> WhaleLargeTradePayload | None:
        snapshot = self.resolve_whale_snapshot(context)
        if snapshot is None or not snapshot.has_large_trade:
            self.remember_no_signal(
                "whale_large_trade_payload_not_resolved",
                whales_domain_keys=sorted(self.whales_domain(context).keys()),
                required_features=sorted(self.required_features()),
            )
            return None

        large_trade = snapshot.large_trade
        whale_side = extract_whale_side(large_trade) or snapshot.whale_side
        side = side_label_to_signal_side(whale_side)

        if self.large_trade_config.require_directional_side and not is_directional_side(side):
            self.remember_no_signal(
                "whale_large_trade_side_not_directional",
                whale_side=whale_side,
                large_trade=serialize_for_metadata(large_trade),
            )
            return None

        notional = extract_large_trade_notional(large_trade)
        if notional < self.large_trade_config.min_large_trade_notional:
            self.remember_no_signal(
                "whale_large_trade_notional_below_threshold",
                notional=notional,
                threshold=self.large_trade_config.min_large_trade_notional,
            )
            return None

        zscore = extract_large_trade_zscore(large_trade) or 0.0
        if zscore < self.large_trade_config.min_large_trade_zscore:
            self.remember_no_signal(
                "whale_large_trade_zscore_below_threshold",
                zscore=zscore,
                threshold=self.large_trade_config.min_large_trade_zscore,
            )
            return None

        if self.large_trade_config.require_reference_price and not snapshot.reference_price:
            self.remember_no_signal("whale_large_trade_reference_price_missing")
            return None

        event_time = extract_event_time(large_trade) or snapshot.timestamp or context.timestamp
        reasons = [
            f"large_trade_notional:{notional:.2f}",
            f"large_trade_zscore:{zscore:.3f}",
        ]
        if snapshot.reference_price:
            reasons.append(f"reference_price:{snapshot.reference_price:.8g}")

        return WhaleLargeTradePayload(
            snapshot=snapshot,
            side=side,
            whale_side=whale_side,
            event_time=event_time,
            reasons=reasons,
            raw={"large_trade": large_trade},
        )

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: WhaleLargeTradePayload,
    ) -> ScoreBreakdown:
        cfg = self.large_trade_config
        snapshot = payload.snapshot
        large_trade = snapshot.large_trade

        notional = extract_large_trade_notional(large_trade)
        zscore = extract_large_trade_zscore(large_trade) or 0.0
        reference_price = extract_reference_price(large_trade) or snapshot.reference_price

        large_trade_component = large_trade_score(
            large_trade,
            min_notional=cfg.min_large_trade_notional,
            min_zscore=cfg.min_large_trade_zscore,
        )
        notional_component = unit_score(notional / max(cfg.min_large_trade_notional * 2.0, 1.0))
        zscore_component = unit_score(zscore / max(cfg.min_large_trade_zscore * 2.0, 1.0))
        freshness_component = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=cfg.stale_feature_max_age_seconds,
        )
        price_component = 1.0 if reference_price is not None and reference_price > 0 else 0.0
        side_component = 1.0 if is_directional_side(payload.side) else 0.0

        score = weighted_score(
            {
                "notional": notional_component,
                "zscore": zscore_component,
                "freshness": freshness_component,
                "price": price_component,
            },
            {
                "notional": cfg.score_notional_weight,
                "zscore": cfg.score_zscore_weight,
                "freshness": cfg.score_freshness_weight,
                "price": cfg.score_price_weight,
            },
        )

        confidence = weighted_score(
            {
                "large_trade": large_trade_component,
                "freshness": freshness_component,
                "price": price_component,
                "side": side_component,
            },
            {
                "large_trade": cfg.confidence_large_trade_weight,
                "freshness": cfg.confidence_freshness_weight,
                "price": cfg.confidence_price_weight,
                "side": cfg.confidence_side_weight,
            },
        )

        reasons = [
            f"notional_score:{notional_component:.3f}",
            f"zscore_score:{zscore_component:.3f}",
        ]
        confirmations: list[str] = []
        if notional >= cfg.min_large_trade_notional:
            confirmations.append("large_trade_notional_confirmed")
        if zscore >= cfg.min_large_trade_zscore:
            confirmations.append("large_trade_zscore_confirmed")
        if price_component > 0:
            confirmations.append("reference_price_available")

        return ScoreBreakdown(
            score=score,
            confidence=confidence,
            components={
                "large_trade": large_trade_component,
                "notional": notional_component,
                "zscore": zscore_component,
                "freshness": freshness_component,
                "price": price_component,
                "side": side_component,
            },
            reasons=reasons,
            confirmations=confirmations,
        )

    def _priority(self, score: float) -> SignalPriority:
        if score >= self.large_trade_config.critical_priority_score:
            return SignalPriority.CRITICAL
        if score >= self.large_trade_config.high_priority_score:
            return SignalPriority.HIGH
        return self.large_trade_config.default_priority

    def _source_features(self, payload: WhaleLargeTradePayload) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *whale_large_trade_source_features(),
                    WHALES_FEATURES.LARGE_TRADE,
                    WHALES_FEATURES.LARGE_TRADE_NOTIONAL,
                    WHALES_FEATURES.LARGE_TRADE_ZSCORE,
                    WHALES_FEATURES.WHALE_SIDE,
                    WHALES_FEATURES.REFERENCE_PRICE,
                ]
            )
        )

    def _tags(self, payload: WhaleLargeTradePayload) -> list[str]:
        tags = [
            self.large_trade_config.tag_whales,
            self.large_trade_config.tag_whale_large_trade,
            self.large_trade_config.tag_large_trade,
        ]
        if payload.side is SignalSide.LONG:
            tags.append(self.large_trade_config.tag_buy_large_trade)
        elif payload.side is SignalSide.SHORT:
            tags.append(self.large_trade_config.tag_sell_large_trade)
        return list(dict.fromkeys(tags))
