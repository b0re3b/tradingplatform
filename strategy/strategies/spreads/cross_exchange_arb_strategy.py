# trading_system/strategy/strategies/spreads/cross_exchange_arb_strategy.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from analytics.spreads.enums import (
    SpreadType,
)
from core.event_bus import EventBus
from core.scheduler import Scheduler
from .base import (
    SPREADS_FEATURES,
    SpreadCompositeSnapshot,
    SpreadsStrategyConfig,
    SpreadsTradingStrategy,
)
from .utils import (
    DECIMAL_ONE,
    DECIMAL_ZERO,
    ScoreBreakdown,
    arbitrage_opportunity_source_features,
    average_score,
    confidence_from_components,
    cross_exchange_contract_error,
    cross_exchange_direction,
    cross_exchange_leg_metadata,
    cross_exchange_source_features,
    cross_exchange_to_signal_side,
    edge_component,
    extract_net_edge,
    extract_net_edge_bps,
    extract_timestamp,
    freshness_score,
    is_directional_side,
    is_stale,
    normalize_label,
    opportunity_status_component,
    quote_component,
    serialize_for_metadata,
    spread_quality_filter_reason,
    unit_score,
    weighted_score,
)
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


@dataclass(slots=True)
class CrossExchangeArbPayload:
    """
    Normalized strategy-level payload for cross-exchange arbitrage.

    Direction convention:
    - opportunity buy_exchange/sell_exchange -> LONG_A_SHORT_B;
    - StrategySignal.side remains generic LONG/SHORT;
    - exact leg construction is metadata for SignalProcessor/SignalBuilder.
    """

    snapshot: SpreadCompositeSnapshot
    side: SignalSide
    spread_direction: str

    net_edge: Decimal
    abs_net_edge: Decimal
    net_edge_bps: Decimal
    abs_net_edge_bps: Decimal

    event_time: datetime | None = None
    reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CrossExchangeArbStrategyConfig(SpreadsStrategyConfig):
    """
    Unified cross-exchange arbitrage strategy config.

    Strategy idea:
    - read normalized arbitrage opportunity / cross-exchange spread from StrategyContext;
    - require tradeable cross-exchange contract and net edge;
    - map buy/sell leg semantics into metadata;
    - return internal StrategySignal only.
    """

    min_net_edge: Decimal = Decimal("0")
    min_edge_bps: Decimal = Decimal("5")
    min_arb_confidence: float = 0.55

    require_cross_exchange_contract: bool = True
    require_valid_quote: bool = True
    require_active_opportunity: bool = True
    require_tradeable_opportunity: bool = True
    require_fees_and_slippage_edge: bool = True

    require_persistence: bool = False
    min_persistence_ms: int = 500

    require_arbitrage_signal_confirmation: bool = False

    allowed_instrument_types: set[str] = field(default_factory=set)
    allowed_buy_exchanges: set[str] = field(default_factory=set)
    allowed_sell_exchanges: set[str] = field(default_factory=set)

    score_edge_weight: float = 0.34
    score_bps_weight: float = 0.24
    score_status_weight: float = 0.16
    score_quote_weight: float = 0.10
    score_persistence_weight: float = 0.08
    score_freshness_weight: float = 0.08

    confidence_primary_weight: float = 0.55
    confidence_context_weight: float = 0.25
    confidence_confirmation_weight: float = 0.15
    confidence_freshness_weight: float = 0.05

    strong_edge_bonus: float = 0.05
    strong_bps_bonus: float = 0.04
    tradeable_opportunity_bonus: float = 0.04
    persistence_bonus: float = 0.03
    confirmation_bonus: float = 0.03

    strong_edge_multiplier: Decimal = Decimal("2")
    strong_bps_multiplier: Decimal = Decimal("2")

    entry_offset_bps_hint: float | None = None
    stop_buffer_bps_hint: float | None = None
    take_profit_bps_hint: float | None = None
    arb_tp_multiplier_hint: float | None = None

    default_priority: SignalPriority = SignalPriority.HIGH
    default_setup_type: SetupType = SetupType.ARBITRAGE

    tag_cross_exchange_arb: str = "cross_exchange_arb"
    tag_net_edge: str = "net_edge"
    tag_tradeable: str = "tradeable"
    tag_long_a_short_b: str = "long_a_short_b"
    tag_short_a_long_b: str = "short_a_long_b"
    tag_persistent: str = "persistent"

    required_spreads_features: tuple[str, ...] = (
        SPREADS_FEATURES.OPPORTUNITY,
    )

    def validate(self) -> None:
        SpreadsStrategyConfig.validate(self)

        if self.min_net_edge < DECIMAL_ZERO:
            raise StrategyConfigError("min_net_edge must be >= 0")

        if self.min_edge_bps < DECIMAL_ZERO:
            raise StrategyConfigError("min_edge_bps must be >= 0")

        if not 0.0 <= float(self.min_arb_confidence) <= 1.0:
            raise StrategyConfigError("min_arb_confidence must be between 0.0 and 1.0")

        if self.min_persistence_ms < 0:
            raise StrategyConfigError("min_persistence_ms must be >= 0")

        if self.strong_edge_multiplier < DECIMAL_ZERO:
            raise StrategyConfigError("strong_edge_multiplier must be >= 0")

        if self.strong_bps_multiplier < DECIMAL_ZERO:
            raise StrategyConfigError("strong_bps_multiplier must be >= 0")

        score_weights = {
            "score_edge_weight": self.score_edge_weight,
            "score_bps_weight": self.score_bps_weight,
            "score_status_weight": self.score_status_weight,
            "score_quote_weight": self.score_quote_weight,
            "score_persistence_weight": self.score_persistence_weight,
            "score_freshness_weight": self.score_freshness_weight,
        }
        confidence_weights = {
            "confidence_primary_weight": self.confidence_primary_weight,
            "confidence_context_weight": self.confidence_context_weight,
            "confidence_confirmation_weight": self.confidence_confirmation_weight,
            "confidence_freshness_weight": self.confidence_freshness_weight,
        }

        for field_name, value in {**score_weights, **confidence_weights}.items():
            if float(value) < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        if sum(score_weights.values()) <= 0:
            raise StrategyConfigError("score weights sum must be > 0")

        if sum(confidence_weights.values()) <= 0:
            raise StrategyConfigError("confidence weights sum must be > 0")

        unit_fields = {
            "strong_edge_bonus": self.strong_edge_bonus,
            "strong_bps_bonus": self.strong_bps_bonus,
            "tradeable_opportunity_bonus": self.tradeable_opportunity_bonus,
            "persistence_bonus": self.persistence_bonus,
            "confirmation_bonus": self.confirmation_bonus,
        }
        for field_name, value in unit_fields.items():
            if not 0.0 <= float(value) <= 1.0:
                raise StrategyConfigError(f"{field_name} must be between 0.0 and 1.0")

        hint_fields = {
            "entry_offset_bps_hint": self.entry_offset_bps_hint,
            "stop_buffer_bps_hint": self.stop_buffer_bps_hint,
            "take_profit_bps_hint": self.take_profit_bps_hint,
            "arb_tp_multiplier_hint": self.arb_tp_multiplier_hint,
        }
        for field_name, value in hint_fields.items():
            if value is not None and value < 0:
                raise StrategyConfigError(f"{field_name} must be >= 0")

        for attr in (
            "tag_cross_exchange_arb",
            "tag_net_edge",
            "tag_tradeable",
            "tag_long_a_short_b",
            "tag_short_a_long_b",
            "tag_persistent",
        ):
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise StrategyConfigError(f"{attr} must be a non-empty string")

        if not self.required_spreads_features:
            raise StrategyConfigError("required_spreads_features cannot be empty")


class CrossExchangeArbStrategy(SpreadsTradingStrategy):
    """
    Unified cross-exchange arbitrage strategy.

    Input:
        StrategyContext with FeatureSource.SPREADS domain data / features.

    Output:
        StrategySignal | None.

    This class does not subscribe to EventBus and does not emit signal.generated.
    SignalProcessor owns routing, filters, confluence, building and risk payloads.
    """

    component_namespace = "strategy.spreads.cross_exchange_arb"
    category: StrategyCategory = StrategyCategory.SPREADS
    default_setup_type: SetupType = SetupType.ARBITRAGE

    def __init__(
        self,
        config: StrategyConfig,
        event_bus: EventBus | None = None,
        scheduler: Scheduler | None = None,
        *,
        definition: StrategyDefinitionConfig | None = None,
        spreads_config: CrossExchangeArbStrategyConfig | None = None,
        service_name: str | None = None,
    ) -> None:
        resolved_spreads_config = spreads_config or CrossExchangeArbStrategyConfig()
        resolved_spreads_config.validate()

        super().__init__(
            config=config,
            event_bus=event_bus,
            scheduler=scheduler,
            definition=definition,
            spreads_config=resolved_spreads_config,
            service_name=service_name,
        )

        self.arb_config: CrossExchangeArbStrategyConfig = resolved_spreads_config

    @property
    def strategy_name(self) -> str:
        return "cross_exchange_arb"

    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            strategy_name=self.strategy_name,
            category=StrategyCategory.SPREADS,
            timeframe=Timeframe.M1,
            tags=[
                self.arb_config.tag_spreads,
                self.arb_config.tag_cross_exchange,
                self.arb_config.tag_arbitrage,
                self.arb_config.tag_cross_exchange_arb,
                self.arb_config.tag_net_edge,
                "analytics_spreads",
            ],
            version="2.0.0",
            description=(
                "Interprets cross-exchange arbitrage opportunities from normalized "
                "StrategyContext and returns internal StrategySignal."
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
                "source": "analytics.spreads",
                "strategy_type": "cross_exchange_arb",
                "base_class": "SpreadsTradingStrategy",
                "canonical_payload": "SpreadCompositeSnapshot",
                "uses_arbitrage_opportunity": True,
                "uses_cross_exchange": True,
                "emits_signal_generated": False,
                "risk_ready_payload_owner": "SignalProcessor",
            },
        )

    def required_features(self) -> set[str]:
        base_required = super().required_features()
        return set(base_required).union(self.arb_config.required_spreads_features)

    async def generate_signal(
        self,
        context: StrategyContext,
    ) -> StrategySignal | None:
        self.validate_context_requirements(context)

        if not self.has_any_spreads_data(
            context,
            tuple(self.arb_config.required_spreads_features),
        ):
            return None

        if self.has_stale_spreads_features(
            context,
            tuple(self.arb_config.required_spreads_features),
        ):
            return None

        payload = self._extract_payload(context)
        if payload is None:
            return None

        if is_stale(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.arb_config.stale_feature_max_age_seconds,
        ):
            return None

        rejection = spread_quality_filter_reason(
            payload.snapshot.to_signal_payload(),
            min_score=self.arb_config.min_score,
            min_confidence=max(
                self.arb_config.min_confidence,
                self.arb_config.min_arb_confidence,
            ),
            require_valid_quote=self.arb_config.require_valid_quote,
            require_edge=True,
            stale_after_seconds=self.arb_config.stale_feature_max_age_seconds,
            now=context.timestamp,
        )
        if rejection is not None:
            return None

        if not self.accepts_spread_snapshot(
            payload.snapshot,
            require_valid_quote=self.arb_config.require_valid_quote,
            require_edge=True,
        ):
            return None

        if not self._passes_contract_filters(payload.snapshot):
            return None

        if not self._passes_arb_filters(payload):
            return None

        breakdown = self._build_score_breakdown(
            context=context,
            payload=payload,
        )

        if breakdown.score < self.arb_config.min_score:
            return None

        if breakdown.confidence < self.arb_config.min_arb_confidence:
            return None

        source_features = self._source_features(payload)
        tags = self._tags(payload)

        reasons = list(
            dict.fromkeys(
                [
                    "cross_exchange_arb_signal",
                    f"side:{payload.side.value}",
                    f"spread_direction:{payload.spread_direction}",
                    *payload.reasons,
                    *breakdown.reasons,
                ]
            )
        )
        confirmations = list(dict.fromkeys(breakdown.confirmations))

        leg_metadata = cross_exchange_leg_metadata(
            payload.snapshot.raw_opportunity
            or payload.snapshot.to_signal_payload()
        )

        metadata = {
            "spreads_setup_family": "cross_exchange_arb",
            "spreads_strategy_version": "2.0.0",
            "score_breakdown": breakdown.to_dict(),
            "tags": tags,
            "snapshot": serialize_for_metadata(payload.snapshot.to_dict()),
            "raw": serialize_for_metadata(payload.raw),
            "event_time": payload.event_time.isoformat() if payload.event_time else None,
            "spread_type": normalize_label(payload.snapshot.spread_type),
            "spread_direction": payload.spread_direction,
            "mapped_side": payload.side.value,
            "net_edge": str(payload.net_edge),
            "abs_net_edge": str(payload.abs_net_edge),
            "net_edge_bps": str(payload.net_edge_bps),
            "abs_net_edge_bps": str(payload.abs_net_edge_bps),
            "opportunity_key": payload.snapshot.opportunity_key,
            "opportunity_status": normalize_label(payload.snapshot.opportunity_status),
            "opportunity_active": payload.snapshot.opportunity_active,
            "opportunity_tradeable": payload.snapshot.opportunity_tradeable,
            "persistence_ms": payload.snapshot.persistence_ms,
            "quote_validity": normalize_label(payload.snapshot.quote_validity),
            "confidence": payload.snapshot.confidence,
            "leg_semantics": leg_metadata,
            "execution_hints": self._execution_hints(),
        }

        return self.build_spread_signal(
            context=context,
            side=payload.side,
            confidence=breakdown.confidence,
            score=breakdown.score,
            setup_type=self.arb_config.default_setup_type,
            reasons=reasons,
            confirmations=confirmations,
            source_features=source_features,
            metadata=metadata,
            priority=self.arb_config.default_priority,
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_payload(
        self,
        context: StrategyContext,
    ) -> CrossExchangeArbPayload | None:
        snapshot = self.resolve_spread_snapshot(context)
        if snapshot is None or not snapshot.has_minimum_data():
            return None

        net_edge = self._net_edge(snapshot)
        if net_edge is None:
            return None

        net_edge_bps = self._net_edge_bps(snapshot)
        if net_edge_bps is None:
            return None

        direction = cross_exchange_direction(
            snapshot.raw_opportunity or snapshot.to_signal_payload()
        )
        side = cross_exchange_to_signal_side(
            snapshot.raw_opportunity or snapshot.to_signal_payload()
        )

        if not is_directional_side(side):
            return None

        event_time = (
            extract_timestamp(snapshot.raw_opportunity)
            or extract_timestamp(snapshot.to_signal_payload())
            or snapshot.timestamp
            or context.timestamp
        )

        reasons = [
            "cross_exchange_arb_context",
            f"spread_type:{normalize_label(snapshot.spread_type)}",
            f"spread_direction:{direction}",
            f"net_edge:{net_edge}",
            f"net_edge_bps:{net_edge_bps}",
            f"confidence:{snapshot.confidence:.4f}",
        ]

        if snapshot.opportunity_key:
            reasons.append(f"opportunity_key:{snapshot.opportunity_key}")

        return CrossExchangeArbPayload(
            snapshot=snapshot,
            side=side,
            spread_direction=direction,
            net_edge=net_edge,
            abs_net_edge=abs(net_edge),
            net_edge_bps=net_edge_bps,
            abs_net_edge_bps=abs(net_edge_bps),
            event_time=event_time,
            reasons=list(dict.fromkeys(reasons)),
            raw={
                "snapshot": snapshot.raw_snapshot,
                "signal": snapshot.raw_signal,
                "opportunity": snapshot.raw_opportunity,
            },
        )

    def _net_edge(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> Decimal | None:
        for candidate in (
            snapshot.net_edge,
            extract_net_edge(snapshot.raw_opportunity),
            snapshot.net_edge_bps,
            snapshot.spread_bps,
        ):
            if candidate is not None:
                return candidate
        return None

    def _net_edge_bps(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> Decimal | None:
        for candidate in (
            snapshot.net_edge_bps,
            extract_net_edge_bps(snapshot.raw_opportunity),
            snapshot.spread_bps,
            snapshot.net_edge,
        ):
            if candidate is not None:
                return candidate
        return None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _passes_contract_filters(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> bool:
        if not self.arb_config.require_cross_exchange_contract:
            return True

        if snapshot.spread_type is not None and snapshot.spread_type is not SpreadType.CROSS_EXCHANGE:
            return False

        contract_error = cross_exchange_contract_error(
            snapshot.raw_opportunity or snapshot.to_signal_payload()
        )
        if contract_error is not None:
            return False

        if self.arb_config.allowed_buy_exchanges:
            buy_exchange = (
                snapshot.raw_opportunity.get("buy_exchange")
                or snapshot.exchange_a
            )
            if str(buy_exchange).lower() not in {
                exchange.lower()
                for exchange in self.arb_config.allowed_buy_exchanges
            }:
                return False

        if self.arb_config.allowed_sell_exchanges:
            sell_exchange = (
                snapshot.raw_opportunity.get("sell_exchange")
                or snapshot.exchange_b
            )
            if str(sell_exchange).lower() not in {
                exchange.lower()
                for exchange in self.arb_config.allowed_sell_exchanges
            }:
                return False

        if self.arb_config.allowed_instrument_types:
            raw_instrument = (
                snapshot.raw_opportunity.get("instrument_type")
                or snapshot.raw_snapshot.get("instrument_type")
                or snapshot.metadata.get("instrument_type")
            )
            label = normalize_label(raw_instrument)
            if label and label not in {
                item.lower()
                for item in self.arb_config.allowed_instrument_types
            }:
                return False

        return True

    def _passes_arb_filters(
        self,
        payload: CrossExchangeArbPayload,
    ) -> bool:
        snapshot = payload.snapshot

        if self.arb_config.require_valid_quote and not snapshot.is_quote_valid:
            return False

        if self.arb_config.require_active_opportunity and not snapshot.opportunity_active:
            return False

        if self.arb_config.require_tradeable_opportunity and not snapshot.opportunity_tradeable:
            return False

        if payload.abs_net_edge < self.arb_config.min_net_edge:
            return False

        if payload.abs_net_edge_bps < self.arb_config.min_edge_bps:
            return False

        if self.arb_config.require_fees_and_slippage_edge:
            if payload.net_edge <= DECIMAL_ZERO and payload.net_edge_bps <= DECIMAL_ZERO:
                return False

        if self.arb_config.require_persistence:
            if snapshot.persistence_ms < self.arb_config.min_persistence_ms:
                return False

        if self.arb_config.require_arbitrage_signal_confirmation:
            if not self._has_arb_signal_confirmation(snapshot):
                return False

        return True

    def _has_arb_signal_confirmation(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> bool:
        if snapshot.signal_type is None:
            return bool(snapshot.raw_signal)

        label = normalize_label(snapshot.signal_type)
        return label in {
            "arbitrage",
            "cross_exchange_arbitrage",
            "opportunity",
            "entry",
            "open",
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _build_score_breakdown(
        self,
        *,
        context: StrategyContext,
        payload: CrossExchangeArbPayload,
    ) -> ScoreBreakdown:
        snapshot = payload.snapshot

        edge_scale = max(self.arb_config.min_net_edge, DECIMAL_ONE)
        bps_scale = max(self.arb_config.min_edge_bps, DECIMAL_ONE)

        edge_component_value = edge_component(
            snapshot.raw_opportunity or snapshot.to_signal_payload(),
            min_edge=edge_scale,
            scale=edge_scale * Decimal("3"),
        )
        bps_component_value = unit_score(
            payload.abs_net_edge_bps / max(bps_scale * Decimal("3"), DECIMAL_ONE)
        )
        status_component_value = opportunity_status_component(
            snapshot.raw_opportunity or snapshot.to_signal_payload()
        )
        quote_component_value = quote_component(snapshot.to_signal_payload())
        persistence_component_value = self._persistence_component(snapshot)
        freshness_component_value = freshness_score(
            event_time=payload.event_time,
            now=context.timestamp,
            stale_after_seconds=self.arb_config.stale_feature_max_age_seconds,
        )

        components = {
            "edge": edge_component_value,
            "bps": bps_component_value,
            "status": status_component_value,
            "quote": quote_component_value,
            "persistence": persistence_component_value,
            "freshness": freshness_component_value,
        }
        weights = {
            "edge": self.arb_config.score_edge_weight,
            "bps": self.arb_config.score_bps_weight,
            "status": self.arb_config.score_status_weight,
            "quote": self.arb_config.score_quote_weight,
            "persistence": self.arb_config.score_persistence_weight,
            "freshness": self.arb_config.score_freshness_weight,
        }

        score = weighted_score(components, weights, default=edge_component_value)
        confidence = confidence_from_components(
            primary=average_score(snapshot.confidence, edge_component_value),
            context=average_score(status_component_value, quote_component_value),
            confirmation=average_score(bps_component_value, persistence_component_value),
            freshness=freshness_component_value,
            primary_weight=self.arb_config.confidence_primary_weight,
            context_weight=self.arb_config.confidence_context_weight,
            confirmation_weight=self.arb_config.confidence_confirmation_weight,
            freshness_weight=self.arb_config.confidence_freshness_weight,
        )

        reasons: list[str] = []
        confirmations: list[str] = [
            "cross_exchange_arb_context",
            f"spread_direction:{payload.spread_direction}",
            f"side:{payload.side.value}",
            f"net_edge:{payload.net_edge}",
            f"net_edge_bps:{payload.net_edge_bps}",
        ]

        if payload.abs_net_edge >= self.arb_config.min_net_edge * self.arb_config.strong_edge_multiplier:
            score += self.arb_config.strong_edge_bonus
            confirmations.append("strong_net_edge")

        if payload.abs_net_edge_bps >= self.arb_config.min_edge_bps * self.arb_config.strong_bps_multiplier:
            score += self.arb_config.strong_bps_bonus
            confirmations.append("strong_edge_bps")

        if snapshot.opportunity_tradeable:
            score += self.arb_config.tradeable_opportunity_bonus
            confirmations.append("tradeable_opportunity")

        if snapshot.persistence_ms >= self.arb_config.min_persistence_ms:
            score += self.arb_config.persistence_bonus
            confirmations.append("persistence_confirmed")

        if self._has_arb_signal_confirmation(snapshot):
            score += self.arb_config.confirmation_bonus
            confirmations.append("arbitrage_signal_confirmation")

        if snapshot.opportunity_key:
            reasons.append(f"opportunity_key:{snapshot.opportunity_key}")

        if snapshot.persistence_ms > 0:
            reasons.append(f"persistence_ms:{snapshot.persistence_ms}")

        if snapshot.quote_validity is not None:
            reasons.append(f"quote_validity:{normalize_label(snapshot.quote_validity)}")

        return ScoreBreakdown(
            score=unit_score(score),
            confidence=unit_score(confidence),
            components=components,
            weights=weights,
            reasons=reasons,
            confirmations=list(dict.fromkeys(confirmations)),
        ).normalize()

    def _persistence_component(
        self,
        snapshot: SpreadCompositeSnapshot,
    ) -> float:
        if self.arb_config.min_persistence_ms <= 0:
            return 1.0 if snapshot.persistence_ms > 0 else 0.0

        return unit_score(
            snapshot.persistence_ms / max(self.arb_config.min_persistence_ms * 3, 1)
        )

    # ------------------------------------------------------------------
    # Source features / tags / metadata helpers
    # ------------------------------------------------------------------

    def _source_features(
        self,
        payload: CrossExchangeArbPayload,
    ) -> list[str]:
        features = [
            *cross_exchange_source_features(),
            *arbitrage_opportunity_source_features(),
            SPREADS_FEATURES.OPPORTUNITY,
            SPREADS_FEATURES.SNAPSHOT,
            SPREADS_FEATURES.SIGNAL,
            SPREADS_FEATURES.SPREAD_TYPE,
            SPREADS_FEATURES.SYMBOL,
            SPREADS_FEATURES.EXCHANGE_A,
            SPREADS_FEATURES.EXCHANGE_B,
            SPREADS_FEATURES.MARKET_TYPE_A,
            SPREADS_FEATURES.MARKET_TYPE_B,
            SPREADS_FEATURES.NET_EDGE,
            SPREADS_FEATURES.NET_EDGE_BPS,
            SPREADS_FEATURES.SPREAD_BPS,
            SPREADS_FEATURES.OPPORTUNITY_KEY,
            SPREADS_FEATURES.OPPORTUNITY_STATUS,
            SPREADS_FEATURES.PERSISTENCE_MS,
            SPREADS_FEATURES.QUOTE_VALIDITY,
            SPREADS_FEATURES.CONFIDENCE,
        ]

        return list(dict.fromkeys(features))

    def _tags(
        self,
        payload: CrossExchangeArbPayload,
    ) -> list[str]:
        tags = [
            self.arb_config.tag_spreads,
            self.arb_config.tag_cross_exchange,
            self.arb_config.tag_arbitrage,
            self.arb_config.tag_cross_exchange_arb,
            self.arb_config.tag_net_edge,
            f"side:{payload.side.value}",
            f"direction:{payload.spread_direction.lower()}",
        ]

        if payload.spread_direction == "LONG_A_SHORT_B":
            tags.append(self.arb_config.tag_long_a_short_b)

        if payload.spread_direction == "SHORT_A_LONG_B":
            tags.append(self.arb_config.tag_short_a_long_b)

        if payload.snapshot.opportunity_tradeable:
            tags.append(self.arb_config.tag_tradeable)

        if payload.snapshot.persistence_ms >= self.arb_config.min_persistence_ms:
            tags.append(self.arb_config.tag_persistent)

        if payload.snapshot.spread_type is not None:
            tags.append(f"type:{normalize_label(payload.snapshot.spread_type)}")

        if payload.snapshot.opportunity_status is not None:
            tags.append(f"status:{normalize_label(payload.snapshot.opportunity_status)}")

        return list(dict.fromkeys(tags))

    def _execution_hints(self) -> dict[str, Any]:
        """
        Execution hints only. Final EntryPlan/ExitPlan/RiskReadySignalPayload
        is owned by SignalProcessor / SignalBuilder.
        """
        return {
            "entry_offset_bps": self.arb_config.entry_offset_bps_hint,
            "stop_buffer_bps": self.arb_config.stop_buffer_bps_hint,
            "take_profit_bps": self.arb_config.take_profit_bps_hint,
            "arb_tp_multiplier": self.arb_config.arb_tp_multiplier_hint,
        }