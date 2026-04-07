from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from .config import CrossExchangeSpreadConfig
from .enums import InstrumentType, OpportunityStatus, SpreadType
from .models import ArbitrageOpportunity, QuoteSnapshot, SpreadSnapshot
from .spread_costs import (
    SpreadCostBreakdown,
    calculate_cost_breakdown,
    edge_bps_after_costs,
    resolve_trade_quantity,
)
from .spread_utils import DECIMAL_ZERO, spread_bps, spread_pct


@dataclass(slots=True)
class OpportunityDetectionResult:
    opportunity: ArbitrageOpportunity | None
    costs: SpreadCostBreakdown | None = None
    reason: str | None = None

    @property
    def found(self) -> bool:
        return self.opportunity is not None

    @property
    def is_profitable(self) -> bool:
        return self.opportunity is not None and self.opportunity.is_profitable


class SpreadOpportunityDetector:
    """
    Доменний detector для cross-exchange arbitrage opportunities.

    Відповідальність:
    - знайти найкращі buy/sell legs
    - порахувати gross edge / net edge
    - врахувати fees / slippage / safety buffer
    - побудувати ArbitrageOpportunity
    - оновити / expire opportunity

    Не відповідає за:
    - EventBus publish
    - storage
    - lifecycle analyzer-а
    """

    def __init__(self, config: CrossExchangeSpreadConfig) -> None:
        self._config = config

    def detect_from_quotes(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        *,
        quantity: Decimal | None = None,
        timestamp: datetime | None = None,
    ) -> OpportunityDetectionResult:
        if quote_a.symbol != quote_b.symbol:
            return OpportunityDetectionResult(
                opportunity=None,
                reason="symbol_mismatch",
            )

        if quote_a.instrument_type != quote_b.instrument_type:
            return OpportunityDetectionResult(
                opportunity=None,
                reason="instrument_type_mismatch",
            )

        if quote_a.instrument_type not in self._config.allowed_instrument_types:
            return OpportunityDetectionResult(
                opportunity=None,
                reason="instrument_type_not_allowed",
            )

        buy_quote, sell_quote = self._select_best_legs(quote_a, quote_b)
        if buy_quote is None or sell_quote is None:
            return OpportunityDetectionResult(
                opportunity=None,
                reason="no_positive_gross_edge",
            )

        trade_quantity = resolve_trade_quantity(self._config, quantity)

        costs = calculate_cost_breakdown(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            quantity=trade_quantity,
            buy_exchange=buy_quote.exchange,
            sell_exchange=sell_quote.exchange,
            config=self._config,
        )

        if costs.gross_edge <= DECIMAL_ZERO:
            return OpportunityDetectionResult(
                opportunity=None,
                costs=costs,
                reason="non_positive_gross_edge_after_leg_selection",
            )

        reference_notional = (buy_quote.ask or buy_quote.mid_price or Decimal("0")) * trade_quantity
        net_edge_bps = edge_bps_after_costs(
            net_edge=costs.net_edge,
            reference_notional=reference_notional,
        )

        if net_edge_bps < self._config.arbitrage_min_bps:
            return OpportunityDetectionResult(
                opportunity=None,
                costs=costs,
                reason="net_edge_bps_below_threshold",
            )

        opportunity = self._build_opportunity(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            quantity=trade_quantity,
            costs=costs,
            timestamp=timestamp,
        )

        return OpportunityDetectionResult(
            opportunity=opportunity,
            costs=costs,
            reason="opportunity_detected",
        )

    def detect_from_snapshot(
        self,
        snapshot: SpreadSnapshot,
        *,
        buy_quote: QuoteSnapshot | None = None,
        sell_quote: QuoteSnapshot | None = None,
        quantity: Decimal | None = None,
        timestamp: datetime | None = None,
    ) -> OpportunityDetectionResult:
        if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
            return OpportunityDetectionResult(
                opportunity=None,
                reason="unsupported_spread_type",
            )

        if buy_quote is None or sell_quote is None:
            metadata_buy_exchange = snapshot.metadata.get("buy_exchange")
            metadata_sell_exchange = snapshot.metadata.get("sell_exchange")

            if not metadata_buy_exchange or not metadata_sell_exchange:
                return OpportunityDetectionResult(
                    opportunity=None,
                    reason="missing_quotes_and_snapshot_metadata",
                )

            buy_price_raw = snapshot.metadata.get("buy_price")
            sell_price_raw = snapshot.metadata.get("sell_price")
            gross_edge_raw = snapshot.metadata.get("gross_edge")

            if buy_price_raw is None or sell_price_raw is None or gross_edge_raw is None:
                return OpportunityDetectionResult(
                    opportunity=None,
                    reason="snapshot_metadata_incomplete",
                )

            trade_quantity = resolve_trade_quantity(self._config, quantity)
            buy_price = Decimal(str(buy_price_raw))
            sell_price = Decimal(str(sell_price_raw))
            gross_edge_per_unit = Decimal(str(gross_edge_raw))
            gross_edge = gross_edge_per_unit * trade_quantity

            estimated_fees = snapshot.estimated_fees or DECIMAL_ZERO
            estimated_slippage = snapshot.estimated_slippage or DECIMAL_ZERO
            safety_buffer = Decimal("0")
            net_edge = snapshot.net_spread if snapshot.net_spread is not None else DECIMAL_ZERO

            costs = SpreadCostBreakdown(
                gross_edge=gross_edge,
                estimated_fees=estimated_fees,
                estimated_slippage=estimated_slippage,
                safety_buffer=safety_buffer,
                net_edge=net_edge,
            )

            reference_notional = buy_price * trade_quantity
            net_edge_bps = edge_bps_after_costs(
                net_edge=net_edge,
                reference_notional=reference_notional,
            )

            if net_edge <= DECIMAL_ZERO:
                return OpportunityDetectionResult(
                    opportunity=None,
                    costs=costs,
                    reason="snapshot_net_edge_not_profitable",
                )

            if net_edge_bps < self._config.arbitrage_min_bps:
                return OpportunityDetectionResult(
                    opportunity=None,
                    costs=costs,
                    reason="snapshot_net_edge_bps_below_threshold",
                )

            opportunity = ArbitrageOpportunity(
                symbol=snapshot.symbol,
                buy_exchange=metadata_buy_exchange,
                sell_exchange=metadata_sell_exchange,
                buy_instrument_type=snapshot.leg_a_type,
                sell_instrument_type=snapshot.leg_b_type,
                buy_price=buy_price,
                sell_price=sell_price,
                gross_edge=gross_edge,
                estimated_fees=estimated_fees,
                estimated_slippage=estimated_slippage,
                net_edge=net_edge,
                spread_pct=snapshot.spread_pct,
                spread_bps=snapshot.spread_bps,
                confidence=self._confidence_from_costs(costs),
                status=OpportunityStatus.ACTIVE,
                timestamp=timestamp or snapshot.timestamp,
                expires_at=(timestamp or snapshot.timestamp) + timedelta(seconds=self._config.cooldown_seconds),
                metadata={
                    "source": "snapshot",
                    "symbol": snapshot.symbol,
                    "buy_exchange": metadata_buy_exchange,
                    "sell_exchange": metadata_sell_exchange,
                    "quantity": str(trade_quantity),
                    "snapshot_timestamp": snapshot.timestamp.isoformat(),
                },
            )

            return OpportunityDetectionResult(
                opportunity=opportunity,
                costs=costs,
                reason="opportunity_from_snapshot",
            )

        return self.detect_from_quotes(
            buy_quote,
            sell_quote,
            quantity=quantity,
            timestamp=timestamp,
        )

    def expire_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        now: datetime | None = None,
    ) -> ArbitrageOpportunity:
        current_time = now or datetime.utcnow()

        if opportunity.expires_at is not None and current_time >= opportunity.expires_at:
            opportunity.status = OpportunityStatus.EXPIRED

        return opportunity

    def reject_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        reason: str | None = None,
    ) -> ArbitrageOpportunity:
        opportunity.status = OpportunityStatus.REJECTED
        if reason is not None:
            opportunity.metadata["reject_reason"] = reason
        return opportunity

    def mark_executed(
        self,
        opportunity: ArbitrageOpportunity,
        execution_metadata: dict[str, str] | None = None,
    ) -> ArbitrageOpportunity:
        opportunity.status = OpportunityStatus.EXECUTED
        if execution_metadata:
            opportunity.metadata.update(execution_metadata)
        return opportunity

    def is_opportunity_active(
        self,
        opportunity: ArbitrageOpportunity,
        now: datetime | None = None,
    ) -> bool:
        current_time = now or datetime.utcnow()

        if opportunity.status != OpportunityStatus.ACTIVE:
            return False

        if opportunity.expires_at is None:
            return True

        return current_time < opportunity.expires_at

    def _select_best_legs(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[QuoteSnapshot | None, QuoteSnapshot | None]:
        if quote_a.ask is None or quote_a.bid is None:
            return None, None

        if quote_b.ask is None or quote_b.bid is None:
            return None, None

        edge_a_to_b = quote_b.bid - quote_a.ask
        edge_b_to_a = quote_a.bid - quote_b.ask

        if edge_a_to_b >= edge_b_to_a and edge_a_to_b > DECIMAL_ZERO:
            return quote_a, quote_b

        if edge_b_to_a > edge_a_to_b and edge_b_to_a > DECIMAL_ZERO:
            return quote_b, quote_a

        return None, None

    def _build_opportunity(
        self,
        buy_quote: QuoteSnapshot,
        sell_quote: QuoteSnapshot,
        quantity: Decimal,
        costs: SpreadCostBreakdown,
        timestamp: datetime | None = None,
    ) -> ArbitrageOpportunity:
        buy_price = buy_quote.ask
        sell_price = sell_quote.bid

        if buy_price is None or sell_price is None:
            raise ValueError("buy_quote.ask and sell_quote.bid must be available")

        reference_buy_notional = buy_price * quantity
        gross_edge_per_unit = sell_price - buy_price

        return ArbitrageOpportunity(
            symbol=buy_quote.symbol,
            buy_exchange=buy_quote.exchange,
            sell_exchange=sell_quote.exchange,
            buy_instrument_type=buy_quote.instrument_type,
            sell_instrument_type=sell_quote.instrument_type,
            buy_price=buy_price,
            sell_price=sell_price,
            gross_edge=costs.gross_edge,
            estimated_fees=costs.estimated_fees,
            estimated_slippage=costs.estimated_slippage,
            net_edge=costs.net_edge,
            spread_pct=spread_pct(gross_edge_per_unit, buy_price),
            spread_bps=spread_bps(gross_edge_per_unit, buy_price),
            confidence=self._confidence_from_costs(costs),
            status=OpportunityStatus.ACTIVE,
            timestamp=timestamp or datetime.utcnow(),
            expires_at=(timestamp or datetime.utcnow()) + timedelta(seconds=self._config.cooldown_seconds),
            metadata={
                "source": "quotes",
                "quantity": str(quantity),
                "reference_buy_notional": str(reference_buy_notional),
                "safety_buffer_bps": str(self._config.safety_buffer_bps),
                "arbitrage_min_bps": str(self._config.arbitrage_min_bps),
            },
        )

    @staticmethod
    def _confidence_from_costs(costs: SpreadCostBreakdown) -> Decimal:
        if costs.gross_edge <= DECIMAL_ZERO:
            return Decimal("0.20")

        if costs.total_costs <= DECIMAL_ZERO:
            return Decimal("0.90")

        coverage_ratio = costs.net_edge / costs.total_costs

        if coverage_ratio >= Decimal("3"):
            return Decimal("0.93")
        if coverage_ratio >= Decimal("2"):
            return Decimal("0.85")
        if coverage_ratio >= Decimal("1"):
            return Decimal("0.75")
        if coverage_ratio > DECIMAL_ZERO:
            return Decimal("0.60")

        return Decimal("0.30")