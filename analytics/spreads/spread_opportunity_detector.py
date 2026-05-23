from __future__ import annotations
from core.logger import get_logger

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from .config import CrossExchangeSpreadConfig
from .enums import OpportunityStatus, SpreadType
from .models import ArbitrageOpportunity, QuoteSnapshot, SpreadSnapshot
from .spread_costs import (
    SpreadCostBreakdown,
    calculate_cost_breakdown,
    edge_bps_after_costs,
    reference_notional_from_quote,
    resolve_trade_quantity,
)
from .spread_utils import (
    DECIMAL_ONE,
    DECIMAL_ZERO,
    normalize_exchange,
    spread_bps,
    spread_pct,
    to_decimal,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_OPPORTUNITY_TTL_SECONDS = 10.0


# ============================================================
# Enums / reason codes
# ============================================================

class OpportunityDetectionReason(str, Enum):
    """
    Stable reason codes for opportunity detection.

    Ці значення можна безпечно використовувати в metadata, dashboard,
    storage або metrics analyzer-а.
    """

    OPPORTUNITY_DETECTED = "opportunity_detected"
    OPPORTUNITY_FROM_SNAPSHOT = "opportunity_from_snapshot"

    SYMBOL_MISMATCH = "symbol_mismatch"
    INSTRUMENT_TYPE_MISMATCH = "instrument_type_mismatch"
    INSTRUMENT_TYPE_NOT_ALLOWED = "instrument_type_not_allowed"

    UNSUPPORTED_SPREAD_TYPE = "unsupported_spread_type"
    MISSING_QUOTES_AND_SNAPSHOT_METADATA = "missing_quotes_and_snapshot_metadata"
    SNAPSHOT_METADATA_INCOMPLETE = "snapshot_metadata_incomplete"
    SNAPSHOT_METADATA_INVALID = "snapshot_metadata_invalid"
    SNAPSHOT_NET_EDGE_NOT_PROFITABLE = "snapshot_net_edge_not_profitable"
    SNAPSHOT_NET_EDGE_BPS_BELOW_THRESHOLD = "snapshot_net_edge_bps_below_threshold"

    NO_POSITIVE_GROSS_EDGE = "no_positive_gross_edge"
    NON_POSITIVE_GROSS_EDGE_AFTER_LEG_SELECTION = "non_positive_gross_edge_after_leg_selection"
    NET_EDGE_NOT_PROFITABLE = "net_edge_not_profitable"
    NET_EDGE_BPS_BELOW_THRESHOLD = "net_edge_bps_below_threshold"

    INVALID_QUANTITY = "invalid_quantity"
    INVALID_QUOTE_PRICES = "invalid_quote_prices"


# ============================================================
# Result model
# ============================================================

@dataclass(slots=True)
class OpportunityDetectionResult:
    """
    Результат пошуку arbitrage opportunity.

    Не містить runtime-залежностей і може використовуватись analyzer-ом
    для публікації EventBus подій або metrics.
    """

    opportunity: ArbitrageOpportunity | None
    costs: SpreadCostBreakdown | None = None
    reason: OpportunityDetectionReason | str | None = None
    net_edge_bps: Decimal | None = None
    quantity: Decimal | None = None
    metadata: dict[str, Any] | None = None

    @property
    def found(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "found", _analytics_args)
        except Exception:
            pass
        return self.opportunity is not None

    @property
    def is_profitable(self) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_profitable", _analytics_args)
        except Exception:
            pass
        return self.opportunity is not None and self.opportunity.is_profitable

    @property
    def reason_value(self) -> str | None:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reason_value", _analytics_args)
        except Exception:
            pass
        if self.reason is None:
            return None
        if isinstance(self.reason, OpportunityDetectionReason):
            return self.reason.value
        return str(self.reason)

    def to_payload(self) -> dict[str, Any]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "to_payload", _analytics_args)
        except Exception:
            pass
        opportunity_payload = None
        if self.opportunity is not None:
            to_payload = getattr(self.opportunity, "to_payload", None)
            opportunity_payload = to_payload() if callable(to_payload) else self.opportunity

        costs_payload = None
        if self.costs is not None:
            to_payload = getattr(self.costs, "to_payload", None)
            costs_payload = to_payload() if callable(to_payload) else self.costs

        return {
            "found": self.found,
            "is_profitable": self.is_profitable,
            "reason": self.reason_value,
            "net_edge_bps": _decimal_to_payload(self.net_edge_bps),
            "quantity": _decimal_to_payload(self.quantity),
            "opportunity": opportunity_payload,
            "costs": costs_payload,
            "metadata": dict(self.metadata or {}),
        }


# ============================================================
# Detector
# ============================================================

class SpreadOpportunityDetector:
    """
    Pure domain detector для cross-exchange arbitrage opportunities.

    Відповідальність:
    - вибрати buy/sell legs;
    - порахувати gross edge / net edge;
    - врахувати fees / slippage / safety buffer;
    - перевірити threshold у bps;
    - побудувати ArbitrageOpportunity;
    - оновити opportunity lifecycle status.

    Не відповідає за:
    - EventBus publish;
    - Scheduler jobs;
    - logging;
    - storage;
    - execution;
    - risk approval;
    - lifecycle analyzer-а.
    """

    def __init__(self, config: CrossExchangeSpreadConfig) -> None:
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
        self._config = config

    # ------------------------------------------------------------------
    # Detection from quotes
    # ------------------------------------------------------------------

    def detect_from_quotes(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
        *,
        quantity: Decimal | None = None,
        timestamp: datetime | None = None,
    ) -> OpportunityDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_from_quotes", _analytics_args)
        except Exception:
            pass
        current_time = timestamp or _utcnow()

        if quote_a.symbol != quote_b.symbol:
            return self._empty_result(
                OpportunityDetectionReason.SYMBOL_MISMATCH,
                quantity=quantity,
                metadata={
                    "quote_a_symbol": quote_a.symbol,
                    "quote_b_symbol": quote_b.symbol,
                },
            )

        if quote_a.instrument_type != quote_b.instrument_type:
            return self._empty_result(
                OpportunityDetectionReason.INSTRUMENT_TYPE_MISMATCH,
                quantity=quantity,
                metadata={
                    "quote_a_instrument_type": quote_a.instrument_type.value,
                    "quote_b_instrument_type": quote_b.instrument_type.value,
                },
            )

        if not self._is_instrument_type_allowed(quote_a):
            return self._empty_result(
                OpportunityDetectionReason.INSTRUMENT_TYPE_NOT_ALLOWED,
                quantity=quantity,
                metadata={
                    "instrument_type": quote_a.instrument_type.value,
                },
            )

        trade_quantity = self._resolve_quantity(quantity)
        if trade_quantity <= DECIMAL_ZERO:
            return self._empty_result(
                OpportunityDetectionReason.INVALID_QUANTITY,
                quantity=trade_quantity,
            )

        buy_quote, sell_quote = self._select_best_legs(quote_a, quote_b)
        if buy_quote is None or sell_quote is None:
            return self._empty_result(
                OpportunityDetectionReason.NO_POSITIVE_GROSS_EDGE,
                quantity=trade_quantity,
                metadata={
                    "quote_a_exchange": quote_a.exchange,
                    "quote_b_exchange": quote_b.exchange,
                },
            )

        costs = calculate_cost_breakdown(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            quantity=trade_quantity,
            buy_exchange=buy_quote.exchange,
            sell_exchange=sell_quote.exchange,
            config=self._config,
        )

        if costs.gross_edge <= DECIMAL_ZERO:
            return self._empty_result(
                OpportunityDetectionReason.NON_POSITIVE_GROSS_EDGE_AFTER_LEG_SELECTION,
                costs=costs,
                quantity=trade_quantity,
            )

        if costs.net_edge <= DECIMAL_ZERO:
            return self._empty_result(
                OpportunityDetectionReason.NET_EDGE_NOT_PROFITABLE,
                costs=costs,
                quantity=trade_quantity,
            )

        reference_notional = reference_notional_from_quote(
            buy_quote,
            trade_quantity,
            prefer_ask=True,
        )

        net_edge_bps = edge_bps_after_costs(
            net_edge=costs.net_edge,
            reference_notional=reference_notional,
        )

        if net_edge_bps < self._config.arbitrage_min_bps:
            return self._empty_result(
                OpportunityDetectionReason.NET_EDGE_BPS_BELOW_THRESHOLD,
                costs=costs,
                net_edge_bps=net_edge_bps,
                quantity=trade_quantity,
                metadata={
                    "threshold_bps": str(self._config.arbitrage_min_bps),
                },
            )

        opportunity = self._build_opportunity(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            quantity=trade_quantity,
            costs=costs,
            timestamp=current_time,
            source="quotes",
        )

        return OpportunityDetectionResult(
            opportunity=opportunity,
            costs=costs,
            reason=OpportunityDetectionReason.OPPORTUNITY_DETECTED,
            net_edge_bps=net_edge_bps,
            quantity=trade_quantity,
            metadata={
                "source": "quotes",
                "threshold_bps": str(self._config.arbitrage_min_bps),
                "reference_notional": str(reference_notional),
            },
        )

    # ------------------------------------------------------------------
    # Detection from snapshot
    # ------------------------------------------------------------------

    def detect_from_snapshot(
        self,
        snapshot: SpreadSnapshot,
        *,
        buy_quote: QuoteSnapshot | None = None,
        sell_quote: QuoteSnapshot | None = None,
        quantity: Decimal | None = None,
        timestamp: datetime | None = None,
    ) -> OpportunityDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "detect_from_snapshot", _analytics_args)
        except Exception:
            pass
        if snapshot.spread_type != SpreadType.CROSS_EXCHANGE:
            return self._empty_result(
                OpportunityDetectionReason.UNSUPPORTED_SPREAD_TYPE,
                quantity=quantity,
                metadata={
                    "spread_type": snapshot.spread_type.value,
                },
            )

        if buy_quote is not None and sell_quote is not None:
            return self.detect_from_quotes(
                buy_quote,
                sell_quote,
                quantity=quantity,
                timestamp=timestamp,
            )

        return self._detect_from_snapshot_metadata(
            snapshot,
            quantity=quantity,
            timestamp=timestamp,
        )

    def _detect_from_snapshot_metadata(
        self,
        snapshot: SpreadSnapshot,
        *,
        quantity: Decimal | None = None,
        timestamp: datetime | None = None,
    ) -> OpportunityDetectionResult:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_detect_from_snapshot_metadata", _analytics_args)
        except Exception:
            pass
        metadata = dict(snapshot.metadata)
        current_time = timestamp or snapshot.timestamp

        buy_exchange = metadata.get("buy_exchange")
        sell_exchange = metadata.get("sell_exchange")

        if not buy_exchange or not sell_exchange:
            return self._empty_result(
                OpportunityDetectionReason.MISSING_QUOTES_AND_SNAPSHOT_METADATA,
                quantity=quantity,
            )

        buy_price = to_decimal(metadata.get("buy_price"))
        sell_price = to_decimal(metadata.get("sell_price"))
        gross_edge_raw = to_decimal(metadata.get("gross_edge"))

        if buy_price is None or sell_price is None or gross_edge_raw is None:
            return self._empty_result(
                OpportunityDetectionReason.SNAPSHOT_METADATA_INCOMPLETE,
                quantity=quantity,
                metadata={
                    "has_buy_price": buy_price is not None,
                    "has_sell_price": sell_price is not None,
                    "has_gross_edge": gross_edge_raw is not None,
                },
            )

        if buy_price <= DECIMAL_ZERO or sell_price <= DECIMAL_ZERO:
            return self._empty_result(
                OpportunityDetectionReason.SNAPSHOT_METADATA_INVALID,
                quantity=quantity,
                metadata={
                    "buy_price": str(buy_price),
                    "sell_price": str(sell_price),
                },
            )

        trade_quantity = self._resolve_quantity(quantity)

        gross_edge = gross_edge_raw * trade_quantity
        estimated_fees = snapshot.estimated_fees or DECIMAL_ZERO
        estimated_slippage = snapshot.estimated_slippage or DECIMAL_ZERO
        safety_buffer = to_decimal(metadata.get("safety_buffer"), default=DECIMAL_ZERO) or DECIMAL_ZERO
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
            return self._empty_result(
                OpportunityDetectionReason.SNAPSHOT_NET_EDGE_NOT_PROFITABLE,
                costs=costs,
                net_edge_bps=net_edge_bps,
                quantity=trade_quantity,
            )

        if net_edge_bps < self._config.arbitrage_min_bps:
            return self._empty_result(
                OpportunityDetectionReason.SNAPSHOT_NET_EDGE_BPS_BELOW_THRESHOLD,
                costs=costs,
                net_edge_bps=net_edge_bps,
                quantity=trade_quantity,
                metadata={
                    "threshold_bps": str(self._config.arbitrage_min_bps),
                },
            )

        opportunity = ArbitrageOpportunity(
            symbol=snapshot.symbol,
            buy_exchange=normalize_exchange(str(buy_exchange)),
            sell_exchange=normalize_exchange(str(sell_exchange)),
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
            timestamp=current_time,
            expires_at=self._expires_at(current_time),
            metadata={
                "source": "snapshot",
                "quantity": str(trade_quantity),
                "snapshot_timestamp": snapshot.timestamp.isoformat(),
                "reference_notional": str(reference_notional),
                "threshold_bps": str(self._config.arbitrage_min_bps),
                "safety_buffer": str(safety_buffer),
                **_safe_metadata_subset(metadata),
            },
        )

        return OpportunityDetectionResult(
            opportunity=opportunity,
            costs=costs,
            reason=OpportunityDetectionReason.OPPORTUNITY_FROM_SNAPSHOT,
            net_edge_bps=net_edge_bps,
            quantity=trade_quantity,
            metadata={
                "source": "snapshot",
                "threshold_bps": str(self._config.arbitrage_min_bps),
                "reference_notional": str(reference_notional),
            },
        )

    # ------------------------------------------------------------------
    # Opportunity lifecycle helpers
    # ------------------------------------------------------------------

    def expire_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        now: datetime | None = None,
    ) -> ArbitrageOpportunity:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "expire_opportunity", _analytics_args)
        except Exception:
            pass
        current_time = now or _utcnow()

        if opportunity.expires_at is not None and current_time >= opportunity.expires_at:
            if hasattr(opportunity, "mark_expired"):
                opportunity.mark_expired()
            else:
                opportunity.status = OpportunityStatus.EXPIRED

        return opportunity

    def reject_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        reason: str | None = None,
    ) -> ArbitrageOpportunity:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "reject_opportunity", _analytics_args)
        except Exception:
            pass
        if hasattr(opportunity, "mark_rejected"):
            opportunity.mark_rejected()
        else:
            opportunity.status = OpportunityStatus.REJECTED

        if reason is not None:
            opportunity.metadata["reject_reason"] = reason

        opportunity.metadata["rejected_at"] = _utcnow().isoformat()
        return opportunity

    def mark_executed(
        self,
        opportunity: ArbitrageOpportunity,
        execution_metadata: dict[str, str] | None = None,
    ) -> ArbitrageOpportunity:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "mark_executed", _analytics_args)
        except Exception:
            pass
        if hasattr(opportunity, "mark_executed"):
            opportunity.mark_executed()
        else:
            opportunity.status = OpportunityStatus.EXECUTED

        opportunity.metadata["executed_at"] = _utcnow().isoformat()

        if execution_metadata:
            opportunity.metadata.update(execution_metadata)

        return opportunity

    def is_opportunity_active(
        self,
        opportunity: ArbitrageOpportunity,
        now: datetime | None = None,
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "is_opportunity_active", _analytics_args)
        except Exception:
            pass
        current_time = now or _utcnow()

        if opportunity.status != OpportunityStatus.ACTIVE:
            return False

        if opportunity.expires_at is None:
            return True

        return current_time < opportunity.expires_at

    # ------------------------------------------------------------------
    # Internal quote/opportunity builders
    # ------------------------------------------------------------------

    def _select_best_legs(
        self,
        quote_a: QuoteSnapshot,
        quote_b: QuoteSnapshot,
    ) -> tuple[QuoteSnapshot | None, QuoteSnapshot | None]:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_select_best_legs", _analytics_args)
        except Exception:
            pass
        if not _has_valid_bid_ask(quote_a) or not _has_valid_bid_ask(quote_b):
            return None, None

        assert quote_a.ask is not None
        assert quote_a.bid is not None
        assert quote_b.ask is not None
        assert quote_b.bid is not None

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
        timestamp: datetime,
        *,
        source: str,
    ) -> ArbitrageOpportunity:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_build_opportunity", _analytics_args)
        except Exception:
            pass
        buy_price = buy_quote.ask
        sell_price = sell_quote.bid

        if buy_price is None or sell_price is None:
            raise ValueError("buy_quote.ask and sell_quote.bid must be available")

        if buy_price <= DECIMAL_ZERO or sell_price <= DECIMAL_ZERO:
            raise ValueError("buy_quote.ask and sell_quote.bid must be > 0")

        gross_edge_per_unit = sell_price - buy_price
        reference_buy_notional = buy_price * quantity

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
            timestamp=timestamp,
            expires_at=self._expires_at(timestamp),
            metadata={
                "source": source,
                "quantity": str(quantity),
                "reference_buy_notional": str(reference_buy_notional),
                "safety_buffer_bps": str(self._config.safety_buffer_bps),
                "arbitrage_min_bps": str(self._config.arbitrage_min_bps),
                "buy_quote_timestamp": buy_quote.timestamp.isoformat(),
                "sell_quote_timestamp": sell_quote.timestamp.isoformat(),
            },
        )

    def _resolve_quantity(self, quantity: Decimal | None) -> Decimal:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_resolve_quantity", _analytics_args)
        except Exception:
            pass
        return resolve_trade_quantity(self._config, quantity)

    def _expires_at(self, timestamp: datetime) -> datetime:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_expires_at", _analytics_args)
        except Exception:
            pass
        ttl_seconds = getattr(
            self._config,
            "opportunity_ttl_seconds",
            DEFAULT_OPPORTUNITY_TTL_SECONDS,
        )
        return timestamp + timedelta(seconds=float(ttl_seconds))

    def _is_instrument_type_allowed(self, quote: QuoteSnapshot) -> bool:
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_is_instrument_type_allowed", _analytics_args)
        except Exception:
            pass
        checker = getattr(self._config, "is_instrument_type_allowed", None)
        if callable(checker):
            return bool(checker(quote.instrument_type))

        return quote.instrument_type in self._config.allowed_instrument_types

    @staticmethod
    def _confidence_from_costs(costs: SpreadCostBreakdown) -> Decimal:
        """
        Heuristic confidence score у діапазоні [0, 1].

        Чим більший net_edge відносно total_costs, тим вища впевненість.
        """
        try:
            _analytics_class_name = "SpreadOpportunityDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_confidence_from_costs", _analytics_args)
        except Exception:
            pass
        if costs.gross_edge <= DECIMAL_ZERO:
            return Decimal("0.20")

        if costs.net_edge <= DECIMAL_ZERO:
            return Decimal("0.30")

        if costs.total_costs <= DECIMAL_ZERO:
            return Decimal("0.90")

        coverage_ratio = costs.net_edge / costs.total_costs

        if coverage_ratio >= Decimal("3"):
            return Decimal("0.93")
        if coverage_ratio >= Decimal("2"):
            return Decimal("0.85")
        if coverage_ratio >= DECIMAL_ONE:
            return Decimal("0.75")
        if coverage_ratio > DECIMAL_ZERO:
            return Decimal("0.60")

        return Decimal("0.30")

    @staticmethod
    def _empty_result(
        reason: OpportunityDetectionReason,
        *,
        costs: SpreadCostBreakdown | None = None,
        net_edge_bps: Decimal | None = None,
        quantity: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OpportunityDetectionResult:
        try:
            _analytics_class_name = "SpreadOpportunityDetector"
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
            _analytics_logger.debug("%s.%s entered | args=%s", _analytics_class_name, "_empty_result", _analytics_args)
        except Exception:
            pass
        return OpportunityDetectionResult(
            opportunity=None,
            costs=costs,
            reason=reason,
            net_edge_bps=net_edge_bps,
            quantity=quantity,
            metadata=metadata or {},
        )


# ============================================================
# Module helpers
# ============================================================

def _utcnow() -> datetime:
    return datetime.utcnow()


def _decimal_to_payload(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _has_valid_bid_ask(quote: QuoteSnapshot) -> bool:
    if quote.ask is None or quote.bid is None:
        return False

    if quote.ask <= DECIMAL_ZERO or quote.bid <= DECIMAL_ZERO:
        return False

    if quote.bid > quote.ask:
        return False

    return True


def _safe_metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Копіює тільки безпечні scalar metadata значення.

    Це захищає opportunity metadata від вкладених mutable-структур
    і випадкового протягування зайвих runtime-об'єктів.
    """
    safe: dict[str, Any] = {}

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, Decimal):
            safe[key] = str(value)
        elif isinstance(value, datetime):
            safe[key] = value.isoformat()

    return safe


__all__ = [
    "OpportunityDetectionReason",
    "OpportunityDetectionResult",
    "SpreadOpportunityDetector",
]