from __future__ import annotations

# ============================================================
# Facade / runtime analyzers
# ============================================================

from analytics.spreads.spread_analyzer import SpreadAnalyzer
from analytics.spreads.base import BaseSpreadAnalyzer
from analytics.spreads.spot_futures_analyzer import SpotFuturesSpreadAnalyzer
from analytics.spreads.cross_exchange_analyzer import CrossExchangeSpreadAnalyzer


# ============================================================
# Config
# ============================================================

from analytics.spreads.config import (
    BaseSpreadConfig,
    SpotFuturesSpreadConfig,
    CrossExchangeSpreadConfig,
)


# ============================================================
# Enums
# ============================================================

from analytics.spreads.enums import (
    StrEnumMixin,
    SpreadType,
    InstrumentType,
    SpreadSignalType,
    SpreadDirection,
    SpreadRegime,
    OpportunityStatus,
    QuoteValidity,
    PricingSource,
    parse_instrument_type,
    parse_spread_type,
    parse_pricing_source,
)


# ============================================================
# Models
# ============================================================

from analytics.spreads.models import (
    QuoteSnapshot,
    FundingSnapshot,
    RollingStats,
    SpreadSnapshot,
    SpreadSignal,
    ArbitrageOpportunity,
    model_to_payload,
)


# ============================================================
# Regime detector
# ============================================================

from analytics.spreads.spread_regime_detector import (
    RegimeDetectionResult,
    RegimeShiftResult,
    SpreadRegimeDetector,
)


# ============================================================
# Signal engine
# ============================================================

from analytics.spreads.spread_signal_engine import (
    SignalBuildReason,
    SignalBuildResult,
    SignalEngineResult,
    SpreadSignalEngine,
)


# ============================================================
# Opportunity detector
# ============================================================

from analytics.spreads.spread_opportunity_detector import (
    OpportunityDetectionReason,
    OpportunityDetectionResult,
    SpreadOpportunityDetector,
)


# ============================================================
# Costs
# ============================================================

from analytics.spreads.spread_costs import (
    CostSide,
    LiquiditySide,
    SpreadCostBreakdown,
    normalize_fee_overrides,
    get_fee_rate,
    estimate_fee_cost,
    estimate_total_fees,
    estimate_simple_slippage_ratio,
    estimate_simple_slippage,
    estimate_slippage_cost_from_quote,
    estimate_total_slippage,
    estimate_safety_buffer_cost,
    gross_edge_from_prices,
    net_edge_after_costs,
    edge_bps_after_costs,
    reference_notional_from_quote,
    resolve_trade_quantity,
    calculate_cost_breakdown,
)


# ============================================================
# Utils
# ============================================================

from analytics.spreads.spread_utils import (
    DECIMAL_ZERO,
    DECIMAL_ONE,
    DECIMAL_TWO,
    DECIMAL_100,
    DECIMAL_10_000,
    DEFAULT_QUANT,
    to_decimal,
    require_decimal,
    quantize_decimal,
    safe_abs,
    safe_min,
    safe_max,
    safe_div,
    clamp_decimal,
    is_positive,
    is_non_negative,
    midpoint,
    spread_abs,
    spread_pct,
    spread_bps,
    abs_spread_bps,
    basis_from_prices,
    basis_pct_from_prices,
    basis_bps_from_prices,
    funding_adjusted_spread,
    notional,
    infer_direction,
    infer_regime,
    normalize_symbol,
    normalize_exchange,
    normalize_pair_key,
    infer_instrument_type,
    now_utc,
    age_ms,
    quote_age_ms,
    is_quote_stale,
    validate_quote_snapshot,
    aligned_quotes,
    quote_time_diff_ms,
    decimal_mean,
    decimal_variance,
    decimal_std,
    compute_zscore,
    ema_next,
    percentile_rank,
    build_rolling_stats,
    RollingDecimalWindow,
)


__all__ = [
    # Facade / runtime analyzers
    "SpreadAnalyzer",
    "BaseSpreadAnalyzer",
    "SpotFuturesSpreadAnalyzer",
    "CrossExchangeSpreadAnalyzer",

    # Config
    "BaseSpreadConfig",
    "SpotFuturesSpreadConfig",
    "CrossExchangeSpreadConfig",

    # Enums
    "StrEnumMixin",
    "SpreadType",
    "InstrumentType",
    "SpreadSignalType",
    "SpreadDirection",
    "SpreadRegime",
    "OpportunityStatus",
    "QuoteValidity",
    "PricingSource",
    "parse_instrument_type",
    "parse_spread_type",
    "parse_pricing_source",

    # Models
    "QuoteSnapshot",
    "FundingSnapshot",
    "RollingStats",
    "SpreadSnapshot",
    "SpreadSignal",
    "ArbitrageOpportunity",
    "model_to_payload",

    # Regime detector
    "RegimeDetectionResult",
    "RegimeShiftResult",
    "SpreadRegimeDetector",

    # Signal engine
    "SignalBuildReason",
    "SignalBuildResult",
    "SignalEngineResult",
    "SpreadSignalEngine",

    # Opportunity detector
    "OpportunityDetectionReason",
    "OpportunityDetectionResult",
    "SpreadOpportunityDetector",

    # Costs
    "CostSide",
    "LiquiditySide",
    "SpreadCostBreakdown",
    "normalize_fee_overrides",
    "get_fee_rate",
    "estimate_fee_cost",
    "estimate_total_fees",
    "estimate_simple_slippage_ratio",
    "estimate_simple_slippage",
    "estimate_slippage_cost_from_quote",
    "estimate_total_slippage",
    "estimate_safety_buffer_cost",
    "gross_edge_from_prices",
    "net_edge_after_costs",
    "edge_bps_after_costs",
    "reference_notional_from_quote",
    "resolve_trade_quantity",
    "calculate_cost_breakdown",

    # Utils
    "DECIMAL_ZERO",
    "DECIMAL_ONE",
    "DECIMAL_TWO",
    "DECIMAL_100",
    "DECIMAL_10_000",
    "DEFAULT_QUANT",
    "to_decimal",
    "require_decimal",
    "quantize_decimal",
    "safe_abs",
    "safe_min",
    "safe_max",
    "safe_div",
    "clamp_decimal",
    "is_positive",
    "is_non_negative",
    "midpoint",
    "spread_abs",
    "spread_pct",
    "spread_bps",
    "abs_spread_bps",
    "basis_from_prices",
    "basis_pct_from_prices",
    "basis_bps_from_prices",
    "funding_adjusted_spread",
    "notional",
    "infer_direction",
    "infer_regime",
    "normalize_symbol",
    "normalize_exchange",
    "normalize_pair_key",
    "infer_instrument_type",
    "now_utc",
    "age_ms",
    "quote_age_ms",
    "is_quote_stale",
    "validate_quote_snapshot",
    "aligned_quotes",
    "quote_time_diff_ms",
    "decimal_mean",
    "decimal_variance",
    "decimal_std",
    "compute_zscore",
    "ema_next",
    "percentile_rank",
    "build_rolling_stats",
    "RollingDecimalWindow",
]