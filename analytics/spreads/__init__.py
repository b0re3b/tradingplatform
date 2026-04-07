from .base import BaseSpreadAnalyzer
from .config import (
    BaseSpreadConfig,
    CrossExchangeSpreadConfig,
    SpotFuturesSpreadConfig,
)
from .cross_exchange_analyzer import CrossExchangeSpreadAnalyzer
from .enums import (
    InstrumentType,
    OpportunityStatus,
    PricingSource,
    QuoteValidity,
    SpreadDirection,
    SpreadRegime,
    SpreadSignalType,
    SpreadType,
)
from .models import (
    ArbitrageOpportunity,
    FundingSnapshot,
    QuoteSnapshot,
    RollingStats,
    SpreadSignal,
    SpreadSnapshot,
)
from .spot_futures_analyzer import SpotFuturesSpreadAnalyzer
from .spread_analyzer import SpreadAnalyzer
from .spread_costs import SpreadCostBreakdown
from .spread_opportunity_detector import (
    OpportunityDetectionResult,
    SpreadOpportunityDetector,
)
from .spread_regime_detector import (
    RegimeDetectionResult,
    RegimeShiftResult,
    SpreadRegimeDetector,
)
from .spread_signal_engine import SignalEngineResult, SpreadSignalEngine
from .spread_utils import RollingDecimalWindow

__all__ = [
    # Facade / analyzers
    "SpreadAnalyzer",
    "BaseSpreadAnalyzer",
    "SpotFuturesSpreadAnalyzer",
    "CrossExchangeSpreadAnalyzer",

    # Config
    "BaseSpreadConfig",
    "SpotFuturesSpreadConfig",
    "CrossExchangeSpreadConfig",

    # Enums
    "SpreadType",
    "InstrumentType",
    "SpreadSignalType",
    "SpreadDirection",
    "SpreadRegime",
    "OpportunityStatus",
    "QuoteValidity",
    "PricingSource",

    # Models
    "QuoteSnapshot",
    "FundingSnapshot",
    "RollingStats",
    "SpreadSnapshot",
    "SpreadSignal",
    "ArbitrageOpportunity",

    # Regime detector
    "SpreadRegimeDetector",
    "RegimeDetectionResult",
    "RegimeShiftResult",

    # Signal engine
    "SpreadSignalEngine",
    "SignalEngineResult",

    # Opportunity detector
    "SpreadOpportunityDetector",
    "OpportunityDetectionResult",

    # Costs
    "SpreadCostBreakdown",

    # Utils public objects
    "RollingDecimalWindow",
]