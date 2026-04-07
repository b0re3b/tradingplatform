from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import FeatureSource, MarketRegime, Timeframe
from .exceptions import ValidationError
from .models import FeatureSnapshot, PortfolioSnapshot, PriceSnapshot, RegimeSnapshot


@dataclass(slots=True)
class DomainContext:
    """
    Універсальний контейнер для даних окремого аналітичного домену.
    Наприклад: orderflow, liquidity, whales тощо.
    """

    source: FeatureSource
    data: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.data, dict):
            raise ValidationError("DomainContext.data must be a dict")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.updated_at = datetime.utcnow()

    def update(self, values: dict[str, Any]) -> None:
        self.data.update(values)
        self.updated_at = datetime.utcnow()

    def clear(self) -> None:
        self.data.clear()
        self.updated_at = datetime.utcnow()


@dataclass(slots=True)
class ContextMetadata:
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    regime: MarketRegime = MarketRegime.UNKNOWN
    correlation_id: str | None = None
    trace_id: str | None = None
    source_event: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("ContextMetadata.symbol cannot be empty")


@dataclass(slots=True)
class StrategyContext:
    """
    Основний context object для strategy layer.

    Його читають усі strategy-класи.
    Він уже містить:
    - price snapshot
    - regime snapshot
    - portfolio snapshot
    - доменні feature-дані
    - feature_map з нормалізованими snapshots
    """

    symbol: str
    timestamp: datetime
    timeframe: Timeframe = Timeframe.M1

    price: PriceSnapshot | None = None
    regime: RegimeSnapshot | None = None
    portfolio: PortfolioSnapshot | None = None

    orderflow: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.ORDERFLOW)
    )
    liquidity: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.LIQUIDITY)
    )
    price_action: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.PRICE_ACTION)
    )
    liquidations: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.LIQUIDATIONS)
    )
    whales: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.WHALES)
    )
    spoofing: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.SPOOFING)
    )
    spreads: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.SPREADS)
    )
    funding: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.FUNDING)
    )
    open_interest: DomainContext = field(
        default_factory=lambda: DomainContext(source=FeatureSource.OPEN_INTEREST)
    )

    feature_map: dict[str, FeatureSnapshot] = field(default_factory=dict)
    freshness_map: dict[str, float] = field(default_factory=dict)
    metadata: ContextMetadata | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol.strip():
            raise ValidationError("StrategyContext.symbol cannot be empty")

        if self.price is not None:
            self.price.validate()

        if self.regime is not None:
            self.regime.validate()

        if self.portfolio is not None:
            self.portfolio.validate()

        for domain in self.iter_domains():
            domain.validate()

        for snapshot in self.feature_map.values():
            snapshot.validate()

        if self.metadata is not None:
            self.metadata.validate()

    def iter_domains(self) -> list[DomainContext]:
        return [
            self.orderflow,
            self.liquidity,
            self.price_action,
            self.liquidations,
            self.whales,
            self.spoofing,
            self.spreads,
            self.funding,
            self.open_interest,
        ]

    def get_domain(self, source: FeatureSource) -> DomainContext:
        mapping = {
            FeatureSource.ORDERFLOW: self.orderflow,
            FeatureSource.LIQUIDITY: self.liquidity,
            FeatureSource.PRICE_ACTION: self.price_action,
            FeatureSource.LIQUIDATIONS: self.liquidations,
            FeatureSource.WHALES: self.whales,
            FeatureSource.SPOOFING: self.spoofing,
            FeatureSource.SPREADS: self.spreads,
            FeatureSource.FUNDING: self.funding,
            FeatureSource.OPEN_INTEREST: self.open_interest,
        }
        return mapping[source]

    def has_feature(self, name: str) -> bool:
        return name in self.feature_map

    def get_feature_snapshot(self, name: str) -> FeatureSnapshot | None:
        return self.feature_map.get(name)

    def get_feature(self, name: str, default: Any = None) -> Any:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return snapshot.value

    def get_normalized_feature(self, name: str, default: float | None = None) -> float | None:
        snapshot = self.feature_map.get(name)
        if snapshot is None:
            return default
        return snapshot.normalized_value if snapshot.normalized_value is not None else default

    def put_feature(self, snapshot: FeatureSnapshot) -> None:
        snapshot.validate()
        self.feature_map[snapshot.name] = snapshot

    def remove_feature(self, name: str) -> None:
        self.feature_map.pop(name, None)

    def update_domain_data(self, source: FeatureSource, values: dict[str, Any]) -> None:
        domain = self.get_domain(source)
        domain.update(values)

    def set_price(self, price: PriceSnapshot) -> None:
        price.validate()
        self.price = price

    def set_regime(self, regime: RegimeSnapshot) -> None:
        regime.validate()
        self.regime = regime

    def set_portfolio(self, portfolio: PortfolioSnapshot) -> None:
        portfolio.validate()
        self.portfolio = portfolio

    @property
    def current_regime(self) -> MarketRegime:
        if self.regime is None:
            return MarketRegime.UNKNOWN
        return self.regime.regime

    @property
    def mid_price(self) -> float | None:
        if self.price is None:
            return None
        return self.price.mid_price