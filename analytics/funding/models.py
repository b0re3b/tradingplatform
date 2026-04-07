from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .enums import (
    FundingBias,
    FundingDataSource,
    FundingDivergenceType,
    FundingEventType,
    FundingExtremeType,
    FundingFlipType,
    FundingPressureDirection,
    FundingPressureLevel,
    FundingRegime,
    FundingSignalType,
    FundingTimeframe,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(slots=True)
class FundingSnapshot:
    """
    Сирий або вже нормалізований funding snapshot для одного символу.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    funding_rate: float = 0.0
    predicted_funding_rate: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    open_interest: float | None = None
    volume_24h: float | None = None
    next_funding_time: datetime | None = None
    event_time: datetime = field(default_factory=utc_now)
    received_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.received_at = ensure_utc(self.received_at)

        if self.next_funding_time is not None:
            self.next_funding_time = ensure_utc(self.next_funding_time)

    @property
    def basis(self) -> float | None:
        """
        Відносне відхилення mark від index, якщо обидва значення доступні.
        """
        if self.mark_price is None or self.index_price is None or self.index_price == 0:
            return None
        return (self.mark_price - self.index_price) / self.index_price

    @property
    def funding_sign(self) -> int:
        if self.funding_rate > 0:
            return 1
        if self.funding_rate < 0:
            return -1
        return 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange",):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        for key in ("event_time", "received_at", "next_funding_time"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        data["basis"] = self.basis
        data["funding_sign"] = self.funding_sign
        return data


@dataclass(slots=True)
class FundingStatistics:
    """
    Агрегована статистика funding на заданому вікні.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    current_rate: float = 0.0
    mean_rate: float = 0.0
    median_rate: float = 0.0
    std_rate: float = 0.0
    min_rate: float = 0.0
    max_rate: float = 0.0

    zscore: float | None = None
    percentile: float | None = None

    sample_size: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.updated_at = ensure_utc(self.updated_at)

        if self.window_start is not None:
            self.window_start = ensure_utc(self.window_start)
        if self.window_end is not None:
            self.window_end = ensure_utc(self.window_end)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        for key in ("window_start", "window_end", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass(slots=True)
class FundingRegimeState:
    """
    Стан funding regime для символу.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    regime: FundingRegime = FundingRegime.UNKNOWN
    bias: FundingBias = FundingBias.NEUTRAL

    current_rate: float = 0.0
    mean_rate: float | None = None
    zscore: float | None = None
    percentile: float | None = None

    confidence: float = 0.0
    changed: bool = False
    previous_regime: FundingRegime | None = None

    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "regime", "bias", "previous_regime"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingExtremeEvent:
    """
    Подія екстремального funding.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    extreme_type: FundingExtremeType = FundingExtremeType.NONE
    regime: FundingRegime = FundingRegime.UNKNOWN
    funding_rate: float = 0.0

    zscore: float | None = None
    percentile: float | None = None

    severity: float = 0.0
    is_reversal_risk: bool = False
    is_squeeze_risk: bool = False

    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.severity = max(0.0, min(1.0, self.severity))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "extreme_type", "regime"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingDivergenceEvent:
    """
    Подія дивергенції funding з ціною, OI, CVD або ліквідаціями.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    divergence_type: FundingDivergenceType = FundingDivergenceType.NONE
    funding_rate: float = 0.0

    price_change_pct: float | None = None
    oi_change_pct: float | None = None
    cvd_change: float | None = None
    long_liquidations: float | None = None
    short_liquidations: float | None = None

    confidence: float = 0.0
    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "divergence_type"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingFlipEvent:
    """
    Подія зміни знаку funding.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    flip_type: FundingFlipType = FundingFlipType.NONE
    previous_rate: float = 0.0
    current_rate: float = 0.0

    flip_magnitude: float = 0.0
    confidence: float = 0.0

    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.flip_magnitude = abs(self.current_rate - self.previous_rate)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "flip_type"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingPressureState:
    """
    Оцінка накопиченого funding pressure / crowded positioning.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    direction: FundingPressureDirection = FundingPressureDirection.NEUTRAL
    level: FundingPressureLevel = FundingPressureLevel.UNKNOWN
    bias: FundingBias = FundingBias.NEUTRAL

    funding_rate: float = 0.0
    pressure_score: float = 0.0
    oi_confirmation: bool = False
    price_stall_confirmation: bool = False

    squeeze_probability: float | None = None
    mean_reversion_probability: float | None = None

    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.pressure_score = max(0.0, min(1.0, self.pressure_score))

        if self.squeeze_probability is not None:
            self.squeeze_probability = max(0.0, min(1.0, self.squeeze_probability))
        if self.mean_reversion_probability is not None:
            self.mean_reversion_probability = max(
                0.0, min(1.0, self.mean_reversion_probability)
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "direction", "level", "bias"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingSignal:
    """
    Нормалізований funding-сигнал для strategies layer.
    """

    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1

    signal_type: FundingSignalType = FundingSignalType.REGIME_CHANGE
    bias: FundingBias = FundingBias.NEUTRAL
    regime: FundingRegime = FundingRegime.UNKNOWN

    score: float = 0.0
    confidence: float = 0.0
    description: str = ""

    supporting_factors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    event_time: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def bullish(self) -> bool:
        return self.score > 0

    @property
    def bearish(self) -> bool:
        return self.score < 0

    @property
    def neutral(self) -> bool:
        return self.score == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("exchange", "timeframe", "signal_type", "bias", "regime"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data


@dataclass(slots=True)
class FundingAnalyticsEvent:
    """
    Уніфікована обгортка для подій, які можна напряму публікувати в EventBus.
    """

    event_type: FundingEventType
    symbol: str
    exchange: FundingDataSource = FundingDataSource.UNKNOWN
    timeframe: FundingTimeframe = FundingTimeframe.H1
    payload: dict[str, Any] = field(default_factory=dict)
    event_time: datetime = field(default_factory=utc_now)
    source: str = "analytics.funding"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper().strip()
        self.event_time = ensure_utc(self.event_time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("event_type", "exchange", "timeframe"):
            value = data.get(key)
            if value is not None:
                data[key] = value.value
        if isinstance(data.get("event_time"), datetime):
            data["event_time"] = data["event_time"].isoformat()
        return data