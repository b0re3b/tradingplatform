from __future__ import annotations

from enum import Enum


class SpoofingSide(str, Enum):
    """
    Сторона, на якій виявлено маніпулятивну активність.
    """
    BID = "bid"
    ASK = "ask"
    UNKNOWN = "unknown"


class SpoofingType(str, Enum):
    """
    Конкретний тип spoofing-патерну.
    """
    FAKE_WALL = "fake_wall"
    ORDER_PULL = "order_pull"
    FAKE_LIQUIDITY = "fake_liquidity"
    LAYERING = "layering"
    FLIP_PRESSURE = "flip_pressure"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class SpoofingPattern(str, Enum):
    """
    Більш прикладний опис патерну для сигналів і алертів.
    """
    SINGLE_LEVEL_SPOOF = "single_level_spoof"
    MULTI_LEVEL_LAYERING = "multi_level_layering"
    PULL_AND_REVERSAL = "pull_and_reversal"
    FAKE_ABSORPTION = "fake_absorption"
    PRESSURE_BLUFF = "pressure_bluff"
    UNKNOWN = "unknown"


class SpoofingSeverity(str, Enum):
    """
    Рівень серйозності події.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SpoofingStatus(str, Enum):
    """
    Стан кандидата/сигналу spoofing.
    """
    DETECTED = "detected"
    TRACKING = "tracking"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SpoofingComponent(str, Enum):
    """
    Компонент spoofing-пакета, який згенерував подію/оцінку.
    """
    PERSISTENCE_TRACKER = "persistence_tracker"
    ORDERBOOK_WALL_DETECTOR = "orderbook_wall_detector"
    ORDER_PULL_DETECTOR = "order_pull_detector"
    FAKE_LIQUIDITY_DETECTOR = "fake_liquidity_detector"
    LAYERING_DETECTOR = "layering_detector"
    FLIP_PRESSURE_DETECTOR = "flip_pressure_detector"
    SPOOFING_SCORE = "spoofing_score"
    ANALYZER = "analyzer"


class LiquidityEventType(str, Enum):
    """
    Тип події в життєвому циклі стінки/ліквідності.
    """
    CREATED = "created"
    UPDATED = "updated"
    PARTIALLY_FILLED = "partially_filled"
    PULLED = "pulled"
    FULLY_FILLED = "fully_filled"
    EXPIRED = "expired"


class DetectorDecision(str, Enum):
    """
    Стандартизоване рішення окремого детектора.
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    INSUFFICIENT_DATA = "insufficient_data"


class OrderbookWallState(str, Enum):
    """
    Поточний стан стінки в orderbook.
    """
    ACTIVE = "active"
    WEAKENING = "weakening"
    PULLED = "pulled"
    FILLED = "filled"
    EXPIRED = "expired"


class ScoreComponent(str, Enum):
    """
    Компоненти фінального spoofing score.
    """
    WALL_SIZE = "wall_size"
    WALL_DISTANCE = "wall_distance"
    PERSISTENCE = "persistence"
    PULL_SPEED = "pull_speed"
    FILL_RATIO = "fill_ratio"
    PRICE_REACTION = "price_reaction"
    REPETITION = "repetition"
    LAYERING = "layering"