from __future__ import annotations

from enum import Enum


class SpoofingSide(str, Enum):
    """
    Сторона orderbook, на якій виявлено потенційну spoofing-активність.

    BID:
        Маніпулятивна або підозріла ліквідність на bid-side.

    ASK:
        Маніпулятивна або підозріла ліквідність на ask-side.

    UNKNOWN:
        Використовується для неповних/нормалізованих подій, де сторону
        неможливо надійно визначити.
    """

    BID = "bid"
    ASK = "ask"
    UNKNOWN = "unknown"


class TradeSide(str, Enum):
    """
    Сторона агресивної угоди / trade-flow події.

    Цей enum замінює raw string values на кшталт "buy" / "sell",
    які раніше могли передаватися напряму в legacy detector-логіці.
    """

    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class SpoofingType(str, Enum):
    """
    Конкретний тип spoofing-поведінки.

    Значення BID_SPOOF / ASK_SPOOF залишені як legacy-compatible типи,
    але нова архітектура має надавати перевагу більш точним типам:
    FAKE_WALL, ORDER_PULL, FAKE_LIQUIDITY, LAYERING, FLIP_PRESSURE.
    """

    FAKE_WALL = "fake_wall"
    ORDER_PULL = "order_pull"
    FAKE_LIQUIDITY = "fake_liquidity"
    LAYERING = "layering"
    FLIP_PRESSURE = "flip_pressure"
    COMPOSITE = "composite"

    # Legacy compatibility from old spoofing_detector.py
    BID_SPOOF = "bid_spoof"
    ASK_SPOOF = "ask_spoof"

    UNKNOWN = "unknown"


class SpoofingPattern(str, Enum):
    """
    Прикладний патерн для detector result, сигналів і алертів.
    """

    SINGLE_LEVEL_SPOOF = "single_level_spoof"
    MULTI_LEVEL_LAYERING = "multi_level_layering"
    PULL_AND_REVERSAL = "pull_and_reversal"
    FAKE_ABSORPTION = "fake_absorption"
    PRESSURE_BLUFF = "pressure_bluff"
    UNKNOWN = "unknown"


class SpoofingSeverity(str, Enum):
    """
    Рівень серйозності фінального spoofing-сигналу.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SpoofingStatus(str, Enum):
    """
    Життєвий стан spoofing-сигналу або високорівневого кандидата.
    """

    DETECTED = "detected"
    TRACKING = "tracking"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class CandidateStatus(str, Enum):
    """
    Життєвий стан внутрішнього кандидата.

    Винесено з legacy spoofing_detector.py, щоб не тримати локальні enum-и
    всередині detector-класів.

    У новій архітектурі цей enum варто використовувати лише для внутрішніх
    state-моделей, якщо вони потрібні. Для фінальних сигналів використовуй
    SpoofingStatus.
    """

    ACTIVE = "active"
    CANCELLED = "cancelled"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


class SpoofingComponent(str, Enum):
    """
    Компонент spoofing-пакета, який згенерував подію, detector result
    або score contribution.
    """

    PERSISTENCE_TRACKER = "persistence_tracker"
    ORDERBOOK_WALL_DETECTOR = "orderbook_wall_detector"
    ORDER_PULL_DETECTOR = "order_pull_detector"
    FAKE_LIQUIDITY_DETECTOR = "fake_liquidity_detector"
    LAYERING_DETECTOR = "layering_detector"
    FLIP_PRESSURE_DETECTOR = "flip_pressure_detector"
    SPOOFING_SCORE = "spoofing_score"
    ANALYZER = "analyzer"

    # Optional compatibility name if old facade is temporarily kept.
    SPOOFING_DETECTOR = "spoofing_detector"


class LiquidityEventType(str, Enum):
    """
    Тип події в життєвому циклі великого orderbook-рівня / liquidity wall.
    """

    CREATED = "created"
    UPDATED = "updated"
    PARTIALLY_FILLED = "partially_filled"
    PULLED = "pulled"
    FULLY_FILLED = "fully_filled"
    EXPIRED = "expired"


class DetectorDecision(str, Enum):
    """
    Стандартизоване рішення окремого detector-а.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    INSUFFICIENT_DATA = "insufficient_data"


class OrderbookWallState(str, Enum):
    """
    Поточний стан tracked wall у PersistenceTracker.
    """

    ACTIVE = "active"
    WEAKENING = "weakening"
    PULLED = "pulled"
    FILLED = "filled"
    EXPIRED = "expired"


class ScoreComponent(str, Enum):
    """
    Компоненти фінального spoofing score.

    Використовуються в ScoreContribution, щоб score breakdown був стабільним,
    типізованим і придатним для dashboard/API.
    """

    WALL_SIZE = "wall_size"
    WALL_DISTANCE = "wall_distance"
    PERSISTENCE = "persistence"
    PULL_SPEED = "pull_speed"
    FILL_RATIO = "fill_ratio"
    PRICE_REACTION = "price_reaction"
    REPETITION = "repetition"
    LAYERING = "layering"


__all__ = [
    "SpoofingSide",
    "TradeSide",
    "SpoofingType",
    "SpoofingPattern",
    "SpoofingSeverity",
    "SpoofingStatus",
    "CandidateStatus",
    "SpoofingComponent",
    "LiquidityEventType",
    "DetectorDecision",
    "OrderbookWallState",
    "ScoreComponent",
]