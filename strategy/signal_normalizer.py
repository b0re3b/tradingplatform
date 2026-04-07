from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import StrategyComponent
from .config import FeatureFreshnessConfig
from .context import StrategyContext
from .enums import FeatureSource
from .exceptions import SignalNormalizationError
from .models import FeatureSnapshot


def utcnow() -> datetime:
    return datetime.utcnow()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass(slots=True)
class NormalizedPayload:
    """
    Результат нормалізації одного analytics payload.
    """

    source: FeatureSource
    symbol: str
    timestamp: datetime
    domain_data: dict[str, Any] = field(default_factory=dict)
    features: list[FeatureSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalNormalizer(StrategyComponent):
    """
    Нормалізує сирі analytics payload у:
    - domain_data для StrategyContext
    - FeatureSnapshot для FeatureStore / context.feature_map

    Підтримує:
    - normalize_event(...)
    - apply_to_context(...)
    - normalize_value helpers
    """

    def __init__(self, config, event_bus=None, logger=None) -> None:
        super().__init__(config=config, event_bus=event_bus, logger=logger)
        self.validate_config()

    def normalize_event(
        self,
        *,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> NormalizedPayload:
        if not isinstance(payload, dict):
            raise SignalNormalizationError("payload must be a dict")

        source = self._resolve_source(event_name, payload)
        symbol = self._extract_symbol(payload)
        ts = self._extract_timestamp(payload, timestamp)

        domain_data = self._extract_domain_data(payload)
        features = self._extract_features(
            source=source,
            symbol=symbol,
            payload=payload,
            timestamp=ts,
        )

        normalized = NormalizedPayload(
            source=source,
            symbol=symbol,
            timestamp=ts,
            domain_data=domain_data,
            features=features,
            metadata={
                "event_name": event_name,
            },
        )

        self.log_debug(
            "Normalized analytics event",
            event_name=event_name,
            source=str(source),
            symbol=symbol,
            features_count=len(features),
        )
        return normalized

    def apply_to_context(
        self,
        context: StrategyContext,
        normalized: NormalizedPayload,
    ) -> StrategyContext:
        context.validate()

        if context.symbol != normalized.symbol:
            raise SignalNormalizationError(
                f"context symbol '{context.symbol}' != normalized symbol '{normalized.symbol}'"
            )

        context.timestamp = normalized.timestamp
        context.update_domain_data(normalized.source, normalized.domain_data)

        for snapshot in normalized.features:
            context.put_feature(snapshot)
            if snapshot.freshness_seconds is not None:
                context.freshness_map[snapshot.name] = snapshot.freshness_seconds

        return context

    def normalize_and_apply(
        self,
        *,
        context: StrategyContext,
        event_name: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> StrategyContext:
        normalized = self.normalize_event(
            event_name=event_name,
            payload=payload,
            timestamp=timestamp,
        )
        return self.apply_to_context(context, normalized)

    def _resolve_source(self, event_name: str, payload: dict[str, Any]) -> FeatureSource:
        explicit_source = payload.get("source")
        if isinstance(explicit_source, FeatureSource):
            return explicit_source

        if isinstance(explicit_source, str):
            try:
                return FeatureSource(explicit_source)
            except ValueError:
                pass

        event_lower = event_name.lower()

        if "orderflow" in event_lower or "cvd" in event_lower or "imbalance" in event_lower:
            return FeatureSource.ORDERFLOW
        if "liquidity" in event_lower or "stop_cluster" in event_lower or "equal_high" in event_lower:
            return FeatureSource.LIQUIDITY
        if "price_action" in event_lower or "market_structure" in event_lower or "fvg" in event_lower:
            return FeatureSource.PRICE_ACTION
        if "liquidation" in event_lower or "squeeze" in event_lower:
            return FeatureSource.LIQUIDATIONS
        if "whale" in event_lower:
            return FeatureSource.WHALES
        if "spoof" in event_lower or "fake_liquidity" in event_lower:
            return FeatureSource.SPOOFING
        if "spread" in event_lower or "basis" in event_lower or "arb" in event_lower:
            return FeatureSource.SPREADS
        if "funding" in event_lower:
            return FeatureSource.FUNDING
        if "open_interest" in event_lower or "oi_" in event_lower or ".oi" in event_lower:
            return FeatureSource.OPEN_INTEREST

        raise SignalNormalizationError(
            f"unable to resolve FeatureSource for event '{event_name}'"
        )

    def _extract_symbol(self, payload: dict[str, Any]) -> str:
        symbol = payload.get("symbol") or payload.get("instrument") or payload.get("market")
        if not isinstance(symbol, str) or not symbol.strip():
            raise SignalNormalizationError("payload does not contain valid symbol")
        return symbol.strip()

    def _extract_timestamp(
        self,
        payload: dict[str, Any],
        fallback: datetime | None = None,
    ) -> datetime:
        raw = payload.get("timestamp") or payload.get("ts") or fallback
        if raw is None:
            return utcnow()

        if isinstance(raw, datetime):
            return raw

        if isinstance(raw, (int, float)):
            # heuristic: milliseconds support
            if raw > 10_000_000_000:
                return datetime.utcfromtimestamp(raw / 1000.0)
            return datetime.utcfromtimestamp(raw)

        raise SignalNormalizationError("unsupported timestamp type in payload")

    def _extract_domain_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Дані домену, які хочемо покласти в context.orderflow/context.whales/...data
        """
        excluded_keys = {
            "symbol",
            "instrument",
            "market",
            "timestamp",
            "ts",
            "source",
            "features",
            "feature_map",
        }
        return {key: value for key, value in payload.items() if key not in excluded_keys}

    def _extract_features(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        explicit_features = payload.get("features")

        if explicit_features is None:
            return self._build_implicit_features(
                source=source,
                symbol=symbol,
                payload=payload,
                timestamp=timestamp,
            )

        if not isinstance(explicit_features, list):
            raise SignalNormalizationError("payload['features'] must be a list")

        result: list[FeatureSnapshot] = []
        for item in explicit_features:
            if not isinstance(item, dict):
                raise SignalNormalizationError("each item in payload['features'] must be a dict")

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise SignalNormalizationError("feature item must contain non-empty 'name'")

            value = item.get("value")
            confidence = self._safe_confidence(item.get("confidence", payload.get("confidence", 0.0)))
            normalized_value = self._safe_normalized_value(item.get("normalized_value", None))
            freshness_seconds = self._resolve_freshness_seconds(
                feature_name=name,
                explicit=item.get("freshness_seconds"),
            )

            snapshot = FeatureSnapshot(
                name=name,
                value=value,
                source=source,
                symbol=symbol,
                timestamp=timestamp,
                confidence=confidence,
                normalized_value=normalized_value,
                freshness_seconds=freshness_seconds,
                metadata=item.get("metadata", {}),
            )
            snapshot.validate()
            result.append(snapshot)

        return result

    def _build_implicit_features(
        self,
        *,
        source: FeatureSource,
        symbol: str,
        payload: dict[str, Any],
        timestamp: datetime,
    ) -> list[FeatureSnapshot]:
        """
        Якщо payload не містить явного списку features, пробуємо автоматично
        перетворити scalar-поля на FeatureSnapshot.
        """
        result: list[FeatureSnapshot] = []
        excluded_keys = {
            "symbol",
            "instrument",
            "market",
            "timestamp",
            "ts",
            "source",
            "features",
            "feature_map",
            "metadata",
        }

        base_confidence = self._safe_confidence(payload.get("confidence", 0.0))

        for key, value in payload.items():
            if key in excluded_keys:
                continue

            if isinstance(value, (int, float, bool, str)):
                normalized_value = self._infer_normalized_value(value)
                snapshot = FeatureSnapshot(
                    name=key,
                    value=value,
                    source=source,
                    symbol=symbol,
                    timestamp=timestamp,
                    confidence=base_confidence,
                    normalized_value=normalized_value,
                    freshness_seconds=self._resolve_freshness_seconds(key, None),
                    metadata={},
                )
                snapshot.validate()
                result.append(snapshot)

        return result

    def _resolve_freshness_seconds(
        self,
        feature_name: str,
        explicit: Any,
    ) -> float | None:
        if explicit is not None:
            if not isinstance(explicit, (int, float)) or explicit <= 0:
                raise SignalNormalizationError(
                    f"freshness_seconds must be positive for feature '{feature_name}'"
                )
            return float(explicit)

        freshness_cfg: FeatureFreshnessConfig = self.config.freshness
        return float(freshness_cfg.get_ttl(feature_name))

    def _safe_confidence(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return clamp(float(value), 0.0, 1.0)
        raise SignalNormalizationError(f"unsupported confidence value: {value!r}")

    def _safe_normalized_value(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return clamp(float(value), -1.0, 1.0)
        raise SignalNormalizationError(f"unsupported normalized_value: {value!r}")

    def _infer_normalized_value(self, value: Any) -> float | None:
        """
        Евристика:
        - bool -> 0/1
        - float/int in [-1,1] -> 그대로
        - інше -> None
        """
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):
            numeric = float(value)
            if -1.0 <= numeric <= 1.0:
                return numeric
            return None

        return None