from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Mapping, Protocol, TypeVar, cast

from core.event_bus import EventBus
from core.logger import TradingLoggerAdapter, get_logger

from strategy.base import (
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
)
from strategy.config import StrategyConfig, StrategyDefinitionConfig
from strategy.enums import (
    FilterDecision,
    MarketRegime,
    SignalSide,
    StrategyCategory,
    Timeframe,
)
from strategy.exceptions import ValidationError
from strategy.models import (
    FilterResult,
    SignalContext,
    StrategyEvaluation,
    StrategySignal,
)


class PriceActionStrategyParamsProtocol(Protocol):
    strategy_name: str
    emit_signal_events: bool
    signal_event_name: str
    freshness_feature_names: tuple[str, ...]

    def validate(self) -> None:
        ...

    @classmethod
    def from_definition(
        cls,
        definition: StrategyDefinitionConfig | None,
    ) -> "PriceActionStrategyParamsProtocol":
        ...


ParamsT = TypeVar("ParamsT", bound=PriceActionStrategyParamsProtocol)


PRICE_ACTION_COMPOSITE_FEATURE_NAMES: tuple[str, ...] = (
    "analytics.price_action",
    "price_action",
    "price_action.composite",
    "analytics.price_action.composite",
)

PRICE_ACTION_MODULE_ALIASES: dict[str, tuple[str, ...]] = {
    "market_structure": (
        "market_structure",
        "structure",
        "price_action.market_structure",
        "analytics.price_action.market_structure",
    ),
    "support_resistance": (
        "support_resistance",
        "sr",
        "price_action.support_resistance",
        "analytics.price_action.support_resistance",
    ),
    "fair_value_gap": (
        "fair_value_gap",
        "fvg",
        "price_action.fair_value_gap",
        "price_action.fvg",
        "analytics.price_action.fair_value_gap",
        "analytics.price_action.fvg",
    ),
    "liquidity_levels": (
        "liquidity_levels",
        "liquidity",
        "price_action.liquidity_levels",
        "price_action.liquidity",
        "analytics.price_action.liquidity_levels",
        "analytics.price_action.liquidity",
    ),
    "trend": (
        "trend",
        "price_action.trend",
        "analytics.price_action.trend",
    ),
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False

    return bool(value)


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        if value > 1_000_000_000_000:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    return None


def enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip().lower()


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_scope_value(value: Any, *, uppercase: bool = False) -> str | None:
    if value is None:
        return None

    raw = str(getattr(value, "value", value)).strip()
    if not raw:
        return None

    return raw.upper() if uppercase else raw.lower()


def _timeframe_raw(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(getattr(value, "value", value)).strip()
    return raw or None


def apply_definition_metadata(
    *,
    params: Any,
    definition: StrategyDefinitionConfig | None,
    skip_fields: set[str] | None = None,
) -> Any:
    """
    Застосовує StrategyDefinitionConfig.metadata до params dataclass.

    Runtime-gating залишається в StrategyConfig / StrategyDefinitionConfig.runtime.
    Metadata використовується тільки для локальних параметрів конкретної strategy.
    """
    skip_fields = skip_fields or {"strategy_name"}

    if definition is None:
        params.validate()
        return params

    if hasattr(params, "strategy_name"):
        params.strategy_name = definition.name or params.strategy_name

    metadata = definition.metadata or {}
    dataclass_fields = getattr(params, "__dataclass_fields__", {})

    for field_name in dataclass_fields.keys():
        if field_name in skip_fields:
            continue

        if field_name in metadata:
            setattr(params, field_name, metadata[field_name])

    params.validate()
    return params


class PriceActionStrategyBase(
    ContextAwareComponent,
    EventEmitterMixin,
    NamedEntityMixin,
    PrioritizedMixin,
):
    """
    Базовий adapter layer для strategy/strategies/price_action.

    Цей клас не містить доменної торгової логіки. Його задача — вирівняти
    конкретні стратегії з актуальним analytics.price_action contract:

        analytics.price_action.updated
            -> PriceActionCompositeState
            -> {market_structure, support_resistance, fair_value_gap,
                liquidity_levels/liquidity, trend}
            -> конкретна strategy

    Важливо:
    - не прив'язується до старих окремих strategy helper-файлів;
    - не дублює StrategyEngine / SignalProcessor;
    - не читає exchange/data напряму;
    - дає backward-compatible aliases, щоб старі strategy-класи можна було
      переписувати поступово.
    """

    category: StrategyCategory = StrategyCategory.PRICE_ACTION
    default_priority: int = 100

    canonical_price_action_feature: str = "analytics.price_action"
    composite_feature_names: tuple[str, ...] = PRICE_ACTION_COMPOSITE_FEATURE_NAMES
    module_aliases: dict[str, tuple[str, ...]] = PRICE_ACTION_MODULE_ALIASES

    def __init__(
        self,
        *,
        config: StrategyConfig,
        strategy_name: str,
        params_cls: type[ParamsT],
        event_bus: EventBus | None = None,
        logger: Logger | TradingLoggerAdapter | None = None,
    ) -> None:
        resolved_logger = logger or get_logger(
            __name__,
            event_type="price_action_strategy",
            strategies=strategy_name,
        )

        super().__init__(
            config=config,
            event_bus=event_bus,
            logger=resolved_logger,
        )

        self._strategy_name = strategy_name
        self._logger = resolved_logger

        self.validate_config()

        self.definition = self._resolve_definition(strategy_name)
        self.params: ParamsT = cast(
            ParamsT,
            params_cls.from_definition(self.definition),  # type: ignore[attr-defined]
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._strategy_name

    @property
    def priority(self) -> int:
        if self.definition is not None:
            return self.definition.priority
        return self.default_priority

    # ------------------------------------------------------------------
    # Config / runtime helpers
    # ------------------------------------------------------------------

    def _resolve_definition(
        self,
        strategy_name: str,
    ) -> StrategyDefinitionConfig | None:
        get_strategy = getattr(self.config, "get_strategy", None)
        if callable(get_strategy):
            return get_strategy(strategy_name)
        return None

    @property
    def runtime(self) -> Any:
        if self.definition is not None:
            return self.definition.runtime
        return self.config.runtime

    def _is_strategy_enabled(self) -> bool:
        is_strategy_enabled = getattr(self.config, "is_strategy_enabled", None)
        if callable(is_strategy_enabled):
            return bool(is_strategy_enabled(self.name))

        if self.definition is not None:
            return bool(self.definition.runtime.enabled)

        return bool(self.config.runtime.enabled)

    def _symbol_allowed(self, symbol: str) -> bool:
        runtime = self.runtime
        return not runtime.symbols or symbol in runtime.symbols

    def _timeframe_allowed(self, timeframe: Timeframe) -> bool:
        runtime = self.runtime
        return not runtime.timeframes or timeframe in runtime.timeframes

    def _passes_runtime_thresholds(self, signal: StrategySignal) -> bool:
        runtime = self.runtime

        if signal.confidence < runtime.min_confidence:
            return False

        if signal.score < runtime.min_score:
            return False

        return True

    def _basic_runtime_gate(self, context: SignalContext) -> StrategyEvaluation | None:
        self.validate_context(context)

        if not self._is_strategy_enabled():
            return self._rejected_evaluation(
                context=context,
                reason="strategy_disabled",
            )

        if not self._symbol_allowed(context.symbol):
            return self._rejected_evaluation(
                context=context,
                reason="symbol_not_allowed",
            )

        if not self._timeframe_allowed(context.timeframe):
            return self._rejected_evaluation(
                context=context,
                reason="timeframe_not_allowed",
            )

        return None

    # ------------------------------------------------------------------
    # Price action analytics extraction
    # ------------------------------------------------------------------

    def _extract_price_action_state(self, context: SignalContext) -> dict[str, Any]:
        """
        Повертає normalized PriceActionCompositeState як dict.

        Підтримує новий contract:
        - context.price_action = composite state;
        - feature analytics.price_action = {state: composite state};
        - feature price_action / price_action.composite = legacy composite.
        """
        candidates: list[tuple[str, Any]] = []

        price_action_attr = getattr(context, "price_action", None)
        if price_action_attr is not None:
            candidates.append(("context.price_action", price_action_attr))

        for feature_name in self.composite_feature_names:
            feature = self._get_context_feature(context, feature_name)
            if feature is not None:
                candidates.append((feature_name, feature))

        for source_name, candidate in candidates:
            state = self._normalize_state_payload(candidate)
            if not state:
                continue

            if self._looks_like_price_action_composite(state):
                result = dict(state)
                result.setdefault("_source_feature", source_name)
                return result

        return {}

    def _extract_price_action_module(
        self,
        context: SignalContext,
        module_name: str,
        *,
        aliases: tuple[str, ...] = (),
        require_scope_match: bool = True,
    ) -> dict[str, Any]:
        """
        Повертає normalized state конкретного analytics.price_action модуля.

        Порядок пошуку:
        1. PriceActionCompositeState у context.price_action / analytics.price_action;
        2. module-specific feature analytics.price_action.<module>;
        3. legacy aliases, які залишені тільки для поетапного переписування.

        Якщо payload має scope і він не збігається з SignalContext, payload
        ігнорується. Якщо частини scope у context немає, вона не блокує payload.
        """
        canonical = self._canonical_module_name(module_name)
        candidates = self._module_candidate_names(canonical, aliases=aliases)

        composite = self._extract_price_action_state(context)
        module_payload = self._extract_module_from_composite(composite, canonical)
        if module_payload:
            normalized = self._normalize_state_payload(module_payload)
            if normalized and (
                not require_scope_match
                or self._analytics_scope_matches_context(context, normalized)
            ):
                normalized = dict(normalized)
                normalized.setdefault("_source_feature", composite.get("_source_feature", "analytics.price_action"))
                normalized.setdefault("_source_module", canonical)
                return normalized

        # Direct access in context.price_action for legacy shapes:
        # context.price_action["trend"], context.price_action["fvg"], etc.
        price_action_mapping = self._mapping_or_empty(getattr(context, "price_action", None))
        for candidate_name in candidates:
            payload = price_action_mapping.get(candidate_name)
            normalized = self._normalize_state_payload(payload)
            if normalized and (
                not require_scope_match
                or self._analytics_scope_matches_context(context, normalized)
            ):
                normalized = dict(normalized)
                normalized.setdefault("_source_feature", f"context.price_action.{candidate_name}")
                normalized.setdefault("_source_module", canonical)
                return normalized

        # Feature map direct module payloads.
        for candidate_name in candidates:
            payload = self._get_context_feature(context, candidate_name)
            normalized = self._normalize_state_payload(payload)
            if normalized and (
                not require_scope_match
                or self._analytics_scope_matches_context(context, normalized)
            ):
                normalized = dict(normalized)
                normalized.setdefault("_source_feature", candidate_name)
                normalized.setdefault("_source_module", canonical)
                return normalized

        return {}

    def _extract_module_from_composite(
        self,
        composite_payload: Mapping[str, Any],
        module_name: str,
    ) -> Mapping[str, Any]:
        if not composite_payload:
            return {}

        state = self._normalize_state_payload(composite_payload)
        if not state:
            return {}

        aliases = self._module_candidate_names(module_name)
        for alias in aliases:
            candidate = state.get(alias)
            candidate_mapping = self._mapping_or_empty(candidate)
            if candidate_mapping:
                return candidate_mapping

        # Some EventBus payloads may be shaped as:
        # {"updated_module": "trend", "state": <module_state>}.
        updated_module = str(state.get("updated_module") or "").strip().lower()
        if updated_module in aliases:
            module_state = self._mapping_or_empty(state.get("state"))
            if module_state:
                return module_state

        return {}

    def _normalize_state_payload(self, payload: Any) -> dict[str, Any]:
        """
        Нормалізує dataclass / dict / EventBus payload у plain dict state.

        Підтримувані форми:
        - dataclass state;
        - {state: dataclass_or_dict};
        - {payload: {state: ...}};
        - {data: {state: ...}};
        - direct state dict.
        """
        mapping = self._mapping_or_empty(payload)
        if not mapping:
            return {}

        for wrapper_key in ("payload", "data"):
            wrapped = mapping.get(wrapper_key)
            wrapped_mapping = self._mapping_or_empty(wrapped)
            if wrapped_mapping:
                mapping = wrapped_mapping
                break

        state = mapping.get("state")
        state_mapping = self._mapping_or_empty(state)
        if state_mapping:
            return dict(state_mapping)

        return dict(mapping)

    def _looks_like_price_action_composite(self, payload: Mapping[str, Any]) -> bool:
        if not payload:
            return False

        module_keys = {
            "market_structure",
            "support_resistance",
            "fair_value_gap",
            "fvg",
            "liquidity_levels",
            "liquidity",
            "trend",
        }
        if any(key in payload for key in module_keys):
            return True

        # A composite may be empty at startup but still scoped as analytics.price_action.
        metadata = self._mapping_or_empty(payload.get("metadata"))
        source = str(
            first_non_empty(
                payload.get("source"),
                payload.get("event_namespace"),
                metadata.get("source"),
                metadata.get("event_namespace"),
            )
            or ""
        ).strip()
        return source == self.canonical_price_action_feature

    def _module_candidate_names(
        self,
        module_name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        canonical = self._canonical_module_name(module_name)
        values: list[str] = [canonical]
        values.extend(self.module_aliases.get(canonical, ()))
        values.extend(aliases)

        deduped: list[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return tuple(deduped)

    @staticmethod
    def _canonical_module_name(module_name: str) -> str:
        normalized = str(module_name or "").strip().lower()
        mapping = {
            "fvg": "fair_value_gap",
            "liquidity": "liquidity_levels",
            "sr": "support_resistance",
            "structure": "market_structure",
        }
        return mapping.get(normalized, normalized)

    def _get_context_feature(self, context: SignalContext, feature_name: str) -> Any:
        get_feature = getattr(context, "get_feature", None)
        if callable(get_feature):
            try:
                return get_feature(feature_name)
            except KeyError:
                return None

        features = getattr(context, "features", None)
        features_mapping = self._mapping_or_empty(features)
        if features_mapping:
            return features_mapping.get(feature_name)

        feature_map = getattr(context, "feature_map", None)
        feature_map_mapping = self._mapping_or_empty(feature_map)
        if feature_map_mapping:
            return feature_map_mapping.get(feature_name)

        return None

    # ------------------------------------------------------------------
    # Analytics scope helpers
    # ------------------------------------------------------------------

    def _analytics_scope_matches_context(
        self,
        context: SignalContext,
        payload: Mapping[str, Any],
    ) -> bool:
        mismatch = self._analytics_scope_mismatch(context, payload)
        if mismatch:
            self._logger.debug(
                "Price action analytics payload skipped because scope does not match | strategy=%s mismatch=%s",
                self.name,
                mismatch,
            )
            return False
        return True

    def _analytics_scope_mismatch(
        self,
        context: SignalContext,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = self._extract_analytics_scope(payload)
        context_scope = self._extract_context_scope(context)

        mismatches: dict[str, Any] = {}
        for field_name in ("exchange", "market_type", "symbol", "timeframe"):
            payload_value = scope.get(field_name)
            context_value = context_scope.get(field_name)

            # symbol/timeframe should usually exist in context. exchange/market_type
            # may not exist yet in older SignalContext versions, so missing values
            # do not block the payload.
            if payload_value is None or context_value is None:
                continue

            if payload_value != context_value:
                mismatches[field_name] = {
                    "payload": payload_value,
                    "context": context_value,
                }

        return mismatches

    def _extract_context_scope(self, context: SignalContext) -> dict[str, Any]:
        metadata = self._mapping_or_empty(getattr(context, "metadata", None))
        market = self._mapping_or_empty(getattr(context, "market", None))

        exchange = first_non_empty(
            getattr(context, "exchange", None),
            metadata.get("exchange"),
            market.get("exchange"),
        )
        market_type = first_non_empty(
            getattr(context, "market_type", None),
            metadata.get("market_type"),
            market.get("market_type"),
        )
        symbol = first_non_empty(
            getattr(context, "symbol", None),
            metadata.get("symbol"),
            market.get("symbol"),
        )
        timeframe = first_non_empty(
            getattr(context, "timeframe", None),
            metadata.get("timeframe"),
            market.get("timeframe"),
        )

        return {
            "exchange": _normalize_scope_value(exchange),
            "market_type": _normalize_scope_value(market_type),
            "symbol": _normalize_scope_value(symbol, uppercase=True),
            "timeframe": _timeframe_raw(timeframe),
        }

    def _extract_analytics_scope(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        state = self._normalize_state_payload(payload)
        metadata = self._mapping_or_empty(state.get("metadata"))

        key = first_non_empty(state.get("key"), metadata.get("key"))
        key_values = list(key) if isinstance(key, (list, tuple)) else []

        exchange = first_non_empty(
            state.get("exchange"),
            metadata.get("exchange"),
            key_values[0] if len(key_values) > 0 else None,
        )
        market_type = first_non_empty(
            state.get("market_type"),
            metadata.get("market_type"),
            key_values[1] if len(key_values) > 1 else None,
        )
        symbol = first_non_empty(
            state.get("symbol"),
            metadata.get("symbol"),
            key_values[2] if len(key_values) > 2 else None,
        )
        timeframe = first_non_empty(
            state.get("timeframe"),
            metadata.get("timeframe"),
            key_values[3] if len(key_values) > 3 else None,
        )

        return {
            "exchange": _normalize_scope_value(exchange),
            "market_type": _normalize_scope_value(market_type),
            "symbol": _normalize_scope_value(symbol, uppercase=True),
            "timeframe": _timeframe_raw(timeframe),
            "exchange_symbol": first_non_empty(
                state.get("exchange_symbol"),
                metadata.get("exchange_symbol"),
            ),
            "key": key_values,
        }

    def _build_analytics_source_metadata(
        self,
        *,
        module_name: str,
        payload: Mapping[str, Any],
        selected_entity: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Єдина metadata-структура для StrategySignal.metadata.

        Конкретні стратегії мають додавати сюди selected gap/swing/level/signal id,
        layer, source topic і state scope, щоб downstream Risk/Execution/Dashboard
        бачили, на якому analytics state було побудовано сигнал.
        """
        state = self._normalize_state_payload(payload)
        source_metadata = self._mapping_or_empty(state.get("metadata"))
        entity = self._mapping_or_empty(selected_entity)

        result: dict[str, Any] = {
            "analytics_namespace": self.canonical_price_action_feature,
            "analytics_module": self._canonical_module_name(module_name),
            "analytics_source_feature": state.get("_source_feature"),
            "analytics_scope": self._extract_analytics_scope(state),
            "analytics_last_update": first_non_empty(
                state.get("last_update"),
                state.get("updated_at"),
                source_metadata.get("last_update"),
                source_metadata.get("updated_at"),
            ),
        }

        state_version = first_non_empty(
            state.get("state_version"),
            source_metadata.get("state_version"),
            source_metadata.get("version"),
        )
        if state_version is not None:
            result["analytics_state_version"] = state_version

        event_topic = first_non_empty(
            state.get("topic"),
            source_metadata.get("topic"),
            source_metadata.get("event_topic"),
        )
        if event_topic is not None:
            result["analytics_event_topic"] = event_topic

        if entity:
            for key in (
                "layer",
                "event_id",
                "gap_id",
                "level_id",
                "swing_id",
                "signal_id",
                "event_type",
                "status",
                "direction",
            ):
                if key in entity and entity[key] is not None:
                    result[f"selected_{key}"] = getattr(entity[key], "value", entity[key])

        if extra:
            result.update(dict(extra))

        return result

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _build_freshness_filter(
        self,
        context: SignalContext,
        *,
        filter_name: str | None = None,
        module_name: str | None = None,
        analytics_payload: Mapping[str, Any] | None = None,
    ) -> FilterResult | None:
        """
        Freshness filter із підтримкою нового analytics.price_action contract.

        Спочатку перевіряє explicit freshness_feature_names з params, щоб не
        ламати існуючі StrategyContextBuilder реалізації. Далі додає canonical
        analytics feature names для модуля / composite.
        """
        candidate_features: list[str] = list(self.params.freshness_feature_names)

        if module_name:
            canonical = self._canonical_module_name(module_name)
            candidate_features.append(self.canonical_price_action_feature)
            candidate_features.append(f"{self.canonical_price_action_feature}.{canonical}")
            candidate_features.extend(self.module_aliases.get(canonical, ()))

        first_found: str | None = None
        stale_feature: str | None = None

        seen: set[str] = set()
        for feature_name in candidate_features:
            if feature_name in seen:
                continue
            seen.add(feature_name)

            has_feature = getattr(context, "has_feature", None)
            if callable(has_feature):
                try:
                    exists = bool(has_feature(feature_name))
                except KeyError:
                    exists = False
            else:
                exists = self._get_context_feature(context, feature_name) is not None

            if not exists:
                continue

            if first_found is None:
                first_found = feature_name

            feature_is_stale = getattr(context, "feature_is_stale", None)
            if callable(feature_is_stale):
                try:
                    if bool(feature_is_stale(feature_name)):
                        stale_feature = feature_name
                        break
                except KeyError:
                    continue

        # Fallback для нового flow: якщо StrategyContextBuilder уже поклав
        # analytics payload, але ще не додав freshness metadata у feature registry.
        if first_found is None and analytics_payload:
            last_update = parse_datetime(
                first_non_empty(
                    analytics_payload.get("last_update"),
                    analytics_payload.get("updated_at"),
                    self._mapping_or_empty(analytics_payload.get("metadata")).get("last_update"),
                )
            )
            if last_update is not None:
                first_found = module_name or self.canonical_price_action_feature

        if first_found is None:
            return None

        is_stale = stale_feature is not None
        feature_name_for_result = stale_feature if is_stale else first_found

        return FilterResult(
            name=filter_name or f"{self.name}_freshness",
            decision=FilterDecision.BLOCK if is_stale else FilterDecision.PASS,
            reason="feature_stale" if is_stale else "feature_fresh",
            score_impact=-1.0 if is_stale else 0.0,
            metadata={
                "feature_name": feature_name_for_result,
                "strategy_name": self.name,
                "analytics_module": self._canonical_module_name(module_name) if module_name else None,
            },
        )

    def _build_regime_filter(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> FilterResult | None:
        runtime = self.runtime

        if not runtime.allowed_regimes:
            return None

        regime = self._resolve_market_regime(context)

        if MarketRegime.UNKNOWN in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
                metadata={
                    "strategy_name": self.name,
                    "side": side.value,
                },
            )

        if regime in runtime.allowed_regimes:
            return FilterResult(
                name="market_regime",
                decision=FilterDecision.PASS,
                reason=f"regime_{regime.value}",
                score_impact=0.0,
                metadata={
                    "strategy_name": self.name,
                    "side": side.value,
                },
            )

        return FilterResult(
            name="market_regime",
            decision=FilterDecision.BLOCK,
            reason=f"regime_{regime.value}_not_allowed_for_{side.value}",
            score_impact=-1.0,
            metadata={
                "strategy_name": self.name,
                "side": side.value,
            },
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluation(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal | None = None,
        passed: bool,
        confidence: float,
        score: float,
        reasons: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        evaluation_metadata = {
            "category": self.category.value,
            "timeframe": context.timeframe.value,
        }

        if metadata:
            evaluation_metadata.update(metadata)

        evaluation = StrategyEvaluation(
            strategy_name=self.name,
            symbol=context.symbol,
            timestamp=context.timestamp,
            signal=signal,
            passed=passed,
            score=score,
            confidence=confidence,
            reasons=list(reasons),
            metadata=evaluation_metadata,
        )

        evaluation.validate()
        return evaluation

    def _rejected_evaluation(
        self,
        *,
        context: SignalContext,
        reason: str,
        confidence: float = 0.0,
        score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        return self._evaluation(
            context=context,
            signal=None,
            passed=False,
            confidence=confidence,
            score=score,
            reasons=[reason],
            metadata=metadata,
        )

    def _finalize_signal_evaluation(
        self,
        *,
        context: SignalContext,
        signal: StrategySignal,
        confidence: float,
        score: float,
        reasons: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> StrategyEvaluation:
        passed = self._passes_runtime_thresholds(signal)

        if not passed:
            signal.to_rejected()

        return self._evaluation(
            context=context,
            signal=signal,
            passed=passed,
            confidence=confidence,
            score=score,
            reasons=reasons,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def maybe_emit_signal(self, signal: StrategySignal) -> None:
        if not self.params.emit_signal_events:
            return

        payload = self._build_signal_event_payload(signal)

        try:
            await self.emit_event(
                self.params.signal_event_name,
                payload,
                source=self.name,
            )
        except RuntimeError:
            self._logger.warning(
                "Signal event skipped because EventBus is not running | signal_id=%s symbol=%s",
                payload.get("signal_id"),
                signal.symbol,
            )
        except Exception:
            self._logger.exception(
                "Failed to emit strategy signal | signal_id=%s symbol=%s",
                payload.get("signal_id"),
                signal.symbol,
            )

    def _build_signal_event_payload(
        self,
        signal: StrategySignal,
    ) -> dict[str, Any]:
        metadata = dict(getattr(signal, "metadata", {}) or {})

        signal_id = (
            getattr(signal, "signal_id", None)
            or getattr(signal, "id", None)
            or metadata.get("signal_id")
        )

        payload = {
            "signal_id": signal_id,
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_name,
            "side": signal.side.value,
            "timeframe": signal.timeframe.value,
            "setup_type": signal.setup_type.value,
            "score": signal.score,
            "confidence": signal.confidence,
            "status": signal.status.value,
            "priority": signal.priority.value,
            "reasons": list(signal.reasons),
            "confirmations": list(signal.confirmations),
            "source_features": list(signal.source_features),
            "metadata": metadata,
        }

        created_at = getattr(signal, "created_at", None) or getattr(signal, "timestamp", None)
        if created_at is not None:
            payload["created_at"] = (
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else created_at
            )

        origin = getattr(signal, "origin", None)
        if origin is not None:
            payload["origin"] = getattr(origin, "value", origin)

        category = getattr(signal, "category", None)
        if category is not None:
            payload["category"] = getattr(category, "value", category)

        trigger_type = getattr(signal, "trigger_type", None)
        if trigger_type is not None:
            payload["trigger_type"] = getattr(trigger_type, "value", trigger_type)

        return payload

    # ------------------------------------------------------------------
    # Market regime helpers
    # ------------------------------------------------------------------

    def _resolve_market_regime(self, context: SignalContext) -> MarketRegime:
        regime: Any = (
            context.regime.regime
            if context.regime is not None
            else MarketRegime.UNKNOWN
        )

        if isinstance(regime, MarketRegime):
            return regime

        raw = enum_value(regime)

        mapping = {
            "trending_up": MarketRegime.TRENDING_UP,
            "trending_down": MarketRegime.TRENDING_DOWN,
            "ranging": MarketRegime.RANGING,
            "breakout": MarketRegime.BREAKOUT,
            "squeeze": MarketRegime.SQUEEZE,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
            "low_volatility": MarketRegime.LOW_VOLATILITY,
            "news_driven": MarketRegime.NEWS_DRIVEN,
            "illiquid": MarketRegime.ILLIQUID,
            "risk_off": MarketRegime.RISK_OFF,
        }

        return mapping.get(raw, MarketRegime.UNKNOWN)

    def _regime_alignment_score(
        self,
        *,
        context: SignalContext,
        side: SignalSide,
    ) -> float:
        regime = self._resolve_market_regime(context)

        if regime == MarketRegime.UNKNOWN:
            return 0.5

        if regime == MarketRegime.RANGING:
            return 0.35

        if regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_DRIVEN}:
            return 0.40

        if regime in {MarketRegime.BREAKOUT, MarketRegime.SQUEEZE}:
            return 0.65

        if side == SignalSide.LONG and regime == MarketRegime.TRENDING_UP:
            return 1.0

        if side == SignalSide.SHORT and regime == MarketRegime.TRENDING_DOWN:
            return 1.0

        return 0.20

    def _resolve_regime(self, context: SignalContext) -> MarketRegime:
        return self._resolve_market_regime(context)

    # ------------------------------------------------------------------
    # Generic parsing helpers
    # ------------------------------------------------------------------

    def _parse_timeframe(self, value: Any) -> Timeframe | None:
        if isinstance(value, Timeframe):
            return value

        raw = enum_value(value)
        if not raw:
            return None

        for timeframe in Timeframe:
            if timeframe.value == raw:
                return timeframe

        return None

    def _mapping_or_empty(self, payload: Any) -> Mapping[str, Any]:
        if payload is None:
            return {}

        if is_dataclass(payload) and not isinstance(payload, type):
            return asdict(payload)

        if hasattr(payload, "__dict__") and not isinstance(payload, Mapping):
            payload = vars(payload)

        if isinstance(payload, Mapping):
            return payload

        return {}

    def _state_mapping_or_empty(self, payload: Any) -> Mapping[str, Any]:
        return self._normalize_state_payload(payload)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_bounded_fields(
        *,
        instance: Any,
        field_names: tuple[str, ...],
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> None:
        for field_name in field_names:
            value = getattr(instance, field_name)

            if not minimum <= value <= maximum:
                raise ValidationError(
                    f"{field_name} must be between {minimum} and {maximum}"
                )

    @staticmethod
    def validate_non_negative_fields(
        *,
        instance: Any,
        field_names: tuple[str, ...],
    ) -> None:
        for field_name in field_names:
            value = getattr(instance, field_name)

            if value < 0:
                raise ValidationError(f"{field_name} must be >= 0")

    # ------------------------------------------------------------------
    # Optional hook for subclasses
    # ------------------------------------------------------------------

    async def maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value
