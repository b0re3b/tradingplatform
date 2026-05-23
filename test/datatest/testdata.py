"""
Жорсткий інтеграційний тест зв'язки:
analytics.* -> StrategyEngine/SignalProcessor -> signal.generated
-> RiskManager boundary -> signal.confirmed -> TradeExecutor -> OrderManager -> filled order.

Файл навмисно тестує glue/event-boundary, а не точність конкретних доменних стратегій.
Він створює synthetic analytics storm: 10 trading events від кожної РЕАЛЬНОЇ
аналітики, реєструє probe-strategy на кожну strategy category і проганяє
повний цикл до execution.order_filled / execution.execution_completed.

Важливо: hybrid НЕ тестується як analytics.hybrid.* подія, бо в проєкті немає
окремого FeatureSource.HYBRID / analytics.hybrid package. Hybrid-стратегії
мають маршрутизуватись як cross-domain strategies на звичайних domain events
через StrategyConfig.routing.route_hybrid_on_domain_signal=True.

Куди класти:
    tests/integration/test_strategy_risk_execution_torture.py

Запуск:
    pytest -q tests/integration/test_strategy_risk_execution_torture.py -s

Очікування:
    - у PYTHONPATH має бути корінь проєкту;
    - пакети core, strategy, risk, execution мають імпортуватись з реального проєкту;
    - реальна біржа НЕ викликається: OrderManager замінений capturing/fill fake-ом.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

import pytest

from strategy.base import BaseStrategy
from strategy.config import (
    BuilderConfig,
    ConflictConfig,
    ConfluenceConfig,
    FilterConfig,
    PortfolioCoordinatorConfig,
    RoutingConfig,
    StrategyConfig,
    StrategyDefinitionConfig,
    StrategyRuntimeConfig,
    VotingConfig,
)
from strategy.engine import StrategyEngine
from strategy.enums import (
    EntryType,
    ExitType,
    FeatureSource,
    MarketRegime,
    SetupType,
    SignalPriority,
    SignalSide,
    StrategyCategory,
    Timeframe,
    TriggerType,
)
from strategy.models import (
    EntryPlan,
    ExecutionPlanDraft,
    ExitPlan,
    InvalidationPlan,
    StrategyContext,
    StrategySignal,
    TargetPlan,
    utcnow,
)

from execution.config import SmartExecutionConfig, TradeExecutorConfig
from execution.enums import ExecutionMode, OrderStatus
from execution.models import OrderRequest, OrderResult
from execution.smart_execution import SmartExecution
from execution.trade_executor import TradeExecutor


try:
    # У реальному проєкті ці enum-и вже існують і саме їх очікує execution.
    from risk.enums import MarginMode, OrderIntent, RiskMode, TradeTier
except Exception as exc:  # pragma: no cover - має спрацювати тільки якщо risk package не підключено.
    pytest.skip(f"risk package is required for this integration test: {exc}", allow_module_level=True)


# -----------------------------------------------------------------------------
# Minimal EventBus harness
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CapturedEvent:
    topic: str
    payload: dict[str, Any]
    priority: Any = None
    source: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class Subscription:
    topic: str
    handler: Callable[[CapturedEvent], Any]
    name: str | None = None


class InMemoryEventBus:
    """
    Small async EventBus compatible з core.event_bus usage in strategy/execution:
    - subscribe(topic, handler, name=...)
    - unsubscribe(subscription)
    - await emit(topic, payload, priority=..., source=...)

    Підтримує exact topics і wildcard suffix 'analytics.*'.
    """

    def __init__(self) -> None:
        self.subscriptions: list[Subscription] = []
        self.events: list[CapturedEvent] = []
        self.failures: list[tuple[CapturedEvent, BaseException]] = []

    def subscribe(
        self,
        topic: str,
        handler: Callable[[CapturedEvent], Any],
        *,
        name: str | None = None,
        **_: Any,
    ) -> Subscription:
        sub = Subscription(topic=topic, handler=handler, name=name)
        self.subscriptions.append(sub)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        self.subscriptions = [sub for sub in self.subscriptions if sub is not subscription]

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        priority: Any = None,
        source: str | None = None,
        **_: Any,
    ) -> None:
        event = CapturedEvent(
            topic=topic,
            payload=dict(payload),
            priority=priority,
            source=source,
        )
        self.events.append(event)

        handlers = [sub.handler for sub in self.subscriptions if self._matches(sub.topic, topic)]
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:  # noqa: BLE001 - test must capture boundary crashes.
                self.failures.append((event, exc))
                raise

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        if pattern.endswith("*"):
            return topic.startswith(pattern[:-1])
        return False

    def by_topic(self, topic: str) -> list[CapturedEvent]:
        return [event for event in self.events if event.topic == topic]

    def count(self, topic: str) -> int:
        return len(self.by_topic(topic))


# -----------------------------------------------------------------------------
# Probe strategies: максимально суворий тест pipeline, не доменної математики.
# -----------------------------------------------------------------------------


# Реальні analytics domains. Hybrid тут навмисно НЕ вказаний: у проєкті немає
# окремої analytics.hybrid аналітики й немає FeatureSource.HYBRID. Hybrid — це
# synthetic/cross-domain strategy contract, який має будуватися зі звичайних
# domain features і маршрутизуватись через route_hybrid_on_domain_signal=True.
ANALYTICS_DOMAIN_TO_CATEGORY: dict[str, StrategyCategory] = {
    "orderflow": StrategyCategory.ORDERFLOW,
    "liquidity": StrategyCategory.LIQUIDITY,
    "price_action": StrategyCategory.PRICE_ACTION,
    "liquidations": StrategyCategory.LIQUIDATIONS,
    "whales": StrategyCategory.WHALES,
    "spoofing": StrategyCategory.SPOOFING,
    "spreads": StrategyCategory.SPREADS,
    "funding": StrategyCategory.FUNDING,
    "open_interest": StrategyCategory.OPEN_INTEREST,
}

ANALYTICS_EVENT_TOPIC: dict[str, str] = {
    "orderflow": "analytics.orderflow.composite.updated",
    "liquidity": "analytics.liquidity.map.updated",
    "price_action": "analytics.price_action.market_structure.updated",
    "liquidations": "analytics.liquidations.cascade.updated",
    "whales": "analytics.whales.activity.updated",
    "spoofing": "analytics.spoofing.signal.updated",
    "spreads": "analytics.spreads.signal.updated",
    "funding": "analytics.funding.signal.updated",
    "open_interest": "analytics.open_interest.signal.updated",
}

CATEGORY_SETUP: dict[StrategyCategory, SetupType] = {
    StrategyCategory.ORDERFLOW: SetupType.CONTINUATION,
    StrategyCategory.LIQUIDITY: SetupType.LIQUIDITY_SWEEP,
    StrategyCategory.PRICE_ACTION: SetupType.BREAKOUT,
    StrategyCategory.LIQUIDATIONS: SetupType.SQUEEZE,
    StrategyCategory.WHALES: SetupType.ABSORPTION,
    StrategyCategory.SPOOFING: SetupType.REVERSAL,
    StrategyCategory.SPREADS: SetupType.MEAN_REVERSION,
    StrategyCategory.FUNDING: SetupType.MOMENTUM,
    StrategyCategory.OPEN_INTEREST: SetupType.BREAKOUT,
    StrategyCategory.HYBRID: SetupType.MOMENTUM,
}


class ProbeStrategy(BaseStrategy):
    """
    Synthetic strategy, яка проходить реальний BaseStrategy.evaluate(),
    реальний SignalProcessor scoring/filter/confluence/builder/portfolio/risk-payload
    і повертає повністю risk-ready StrategySignal.
    """

    default_timeframe = Timeframe.M1
    default_trigger_type = TriggerType.PRIMARY

    def __init__(
        self,
        *,
        config: StrategyConfig,
        category: StrategyCategory,
        strategy_name: str,
        side_offset: int,
    ) -> None:
        self.category = category
        self.default_setup_type = CATEGORY_SETUP[category]
        self.side_offset = side_offset
        definition = StrategyDefinitionConfig(
            name=strategy_name,
            category=category,
            runtime=StrategyRuntimeConfig(
                enabled=True,
                symbols=[],
                timeframes=[Timeframe.M1],
                allowed_regimes=[MarketRegime.UNKNOWN],
                cooldown_seconds=0,
                max_signal_age_seconds=300,
                min_confidence=0.10,
                min_score=0.10,
            ),
            required_features=set(),
            weight=1.0,
            priority=10 + side_offset,
            tags=["probe", category.value, "torture"],
        )
        super().__init__(config=config, definition=definition, service_name=f"strategy.probe.{strategy_name}")

    async def generate_signal(self, context: StrategyContext) -> StrategySignal | None:
        domain = self.category.value
        event_index = int(context.metadata.get("event_index", 0) or 0)
        base_price = float(context.metadata.get("last_price") or 50_000.0)
        # Для full-cycle тесту всі probe-стратегії в одному batch мають давати
        # один dominant side. Інакше ConfluenceEngine чесно карає side-conflict
        # penalty і batch може бути відхилений до risk/execution boundary.
        # Окрему перевірку side-conflict краще тримати в окремому unit/integration
        # тесті confluence, а тут ми тестуємо повний життєвий цикл.
        side = SignalSide.LONG if event_index % 2 == 0 else SignalSide.SHORT

        if side is SignalSide.LONG:
            entry = base_price
            stop = base_price * 0.992
            take = base_price * 1.018
        else:
            entry = base_price
            stop = base_price * 1.008
            take = base_price * 0.982

        confidence = min(0.99, 0.78 + (event_index % 5) * 0.035)
        score = min(0.99, 0.76 + (self.side_offset * 0.025) + (event_index % 4) * 0.03)

        signal = self.build_signal(
            context=context,
            side=side,
            confidence=confidence,
            score=score,
            setup_type=self.default_setup_type,
            trigger_type=TriggerType.PRIMARY,
            priority=SignalPriority.HIGH,
            reasons=[
                f"{domain}:probe_signal_{event_index}",
                "torture_test_full_pipeline",
            ],
            confirmations=[
                f"{domain}:domain_contract_present",
                "synthetic_risk_ready_plans_present",
            ],
            source_features=[
                f"{domain}.signal_score",
                f"{domain}.confidence",
                "last_price",
            ],
            metadata={
                "entry_price": entry,
                "stop_loss": stop,
                "take_profit": take,
                "last_price": base_price,
                "requested_size": 0.003 + (event_index % 3) * 0.001,
                "requested_leverage": 3.0,
                "requested_margin": 25.0,
                "exchange": "binance",
                "market_type": "usdm_futures",
                "margin_mode": "isolated",
                "order_intent": "open",
                "tier": "normal",
                "liquidity_class": "normal",
                "execution_quality": "good",
                "priority_score": score,
                "expected_reward": abs(take - entry),
                "expected_loss": abs(entry - stop),
                "expected_win_probability": confidence,
                "torture_domain": domain,
                "torture_event_index": event_index,
            },
        )

        signal.entry_plan = EntryPlan(
            entry_type=EntryType.MARKET,
            price=entry,
            max_slippage_bps=5.0,
            confirmation_required=False,
            notes=["prebuilt by torture probe strategy"],
        )
        signal.invalidation_plan = InvalidationPlan(
            price=stop,
            reason="probe_stop_invalidated",
            timeout_seconds=900,
            conditions=["price_crosses_stop"],
        )
        signal.exit_plan = ExitPlan(
            exit_types=[ExitType.TAKE_PROFIT, ExitType.STOP_LOSS],
            stop_loss=stop,
            take_profit_levels=[TargetPlan(price=take, size_fraction=1.0, rr=2.0, label="tp1")],
            max_holding_seconds=1_800,
        )
        signal.execution_plan = ExecutionPlanDraft(
            symbol=context.symbol,
            side=side,
            entry=signal.entry_plan,
            exit=signal.exit_plan,
            invalidation=signal.invalidation_plan,
            leverage=3.0,
            reduce_only=False,
            post_only=False,
            expected_holding_seconds=1_800,
            metadata={"created_by": "ProbeStrategy"},
        )
        return signal


# -----------------------------------------------------------------------------
# Strict fake risk manager: consumes signal.generated, emits signal.confirmed.
# -----------------------------------------------------------------------------


class StrictRiskManagerHarness:
    """
    Цей harness навмисно не замінює strategy risk-payload conversion.
    Він перевіряє, що signal.generated має поля, які потрібні RiskManager,
    і емiтить signal.confirmed у форматі, який очікує TradeExecutor.
    """

    REQUIRED_SIGNAL_FIELDS = {
        "signal_id",
        "symbol",
        "side",
        "entry_price",
        "stop_loss",
        "strategy_name",
        "confidence",
        "edge_score",
        "order_intent",
    }

    def __init__(self, event_bus: InMemoryEventBus, *, max_risk_per_trade: float = 12.0) -> None:
        self.event_bus = event_bus
        self.max_risk_per_trade = max_risk_per_trade
        self.decisions: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.event_bus.subscribe("signal.generated", self.on_signal_generated, name="strict_risk_harness")

    async def on_signal_generated(self, event: CapturedEvent) -> None:
        payload = dict(event.payload)
        missing = sorted(field for field in self.REQUIRED_SIGNAL_FIELDS if payload.get(field) in (None, ""))
        if missing:
            self.rejections.append({"payload": payload, "reason": f"missing_fields:{missing}"})
            await self.event_bus.emit(
                "risk.position_blocked",
                {**payload, "reason": f"missing_fields:{missing}"},
                source="test.strict_risk_harness",
            )
            return

        entry = float(payload["entry_price"])
        stop = float(payload["stop_loss"])
        if entry <= 0 or stop <= 0 or entry == stop:
            self.rejections.append({"payload": payload, "reason": "invalid_entry_stop"})
            await self.event_bus.emit(
                "risk.position_blocked",
                {**payload, "reason": "invalid_entry_stop"},
                source="test.strict_risk_harness",
            )
            return

        side = str(payload["side"]).lower()
        if side == "long" and stop >= entry:
            reason = "long_stop_not_below_entry"
        elif side == "short" and stop <= entry:
            reason = "short_stop_not_above_entry"
        else:
            reason = ""

        if reason:
            self.rejections.append({"payload": payload, "reason": reason})
            await self.event_bus.emit("risk.position_blocked", {**payload, "reason": reason})
            return

        # Deliberately conservative but deterministic sizing.
        requested_size = float(payload.get("requested_size") or 0.002)
        final_size = max(0.001, min(requested_size, 0.01))
        final_leverage = float(payload.get("requested_leverage") or 3.0)
        final_notional = final_size * entry
        final_margin = final_notional / final_leverage
        final_risk_amount = min(self.max_risk_per_trade, abs(entry - stop) * final_size)

        # Strings are intentional: TradeExecutor must parse EventBus payloads robustly.
        decision = {
            **payload,
            "risk_decision_id": f"risk_{uuid4().hex}",
            "decision": "allow",
            "symbol": payload["symbol"],
            "side": side,
            "order_intent": str(payload.get("order_intent") or "open"),
            "final_size": final_size,
            "final_leverage": final_leverage,
            "final_tier": str(payload.get("tier") or "normal"),
            "final_risk_amount": final_risk_amount,
            "final_margin": final_margin,
            "final_notional": final_notional,
            "reservation_id": f"res_{uuid4().hex}",
            "reservation_expires_at": (datetime.now(timezone.utc) + timedelta(minutes=3)).timestamp(),
            "risk_mode": str(payload.get("risk_mode") or "normal"),
            "margin_mode": str(payload.get("margin_mode") or "isolated"),
            "exchange": payload.get("exchange") or "binance",
            "market_type": payload.get("market_type") or "usdm_futures",
            "metadata": {
                **dict(payload.get("metadata") or {}),
                "risk_harness": "strict",
                "source_signal_id": payload["signal_id"],
            },
        }
        self.decisions.append(decision)
        await self.event_bus.emit("signal.confirmed", decision, source="test.strict_risk_harness")


# -----------------------------------------------------------------------------
# Capturing execution dependencies
# -----------------------------------------------------------------------------


class CapturingOrderManager:
    """Fake OrderManager, який не ходить на Binance, але повертає FILLED OrderResult."""

    def __init__(self, event_bus: InMemoryEventBus) -> None:
        self.event_bus = event_bus
        self.requests: list[OrderRequest] = []
        self.results: list[OrderResult] = []
        self.cancel_all_calls: list[dict[str, Any]] = []

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        request.validate()
        self.requests.append(request)

        fill_price = request.price or float(request.metadata.get("entry_price") or request.metadata.get("last_price") or 50_000.0)
        qty = float(request.quantity or request.metadata.get("final_size") or 0.001)
        payload = {
            "exchange": request.exchange,
            "market_type": request.market_type,
            "symbol": request.symbol,
            "order_id": f"ord_{len(self.requests)}",
            "client_order_id": request.client_order_id or f"cid_{len(self.requests)}",
            "status": "filled",
            "side": request.side.value,
            "type": request.order_type.value,
            "price": fill_price,
            "avg_price": fill_price,
            "orig_qty": qty,
            "executed_qty": qty,
            "cum_quote": qty * fill_price,
            "position_side": request.position_side.value if request.position_side else None,
            "reduce_only": request.reduce_only,
            "close_position": request.close_position,
            "update_time": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        result = OrderResult.from_exchange_order(payload, request=request, metadata={"fake_fill": True})
        assert result.status is OrderStatus.FILLED
        self.results.append(result)

        await self.event_bus.emit(
            "execution.order_filled",
            result.to_event_payload(),
            source="test.capturing_order_manager",
        )
        return result

    async def cancel_all_orders(
        self,
        *,
        symbol: str,
        exchange: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        call = {"symbol": symbol, "exchange": exchange, "reason": reason}
        self.cancel_all_calls.append(call)
        return {"cancelled": True, **call}


class EmptyPositionManager:
    def get_position(self, *, symbol: str, side: Any | None = None) -> None:
        return None

    def list_positions(self, *, symbol: str | None = None, include_closed: bool = False) -> list[Any]:
        return []


# -----------------------------------------------------------------------------
# Synthetic analytics payloads
# -----------------------------------------------------------------------------


def _analytics_payload(domain: str, index: int) -> dict[str, Any]:
    side = "long" if index % 2 == 0 else "short"
    price = 50_000.0 + index * 17.0 + len(domain)
    ts = datetime.now(timezone.utc).isoformat()

    # Payload intentionally contains many aliases because SignalNormalizer has
    # domain-specific adapters. This makes the test brutal for schema drift.
    return {
        "event_id": f"{domain}-{index}",
        # SignalNormalizer._resolve_source() приймає тільки реальні FeatureSource values.
        # Для HYBRID у поточних enums немає FeatureSource.HYBRID: за контрактом
        # FeatureSource.from_strategy_category(StrategyCategory.HYBRID) -> FeatureSource.SYSTEM.
        "source": FeatureSource.from_strategy_category(ANALYTICS_DOMAIN_TO_CATEGORY[domain]).value,
        "source_topic": ANALYTICS_EVENT_TOPIC[domain],
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "usdm_futures",
        "timeframe": "1m",
        "timestamp": ts,
        "created_at": ts,
        "event_index": index,
        "last_price": price,
        "price": price,
        "close": price,
        "side": side,
        "direction": side,
        "bias": side,
        "signal": {
            "detected": True,
            "side": side,
            "direction": side,
            "score": 0.85,
            "confidence": 0.86,
            "entry_price": price,
            "stop_loss": price * (0.992 if side == "long" else 1.008),
            "take_profit": price * (1.018 if side == "long" else 0.982),
        },
        "analysis": {
            "score": 0.85,
            "confidence": 0.86,
            "side": side,
            "direction": side,
            "regime": "trending_up" if side == "long" else "trending_down",
        },
        "result": {
            "score": 0.85,
            "confidence": 0.86,
            "side": side,
            "direction": side,
        },
        "features": {
            "signal_score": 0.85,
            "confidence": 0.86,
            "dominant_side": side,
            "flow_imbalance": 0.72 if side == "long" else -0.72,
            "liquidity_score": 0.88,
            "spread_bps": 2.0,
            "volatility_zscore": 1.0,
            "funding_alignment": 0.2,
            "last_price": price,
        },
        domain: {
            "signal_score": 0.85,
            "confidence": 0.86,
            "dominant_side": side,
            "side": side,
            "direction": side,
            "entry_price": price,
            "stop_loss": price * (0.992 if side == "long" else 1.008),
            "take_profit": price * (1.018 if side == "long" else 0.982),
        },
        "metadata": {
            "domain": domain,
            "event_index": index,
            "test": "strategy_risk_execution_torture",
        },
    }


def _build_strategy_config() -> StrategyConfig:
    config = StrategyConfig(
        runtime=StrategyRuntimeConfig(
            enabled=True,
            symbols=[],
            timeframes=[Timeframe.M1],
            allowed_regimes=[MarketRegime.UNKNOWN],
            cooldown_seconds=0,
            max_signal_age_seconds=300,
            min_confidence=0.10,
            min_score=0.10,
        ),
        routing=RoutingConfig(
            reevaluate_on_any_update=True,
            route_hybrid_on_domain_signal=True,
            min_domains_for_hybrid_route=3,
            require_fresh_domains_for_hybrid_route=True,
            hybrid_route_stale_seconds=300,
            allow_partial_context=True,
            stale_feature_threshold_seconds=300,
            event_to_categories={
                # Explicit domain routes make the test resilient to default routing
                # changes. HYBRID is no longer appended here unconditionally;
                # SignalRouter may add it only when the StrategyContext already
                # contains enough fresh independent domain contracts.
                "analytics.orderflow.": [StrategyCategory.ORDERFLOW],
                "analytics.liquidity.": [StrategyCategory.LIQUIDITY],
                "analytics.price_action.": [StrategyCategory.PRICE_ACTION],
                "analytics.liquidations.": [StrategyCategory.LIQUIDATIONS],
                "analytics.whales.": [StrategyCategory.WHALES],
                "analytics.spoofing.": [StrategyCategory.SPOOFING],
                "analytics.spreads.": [StrategyCategory.SPREADS],
                "analytics.funding.": [StrategyCategory.FUNDING],
                "analytics.open_interest.": [StrategyCategory.OPEN_INTEREST],
            },
        ),
        confluence=ConfluenceConfig(enabled=True),
        voting=VotingConfig(
            min_confirmations=0,
            min_total_votes=1,
            require_primary_trigger=False,
            allow_single_strategy_confirmation=True,
        ),
        conflict=ConflictConfig(
            reject_on_side_conflict=False,
            reject_on_regime_conflict=False,
            max_total_penalty=10.0,
        ),
        filters=FilterConfig(
            enabled=True,
            min_signal_confidence=0.10,
            min_signal_score=0.10,
            min_risk_reward=0.10,
            enable_cooldown_filter=False,
            enable_freshness_filter=False,
            enable_portfolio_filter=False,
            enable_regime_filter=False,
            enable_volatility_filter=True,
            enable_liquidity_filter=True,
            enable_spread_filter=True,
            enable_funding_filter=True,
            max_spread_bps=50.0,
            min_liquidity_score=0.10,
            max_volatility_zscore=10.0,
            min_funding_alignment=-1.0,
        ),
        builders=BuilderConfig(
            default_entry_type=EntryType.MARKET,
            default_rr_ratio=2.0,
            enable_partial_take_profit=False,
            default_partial_tp_levels=[1.0],
            require_invalidation=True,
        ),
        portfolio=PortfolioCoordinatorConfig(
            enabled=True,
            max_signals_per_symbol=10_000,
            deduplicate_by_side=False,
            merge_similar_signals=False,
            correlation_guard_enabled=False,
            symbol_cooldown_seconds=0,
            side_cooldown_seconds=0,
            repeated_signal_suppression_seconds=0,
            volatility_throttle_enabled=False,
            high_volatility_max_signals_per_symbol=10_000,
        ),
    )
    config.validate()
    return config


def _register_probe_strategies(engine: StrategyEngine, config: StrategyConfig) -> list[ProbeStrategy]:
    probes: list[ProbeStrategy] = []
    for category in StrategyCategory:
        # 3 стратегії на категорію: це жорсткіше за один signal і добре ловить
        # помилки dedup/confluence/portfolio coordination.
        for n in range(3):
            strategy = ProbeStrategy(
                config=config,
                category=category,
                strategy_name=f"probe_{category.value}_{n}",
                side_offset=n,
            )
            engine.add_strategy(strategy)
            probes.append(strategy)
    assert engine.registry.count() == len(probes)
    return probes


def _market_context(_: Any) -> dict[str, Any]:
    return {
        "bid": 49_999.5,
        "ask": 50_000.5,
        "mark_price": 50_000.0,
        "last_price": 50_000.0,
        "tick_size": 0.1,
        "step_size": 0.001,
        "min_notional": 5.0,
        "available_depth_notional": 1_000_000.0,
        "expected_slippage_bps": 1.0,
    }


@pytest.mark.asyncio
async def test_analytics_strategy_risk_execution_torture_full_cycle() -> None:
    event_bus = InMemoryEventBus()
    config = _build_strategy_config()

    engine = StrategyEngine(config=config, event_bus=event_bus)
    probes = _register_probe_strategies(engine, config)
    assert len(probes) == len(StrategyCategory) * 3

    risk_harness = StrictRiskManagerHarness(event_bus)
    order_manager = CapturingOrderManager(event_bus)
    trade_executor = TradeExecutor(
        TradeExecutorConfig(
            enabled=True,
            auto_subscribe=True,
            register_scheduler_jobs=False,
            allow_new_entries=True,
            max_concurrent_executions=50,
            per_symbol_execution_lock=False,
            reject_expired_risk_reservations=True,
        ),
        order_manager=order_manager,
        position_manager=EmptyPositionManager(),
        sltp_manager=None,
        smart_execution=SmartExecution(
            SmartExecutionConfig(
                enabled=True,
                default_mode=ExecutionMode.MARKET,
                fallback_mode=ExecutionMode.MARKET,
                max_slippage_bps=50.0,
                max_spread_bps=50.0,
                allow_order_splitting=False,
                min_leg_notional=5.0,
            )
        ),
        event_bus=event_bus,
        scheduler=None,
        market_context_provider=_market_context,
        auto_subscribe=True,
        register_scheduler_jobs=False,
    )
    trade_executor.register()

    batches = []
    per_domain_batch_counts: Counter[str] = Counter()

    # 10 analytics signals/events від кожної РЕАЛЬНОЇ аналітики.
    # Hybrid не має окремого analytics.hybrid event; він має приходити як
    # додаткова StrategyCategory.HYBRID evaluation на domain events.
    for domain in ANALYTICS_DOMAIN_TO_CATEGORY:
        for index in range(10):
            payload = _analytics_payload(domain, index)
            batch = await engine.process_analytics_event(
                event_name=ANALYTICS_EVENT_TOPIC[domain],
                payload=payload,
            )
            batches.append((domain, batch))
            per_domain_batch_counts[domain] += 1

    assert not event_bus.failures
    assert per_domain_batch_counts == Counter({domain: 10 for domain in ANALYTICS_DOMAIN_TO_CATEGORY})

    rejected_batches = [(domain, batch) for domain, batch in batches if not batch.accepted]
    assert not rejected_batches, [
        {
            "domain": domain,
            "reasons": batch.reasons,
            "metadata": batch.metadata,
        }
        for domain, batch in rejected_batches[:10]
    ]

    # Boundary checks: strategy -> risk.
    signal_generated = event_bus.by_topic("signal.generated")
    signal_confirmed = event_bus.by_topic("signal.confirmed")
    assert len(signal_generated) >= len(ANALYTICS_DOMAIN_TO_CATEGORY) * 10
    assert len(signal_confirmed) == len(signal_generated)
    assert len(risk_harness.rejections) == 0
    assert len(risk_harness.decisions) == len(signal_generated)

    # Boundary checks: risk -> execution.
    assert event_bus.count("execution.trade_requested") == len(signal_confirmed)
    assert event_bus.count("execution.trade_accepted") == len(signal_confirmed)
    assert event_bus.count("execution.execution_plan_created") == len(signal_confirmed)
    assert event_bus.count("execution.execution_submitted") == len(signal_confirmed)
    assert event_bus.count("execution.execution_completed") == len(signal_confirmed)
    assert event_bus.count("execution.trade_rejected") == 0
    assert event_bus.count("execution.execution_failed") == 0

    # Order boundary: execution -> exchange adapter fake.
    assert len(order_manager.requests) == len(signal_confirmed)
    assert len(order_manager.results) == len(signal_confirmed)
    assert all(result.status is OrderStatus.FILLED for result in order_manager.results)

    # Жорстка перевірка domain strategy coverage: кожна реальна аналітика
    # має дати raw strategy signals. Це перевіряється ДО ConfluenceEngine,
    # бо фінальний selected/merged signal може бути domain або HYBRID.
    raw_by_domain: Counter[str] = Counter()
    routed_categories_by_event: list[set[str]] = []

    for domain, batch in batches:
        categories = {str(getattr(signal.category, "value", signal.category)) for signal in batch.raw_signals}
        routed_categories_by_event.append(categories)
        expected_category = ANALYTICS_DOMAIN_TO_CATEGORY[domain].value
        assert expected_category in categories, {
            "domain": domain,
            "expected_category": expected_category,
            "actual_categories": sorted(categories),
            "batch_reasons": batch.reasons,
            "batch_metadata": batch.metadata,
        }

        for signal in batch.raw_signals:
            metadata = dict(signal.metadata or {})
            signal_domain = metadata.get("torture_domain") or getattr(signal.category, "value", signal.category)
            raw_by_domain[str(signal_domain)] += 1

    for domain in ANALYTICS_DOMAIN_TO_CATEGORY:
        assert raw_by_domain[domain] >= 10, raw_by_domain

    generated_by_domain: Counter[str] = Counter()
    completed_by_domain: Counter[str] = Counter()

    for event in signal_generated:
        metadata = dict(event.payload.get("metadata") or {})
        domain = metadata.get("torture_domain") or metadata.get("category")
        generated_by_domain[str(domain)] += 1

    for event in event_bus.by_topic("execution.execution_completed"):
        metadata = dict(event.payload.get("metadata") or {})
        domain = metadata.get("torture_domain") or metadata.get("category")
        completed_by_domain[str(domain)] += 1

    assert sum(generated_by_domain.values()) == len(signal_generated), generated_by_domain
    assert sum(completed_by_domain.values()) == event_bus.count("execution.execution_completed"), completed_by_domain

    # HYBRID не має бути окремою analytics-подією і не має запускатись
    # на першому ж однодоменному контексті. Він з'являється тільки після того,
    # як SignalRouter побачив мінімум 3 свіжі незалежні domain contracts.
    assert not any(event.topic.startswith("analytics.hybrid") for event in event_bus.events)
    assert StrategyCategory.HYBRID.value not in routed_categories_by_event[0]
    assert any(StrategyCategory.HYBRID.value in categories for categories in routed_categories_by_event), routed_categories_by_event
    assert generated_by_domain[StrategyCategory.HYBRID.value] > 0, generated_by_domain
    assert completed_by_domain[StrategyCategory.HYBRID.value] > 0, completed_by_domain

    # Стратегічні інваріанти, які часто ламаються при неправильній зв'язці.
    for event in signal_generated:
        payload = event.payload
        assert payload["exchange"] == "binance"
        assert str(payload["market_type"]).lower() in {"usdm_futures", "usdm-futures"}
        assert payload["entry_price"] > 0
        assert payload["stop_loss"] > 0
        assert payload["take_profit"] > 0
        assert payload["side"] in {"long", "short", SignalSide.LONG, SignalSide.SHORT}
        assert payload["strategy_name"].startswith("probe_")

    for request in order_manager.requests:
        assert request.exchange == "binance"
        assert request.market_type == "usdm_futures"
        assert request.symbol == "BTCUSDT"
        assert request.quantity is not None and request.quantity > 0
        assert request.execution_id
        assert request.signal_id
        assert request.strategy_name
        assert request.metadata.get("reservation_id") or request.reservation_id


@pytest.mark.asyncio
async def test_torture_detects_bad_risk_payload_and_does_not_execute() -> None:
    """
    Негативний контроль: risk harness має заблокувати пошкоджений signal.generated,
    а execution не має отримати жодного signal.confirmed/order.
    """
    event_bus = InMemoryEventBus()
    risk_harness = StrictRiskManagerHarness(event_bus)
    order_manager = CapturingOrderManager(event_bus)

    trade_executor = TradeExecutor(
        TradeExecutorConfig(auto_subscribe=True, register_scheduler_jobs=False),
        order_manager=order_manager,
        position_manager=EmptyPositionManager(),
        smart_execution=SmartExecution(SmartExecutionConfig(default_mode=ExecutionMode.MARKET)),
        event_bus=event_bus,
        market_context_provider=_market_context,
        auto_subscribe=True,
        register_scheduler_jobs=False,
    )
    trade_executor.register()

    await event_bus.emit(
        "signal.generated",
        {
            # intentionally missing entry_price/stop_loss/strategy_name/confidence/edge_score
            "signal_id": "broken_signal",
            "symbol": "BTCUSDT",
            "side": "long",
            "order_intent": "open",
        },
        source="test.negative_control",
    )

    assert len(risk_harness.rejections) == 1
    assert event_bus.count("risk.position_blocked") == 1
    assert event_bus.count("signal.confirmed") == 0
    assert event_bus.count("execution.trade_requested") == 0
    assert len(order_manager.requests) == 0