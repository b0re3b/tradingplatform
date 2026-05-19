# # tests/strategy/funding/conftest.py
#
# from __future__ import annotations
#
# import inspect
# import json
# from dataclasses import dataclass, field
# from datetime import datetime, timedelta, timezone
# from enum import Enum
# from types import SimpleNamespace
# from typing import Any, Callable, Mapping
#
# import pytest
#
# from analytics.funding.enums import (
#     FundingBias,
#     FundingDivergenceType,
#     FundingExtremeType,
#     FundingFlipType,
#     FundingPressureDirection,
#     FundingPressureLevel,
#     FundingRegime,
#     FundingSignalType,
#     FundingTimeframe,
# )
# from core.event_bus import Event, EventPriority
#
# from strategy.strategies.funding.base import (
#     BaseFundingStrategyConfig,
#     FundingSetupStatus,
#     FundingStrategyDirection,
#     FundingStrategyScope,
#     FundingStrategyState,
# )
# from strategy.strategies.funding.funding_divergence_strategy import (
#     FundingDivergenceStrategy,
#     FundingDivergenceStrategyConfig,
# )
# from strategy.strategies.funding.funding_extreme_reversal_strategy import (
#     FundingExtremeReversalStrategy,
#     FundingExtremeReversalStrategyConfig,
# )
#
#
# # =============================================================================
# # Generic constants / helpers
# # =============================================================================
#
#
# UTC = timezone.utc
#
# DEFAULT_SYMBOL = "BTCUSDT"
# DEFAULT_EXCHANGE = "binance"
# DEFAULT_MARKET_TYPE = "usdm_futures"
# DEFAULT_TIMEFRAME = FundingTimeframe.H1
# DEFAULT_EXCHANGE_SYMBOL = "BTCUSDT"
#
# ALT_SYMBOL = "ETHUSDT"
# ALT_EXCHANGE = "bybit"
# ALT_MARKET_TYPE = "linear"
# ALT_TIMEFRAME = FundingTimeframe.H4
# ALT_EXCHANGE_SYMBOL = "ETHUSDT"
#
# COINM_MARKET_TYPE = "coinm_futures"
# COINM_EXCHANGE_SYMBOL = "BTCUSD_PERP"
#
#
# def now_utc() -> datetime:
#     return datetime.now(tz=UTC)
#
#
# def iso_now() -> str:
#     return now_utc().isoformat()
#
#
# def iso_ago(seconds: float) -> str:
#     return (now_utc() - timedelta(seconds=seconds)).isoformat()
#
#
# def iso_after(seconds: float) -> str:
#     return (now_utc() + timedelta(seconds=seconds)).isoformat()
#
#
# def enum_value(value: Any) -> Any:
#     if isinstance(value, Enum):
#         return value.value
#     return value
#
#
# def enum_name_value(enum_cls: type[Enum], name: str, fallback: str) -> str:
#     item = getattr(enum_cls, name, None)
#     if isinstance(item, Enum):
#         return str(item.value)
#     return fallback
#
#
# def timeframe_value(value: Any = DEFAULT_TIMEFRAME) -> str:
#     raw = enum_value(value)
#     return str(raw)
#
#
# def normalized_symbol(value: str = DEFAULT_SYMBOL) -> str:
#     return value.replace("/", "").replace("-", "").replace("_", "").upper()
#
#
# def normalized_exchange(value: str = DEFAULT_EXCHANGE) -> str:
#     return value.lower().strip()
#
#
# def normalized_market_type(value: str = DEFAULT_MARKET_TYPE) -> str:
#     return value.lower().strip()
#
#
# def normalized_exchange_symbol(
#     value: str | None = None,
#     *,
#     fallback_symbol: str = DEFAULT_SYMBOL,
# ) -> str:
#     return str(value or fallback_symbol).strip()
#
#
# def merge_payload(
#     base: dict[str, Any],
#     overrides: Mapping[str, Any] | None = None,
# ) -> dict[str, Any]:
#     data = dict(base)
#     if overrides:
#         data.update(dict(overrides))
#     return data
#
#
# def scope_dict(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
# ) -> dict[str, str]:
#     symbol_value = normalized_symbol(symbol)
#     return {
#         "exchange": normalized_exchange(exchange),
#         "market_type": normalized_market_type(market_type),
#         "symbol": symbol_value,
#         "timeframe": timeframe_value(timeframe),
#         "exchange_symbol": normalized_exchange_symbol(
#             exchange_symbol,
#             fallback_symbol=symbol_value,
#         ),
#     }
#
#
# def make_scope(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
# ) -> FundingStrategyScope:
#     return FundingStrategyScope(
#         exchange=exchange,
#         market_type=market_type,
#         symbol=symbol,
#         timeframe=timeframe_value(timeframe),
#         exchange_symbol=exchange_symbol,
#     )
#
#
# def state_key(
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
# ) -> str:
#     return make_scope(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#     ).key
#
#
# def legacy_state_key(
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
# ) -> str:
#     return f"{normalized_symbol(symbol)}:{normalized_exchange(exchange)}"
#
#
# def payload_metadata(
#     *,
#     fixture: str,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     extra: Mapping[str, Any] | None = None,
# ) -> dict[str, Any]:
#     symbol_value = normalized_symbol(symbol)
#     data: dict[str, Any] = {
#         "fixture": fixture,
#         "scope": scope_dict(
#             symbol=symbol_value,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#         ),
#         "exchange_symbol": normalized_exchange_symbol(
#             exchange_symbol,
#             fallback_symbol=symbol_value,
#         ),
#     }
#     if extra:
#         data.update(dict(extra))
#     return data
#
#
# def scoped_base_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     event_time: str | datetime | None = None,
#     fixture: str,
#     metadata_extra: Mapping[str, Any] | None = None,
# ) -> dict[str, Any]:
#     symbol_value = normalized_symbol(symbol)
#     return {
#         "symbol": symbol_value,
#         "exchange": normalized_exchange(exchange),
#         "market_type": normalized_market_type(market_type),
#         "timeframe": timeframe_value(timeframe),
#         "exchange_symbol": normalized_exchange_symbol(
#             exchange_symbol,
#             fallback_symbol=symbol_value,
#         ),
#         "event_time": event_time or iso_now(),
#         "metadata": payload_metadata(
#             fixture=fixture,
#             symbol=symbol_value,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#             extra=metadata_extra,
#         ),
#     }
#
#
# def last_emitted_payload(event_bus: "SpyEventBus") -> dict[str, Any]:
#     assert event_bus.emitted, "Expected at least one emitted event"
#     return dict(event_bus.emitted[-1].payload)
#
#
# def last_emitted_topic(event_bus: "SpyEventBus") -> str:
#     assert event_bus.emitted, "Expected at least one emitted event"
#     return event_bus.emitted[-1].topic
#
#
# def emitted_topics(event_bus: "SpyEventBus") -> list[str]:
#     return [record.topic for record in event_bus.emitted]
#
#
# def payloads_for_topic(event_bus: "SpyEventBus", topic: str) -> list[dict[str, Any]]:
#     return [record.payload for record in event_bus.emitted if record.topic == topic]
#
#
# def last_payload_for_topic(event_bus: "SpyEventBus", topic: str) -> dict[str, Any]:
#     payloads = payloads_for_topic(event_bus, topic)
#     assert payloads, f"No payloads for topic={topic!r}; topics={emitted_topics(event_bus)}"
#     return payloads[-1]
#
#
# def assert_scope_payload(
#     payload: Mapping[str, Any],
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
# ) -> None:
#     expected = scope_dict(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#     assert payload["symbol"] == expected["symbol"]
#     assert payload["exchange"] == expected["exchange"]
#     assert payload["market_type"] == expected["market_type"]
#     assert payload["timeframe"] == expected["timeframe"]
#     assert payload["exchange_symbol"] == expected["exchange_symbol"]
#     assert payload["scope"] == expected
#
#
# def assert_scope_contract(
#     record: "EmittedEventRecord",
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
# ) -> None:
#     expected = scope_dict(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#     payload = record.payload
#
#     assert payload["symbol"] == expected["symbol"]
#     assert payload["exchange"] == expected["exchange"]
#     assert payload["market_type"] == expected["market_type"]
#     assert payload["timeframe"] == expected["timeframe"]
#     assert payload["exchange_symbol"] == expected["exchange_symbol"]
#     assert payload["scope"] == expected
#
#     assert record.headers["symbol"] == expected["symbol"]
#     assert record.headers["exchange"] == expected["exchange"]
#     assert record.headers["market_type"] == expected["market_type"]
#     assert record.headers["timeframe"] == expected["timeframe"]
#     assert record.headers["exchange_symbol"] == expected["exchange_symbol"]
#     assert record.headers["scope"] == expected
#
#     if "state" in payload:
#         assert payload["state"]["scope"] == expected
#         assert payload["state"]["key"] == state_key(
#             symbol=symbol,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#         )
#         assert payload["state"]["legacy_key"] == legacy_state_key(
#             symbol=symbol,
#             exchange=exchange,
#         )
#
#     if "funding_context" in payload:
#         assert payload["funding_context"]["scope"] == expected
#
#     if "analytics_context" in payload:
#         assert payload["analytics_context"]["scope"] == expected
#
#
# def assert_state_scope(
#     state: FundingStrategyState,
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
# ) -> None:
#     expected = scope_dict(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#     assert state.symbol == expected["symbol"]
#     assert state.exchange == expected["exchange"]
#     assert state.market_type == expected["market_type"]
#     assert state.timeframe.value == expected["timeframe"]
#     assert state.exchange_symbol == expected["exchange_symbol"]
#     assert state.scope.to_dict() == expected
#     assert state.key == state_key(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#     )
#     assert state.legacy_key == legacy_state_key(symbol=symbol, exchange=exchange)
#
#
# # =============================================================================
# # EventBus / Scheduler / Parquet spies
# # =============================================================================
#
#
# @dataclass(slots=True)
# class SpySubscription:
#     pattern: str
#     handler: Callable[..., Any]
#     name: str | None = None
#
#
# @dataclass(slots=True)
# class EmittedEventRecord:
#     topic: str
#     payload: dict[str, Any]
#     priority: Any = None
#     source: str | None = None
#     correlation_id: str | None = None
#     headers: dict[str, Any] = field(default_factory=dict)
#     kwargs: dict[str, Any] = field(default_factory=dict)
#
#
# class SpyEventBus:
#     """
#     Minimal EventBus test double.
#
#     Keeps `_subscriptions` because BaseFundingStrategy snapshots EventBus internals
#     during register().
#     """
#
#     def __init__(self, *, emit_result: bool = True) -> None:
#         self.emit_result = emit_result
#         self.raise_on_emit: Exception | None = None
#
#         self._subscriptions: list[SpySubscription] = []
#         self.subscribed: list[SpySubscription] = []
#         self.unsubscribed: list[SpySubscription] = []
#         self.emitted: list[EmittedEventRecord] = []
#
#     def subscribe(
#         self,
#         pattern: str,
#         handler: Callable[..., Any],
#         *,
#         name: str | None = None,
#         **_: Any,
#     ) -> SpySubscription:
#         subscription = SpySubscription(pattern=pattern, handler=handler, name=name)
#         self._subscriptions.append(subscription)
#         self.subscribed.append(subscription)
#         return subscription
#
#     def unsubscribe(self, subscription: SpySubscription) -> None:
#         self.unsubscribed.append(subscription)
#         if subscription in self._subscriptions:
#             self._subscriptions.remove(subscription)
#
#     async def emit(
#         self,
#         topic: str,
#         payload: Mapping[str, Any] | None = None,
#         *,
#         priority: Any = None,
#         source: str | None = None,
#         correlation_id: str | None = None,
#         headers: Mapping[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> bool:
#         if self.raise_on_emit is not None:
#             raise self.raise_on_emit
#
#         self.emitted.append(
#             EmittedEventRecord(
#                 topic=topic,
#                 payload=dict(payload or {}),
#                 priority=priority,
#                 source=source,
#                 correlation_id=correlation_id,
#                 headers=dict(headers or {}),
#                 kwargs=dict(kwargs),
#             )
#         )
#         return self.emit_result
#
#     def clear(self) -> None:
#         self.emitted.clear()
#         self.subscribed.clear()
#         self.unsubscribed.clear()
#
#     def topics(self) -> list[str]:
#         return emitted_topics(self)
#
#     def payloads_for(self, topic: str) -> list[dict[str, Any]]:
#         return payloads_for_topic(self, topic)
#
#     def last_payload_for(self, topic: str) -> dict[str, Any]:
#         return last_payload_for_topic(self, topic)
#
#
# @dataclass(slots=True)
# class SpyScheduledJob:
#     job_id: str
#     name: str
#     func: Callable[..., Any]
#     interval: float
#     kwargs: dict[str, Any]
#     run_immediately: bool
#     max_retries: int
#     retry_delay: float
#     timeout: float | None
#     allow_overlap: bool
#     enabled: bool
#
#
# class SpyScheduler:
#     def __init__(self) -> None:
#         self.jobs: dict[str, SpyScheduledJob] = {}
#         self.added_jobs: list[SpyScheduledJob] = []
#         self.removed_job_ids: list[str] = []
#
#     def get_job_by_name(self, name: str) -> SpyScheduledJob | None:
#         for job in self.jobs.values():
#             if job.name == name:
#                 return job
#         return None
#
#     def add_interval_job(
#         self,
#         *,
#         name: str,
#         func: Callable[..., Any],
#         interval: float,
#         kwargs: Mapping[str, Any] | None = None,
#         run_immediately: bool = False,
#         max_retries: int = 0,
#         retry_delay: float = 0.0,
#         timeout: float | None = None,
#         allow_overlap: bool = False,
#         enabled: bool = True,
#         **_: Any,
#     ) -> str:
#         job_id = f"job-{len(self.jobs) + 1}"
#         job = SpyScheduledJob(
#             job_id=job_id,
#             name=name,
#             func=func,
#             interval=interval,
#             kwargs=dict(kwargs or {}),
#             run_immediately=run_immediately,
#             max_retries=max_retries,
#             retry_delay=retry_delay,
#             timeout=timeout,
#             allow_overlap=allow_overlap,
#             enabled=enabled,
#         )
#         self.jobs[job_id] = job
#         self.added_jobs.append(job)
#         return job_id
#
#     def remove_job(self, job_id: str) -> None:
#         if job_id not in self.jobs:
#             raise KeyError(job_id)
#         self.removed_job_ids.append(job_id)
#         del self.jobs[job_id]
#
#
# class SpyParquetStorage:
#     """
#     Fake storage for BaseFundingStrategy generated-signal parquet path.
#
#     Base calls:
#         append_records(dataset=..., records=[...])
#     """
#
#     def __init__(self, *, raise_on_append: Exception | None = None) -> None:
#         self.raise_on_append = raise_on_append
#         self.append_calls: list[dict[str, Any]] = []
#         self.records_by_dataset: dict[str, list[dict[str, Any]]] = {}
#
#     def append_records(self, *, dataset: str, records: list[dict[str, Any]]) -> None:
#         if self.raise_on_append is not None:
#             raise self.raise_on_append
#
#         copied = [dict(record) for record in records]
#         self.append_calls.append({"dataset": dataset, "records": copied})
#         self.records_by_dataset.setdefault(dataset, []).extend(copied)
#
#     def all_records(self) -> list[dict[str, Any]]:
#         result: list[dict[str, Any]] = []
#         for records in self.records_by_dataset.values():
#             result.extend(records)
#         return result
#
#
# # =============================================================================
# # Event factory
# # =============================================================================
#
#
# def make_event(
#     topic: str,
#     payload: Mapping[str, Any],
#     *,
#     correlation_id: str = "test-correlation-id",
#     priority: Any = EventPriority.NORMAL,
#     source: str = "pytest",
#     headers: Mapping[str, Any] | None = None,
# ) -> Event | SimpleNamespace:
#     """
#     Build core.event_bus.Event while being tolerant to constructor differences.
#
#     Fallback object gives handlers the attributes they use:
#     payload, correlation_id, topic/event_type/name and headers.
#     """
#     event_headers = dict(headers or {"test": True})
#
#     constructor_attempts = [
#         {
#             "topic": topic,
#             "payload": dict(payload),
#             "priority": priority,
#             "source": source,
#             "correlation_id": correlation_id,
#             "headers": event_headers,
#         },
#         {
#             "event_type": topic,
#             "payload": dict(payload),
#             "priority": priority,
#             "source": source,
#             "correlation_id": correlation_id,
#             "headers": event_headers,
#         },
#         {
#             "name": topic,
#             "payload": dict(payload),
#             "priority": priority,
#             "source": source,
#             "correlation_id": correlation_id,
#             "headers": event_headers,
#         },
#     ]
#
#     try:
#         signature = inspect.signature(Event)
#         params = set(signature.parameters)
#         for candidate in constructor_attempts:
#             kwargs = {key: value for key, value in candidate.items() if key in params}
#             if "payload" not in kwargs:
#                 continue
#             try:
#                 return Event(**kwargs)
#             except TypeError:
#                 continue
#     except (TypeError, ValueError):
#         pass
#
#     for candidate in constructor_attempts:
#         try:
#             return Event(**candidate)
#         except TypeError:
#             continue
#
#     try:
#         return Event(topic, dict(payload))
#     except TypeError:
#         return SimpleNamespace(
#             topic=topic,
#             event_type=topic,
#             name=topic,
#             payload=dict(payload),
#             priority=priority,
#             source=source,
#             correlation_id=correlation_id,
#             headers=event_headers,
#         )
#
#
# # =============================================================================
# # Payload builders
# # =============================================================================
#
#
# def snapshot_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     funding_rate: float = 0.0007,
#     predicted_funding_rate: float | None = 0.00072,
#     mark_price: float | None = 50_000.0,
#     index_price: float | None = 49_980.0,
#     open_interest: float | None = 1_000_000.0,
#     volume_24h: float | None = 2_500_000.0,
#     next_funding_time: str | datetime | None = None,
#     event_time: str | datetime | None = None,
#     received_at: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="snapshot",
#             ),
#             "funding_rate": funding_rate,
#             "predicted_funding_rate": predicted_funding_rate,
#             "mark_price": mark_price,
#             "index_price": index_price,
#             "open_interest": open_interest,
#             "volume_24h": volume_24h,
#             "next_funding_time": next_funding_time or iso_after(8 * 60 * 60),
#             "received_at": received_at or iso_now(),
#         },
#         overrides,
#     )
#
#
# def statistics_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     current_rate: float = 0.0007,
#     mean_rate: float = 0.0003,
#     median_rate: float = 0.00028,
#     std_rate: float = 0.00015,
#     min_rate: float = -0.0002,
#     max_rate: float = 0.0012,
#     zscore: float | None = 2.0,
#     percentile: float | None = 90.0,
#     sample_size: int = 100,
#     window_start: str | datetime | None = None,
#     window_end: str | datetime | None = None,
#     updated_at: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=updated_at,
#                 fixture="statistics",
#             ),
#             "current_rate": current_rate,
#             "mean_rate": mean_rate,
#             "median_rate": median_rate,
#             "std_rate": std_rate,
#             "min_rate": min_rate,
#             "max_rate": max_rate,
#             "zscore": zscore,
#             "percentile": percentile,
#             "sample_size": sample_size,
#             "window_start": window_start or iso_ago(60 * 60),
#             "window_end": window_end or iso_now(),
#             "updated_at": updated_at or iso_now(),
#         },
#         overrides,
#     )
#
#
# def regime_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     regime: Any = FundingRegime.POSITIVE,
#     bias: Any = FundingBias.LONG_BIAS,
#     current_rate: float = 0.0008,
#     mean_rate: float = 0.0003,
#     zscore: float = 2.0,
#     percentile: float = 90.0,
#     confidence: float = 0.85,
#     changed: bool = False,
#     previous_regime: Any | None = None,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="regime",
#             ),
#             "regime": enum_value(regime),
#             "bias": enum_value(bias),
#             "current_rate": current_rate,
#             "mean_rate": mean_rate,
#             "zscore": zscore,
#             "percentile": percentile,
#             "confidence": confidence,
#             "changed": changed,
#             "previous_regime": enum_value(previous_regime) if previous_regime is not None else None,
#         },
#         overrides,
#     )
#
#
# def pressure_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     direction: Any = FundingPressureDirection.LONG,
#     level: Any = FundingPressureLevel.HIGH,
#     bias: Any = FundingBias.LONG_BIAS,
#     funding_rate: float = 0.0009,
#     pressure_score: float = 0.82,
#     oi_confirmation: bool = True,
#     price_stall_confirmation: bool = True,
#     squeeze_probability: float = 0.72,
#     mean_reversion_probability: float = 0.68,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="pressure",
#             ),
#             "direction": enum_value(direction),
#             "level": enum_value(level),
#             "bias": enum_value(bias),
#             "funding_rate": funding_rate,
#             "pressure_score": pressure_score,
#             "oi_confirmation": oi_confirmation,
#             "price_stall_confirmation": price_stall_confirmation,
#             "squeeze_probability": squeeze_probability,
#             "mean_reversion_probability": mean_reversion_probability,
#         },
#         overrides,
#     )
#
#
# def positive_extreme_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     extreme_type: Any | None = None,
#     regime: Any = FundingRegime.POSITIVE,
#     funding_rate: float = 0.0012,
#     zscore: float = 3.0,
#     percentile: float = 98.0,
#     severity: float = 0.90,
#     is_reversal_risk: bool = True,
#     is_squeeze_risk: bool = True,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     resolved_extreme_type = extreme_type or FundingExtremeType.ZSCORE_HIGH
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="positive_extreme",
#             ),
#             "extreme_type": enum_value(resolved_extreme_type),
#             "regime": enum_value(regime),
#             "funding_rate": funding_rate,
#             "zscore": zscore,
#             "percentile": percentile,
#             "severity": severity,
#             "is_reversal_risk": is_reversal_risk,
#             "is_squeeze_risk": is_squeeze_risk,
#         },
#         overrides,
#     )
#
#
# def negative_extreme_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     extreme_type: Any | None = None,
#     regime: Any = FundingRegime.NEGATIVE,
#     funding_rate: float = -0.0012,
#     zscore: float = -3.0,
#     percentile: float = 2.0,
#     severity: float = 0.90,
#     is_reversal_risk: bool = True,
#     is_squeeze_risk: bool = True,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     resolved_extreme_type = extreme_type or FundingExtremeType.ZSCORE_LOW
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="negative_extreme",
#             ),
#             "extreme_type": enum_value(resolved_extreme_type),
#             "regime": enum_value(regime),
#             "funding_rate": funding_rate,
#             "zscore": zscore,
#             "percentile": percentile,
#             "severity": severity,
#             "is_reversal_risk": is_reversal_risk,
#             "is_squeeze_risk": is_squeeze_risk,
#         },
#         overrides,
#     )
#
#
# def bullish_divergence_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     divergence_type: Any | None = None,
#     funding_rate: float = -0.0006,
#     price_change_pct: float = 0.012,
#     oi_change_pct: float = 0.024,
#     cvd_change: float = 100_000.0,
#     long_liquidations: float = 5_000.0,
#     short_liquidations: float = 45_000.0,
#     confidence: float = 0.82,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     resolved_type = divergence_type or FundingDivergenceType.PRICE_UP_FUNDING_DOWN
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="bullish_divergence",
#             ),
#             "divergence_type": enum_value(resolved_type),
#             "funding_rate": funding_rate,
#             "price_change_pct": price_change_pct,
#             "oi_change_pct": oi_change_pct,
#             "cvd_change": cvd_change,
#             "long_liquidations": long_liquidations,
#             "short_liquidations": short_liquidations,
#             "confidence": confidence,
#         },
#         overrides,
#     )
#
#
# def bearish_divergence_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     divergence_type: Any | None = None,
#     funding_rate: float = 0.0008,
#     price_change_pct: float = -0.012,
#     oi_change_pct: float = 0.024,
#     cvd_change: float = -100_000.0,
#     long_liquidations: float = 45_000.0,
#     short_liquidations: float = 5_000.0,
#     confidence: float = 0.82,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     resolved_type = divergence_type or FundingDivergenceType.PRICE_DOWN_FUNDING_UP
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="bearish_divergence",
#             ),
#             "divergence_type": enum_value(resolved_type),
#             "funding_rate": funding_rate,
#             "price_change_pct": price_change_pct,
#             "oi_change_pct": oi_change_pct,
#             "cvd_change": cvd_change,
#             "long_liquidations": long_liquidations,
#             "short_liquidations": short_liquidations,
#             "confidence": confidence,
#         },
#         overrides,
#     )
#
#
# def positive_to_negative_flip_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     previous_rate: float = 0.0008,
#     current_rate: float = -0.0002,
#     flip_magnitude: float = 0.0010,
#     confidence: float = 0.82,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="positive_to_negative_flip",
#             ),
#             "flip_type": enum_value(FundingFlipType.POSITIVE_TO_NEGATIVE),
#             "previous_rate": previous_rate,
#             "current_rate": current_rate,
#             "flip_magnitude": flip_magnitude,
#             "confidence": confidence,
#         },
#         overrides,
#     )
#
#
# def negative_to_positive_flip_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     previous_rate: float = -0.0008,
#     current_rate: float = 0.0002,
#     flip_magnitude: float = 0.0010,
#     confidence: float = 0.82,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     return merge_payload(
#         {
#             **scoped_base_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#                 fixture="negative_to_positive_flip",
#             ),
#             "flip_type": enum_value(FundingFlipType.NEGATIVE_TO_POSITIVE),
#             "previous_rate": previous_rate,
#             "current_rate": current_rate,
#             "flip_magnitude": flip_magnitude,
#             "confidence": confidence,
#         },
#         overrides,
#     )
#
#
# def funding_signal_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     signal_type: Any = FundingSignalType.REVERSION_SETUP,
#     bias: Any = FundingBias.NEUTRAL,
#     regime: Any = FundingRegime.UNKNOWN,
#     score: float = 0.70,
#     confidence: float = 0.80,
#     signal_origin: str = "pressure_reversion",
#     description: str = "pytest funding signal",
#     supporting_factors: list[str] | None = None,
#     tags: list[str] | None = None,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     payload = {
#         **scoped_base_payload(
#             symbol=symbol,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#             event_time=event_time,
#             fixture="funding_signal",
#             metadata_extra={"signal_origin": signal_origin},
#         ),
#         "signal_type": enum_value(signal_type),
#         "bias": enum_value(bias),
#         "regime": enum_value(regime),
#         "score": score,
#         "confidence": confidence,
#         "description": description,
#         "supporting_factors": supporting_factors or ["pytest"],
#         "tags": tags or ["pytest"],
#     }
#     return merge_payload(payload, overrides)
#
#
# def funding_signal_envelope_payload(
#     *,
#     signal: Mapping[str, Any] | None = None,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     base = scoped_base_payload(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#         event_time=event_time,
#         fixture="funding_signal_envelope",
#     )
#     payload = {
#         **base,
#         "event_type": "signal",
#         "source": "funding_analyzer",
#         "payload": dict(signal or funding_signal_payload(
#             symbol=symbol,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#             event_time=event_time,
#         )),
#     }
#     return merge_payload(payload, overrides)
#
#
# def funding_updated_payload(
#     *,
#     symbol: str = DEFAULT_SYMBOL,
#     exchange: str = DEFAULT_EXCHANGE,
#     market_type: str = DEFAULT_MARKET_TYPE,
#     timeframe: Any = DEFAULT_TIMEFRAME,
#     exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     snapshot: Mapping[str, Any] | None = None,
#     statistics: Mapping[str, Any] | None = None,
#     regime_state: Mapping[str, Any] | None = None,
#     pressure_state: Mapping[str, Any] | None = None,
#     extreme_event: Mapping[str, Any] | None = None,
#     divergence_event: Mapping[str, Any] | None = None,
#     flip_event: Mapping[str, Any] | None = None,
#     signal: Mapping[str, Any] | None = None,
#     event_time: str | datetime | None = None,
#     **overrides: Any,
# ) -> dict[str, Any]:
#     base = scoped_base_payload(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#         event_time=event_time,
#         fixture="funding_updated",
#     )
#     nested_scope = scope_dict(
#         symbol=symbol,
#         exchange=exchange,
#         market_type=market_type,
#         timeframe=timeframe,
#         exchange_symbol=exchange_symbol,
#     )
#
#     payload = {
#         **base,
#         "event_type": "snapshot",
#         "source": "funding_analyzer",
#         "payload": {
#             **nested_scope,
#             "scope": nested_scope,
#             "snapshot": dict(snapshot or snapshot_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 event_time=event_time,
#             )),
#             "statistics": dict(statistics or statistics_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#             )),
#             "regime_state": dict(regime_state or {}),
#             "pressure_state": dict(pressure_state or {}),
#             "extreme_event": dict(extreme_event or {}),
#             "divergence_event": dict(divergence_event or {}),
#             "flip_event": dict(flip_event or {}),
#             "signal": dict(signal or {}),
#         },
#     }
#     return merge_payload(payload, overrides)
#
#
# # =============================================================================
# # Event fixtures
# # =============================================================================
#
#
# @pytest.fixture
# def event_bus_spy() -> SpyEventBus:
#     return SpyEventBus()
#
#
# @pytest.fixture
# def scheduler_spy() -> SpyScheduler:
#     return SpyScheduler()
#
#
# @pytest.fixture
# def parquet_storage_spy() -> SpyParquetStorage:
#     return SpyParquetStorage()
#
#
# @pytest.fixture
# def failing_parquet_storage_spy() -> SpyParquetStorage:
#     return SpyParquetStorage(raise_on_append=RuntimeError("parquet append failed intentionally"))
#
#
# @pytest.fixture
# def make_test_event() -> Callable[..., Event | SimpleNamespace]:
#     return make_event
#
#
# @pytest.fixture
# def make_regime_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.regime", regime_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_pressure_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.pressure", pressure_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_positive_extreme_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.extreme", positive_extreme_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_negative_extreme_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.extreme", negative_extreme_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_bullish_divergence_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.divergence", bullish_divergence_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_bearish_divergence_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.divergence", bearish_divergence_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_positive_to_negative_flip_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.flip", positive_to_negative_flip_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_negative_to_positive_flip_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.flip", negative_to_positive_flip_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_funding_signal_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.signal", funding_signal_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_funding_signal_envelope_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.signal", funding_signal_envelope_payload(**kwargs))
#
#     return factory
#
#
# @pytest.fixture
# def make_funding_updated_event() -> Callable[..., Event | SimpleNamespace]:
#     def factory(**kwargs: Any) -> Event | SimpleNamespace:
#         return make_event("analytics.funding.updated", funding_updated_payload(**kwargs))
#
#     return factory
#
#
# # =============================================================================
# # Config fixtures
# # =============================================================================
#
#
# @pytest.fixture
# def base_strategy_config() -> BaseFundingStrategyConfig:
#     return BaseFundingStrategyConfig(
#         setup_ttl_sec=120.0,
#         cooldown_sec=30.0,
#         event_stale_after_sec=300.0,
#         state_lock_timeout_sec=0.05,
#         default_market_type=DEFAULT_MARKET_TYPE,
#         default_timeframe=DEFAULT_TIMEFRAME,
#         enable_scheduler_cleanup=True,
#         cleanup_interval_sec=10.0,
#         cleanup_job_timeout_sec=2.0,
#         cleanup_job_name="pytest_funding_runtime.cleanup_expired_states",
#         strategy_namespace="strategy.funding.test",
#         source_name="pytest_funding_strategy",
#         service_name="pytest_funding_strategy",
#         enable_funding_updated_subscription=True,
#         enable_funding_signal_subscription=True,
#         recent_signals_maxlen=10,
#         signals_per_type_maxlen=5,
#         enable_generated_signal_parquet_history=False,
#         generated_signal_parquet_base_path="data/parquet",
#         generated_signal_parquet_dataset_name="pytest_strategy_funding_signals",
#         generated_signal_parquet_flush_interval_sec=10.0,
#         generated_signal_parquet_flush_timeout_sec=2.0,
#         generated_signal_parquet_flush_batch_size=2,
#         generated_signal_parquet_flush_job_name="pytest_funding_runtime.flush_generated_signals",
#     )
#
#
# @pytest.fixture
# def base_strategy_config_with_parquet(base_strategy_config: BaseFundingStrategyConfig) -> BaseFundingStrategyConfig:
#     base_strategy_config.enable_generated_signal_parquet_history = True
#     base_strategy_config.generated_signal_parquet_flush_batch_size = 2
#     return base_strategy_config
#
#
# @pytest.fixture
# def extreme_reversal_config() -> FundingExtremeReversalStrategyConfig:
#     return FundingExtremeReversalStrategyConfig(
#         setup_ttl_sec=120.0,
#         cooldown_sec=30.0,
#         event_stale_after_sec=300.0,
#         state_lock_timeout_sec=0.05,
#         default_market_type=DEFAULT_MARKET_TYPE,
#         default_timeframe=DEFAULT_TIMEFRAME,
#         cleanup_interval_sec=10.0,
#         cleanup_job_timeout_sec=2.0,
#         cleanup_job_name="funding_extreme_reversal.cleanup_expired_states",
#         recent_signals_maxlen=10,
#         signals_per_type_maxlen=5,
#         enable_generated_signal_parquet_history=False,
#         generated_signal_parquet_dataset_name="pytest_strategy_funding_extreme_reversal_signals",
#         generated_signal_parquet_flush_batch_size=2,
#         generated_signal_parquet_flush_job_name="funding_extreme_reversal.flush_generated_signals",
#         min_extreme_severity=0.60,
#         min_pressure_score=0.55,
#         min_regime_confidence=0.15,
#         min_mean_reversion_probability=0.50,
#         min_squeeze_probability=0.50,
#         min_divergence_confidence=0.45,
#         min_signal_confidence=0.45,
#         min_signal_abs_score=0.35,
#         require_reversal_risk=True,
#         require_squeeze_risk_or_reversion_probability=True,
#         require_high_pressure_level=True,
#     )
#
#
# @pytest.fixture
# def divergence_config() -> FundingDivergenceStrategyConfig:
#     return FundingDivergenceStrategyConfig(
#         setup_ttl_sec=120.0,
#         cooldown_sec=30.0,
#         event_stale_after_sec=300.0,
#         state_lock_timeout_sec=0.05,
#         default_market_type=DEFAULT_MARKET_TYPE,
#         default_timeframe=DEFAULT_TIMEFRAME,
#         cleanup_interval_sec=10.0,
#         cleanup_job_timeout_sec=2.0,
#         cleanup_job_name="funding_divergence.cleanup_expired_states",
#         recent_signals_maxlen=10,
#         signals_per_type_maxlen=5,
#         enable_generated_signal_parquet_history=False,
#         generated_signal_parquet_dataset_name="pytest_strategy_funding_divergence_signals",
#         generated_signal_parquet_flush_batch_size=2,
#         generated_signal_parquet_flush_job_name="funding_divergence.flush_generated_signals",
#         min_divergence_confidence=0.50,
#         min_pressure_score=0.35,
#         min_regime_confidence=0.10,
#         min_extreme_severity=0.45,
#         min_signal_confidence=0.45,
#         min_signal_abs_score=0.30,
#         require_non_neutral_regime=True,
#         require_pressure_alignment=False,
#         require_pressure_present=False,
#     )
#
#
# # =============================================================================
# # Strategy fixtures
# # =============================================================================
#
#
# @pytest.fixture
# def extreme_reversal_strategy(
#     event_bus_spy: SpyEventBus,
#     scheduler_spy: SpyScheduler,
#     parquet_storage_spy: SpyParquetStorage,
#     extreme_reversal_config: FundingExtremeReversalStrategyConfig,
# ) -> FundingExtremeReversalStrategy:
#     return FundingExtremeReversalStrategy(
#         event_bus=event_bus_spy,  # type: ignore[arg-type]
#         scheduler=scheduler_spy,  # type: ignore[arg-type]
#         parquet_storage=parquet_storage_spy,
#         config=extreme_reversal_config,
#     )
#
#
# @pytest.fixture
# def divergence_strategy(
#     event_bus_spy: SpyEventBus,
#     scheduler_spy: SpyScheduler,
#     parquet_storage_spy: SpyParquetStorage,
#     divergence_config: FundingDivergenceStrategyConfig,
# ) -> FundingDivergenceStrategy:
#     return FundingDivergenceStrategy(
#         event_bus=event_bus_spy,  # type: ignore[arg-type]
#         scheduler=scheduler_spy,  # type: ignore[arg-type]
#         parquet_storage=parquet_storage_spy,
#         config=divergence_config,
#     )
#
#
# # =============================================================================
# # Scenario fixtures
# # =============================================================================
#
#
# @pytest.fixture
# def crowded_longs_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.POSITIVE,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             confidence=0.86,
#             current_rate=0.0009,
#             mean_rate=0.0003,
#             zscore=2.0,
#             percentile=92.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.LONG,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             funding_rate=0.0009,
#             pressure_score=0.84,
#             squeeze_probability=0.74,
#             mean_reversion_probability=0.70,
#         ),
#         "extreme": positive_extreme_payload(
#             severity=0.91,
#             extreme_type=FundingExtremeType.ZSCORE_HIGH,
#             is_reversal_risk=True,
#             is_squeeze_risk=True,
#         ),
#     }
#
#
# @pytest.fixture
# def crowded_shorts_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.NEGATIVE,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             confidence=0.86,
#             current_rate=-0.0009,
#             mean_rate=-0.0003,
#             zscore=-2.0,
#             percentile=8.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.SHORT,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             funding_rate=-0.0009,
#             pressure_score=0.84,
#             squeeze_probability=0.74,
#             mean_reversion_probability=0.70,
#         ),
#         "extreme": negative_extreme_payload(
#             severity=0.91,
#             extreme_type=FundingExtremeType.ZSCORE_LOW,
#             is_reversal_risk=True,
#             is_squeeze_risk=True,
#         ),
#     }
#
#
# @pytest.fixture
# def bullish_divergence_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.NEGATIVE,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             confidence=0.82,
#             current_rate=-0.0007,
#             mean_rate=-0.0002,
#             zscore=-2.1,
#             percentile=8.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.SHORT,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             funding_rate=-0.0008,
#             pressure_score=0.72,
#             squeeze_probability=0.62,
#             mean_reversion_probability=0.64,
#         ),
#         "divergence": bullish_divergence_payload(
#             confidence=0.84,
#             divergence_type=FundingDivergenceType.PRICE_UP_FUNDING_DOWN,
#         ),
#     }
#
#
# @pytest.fixture
# def bearish_divergence_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.POSITIVE,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             confidence=0.82,
#             current_rate=0.0007,
#             mean_rate=0.0002,
#             zscore=2.1,
#             percentile=92.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.LONG,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             funding_rate=0.0008,
#             pressure_score=0.72,
#             squeeze_probability=0.62,
#             mean_reversion_probability=0.64,
#         ),
#         "divergence": bearish_divergence_payload(
#             confidence=0.84,
#             divergence_type=FundingDivergenceType.PRICE_DOWN_FUNDING_UP,
#         ),
#     }
#
#
# @pytest.fixture
# def bullish_liquidation_divergence_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.NEGATIVE,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             confidence=0.82,
#             current_rate=-0.0007,
#             mean_rate=-0.0002,
#             zscore=-2.1,
#             percentile=8.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.SHORT,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_SHORTS,
#             funding_rate=-0.0008,
#             pressure_score=0.72,
#             squeeze_probability=0.62,
#             mean_reversion_probability=0.64,
#         ),
#         "divergence": bullish_divergence_payload(
#             confidence=0.84,
#             divergence_type=FundingDivergenceType.LIQUIDATIONS_SHORTS_WITH_NEGATIVE_FUNDING,
#             short_liquidations=150_000.0,
#         ),
#     }
#
#
# @pytest.fixture
# def bearish_liquidation_divergence_context() -> dict[str, dict[str, Any]]:
#     return {
#         "regime": regime_payload(
#             regime=FundingRegime.POSITIVE,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             confidence=0.82,
#             current_rate=0.0007,
#             mean_rate=0.0002,
#             zscore=2.1,
#             percentile=92.0,
#         ),
#         "pressure": pressure_payload(
#             direction=FundingPressureDirection.LONG,
#             level=FundingPressureLevel.HIGH,
#             bias=FundingBias.OVERCROWDED_LONGS,
#             funding_rate=0.0008,
#             pressure_score=0.72,
#             squeeze_probability=0.62,
#             mean_reversion_probability=0.64,
#         ),
#         "divergence": bearish_divergence_payload(
#             confidence=0.84,
#             divergence_type=FundingDivergenceType.LIQUIDATIONS_LONGS_WITH_POSITIVE_FUNDING,
#             long_liquidations=150_000.0,
#         ),
#     }
#
#
# @pytest.fixture
# def scoped_context_factory() -> Callable[..., dict[str, dict[str, Any]]]:
#     def factory(
#         *,
#         symbol: str = DEFAULT_SYMBOL,
#         exchange: str = DEFAULT_EXCHANGE,
#         market_type: str = DEFAULT_MARKET_TYPE,
#         timeframe: Any = DEFAULT_TIMEFRAME,
#         exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#         direction: FundingStrategyDirection = FundingStrategyDirection.LONG,
#     ) -> dict[str, dict[str, Any]]:
#         if direction == FundingStrategyDirection.LONG:
#             return {
#                 "regime": regime_payload(
#                     symbol=symbol,
#                     exchange=exchange,
#                     market_type=market_type,
#                     timeframe=timeframe,
#                     exchange_symbol=exchange_symbol,
#                     regime=FundingRegime.NEGATIVE,
#                     bias=FundingBias.OVERCROWDED_SHORTS,
#                     confidence=0.82,
#                     current_rate=-0.0007,
#                     mean_rate=-0.0002,
#                     zscore=-2.1,
#                     percentile=8.0,
#                 ),
#                 "pressure": pressure_payload(
#                     symbol=symbol,
#                     exchange=exchange,
#                     market_type=market_type,
#                     timeframe=timeframe,
#                     exchange_symbol=exchange_symbol,
#                     direction=FundingPressureDirection.SHORT,
#                     level=FundingPressureLevel.HIGH,
#                     bias=FundingBias.OVERCROWDED_SHORTS,
#                     funding_rate=-0.0008,
#                     pressure_score=0.72,
#                     squeeze_probability=0.62,
#                     mean_reversion_probability=0.64,
#                 ),
#                 "divergence": bullish_divergence_payload(
#                     symbol=symbol,
#                     exchange=exchange,
#                     market_type=market_type,
#                     timeframe=timeframe,
#                     exchange_symbol=exchange_symbol,
#                     confidence=0.84,
#                 ),
#                 "extreme": negative_extreme_payload(
#                     symbol=symbol,
#                     exchange=exchange,
#                     market_type=market_type,
#                     timeframe=timeframe,
#                     exchange_symbol=exchange_symbol,
#                     severity=0.91,
#                 ),
#             }
#
#         return {
#             "regime": regime_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 regime=FundingRegime.POSITIVE,
#                 bias=FundingBias.OVERCROWDED_LONGS,
#                 confidence=0.82,
#                 current_rate=0.0007,
#                 mean_rate=0.0002,
#                 zscore=2.1,
#                 percentile=92.0,
#             ),
#             "pressure": pressure_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 direction=FundingPressureDirection.LONG,
#                 level=FundingPressureLevel.HIGH,
#                 bias=FundingBias.OVERCROWDED_LONGS,
#                 funding_rate=0.0008,
#                 pressure_score=0.72,
#                 squeeze_probability=0.62,
#                 mean_reversion_probability=0.64,
#             ),
#             "divergence": bearish_divergence_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 confidence=0.84,
#             ),
#             "extreme": positive_extreme_payload(
#                 symbol=symbol,
#                 exchange=exchange,
#                 market_type=market_type,
#                 timeframe=timeframe,
#                 exchange_symbol=exchange_symbol,
#                 severity=0.91,
#             ),
#         }
#
#     return factory
#
#
# # =============================================================================
# # Assertion fixtures
# # =============================================================================
#
#
# @pytest.fixture
# def assert_last_event() -> Callable[..., EmittedEventRecord]:
#     def assertion(
#         event_bus: SpyEventBus,
#         *,
#         topic: str | None = None,
#         event_kind: str | None = None,
#         symbol: str = DEFAULT_SYMBOL,
#         exchange: str = DEFAULT_EXCHANGE,
#         market_type: str = DEFAULT_MARKET_TYPE,
#         timeframe: Any = DEFAULT_TIMEFRAME,
#         exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#         direction: FundingStrategyDirection | None = None,
#         strategy: str | None = None,
#         strategy_namespace: str | None = None,
#     ) -> EmittedEventRecord:
#         assert event_bus.emitted, "Expected at least one emitted event"
#
#         record = event_bus.emitted[-1]
#         if topic is not None:
#             assert record.topic == topic
#
#         payload = record.payload
#         assert_scope_contract(
#             record,
#             symbol=symbol,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#         )
#
#         if event_kind is not None:
#             assert payload["event_kind"] == event_kind
#
#         if direction is not None:
#             assert payload["direction"] == direction.value
#
#         if strategy is not None:
#             assert payload["strategy"] == strategy
#             assert payload["strategy_name"] == strategy
#
#         if strategy_namespace is not None:
#             assert payload["strategy_namespace"] == strategy_namespace
#
#         return record
#
#     return assertion
#
#
# @pytest.fixture
# def assert_state_status() -> Callable[..., FundingStrategyState]:
#     def assertion(
#         strategy: Any,
#         *,
#         status: FundingSetupStatus,
#         symbol: str = DEFAULT_SYMBOL,
#         exchange: str = DEFAULT_EXCHANGE,
#         market_type: str = DEFAULT_MARKET_TYPE,
#         timeframe: Any = DEFAULT_TIMEFRAME,
#         exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#         direction: FundingStrategyDirection | None = None,
#         reason: str | None = None,
#     ) -> FundingStrategyState:
#         state = strategy.get_state(
#             symbol,
#             exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#         )
#         assert_state_scope(
#             state,
#             symbol=symbol,
#             exchange=exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#         )
#         assert state.status == status
#
#         if direction is not None:
#             assert state.direction == direction
#
#         if reason is not None:
#             assert state.reason == reason
#
#         return state
#
#     return assertion
#
#
# @pytest.fixture
# def state_for_scope() -> Callable[..., FundingStrategyState]:
#     def getter(
#         strategy: Any,
#         *,
#         symbol: str = DEFAULT_SYMBOL,
#         exchange: str = DEFAULT_EXCHANGE,
#         market_type: str = DEFAULT_MARKET_TYPE,
#         timeframe: Any = DEFAULT_TIMEFRAME,
#         exchange_symbol: str | None = DEFAULT_EXCHANGE_SYMBOL,
#     ) -> FundingStrategyState:
#         return strategy.get_state(
#             symbol,
#             exchange,
#             market_type=market_type,
#             timeframe=timeframe,
#             exchange_symbol=exchange_symbol,
#         )
#
#     return getter
#
#
# @pytest.fixture
# def make_scope_fixture() -> Callable[..., FundingStrategyScope]:
#     return make_scope
#
#
# @pytest.fixture
# def scope_dict_fixture() -> Callable[..., dict[str, str]]:
#     return scope_dict
#
#
# @pytest.fixture
# def assert_scope_contract_fixture() -> Callable[..., None]:
#     return assert_scope_contract
#
#
# @pytest.fixture
# def assert_state_scope_fixture() -> Callable[..., None]:
#     return assert_state_scope
#
#
# @pytest.fixture
# def json_loads_field() -> Callable[[Mapping[str, Any], str], Any]:
#     def loader(record: Mapping[str, Any], field_name: str) -> Any:
#         value = record.get(field_name)
#         assert isinstance(value, str), f"{field_name} must be JSON string"
#         return json.loads(value)
#
#     return loader