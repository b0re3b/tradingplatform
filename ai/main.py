"""
Реальний entry point для AI/news пакету.

Куди покласти:
    trading_system/ai/main.py

Запуск із кореня проєкту:
    python -m ai.main

Або якщо пакет імпортується як trading_system:
    python -m trading_system.ai.main

Що робить:
    - бере реальні адреси сайтів з build_default_news_ai_config();
    - створює NewsAIService з EventBus і Scheduler;
    - запускає повний pipeline: collect -> deduplicate -> normalize -> features -> LLM fallback/rules -> score -> publish events;
    - друкує в консоль новини та результати оцінювання;
    - підтримує one-shot запуск і watch-режим через core Scheduler.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import signal
from dataclasses import replace
from typing import Any, Iterable

try:
    from core.event_bus import EventBus
    from core.scheduler import Scheduler
except ModuleNotFoundError:  # pragma: no cover
    from core.event_bus import EventBus
    from core.scheduler import Scheduler

try:
    from .config import NewsAIConfig, NewsSourceConfig, build_default_news_ai_config
    from .news_service import NewsAIService
except ImportError:  # pragma: no cover
    from ai.config import NewsAIConfig, NewsSourceConfig, build_default_news_ai_config
    from ai.news_service import NewsAIService


# --------------------------------------------------------------------------------------
# Small compatibility helpers for slightly different core implementations.
# --------------------------------------------------------------------------------------

async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def start_component(component: Any) -> None:
    start = getattr(component, "start", None)
    if callable(start):
        await maybe_await(start())


async def stop_component(component: Any) -> None:
    stop = getattr(component, "stop", None)
    if callable(stop):
        await maybe_await(stop())


async def subscribe(event_bus: EventBus, topic: str, handler: Any) -> None:
    await maybe_await(event_bus.subscribe(topic, handler))


def payload_from_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        payload = event.get("payload", event)
        return payload if isinstance(payload, dict) else {}

    payload = getattr(event, "payload", None)
    return payload if isinstance(payload, dict) else {}


def topic_from_event(event: Any, fallback: str = "news.event") -> str:
    if isinstance(event, dict):
        return str(event.get("topic", fallback))
    return str(getattr(event, "topic", fallback))


def make_scheduler(event_bus: EventBus) -> Scheduler:
    try:
        return Scheduler(event_bus=event_bus)
    except TypeError:
        return Scheduler()


# --------------------------------------------------------------------------------------
# Config assembly: real sources only.
# --------------------------------------------------------------------------------------


def _filter_sources(
    sources: Iterable[NewsSourceConfig],
    *,
    source_names: tuple[str, ...],
    limit_sources: int | None,
    max_items_per_source: int | None,
    request_timeout_seconds: float | None,
    disable_min_fetch_interval: bool,
) -> tuple[NewsSourceConfig, ...]:
    selected = tuple(sources)

    if source_names:
        wanted = {name.strip() for name in source_names if name.strip()}
        selected = tuple(source for source in selected if source.name in wanted)

    if limit_sources is not None and limit_sources > 0:
        selected = selected[:limit_sources]

    patched: list[NewsSourceConfig] = []
    for source in selected:
        updates: dict[str, Any] = {}

        if max_items_per_source is not None and max_items_per_source > 0:
            updates["max_items_per_fetch"] = min(
                source.max_items_per_fetch,
                max_items_per_source,
            )

        if request_timeout_seconds is not None and request_timeout_seconds > 0:
            updates["request_timeout_seconds"] = request_timeout_seconds

        if disable_min_fetch_interval:
            updates["min_fetch_interval_seconds"] = 0.0

        patched.append(replace(source, **updates) if updates else source)

    return tuple(patched)


def build_runtime_config(args: argparse.Namespace) -> NewsAIConfig:
    base = build_default_news_ai_config()

    sources = _filter_sources(
        base.source_configs,
        source_names=tuple(args.source or ()),
        limit_sources=args.limit_sources,
        max_items_per_source=args.max_items_per_source,
        request_timeout_seconds=args.timeout,
        disable_min_fetch_interval=args.disable_min_fetch_interval,
    )

    if not sources:
        available = ", ".join(source.name for source in base.source_configs)
        raise SystemExit(
            "Не залишилось жодного джерела для запуску. "
            f"Перевір --source/--limit-sources. Доступні джерела: {available}"
        )

    return replace(
        base,
        collect_interval_seconds=args.interval,
        startup_collect_enabled=args.watch,
        publish_raw_fetched_event=args.print_raw,
        publish_scored_event=True,
        publish_high_impact_event=True,
        publish_failed_events=True,
        max_items_per_cycle=args.max_items_per_cycle,
        max_concurrent_sources=args.max_concurrent_sources,
        source_configs=sources,
        service_name=args.service_name,
        metadata={
            **base.metadata,
            "runner": "ai.news.main",
            "mode": "real_sources",
            "selected_source_count": len(sources),
        },
    )


# --------------------------------------------------------------------------------------
# Console rendering.
# --------------------------------------------------------------------------------------


def fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_list(values: Any) -> str:
    if not values:
        return "-"
    if isinstance(values, str):
        return values
    return ", ".join(str(value) for value in values if str(value).strip()) or "-"


def unique_keywords(features: dict[str, Any]) -> list[str]:
    fields = (
        "matched_urgent_keywords",
        "matched_regulatory_keywords",
        "matched_macro_keywords",
        "matched_hack_keywords",
        "matched_listing_keywords",
        "matched_delisting_keywords",
        "matched_positive_keywords",
        "matched_negative_keywords",
        "matched_high_impact_keywords",
        "matched_low_quality_keywords",
    )

    values: list[str] = []
    for field in fields:
        raw = features.get(field) or ()
        if isinstance(raw, str):
            raw = (raw,)
        values.extend(str(item).strip() for item in raw if str(item).strip())

    return list(dict.fromkeys(values))


def print_raw_payload(payload: dict[str, Any]) -> None:
    batch = payload.get("batch") or {}
    items = batch.get("items") or ()

    print("\n" + "." * 120)
    print(f"[RAW FETCHED] count={batch.get('count', len(items))}")
    print("." * 120)

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        print(f"{index:02d}. {item.get('title', 'Без заголовка')}")
        print(f"    source: {item.get('source_name')} | published: {item.get('published_at') or '-'}")
        print(f"    url   : {item.get('url') or '-'}")


def print_scored_payload(payload: dict[str, Any], *, high_impact: bool = False) -> None:
    item = payload.get("item") or {}
    features = payload.get("features") or {}
    score = payload.get("score") or {}
    llm_result = payload.get("llm_result") or {}

    title = item.get("title") or "Без заголовка"
    categories = score.get("categories") or item.get("categories") or ()
    keywords = unique_keywords(features)

    border = "!" if high_impact else "="
    label = "HIGH IMPACT" if high_impact else "SCORED"

    print("\n" + border * 120)
    print(f"[{label}] {title}")
    print("-" * 120)
    print(f"news_id        : {payload.get('news_id') or item.get('news_id') or '-'}")
    print(f"source         : {item.get('source_name') or '-'}")
    print(f"url            : {item.get('url') or '-'}")
    print(f"published      : {item.get('published_at') or '-'}")
    print(f"symbols        : {fmt_list(item.get('symbols'))}")
    print(f"categories     : {fmt_list(categories)}")
    print(f"sentiment      : {score.get('sentiment') or '-'} ({fmt_float(score.get('sentiment_score'))})")
    print(f"market_bias    : {score.get('market_bias') or '-'}")
    print(f"impact         : {score.get('impact_level') or '-'} ({fmt_float(score.get('impact_score'))})")
    print(f"urgency        : {score.get('urgency_level') or '-'} ({fmt_float(score.get('urgency_score'))})")
    print(f"relevance      : {score.get('relevance_level') or '-'} ({fmt_float(score.get('relevance_score'))})")
    print(f"confidence     : {fmt_float(score.get('confidence_score'))}")
    print(f"novelty        : {fmt_float(score.get('novelty_score'))}")
    print(f"source_rep     : {fmt_float(score.get('source_reputation_score'))}")
    print(f"time_horizon   : {score.get('time_horizon') or '-'}")
    print(f"alert_types    : {fmt_list(score.get('alert_types'))}")
    print(f"llm_status     : {score.get('llm_status') or llm_result.get('status') or '-'}")

    if keywords:
        print(f"keywords       : {', '.join(keywords)}")

    if score.get("summary"):
        print(f"summary        : {score['summary']}")
    if score.get("explanation"):
        print(f"explanation    : {score['explanation']}")
    if score.get("trading_notes"):
        print(f"trading_notes  : {score['trading_notes']}")

    if high_impact:
        alert = payload.get("alert") or {}
        if alert.get("message"):
            print(f"alert_message  : {alert['message']}")

    print(border * 120)


def print_failure(event: Any) -> None:
    payload = payload_from_event(event)
    print("\n" + "#" * 120)
    print(f"[FAILED] {topic_from_event(event)}")
    print(payload)
    print("#" * 120)


def print_sources(config: NewsAIConfig) -> None:
    print("\n" + "=" * 120)
    print("REAL NEWS SOURCES FROM CONFIG")
    print("=" * 120)
    for index, source in enumerate(config.source_configs, start=1):
        address = source.url or source.api_url or "-"
        print(
            f"{index:02d}. {source.name} | {source.source_type} | "
            f"items={source.max_items_per_fetch} | timeout={source.request_timeout_seconds}s"
        )
        print(f"    {address}")
    print("=" * 120)


def print_summary(result: Any, *, printed_scored: int, printed_high_impact: int, printed_failures: int) -> None:
    print("\n" + "=" * 120)
    print("RUN SUMMARY")
    print("=" * 120)
    print(f"duration_ms              : {getattr(result, 'duration_ms', 0.0):.2f}")
    print(f"collected_count          : {getattr(result, 'collected_count', 0)}")
    print(f"raw_unique_count         : {getattr(result, 'raw_unique_count', 0)}")
    print(f"processed_count          : {getattr(result, 'processed_count', 0)}")
    print(f"normalized_unique_count  : {getattr(result, 'normalized_unique_count', 0)}")
    print(f"scored_count             : {getattr(result, 'scored_count', 0)}")
    print(f"high_impact_count        : {getattr(result, 'high_impact_count', 0)}")
    print(f"failed_count             : {getattr(result, 'failed_count', 0)}")
    print(f"printed_scored_events    : {printed_scored}")
    print(f"printed_high_impact      : {printed_high_impact}")
    print(f"printed_failures         : {printed_failures}")

    errors = tuple(getattr(result, "errors", ()) or ())
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"- {error}")

    metadata = getattr(result, "metadata", {}) or {}
    collector_stats = metadata.get("collector_stats") or {}
    sources = collector_stats.get("sources") or ()
    if sources:
        print("\nSOURCE HEALTH:")
        for source in sources:
            print(
                f"- {source.get('source_name')}: "
                f"status={source.get('status')}, "
                f"last_fetch_status={source.get('last_fetch_status')}, "
                f"items={source.get('total_items_fetched')}, "
                f"error={source.get('last_error') or '-'}"
            )

    if getattr(result, "scored_count", 0) == 0:
        print(
            "\nЖодна новина не дійшла до фінального scoring. "
            "Найчастіші причини: джерела недоступні, RSS/HTML змінив формат, "
            "усі новини стали duplicate або text/title порожні після нормалізації."
        )


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass


# --------------------------------------------------------------------------------------
# Runtime.
# --------------------------------------------------------------------------------------


async def run_once(service: NewsAIService) -> Any:
    run_now = getattr(service, "run_now", None)
    if callable(run_now):
        return await maybe_await(run_now())
    return await maybe_await(service.collect_once())


async def run_watch(service: NewsAIService, scheduler: Scheduler, stop_event: asyncio.Event) -> None:
    service.register()
    await start_component(scheduler)
    print("\nWatch mode запущено через core Scheduler. Натисни Ctrl+C для зупинки.")
    await stop_event.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real AI/news package")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="запускати збір новин періодично через core Scheduler",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="інтервал збору новин у watch-режимі / config.collect_interval_seconds",
    )
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=0,
        help="скільки перших джерел із config використати; 0 = усі джерела",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="запустити тільки конкретне джерело за name; можна передавати кілька разів",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="показати реальні джерела з config і завершити роботу",
    )
    parser.add_argument(
        "--max-items-per-source",
        type=int,
        default=None,
        help="обмежити кількість item-ів на одне джерело для локального запуску",
    )
    parser.add_argument(
        "--max-items-per-cycle",
        type=int,
        default=200,
        help="глобальний ліміт item-ів на один цикл NewsAIService",
    )
    parser.add_argument(
        "--max-concurrent-sources",
        type=int,
        default=5,
        help="скільки джерел опитувати паралельно",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="перевизначити timeout для кожного джерела",
    )
    parser.add_argument(
        "--disable-min-fetch-interval",
        action="store_true",
        help="скинути min_fetch_interval_seconds до 0 для ручних локальних запусків",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="друкувати raw batch перед нормалізацією і scoring",
    )
    parser.add_argument(
        "--service-name",
        default="news_ai_service",
        help="source/service_name для EventBus подій",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="рівень логів у консолі",
    )
    return parser


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    if args.max_items_per_cycle <= 0:
        raise SystemExit("--max-items-per-cycle must be > 0")
    if args.max_concurrent_sources <= 0:
        raise SystemExit("--max-concurrent-sources must be > 0")

    config = build_runtime_config(args)
    print_sources(config)

    if args.list_sources:
        return

    event_bus = EventBus()
    scheduler = make_scheduler(event_bus)
    service = NewsAIService(
        event_bus=event_bus,
        scheduler=scheduler,
        config=config,
    )

    printed_scored: list[dict[str, Any]] = []
    printed_high_impact: list[dict[str, Any]] = []
    printed_failures: list[dict[str, Any]] = []

    async def on_raw_fetched(event: Any) -> None:
        print_raw_payload(payload_from_event(event))

    async def on_scored(event: Any) -> None:
        payload = payload_from_event(event)
        printed_scored.append(payload)
        print_scored_payload(payload)

    async def on_high_impact(event: Any) -> None:
        payload = payload_from_event(event)
        printed_high_impact.append(payload)
        print_scored_payload(payload, high_impact=True)

    async def on_failed(event: Any) -> None:
        payload = payload_from_event(event)
        printed_failures.append(payload)
        print_failure(event)

    if args.print_raw:
        await subscribe(event_bus, "news.raw_fetched", on_raw_fetched)

    await subscribe(event_bus, "news.scored", on_scored)
    await subscribe(event_bus, "news.high_impact", on_high_impact)
    await subscribe(event_bus, "news.scoring_failed", on_failed)
    await subscribe(event_bus, "news.pipeline_failed", on_failed)
    await subscribe(event_bus, "news.publish_failed", on_failed)

    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)

    try:
        await start_component(event_bus)

        if args.watch:
            await run_watch(service, scheduler, stop_event)
            return

        result = await run_once(service)

        # EventBus can dispatch through an internal queue, so give it a short drain window.
        await asyncio.sleep(0.25)

        print_summary(
            result,
            printed_scored=len(printed_scored),
            printed_high_impact=len(printed_high_impact),
            printed_failures=len(printed_failures),
        )

    finally:
        await stop_component(scheduler)
        await stop_component(event_bus)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()