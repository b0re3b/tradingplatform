from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable

from data.market_models import DirtyReason, MarketScope, now_ms


@dataclass(slots=True)
class DirtyItem:
    scope: MarketScope
    reasons: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    first_dirty_ms: int = field(default_factory=now_ms)
    last_dirty_ms: int = field(default_factory=now_ms)
    count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark(self, *, reason: str, source: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.reasons.add(str(reason))
        if source:
            self.sources.add(str(source))
        self.last_dirty_ms = now_ms()
        self.count += 1
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "reasons": sorted(self.reasons),
            "sources": sorted(self.sources),
            "first_dirty_ms": self.first_dirty_ms,
            "last_dirty_ms": self.last_dirty_ms,
            "count": self.count,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DirtyRegistryStats:
    dirty_items: int = 0
    total_marks: int = 0
    total_pops: int = 0
    total_clears: int = 0
    last_mark_ms: int | None = None
    last_pop_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dirty_items": self.dirty_items,
            "total_marks": self.total_marks,
            "total_pops": self.total_pops,
            "total_clears": self.total_clears,
            "last_mark_ms": self.last_mark_ms,
            "last_pop_ms": self.last_pop_ms,
        }


class DirtySymbolRegistry:
    """
    Async-safe dirty-scope registry for state-driven market analytics.

    The registry coalesces repeated raw updates into one dirty item per scope.
    Analytics schedulers can pop dirty scopes at a controlled cadence, preventing
    raw market data from creating EventBus or scheduler backlogs.
    """

    def __init__(self, *, service_name: str = "dirty_symbol_registry") -> None:
        self.service_name = service_name
        self._items: dict[str, DirtyItem] = {}
        self._lock = asyncio.Lock()
        self._stats = DirtyRegistryStats()

    async def mark_dirty(
        self,
        scope: MarketScope,
        *,
        reason: DirtyReason | str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        reason_value = reason.value if isinstance(reason, DirtyReason) else str(reason)
        async with self._lock:
            item = self._items.get(scope.key)
            if item is None:
                item = DirtyItem(scope=scope)
                self._items[scope.key] = item
            item.mark(reason=reason_value, source=source, metadata=metadata)
            self._stats.total_marks += 1
            self._stats.last_mark_ms = now_ms()
            self._stats.dirty_items = len(self._items)

    async def mark_many(
        self,
        scopes: Iterable[MarketScope],
        *,
        reason: DirtyReason | str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for scope in scopes:
            await self.mark_dirty(scope, reason=reason, source=source, metadata=metadata)

    async def pop_dirty(
        self,
        *,
        limit: int | None = None,
        reasons: set[str] | None = None,
        source: str | None = None,
    ) -> list[DirtyItem]:
        async with self._lock:
            keys: list[str] = []
            for key, item in self._items.items():
                if reasons and not (item.reasons & reasons):
                    continue
                if source and source not in item.sources:
                    continue
                keys.append(key)
                if limit is not None and len(keys) >= limit:
                    break

            items = [self._items.pop(key) for key in keys]
            self._stats.total_pops += len(items)
            self._stats.last_pop_ms = now_ms()
            self._stats.dirty_items = len(self._items)
            return items

    async def snapshot_dirty(
        self,
        *,
        limit: int | None = None,
        reasons: set[str] | None = None,
        source: str | None = None,
    ) -> list[DirtyItem]:
        async with self._lock:
            result: list[DirtyItem] = []
            for item in self._items.values():
                if reasons and not (item.reasons & reasons):
                    continue
                if source and source not in item.sources:
                    continue
                result.append(
                    DirtyItem(
                        scope=item.scope,
                        reasons=set(item.reasons),
                        sources=set(item.sources),
                        first_dirty_ms=item.first_dirty_ms,
                        last_dirty_ms=item.last_dirty_ms,
                        count=item.count,
                        metadata=dict(item.metadata),
                    )
                )
                if limit is not None and len(result) >= limit:
                    break
            return result

    async def clear(self, scope: MarketScope | None = None) -> int:
        async with self._lock:
            if scope is None:
                count = len(self._items)
                self._items.clear()
            else:
                count = 1 if self._items.pop(scope.key, None) is not None else 0
            self._stats.total_clears += count
            self._stats.dirty_items = len(self._items)
            return count

    async def size(self) -> int:
        async with self._lock:
            return len(self._items)

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            self._stats.dirty_items = len(self._items)
            return self._stats.to_dict()
