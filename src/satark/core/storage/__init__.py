"""In-memory and file-backed event storage abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from satark.core.events import Event


class EventStore(ABC):
    """Abstract store for normalized events."""

    @abstractmethod
    def put(self, events: Sequence[Event]) -> None:
        """Persist events."""

    @abstractmethod
    def get(self, event_id: UUID) -> Event | None:
        """Fetch a single event by id."""

    @abstractmethod
    def list_events(self, *, limit: int | None = None) -> list[Event]:
        """List stored events, optionally capped."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all events."""


class InMemoryEventStore(EventStore):
    """Simple dict-backed store suitable for tests and research workflows."""

    def __init__(self) -> None:
        self._events: dict[UUID, Event] = {}

    def put(self, events: Sequence[Event]) -> None:
        for event in events:
            self._events[event.id] = event

    def get(self, event_id: UUID) -> Event | None:
        return self._events.get(event_id)

    def list_events(self, *, limit: int | None = None) -> list[Event]:
        values = list(self._events.values())
        values.sort(key=lambda e: e.timestamp)
        if limit is not None:
            return values[:limit]
        return values

    def clear(self) -> None:
        self._events.clear()

    def extend(self, events: Iterable[Event]) -> None:
        """Alias for :meth:`put` accepting any iterable."""
        self.put(list(events))


class JsonlEventStore(EventStore):
    """Append-oriented JSONL file store for research and lightweight persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: InMemoryEventStore = InMemoryEventStore()
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        events: list[Event] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(Event.model_validate_json(line))
        self._cache.put(events)

    def put(self, events: Sequence[Event]) -> None:
        self._cache.put(events)
        with self.path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(event.model_dump_json() + "\n")

    def get(self, event_id: UUID) -> Event | None:
        return self._cache.get(event_id)

    def list_events(self, *, limit: int | None = None) -> list[Event]:
        return self._cache.list_events(limit=limit)

    def clear(self) -> None:
        self._cache.clear()
        if self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return all events as plain dictionaries."""
        return [e.model_dump(mode="json") for e in self.list_events()]
