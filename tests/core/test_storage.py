"""Tests for storage backends."""

from pathlib import Path

from satark.core.events import Event, EventCategory
from satark.core.storage import InMemoryEventStore, JsonlEventStore


def test_in_memory_store() -> None:
    store = InMemoryEventStore()
    event = Event(category=EventCategory.DNS_QUERY, source="dns")
    store.put([event])
    assert store.get(event.id) == event
    assert store.list_events() == [event]
    store.clear()
    assert store.list_events() == []


def test_jsonl_store(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = JsonlEventStore(path)
    event = Event(category=EventCategory.WEB_REQUEST, source="proxy", actor="u1")
    store.put([event])
    reloaded = JsonlEventStore(path)
    assert reloaded.get(event.id) is not None
    assert reloaded.as_dicts()[0]["actor"] == "u1"
