"""Tests for the entity graph and timeline."""

from satark.core.events import Event, EventCategory
from satark.graph import EntityGraph, build_timeline, find_attack_paths


def test_entity_graph_ingest_and_path() -> None:
    graph = EntityGraph()
    event = Event(
        category=EventCategory.USB_INSERTION,
        source="endpoint",
        actor="alice",
        host="ws1",
        target="USB-9",
    )
    graph.ingest_event(event)
    assert graph.node_count() >= 4
    path = graph.shortest_path("actor:alice", f"event:{event.id}")
    assert path
    paths = find_attack_paths(graph, "actor:alice", "target:USB-9")
    assert paths
    assert "→" in paths[0].summary


def test_timeline_orders_events() -> None:
    from datetime import UTC, datetime, timedelta

    base = datetime(2024, 1, 1, tzinfo=UTC)
    events = [
        Event(
            category=EventCategory.LOGIN,
            source="idp",
            timestamp=base + timedelta(hours=2),
            action="login",
        ),
        Event(
            category=EventCategory.USB_INSERTION, source="endpoint", timestamp=base, action="usb"
        ),
    ]
    timeline = build_timeline(events)
    assert timeline[0].summary == "usb"
    assert timeline[1].summary == "login"
