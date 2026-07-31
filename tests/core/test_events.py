"""Unit tests for the canonical Event model."""

from datetime import UTC, datetime

from satark.core.events import Event, EventCategory


def test_event_defaults_and_immutability() -> None:
    event = Event(
        category=EventCategory.USB_INSERTION,
        source="endpoint",
        actor="alice",
        attributes={"count": 3},
    )
    assert event.category is EventCategory.USB_INSERTION
    assert event.actor == "alice"
    updated = event.with_attribute("device", "USB-1")
    assert "device" not in event.attributes
    assert updated.attributes["device"] == "USB-1"
    assert updated.id == event.id


def test_event_timestamp_utc() -> None:
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    event = Event(category=EventCategory.LOGIN, source="idp", timestamp=ts)
    assert event.timestamp == ts
