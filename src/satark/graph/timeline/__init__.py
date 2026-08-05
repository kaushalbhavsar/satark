"""Timeline construction from events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from satark.core.events import Event


@dataclass(frozen=True)
class TimelineEntry:
    """A single ordered timeline entry."""

    timestamp: datetime
    event_id: str
    summary: str
    actor: str | None
    category: str


def build_timeline(events: Sequence[Event]) -> list[TimelineEntry]:
    """Build a chronologically ordered timeline from events."""
    ordered = sorted(events, key=lambda e: e.timestamp)
    return [
        TimelineEntry(
            timestamp=e.timestamp,
            event_id=str(e.id),
            summary=e.action or e.category.value,
            actor=e.actor,
            category=e.category.value,
        )
        for e in ordered
    ]
