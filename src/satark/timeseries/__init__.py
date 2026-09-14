"""Build fixed-interval activity time series from canonical SATARK events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal
from uuid import UUID

from satark.core.events import Event, EventCategory

GroupBy = Literal["actor", "host", "source"]


@dataclass(frozen=True)
class TimeSeriesPoint:
    """Aggregated activity for one group during one fixed time bucket."""

    bucket_start: datetime
    group: str
    values: dict[EventCategory, float]
    event_ids: tuple[UUID, ...]

    @property
    def total(self) -> float:
        """Total activity represented by this bucket."""
        return sum(self.values.values())


def build_time_series(
    events: Sequence[Event],
    *,
    interval: timedelta = timedelta(hours=1),
    group_by: GroupBy = "actor",
    categories: Sequence[EventCategory] | None = None,
    fill_gaps: bool = True,
) -> dict[str, list[TimeSeriesPoint]]:
    """Aggregate event counts into chronological, optionally gap-filled buckets.

    A value uses ``event.attributes['count']`` when present, otherwise one.
    Missing buckets between a group's first and last observation are emitted with
    zero values and no event IDs, which is useful for baseline and ML workflows.
    """
    seconds = interval.total_seconds()
    if seconds <= 0:
        raise ValueError("interval must be positive")
    selected = tuple(categories or tuple(EventCategory))
    grouped: dict[str, dict[datetime, list[Event]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.category not in selected:
            continue
        group = _group(event, group_by)
        grouped[group][_bucket_start(event.timestamp, seconds)].append(event)

    result: dict[str, list[TimeSeriesPoint]] = {}
    for group, buckets in grouped.items():
        times = sorted(buckets)
        if not times:
            continue
        if fill_gaps:
            cursor = times[0]
            while cursor <= times[-1]:
                buckets.setdefault(cursor, [])
                cursor += interval
        result[group] = [
            _point(timestamp, group, buckets[timestamp], selected)
            for timestamp in sorted(buckets)
        ]
    return result


def _point(
    timestamp: datetime,
    group: str,
    events: Sequence[Event],
    categories: Sequence[EventCategory],
) -> TimeSeriesPoint:
    values = {category: 0.0 for category in categories}
    for event in events:
        count = float(event.attributes.get("count", 1))
        if not isfinite(count) or count < 0:
            raise ValueError("Activity counts must be finite and nonnegative")
        values[event.category] += count
    return TimeSeriesPoint(timestamp, group, values, tuple(event.id for event in events))


def _group(event: Event, group_by: GroupBy) -> str:
    value = getattr(event, group_by)
    return value or "unknown"


def _bucket_start(timestamp: datetime, seconds: float) -> datetime:
    utc_timestamp = timestamp.astimezone(UTC)
    floored = int(utc_timestamp.timestamp() // seconds * seconds)
    return datetime.fromtimestamp(floored, tz=UTC)
