# Events

Every record entering SATARK becomes an immutable Pydantic `Event` before
analysis. This lets a plugin work with a stable schema even when data comes
from different endpoint agents, cloud providers, or exports.

Plugins should not pass vendor-specific dictionaries to the engine. A copy of
the original record may be retained in `Event.raw`, but reusable detections and
scores should use the normalized fields and `attributes`.

## Fields

| Field | Meaning |
| --- | --- |
| `id` | Generated UUID used to connect events, evidence, and detections |
| `category` | Controlled high-level event type |
| `source` | Logical collector or source, such as `endpoint.agent` |
| `timestamp` | Time of the activity; defaults to the current UTC time when omitted |
| `actor` | User, service, process, or other actor identifier |
| `target` | Resource affected by the activity |
| `host` | Endpoint or workload identifier |
| `action` | Optional verb such as `file_read` or `usb_activity` |
| `attributes` | Domain-specific typed details used by a plugin |
| `tags` | Short labels used for routing or heuristic matching |
| `raw` | Original payload retained for investigation |

## Constructing an event

The repository implements a typed Pydantic `Event` in `satark.core.events` with fields such as `category`, `source`, `actor`, `target`, `host`, `action`, `timestamp`, `attributes`, `tags`, and `raw`.

```python
from datetime import UTC, datetime
from satark.core.events import Event, EventCategory

event = Event(
    category=EventCategory.USB_INSERTION,
    source="endpoint.agent",
    actor="alice",
    host="workstation-1",
    action="usb_activity",
    timestamp=datetime.now(UTC),
    attributes={"count": 3},
    tags=["insider", "usb"],
)
```

Events are frozen. To add or replace an attribute, create a copy:

```python
updated = event.with_attribute("device_id", "USB-42")
```

## Categories

`EventCategory` currently includes `file_access`, `file_read`, `file_write`,
`login`, `authentication`, `process_execution`, `network_connection`,
`email_received`, `web_request`, `dns_query`, `git_commit`, `cloud_api_call`,
`usb_insertion`, and `custom`. Use `custom` only when none of the existing
categories accurately describe the activity; document the meaning in your
plugin.

## Normalization rules

Normalize once at the boundary. Parse timestamps into `datetime` values,
preserve a stable actor identifier, and put vendor-only names in `attributes`.
Do not use a tag to replace a category: `category` expresses the event class;
tags add context such as `phishing` or `ransomware`.
