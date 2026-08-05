# Events

All security data entering SATARK is normalized into the canonical `Event` model before analysis.

## Principles

- Plugins quarantine vendor payloads in `Event.raw`.
- The engine and scoring layers never depend on vendor schemas.
- Events are immutable Pydantic models (`frozen=True`).

## Categories

| Category | Enum value |
|----------|------------|
| File access | `file_access` |
| File read / write | `file_read`, `file_write` |
| Login | `login` |
| Authentication | `authentication` |
| Process execution | `process_execution` |
| Network connection | `network_connection` |
| Email received | `email_received` |
| Web request | `web_request` |
| DNS query | `dns_query` |
| Git commit | `git_commit` |
| Cloud API call | `cloud_api_call` |
| USB insertion | `usb_insertion` |
| Custom | `custom` |

## Example

```python
from datetime import UTC, datetime
from satark.core.events import Event, EventCategory

event = Event(
    category=EventCategory.USB_INSERTION,
    source="endpoint.agent",
    actor="alice",
    host="workstation-1",
    timestamp=datetime.now(UTC),
    attributes={"count": 3, "device_id": "USB-42"},
    tags=["insider", "usb"],
)

updated = event.with_attribute("mount_point", "/media/usb")
```

## Fields

| Field | Description |
|-------|-------------|
| `id` | UUID (auto-generated) |
| `category` | `EventCategory` |
| `source` | Logical collector / source name |
| `timestamp` | Event time (timezone-aware preferred) |
| `actor` | User, service, or process identity |
| `target` | Resource acted upon |
| `host` | Host or endpoint |
| `action` | Verb describing what happened |
| `attributes` | Domain-specific structured details |
| `tags` | Free-form labels for correlation / rules |
| `raw` | Original vendor payload (quarantined) |

See also the [Events API](../api/events.md).
