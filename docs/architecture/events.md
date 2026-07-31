# Events

All security data must be normalized into the canonical `Event` model before analysis.

## Categories

File access, login, process execution, network connection, email, web request, DNS, git commit, cloud API call, USB insertion, authentication, and custom.

## Principles

- Plugins quarantine vendor payloads in `Event.raw`.
- The engine and scoring layers never depend on vendor schemas.
- Events are immutable Pydantic models.

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
```
