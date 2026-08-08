# Events

Everything in SATARK is normalized into a common **Event** before analysis.

Plugins must not expose vendor-specific formats directly to the engine. Original payloads may be retained in `Event.raw`, but scoring logic should use the normalized fields and attributes.

## Implemented model

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

## Conceptual example

The following shape is a **conceptual** illustration of a domain-agnostic event. It is useful for discussion and design notes; field names and constructors may differ from the implemented API above.

```python
# Conceptual example — not a guaranteed public constructor
Event(
    source="github",
    actor="alice",
    action="clone_repository",
    target="customer-data",
    timestamp="2026-08-05T10:00:00Z",
)
```

!!! note
    Prefer the implemented `satark.core.events.Event` API in application code. Use conceptual examples only for architecture discussion.

## Categories

Examples of categories the framework recognizes include file access, login, process execution, network connection, email, web request, DNS, git commit, cloud API call, USB insertion, and authentication.
