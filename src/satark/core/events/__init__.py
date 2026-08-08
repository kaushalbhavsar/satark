"""Normalized security event model.

Purpose
-------
All security data entering SATARK is normalized into :class:`Event` before
analysis. Plugins must never pass vendor-specific formats to the engine.

Architecture
------------
Events are immutable Pydantic models with a stable schema. Domain-specific
details live in ``attributes`` and ``raw`` (quarantined, never used for
scoring directly by the engine).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventCategory(StrEnum):
    """High-level event categories spanning security domains."""

    FILE_ACCESS = "file_access"
    LOGIN = "login"
    PROCESS_EXECUTION = "process_execution"
    NETWORK_CONNECTION = "network_connection"
    EMAIL_RECEIVED = "email_received"
    WEB_REQUEST = "web_request"
    DNS_QUERY = "dns_query"
    GIT_COMMIT = "git_commit"
    CLOUD_API_CALL = "cloud_api_call"
    USB_INSERTION = "usb_insertion"
    AUTHENTICATION = "authentication"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    CUSTOM = "custom"


class Event(BaseModel):
    """Canonical security event consumed by the SATARK engine.

    Examples
    --------
    >>> from datetime import datetime, UTC
    >>> Event(
    ...     category=EventCategory.USB_INSERTION,
    ...     source="endpoint.agent",
    ...     actor="alice",
    ...     timestamp=datetime.now(UTC),
    ...     attributes={"device_id": "USB-42"},
    ... )
    Event(...)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    category: EventCategory
    source: str = Field(description="Logical source or collector name")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str | None = Field(default=None, description="User, service, or process identity")
    target: str | None = Field(default=None, description="Resource acted upon")
    host: str | None = None
    action: str | None = Field(default=None, description="Verb describing what happened")
    attributes: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = Field(
        default=None,
        description="Original vendor payload quarantined from scoring logic",
    )

    def with_attribute(self, key: str, value: Any) -> Event:
        """Return a copy with an additional attribute (immutability-friendly)."""
        merged = {**self.attributes, key: value}
        return self.model_copy(update={"attributes": merged})
