"""Evidence attached to detections and scores."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceKind(StrEnum):
    """Kinds of supporting evidence for a finding."""

    EVENT = "event"
    STATISTIC = "statistic"
    RULE_MATCH = "rule_match"
    BEHAVIORAL = "behavioral"
    CONTEXT = "context"
    ARTIFACT = "artifact"


class Evidence(BaseModel):
    """A single piece of evidence supporting a detection or score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvidenceKind
    summary: str
    source_event_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
