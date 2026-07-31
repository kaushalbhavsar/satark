"""Detection models produced by plugins."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from satark.core.models.evidence import Evidence
from satark.core.models.knowledge_ref import KnowledgeReference


class DetectionSeverity(StrEnum):
    """Ordered severity levels for detections."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Detection(BaseModel):
    """A reproducible detection produced without requiring AI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    plugin: str
    rule_id: str | None = None
    title: str
    description: str
    severity: DetectionSeverity = DetectionSeverity.MEDIUM
    event_ids: list[UUID] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    knowledge: list[KnowledgeReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
