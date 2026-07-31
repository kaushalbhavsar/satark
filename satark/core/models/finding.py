"""Aggregated finding combining detections, scores, and explanations."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from satark.core.models.detection import Detection
from satark.core.models.score import ScoreBreakdown


class Finding(BaseModel):
    """End-to-end analysis output for one or more related events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    detection: Detection
    score: ScoreBreakdown
    explanation: str
    recommendations: list[str] = Field(default_factory=list)
    ai_assisted: bool = Field(
        default=False,
        description="True when AI enriched explanation; detections remain reproducible without AI",
    )
