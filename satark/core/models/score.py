"""Transparent risk score models.

SATARK never outputs only a numerical score. Every score includes contributing
factors, evidence, confidence, reasoning, and references.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from satark.core.models.evidence import Evidence
from satark.core.models.knowledge_ref import KnowledgeReference


class ScoreFactor(BaseModel):
    """A named contribution to an overall risk score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    contribution: float = Field(ge=-1.0, le=1.0, description="Signed contribution toward risk")
    description: str
    evidence: list[Evidence] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """Explainable risk score with full transparency.

    Answers: "Why was this event classified as malicious?"
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(ge=0.0, le=1.0, description="Overall risk in [0, 1]")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the score")
    factors: list[ScoreFactor] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str
    references: list[KnowledgeReference] = Field(default_factory=list)

    @property
    def is_elevated(self) -> bool:
        """Return True when risk exceeds a conventional elevated threshold."""
        return self.value >= 0.7
