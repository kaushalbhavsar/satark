"""Transparent risk scoring utilities."""

from __future__ import annotations

from collections.abc import Sequence

from satark.core.models.evidence import Evidence
from satark.core.models.knowledge_ref import KnowledgeReference
from satark.core.models.score import ScoreBreakdown, ScoreFactor


def clamp01(value: float) -> float:
    """Clamp a float into the inclusive [0, 1] range."""
    return max(0.0, min(1.0, value))


def aggregate_score(
    factors: Sequence[ScoreFactor],
    *,
    confidence: float,
    reasoning: str,
    evidence: Sequence[Evidence] | None = None,
    references: Sequence[KnowledgeReference] | None = None,
    baseline: float = 0.0,
) -> ScoreBreakdown:
    """Aggregate signed factor contributions into an explainable score.

    Positive contributions increase risk; negative contributions reduce it.
    The result is always clamped to [0, 1].
    """
    total = baseline + sum(f.contribution for f in factors)
    all_evidence: list[Evidence] = list(evidence or [])
    for factor in factors:
        all_evidence.extend(factor.evidence)
    return ScoreBreakdown(
        value=clamp01(total),
        confidence=clamp01(confidence),
        factors=list(factors),
        evidence=all_evidence,
        reasoning=reasoning,
        references=list(references or []),
    )
