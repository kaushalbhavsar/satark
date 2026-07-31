"""AI-assisted explanation enrichment (optional)."""

from __future__ import annotations

from satark.ai.agents import InvestigationAgent
from satark.core.models.finding import Finding
from satark.scoring.explainability import format_explanation


def enrich_explanation(finding: Finding, agent: InvestigationAgent | None = None) -> Finding:
    """Optionally enrich a finding's explanation via AI without altering the detection."""
    if agent is None or not agent.enabled:
        return finding
    summary = agent.summarize_finding(finding)
    recommendations = agent.recommend(finding)
    return finding.model_copy(
        update={
            "explanation": summary,
            "recommendations": recommendations,
            "ai_assisted": True,
        }
    )


def deterministic_explanation(finding: Finding) -> str:
    """Always-available non-AI explanation."""
    return format_explanation(finding.detection, finding.score)
