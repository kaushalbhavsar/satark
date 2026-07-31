"""Explainability helpers for scored detections."""

from __future__ import annotations

from satark.core.models.detection import Detection
from satark.core.models.score import ScoreBreakdown


def format_explanation(detection: Detection, score: ScoreBreakdown) -> str:
    """Build a structured explanation answering why a detection is malicious."""
    lines = [
        f"Detection: {detection.title}",
        f"Severity: {detection.severity.value}",
        f"Risk: {score.value:.2f} (confidence {score.confidence:.2f})",
        f"Reasoning: {score.reasoning}",
    ]
    if score.factors:
        lines.append("Contributing factors:")
        for factor in score.factors:
            lines.append(f"  - {factor.name}: {factor.contribution:+.2f} — {factor.description}")
    if score.evidence:
        lines.append("Evidence:")
        for item in score.evidence:
            lines.append(f"  - [{item.kind.value}] {item.summary}")
    if score.references:
        lines.append("References:")
        for ref in score.references:
            label = ref.name or ref.identifier
            lines.append(f"  - {ref.source.value}:{ref.identifier} ({label})")
    return "\n".join(lines)


def why_malicious(detection: Detection, score: ScoreBreakdown) -> str:
    """Short answer to: Why was this event classified as malicious?"""
    top = sorted(score.factors, key=lambda f: abs(f.contribution), reverse=True)[:3]
    if not top:
        return score.reasoning
    parts = [f"{f.name} ({f.contribution:+.2f})" for f in top]
    return f"{detection.title} scored {score.value:.2f} due to: " + ", ".join(parts) + "."
