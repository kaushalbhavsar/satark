"""Finding prioritization utilities."""

from __future__ import annotations

from collections.abc import Sequence

from satark.core.models.detection import DetectionSeverity
from satark.core.models.finding import Finding

_SEVERITY_RANK: dict[DetectionSeverity, int] = {
    DetectionSeverity.INFO: 0,
    DetectionSeverity.LOW: 1,
    DetectionSeverity.MEDIUM: 2,
    DetectionSeverity.HIGH: 3,
    DetectionSeverity.CRITICAL: 4,
}


def prioritize(findings: Sequence[Finding]) -> list[Finding]:
    """Sort findings by severity rank then risk score (descending)."""
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_RANK.get(f.detection.severity, 0),
            f.score.value,
            f.score.confidence,
        ),
        reverse=True,
    )


def priority_score(finding: Finding) -> float:
    """Compute a single priority metric combining severity and risk."""
    severity = _SEVERITY_RANK.get(finding.detection.severity, 0) / 4.0
    return 0.6 * finding.score.value + 0.3 * severity + 0.1 * finding.score.confidence
