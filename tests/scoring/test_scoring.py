"""Tests for scoring and explainability."""

from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.finding import Finding
from satark.core.models.score import ScoreBreakdown, ScoreFactor
from satark.scoring import (
    aggregate_score,
    evidence_confidence,
    prioritize,
    why_malicious,
)


def test_aggregate_score_clamps_and_explains() -> None:
    factors = [
        ScoreFactor(name="a", contribution=0.8, description="high"),
        ScoreFactor(name="b", contribution=0.5, description="also high"),
    ]
    score = aggregate_score(factors, confidence=1.5, reasoning="test", baseline=0.2)
    assert score.value == 1.0
    assert score.confidence == 1.0
    assert len(score.factors) == 2


def test_evidence_confidence() -> None:
    evidence = [
        Evidence(kind=EvidenceKind.EVENT, summary="e1", weight=1.0),
        Evidence(kind=EvidenceKind.STATISTIC, summary="e2", weight=0.5),
    ]
    conf = evidence_confidence(evidence)
    assert 0.0 < conf <= 1.0
    assert evidence_confidence([]) == 0.0


def test_why_malicious_and_prioritize() -> None:
    detection = Detection(
        plugin="test",
        title="Test detection",
        description="desc",
        severity=DetectionSeverity.HIGH,
    )
    score = ScoreBreakdown(
        value=0.9,
        confidence=0.8,
        factors=[
            ScoreFactor(name="volume", contribution=0.4, description="spike"),
            ScoreFactor(name="severity", contribution=0.3, description="high"),
        ],
        reasoning="Because of a spike.",
    )
    finding = Finding(detection=detection, score=score, explanation="x")
    answer = why_malicious(detection, score)
    assert "0.90" in answer or "0.9" in answer
    assert "volume" in answer
    ordered = prioritize([finding])
    assert ordered[0] is finding
