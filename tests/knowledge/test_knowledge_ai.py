"""Tests for knowledge providers and AI assistants."""

from satark.ai import InvestigationAgent, NullLLM, enrich_explanation
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.finding import Finding
from satark.core.models.score import ScoreBreakdown
from satark.knowledge.mitre_attack import default_attack_provider


def test_attack_provider_seed() -> None:
    provider = default_attack_provider()
    entry = provider.get("T1091")
    assert entry is not None
    assert entry.name.startswith("Replication")
    refs = provider.search("phishing")
    assert any(r.identifier == "T1566" for r in refs)


def test_ai_disabled_does_not_mutate_truth() -> None:
    finding = Finding(
        detection=Detection(
            plugin="t",
            title="t",
            description="d",
            severity=DetectionSeverity.LOW,
        ),
        score=ScoreBreakdown(value=0.2, confidence=0.5, reasoning="r"),
        explanation="original",
    )
    agent = InvestigationAgent(client=NullLLM(), enabled=False)
    enriched = enrich_explanation(finding, agent)
    assert enriched.explanation == "original"
    assert enriched.ai_assisted is False
