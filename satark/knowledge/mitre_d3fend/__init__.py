"""MITRE D3FEND defensive technique provider (seed)."""

from __future__ import annotations

from satark.core.models.knowledge_ref import KnowledgeSource
from satark.knowledge import KnowledgeEntry, StaticKnowledgeProvider

_SEED: list[KnowledgeEntry] = [
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_D3FEND,
        identifier="D3-SFA",
        name="System File Analysis",
        description="Analyzing file attributes and contents for malicious indicators.",
        url="https://d3fend.mitre.org/technique/d3f:SystemFileAnalysis/",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_D3FEND,
        identifier="D3-UA",
        name="User Behavior Analysis",
        description="Detect anomalous user activity patterns.",
    ),
]


def default_d3fend_provider(*, version: str = "0.12") -> StaticKnowledgeProvider:
    """Return a seeded MITRE D3FEND provider."""
    return StaticKnowledgeProvider(KnowledgeSource.MITRE_D3FEND, _SEED, version=version)
