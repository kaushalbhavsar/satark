"""CAPEC attack pattern provider (seed)."""

from __future__ import annotations

from satark.core.models.knowledge_ref import KnowledgeSource
from satark.knowledge import KnowledgeEntry, StaticKnowledgeProvider

_SEED: list[KnowledgeEntry] = [
    KnowledgeEntry(
        source=KnowledgeSource.CAPEC,
        identifier="CAPEC-112",
        name="Brute Force",
        description="Adversary tries repeatedly to guess credentials.",
        url="https://capec.mitre.org/data/definitions/112.html",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.CAPEC,
        identifier="CAPEC-98",
        name="Phishing",
        description="Adversary tricks users into revealing sensitive information.",
    ),
]


def default_capec_provider(*, version: str = "3.9") -> StaticKnowledgeProvider:
    """Return a seeded CAPEC provider."""
    return StaticKnowledgeProvider(KnowledgeSource.CAPEC, _SEED, version=version)
