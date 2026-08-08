"""CWE weakness provider (seed)."""

from __future__ import annotations

from satark.core.models.knowledge_ref import KnowledgeSource
from satark.knowledge import KnowledgeEntry, StaticKnowledgeProvider

_SEED: list[KnowledgeEntry] = [
    KnowledgeEntry(
        source=KnowledgeSource.CWE,
        identifier="CWE-79",
        name="Cross-site Scripting",
        description="Improper neutralization of input during web page generation.",
        url="https://cwe.mitre.org/data/definitions/79.html",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.CWE,
        identifier="CWE-287",
        name="Improper Authentication",
        description="When an actor claims an identity, the software does not prove it.",
    ),
]


def default_cwe_provider(*, version: str = "4.15") -> StaticKnowledgeProvider:
    """Return a seeded CWE provider."""
    return StaticKnowledgeProvider(KnowledgeSource.CWE, _SEED, version=version)
