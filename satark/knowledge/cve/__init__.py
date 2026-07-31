"""CVE knowledge provider stub."""

from __future__ import annotations

from satark.core.models.knowledge_ref import KnowledgeSource
from satark.knowledge import KnowledgeEntry, StaticKnowledgeProvider


def default_cve_provider(
    entries: list[KnowledgeEntry] | None = None,
    *,
    version: str = "nvd-local",
) -> StaticKnowledgeProvider:
    """Return a CVE provider (empty by default; inject entries as needed)."""
    return StaticKnowledgeProvider(KnowledgeSource.CVE, entries or [], version=version)
