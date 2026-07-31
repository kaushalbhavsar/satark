"""Replaceable knowledge providers (MITRE ATT&CK, D3FEND, CAPEC, CVE, CWE)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from satark.core.models.knowledge_ref import KnowledgeReference, KnowledgeSource


@dataclass(frozen=True)
class KnowledgeEntry:
    """A versioned knowledge-base entry."""

    source: KnowledgeSource
    identifier: str
    name: str
    description: str = ""
    version: str = "1.0"
    url: str | None = None

    def as_reference(self) -> KnowledgeReference:
        """Convert to a :class:`KnowledgeReference`."""
        return KnowledgeReference(
            source=self.source,
            identifier=self.identifier,
            name=self.name,
            url=self.url,  # type: ignore[arg-type]
            version=self.version,
        )


class KnowledgeProvider(ABC):
    """Abstract knowledge provider — independently versioned and replaceable."""

    @property
    @abstractmethod
    def source(self) -> KnowledgeSource:
        """Knowledge source this provider serves."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Dataset version string."""

    @abstractmethod
    def get(self, identifier: str) -> KnowledgeEntry | None:
        """Lookup an entry by identifier."""

    @abstractmethod
    def search(self, query: str) -> list[KnowledgeEntry]:
        """Search entries by free-text query."""


class StaticKnowledgeProvider(KnowledgeProvider):
    """In-memory provider useful for tests and offline research."""

    def __init__(
        self,
        source: KnowledgeSource,
        entries: list[KnowledgeEntry],
        *,
        version: str = "1.0",
    ) -> None:
        self._source = source
        self._version = version
        self._entries = {e.identifier.upper(): e for e in entries}

    @property
    def source(self) -> KnowledgeSource:
        return self._source

    @property
    def version(self) -> str:
        return self._version

    def get(self, identifier: str) -> KnowledgeEntry | None:
        return self._entries.get(identifier.upper())

    def search(self, query: str) -> list[KnowledgeEntry]:
        q = query.lower()
        return [
            e
            for e in self._entries.values()
            if q in e.identifier.lower() or q in e.name.lower() or q in e.description.lower()
        ]
