"""Lightweight RAG stubs for knowledge retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A retrievable knowledge document."""

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class InMemoryRetriever:
    """Naive keyword retriever for research prototypes."""

    def __init__(self) -> None:
        self._docs: list[Document] = []

    def add(self, document: Document) -> None:
        """Index a document."""
        self._docs.append(document)

    def search(self, query: str, *, limit: int = 5) -> list[Document]:
        """Return documents whose text contains any query token."""
        tokens = [t.lower() for t in query.split() if t]
        scored: list[tuple[int, Document]] = []
        for doc in self._docs:
            hay = doc.text.lower()
            score = sum(1 for t in tokens if t in hay)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]
