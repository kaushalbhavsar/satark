"""Embedding stubs — replaceable providers for future vector search."""

from __future__ import annotations

from collections.abc import Sequence


class HashEmbedding:
    """Deterministic bag-of-words style embedding for tests (not semantic)."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Embed text into a fixed-size float vector."""
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            idx = hash(token) % self.dimensions
            vector[idx] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]
