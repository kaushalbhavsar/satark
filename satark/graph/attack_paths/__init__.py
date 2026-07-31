"""Attack-path discovery over the entity graph."""

from __future__ import annotations

from dataclasses import dataclass

from satark.graph.entities import EntityGraph


@dataclass(frozen=True)
class AttackPath:
    """A path between entities that may represent an attack chain."""

    nodes: tuple[str, ...]
    length: int

    @property
    def summary(self) -> str:
        """Human-readable path summary."""
        return " → ".join(self.nodes)


def find_attack_paths(
    graph: EntityGraph,
    source: str,
    target: str,
    *,
    max_length: int = 8,
) -> list[AttackPath]:
    """Find simple paths from source to target up to ``max_length`` hops."""
    import networkx as nx

    if source not in graph.graph or target not in graph.graph:
        return []
    paths: list[AttackPath] = []
    for path in nx.all_simple_paths(graph.graph, source, target, cutoff=max_length):
        paths.append(AttackPath(nodes=tuple(path), length=len(path) - 1))
    paths.sort(key=lambda p: p.length)
    return paths
