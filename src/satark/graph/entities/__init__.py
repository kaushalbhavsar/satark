"""Entity-relationship graph for correlating security entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

import networkx as nx

from satark.core.events import Event


class EntityType(StrEnum):
    """Node types in the SATARK entity graph."""

    ACTOR = "actor"
    HOST = "host"
    TARGET = "target"
    EVENT = "event"
    PROCESS = "process"
    FILE = "file"
    IP = "ip"
    EMAIL = "email"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Entity:
    """A graph node representing a security-relevant entity."""

    id: str
    type: EntityType
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    """A directed edge between entities."""

    source: str
    target: str
    relation: str
    event_id: UUID | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class EntityGraph:
    """NetworkX-backed entity graph for correlation and attack-path analysis."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph[str] = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph[str]:
        """Underlying NetworkX digraph (read-only use recommended)."""
        return self._graph

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node."""
        self._graph.add_node(
            entity.id,
            type=entity.type.value,
            label=entity.label,
            **entity.attributes,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a directed relationship edge."""
        self._graph.add_edge(
            relationship.source,
            relationship.target,
            relation=relationship.relation,
            event_id=str(relationship.event_id) if relationship.event_id else None,
            **relationship.attributes,
        )

    def ingest_event(self, event: Event) -> None:
        """Derive entities and relationships from a normalized event."""
        event_node = f"event:{event.id}"
        self.add_entity(
            Entity(
                id=event_node,
                type=EntityType.EVENT,
                label=event.category.value,
                attributes={"timestamp": event.timestamp.isoformat()},
            )
        )
        if event.actor:
            actor_id = f"actor:{event.actor}"
            self.add_entity(Entity(id=actor_id, type=EntityType.ACTOR, label=event.actor))
            self.add_relationship(
                Relationship(
                    source=actor_id,
                    target=event_node,
                    relation="performed",
                    event_id=event.id,
                )
            )
        if event.host:
            host_id = f"host:{event.host}"
            self.add_entity(Entity(id=host_id, type=EntityType.HOST, label=event.host))
            self.add_relationship(
                Relationship(
                    source=event_node,
                    target=host_id,
                    relation="on_host",
                    event_id=event.id,
                )
            )
        if event.target:
            target_id = f"target:{event.target}"
            self.add_entity(Entity(id=target_id, type=EntityType.TARGET, label=event.target))
            self.add_relationship(
                Relationship(
                    source=event_node,
                    target=target_id,
                    relation="affected",
                    event_id=event.id,
                )
            )

    def neighbors(self, entity_id: str) -> list[str]:
        """Return adjacent entity ids."""
        if entity_id not in self._graph:
            return []
        return list(self._graph.neighbors(entity_id))

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return the shortest path between two entities, or empty if none."""
        try:
            return list(nx.shortest_path(self._graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def node_count(self) -> int:
        """Number of entities in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Number of relationships in the graph."""
        return self._graph.number_of_edges()
