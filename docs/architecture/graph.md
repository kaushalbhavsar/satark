# Graph Correlation

`satark.graph` builds entity graphs from normalized events for correlation and attack-path analysis.

## EntityGraph

```python
from satark.graph import EntityGraph, build_timeline, find_attack_paths

graph = EntityGraph()
for event in events:
    graph.ingest_event(event)

path = graph.shortest_path("actor:alice", "target:USB-9")
paths = find_attack_paths(graph, "actor:alice", "target:USB-9")
timeline = build_timeline(events)
```

## Node types

`actor`, `host`, `target`, `event`, `process`, `file`, `ip`, `email`, `custom`

## Outputs

| Helper | Output |
|--------|--------|
| `build_timeline` | Chronological `TimelineEntry` list |
| `find_attack_paths` | Simple paths as `AttackPath` objects |
| `EntityGraph.shortest_path` | Single shortest path or `[]` |

See [Graph API](../api/graph.md).
