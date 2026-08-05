# Analysis Engine

The `AnalysisEngine` is the domain-agnostic orchestrator.

## Responsibilities

- Register plugins by name
- Ingest normalized events (or raw records via a plugin)
- Store events (`InMemoryEventStore` by default)
- Run one plugin or all plugins through an `AnalysisPipeline`
- Aggregate findings and flag elevated risk against a threshold

## Example

```python
from satark.core.config import load_settings
from satark.core.engine import AnalysisEngine
from satark.plugins import create_plugin

settings = load_settings(risk_threshold=0.7)
engine = AnalysisEngine(
    plugins=[create_plugin("insider"), create_plugin("identity")],
    settings=settings,
)

events = engine.ingest_raw("insider", records)
result = engine.analyze(plugin_name="insider", events=events)

print(result.events_processed, len(result.findings), len(result.elevated))
```

## Key types

| Type | Role |
|------|------|
| `AnalysisEngine` | Top-level orchestrator |
| `AnalysisResult` | Findings + elevated subset + event count |
| `AnalysisPipeline` | Linear composition of plugin stages |
| `SatarkSettings` | Env-backed config (`SATARK_*`) |
| `EventStore` | Persistence abstraction |

## Storage

| Backend | Use case |
|---------|----------|
| `InMemoryEventStore` | Tests, notebooks, short runs |
| `JsonlEventStore` | Lightweight append-only research persistence |

```python
from pathlib import Path
from satark.core.storage import JsonlEventStore

store = JsonlEventStore(Path("data/events.jsonl"))
engine = AnalysisEngine(plugins=[create_plugin("web")], store=store)
```

See [Engine API](../api/engine.md) and [Storage API](../api/storage.md).
