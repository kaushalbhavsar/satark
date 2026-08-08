# Plugin contract

Plugins implement a shared lifecycle:

```text
collect → normalize → detect → score → explain
```

| Stage | Responsibility |
|-------|----------------|
| `collect` | Gather raw records from a source (optional) |
| `normalize` | Convert raw records into `Event` objects |
| `detect` | Produce reproducible detections (**no AI required**) |
| `score` | Attach a transparent `ScoreBreakdown` |
| `explain` | Produce human-readable reasoning |

## Independence

Plugins **must not** depend directly on each other. Share only core models and utilities (`satark.core`, `satark.scoring`, `satark.rules`, `satark.knowledge`, `satark.graph`).

## AI is optional

AI may assist with summarization, explanation enrichment, investigation help, reporting, and recommendations. Detections must remain reproducible without an LLM.

## Registry

```python
from satark.plugins import builtin_plugins, create_plugin

print(builtin_plugins())
plugin = create_plugin("insider")
```
