# Plugin Contract

Every plugin implements a common interface.

## Stages

| Stage | Role | Required |
|-------|------|----------|
| `collect()` | Gather raw records from a source | Optional (default: empty) |
| `normalize()` | Convert raw records → `Event` | Yes |
| `detect()` | Produce reproducible `Detection`s | Yes |
| `score()` | Attach transparent `ScoreBreakdown` | Yes |
| `explain()` | Human-readable reasoning | Optional (default provided) |
| `run()` | Full collect→…→explain pipeline | Provided by base class |

## Rules

1. Plugins **must remain independent** — never import another plugin.
2. `detect()` **must not require AI**.
3. Vendor formats stop at `normalize()` / `Event.raw`.
4. Prefer composition over deep inheritance; use helpers from core/scoring/rules/knowledge.

## Metadata

```python
from satark.core.plugin import PluginMeta

PluginMeta(
    name="insider",
    version="0.1.0",
    domain="insider",
    description="Behavioral analytics for insider threat signals",
    author="SATARK",
)
```

## Context

`PluginContext` carries runtime config (`config` dict) and a `dry_run` flag. Extra keys are allowed for research workflows.

See [Writing a Plugin](../guides/writing-a-plugin.md) and [Plugin API](../api/plugin.md).
