# Plugin Contract

Every plugin implements:

| Stage | Role |
|-------|------|
| `collect()` | Gather raw records from a source |
| `normalize()` | Convert raw records → `Event` |
| `detect()` | Produce reproducible `Detection`s (no AI required) |
| `score()` | Attach transparent `ScoreBreakdown` |
| `explain()` | Human-readable reasoning |

Plugins must remain independent—never import or call another plugin.

See [Writing a Plugin](../guides/writing-a-plugin.md).
