# Plugin contract

Plugins implement a shared lifecycle and inherit from the abstract `Plugin`
base class:

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

## Implementing a plugin

Implement `meta`, `normalize`, `detect`, and `score`. `collect` is optional;
the default returns no records. `explain` has a generic implementation based on
the score factors, but a domain plugin may override it.

```python
class ExamplePlugin(Plugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(name="example", domain="example")

    def normalize(self, records, context) -> list[Event]:
        ...

    def detect(self, events, context) -> list[Detection]:
        ...

    def score(self, detection, events, context) -> ScoreBreakdown:
        ...
```

Register a built-in plugin in `satark.plugins.registry`. Test normalization,
positive and negative detection cases, score bounds, and any input parsing
errors. Keep network access, model loading, and vendor SDK setup outside
`detect()` where possible so the method remains deterministic and easy to test.
