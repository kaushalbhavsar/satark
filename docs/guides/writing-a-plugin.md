# Writing a Plugin

1. Subclass `satark.core.plugin.Plugin`.
2. Implement `meta`, `normalize`, `detect`, and `score` (override `collect` / `explain` as needed).
3. Register via `satark.plugins.registry` or pass the instance to `AnalysisEngine`.
4. Add unit tests under `tests/plugins/`.
5. Document purpose, architecture, examples, and use cases.

## Skeleton

```python
from collections.abc import Sequence
from typing import Any

from satark.core.events import Event, EventCategory
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.score import ScoreBreakdown, ScoreFactor
from satark.core.plugin import Plugin, PluginContext, PluginMeta
from satark.scoring.risk import aggregate_score

class MyPlugin(Plugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(name="myplugin", domain="custom", description="Demo plugin")

    def normalize(self, records: Sequence[dict[str, Any]], context: PluginContext) -> list[Event]:
        return [
            Event(category=EventCategory.CUSTOM, source="myplugin", attributes=dict(r))
            for r in records
        ]

    def detect(self, events: Sequence[Event], context: PluginContext) -> list[Detection]:
        return []

    def score(self, detection, events, context) -> ScoreBreakdown:
        return aggregate_score(
            [ScoreFactor(name="base", contribution=0.2, description="demo")],
            confidence=0.5,
            reasoning="Demo score",
        )
```

**Do not** depend on another plugin. Share only core models and utilities.
