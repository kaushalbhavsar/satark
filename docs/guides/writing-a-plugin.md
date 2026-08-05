# Writing a Plugin

## Checklist

1. Subclass `satark.core.plugin.Plugin`
2. Implement `meta`, `normalize`, `detect`, and `score`
3. Optionally override `collect` / `explain`
4. Register in `satark.plugins.registry` (or pass the instance to `AnalysisEngine`)
5. Add unit tests under `tests/plugins/`
6. Document purpose, architecture, examples, and use cases

## Skeleton

```python
from collections.abc import Sequence
from typing import Any

from satark.core.events import Event, EventCategory
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.score import ScoreBreakdown, ScoreFactor
from satark.core.plugin import Plugin, PluginContext, PluginMeta
from satark.scoring.risk import aggregate_score


class MyPlugin(Plugin):
    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="myplugin",
            domain="custom",
            description="Demo plugin",
        )

    def normalize(
        self,
        records: Sequence[dict[str, Any]],
        context: PluginContext,
    ) -> list[Event]:
        return [
            Event(
                category=EventCategory.CUSTOM,
                source="myplugin",
                attributes=dict(record),
            )
            for record in records
        ]

    def detect(
        self,
        events: Sequence[Event],
        context: PluginContext,
    ) -> list[Detection]:
        return [
            Detection(
                plugin=self.meta.name,
                rule_id="myplugin.demo",
                title="Demo detection",
                description="Example rule fired",
                severity=DetectionSeverity.LOW,
                event_ids=[events[0].id] if events else [],
                evidence=[
                    Evidence(
                        kind=EvidenceKind.CONTEXT,
                        summary="Demo evidence",
                    )
                ],
            )
        ] if events else []

    def score(
        self,
        detection: Detection,
        events: Sequence[Event],
        context: PluginContext,
    ) -> ScoreBreakdown:
        return aggregate_score(
            [
                ScoreFactor(
                    name="base",
                    contribution=0.2,
                    description="Demo contribution",
                )
            ],
            confidence=0.5,
            reasoning="Demo score for documentation",
            evidence=detection.evidence,
        )
```

## Do / Don't

**Do**

- Keep `detect()` deterministic and AI-free
- Attach evidence and knowledge references
- Keep functions small and typed

**Don't**

- Import another plugin
- Score from `Event.raw` vendor fields in the engine
- Return a bare float as the only scoring output
