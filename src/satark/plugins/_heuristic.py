"""Base helpers for stub domain plugins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from satark.core.events import Event, EventCategory
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.score import ScoreBreakdown, ScoreFactor
from satark.core.plugin import Plugin, PluginContext, PluginMeta
from satark.scoring.risk import aggregate_score


class HeuristicDomainPlugin(Plugin):
    """Reusable heuristic plugin for domain stubs.

    Matches events whose category is in ``watched_categories`` and whose tags
    intersect ``watched_tags``, emitting a single detection when any match.
    """

    def __init__(
        self,
        *,
        name: str,
        domain: str,
        description: str,
        watched_categories: set[EventCategory],
        watched_tags: set[str],
        rule_id: str,
        title: str,
        severity: DetectionSeverity = DetectionSeverity.MEDIUM,
    ) -> None:
        self._meta = PluginMeta(
            name=name,
            version="0.1.0",
            domain=domain,
            description=description,
            author="SATARK",
        )
        self.watched_categories = watched_categories
        self.watched_tags = watched_tags
        self.rule_id = rule_id
        self.title = title
        self.severity = severity

    @property
    def meta(self) -> PluginMeta:
        return self._meta

    def normalize(
        self,
        records: Sequence[dict[str, Any]],
        context: PluginContext,
    ) -> list[Event]:
        events: list[Event] = []
        for record in records:
            category = EventCategory(str(record.get("category", EventCategory.CUSTOM.value)))
            tags = record.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            events.append(
                Event(
                    category=category,
                    source=str(record.get("source", f"{self.meta.name}.collector")),
                    actor=record.get("actor"),
                    target=record.get("target"),
                    host=record.get("host"),
                    action=record.get("action"),
                    attributes={
                        k: v
                        for k, v in record.items()
                        if k
                        not in {
                            "category",
                            "source",
                            "timestamp",
                            "actor",
                            "target",
                            "host",
                            "action",
                            "tags",
                        }
                    },
                    tags=list(tags),
                    raw=dict(record),
                )
            )
        return events

    def detect(self, events: Sequence[Event], context: PluginContext) -> list[Detection]:
        matched = [
            e
            for e in events
            if e.category in self.watched_categories or self.watched_tags.intersection(e.tags)
        ]
        if not matched:
            return []
        evidence = [
            Evidence(
                kind=EvidenceKind.EVENT,
                summary=f"{e.category.value} from {e.source}",
                source_event_id=str(e.id),
                weight=0.7,
            )
            for e in matched
        ]
        return [
            Detection(
                plugin=self.meta.name,
                rule_id=self.rule_id,
                title=self.title,
                description=self.meta.description,
                severity=self.severity,
                event_ids=[e.id for e in matched],
                evidence=evidence,
                tags=[self.meta.domain],
            )
        ]

    def score(
        self,
        detection: Detection,
        events: Sequence[Event],
        context: PluginContext,
    ) -> ScoreBreakdown:
        factor = ScoreFactor(
            name="domain_match",
            contribution=0.55,
            description=f"{self.meta.domain} heuristic matched {len(detection.event_ids)} event(s)",
            evidence=list(detection.evidence),
        )
        return aggregate_score(
            [factor],
            confidence=0.6,
            reasoning=f"Heuristic domain plugin '{self.meta.name}' matched watched signals.",
            evidence=detection.evidence,
            baseline=0.1,
        )
