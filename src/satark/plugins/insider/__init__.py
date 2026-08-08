"""Insider-threat analytics plugin.

Purpose
-------
Detect anomalous endpoint behaviors associated with insider threats — USB
activity spikes, unusual file access, and related behavioral patterns.

Architecture
------------
Implements the SATARK plugin contract. Detection is statistical and rule-based
(reproducible without AI). Optional ML backends can be layered later.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from satark.core.events import Event, EventCategory
from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.knowledge_ref import KnowledgeReference, KnowledgeSource
from satark.core.models.score import ScoreBreakdown, ScoreFactor
from satark.core.plugin import Plugin, PluginContext, PluginMeta
from satark.knowledge.mitre_attack import default_attack_provider
from satark.scoring.confidence import evidence_confidence
from satark.scoring.risk import aggregate_score


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now().astimezone()
    text = str(value)
    # Support both "Z" and offset-aware ISO strings
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


class InsiderThreatPlugin(Plugin):
    """Detect insider-threat patterns from normalized endpoint events."""

    def __init__(
        self,
        *,
        usb_spike_threshold: float = 3.0,
        file_spike_threshold: float = 3.0,
    ) -> None:
        self.usb_spike_threshold = usb_spike_threshold
        self.file_spike_threshold = file_spike_threshold
        self._attack = default_attack_provider()

    @property
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="insider",
            version="0.1.0",
            domain="insider",
            description="Behavioral analytics for insider threat and data exfiltration signals",
            author="SATARK",
        )

    def collect(self, context: PluginContext) -> Iterable[dict[str, Any]]:
        path = context.config.get("data_path")
        if not path:
            return []
        import csv
        from pathlib import Path

        records: list[dict[str, Any]] = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records.extend(dict(row) for row in reader)
        return records

    def normalize(
        self,
        records: Sequence[dict[str, Any]],
        context: PluginContext,
    ) -> list[Event]:
        events: list[Event] = []
        for record in records:
            # Support both wide (feature columns) and long (category) formats
            if "category" in record:
                category = EventCategory(str(record["category"]))
                events.append(
                    Event(
                        category=category,
                        source=str(record.get("source", "insider.collector")),
                        timestamp=_parse_timestamp(record.get("timestamp")),
                        actor=record.get("actor") or record.get("user"),
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
                                "user",
                                "target",
                                "host",
                                "action",
                            }
                        },
                        raw=dict(record),
                    )
                )
                continue

            # Wide format: usb_events / file_reads / file_writes counts per row
            ts = _parse_timestamp(record.get("timestamp"))
            actor = record.get("actor") or record.get("user")
            host = record.get("host")
            source = str(record.get("source", "insider.collector"))

            usb_count = float(record.get("usb_events") or 0)
            if usb_count > 0:
                events.append(
                    Event(
                        category=EventCategory.USB_INSERTION,
                        source=source,
                        timestamp=ts,
                        actor=actor,
                        host=host,
                        action="usb_activity",
                        attributes={"count": usb_count},
                        tags=["insider", "usb"],
                        raw=dict(record),
                    )
                )
            file_reads = float(record.get("file_reads") or 0)
            if file_reads > 0:
                events.append(
                    Event(
                        category=EventCategory.FILE_READ,
                        source=source,
                        timestamp=ts,
                        actor=actor,
                        host=host,
                        action="file_read",
                        attributes={"count": file_reads},
                        tags=["insider", "file"],
                        raw=dict(record),
                    )
                )
            file_writes = float(record.get("file_writes") or 0)
            if file_writes > 0:
                events.append(
                    Event(
                        category=EventCategory.FILE_WRITE,
                        source=source,
                        timestamp=ts,
                        actor=actor,
                        host=host,
                        action="file_write",
                        attributes={"count": file_writes},
                        tags=["insider", "file"],
                        raw=dict(record),
                    )
                )
        return events

    def detect(self, events: Sequence[Event], context: PluginContext) -> list[Detection]:
        by_actor: dict[str, list[Event]] = defaultdict(list)
        for event in events:
            key = event.actor or "unknown"
            by_actor[key].append(event)

        detections: list[Detection] = []
        for actor, actor_events in by_actor.items():
            usb_counts = [
                float(e.attributes.get("count", 1))
                for e in actor_events
                if e.category == EventCategory.USB_INSERTION
            ]
            file_counts = [
                float(e.attributes.get("count", 1))
                for e in actor_events
                if e.category in {EventCategory.FILE_READ, EventCategory.FILE_WRITE}
            ]

            if usb_counts and self._is_spike(usb_counts, self.usb_spike_threshold):
                event_ids = [
                    e.id for e in actor_events if e.category == EventCategory.USB_INSERTION
                ]
                technique = self._attack.get("T1091")
                knowledge = [technique.as_reference()] if technique else []
                detections.append(
                    Detection(
                        plugin=self.meta.name,
                        rule_id="insider.usb_spike",
                        title=f"Anomalous USB activity for {actor}",
                        description=(
                            f"USB event volume for actor '{actor}' exceeds "
                            f"{self.usb_spike_threshold}× the actor baseline."
                        ),
                        severity=DetectionSeverity.HIGH,
                        event_ids=event_ids,
                        evidence=[
                            Evidence(
                                kind=EvidenceKind.BEHAVIORAL,
                                summary=f"USB counts={usb_counts}",
                                details={"counts": usb_counts, "actor": actor},
                                weight=0.9,
                            )
                        ],
                        knowledge=knowledge,
                        tags=["insider", "usb", "exfiltration"],
                    )
                )

            if file_counts and self._is_spike(file_counts, self.file_spike_threshold):
                event_ids = [
                    e.id
                    for e in actor_events
                    if e.category in {EventCategory.FILE_READ, EventCategory.FILE_WRITE}
                ]
                technique = self._attack.get("T1020")
                knowledge = [technique.as_reference()] if technique else []
                detections.append(
                    Detection(
                        plugin=self.meta.name,
                        rule_id="insider.file_spike",
                        title=f"Anomalous file activity for {actor}",
                        description=(
                            f"File read/write volume for actor '{actor}' exceeds "
                            f"{self.file_spike_threshold}× the actor baseline."
                        ),
                        severity=DetectionSeverity.MEDIUM,
                        event_ids=event_ids,
                        evidence=[
                            Evidence(
                                kind=EvidenceKind.STATISTIC,
                                summary=f"File activity counts={file_counts}",
                                details={"counts": file_counts, "actor": actor},
                                weight=0.8,
                            )
                        ],
                        knowledge=knowledge,
                        tags=["insider", "file", "exfiltration"],
                    )
                )
        return detections

    def score(
        self,
        detection: Detection,
        events: Sequence[Event],
        context: PluginContext,
    ) -> ScoreBreakdown:
        related = [e for e in events if e.id in set(detection.event_ids)]
        volume = sum(float(e.attributes.get("count", 1)) for e in related)
        volume_factor = ScoreFactor(
            name="activity_volume",
            contribution=min(0.5, volume / 20.0),
            description=f"Aggregated activity volume={volume}",
            evidence=list(detection.evidence),
        )
        severity_boost = {
            DetectionSeverity.INFO: 0.05,
            DetectionSeverity.LOW: 0.15,
            DetectionSeverity.MEDIUM: 0.3,
            DetectionSeverity.HIGH: 0.45,
            DetectionSeverity.CRITICAL: 0.6,
        }[detection.severity]
        severity_factor = ScoreFactor(
            name="severity",
            contribution=severity_boost,
            description=f"Detection severity={detection.severity.value}",
        )
        references = list(detection.knowledge) or [
            KnowledgeReference(
                source=KnowledgeSource.MITRE_ATTACK,
                identifier="T1020",
                name="Automated Exfiltration",
            )
        ]
        confidence = evidence_confidence(detection.evidence)
        return aggregate_score(
            [volume_factor, severity_factor],
            confidence=max(confidence, 0.5),
            reasoning=(
                f"Insider-threat rule '{detection.rule_id}' fired with "
                f"supporting behavioral evidence for elevated risk."
            ),
            evidence=detection.evidence,
            references=references,
            baseline=0.1,
        )

    @staticmethod
    def _is_spike(values: list[float], threshold: float) -> bool:
        if len(values) < 2:
            return values[0] >= threshold if values else False
        baseline = sum(values[:-1]) / len(values[:-1])
        if baseline <= 0:
            return values[-1] >= threshold
        return values[-1] >= baseline * threshold
