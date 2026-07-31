"""Plugin contract for SATARK security analytics plugins.

Every plugin implements a common interface with stages:

* ``collect()`` — gather raw records
* ``normalize()`` — convert to :class:`~satark.core.events.Event`
* ``detect()`` — produce reproducible detections
* ``score()`` — attach transparent risk scores
* ``explain()`` — human-readable reasoning

Plugins must remain independent and must never depend on another plugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from satark.core.events import Event
from satark.core.models.detection import Detection
from satark.core.models.finding import Finding
from satark.core.models.score import ScoreBreakdown


class PluginMeta(BaseModel):
    """Descriptive metadata for a plugin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str = "0.1.0"
    domain: str = Field(description="Security domain, e.g. insider, malware, phishing")
    description: str = ""
    author: str | None = None


class PluginContext(BaseModel):
    """Runtime context passed to plugin stages."""

    model_config = ConfigDict(frozen=True, extra="allow")

    config: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class Plugin(ABC):
    """Abstract base for all SATARK analytics plugins.

    Prefer composition for helpers; subclasses implement the stage methods.
    Detections produced by :meth:`detect` must be reproducible without AI.
    """

    @property
    @abstractmethod
    def meta(self) -> PluginMeta:
        """Return plugin metadata."""

    def collect(self, context: PluginContext) -> Iterable[dict[str, Any]]:
        """Collect raw vendor/source records.

        Default implementation yields nothing; override when the plugin owns
        its own data sources.
        """
        return []

    @abstractmethod
    def normalize(
        self,
        records: Sequence[dict[str, Any]],
        context: PluginContext,
    ) -> list[Event]:
        """Normalize raw records into canonical :class:`Event` objects."""

    @abstractmethod
    def detect(self, events: Sequence[Event], context: PluginContext) -> list[Detection]:
        """Produce detections from normalized events (no AI required)."""

    @abstractmethod
    def score(
        self,
        detection: Detection,
        events: Sequence[Event],
        context: PluginContext,
    ) -> ScoreBreakdown:
        """Compute a transparent risk score for a detection."""

    def explain(
        self,
        detection: Detection,
        score: ScoreBreakdown,
        events: Sequence[Event],
        context: PluginContext,
    ) -> str:
        """Return a human-readable explanation for a scored detection."""
        factor_lines = "; ".join(
            f"{f.name} ({f.contribution:+.2f}): {f.description}" for f in score.factors
        )
        return (
            f"{detection.title}: risk={score.value:.2f}, confidence={score.confidence:.2f}. "
            f"{score.reasoning}" + (f" Factors: {factor_lines}." if factor_lines else "")
        )

    def run(self, context: PluginContext | None = None) -> list[Finding]:
        """Execute the full plugin pipeline: collect → normalize → detect → score → explain."""
        ctx = context or PluginContext()
        records = list(self.collect(ctx))
        events = self.normalize(records, ctx)
        detections = self.detect(events, ctx)
        findings: list[Finding] = []
        for detection in detections:
            score = self.score(detection, events, ctx)
            explanation = self.explain(detection, score, events, ctx)
            findings.append(
                Finding(
                    detection=detection,
                    score=score,
                    explanation=explanation,
                )
            )
        return findings
