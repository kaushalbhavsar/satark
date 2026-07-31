"""Analysis pipelines composing plugin stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from satark.core.events import Event
from satark.core.models.finding import Finding
from satark.core.plugin import Plugin, PluginContext


@dataclass
class PipelineResult:
    """Result of running a pipeline over a set of events or raw records."""

    events: list[Event] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)


class AnalysisPipeline:
    """Compose one or more plugins into a linear analysis pipeline.

    Plugins never call each other; the pipeline orchestrates stages.
    """

    def __init__(self, plugins: Sequence[Plugin]) -> None:
        if not plugins:
            msg = "AnalysisPipeline requires at least one plugin"
            raise ValueError(msg)
        self._plugins = list(plugins)

    @property
    def plugins(self) -> list[Plugin]:
        """Registered plugins in execution order."""
        return list(self._plugins)

    def run_records(
        self,
        records: Sequence[dict[str, Any]],
        context: PluginContext | None = None,
        *,
        plugin_name: str | None = None,
    ) -> PipelineResult:
        """Normalize and analyze raw records with a single plugin (by name or first)."""
        ctx = context or PluginContext()
        plugin = self._select(plugin_name)
        events = plugin.normalize(records, ctx)
        return self._analyze(plugin, events, ctx)

    def run_events(
        self,
        events: Sequence[Event],
        context: PluginContext | None = None,
        *,
        plugin_name: str | None = None,
    ) -> PipelineResult:
        """Analyze already-normalized events with a selected plugin."""
        ctx = context or PluginContext()
        plugin = self._select(plugin_name)
        return self._analyze(plugin, list(events), ctx)

    def run_all(
        self,
        events: Sequence[Event],
        context: PluginContext | None = None,
    ) -> PipelineResult:
        """Run every registered plugin against the same event set."""
        ctx = context or PluginContext()
        all_findings: list[Finding] = []
        for plugin in self._plugins:
            result = self._analyze(plugin, list(events), ctx)
            all_findings.extend(result.findings)
        return PipelineResult(events=list(events), findings=all_findings)

    def _select(self, plugin_name: str | None) -> Plugin:
        if plugin_name is None:
            return self._plugins[0]
        for plugin in self._plugins:
            if plugin.meta.name == plugin_name:
                return plugin
        available = ", ".join(p.meta.name for p in self._plugins)
        msg = f"Unknown plugin '{plugin_name}'. Available: {available}"
        raise KeyError(msg)

    def _analyze(
        self,
        plugin: Plugin,
        events: list[Event],
        context: PluginContext,
    ) -> PipelineResult:
        detections = plugin.detect(events, context)
        findings: list[Finding] = []
        for detection in detections:
            score = plugin.score(detection, events, context)
            explanation = plugin.explain(detection, score, events, context)
            findings.append(Finding(detection=detection, score=score, explanation=explanation))
        return PipelineResult(events=events, findings=findings)
