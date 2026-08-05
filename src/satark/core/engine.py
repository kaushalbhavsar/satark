"""SATARK analysis engine — orchestrates plugins, storage, scoring, and knowledge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from satark.core.config import SatarkSettings, load_settings
from satark.core.events import Event
from satark.core.models.finding import Finding
from satark.core.pipelines import AnalysisPipeline, PipelineResult
from satark.core.plugin import Plugin, PluginContext
from satark.core.storage import EventStore, InMemoryEventStore


@dataclass
class AnalysisResult:
    """Top-level engine result."""

    findings: list[Finding] = field(default_factory=list)
    events_processed: int = 0
    elevated: list[Finding] = field(default_factory=list)

    @property
    def has_elevated_risk(self) -> bool:
        """True when any finding exceeds the elevated risk threshold."""
        return bool(self.elevated)


class AnalysisEngine:
    """Domain-agnostic engine for security analytics.

    The engine does not understand vendor formats. Plugins normalize data into
    Events; the engine stores them, runs pipelines, and aggregates findings.
    """

    def __init__(
        self,
        plugins: Sequence[Plugin] | None = None,
        *,
        store: EventStore | None = None,
        settings: SatarkSettings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.store: EventStore = store or InMemoryEventStore()
        self._plugins: dict[str, Plugin] = {}
        if plugins:
            for plugin in plugins:
                self.register(plugin)

    def register(self, plugin: Plugin) -> None:
        """Register a plugin by its metadata name."""
        name = plugin.meta.name
        if name in self._plugins:
            msg = f"Plugin already registered: {name}"
            raise ValueError(msg)
        self._plugins[name] = plugin

    def list_plugins(self) -> list[str]:
        """Return registered plugin names."""
        return sorted(self._plugins)

    def get_plugin(self, name: str) -> Plugin:
        """Fetch a registered plugin or raise KeyError."""
        try:
            return self._plugins[name]
        except KeyError as exc:
            msg = f"Plugin not found: {name}"
            raise KeyError(msg) from exc

    def ingest(self, events: Sequence[Event]) -> int:
        """Store normalized events and return the count ingested."""
        self.store.put(events)
        return len(events)

    def ingest_raw(
        self,
        plugin_name: str,
        records: Sequence[dict[str, Any]],
        context: PluginContext | None = None,
    ) -> list[Event]:
        """Normalize raw records via a plugin and store the resulting events."""
        plugin = self.get_plugin(plugin_name)
        ctx = context or PluginContext()
        events = plugin.normalize(records, ctx)
        self.ingest(events)
        return events

    def analyze(
        self,
        *,
        plugin_name: str | None = None,
        events: Sequence[Event] | None = None,
        context: PluginContext | None = None,
    ) -> AnalysisResult:
        """Analyze events with one or all plugins."""
        source_events = list(events) if events is not None else self.store.list_events()
        if not self._plugins:
            return AnalysisResult(events_processed=len(source_events))

        pipeline = AnalysisPipeline(list(self._plugins.values()))
        ctx = context or PluginContext(config={"risk_threshold": self.settings.risk_threshold})

        result: PipelineResult
        if plugin_name is not None:
            result = pipeline.run_events(source_events, ctx, plugin_name=plugin_name)
        else:
            result = pipeline.run_all(source_events, ctx)

        elevated = [f for f in result.findings if f.score.value >= self.settings.risk_threshold]
        return AnalysisResult(
            findings=result.findings,
            events_processed=len(source_events),
            elevated=elevated,
        )

    def run_plugin(self, plugin_name: str, context: PluginContext | None = None) -> AnalysisResult:
        """Run a plugin's full collect→normalize→detect→score→explain pipeline."""
        plugin = self.get_plugin(plugin_name)
        findings = plugin.run(context)
        events = self.store.list_events()
        elevated = [f for f in findings if f.score.value >= self.settings.risk_threshold]
        return AnalysisResult(
            findings=findings,
            events_processed=len(events),
            elevated=elevated,
        )
