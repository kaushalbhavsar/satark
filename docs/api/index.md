# API reference

This page generates reference material from the implementation. The stable
conceptual API is the event → detection → score → finding flow; individual
classes remain subject to alpha-version changes.

## Typical imports

```python
from satark.core.engine import AnalysisEngine
from satark.core.events import Event, EventCategory
from satark.core.plugin import Plugin, PluginContext, PluginMeta
from satark.core.models import Detection, Evidence, Finding, ScoreBreakdown
from satark.plugins import create_plugin
```

## Core

::: satark.core.engine.AnalysisEngine

::: satark.core.plugin.Plugin

::: satark.core.events.Event

## Scoring

::: satark.scoring.risk.aggregate_score

::: satark.scoring.explainability.why_malicious

## Plugins

::: satark.plugins.registry.create_plugin

::: satark.plugins.insider.InsiderThreatPlugin

::: satark.plugins.insider.lstm.LstmInsiderDetector

## Supporting APIs

`satark.core.storage` provides `InMemoryEventStore` for experiments and
`JsonlEventStore` for lightweight persistent event logs. `satark.graph` offers
`EntityGraph`, `build_timeline`, and `find_attack_paths`. `satark.rules`
contains regex, Sigma-like, STIX-like, and custom predicate rule engines; YARA
is a placeholder that requires external integration.

`satark.knowledge` provides static, versioned lookup providers. The included
catalogs are small seeds, not a replacement for a maintained upstream data-sync
process.

For deeper module docs, browse the source under `src/satark/`.
