# Architecture Overview

SATARK separates a **domain-agnostic core** from **domain plugins**.

```text
┌─────────────────────────────────────────────────────────┐
│                     AnalysisEngine                       │
│  storage · pipelines · config · CLI                      │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
        ┌───────▼───────┐         ┌───────▼───────┐
        │   Scoring     │         │    Graph      │
        │ risk/confidence│        │ entities/paths│
        └───────────────┘         └───────────────┘
                │
        ┌───────▼───────────────────────────────────────┐
        │ Plugins: insider · malware · phishing · …     │
        │ collect → normalize → detect → score → explain│
        └───────────────────────────────────────────────┘
                │
        ┌───────▼───────┐    ┌──────────────┐
        │ Rules engines │    │ Knowledge    │
        │ regex/sigma/… │    │ ATT&CK/CWE…  │
        └───────────────┘    └──────────────┘
```

## Pipeline flow

```text
Raw source
  → Plugin.collect()
  → Plugin.normalize()   → Event
  → Plugin.detect()      → Detection
  → Plugin.score()       → ScoreBreakdown
  → Plugin.explain()     → Finding
```

The engine can also accept already-normalized events and run one or all registered plugins.

## Packages

| Package | Responsibility |
|---------|----------------|
| [`satark.core`](engine.md) | Events, engine, plugin contract, pipelines, storage, CLI, config |
| [`satark.scoring`](scoring.md) | Risk, confidence, prioritization, explainability |
| [`satark.graph`](graph.md) | Entities, relationships, timelines, attack paths |
| [`satark.rules`](rules.md) | Regex, Sigma-like, STIX stub, YARA stub, custom rules |
| [`satark.ai`](ai.md) | Optional LLM assistants — never the source of truth |
| [`satark.knowledge`](knowledge.md) | Replaceable MITRE / CAPEC / CVE / CWE providers |
| [`satark.plugins`](../plugins/overview.md) | Domain analytics modules |

## Invariants

1. **Everything is an Event** before scoring.
2. **Plugins are independent** — no cross-plugin imports.
3. **Detections are reproducible without AI.**
4. **Scores are explainable** — factors, evidence, confidence, reasoning, references.
5. **Knowledge providers are swappable** and versioned.
