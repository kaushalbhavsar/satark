# Architecture

SATARK separates a domain-agnostic core from domain plugins. The boundary is
intentional: core code operates on common models, while a plugin knows how to
interpret a particular telemetry domain.

## Package layout

```text
src/satark/
    core/         # engine, events, models, pipelines, storage, cli, config
    scoring/      # risk, confidence, prioritization, explainability
    graph/        # entities, relationships, timeline, attack paths
    rules/        # yara, sigma, regex, stix, custom
    ai/           # optional assistants (never source of truth)
    knowledge/    # mitre_attack, mitre_d3fend, capec, cve, cwe
    plugins/
        insider/
        malware/
        phishing/
        web/
        email/
        cloud/
        identity/
```

## Core model and ownership

The core understands `Event`, `Detection`, `ScoreBreakdown`, and `Finding`. It
does not understand vendor schemas or domain heuristics. A plugin normalizes raw
records into events and owns the detection and scoring logic.

| Component | Responsibility |
| --- | --- |
| `AnalysisEngine` | Registers plugins, ingests events, selects a pipeline, and applies the elevated-risk cutoff |
| `EventStore` | Holds events in memory or as JSONL; it is not a database or queue |
| `AnalysisPipeline` | Runs detect, score, and explain against one or more plugins |
| `Plugin` | Defines raw collection, normalization, detection, scoring, and explanation |
| `scoring` | Reusable aggregation, confidence, ordering, and formatting helpers |
| `knowledge` | Versioned reference data such as ATT&CK and CAPEC |
| `graph` | Optional correlation utilities; it is not automatically populated by the engine |

## Execution paths

```text
Plugin.run(): collect → normalize → detect → score → explain

Engine ingest path: raw records → plugin.normalize → EventStore
                                  ↓
Engine analysis path: Events → plugin.detect → score → explain → Findings
```

`Plugin.run()` is appropriate when the plugin owns data collection. Use
`AnalysisEngine.ingest_raw()` followed by `analyze()` when the caller has the
records already. `run_all()` evaluates every registered plugin against the same
normalized event collection; it does not make plugins depend on one another.

## Configuration

`SatarkSettings` reads environment variables beginning with `SATARK_` and an
optional `.env` file. Common settings include `SATARK_RISK_THRESHOLD`,
`SATARK_DATA_DIR`, `SATARK_ENABLE_AI`, and `SATARK_LOG_LEVEL`. The engine's
default elevated threshold is `0.7`. Thresholding marks findings as elevated;
it does not change their underlying score.

## AI boundary

The AI modules receive existing findings and may produce summaries or
recommendations. They do not run inside a plugin's detection stage. The
included `NullLLM` and `EchoLLM` are safe defaults/test doubles; applications
must provide their own real `LLMClient` implementation.

The `AnalysisEngine` registers plugins, stores events, and runs pipelines. Plugins never call each other.

## Related concepts

- [Events](concepts/events.md)
- [Findings](concepts/findings.md)
- [Scoring](concepts/scoring.md)
- [Plugins](concepts/plugins.md)
