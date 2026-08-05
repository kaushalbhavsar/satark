# Architecture

SATARK separates a **domain-agnostic core** from **domain plugins**.

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

## Core remains domain-agnostic

The core understands **Events**, **Detections**, **ScoreBreakdowns**, and **Findings**. It does not understand vendor schemas or domain heuristics. Plugins normalize raw data into events and own domain logic.

## Pipeline

```text
collect → normalize → detect → score → explain
```

The `AnalysisEngine` registers plugins, stores events, and runs pipelines. Plugins never call each other.

## Related concepts

- [Events](concepts/events.md)
- [Findings](concepts/findings.md)
- [Scoring](concepts/scoring.md)
- [Plugins](concepts/plugins.md)
