# Architecture

SATARK is a plugin-first security analytics framework.

## High-level layout

```text
src/satark/
  core/        # domain-agnostic engine, events, models, pipelines, storage, CLI
  scoring/     # explainable risk, confidence, prioritization
  graph/       # entity correlation and attack paths
  rules/       # rule engines
  ai/          # optional assistants (not the source of truth)
  knowledge/   # replaceable knowledge providers
  plugins/     # domain detectors (insider, malware, phishing, …)
```

## Invariants

1. Everything is normalized to an **Event** before scoring.
2. The **core stays domain-agnostic**.
3. Plugins are **independent** (no cross-plugin imports).
4. Detections are **reproducible without AI**.
5. Scores are **explainable** (factors, evidence, confidence, reasoning, references).

See [docs/architecture.md](docs/architecture.md) for the published documentation version.
