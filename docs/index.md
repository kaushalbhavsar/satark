# SATARK Documentation

SATARK (Scalable Automated Technology for Analysis and Ranking of Known Threats)
is a Python framework for building security analytics that an analyst can inspect.
It provides a shared event schema, plugin lifecycle, evidence model, transparent
risk scoring, knowledge references, lightweight graph utilities, and an optional
AI layer for writing assistance.

SATARK is currently an alpha project. The insider plugin contains the most
substantial detection logic. The other built-in domain plugins are deliberately
small heuristics intended as extension examples, not complete detection products.

## What SATARK does

```text
Raw telemetry → normalized Events → reproducible Detections
              → evidence-backed Scores → analyst-facing Findings
```

The core does not parse vendor-specific schemas or decide what is malicious.
Plugins own normalization and detection behavior; the core runs them consistently.

## Start here

1. [Getting started](getting-started.md) installs SATARK and runs a sample.
2. [Data onboarding](data-onboarding.md) explains how to prepare telemetry.
3. [Architecture](architecture.md) describes the execution model.
4. [Insider threats](plugins/insider.md) documents the implemented detector and
   optional LSTM workflow.

## Capability status

| Area | Current capability | Maturity |
| --- | --- | --- |
| Event ingestion and storage | Canonical Pydantic events; memory and JSONL stores | Usable for research/prototypes |
| Insider analytics | Per-actor USB and file-activity spikes | Implemented |
| LSTM anomaly detection | Optional sequence model with baseline calibration | Experimental |
| Malware, phishing, web, email, cloud, identity | Category/tag heuristics | Extension scaffolding |
| Scoring and explanations | Factors, evidence, confidence, knowledge references | Implemented |
| Graph utilities | Entity graph, timeline, simple path discovery | Implemented utility layer |
| AI | Optional summaries and recommendations | No model provider included |

## Principles

- **Detections are reproducible.** A plugin's `detect()` method must run without
  an LLM.
- **Scores show their work.** A number without factors, evidence, and reasoning
  is not a SATARK score.
- **Raw telemetry is quarantined.** Preserve it for investigation, but rely on
  normalized fields for reusable logic.
- **AI assists the analyst.** It may summarize a finding or suggest follow-up
  work; it does not alter a detection or make a score authoritative.

See the [API reference](api/index.md) for the public Python surfaces and
[research origins](research/origins.md) for project provenance.
