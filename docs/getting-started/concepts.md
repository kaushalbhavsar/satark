# Concepts

## Event

The canonical unit of telemetry in SATARK. Every source — USB activity, login, process start, cloud API call — is normalized into an [`Event`](../architecture/events.md) before analysis.

Vendor payloads may be retained in `Event.raw`, but scoring and the engine never depend on vendor schemas.

## Plugin

A domain module that implements the stages:

`collect` → `normalize` → `detect` → `score` → `explain`

Plugins are independent. They must not import or call other plugins. Shared code lives in `satark.core`, `satark.scoring`, `satark.rules`, and `satark.knowledge`.

## Detection

A reproducible signal produced by `detect()`. Detections must not require AI. They carry severity, evidence, optional rule IDs, and knowledge references.

## ScoreBreakdown

Transparent risk output. Always includes:

- numeric risk (`value`) and `confidence`
- contributing `factors`
- `evidence`
- narrative `reasoning`
- knowledge `references`

SATARK never returns “just a number.”

## Finding

The end-to-end unit returned to analysts: a detection + score + explanation (+ optional AI enrichment flags/recommendations).

## Engine

[`AnalysisEngine`](../architecture/engine.md) registers plugins, stores events, and runs analysis pipelines.

## Knowledge reference

A pointer to MITRE ATT&CK, D3FEND, CAPEC, CWE, CVE, or a custom catalog entry. Providers are replaceable and versioned.
