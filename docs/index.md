# SATARK

**Scalable Automated Technology for Analysis and Ranking of Known Threats**

SATARK is an open-source Python framework for security analytics. It gives researchers and engineers a shared engine to collect, normalize, analyze, score, correlate, and explain security events — across insider threat, malware, phishing, identity, cloud, web, email, and future domains.

## Core idea

Everything in SATARK is an **Event**.

Plugins never pass vendor-specific formats into the engine. They normalize raw telemetry into a common model, then produce reproducible detections and transparent risk scores.

## What you get

| Capability | Package |
|------------|---------|
| Engine, events, plugin contract, CLI | `satark.core` |
| Explainable risk scoring | `satark.scoring` |
| Entity graphs & attack paths | `satark.graph` |
| Rule engines (regex, Sigma-like, …) | `satark.rules` |
| Optional AI assistants | `satark.ai` |
| MITRE / CAPEC / CWE / CVE providers | `satark.knowledge` |
| Domain plugins | `satark.plugins` |

## Design principles

- Plugin-first architecture
- Clean separation of concerns
- Domain-agnostic core
- AI-assisted analysis (never the source of truth)
- Explainable detections
- Research-friendly and enterprise-ready
- Test-driven, typed, documented

## Start here

1. [Install](getting-started/installation.md)
2. [Quickstart](getting-started/quickstart.md)
3. [Concepts](getting-started/concepts.md)
4. [Architecture overview](architecture/overview.md)
5. [Write a plugin](guides/writing-a-plugin.md)

!!! tip "CLI at a glance"
    ```bash
    satark list-plugins
    satark analyze -p insider -d examples/data/sample_insider.csv
    ```
