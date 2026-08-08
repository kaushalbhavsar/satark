# SATARK Documentation

**SATARK** — *Scalable Automated Technology for Analysis and Ranking of Known Threats* — is an open-source security analytics framework for building explainable detection pipelines across multiple security domains.

Official project website: [https://satark.org/](https://satark.org/)

## What problem it solves

Security teams and researchers often rebuild the same plumbing for every detector: ingest, normalize, score, explain, and map findings to knowledge bases. SATARK provides a **shared, plugin-first architecture** so those capabilities can be reused across insider threat, malware, phishing, identity, cloud, web, and email analytics.

## Who it is for

- Security researchers prototyping detections
- Engineers building reusable analytics plugins
- Teams that need transparent risk scores (not unexplained numbers)
- Contributors extending domain coverage without rewriting the core

## Current status

SATARK is an **early open-source rebuild (alpha)**. APIs, packaging, and plugin coverage are evolving. Do not treat the project as production-ready unless your own evaluation supports that conclusion.

## Supported plugin domains

| Domain | Plugin name | Status |
|--------|-------------|--------|
| Insider threats | `insider` | Implemented (behavioral spikes) |
| Malware | `malware` | Heuristic stub |
| Phishing | `phishing` | Heuristic stub |
| Web | `web` | Heuristic stub |
| Email | `email` | Heuristic stub |
| Cloud | `cloud` | Heuristic stub |
| Identity | `identity` | Heuristic stub |

## Where to start

1. [Getting started](getting-started.md)
2. [Architecture](architecture.md)
3. [Events](concepts/events.md)
4. [Plugins](plugins/index.md)
5. [API reference](api/index.md)
