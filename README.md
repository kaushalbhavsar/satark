# SATARK

**Scalable Automated Technology for Analysis and Ranking of Known Threats**

SATARK is an open-source security analytics framework that unifies threat detection across domains—insider threats, malware, phishing, identity, cloud, web, email, and more—on a shared, plugin-first architecture.

> Everything in SATARK is an **Event**. Plugins normalize vendor data into a common model, then detect, score, and explain findings with full transparency.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Design Principles

- Plugin-first architecture
- Clean separation of concerns
- Domain-agnostic core
- AI-assisted analysis (never the source of truth)
- Explainable detections
- Research-friendly and enterprise-ready
- Test-driven, typed, documented

## Quick Start

### Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Install

```bash
uv sync
```

Or with pip:

```bash
pip install -e ".[dev]"
```

### Analyze sample insider telemetry

```bash
uv run python examples/run_insider_analysis.py
```

### CLI

```bash
uv run satark version
uv run satark list-plugins
uv run satark analyze --plugin insider --data examples/data/sample_insider.csv
```

## Architecture

```
Raw sources → Plugin.collect() → Plugin.normalize() → Event
     → Plugin.detect() → Detection
     → Plugin.score()  → ScoreBreakdown (factors, evidence, confidence, reasoning, references)
     → Plugin.explain() → Finding
```

Plugins never depend on each other. The engine orchestrates pipelines and storage.

### Repository layout

```
satark/
  core/        # engine, events, models, pipelines, storage, cli, config
  scoring/     # risk, confidence, prioritization, explainability
  graph/       # entities, relationships, timeline, attack_paths
  rules/       # yara, sigma, regex, stix, custom
  ai/          # agents, prompts, rag, explain, embeddings
  knowledge/   # mitre_attack, mitre_d3fend, capec, cve, cwe
  plugins/     # insider, malware, phishing, web, email, cloud, identity
tests/
examples/
docs/
```

## Risk Scoring

SATARK never returns only a number. Every score includes:

- contributing factors
- evidence
- confidence
- reasoning
- knowledge references (MITRE ATT&CK, D3FEND, CAPEC, CWE, CVE)

## AI Integration

LLMs assist with summarization, investigation guidance, report generation, and explanation enrichment. **Detections are always reproducible without AI.**

## Legacy Demo

The original LSTM USB anomaly script is preserved at [`examples/legacy/lstm_usb_anomaly.py`](examples/legacy/lstm_usb_anomaly.py). The insider plugin provides a framework-native, dependency-light successor focused on explainable behavioral spikes.

## Development

```bash
uv sync
uv run pytest
uv run ruff check satark tests
uv run black --check satark tests
uv run mypy satark
```

Docs (MkDocs Material):

```bash
uv run mkdocs serve
```

## License

MIT © Dr. Kaushal Bhavsar

## Contributing

Contributions that add plugins, knowledge providers, or detection capabilities on the shared architecture are welcome. Open a pull request with tests and documentation.
