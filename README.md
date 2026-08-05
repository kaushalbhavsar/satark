# SATARK

**Scalable Automated Technology for Analysis and Ranking of Known Threats**

Open-source Python framework for security analytics across insider threat, malware, phishing, identity, cloud, web, and email — built on one shared engine.

> Everything is an **Event**. Plugins normalize source data, then detect, score, and explain findings with full transparency.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why SATARK

- **Plugin-first** — domain detectors share one contract; plugins never depend on each other
- **Domain-agnostic core** — vendor formats stop at the plugin boundary
- **Explainable risk** — every score includes factors, evidence, confidence, reasoning, and references
- **AI-assisted, not AI-owned** — detections stay reproducible without LLMs
- **Knowledge-mapped** — findings can map to MITRE ATT&CK, D3FEND, CAPEC, CWE, and CVE

## Requirements

- Python 3.13+
- `pip` and `venv`

## Install

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Check the install:

```bash
satark version
satark list-plugins
```

## Quick start

### CLI

```bash
satark analyze --plugin insider --data examples/data/sample_insider.csv
```

### Example script

```bash
python examples/run_insider_analysis.py
```

### Python API

```python
from satark.core.engine import AnalysisEngine
from satark.plugins import create_plugin

engine = AnalysisEngine(plugins=[create_plugin("insider")])

events = engine.ingest_raw("insider", [
    {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "user": "alice",
        "usb_events": 1,
        "file_reads": 2,
        "file_writes": 1,
    },
    {
        "timestamp": "2024-01-01T01:00:00+00:00",
        "user": "alice",
        "usb_events": 9,
        "file_reads": 40,
        "file_writes": 20,
    },
])

result = engine.analyze(plugin_name="insider", events=events)

for finding in result.findings:
    print(finding.detection.title)
    print(f"risk={finding.score.value:.2f} confidence={finding.score.confidence:.2f}")
    print(finding.explanation)
```

## How it works

```text
Raw source
  → collect()
  → normalize()   # → Event
  → detect()      # → Detection (no AI required)
  → score()       # → ScoreBreakdown
  → explain()     # → Finding
```

The engine stores events, runs plugins, and aggregates findings. Scores always answer:

**“Why was this classified as malicious?”**

## Project layout

```text
satark/
  core/         # engine, events, models, pipelines, storage, cli, config
  scoring/      # risk, confidence, prioritization, explainability
  graph/        # entities, relationships, timeline, attack paths
  rules/        # yara, sigma, regex, stix, custom
  ai/           # agents, prompts, rag, explain, embeddings
  knowledge/    # mitre_attack, mitre_d3fend, capec, cve, cwe
  plugins/      # insider, malware, phishing, web, email, cloud, identity
tests/
examples/
docs/
```

### Built-in plugins

| Plugin     | Domain   | Status                          |
|------------|----------|---------------------------------|
| `insider`  | Insider  | Behavioral USB/file spike logic |
| `malware`  | Malware  | Heuristic stub                  |
| `phishing` | Phishing | Heuristic stub                  |
| `web`      | Web      | Heuristic stub                  |
| `email`    | Email    | Heuristic stub                  |
| `cloud`    | Cloud    | Heuristic stub                  |
| `identity` | Identity | Heuristic stub                  |

## Development

```bash
source .venv/bin/activate
pip install -e ".[dev,docs]"

pytest
ruff check satark tests
black --check satark tests examples --exclude examples/legacy
mypy satark
```

Docs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Full documentation (architecture, guides, and API reference) lives in [`docs/`](docs/) and builds with MkDocs Material.

## Writing a plugin

Subclass `satark.core.plugin.Plugin` and implement:

1. `normalize()` — raw records → `Event`
2. `detect()` — reproducible detections
3. `score()` — transparent `ScoreBreakdown`
4. `explain()` — optional; default explanation is provided

See [docs/guides/writing-a-plugin.md](docs/guides/writing-a-plugin.md).

## Legacy demo

The original LSTM USB anomaly script lives at [`examples/legacy/lstm_usb_anomaly.py`](examples/legacy/lstm_usb_anomaly.py). Prefer the `insider` plugin for framework-native analysis.

## License

MIT © [Dr. Kaushal Bhavsar](https://bhavsar.ai)

## Contributing

PRs that add plugins, knowledge providers, tests, or docs on the shared architecture are welcome.
