# Getting started

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) is recommended for repository development

## Install from a checkout

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
uv sync --group dev
```

Install optional machine-learning dependencies only when using the LSTM
detector:

```bash
uv sync --extra ml
```

For a conventional virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Confirm the installation

```bash
uv run satark version
uv run satark list-plugins
uv run pytest
```

`list-plugins` displays the built-in plugin names and their current
descriptions. Availability does not mean every plugin is feature-complete; see
the individual plugin pages for their scope.

## Run the included insider example

```bash
uv run python examples/run_insider_analysis.py
```

The script writes sample hourly telemetry, normalizes it, runs the insider
plugin, prioritizes the findings, and prints their risk and ATT&CK references.

You can run the same data through the CLI:

```bash
uv run satark analyze --plugin insider --data examples/data/sample_insider.csv
```

Use `--threshold 0.5` to change the engine's elevated-risk cutoff, and
`--explain/--no-explain` to control explanation output. Supported CLI input
formats are CSV, JSON (a list of records), and JSONL (one object per line).

## A minimal library workflow

```python
from satark.core.engine import AnalysisEngine
from satark.core.plugin import PluginContext
from satark.plugins import create_plugin

records = [
    {"timestamp": "2026-09-14T09:00:00+00:00", "user": "alice", "usb_events": 1},
    {"timestamp": "2026-09-14T10:00:00+00:00", "user": "alice", "usb_events": 8},
]

engine = AnalysisEngine(plugins=[create_plugin("insider")])
events = engine.ingest_raw("insider", records, PluginContext())
result = engine.analyze(plugin_name="insider", events=events)

for finding in result.findings:
    print(finding.detection.title, finding.score.value)
```

Continue with [data onboarding](data-onboarding.md) before using operational
telemetry.
