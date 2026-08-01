# Quickstart

Activate your virtual environment first (`source .venv/bin/activate`).

## Generate and analyze sample insider data

```bash
python examples/run_insider_analysis.py
```

## Use the CLI

```bash
satark list-plugins
satark analyze -p insider -d examples/data/sample_insider.csv
```

## Programmatic usage

```python
from satark.core.engine import AnalysisEngine
from satark.plugins import create_plugin

engine = AnalysisEngine(plugins=[create_plugin("insider")])
events = engine.ingest_raw("insider", [
    {"timestamp": "2024-01-01T00:00:00+00:00", "user": "alice",
     "usb_events": 1, "file_reads": 2, "file_writes": 1},
    {"timestamp": "2024-01-01T01:00:00+00:00", "user": "alice",
     "usb_events": 9, "file_reads": 40, "file_writes": 20},
])
result = engine.analyze(plugin_name="insider", events=events)
for finding in result.findings:
    print(finding.detection.title, finding.score.value, finding.explanation)
```
