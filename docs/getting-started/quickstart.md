# Quickstart

Activate your environment first:

```bash
source .venv/bin/activate
```

## 1. Run the sample insider analysis

```bash
python examples/run_insider_analysis.py
```

This writes `examples/data/sample_insider.csv` (if needed) and prints explainable findings with MITRE references.

## 2. Use the CLI

```bash
satark list-plugins
satark analyze -p insider -d examples/data/sample_insider.csv
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-p` / `--plugin` | Plugin name (default: `insider`) |
| `-d` / `--data` | CSV, JSON, or JSONL of raw records |
| `-t` / `--threshold` | Elevated risk threshold (default `0.7`) |
| `--explain` / `--no-explain` | Print why-malicious answers |

## 3. Use the Python API

```python
from satark.core.engine import AnalysisEngine
from satark.plugins import create_plugin
from satark.scoring import why_malicious, prioritize

engine = AnalysisEngine(plugins=[create_plugin("insider")])

records = [
    {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "user": "alice",
        "host": "ws1",
        "usb_events": 1,
        "file_reads": 2,
        "file_writes": 1,
    },
    {
        "timestamp": "2024-01-01T01:00:00+00:00",
        "user": "alice",
        "host": "ws1",
        "usb_events": 9,
        "file_reads": 40,
        "file_writes": 20,
    },
]

events = engine.ingest_raw("insider", records)
result = engine.analyze(plugin_name="insider", events=events)

for finding in prioritize(result.findings):
    print(finding.detection.title)
    print(why_malicious(finding.detection, finding.score))
    print(f"risk={finding.score.value:.2f}")
```

## Next steps

- [Core concepts](concepts.md)
- [Event model](../architecture/events.md)
- [Plugin contract](../architecture/plugins.md)
- [Write your own plugin](../guides/writing-a-plugin.md)
