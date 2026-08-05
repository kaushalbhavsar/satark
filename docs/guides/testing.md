# Testing Guide

SATARK is developed test-first. Every feature should include unit tests.

## Run tests

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Coverage is enabled by default via `pyproject.toml`.

## Layout

```text
tests/
  core/        # events, engine, storage, cli
  scoring/
  graph/
  rules/
  knowledge/
  plugins/
```

## What to test

| Layer | Focus |
|-------|-------|
| Models | Validation, immutability, helpers |
| Plugins | normalize → detect → score on fixtures |
| Engine | registration, ingest, analyze, thresholds |
| Scoring | clamping, explainability text, prioritization |
| Rules / graph / knowledge | matchers, paths, lookups |

## Example plugin test

```python
from satark.core.plugin import PluginContext
from satark.plugins.insider import InsiderThreatPlugin

def test_usb_spike_detection() -> None:
    plugin = InsiderThreatPlugin(usb_spike_threshold=2.0)
    records = [
        {"timestamp": "2024-01-01T00:00:00+00:00", "user": "bob",
         "usb_events": 1, "file_reads": 1, "file_writes": 0},
        {"timestamp": "2024-01-01T01:00:00+00:00", "user": "bob",
         "usb_events": 8, "file_reads": 1, "file_writes": 0},
    ]
    events = plugin.normalize(records, PluginContext())
    detections = plugin.detect(events, PluginContext())
    assert detections
    score = plugin.score(detections[0], events, PluginContext())
    assert 0.0 <= score.value <= 1.0
```

## Lint & types

```bash
ruff check satark tests
black --check satark tests examples --exclude examples/legacy
mypy satark
```
