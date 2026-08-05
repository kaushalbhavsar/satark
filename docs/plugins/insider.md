# Insider Threat Plugin

## Purpose

Detect anomalous endpoint behaviors associated with insider threats and data exfiltration — especially USB activity spikes and unusual file read/write volumes.

## Architecture

Implements the full plugin contract:

1. **normalize** — wide CSV feature rows (`usb_events`, `file_reads`, `file_writes`) or long-form category records → `Event`
2. **detect** — per-actor spike detection against configurable multipliers
3. **score** — volume + severity factors with evidence and ATT&CK references
4. **explain** — default plugin explanation plus scoring helpers

## ATT&CK mapping

| Signal | Technique |
|--------|-----------|
| USB spike | T1091 Replication Through Removable Media |
| File activity spike | T1020 Automated Exfiltration |

## Usage

```python
from satark.plugins import create_plugin
from satark.core.plugin import PluginContext

plugin = create_plugin("insider")
events = plugin.normalize(records, PluginContext())
detections = plugin.detect(events, PluginContext())
score = plugin.score(detections[0], events, PluginContext())
```

CLI:

```bash
satark analyze -p insider -d examples/data/sample_insider.csv
```

## Configuration knobs

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `usb_spike_threshold` | `3.0` | Last point ≥ baseline × threshold |
| `file_spike_threshold` | `3.0` | Same for file activity |

## Legacy LSTM demo

The original TensorFlow LSTM script is preserved at `examples/legacy/lstm_usb_anomaly.py` for research comparison. Prefer this plugin for framework-native, explainable analysis.
