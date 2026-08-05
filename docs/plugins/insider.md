# Insider threats

The `insider` plugin detects anomalous USB and file activity volumes per actor, maps findings to MITRE ATT&CK techniques such as T1091 and T1020 when available, and produces explainable scores.

## Status

Implemented behavioral spike detection (framework-native). A legacy LSTM demo remains at `examples/legacy/lstm_usb_anomaly.py` for research comparison.

## Usage

```bash
uv run satark analyze -p insider -d examples/data/sample_insider.csv
```

```python
from satark.plugins import create_plugin
from satark.core.plugin import PluginContext

plugin = create_plugin("insider")
events = plugin.normalize(records, PluginContext())
detections = plugin.detect(events, PluginContext())
```
