# Insider Threat Plugin

The `insider` plugin detects anomalous USB and file activity volumes per actor, maps findings to MITRE ATT&CK (e.g. T1091, T1020), and produces explainable risk scores.

## Usage

```python
from satark.plugins import create_plugin
from satark.core.plugin import PluginContext

plugin = create_plugin("insider")
events = plugin.normalize(records, PluginContext())
detections = plugin.detect(events, PluginContext())
```

## Legacy LSTM demo

The original TensorFlow LSTM script lives at `examples/legacy/lstm_usb_anomaly.py` for research comparison. The plugin path is the supported, framework-native approach.
