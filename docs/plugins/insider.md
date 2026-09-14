# Insider threats

The `insider` plugin detects anomalous USB and file activity volumes per actor, maps findings to MITRE ATT&CK techniques such as T1091 and T1020 when available, and produces explainable scores.

## Status

Implemented behavioral spike detection (framework-native). An optional LSTM
backend is available for model-derived, per-actor anomaly evidence; it does
not replace deterministic detections or their transparent scoring.

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

## Optional LSTM anomaly evidence

Install the ML dependencies, then fit on a known-normal baseline before
evaluating later telemetry. The detector aggregates USB insertion, file-read,
and file-write counts by actor and timestamp. Its scaler is fit only on the
training portion of the baseline, and its threshold is calibrated against the
held-out baseline tail.

The split is performed on timestamp buckets separately for every actor before
windows are constructed, so training and calibration share no observations.
Both partitions must contain at least `sequence_length` buckets. With the
defaults (20 steps and 20% validation), supply at least 100 buckets per actor.
Inference requires at least 20 buckets per actor and scores each window's final
bucket, including the final bucket in the dataset. The model and scaler are
shared across actors, but windows never cross actor boundaries. Use consistent
time intervals; this backend does not resample or fill missing intervals.
The current insider wide-record normalizer omits entirely zero-activity rows;
represent those explicitly as zero-count Events if they should be time steps.
Risk and confidence are heuristic review aids, not calibrated probabilities.

```bash
uv sync --extra ml
```

```python
from satark.plugins import LstmInsiderDetector

detector = LstmInsiderDetector(sequence_length=20)
detector.fit(known_normal_events)
ml_findings = detector.analyze(new_events)
```

`ml_findings` are standard SATARK `Finding` objects with event IDs,
reconstruction-error evidence, and a deliberately moderate explainable score.
