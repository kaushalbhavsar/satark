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

## Deterministic spike detector

The framework-native detector groups events by `actor` and evaluates USB and
file activity separately. For each actor, it uses all values except the latest
as a baseline. The latest value is suspicious when it is at least the configured
multiple of that baseline. The default multiple is `3.0` for both USB and file
activity. If there is only one value, it is compared directly with the threshold.

Wide input rows containing `usb_events`, `file_reads`, and `file_writes` become
USB insertion, file-read, and file-write events. The detector combines reads
and writes when evaluating file activity. USB detections are high severity and
map to ATT&CK T1091 where available; file detections are medium severity and
map to T1020.

The score starts from `0.1`, adds an activity-volume factor up to `0.5`, and
adds a severity factor. It is capped at `1.0`. This makes the score easy to
trace, but it is a prioritization policy rather than a probability of insider
malice.

```python
from satark.plugins.insider import InsiderThreatPlugin

plugin = InsiderThreatPlugin(usb_spike_threshold=4.0, file_spike_threshold=2.5)
```

Order matters: the latest event is the candidate spike. Sort records by
timestamp before normalizing them if the source does not already do so.

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

## Choosing between detectors

Use the deterministic detector first when you have a clear activity policy and
need an immediately inspectable answer. Use the LSTM only with a sufficiently
large, trusted normal baseline and measure its false-positive behavior against
held-out telemetry. Run both approaches side by side during evaluation; their
agreement and disagreement are useful investigation signals.

Neither detector establishes intent or proves exfiltration. Preserve the event
IDs, review the original telemetry, and correlate with host, identity, network,
and case-management context before taking action.
