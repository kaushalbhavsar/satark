# Data onboarding

SATARK accepts raw records through plugins. A plugin converts each record into
one or more canonical `Event` objects. Start by choosing the plugin and then
make the input format explicit, versioned, and testable.

## CLI file formats

The `satark analyze` command reads CSV, JSON, and JSONL. JSON must be a list
of objects; JSONL has one JSON object per nonempty line. CSV column names become
record keys and values are strings, so plugins must parse numbers and timestamps.

```bash
uv run satark analyze -p insider -d data/telemetry.csv
```

## Insider wide records

The insider plugin accepts a wide row with a timestamp, actor, and counts:

```csv
timestamp,user,host,usb_events,file_reads,file_writes
2026-09-14T09:00:00+00:00,alice,workstation-7,0,4,1
2026-09-14T10:00:00+00:00,alice,workstation-7,1,5,2
```

`user` and `actor` are interchangeable. A row becomes up to three events:
USB insertion, file read, and file write. Counts are parsed as floats. Zero
counts do not create an event in the current wide-record normalizer.

The plugin also accepts long records with `category`, for example:

```json
{
  "timestamp": "2026-09-14T10:00:00Z",
  "category": "usb_insertion",
  "actor": "alice",
  "source": "endpoint.agent",
  "count": 8
}
```

Use a valid [event category](concepts/events.md#categories). Unknown categories
raise validation errors rather than silently becoming a different signal.

## Data quality checklist

- Use timezone-aware ISO-8601 timestamps, preferably UTC.
- Keep the aggregation interval consistent: hourly rows should stay hourly.
- Keep a stable actor identifier; do not mix display names and account IDs.
- Preserve source and host where available.
- Treat missing counts intentionally. Empty values are interpreted as zero by
  the insider wide-record parser.
- Retain original records outside SATARK if they include sensitive information.
  `Event.raw` can retain a copy for investigation but should not become the
  basis of a generic scoring rule.

## Baseline versus incoming data

The deterministic insider detector can operate on a single ordered dataset.
The optional LSTM needs two datasets: known-normal baseline telemetry for
training/calibration and later incoming telemetry for analysis. See
[Insider threats](plugins/insider.md#optional-lstm-anomaly-evidence) for the
minimum bucket requirements and code sample.

## Inspect normalized events

Normalize a small sample before running a large import:

```python
from satark.core.plugin import PluginContext
from satark.plugins import create_plugin

plugin = create_plugin("insider")
events = plugin.normalize(records[:10], PluginContext())
for event in events:
    print(event.timestamp, event.category, event.actor, event.attributes)
```

This is the best place to find parsing mistakes, unexpected users, bad time
zones, or missing source fields.
