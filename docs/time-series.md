# Time-series analysis

SATARK's `timeseries` module turns canonical events into fixed-interval activity
series. It is designed for baselines, anomaly detection, charts, and models
such as the insider LSTM.

## Build a series

```python
from datetime import timedelta

from satark.core.events import EventCategory
from satark.timeseries import build_time_series

series_by_actor = build_time_series(
    events,
    interval=timedelta(hours=1),
    group_by="actor",
    categories=[
        EventCategory.USB_INSERTION,
        EventCategory.FILE_READ,
        EventCategory.FILE_WRITE,
    ],
    fill_gaps=True,
)
```

The result is a dictionary keyed by actor. Each `TimeSeriesPoint` contains the
UTC bucket start time, selected category values, total activity, and the IDs of
events contributing to that bucket.

## Bucketing behavior

Events are placed into UTC-aligned fixed buckets. The default interval is one
hour. An event's `attributes["count"]` is used as its activity value; otherwise
the event contributes one. Counts must be finite and nonnegative.

With `fill_gaps=True` (the default), SATARK adds zero-valued buckets between
the first and last observed bucket for each group. This means a two-hour idle
period becomes visible instead of disappearing from a model input sequence.
No buckets are created before the first event or after the last event because
the data alone does not establish that observation window.

## Grouping

Choose `group_by="actor"`, `"host"`, or `"source"`. Missing values are grouped
as `unknown`. Keep the same grouping and interval for baseline and incoming data
when training or evaluating an anomaly detector.

## LSTM integration

`LstmInsiderDetector` now uses this bucketing layer internally for USB insertion,
file reads, and file writes. It receives continuous hourly actor sequences even
when there are no matching events in intermediate buckets. The original event
IDs remain attached to anomaly findings; zero-filled buckets naturally have no
event IDs.
