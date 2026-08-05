# Examples

Runnable examples live under `examples/`.

## Insider analysis walkthrough

```bash
source .venv/bin/activate
python examples/run_insider_analysis.py
```

What it does:

1. Writes sample CSV telemetry to `examples/data/sample_insider.csv`
2. Normalizes rows through the `insider` plugin
3. Detects USB/file spikes for actor `alice`
4. Prints explainable risk scores and ATT&CK references

Equivalent CLI:

```bash
satark analyze -p insider -d examples/data/sample_insider.csv
```

## Sample data schema (wide format)

| Column | Type | Meaning |
|--------|------|---------|
| `timestamp` | ISO-8601 | Event time |
| `user` / `actor` | string | Identity |
| `host` | string | Endpoint |
| `usb_events` | number | USB activity count |
| `file_reads` | number | File read volume |
| `file_writes` | number | File write volume |

## Long-form event records

Plugins that use `HeuristicDomainPlugin` accept category-oriented dicts:

```json
{
  "category": "login",
  "source": "idp",
  "actor": "alice",
  "host": "vpn-gw",
  "tags": ["identity", "bruteforce"],
  "action": "failed_login"
}
```

## Legacy LSTM demo

```text
examples/legacy/lstm_usb_anomaly.py
```

Requires optional ML dependencies (`pip install -e ".[ml]"` plus TensorFlow if you run that script as originally written). Prefer the insider plugin for framework-native workflows.

## Graph correlation snippet

```python
from satark.core.events import Event, EventCategory
from satark.graph import EntityGraph, build_timeline, find_attack_paths

events = [
    Event(category=EventCategory.LOGIN, source="idp", actor="alice", host="ws1"),
    Event(
        category=EventCategory.USB_INSERTION,
        source="endpoint",
        actor="alice",
        host="ws1",
        target="USB-9",
    ),
]

graph = EntityGraph()
for event in events:
    graph.ingest_event(event)

print(build_timeline(events))
print(find_attack_paths(graph, "actor:alice", "target:USB-9"))
```
