# Log ingestion and schema normalization

SATARK can now parse CEF, LEEF, RFC 3164 Syslog, RFC 5424 Syslog, and nested
JSON into a neutral record. The record can then be projected into ECS or an
OCSF Base Event envelope.

It can also accept a valid OCSF Base Event directly and convert it into a
canonical SATARK `Event` for analysis.

## Scope

The parsers are source-format adapters. They preserve fields that do not have a
safe generic mapping. The ECS adapter emits commonly used fields such as
`@timestamp`, `event.*`, `host.name`, `user.name`, source/destination IPs,
`log.syslog.*`, and `event.original`.

The OCSF adapter emits a Base Event envelope with its required base identifiers,
metadata, timestamp, message, raw data, and unmapped attributes. It deliberately
uses the Uncategorized base class (`class_uid: 0`, `category_uid: 0`) because a
concrete OCSF class must be chosen based on source semantics. A source-specific
adapter should set the appropriate class UID and populate that class's required
fields before strict OCSF validation.

## CEF

```python
from satark.ingestion import parse_cef, to_ecs

record = parse_cef(
    "CEF:0|Acme|Firewall|1.0|100|Blocked connection|8|src=10.0.0.1 dst=10.0.0.2"
)
ecs = to_ecs(record)
```

CEF headers map vendor, product, product version, signature ID, event name, and
severity. Its extension key-value pairs remain available in `record["fields"]`
and in the ECS `satark.unmapped` object.

## LEEF

```python
from satark.ingestion import parse_leef, to_ocsf

record = parse_leef(
    "LEEF:2.0|IBM|QRadar|7.5|42|^|devTime=2026-09-14T00:00:00Z^src=10.0.0.1"
)
ocsf_base_event = to_ocsf(record)
```

LEEF 1.0 uses tab-separated extension fields. LEEF 2.0 declares its extension
delimiter after the five header fields; the parser supports that declared
delimiter.

## Syslog

```python
from satark.ingestion import parse_syslog, to_ecs

record = parse_syslog(
    '<34>1 2026-09-14T10:00:00Z host app 123 ID47 [meta@32473 key="value"] message'
)
ecs = to_ecs(record)
```

RFC 5424 parsing retains priority, facility, severity, app name, process ID,
message ID, and structured data. RFC 3164 messages lack a year and timezone;
pass `year=` to `parse_syslog()` when the collector's current year would be
incorrect. RFC 3164 timestamps are treated as UTC by this adapter.

## Nested JSON

```python
from satark.ingestion import parse_json, to_ecs

record = parse_json({
    "@timestamp": "2026-09-14T10:00:00Z",
    "host": {"name": "host-a"},
    "user": {"name": "alice"},
    "source": {"ip": "10.0.0.1"},
    "message": "authentication succeeded",
})
ecs = to_ecs(record)
```

The JSON adapter looks for common nested names such as `@timestamp`,
`host.name`, `user.name`, `source.ip`, `destination.ip`, and `event.*`. It
retains the complete payload under unmapped fields so information is not lost.

## From normalized schema to SATARK analysis

Use `ocsf_to_event()` for direct OCSF Base Event ingestion:

```python
from satark.core.engine import AnalysisEngine
from satark.ingestion import ocsf_to_event
from satark.plugins import create_plugin

event = ocsf_to_event(ocsf_json_object)
engine = AnalysisEngine(plugins=[create_plugin("identity")])
result = engine.analyze(plugin_name="identity", events=[event])
```

For OCSF files, load JSON objects, JSON arrays, or JSONL directly:

```python
from pathlib import Path
from satark.ingestion import load_ocsf_events

events = load_ocsf_events(Path("data/ocsf-events.jsonl"))
```

The adapter maps common OCSF semantics such as Authentication, Login, Process,
File, Network, Email, Web, DNS, and API activity to the corresponding SATARK
event categories. Unknown OCSF classes become `custom` events and preserve
OCSF identifiers and unmapped content in `attributes` and `raw`.

ECS output remains an interchange format. Add a source-specific ECS adapter
when you need to turn its category and action conventions into SATARK events.
