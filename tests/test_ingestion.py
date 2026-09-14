"""Coverage for log-format parsing and standard-schema projections."""

from pathlib import Path

from satark.core.events import Event, EventCategory
from satark.ingestion import (
    load_ocsf_events,
    ocsf_to_event,
    parse_cef,
    parse_json,
    parse_leef,
    parse_syslog,
    to_ecs,
    to_ocsf,
)
from satark.timeseries import build_time_series


def test_cef_to_ecs_preserves_network_and_source_fields() -> None:
    record = parse_cef(
        "CEF:0|Acme|Firewall|1.0|100|Blocked connection|8|src=10.0.0.1 dst=10.0.0.2 msg=blocked"
    )
    ecs = to_ecs(record)
    assert ecs["event"]["id"] == "100"
    assert ecs["event"]["severity"] == 8
    assert ecs["source"]["ip"] == "10.0.0.1"
    assert ecs["satark"]["unmapped"]["dst"] == "10.0.0.2"


def test_leef_2_delimiter_is_parsed() -> None:
    record = parse_leef("LEEF:2.0|IBM|QRadar|7.5|42|^|devTime=2026-09-14T00:00:00Z^src=10.0.0.1")
    assert record["event_id"] == "42"
    assert record["fields"]["src"] == "10.0.0.1"


def test_rfc5424_to_ecs_retains_syslog_metadata() -> None:
    record = parse_syslog(
        '<34>1 2026-09-14T10:00:00Z host app 123 ID47 [example@32473 key="value"] message'
    )
    ecs = to_ecs(record)
    assert ecs["log"]["syslog"]["priority"] == 34
    assert ecs["log"]["syslog"]["facility"]["code"] == 4
    assert ecs["log"]["syslog"]["structured_data"]["example@32473"]["key"] == "value"


def test_rfc3164_and_nested_json_to_ocsf() -> None:
    syslog = parse_syslog("<13>Sep 14 12:34:56 host sshd[99]: accepted", year=2026)
    assert syslog["app"] == "sshd"
    record = parse_json({
        "@timestamp": "2026-09-14T10:00:00Z",
        "host": {"name": "host-a"},
        "user": {"name": "alice"},
    })
    ocsf = to_ocsf(record)
    assert ocsf["class_uid"] == 0
    assert ocsf["metadata"]["product"]["name"] == "json"
    assert ocsf["device"]["hostname"] == "host-a"


def test_ocsf_event_converts_directly_to_satark_event() -> None:
    event = ocsf_to_event({
        "class_uid": 3002,
        "category_uid": 3,
        "activity_id": 1,
        "severity_id": 3,
        "time": 1_789_379_200_000,
        "metadata": {"product": {"name": "identity-provider"}},
        "class_name": "Authentication",
        "actor": {"user": {"name": "alice"}},
        "device": {"hostname": "host-a"},
    })
    assert event.category is EventCategory.AUTHENTICATION
    assert event.actor == "alice"
    assert event.source == "identity-provider"


def test_ocsf_jsonl_file_loads_directly(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"class_uid":0,"category_uid":0,"activity_id":0,"severity_id":0,'
        '"time":1789379200000,"metadata":{"product":{"name":"test"}}}\n',
        encoding="utf-8",
    )
    events = load_ocsf_events(path)
    assert len(events) == 1
    assert events[0].source == "test"


def test_time_series_groups_counts_and_fills_hourly_gaps() -> None:
    first = Event(
        category=EventCategory.USB_INSERTION,
        source="test",
        actor="alice",
        timestamp="2026-09-14T00:10:00Z",
        attributes={"count": 2},
    )
    last = Event(
        category=EventCategory.FILE_READ,
        source="test",
        actor="alice",
        timestamp="2026-09-14T02:10:00Z",
        attributes={"count": 4},
    )
    series = build_time_series(
        [first, last],
        categories=[EventCategory.USB_INSERTION, EventCategory.FILE_READ],
    )
    points = series["alice"]
    assert len(points) == 3
    assert points[0].values[EventCategory.USB_INSERTION] == 2
    assert points[1].total == 0
    assert points[2].values[EventCategory.FILE_READ] == 4
