"""Projection of neutral parsed records to ECS and OCSF base-event envelopes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def to_ecs(record: dict[str, Any]) -> dict[str, Any]:
    """Project a parsed record to an ECS-shaped event while preserving source fields."""
    event: dict[str, Any] = {
        "@timestamp": record.get("timestamp") or datetime.now(UTC).isoformat(),
        "message": record.get("message") or record.get("event_name") or "",
        "event": {"kind": "event", "original": record.get("raw", "")},
        "labels": {"satark.ingestion.format": record.get("format", "unknown")},
    }
    _put(event["event"], "id", record.get("event_id"))
    _put(event["event"], "action", record.get("event_name"))
    _put(event["event"], "severity", _integer(record.get("severity")))
    _put_nested(event, "host.name", record.get("host"))
    _put_nested(event, "user.name", record.get("user"))
    _put_nested(event, "source.ip", record.get("source_ip"))
    _put_nested(event, "destination.ip", record.get("destination_ip"))
    if record.get("format") == "syslog":
        log = {"syslog": {}}
        _put(log["syslog"], "priority", record.get("priority"))
        _put(log["syslog"], "version", record.get("version"))
        _put(log["syslog"], "appname", record.get("app"))
        _put(log["syslog"], "procid", record.get("process_id"))
        _put(log["syslog"], "msgid", record.get("event_id"))
        _put_nested(log["syslog"], "facility.code", record.get("facility"))
        _put_nested(log["syslog"], "severity.code", record.get("severity"))
        _put(log["syslog"], "structured_data", record.get("structured_data") or None)
        event["log"] = log
    event["satark"] = {"unmapped": record.get("fields", {})}
    return event


def to_ocsf(record: dict[str, Any]) -> dict[str, Any]:
    """Project a parsed record to an OCSF Base Event envelope.

    This produces required base identifiers with an Uncategorized class. Choose
    a concrete OCSF event class and its required attributes in a source adapter
    before submitting the result to a strict OCSF validator.
    """
    time = _epoch_millis(record.get("timestamp"))
    result: dict[str, Any] = {
        "class_uid": 0,
        "category_uid": 0,
        "activity_id": 0,
        "severity_id": _ocsf_severity(record.get("severity")),
        "time": time,
        "metadata": {
            "version": "1.0.0",
            "product": {"name": record.get("product") or record.get("format", "unknown")},
        },
        "message": record.get("message") or record.get("event_name") or "",
        "raw_data": record.get("raw", ""),
        "unmapped": record.get("fields", {}),
    }
    if record.get("host"):
        result["device"] = {"hostname": record["host"]}
    if record.get("user"):
        result["actor"] = {"user": {"name": record["user"]}}
    if record.get("source_ip"):
        result["src_endpoint"] = {"ip": record["source_ip"]}
    if record.get("destination_ip"):
        result["dst_endpoint"] = {"ip": record["destination_ip"]}
    return result


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        target[key] = value


def _put_nested(target: dict[str, Any], path: str, value: Any) -> None:
    if value is None or value == "":
        return
    parent, key = path.rsplit(".", 1)
    cursor = target
    for segment in parent.split("."):
        cursor = cursor.setdefault(segment, {})
    cursor[key] = value


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _epoch_millis(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.now(UTC).timestamp() * 1000)


def _ocsf_severity(value: Any) -> int:
    severity = _integer(value)
    if severity is None:
        return 0
    return min(6, max(0, severity))
