"""Direct conversion from OCSF JSON records to SATARK events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from satark.core.events import Event, EventCategory


def parse_ocsf(value: str | dict[str, Any]) -> dict[str, Any]:
    """Read one OCSF JSON object and validate its base-event shape."""
    record = json.loads(value) if isinstance(value, str) else value
    if not isinstance(record, dict):
        raise ValueError("OCSF input must be a JSON object")
    for field in ("class_uid", "category_uid", "activity_id", "severity_id", "time", "metadata"):
        if field not in record:
            raise ValueError(f"OCSF input is missing required base field: {field}")
    return record


def ocsf_to_event(value: str | dict[str, Any]) -> Event:
    """Map a single OCSF Base Event into SATARK's canonical event model."""
    record = parse_ocsf(value)
    metadata = _mapping(record.get("metadata"))
    product = _mapping(metadata.get("product"))
    actor = _mapping(record.get("actor"))
    user = _mapping(actor.get("user"))
    device = _mapping(record.get("device"))
    source = _mapping(record.get("src_endpoint"))
    destination = _mapping(record.get("dst_endpoint"))
    timestamp = _timestamp(record["time"])
    action = record.get("activity_name") or record.get("class_name") or record.get("message")
    return Event(
        category=_category(record),
        source=str(product.get("name") or metadata.get("product_name") or "ocsf"),
        timestamp=timestamp,
        actor=_string(user.get("name") or actor.get("name")),
        target=_string(destination.get("hostname") or destination.get("ip")),
        host=_string(device.get("hostname") or device.get("name")),
        action=_string(action),
        attributes={
            "ocsf_class_uid": record["class_uid"],
            "ocsf_category_uid": record["category_uid"],
            "ocsf_activity_id": record["activity_id"],
            "ocsf_severity_id": record["severity_id"],
            "source_ip": source.get("ip"),
            "destination_ip": destination.get("ip"),
            "message": record.get("message"),
            "unmapped": record.get("unmapped", {}),
        },
        tags=["ocsf", *_tags(record)],
        raw=record,
    )


def load_ocsf_events(path: Path) -> list[Event]:
    """Load an OCSF JSON object, JSON array, or JSONL file into SATARK events."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
    return [ocsf_to_event(record) for record in records]


def _category(record: dict[str, Any]) -> EventCategory:
    text = " ".join(
        str(record.get(key, "")) for key in ("class_name", "activity_name", "message")
    ).lower()
    mappings = (
        ("authentication", EventCategory.AUTHENTICATION),
        ("login", EventCategory.LOGIN),
        ("process", EventCategory.PROCESS_EXECUTION),
        ("file", EventCategory.FILE_ACCESS),
        ("network", EventCategory.NETWORK_CONNECTION),
        ("email", EventCategory.EMAIL_RECEIVED),
        ("web", EventCategory.WEB_REQUEST),
        ("dns", EventCategory.DNS_QUERY),
        ("api", EventCategory.CLOUD_API_CALL),
    )
    return next(
        (category for keyword, category in mappings if keyword in text),
        EventCategory.CUSTOM,
    )


def _timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError("OCSF time must be epoch milliseconds or an ISO-8601 timestamp")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _tags(record: dict[str, Any]) -> list[str]:
    class_name = record.get("class_name")
    return [str(class_name).lower().replace(" ", "_")] if class_name else []
