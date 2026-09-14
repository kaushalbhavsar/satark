"""Parsers that turn log transport formats into a neutral dictionary."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any


class ParseError(ValueError):
    """Raised when an input cannot be parsed as its declared format."""


def _split_escaped(value: str, separator: str) -> list[str]:
    """Split a CEF header without treating escaped separators as delimiters."""
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for character in value:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == separator:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    parts.append("".join(current))
    return parts


def _key_values(value: str, *, delimiter: str | None = None) -> dict[str, str]:
    if delimiter is not None:
        pairs = value.split(delimiter)
        return {
            key.strip(): item.strip()
            for item in pairs
            if "=" in item
            for key, item in [item.split("=", 1)]
        }
    return {key: item for key, item in re.findall(r"(\S+?)=(.*?)(?=\s+\S+=|$)", value)}


def parse_cef(line: str) -> dict[str, Any]:
    """Parse a CEF line into neutral fields, preserving extension key-values."""
    marker = line.find("CEF:")
    if marker < 0:
        raise ParseError("CEF input must contain 'CEF:'")
    prefix, payload = line[:marker].strip(), line[marker + 4 :]
    parts = _split_escaped(payload, "|")
    if len(parts) < 7:
        raise ParseError("CEF input needs seven header fields")
    version, vendor, product, device_version, signature, name, severity, *rest = parts
    extension = "|".join(rest)
    fields = _key_values(extension)
    return {
        "format": "cef",
        "raw": line,
        "transport_prefix": prefix or None,
        "version": version,
        "vendor": vendor,
        "product": product,
        "product_version": device_version,
        "event_id": signature,
        "event_name": name,
        "severity": severity,
        "message": fields.get("msg") or name,
        "timestamp": fields.get("rt") or fields.get("end"),
        "host": fields.get("dhost") or fields.get("shost"),
        "user": fields.get("duser") or fields.get("suser"),
        "source_ip": fields.get("src"),
        "destination_ip": fields.get("dst"),
        "fields": fields,
    }


def parse_leef(line: str) -> dict[str, Any]:
    """Parse LEEF 1.0/2.0 header fields and its delimited extension."""
    marker = line.find("LEEF:")
    if marker < 0:
        raise ParseError("LEEF input must contain 'LEEF:'")
    prefix, payload = line[:marker].strip(), line[marker + 5 :]
    parts = payload.split("|")
    if len(parts) < 5:
        raise ParseError("LEEF input needs five header fields")
    version, vendor, product, product_version, event_id, *rest = parts
    extension = "|".join(rest)
    delimiter = "\t"
    if version.startswith("2.0"):
        if len(extension) < 2 or extension[1] != "|":
            raise ParseError("LEEF 2.0 input must declare an extension delimiter")
        delimiter, extension = extension[0], extension[2:]
    fields = _key_values(extension, delimiter=delimiter)
    return {
        "format": "leef",
        "raw": line,
        "transport_prefix": prefix or None,
        "version": version,
        "vendor": vendor,
        "product": product,
        "product_version": product_version,
        "event_id": event_id,
        "event_name": fields.get("eventname") or event_id,
        "severity": fields.get("sev"),
        "message": fields.get("msg") or fields.get("eventname") or event_id,
        "timestamp": fields.get("devTime") or fields.get("time"),
        "host": fields.get("dev") or fields.get("hostname"),
        "user": fields.get("usrName"),
        "source_ip": fields.get("src"),
        "destination_ip": fields.get("dst"),
        "fields": fields,
    }


_RFC5424 = re.compile(
    r"^(?:<(\d{1,3})>)?(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$"
)
_RFC3164 = re.compile(
    r"^(?:<(\d{1,3})>)?([A-Z][a-z]{2}\s{1,2}\d{1,2}\s\d\d:\d\d:\d\d)\s+(\S+)\s+(.*)$"
)


def parse_syslog(line: str, *, year: int | None = None) -> dict[str, Any]:
    """Parse an RFC 5424 or RFC 3164 syslog line into neutral fields."""
    match_5424 = _RFC5424.match(line)
    if match_5424:
        priority, version, timestamp, host, app, process, message_id, remainder = (
            match_5424.groups()
        )
        structured, message = _structured_data(remainder)
        return _syslog_result(
            line,
            "rfc5424",
            priority,
            timestamp,
            host,
            app,
            process,
            message_id,
            message,
            structured,
            version,
        )
    match_3164 = _RFC3164.match(line)
    if not match_3164:
        raise ParseError("Input is not recognizable RFC 3164 or RFC 5424 syslog")
    priority, timestamp_text, host, message = match_3164.groups()
    parsed = datetime.strptime(timestamp_text, "%b %d %H:%M:%S").replace(
        year=year or datetime.now(UTC).year, tzinfo=UTC
    )
    app, process = _legacy_tag(message)
    return _syslog_result(
        line,
        "rfc3164",
        priority,
        parsed.isoformat(),
        host,
        app,
        process,
        None,
        message,
        {},
        None,
    )


def _structured_data(value: str) -> tuple[dict[str, dict[str, str]], str]:
    if value == "-":
        return {}, ""
    if not value.startswith("["):
        return {}, value
    blocks = re.findall(r"\[([^\]]+)\]", value)
    end = value.rfind("]") + 1
    structured: dict[str, dict[str, str]] = {}
    for block in blocks:
        identifier, *pairs = block.split(" ")
        structured[identifier] = _key_values(" ".join(pairs).replace('"', ""))
    return structured, value[end:].lstrip()


def _legacy_tag(message: str) -> tuple[str | None, str | None]:
    match = re.match(r"([^:\[]+)(?:\[(\d+)\])?:", message)
    return (match.group(1), match.group(2)) if match else (None, None)


def _syslog_result(
    raw: str,
    fmt: str,
    priority: str | None,
    timestamp: str,
    host: str,
    app: str | None,
    process: str | None,
    message_id: str | None,
    message: str,
    structured: dict[str, Any],
    version: str | None,
) -> dict[str, Any]:
    priority_value = int(priority) if priority is not None else None
    return {
        "format": "syslog",
        "syslog_format": fmt,
        "raw": raw,
        "timestamp": timestamp,
        "host": host,
        "app": app,
        "process_id": process,
        "event_id": message_id,
        "message": message,
        "severity": priority_value % 8 if priority_value is not None else None,
        "facility": priority_value // 8 if priority_value is not None else None,
        "priority": priority_value,
        "version": version,
        "structured_data": structured,
        "fields": {},
    }


def parse_json(value: str | dict[str, Any]) -> dict[str, Any]:
    """Accept nested JSON and retain it as source fields for projection."""
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ParseError("JSON input must be an object")
    return {
        "format": "json",
        "raw": json.dumps(payload, separators=(",", ":")),
        "timestamp": _nested(payload, "@timestamp", "timestamp", "time"),
        "host": _nested(payload, "host.name", "hostname", "host"),
        "user": _nested(payload, "user.name", "user", "username"),
        "source_ip": _nested(payload, "source.ip", "src_ip", "src"),
        "destination_ip": _nested(payload, "destination.ip", "dst_ip", "dst"),
        "event_id": _nested(payload, "event.id", "id"),
        "event_name": _nested(payload, "event.action", "action", "event_type"),
        "severity": _nested(payload, "event.severity", "severity"),
        "message": _nested(payload, "message", "msg"),
        "fields": payload,
    }


def _nested(value: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = value
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None
