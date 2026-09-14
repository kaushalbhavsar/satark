"""Parse common security log formats and project them into ECS or OCSF envelopes.

The adapters intentionally preserve unmapped fields. They do not claim full
event-class validation: selecting an OCSF event class remains source-specific.
"""

from satark.ingestion.formats import ParseError, parse_cef, parse_json, parse_leef, parse_syslog
from satark.ingestion.ocsf import load_ocsf_events, ocsf_to_event, parse_ocsf
from satark.ingestion.schemas import to_ecs, to_ocsf

__all__ = [
    "ParseError",
    "parse_cef",
    "parse_json",
    "parse_leef",
    "parse_ocsf",
    "parse_syslog",
    "ocsf_to_event",
    "load_ocsf_events",
    "to_ecs",
    "to_ocsf",
]
