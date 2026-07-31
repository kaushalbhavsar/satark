"""YARA rule engine placeholder.

YARA scanning is optional and requires the ``yara-python`` package at runtime.
This module provides a stub interface so plugins can depend on a stable API.
"""

from __future__ import annotations

from collections.abc import Sequence

from satark.core.events import Event
from satark.rules.regex import RuleEngine, RuleMatch


class YaraRuleEngine(RuleEngine):
    """Stub YARA engine — raise if used without yara-python installed."""

    def __init__(self, rules_path: str | None = None) -> None:
        self.rules_path = rules_path

    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        """Match events against YARA rules (not implemented without yara-python)."""
        msg = (
            "YaraRuleEngine requires the optional 'yara-python' dependency. "
            "Install it and provide compiled rules to enable scanning."
        )
        raise NotImplementedError(msg)
