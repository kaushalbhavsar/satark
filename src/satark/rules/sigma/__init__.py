"""Sigma rule engine placeholder for SIEM-style detections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from satark.core.events import Event
from satark.rules.regex import RuleEngine, RuleMatch


class SigmaRuleEngine(RuleEngine):
    """Minimal Sigma-like field equality matcher for research use."""

    def __init__(self, rules: Sequence[dict[str, Any]] | None = None) -> None:
        self._rules = list(rules or [])

    def add_rule(self, rule_id: str, title: str, selection: dict[str, Any]) -> None:
        """Add a rule that matches when all selection field/value pairs equal."""
        self._rules.append({"id": rule_id, "title": title, "selection": selection})

    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for rule in self._rules:
            selection: dict[str, Any] = rule["selection"]
            for event in events:
                ok = True
                for key, expected in selection.items():
                    actual = event.attributes.get(key, getattr(event, key, None))
                    if actual != expected:
                        ok = False
                        break
                if ok:
                    matches.append(
                        RuleMatch(
                            rule_id=rule["id"],
                            rule_type="sigma",
                            title=rule["title"],
                            event_ids=(str(event.id),),
                            details={"selection": selection},
                        )
                    )
        return matches
