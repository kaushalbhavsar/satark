"""Rule matching abstractions (YARA, Sigma, regex, STIX, custom)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from satark.core.events import Event


@dataclass(frozen=True)
class RuleMatch:
    """Result of applying a rule to one or more events."""

    rule_id: str
    rule_type: str
    title: str
    event_ids: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)


class RuleEngine(ABC):
    """Abstract rule engine interface."""

    @abstractmethod
    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        """Evaluate rules against events and return matches."""


class RegexRuleEngine(RuleEngine):
    """Simple attribute/value regex rule engine."""

    def __init__(self, rules: Sequence[dict[str, Any]] | None = None) -> None:
        import re

        self._re = re
        self._rules = list(rules or [])

    def add_rule(
        self,
        rule_id: str,
        title: str,
        field_name: str,
        pattern: str,
    ) -> None:
        """Register a regex rule against an event attribute or top-level field."""
        self._rules.append(
            {
                "id": rule_id,
                "title": title,
                "field": field_name,
                "pattern": pattern,
            }
        )

    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for rule in self._rules:
            compiled = self._re.compile(rule["pattern"])
            for event in events:
                value = event.attributes.get(rule["field"])
                if value is None:
                    value = getattr(event, rule["field"], None)
                if value is not None and compiled.search(str(value)):
                    matches.append(
                        RuleMatch(
                            rule_id=rule["id"],
                            rule_type="regex",
                            title=rule["title"],
                            event_ids=(str(event.id),),
                            details={"field": rule["field"], "value": str(value)},
                        )
                    )
        return matches
