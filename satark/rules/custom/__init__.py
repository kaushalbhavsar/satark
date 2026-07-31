"""Custom callable-based rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from satark.core.events import Event
from satark.rules.regex import RuleEngine, RuleMatch

EventPredicate = Callable[[Event], bool]


class CustomRuleEngine(RuleEngine):
    """Rule engine backed by arbitrary Python predicates."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, str, EventPredicate]] = []

    def add_rule(self, rule_id: str, title: str, predicate: EventPredicate) -> None:
        """Register a custom predicate rule."""
        self._rules.append((rule_id, title, predicate))

    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for rule_id, title, predicate in self._rules:
            for event in events:
                if predicate(event):
                    matches.append(
                        RuleMatch(
                            rule_id=rule_id,
                            rule_type="custom",
                            title=title,
                            event_ids=(str(event.id),),
                        )
                    )
        return matches
