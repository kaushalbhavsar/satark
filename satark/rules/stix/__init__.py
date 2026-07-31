"""STIX pattern helpers (lightweight stub for research workflows)."""

from __future__ import annotations

from collections.abc import Sequence

from satark.core.events import Event
from satark.rules.regex import RuleEngine, RuleMatch


class StixRuleEngine(RuleEngine):
    """Placeholder STIX engine matching on event category tags."""

    def __init__(self, indicators: Sequence[str] | None = None) -> None:
        self._indicators = list(indicators or [])

    def add_indicator(self, indicator: str) -> None:
        """Register a simple indicator string matched against event tags."""
        self._indicators.append(indicator)

    def match(self, events: Sequence[Event]) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for event in events:
            for indicator in self._indicators:
                if indicator in event.tags or indicator == event.category.value:
                    matches.append(
                        RuleMatch(
                            rule_id=f"stix:{indicator}",
                            rule_type="stix",
                            title=f"STIX indicator match: {indicator}",
                            event_ids=(str(event.id),),
                            details={"indicator": indicator},
                        )
                    )
        return matches
