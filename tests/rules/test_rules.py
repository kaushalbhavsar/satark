"""Tests for rule engines."""

from satark.core.events import Event, EventCategory
from satark.rules import CustomRuleEngine, RegexRuleEngine, SigmaRuleEngine


def test_regex_rule_engine() -> None:
    engine = RegexRuleEngine()
    engine.add_rule("r1", "Suspicious process", "action", r"powershell|cmd")
    events = [
        Event(category=EventCategory.PROCESS_EXECUTION, source="edr", action="powershell.exe"),
        Event(category=EventCategory.PROCESS_EXECUTION, source="edr", action="notepad.exe"),
    ]
    matches = engine.match(events)
    assert len(matches) == 1
    assert matches[0].rule_id == "r1"


def test_sigma_and_custom() -> None:
    sigma = SigmaRuleEngine()
    sigma.add_rule("s1", "Login alice", {"actor": "alice", "category": EventCategory.LOGIN})
    event = Event(category=EventCategory.LOGIN, source="idp", actor="alice")
    assert sigma.match([event])

    custom = CustomRuleEngine()
    custom.add_rule("c1", "Has host", lambda e: e.host is not None)
    assert not custom.match([event])
    hosted = event.model_copy(update={"host": "ws1"})
    assert custom.match([hosted])
