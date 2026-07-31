"""Tests for the analysis engine and plugin registry."""

from satark.core.engine import AnalysisEngine
from satark.core.events import Event, EventCategory
from satark.plugins import builtin_plugins, create_plugin


def test_builtin_plugins_registered() -> None:
    names = builtin_plugins()
    assert "insider" in names
    assert "malware" in names
    assert len(names) == 7


def test_engine_ingest_and_analyze_insider() -> None:
    plugin = create_plugin("insider")
    engine = AnalysisEngine(plugins=[plugin])
    records = [
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "user": "alice",
            "host": "ws1",
            "usb_events": "1",
            "file_reads": "2",
            "file_writes": "1",
        },
        {
            "timestamp": "2024-01-01T01:00:00+00:00",
            "user": "alice",
            "host": "ws1",
            "usb_events": "1",
            "file_reads": "2",
            "file_writes": "1",
        },
        {
            "timestamp": "2024-01-01T02:00:00+00:00",
            "user": "alice",
            "host": "ws1",
            "usb_events": "10",
            "file_reads": "50",
            "file_writes": "30",
        },
    ]
    events = engine.ingest_raw("insider", records)
    assert len(events) >= 3
    result = engine.analyze(plugin_name="insider", events=events)
    assert result.events_processed == len(events)
    assert result.findings
    finding = result.findings[0]
    assert finding.score.reasoning
    assert finding.score.factors
    assert finding.explanation


def test_engine_rejects_duplicate_plugin() -> None:
    plugin = create_plugin("web")
    engine = AnalysisEngine(plugins=[plugin])
    try:
        engine.register(create_plugin("web"))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_analyze_empty_without_plugins() -> None:
    engine = AnalysisEngine()
    event = Event(category=EventCategory.DNS_QUERY, source="resolver")
    engine.ingest([event])
    result = engine.analyze()
    assert result.events_processed == 1
    assert result.findings == []
