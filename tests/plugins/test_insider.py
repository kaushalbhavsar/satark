"""Tests for plugin contract behavior."""

from satark.core.plugin import PluginContext
from satark.plugins.insider import InsiderThreatPlugin
from satark.plugins.malware import MalwarePlugin


def test_insider_plugin_pipeline() -> None:
    plugin = InsiderThreatPlugin(usb_spike_threshold=2.0, file_spike_threshold=2.0)
    records = [
        {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "user": "bob",
            "usb_events": 1,
            "file_reads": 1,
            "file_writes": 0,
        },
        {
            "timestamp": "2024-01-01T01:00:00+00:00",
            "user": "bob",
            "usb_events": 1,
            "file_reads": 1,
            "file_writes": 0,
        },
        {
            "timestamp": "2024-01-01T02:00:00+00:00",
            "user": "bob",
            "usb_events": 8,
            "file_reads": 20,
            "file_writes": 10,
        },
    ]
    findings = plugin.run(PluginContext())
    # collect returns [] without data_path; normalize via manual call
    events = plugin.normalize(records, PluginContext())
    detections = plugin.detect(events, PluginContext())
    assert detections
    score = plugin.score(detections[0], events, PluginContext())
    assert 0.0 <= score.value <= 1.0
    assert score.confidence > 0
    assert findings == []  # no data_path collect


def test_malware_plugin_tag_match() -> None:
    plugin = MalwarePlugin()
    events = plugin.normalize(
        [{"category": "process_execution", "source": "edr", "tags": "malware", "actor": "svc"}],
        PluginContext(),
    )
    detections = plugin.detect(events, PluginContext())
    assert len(detections) == 1
    assert detections[0].plugin == "malware"
