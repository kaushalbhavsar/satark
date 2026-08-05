"""Phishing analytics plugin stub."""

from satark.core.events import EventCategory
from satark.core.models.detection import DetectionSeverity
from satark.plugins._heuristic import HeuristicDomainPlugin


class PhishingPlugin(HeuristicDomainPlugin):
    """Detect phishing-oriented email and web signals."""

    def __init__(self) -> None:
        super().__init__(
            name="phishing",
            domain="phishing",
            description="Phishing email and lure indicators",
            watched_categories={EventCategory.EMAIL_RECEIVED, EventCategory.WEB_REQUEST},
            watched_tags={"phishing", "spearphish"},
            rule_id="phishing.heuristic",
            title="Potential phishing activity",
            severity=DetectionSeverity.HIGH,
        )
