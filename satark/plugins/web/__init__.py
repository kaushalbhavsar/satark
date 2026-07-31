"""Web security analytics plugin stub."""

from satark.core.events import EventCategory
from satark.core.models.detection import DetectionSeverity
from satark.plugins._heuristic import HeuristicDomainPlugin


class WebPlugin(HeuristicDomainPlugin):
    """Detect suspicious web request patterns."""

    def __init__(self) -> None:
        super().__init__(
            name="web",
            domain="web",
            description="Web application attack and abuse signals",
            watched_categories={EventCategory.WEB_REQUEST},
            watched_tags={"xss", "sqli", "web-attack"},
            rule_id="web.heuristic",
            title="Suspicious web activity",
            severity=DetectionSeverity.MEDIUM,
        )
