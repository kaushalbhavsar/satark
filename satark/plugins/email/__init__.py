"""Email security analytics plugin stub."""

from satark.core.events import EventCategory
from satark.core.models.detection import DetectionSeverity
from satark.plugins._heuristic import HeuristicDomainPlugin


class EmailPlugin(HeuristicDomainPlugin):
    """Detect suspicious email activity."""

    def __init__(self) -> None:
        super().__init__(
            name="email",
            domain="email",
            description="Email-borne threat signals",
            watched_categories={EventCategory.EMAIL_RECEIVED},
            watched_tags={"email", "bEC", "spoof"},
            rule_id="email.heuristic",
            title="Suspicious email activity",
            severity=DetectionSeverity.MEDIUM,
        )
