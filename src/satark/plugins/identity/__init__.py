"""Identity security analytics plugin stub."""

from satark.core.events import EventCategory
from satark.core.models.detection import DetectionSeverity
from satark.plugins._heuristic import HeuristicDomainPlugin


class IdentityPlugin(HeuristicDomainPlugin):
    """Detect suspicious authentication and identity signals."""

    def __init__(self) -> None:
        super().__init__(
            name="identity",
            domain="identity",
            description="Authentication anomalies and identity abuse",
            watched_categories={EventCategory.LOGIN, EventCategory.AUTHENTICATION},
            watched_tags={"identity", "bruteforce", "mfa-bypass"},
            rule_id="identity.heuristic",
            title="Suspicious identity activity",
            severity=DetectionSeverity.HIGH,
        )
