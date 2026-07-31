"""Cloud security analytics plugin stub."""

from satark.core.events import EventCategory
from satark.core.models.detection import DetectionSeverity
from satark.plugins._heuristic import HeuristicDomainPlugin


class CloudPlugin(HeuristicDomainPlugin):
    """Detect suspicious cloud API activity."""

    def __init__(self) -> None:
        super().__init__(
            name="cloud",
            domain="cloud",
            description="Cloud API abuse and misconfiguration signals",
            watched_categories={EventCategory.CLOUD_API_CALL},
            watched_tags={"cloud", "iam-abuse", "exfil"},
            rule_id="cloud.heuristic",
            title="Suspicious cloud API activity",
            severity=DetectionSeverity.HIGH,
        )
