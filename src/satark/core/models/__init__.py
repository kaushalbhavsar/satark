"""Shared domain models used across the SATARK core."""

from satark.core.models.detection import Detection, DetectionSeverity
from satark.core.models.evidence import Evidence, EvidenceKind
from satark.core.models.finding import Finding
from satark.core.models.knowledge_ref import KnowledgeReference, KnowledgeSource
from satark.core.models.score import ScoreBreakdown, ScoreFactor

__all__ = [
    "Detection",
    "DetectionSeverity",
    "Evidence",
    "EvidenceKind",
    "Finding",
    "KnowledgeReference",
    "KnowledgeSource",
    "ScoreBreakdown",
    "ScoreFactor",
]
