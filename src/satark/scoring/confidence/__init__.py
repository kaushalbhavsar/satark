"""Confidence estimation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from satark.core.models.evidence import Evidence
from satark.scoring.risk import clamp01


def evidence_confidence(
    evidence: Sequence[Evidence],
    *,
    min_items: int = 1,
    max_items: int = 5,
) -> float:
    """Estimate confidence from evidence quantity and weights.

    More weighted evidence increases confidence, saturating at ``max_items``.
    """
    if not evidence:
        return 0.0
    weight_sum = sum(e.weight for e in evidence)
    count_factor = min(len(evidence), max_items) / max(max_items, 1)
    weight_factor = clamp01(weight_sum / max(min_items, 1))
    return clamp01(0.5 * count_factor + 0.5 * weight_factor)
