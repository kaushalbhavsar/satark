# Scoring & Explainability

SATARK answers: **“Why was this event classified as malicious?”**

## ScoreBreakdown

| Field | Meaning |
|-------|---------|
| `value` | Overall risk in `[0, 1]` |
| `confidence` | Confidence in `[0, 1]` |
| `factors` | Named signed contributions with descriptions/evidence |
| `evidence` | Supporting telemetry / statistics / rule matches |
| `reasoning` | Narrative explanation |
| `references` | Knowledge-base mappings |

## Helpers

```python
from satark.scoring import (
    aggregate_score,
    evidence_confidence,
    format_explanation,
    prioritize,
    why_malicious,
)

score = aggregate_score(
    factors,
    confidence=evidence_confidence(evidence),
    reasoning="USB volume exceeded actor baseline.",
    evidence=evidence,
    references=refs,
)

print(why_malicious(detection, score))
print(format_explanation(detection, score))

ordered = prioritize(findings)
```

## Prioritization

Findings are sorted by severity rank, then risk, then confidence. Use `priority_score(finding)` for a single combined metric.

See [Scoring API](../api/scoring.md).
