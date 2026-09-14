# Scoring

SATARK does not treat a risk score as a verdict. A `ScoreBreakdown` always
contains a bounded risk value, confidence, factors, evidence, reasoning, and
optional knowledge references.

Every score should include:

- Numerical score (`value` in `[0, 1]`)
- Contributing factors
- Evidence
- Confidence
- Explanation / reasoning
- Relevant knowledge references (for example MITRE ATT&CK, CAPEC, CWE, CVE)

## How aggregation works

`aggregate_score()` adds a baseline to signed factor contributions and clamps
the result to `[0, 1]`. Positive factors raise risk and negative factors reduce
it. The helper does not infer probability or calibrate a model; it records a
transparent policy chosen by the plugin author.

`evidence_confidence()` combines evidence count and weight, also clamped to
`[0, 1]`. It measures the amount of supporting material, not the truth of a
detection.

## Implemented helpers

```python
from satark.scoring import aggregate_score, why_malicious, format_explanation

score = aggregate_score(
    factors,
    confidence=0.7,
    reasoning="USB volume exceeded the actor baseline.",
    evidence=evidence,
    references=references,
)

print(why_malicious(detection, score))
print(format_explanation(detection, score))
```

`prioritize()` sorts by detection severity first, then risk and confidence.
`priority_score()` provides a simple combined value for queues; use the full
finding when an analyst needs to understand why it appears in that order.

## Writing good factors

Use one factor per meaningful contribution. Give it a stable name, a bounded
contribution, a description with the measured value, and the supporting
evidence. Avoid hiding a complex model behind one opaque factor. If a model is
used, record its output, threshold, feature set, and artifact version in
evidence details.

## Design rule

If a detector cannot explain why a score is elevated, it is not ready for
SATARK.
