# Scoring

SATARK must **not** output unexplained risk scores.

Every score should include:

- Numerical score (`value` in `[0, 1]`)
- Contributing factors
- Evidence
- Confidence
- Explanation / reasoning
- Relevant knowledge references (for example MITRE ATT&CK, CAPEC, CWE, CVE)

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

## Design rule

If a detector cannot explain *why* a score is elevated, it is not ready for SATARK.
