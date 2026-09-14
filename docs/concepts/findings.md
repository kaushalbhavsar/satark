# Findings

A `Finding` is the end-to-end output an analyst sees: a reproducible detection,
an explainable score, and an explanation. Findings are immutable Pydantic
models and can be serialized with `model_dump(mode="json")` or
`model_dump_json()`.

## Composition

| Piece | Role |
|-------|------|
| `Detection` | Reproducible signal from `detect()` (no AI required) |
| `ScoreBreakdown` | Transparent risk with factors, evidence, confidence, reasoning, references |
| `explanation` | Human-readable narrative |
| `recommendations` | Optional next steps (may be AI-assisted) |
| `ai_assisted` | Flag set only when AI enrichment was applied |

## Flow and use

```text
Events → Detection → ScoreBreakdown → Finding
```

Use `satark.scoring.prioritize` to order findings by severity, risk, and
confidence. `AnalysisResult.elevated` is a filtered subset based on the
engine's configured risk threshold; a non-elevated finding is still a finding
worth retaining for review or correlation.

```python
from satark.scoring.prioritization import prioritize

for finding in prioritize(result.findings):
    print(finding.detection.severity, finding.score.value, finding.explanation)
```

## Evidence and references

A `Detection` points to source event IDs and can include `Evidence` objects.
Evidence has a kind, readable summary, optional event ID, structured details,
and a weight from 0 to 1. Scores may reference MITRE ATT&CK, D3FEND, CAPEC,
CVE, CWE, or a custom provider through `KnowledgeReference`.

## AI boundary

AI may enrich explanations or recommendations. When it does, `ai_assisted` is
set to `True`; the underlying detection and score still remain reproducible with
AI disabled.
