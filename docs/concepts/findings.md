# Findings

A **Finding** is the end-to-end output an analyst sees: a detection, an explainable score, and an explanation.

## Composition

| Piece | Role |
|-------|------|
| `Detection` | Reproducible signal from `detect()` (no AI required) |
| `ScoreBreakdown` | Transparent risk with factors, evidence, confidence, reasoning, references |
| `explanation` | Human-readable narrative |
| `recommendations` | Optional next steps (may be AI-assisted) |
| `ai_assisted` | Flag set only when AI enrichment was applied |

## Flow

```text
Events → Detection → ScoreBreakdown → Finding
```

Findings can be prioritized with `satark.scoring.prioritize`.

## AI boundary

AI may enrich explanations or recommendations. The underlying detection and score must remain reproducible with AI disabled.
