# Scoring & Explainability

SATARK answers: **“Why was this event classified as malicious?”**

A `ScoreBreakdown` always includes:

- `value` — risk in `[0, 1]`
- `confidence` — confidence in `[0, 1]`
- `factors` — named contributions with descriptions and evidence
- `evidence` — supporting telemetry / statistics
- `reasoning` — narrative explanation
- `references` — knowledge-base mappings

Use `satark.scoring.why_malicious()` for a concise answer and `format_explanation()` for a full report.
