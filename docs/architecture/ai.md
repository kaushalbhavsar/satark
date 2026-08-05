# AI Integration

AI is an **assistant**, not the source of truth.

## Allowed uses

- Summarization
- Investigation assistance
- Report generation
- Detection explanation enrichment
- Analyst recommendations

## Hard rule

**Detections must always be reproducible without AI.**

`detect()` and core scoring must work with `SATARK_ENABLE_AI=false` (default).

## Usage

```python
from satark.ai import InvestigationAgent, EchoLLM, enrich_explanation

agent = InvestigationAgent(client=EchoLLM(), enabled=True)
enriched = enrich_explanation(finding, agent)
assert enriched.ai_assisted is True
```

When AI is disabled, `enrich_explanation` returns the original finding unchanged.

See [AI API](../api/ai.md) and [AI assistant guide](../guides/ai-assistants.md).
