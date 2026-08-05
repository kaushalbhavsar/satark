# AI Assistants Guide

Use AI only to enrich explanations and recommendations.

## Enable explicitly

```python
from satark.ai import InvestigationAgent, EchoLLM, enrich_explanation
from satark.core.config import load_settings

settings = load_settings(enable_ai=True)
agent = InvestigationAgent(client=EchoLLM(), enabled=settings.enable_ai)

finding = ...  # from plugin/engine
enriched = enrich_explanation(finding, agent)
print(enriched.explanation)
print(enriched.recommendations)
```

## Bring your own LLM

Implement a client with a `complete(prompt: str) -> str` method:

```python
class MyClient:
    def complete(self, prompt: str) -> str:
        return call_my_provider(prompt)

agent = InvestigationAgent(client=MyClient(), enabled=True)
```

## Safety checklist

- [ ] `detect()` works with AI disabled
- [ ] Scores do not depend on LLM output
- [ ] `Finding.ai_assisted` is set only when enrichment ran
- [ ] Prompts do not invent telemetry not present in evidence
