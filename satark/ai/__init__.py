"""AI integration: assistants only — detections remain reproducible without AI."""

from satark.ai.agents import EchoLLM, InvestigationAgent, NullLLM
from satark.ai.explain import deterministic_explanation, enrich_explanation
from satark.ai.prompts import FINDING_SUMMARY_PROMPT, render_prompt
from satark.ai.rag import Document, InMemoryRetriever

__all__ = [
    "Document",
    "EchoLLM",
    "FINDING_SUMMARY_PROMPT",
    "InMemoryRetriever",
    "InvestigationAgent",
    "NullLLM",
    "deterministic_explanation",
    "enrich_explanation",
    "render_prompt",
]
