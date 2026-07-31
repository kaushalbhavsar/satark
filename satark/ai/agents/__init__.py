"""AI assistants for summarization and explanation — never the source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from satark.core.models.finding import Finding


class LLMClient(Protocol):
    """Minimal protocol for optional LLM backends."""

    def complete(self, prompt: str) -> str:
        """Return a completion for the given prompt."""


@dataclass
class NullLLM:
    """Deterministic no-op LLM used when AI is disabled."""

    def complete(self, prompt: str) -> str:
        return ""


@dataclass
class EchoLLM:
    """Test double that echoes a truncated prompt."""

    max_chars: int = 200

    def complete(self, prompt: str) -> str:
        text = prompt.strip().replace("\n", " ")
        return text[: self.max_chars]


class InvestigationAgent:
    """AI assistant for investigation support.

    Detections remain reproducible without this agent. AI only enriches
    summaries, explanations, and analyst recommendations.
    """

    def __init__(self, client: LLMClient | None = None, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self.client: LLMClient = client or NullLLM()

    def summarize_finding(self, finding: Finding) -> str:
        """Produce an optional AI summary; falls back to deterministic text."""
        base = finding.explanation
        if not self.enabled:
            return base
        prompt = (
            "Summarize this security finding for an analyst.\n"
            f"Title: {finding.detection.title}\n"
            f"Risk: {finding.score.value}\n"
            f"Explanation: {finding.explanation}\n"
        )
        response = self.client.complete(prompt).strip()
        return response or base

    def recommend(self, finding: Finding) -> list[str]:
        """Suggest analyst next steps (AI-assisted when enabled)."""
        defaults = [
            "Validate supporting evidence against original telemetry.",
            "Correlate with related entities in the graph timeline.",
            "Map findings to MITRE ATT&CK techniques for reporting.",
        ]
        if not self.enabled:
            return defaults
        prompt = f"Recommend investigation steps for: {finding.detection.title}"
        response = self.client.complete(prompt).strip()
        if not response:
            return defaults
        return [response, *defaults]
