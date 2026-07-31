"""Prompt templates for AI-assisted workflows."""

from __future__ import annotations

FINDING_SUMMARY_PROMPT = """\
You are a security analyst assistant. Summarize the finding briefly.
Title: {title}
Severity: {severity}
Risk: {risk}
Reasoning: {reasoning}
"""

EXPLANATION_PROMPT = """\
Explain why this detection may indicate malicious activity.
Detection: {title}
Evidence: {evidence}
Factors: {factors}
Keep the explanation factual; do not invent telemetry.
"""


def render_prompt(template: str, **kwargs: str) -> str:
    """Render a prompt template with keyword substitutions."""
    return template.format(**kwargs)
