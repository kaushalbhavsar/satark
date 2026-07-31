"""Core engine, event model, pipelines, storage, and CLI.

Everything in SATARK is an Event. The core package provides a domain-agnostic
foundation that plugins build upon without exposing vendor-specific formats.
"""

from satark.core.engine import AnalysisEngine, AnalysisResult
from satark.core.plugin import Plugin, PluginContext, PluginMeta

__all__ = [
    "AnalysisEngine",
    "AnalysisResult",
    "Plugin",
    "PluginContext",
    "PluginMeta",
]
