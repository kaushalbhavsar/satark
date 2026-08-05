"""Plugin package: domain-specific analytics modules."""

from satark.plugins.insider import InsiderThreatPlugin
from satark.plugins.registry import builtin_plugins, create_plugin

__all__ = ["InsiderThreatPlugin", "builtin_plugins", "create_plugin"]
