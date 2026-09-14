"""Plugin package: domain-specific analytics modules."""

from satark.plugins.insider import InsiderThreatPlugin
from satark.plugins.insider.lstm import LstmInsiderDetector
from satark.plugins.registry import builtin_plugins, create_plugin

__all__ = ["InsiderThreatPlugin", "LstmInsiderDetector", "builtin_plugins", "create_plugin"]
