"""Plugin registry for built-in SATARK plugins."""

from __future__ import annotations

from collections.abc import Callable

from satark.core.plugin import Plugin
from satark.plugins.cloud import CloudPlugin
from satark.plugins.email import EmailPlugin
from satark.plugins.identity import IdentityPlugin
from satark.plugins.insider import InsiderThreatPlugin
from satark.plugins.malware import MalwarePlugin
from satark.plugins.phishing import PhishingPlugin
from satark.plugins.web import WebPlugin

_REGISTRY: dict[str, Callable[[], Plugin]] = {
    "insider": InsiderThreatPlugin,
    "malware": MalwarePlugin,
    "phishing": PhishingPlugin,
    "web": WebPlugin,
    "email": EmailPlugin,
    "cloud": CloudPlugin,
    "identity": IdentityPlugin,
}


def builtin_plugins() -> list[str]:
    """Return names of built-in plugins."""
    return sorted(_REGISTRY)


def create_plugin(name: str) -> Plugin:
    """Instantiate a built-in plugin by name."""
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(builtin_plugins())
        msg = f"Unknown plugin '{name}'. Available: {available}"
        raise KeyError(msg) from exc
    return factory()
