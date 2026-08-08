"""MITRE ATT&CK knowledge provider with a curated seed set."""

from __future__ import annotations

from satark.core.models.knowledge_ref import KnowledgeSource
from satark.knowledge import KnowledgeEntry, StaticKnowledgeProvider

_SEED: list[KnowledgeEntry] = [
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_ATTACK,
        identifier="T1059",
        name="Command and Scripting Interpreter",
        description="Adversaries may abuse command interpreters to execute commands.",
        url="https://attack.mitre.org/techniques/T1059/",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_ATTACK,
        identifier="T1091",
        name="Replication Through Removable Media",
        description="Adversaries may move onto systems by copying malware to removable media.",
        url="https://attack.mitre.org/techniques/T1091/",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_ATTACK,
        identifier="T1020",
        name="Automated Exfiltration",
        description="Adversaries may exfiltrate data using automated processing.",
        url="https://attack.mitre.org/techniques/T1020/",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_ATTACK,
        identifier="T1078",
        name="Valid Accounts",
        description="Adversaries may obtain and abuse credentials of existing accounts.",
        url="https://attack.mitre.org/techniques/T1078/",
    ),
    KnowledgeEntry(
        source=KnowledgeSource.MITRE_ATTACK,
        identifier="T1566",
        name="Phishing",
        description="Adversaries may send phishing messages to gain access.",
        url="https://attack.mitre.org/techniques/T1566/",
    ),
]


def default_attack_provider(*, version: str = "16.0") -> StaticKnowledgeProvider:
    """Return a seeded MITRE ATT&CK provider."""
    return StaticKnowledgeProvider(KnowledgeSource.MITRE_ATTACK, _SEED, version=version)
