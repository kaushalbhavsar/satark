"""Knowledge-base reference models (MITRE, CAPEC, CVE, CWE)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class KnowledgeSource(StrEnum):
    """Supported knowledge providers."""

    MITRE_ATTACK = "mitre_attack"
    MITRE_D3FEND = "mitre_d3fend"
    CAPEC = "capec"
    CVE = "cve"
    CWE = "cwe"
    CUSTOM = "custom"


class KnowledgeReference(BaseModel):
    """A reference to an external knowledge-base entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: KnowledgeSource
    identifier: str = Field(description="e.g. T1059, CAPEC-112, CVE-2024-1234")
    name: str | None = None
    url: HttpUrl | None = None
    version: str | None = Field(default=None, description="Provider dataset version")
