# Knowledge Layer

Findings should map to external knowledge wherever possible.

## Providers

| Provider | Module | Seeded? |
|----------|--------|---------|
| MITRE ATT&CK | `satark.knowledge.mitre_attack` | Yes (curated) |
| MITRE D3FEND | `satark.knowledge.mitre_d3fend` | Yes (curated) |
| CAPEC | `satark.knowledge.capec` | Yes (curated) |
| CWE | `satark.knowledge.cwe` | Yes (curated) |
| CVE | `satark.knowledge.cve` | Empty by default (inject entries) |

Providers are **replaceable** and independently **versioned**.

## Example

```python
from satark.knowledge.mitre_attack import default_attack_provider

attack = default_attack_provider(version="16.0")
entry = attack.get("T1091")
ref = entry.as_reference()  # KnowledgeReference for ScoreBreakdown
```

See [Knowledge API](../api/knowledge.md) and [Mapping findings guide](../guides/knowledge-mapping.md).
