# Knowledge Mapping Guide

Attach knowledge references when scoring so findings map to industry frameworks.

## Pattern

```python
from satark.knowledge.mitre_attack import default_attack_provider
from satark.core.models.knowledge_ref import KnowledgeReference, KnowledgeSource

attack = default_attack_provider()
technique = attack.get("T1566")

references = []
if technique is not None:
    references.append(technique.as_reference())
else:
    references.append(
        KnowledgeReference(
            source=KnowledgeSource.MITRE_ATTACK,
            identifier="T1566",
            name="Phishing",
        )
    )

score = aggregate_score(
    factors,
    confidence=0.7,
    reasoning="Phishing lure matched selection criteria.",
    references=references,
)
```

## Tips

- Prefer provider lookups so `version` and `url` stay consistent.
- Include multiple frameworks when useful (ATT&CK + D3FEND + CWE).
- Keep identifiers canonical (`T1059`, `CAPEC-98`, `CWE-79`, `CVE-2024-1234`).
- CVE provider starts empty — load your own dataset for production use.
