# Architecture Overview

SATARK separates a **domain-agnostic core** from **domain plugins**.

```
┌─────────────────────────────────────────────────────────┐
│                     AnalysisEngine                       │
│  storage · pipelines · config · CLI                      │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
        ┌───────▼───────┐         ┌───────▼───────┐
        │   Scoring     │         │    Graph      │
        │ risk/confidence│        │ entities/paths│
        └───────────────┘         └───────────────┘
                │
        ┌───────▼───────────────────────────────────────┐
        │ Plugins: insider · malware · phishing · …     │
        │ collect → normalize → detect → score → explain│
        └───────────────────────────────────────────────┘
                │
        ┌───────▼───────┐    ┌──────────────┐
        │ Rules engines │    │ Knowledge    │
        │ regex/sigma/… │    │ ATT&CK/CWE…  │
        └───────────────┘    └──────────────┘
```

## Packages

| Package | Responsibility |
|---------|----------------|
| `satark.core` | Events, engine, plugin contract, pipelines, storage, CLI |
| `satark.scoring` | Transparent risk, confidence, prioritization, explainability |
| `satark.graph` | Entity correlation, timelines, attack paths |
| `satark.rules` | Rule engines (regex, Sigma-like, STIX stub, YARA stub, custom) |
| `satark.ai` | Optional LLM assistants (never source of truth) |
| `satark.knowledge` | Replaceable MITRE/CAPEC/CVE/CWE providers |
| `satark.plugins` | Domain analytics modules |
