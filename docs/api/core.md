# Core API

The `satark.core` package is the domain-agnostic foundation.

## Subpages

| Page | Contents |
|------|----------|
| [Events](events.md) | `Event`, `EventCategory` |
| [Engine](engine.md) | `AnalysisEngine`, pipelines |
| [Plugin](plugin.md) | Plugin contract types |
| [Models](models.md) | Detection, evidence, score, finding |
| [Storage](storage.md) | Event stores |
| [Config](config.md) | Settings |

## Package exports

```python
from satark.core import (
    AnalysisEngine,
    AnalysisResult,
    Plugin,
    PluginContext,
    PluginMeta,
)
```
