# Plugins Overview

Domain plugins live under `satark.plugins` and are registered in `satark.plugins.registry`.

## Built-in plugins

| Name | Class | Status | Notes |
|------|-------|--------|-------|
| `insider` | `InsiderThreatPlugin` | Implemented | USB/file volume spikes, ATT&CK mapping |
| `malware` | `MalwarePlugin` | Heuristic stub | Process/file + malware tags |
| `phishing` | `PhishingPlugin` | Heuristic stub | Email/web + phishing tags |
| `web` | `WebPlugin` | Heuristic stub | Web requests / XSS/SQLi tags |
| `email` | `EmailPlugin` | Heuristic stub | Email-borne signals |
| `cloud` | `CloudPlugin` | Heuristic stub | Cloud API abuse signals |
| `identity` | `IdentityPlugin` | Heuristic stub | Login / auth anomalies |

## Factory

```python
from satark.plugins import builtin_plugins, create_plugin

print(builtin_plugins())
plugin = create_plugin("insider")
```

## Independence rule

Plugins must not depend on other plugins. Share only:

- `satark.core` models and contracts
- `satark.scoring` helpers
- `satark.rules` engines
- `satark.knowledge` providers
- `satark.graph` utilities

See [Writing a Plugin](../guides/writing-a-plugin.md).
