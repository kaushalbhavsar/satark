# Plugins

Domain plugins live under `src/satark/plugins/` and are registered in `satark.plugins.registry`.

## Built-in plugins

| Plugin | Domain | Docs |
|--------|--------|------|
| `insider` | Insider threats | [insider](insider.md) |
| `malware` | Malware | [malware](malware.md) |
| `phishing` | Phishing | [phishing](phishing.md) |
| `web` | Web | [web](web.md) |
| `email` | Email | [email](email.md) |
| `cloud` | Cloud | [cloud](cloud.md) |
| `identity` | Identity | [identity](identity.md) |

## Contract reminder

See [Plugin contract](../concepts/plugins.md) for the `collect → normalize → detect → score → explain` lifecycle and independence rules.
