# Plugins

Domain plugins live under `src/satark/plugins/` and are registered in
`satark.plugins.registry`. Every plugin accepts raw records, normalizes them
into canonical events, produces detections, attaches score factors, and creates
an explanation.

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

## Current heuristic plugins

The non-insider plugins share `HeuristicDomainPlugin`. They emit one detection
when an event matches any configured category or tag, add evidence for each
matched event, and produce a medium-confidence heuristic score. They do not
parse vendor-native formats, correlate sequences, or inspect payload content.

Use these plugins to exercise the framework or as small starting points for a
production-quality plugin. Their current rules are documented on the individual
pages and should not be interpreted as comprehensive coverage.
