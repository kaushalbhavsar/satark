# Contributing to SATARK

Thanks for helping grow SATARK.

## Setup

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
uv sync --group dev --group docs
uv run pytest
```

## Expectations

1. Typed public APIs
2. Unit tests for new behavior
3. No cross-plugin dependencies
4. Detections remain reproducible without AI
5. Docs updates when behavior changes

## Pull requests

- Keep changes focused
- Prefer composition over inheritance
- Avoid unexplained risk scores
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

See [docs/concepts/plugins.md](docs/concepts/plugins.md) for the plugin contract.
