# Contributing

Thanks for helping grow SATARK.

## Development setup

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest
```

## Contribution types we welcome

- New domain plugins
- Knowledge provider backends / dataset loaders
- Rule engines and detection content
- Scoring / explainability improvements
- Docs and runnable examples
- Tests and typing fixes

## Pull request expectations

1. Typed public APIs
2. Module docs (purpose + examples where useful)
3. Unit tests for new behavior
4. No cross-plugin dependencies
5. Detections remain reproducible without AI

## Coding standards

- Modern Python 3.13+
- Prefer composition over inheritance
- Pydantic models or dataclasses for structured data
- Small, focused functions
- Avoid global state and circular imports

## Docs site

```bash
mkdocs serve
```

Update navigation in `mkdocs.yml` when adding pages.
