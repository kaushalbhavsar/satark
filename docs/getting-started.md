# Getting started

## Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended for this repository)

## Install

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
uv sync --group docs --group dev
```

Or with pip and venv:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify

```bash
uv run satark version
uv run satark list-plugins
uv run pytest
```

## Analyze sample insider data

```bash
uv run python examples/run_insider_analysis.py
uv run satark analyze -p insider -d examples/data/sample_insider.csv
```

## Local documentation

Documentation only:

```bash
uv sync --group docs
uv run mkdocs serve
```

MkDocs serves documentation at `http://127.0.0.1:8000/` during local docs development.

## Complete website (marketing site + docs)

```bash
rm -rf public
mkdir -p public
cp -R website/. public/
uv run mkdocs build --site-dir public/docs
python -m http.server 8000 --directory public
```

Then open:

- [http://localhost:8000/](http://localhost:8000/)
- [http://localhost:8000/docs/](http://localhost:8000/docs/)
- [http://localhost:8000/history/](http://localhost:8000/history/)
- [http://localhost:8000/research/](http://localhost:8000/research/)

## Next

- [Architecture](architecture.md)
- [Plugin contract](concepts/plugins.md)
- [Scoring](concepts/scoring.md)
