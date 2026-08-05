# Installation

## Requirements

- **Python 3.13+**
- `pip` and `venv` (included with CPython)

## Clone and create a virtual environment

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## Install the package

Runtime only:

```bash
pip install -e .
```

With test and lint tools:

```bash
pip install -e ".[dev]"
```

With documentation tools (MkDocs Material + mkdocstrings):

```bash
pip install -e ".[dev,docs]"
```

Optional ML extras used by the legacy LSTM demo:

```bash
pip install -e ".[ml]"
```

Everything:

```bash
pip install -e ".[all]"
```

## Verify

```bash
satark version
satark list-plugins
pytest
```

## Build these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

To produce a static site:

```bash
mkdocs build
```

Output is written to `site/`.
