# Installation

## Requirements

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## With uv

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
uv sync
```

## With pip

```bash
pip install -e ".[dev]"
```

## Verify

```bash
uv run satark version
uv run pytest
```
