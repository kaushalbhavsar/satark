# Installation

## Requirements

- Python **3.13+**
- `pip` and `venv` (included with Python)

## Create a virtual environment

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For documentation tooling as well:

```bash
pip install -e ".[dev,docs]"
```

## Verify

```bash
satark version
pytest
```
