# SATARK

**Scalable Automated Technology for Analysis and Ranking of Known Threats**

SATARK is an open-source security analytics framework for building explainable detection pipelines across insider threats, malware, phishing, cloud, identity, email, and web security.

> Current status: **early open-source rebuild (alpha)**. APIs and packaging may change.

The official SATARK website and documentation are published at [https://satark.org](https://satark.org).

- Website: [https://satark.org/](https://satark.org/)
- Documentation: [https://satark.org/docs/](https://satark.org/docs/)
- History: [https://satark.org/history/](https://satark.org/history/)
- Research: [https://satark.org/research/](https://satark.org/research/)
- Community: [https://satark.org/community/](https://satark.org/community/)
- Trademark: [https://satark.org/trademark/](https://satark.org/trademark/) (India TM Application No. 7223965, Class 42)
- Repository: [https://github.com/kaushalbhavsar/satark](https://github.com/kaushalbhavsar/satark)

## Repository structure

```text
satark/
├── src/satark/          # Python package (core, scoring, graph, rules, ai, knowledge, plugins)
├── website/             # Static site for satark.org
├── docs/                # MkDocs Material documentation (published under /docs/)
├── examples/
├── tests/
├── mkdocs.yml
└── pyproject.toml
```

## Installation

Requires Python 3.13+.

```bash
git clone https://github.com/kaushalbhavsar/satark.git
cd satark
uv sync --group dev --group docs
```

## Local development

### Documentation only

```bash
uv sync --group docs
uv run mkdocs serve
```

### Complete website (site + docs)

```bash
rm -rf public
mkdir -p public
cp -R website/. public/
uv run mkdocs build --site-dir public/docs
python -m http.server 8000 --directory public
```

Then visit:

- http://localhost:8000/
- http://localhost:8000/docs/
- http://localhost:8000/history/
- http://localhost:8000/research/

### Quick demo (insider plugin)

```bash
uv sync --group dev
uv run satark list-plugins
uv run satark analyze -p insider -d examples/data/sample_insider.csv --threshold 0.5
```

Or run the scripted demo:

```bash
./scripts/demo_insider.sh
```

Screenshots from this run are embedded on the website ([architecture page](https://satark.org/architecture/) and homepage) under `website/assets/images/demo/`.

### Tests

```bash
uv sync --group dev
uv run pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [community page](https://satark.org/community/).

## Security

Please follow [SECURITY.md](SECURITY.md) for private vulnerability disclosure.

## Creator

Created by **Kaushal Bhavsar**.

## License

MIT — see [LICENSE](LICENSE).
