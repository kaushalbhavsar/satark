# CLI Guide

The `satark` CLI is built with Typer and Rich.

## Commands

### `satark version`

Print the package version.

### `satark list-plugins`

Show built-in plugins with domain and description.

### `satark analyze`

Run a plugin against optional input data.

```bash
satark analyze \
  --plugin insider \
  --data examples/data/sample_insider.csv \
  --threshold 0.7 \
  --explain
```

Supported data formats:

- `.csv` — header row required
- `.json` — list of objects
- `.jsonl` — one JSON object per line

### `satark explain-finding`

Print a why-malicious summary for a serialized `Finding` JSON file:

```bash
satark explain-finding path/to/finding.json
```

## Environment configuration

Settings load from environment variables prefixed with `SATARK_`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SATARK_LOG_LEVEL` | `INFO` | Log level |
| `SATARK_DATA_DIR` | `./data` | Data directory |
| `SATARK_ENABLE_AI` | `false` | Enable AI assistants |
| `SATARK_RISK_THRESHOLD` | `0.7` | Elevated risk cutoff |
| `SATARK_KNOWLEDGE_VERSION` | `latest` | Preferred knowledge version |
