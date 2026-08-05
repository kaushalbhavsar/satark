# Domain Plugin Stubs

The following plugins share `HeuristicDomainPlugin`: they match watched categories and/or tags, then emit a single heuristic detection.

## Malware

- **Name:** `malware`
- Watches: `process_execution`, `file_write`
- Tags: `malware`, `ransomware`, `trojan`

## Phishing

- **Name:** `phishing`
- Watches: `email_received`, `web_request`
- Tags: `phishing`, `spearphish`

## Web

- **Name:** `web`
- Watches: `web_request`
- Tags: `xss`, `sqli`, `web-attack`

## Email

- **Name:** `email`
- Watches: `email_received`
- Tags: `email`, `bEC`, `spoof`

## Cloud

- **Name:** `cloud`
- Watches: `cloud_api_call`
- Tags: `cloud`, `iam-abuse`, `exfil`

## Identity

- **Name:** `identity`
- Watches: `login`, `authentication`
- Tags: `identity`, `bruteforce`, `mfa-bypass`

## Example record

```python
records = [
    {
        "category": "process_execution",
        "source": "edr",
        "actor": "svc",
        "tags": "malware",
        "action": "powershell.exe",
    }
]
plugin = create_plugin("malware")
events = plugin.normalize(records, PluginContext())
print(plugin.detect(events, PluginContext()))
```

These stubs are intentional extension points — replace heuristics with richer detectors while keeping the shared contract.
