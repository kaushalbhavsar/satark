# Rules

Rule engines evaluate normalized events and return `RuleMatch` objects. Plugins may compose these engines inside `detect()`.

## Engines

| Engine | Status | Notes |
|--------|--------|-------|
| `RegexRuleEngine` | Ready | Match attributes / fields by regex |
| `SigmaRuleEngine` | Ready (minimal) | Field equality “selection” matcher |
| `CustomRuleEngine` | Ready | Arbitrary Python predicates |
| `StixRuleEngine` | Stub / light | Indicator vs tags/category |
| `YaraRuleEngine` | Stub | Requires optional `yara-python` |

## Example

```python
from satark.rules import RegexRuleEngine, CustomRuleEngine

regex = RegexRuleEngine()
regex.add_rule("r1", "Suspicious shell", "action", r"powershell|cmd\\.exe")

custom = CustomRuleEngine()
custom.add_rule("c1", "Off-hours actor", lambda e: e.actor == "alice")

matches = regex.match(events) + custom.match(events)
```

See [Rules API](../api/rules.md).
