# Identity

The `identity` plugin is a heuristic scaffold. It produces one high-severity
finding when an event category is `login` or `authentication`, or its tags
include `identity`, `bruteforce`, or `mfa-bypass`.

It does not calculate failed-login rates, establish sign-in baselines, inspect
geography, or interpret identity-provider risk signals. A full plugin should
normalize outcome, factor type, application, IP address, device, and user into
attributes, then create deterministic evidence for events such as repeated
failures, impossible travel, token abuse, or factor reset.
