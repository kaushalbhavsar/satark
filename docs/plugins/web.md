# Web

The `web` plugin is a heuristic scaffold for web-application telemetry. It
produces one medium-severity finding when an event category is `web_request` or
when tags contain `xss`, `sqli`, or `web-attack`.

It does not parse HTTP access logs, decode requests, perform signature matching,
or distinguish scanners from exploitation. A practical extension should retain
method, route, response status, source IP, user agent, and sanitized request
features in `attributes`, then attach exact matched evidence to each detection.
