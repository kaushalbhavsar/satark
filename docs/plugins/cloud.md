# Cloud

The `cloud` plugin is a heuristic scaffold for cloud audit telemetry. It
produces one high-severity finding when an event is `cloud_api_call` or tags
include `cloud`, `iam-abuse`, or `exfil`.

It does not parse CloudTrail, Azure Activity Logs, or Google Cloud Audit Logs.
Extend it by normalizing principal, API operation, resource, result, source IP,
and region into stable fields; then add rules for actions such as access-key
creation, policy changes, unusual data export, or disabled logging.
