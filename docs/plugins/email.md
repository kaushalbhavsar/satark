# Email

The `email` plugin is a heuristic scaffold. It produces one medium-severity
finding when an event is `email_received` or its tags include `email`, `bEC`,
or `spoof`. Tag comparisons are case-sensitive, so normalize tag values in your
source adapter.

It does not verify sender authentication, inspect attachment content, analyze
links, or model user-reported messages. Those additions should surface their
specific signals as evidence so the resulting score remains auditable.
