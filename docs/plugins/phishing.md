# Phishing

The `phishing` plugin is a heuristic scaffold. It produces one high-severity
finding when an event is `email_received` or `web_request`, or has the tag
`phishing` or `spearphish`.

It does not inspect message headers, URLs, attachments, sender authentication,
or page content. A fuller implementation should normalize those details into
attributes, attach specific evidence such as SPF/DKIM results or URL matches,
and make deterministic decisions before any optional AI summarization.
