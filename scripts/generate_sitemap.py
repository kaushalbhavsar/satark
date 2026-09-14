#!/usr/bin/env python3
"""Generate sitemap.xml from a built static site directory."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

# Higher priority for primary site sections; docs default lower.
PRIORITY_OVERRIDES: dict[str, str] = {
    "/": "1.0",
    "/architecture/": "0.9",
    "/history/": "0.9",
    "/research/": "0.9",
    "/community/": "0.8",
    "/docs/": "0.85",
}


def discover_urls(public_dir: Path) -> list[str]:
    """Return canonical path URLs (with trailing slash) for each HTML page."""
    urls: set[str] = set()

    for html in sorted(public_dir.rglob("*.html")):
        rel = html.relative_to(public_dir)
        if rel.name != "index.html":
            continue
        if rel.parent == Path("."):
            urls.add("/")
            continue
        urls.add(f"/{rel.parent.as_posix()}/")

    return sorted(urls)


def priority_for(path: str) -> str:
    if path in PRIORITY_OVERRIDES:
        return PRIORITY_OVERRIDES[path]
    if path.startswith("/docs/"):
        return "0.7"
    return "0.6"


def build_sitemap(urls: list[str], base_url: str, lastmod: str) -> str:
    base = base_url.rstrip("/")
    urlset = ET.Element(
        "urlset",
        xmlns="http://www.sitemaps.org/schemas/sitemap/0.9",
    )

    for path in urls:
        url = ET.SubElement(urlset, "url")
        loc = ET.SubElement(url, "loc")
        loc.text = f"{base}{path}"
        mod = ET.SubElement(url, "lastmod")
        mod.text = lastmod
        pri = ET.SubElement(url, "priority")
        pri.text = priority_for(path)

    ET.indent(urlset, space="  ")
    return ET.tostring(urlset, encoding="unicode", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=Path("public"),
        help="Built site root (default: public)",
    )
    parser.add_argument(
        "--base-url",
        default="https://satark.org",
        help="Canonical site origin (default: https://satark.org)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: <public-dir>/sitemap.xml)",
    )
    args = parser.parse_args()

    public_dir = args.public_dir.resolve()
    if not public_dir.is_dir():
        msg = f"Public directory not found: {public_dir}"
        raise SystemExit(msg)

    urls = discover_urls(public_dir)
    if not urls:
        msg = f"No index.html pages found under {public_dir}"
        raise SystemExit(msg)

    lastmod = datetime.now(tz=UTC).date().isoformat()
    xml_body = build_sitemap(urls, args.base_url, lastmod)
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body + "\n"

    output = args.output or (public_dir / "sitemap.xml")
    output.write_text(xml, encoding="utf-8")
    print(f"Wrote {len(urls)} URLs to {output}")


if __name__ == "__main__":
    main()
