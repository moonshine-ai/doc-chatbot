#!/usr/bin/env python3
"""Fetch a page through ScrapingBee with return_page_markdown and save it as a .md file."""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SCRAPINGBEE_API = "https://app.scrapingbee.com/api/v1"


def _default_output_path(page_url: str) -> str:
    path = urllib.parse.urlparse(page_url).path.rstrip("/")
    base = path.split("/")[-1] if path else "index"
    if not base or base == "":
        base = "index"
    safe = re.sub(r"[^\w.\-]+", "_", base, flags=re.UNICODE)
    return f"{safe}.md"


def fetch_markdown(page_url: str, api_key: str, render_js: bool) -> bytes:
    params = {
        "api_key": api_key,
        "url": page_url,
        "return_page_markdown": "true",
        "render_js": "true" if render_js else "false",
    }
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{SCRAPINGBEE_API}?{qs}", method="GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape a URL via ScrapingBee (return_page_markdown) and write Markdown to a file."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.saikat.us/en/policies",
        help="Page to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Output .md path (default: derived from the URL path, e.g. policies.md)",
    )
    parser.add_argument(
        "--no-render-js",
        action="store_true",
        help="Pass render_js=false (faster; use if the page is static HTML).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SCRAPINGBEE_API_KEY")
    if not api_key:
        print("SCRAPINGBEE_API_KEY is not set.", file=sys.stderr)
        return 1

    out_path = args.output or _default_output_path(args.url)

    try:
        body = fetch_markdown(args.url, api_key, render_js=not args.no_render_js)
    except urllib.error.HTTPError as e:
        print(f"ScrapingBee HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(body)

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
