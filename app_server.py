#!/usr/bin/env python3
"""
Serve a single-page question search app and JSON API.

- Serves the SPA at / (white background, Moonshine logo, search box).
- API: POST /api/search with JSON {"question": "..."} or GET /api/search?q=...
  Returns JSON with results (distance < 0.3 only), including asciidoc content
  from documentation/documentation/asciidoc/ (full doc or named section by source).

Run:
  python app_server.py [--port 8080] [--doc-root PATH] [--embeddings PATH]

Requires: question_search, doc_content, and optionally the documentation repo
at --doc-root for full result content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import from project
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from doc_content import get_doc_content
from question_search import QuestionSearch

try:
    from flask import Flask, request, send_from_directory
except ImportError:
    print("Install Flask: pip install flask", file=sys.stderr)
    sys.exit(1)

DEFAULT_PORT = 8080
DEFAULT_DOC_ROOT = SCRIPT_DIR / "documentation" / "documentation" / "asciidoc"
DEFAULT_EMBEDDINGS = SCRIPT_DIR / "question-embeddings.json"
DISTANCE_THRESHOLD = 0.3
MAX_RESULTS = 20

app = Flask(__name__, static_folder=None)
app.config["JSON_SORT_KEYS"] = False

# Set at startup
_search: QuestionSearch | None = None
_doc_root: Path | None = None


def _get_search() -> QuestionSearch:
    global _search
    if _search is None:
        raise RuntimeError("QuestionSearch not initialized")
    return _search


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")


@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    question = None
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        question = data.get("question") or data.get("q")
    if question is None:
        question = request.args.get("q", "").strip()
    if not question:
        return {"error": "Missing question (use ?q=... or JSON {\"question\": \"...\"})"}, 400

    try:
        search = _get_search()
    except RuntimeError:
        return {"error": "Search not initialized"}, 503

    # Top results; we'll filter by distance and attach content
    raw = search.query(question, n=MAX_RESULTS)
    results = []
    for sim, sentence, source in raw:
        distance = 1.0 - sim
        if distance >= DISTANCE_THRESHOLD:
            continue
        entry = {
            "distance": round(distance, 4),
            "sentence": sentence,
            "source": source or "",
            "content_adoc": None,
            "content_error": None,
        }
        if _doc_root and _doc_root.is_dir() and source:
            content, err = get_doc_content(_doc_root, source)
            entry["content_adoc"] = content
            if err:
                entry["content_error"] = err
        results.append(entry)

    return {"results": results}


def main() -> int:
    global _search, _doc_root

    parser = argparse.ArgumentParser(description="Serve question search SPA and API")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--doc-root",
        type=Path,
        default=DEFAULT_DOC_ROOT,
        help=f"Documentation asciidoc root (default: {DEFAULT_DOC_ROOT})",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"JSONL embeddings (default: {DEFAULT_EMBEDDINGS})",
    )
    parser.add_argument(
        "--no-doc",
        action="store_true",
        help="Do not load doc root; API will not include content_adoc",
    )
    args = parser.parse_args()

    _doc_root = None if args.no_doc else args.doc_root.resolve()
    if _doc_root and not _doc_root.is_dir():
        print(f"Warning: doc-root not found: {_doc_root}", file=sys.stderr)
        _doc_root = None

    print("Loading question embeddings...", file=sys.stderr)
    try:
        _search = QuestionSearch(args.embeddings)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"Loaded {len(_search)} questions.", file=sys.stderr)
    if _doc_root:
        print(f"Doc root: {_doc_root}", file=sys.stderr)
    else:
        print("Doc root disabled; results will not include content_adoc.", file=sys.stderr)

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
