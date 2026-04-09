#!/usr/bin/env python3
"""
Serve a single-page question search app and JSON API.

- Serves the SPA at / (white background, Moonshine logo, search box).
- API: POST /api/search with JSON {"question": "..."} or GET /api/search?q=...
  POST /api/suggest with JSON {"q": "..."} — top embedding-nearest questions for live suggestions.
  Search returns JSON with results (distance < 0.3 only), including asciidoc content
  from documentation/documentation/asciidoc/ (full doc or named section by source).

Run:
  python app_server.py [--port 8080] [--doc-root PATH] [--embeddings PATH]

Requires: question_search, doc_content, and optionally the documentation repo
at --doc-root for full result content.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Ensure we can import from project
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from question_search import QuestionSearch
from saikat_policy_anchors import policies_url_for_stub

# -----------------------------------------------------------------------------
# Doc content extraction (inlined so the server always uses this code)
# -----------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(={1,6})\s+(.+)$")
_ANCHOR_RE = re.compile(r"^\[#([\w\-]+)\]")
_INCLUDE_RE = re.compile(r"^include::([^\[]+)\[")


def _asciidoc_slug(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def _resolve_includes(adoc_path: Path, depth: int = 0) -> list[str]:
    if depth > 8 or not adoc_path.exists():
        return []
    lines = adoc_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = []
    for line in lines:
        m = _INCLUDE_RE.match(line)
        if m:
            include_path = adoc_path.parent / m.group(1)
            result.extend(_resolve_includes(include_path, depth + 1))
        else:
            result.append(line)
    return result


def _adoc_path_from_source(doc_root: Path, source_path: str) -> Path | None:
    path_part = source_path.replace("\\", "/").strip("/")
    if path_part.endswith(".faq"):
        path_part = path_part[:-4]
    if not path_part.endswith(".adoc"):
        path_part = path_part + ".adoc"
    full = doc_root / path_part
    if full.exists():
        return full
    alt = doc_root / (source_path.strip("/").replace(".faq", "") + ".adoc")
    return alt if alt.exists() else None


def _extract_section_as_adoc(lines: list[str], section_slug: str) -> str | None:
    section_slug = (section_slug or "").strip().lower()
    pending_anchor: str | None = None
    in_literal = False
    i = 0
    collecting: list[str] = []
    section_level: int | None = None

    while i < len(lines):
        line = lines[i]
        if line.strip() in ("----", "...."):
            in_literal = not in_literal
            if collecting:
                collecting.append(line)
            i += 1
            continue
        if in_literal:
            if collecting:
                collecting.append(line)
            i += 1
            continue
        m_anchor = _ANCHOR_RE.match(line.strip())
        if m_anchor:
            pending_anchor = m_anchor.group(1)
            if collecting:
                collecting.append(line)
            i += 1
            continue
        m_head = _HEADING_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            heading = m_head.group(2).strip()
            anchor = pending_anchor or _asciidoc_slug(heading)
            pending_anchor = None
            heading_slug = _asciidoc_slug(heading)
            if section_level is not None:
                anchor_match = anchor.strip().lower() == section_slug
                heading_match = heading_slug.strip().lower() == section_slug
                if (anchor_match or heading_match) and level < section_level:
                    collecting = [line]
                    section_level = level
                    i += 1
                    continue
                if level <= section_level:
                    return "\n".join(collecting)
                collecting.append(line)
                i += 1
                continue
            anchor_match = anchor.strip().lower() == section_slug
            heading_match = heading_slug.strip().lower() == section_slug
            if anchor_match or heading_match:
                if not collecting or level < section_level:
                    collecting = [line]
                    section_level = level
            i += 1
            continue
        pending_anchor = None
        if collecting:
            collecting.append(line)
        i += 1
    if collecting:
        return "\n".join(collecting)
    return None


def get_doc_content(doc_root: Path, source: str) -> tuple[str | None, str | None]:
    if "#" in source:
        path_part, section_slug = source.split("#", 1)
        section_slug = section_slug.strip()
    else:
        path_part = source
        section_slug = "whole-document"
    path_part = path_part.strip()
    adoc_path = _adoc_path_from_source(doc_root, path_part)
    if not adoc_path or not adoc_path.is_file():
        return None, "File not found"
    try:
        lines = _resolve_includes(adoc_path)
    except Exception as e:
        return None, str(e)
    if not lines:
        return None, "Empty or unreadable file"
    if section_slug == "whole-document":
        return "\n".join(lines), None
    content = _extract_section_as_adoc(lines, section_slug)
    if content is None:
        return None, f"Section '{section_slug}' not found"
    return content, None

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
SUGGEST_MIN_LEN = 2
SUGGEST_MAX_ITEMS = 6

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


@app.route("/saikat-policies")
def saikat_policies_index():
    """Saikat campaign–styled policy Q&A (uses same /api/search as the main SPA)."""
    return send_from_directory(SCRIPT_DIR / "data", "saikat-policies-index.html")


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
    for sim, sentence, source, answer, stub in raw:
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
        if answer:
            entry["answer"] = answer
        if stub:
            entry["stub"] = stub
        policy_url = policies_url_for_stub(stub)
        if policy_url:
            entry["policy_url"] = policy_url
        if _doc_root and _doc_root.is_dir() and source:
            content, err = get_doc_content(_doc_root, source)
            entry["content_adoc"] = content
            if err:
                entry["content_error"] = err
        results.append(entry)

    return {"results": results}


@app.route("/api/suggest", methods=["GET", "POST"])
def api_suggest():
    """
    Return the closest FAQ questions to the partial query (embedding similarity),
    for live suggestions while typing. No doc content; looser than /api/search
    filtering — always returns up to SUGGEST_MAX_ITEMS nearest neighbors.
    """
    q = None
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        q = data.get("question") or data.get("q")
    if q is None:
        q = request.args.get("q", "").strip()
    if not q or len(q.strip()) < SUGGEST_MIN_LEN:
        return {"suggestions": []}

    try:
        search = _get_search()
    except RuntimeError:
        return {"error": "Search not initialized"}, 503

    raw = search.query(q.strip(), n=SUGGEST_MAX_ITEMS)
    suggestions = []
    for sim, sentence, source, answer, stub in raw:
        distance = 1.0 - sim
        item = {
            "sentence": sentence,
            "distance": round(distance, 4),
            "source": source or "",
        }
        if answer:
            item["answer"] = answer
        if stub:
            item["stub"] = stub
        policy_url = policies_url_for_stub(stub)
        if policy_url:
            item["policy_url"] = policy_url
        suggestions.append(item)

    return {"suggestions": suggestions}


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
