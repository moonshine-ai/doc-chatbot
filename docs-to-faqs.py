"""
doc_crawler.py — Build an anchor-aware section index from documentation source.

TWO MODES
─────────
1. ASCIIDOC (recommended for raspberrypi/documentation)
   Parse the locally cloned Git repo directly — no HTTP, no Cloudflare.

   Setup:
       git clone https://github.com/raspberrypi/documentation.git
       pip install sentence-transformers chromadb beautifulsoup4

   Run:
       python doc_crawler.py --mode asciidoc --repo ./documentation

2. PLAYWRIGHT (fallback for any live site behind a JS challenge)
   Drives a real browser, so Cloudflare and JS-rendered pages work fine.

   Setup:
       pip install playwright beautifulsoup4 sentence-transformers chromadb
       playwright install chromium

   Run:
       python doc_crawler.py --mode playwright --url https://example.com/docs/

OUTPUT
──────
  sections.json   — flat list of sections, each with a full anchor URL

FAQ GENERATION
──────────────
  From AsciiDoc (directory):  python docs-to-faqs.py --faq --adoc ./documentation/asciidoc/computers/os
  From Markdown (one or more files):  python docs-to-faqs.py --faq --markdown data/saikat-policies.md
  Optional:  --question-prefix "How will he"  → one required Q per section starting with that phrase, section title quoted verbatim, with extra words if needed for grammar (e.g. "… provide Paid Parental Leave?").
  Writes a sibling <stem>.faq.txt next to each source file (Q:/A: lines; answers 10–15 words from source only). Add --anthropic for Claude instead of Ollama. Use --force to replace an existing .faq.txt.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None

try:
    import anthropic
except ImportError:
    anthropic = None


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class Section:
    url: str  # full URL including #anchor
    page_url: str  # URL without fragment
    anchor: str  # id value on the heading
    level: int  # 1 = h1/=, 2 = h2/==, …
    heading: str  # visible heading text
    body: str  # plain text until the next heading
    page_title: str = ""  # title of the page this section belongs to
    breadcrumb: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list, repr=False)

    def embed_text(self) -> str:
        """Context-enriched text for the embedding model."""
        crumb = " > ".join(self.breadcrumb + [self.heading])
        return f"{crumb}\n\n{self.body}".strip()

    def to_dict(self, include_children: bool = True) -> dict:
        d = dict(
            url=self.url,
            page_url=self.page_url,
            anchor=self.anchor,
            level=self.level,
            heading=self.heading,
            page_title=self.page_title,
            breadcrumb=self.breadcrumb,
            body=self.body,
        )
        if include_children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# ── Shared: hierarchy builder ──────────────────────────────────────────────────


def build_hierarchy(flat: list[Section]) -> list[Section]:
    """
    Convert a flat ordered list of Sections into a nested tree in-place.
    Sets .breadcrumb and .children on every section; returns top-level roots.
    """
    for s in flat:
        s.children = []

    roots: list[Section] = []
    stack: list[Section] = []  # invariant: strictly increasing levels

    for sec in flat:
        while stack and stack[-1].level >= sec.level:
            stack.pop()

        sec.breadcrumb = [s.heading for s in stack]

        if stack:
            stack[-1].children.append(sec)
        else:
            roots.append(sec)

        stack.append(sec)

    return roots


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — AsciiDoc source parser
# ══════════════════════════════════════════════════════════════════════════════

# AsciiDoc heading levels: = H1, == H2, === H3, …
_HEADING_RE = re.compile(r"^(={1,6})\s+(.+)$")
# Explicit block anchor: [#some-id] on its own line immediately before a heading
_ANCHOR_RE = re.compile(r"^\[#([\w\-]+)\]")
# include directive: include::path/to/file.adoc[...]
_INCLUDE_RE = re.compile(r"^include::([^\[]+)\[")


def _asciidoc_slug(text: str) -> str:
    """
    Replicate Asciidoctor's default anchor ID generation:
      - lowercase
      - spaces and most punctuation → hyphens
      - strip leading/trailing hyphens
      - collapse consecutive hyphens

    The published RPi docs use this for headings without an explicit [#id].
    """
    s = text.lower()
    s = re.sub(
        r"[^\w\s-]", "", s
    )  # drop punctuation (keep word chars, spaces, hyphens)
    s = re.sub(r"[\s_]+", "-", s)  # spaces/underscores → hyphens
    s = re.sub(r"-{2,}", "-", s)  # collapse repeated hyphens
    return s.strip("-")


def _resolve_includes(adoc_path: Path, depth: int = 0) -> list[str]:
    """
    Recursively expand include:: directives, returning a flat list of lines.
    Limits recursion to 8 levels to guard against cycles.
    """
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


def _adoc_file_to_url(adoc_path: Path, repo_root: Path) -> str:
    """
    Map an .adoc file path inside the cloned repo to the published URL.

    RPi convention:
      documentation/asciidoc/computers/config_txt.adoc
        → https://www.raspberrypi.com/documentation/computers/config_txt.html

      documentation/asciidoc/computers/config_txt/video.adoc
        → (included into config_txt.adoc; parent file owns the URL)
    """
    base = "https://www.raspberrypi.com/documentation/"
    try:
        rel = adoc_path.relative_to(repo_root / "documentation" / "asciidoc")
    except ValueError:
        rel = adoc_path.relative_to(repo_root)

    html_rel = str(rel.with_suffix(".html"))
    return urljoin(base, html_rel)


def parse_asciidoc_repo(
    repo_root: str | Path,
) -> list[Section]:
    """
    Walk all top-level .adoc files under <repo_root>/documentation/asciidoc/,
    expand include:: directives, and return a flat list of Sections with
    breadcrumbs and anchor URLs set.

    Only top-level files are walked directly (included files are inlined);
    this preserves the correct page-URL-per-section mapping.
    """
    repo_root = Path(repo_root)
    adoc_dir = repo_root / "documentation" / "asciidoc"

    if not adoc_dir.exists():
        sys.exit(f"[asciidoc] Cannot find {adoc_dir}. Did you clone the repo?")

    # A file is "top-level" if there is no sibling directory with the same
    # stem (which would mean the file is the parent that includes a subdirectory).
    # We DO want the parent; we SKIP files that live inside such a subdirectory,
    # because they'll be inlined via include:: resolution.
    top_level_files: list[Path] = []
    for f in sorted(adoc_dir.rglob("*.adoc")):
        parent_adoc = f.parent.parent / (f.parent.name + ".adoc")
        if parent_adoc.exists():
            continue  # this file will be reached via include::
        top_level_files.append(f)

    all_flat: list[Section] = []

    for adoc_file in top_level_files:
        page_url = _adoc_file_to_url(adoc_file, repo_root)
        lines = _resolve_includes(adoc_file)
        sections = _parse_adoc_lines(lines, page_url)
        build_hierarchy(sections)  # sets breadcrumbs within this page
        all_flat.extend(sections)
        print(f"[asciidoc] Parsed {len(sections):3d} sections  {adoc_file.name}")

    # Re-run globally so cross-page hierarchy is consistent
    build_hierarchy(all_flat)
    return all_flat


def _parse_adoc_lines(lines: list[str], page_url: str) -> list[Section]:
    """
    Extract sections from a resolved (includes already expanded) AsciiDoc
    line list for a single page.
    """
    # Page title: first H1 (= ) on the page, or first heading of any level if no H1.
    # Skip literal blocks (----/....) so we don't use code lines as the title.
    page_title = ""
    in_literal = False
    for line in lines:
        if line.strip() in ("----", "...."):
            in_literal = not in_literal
            continue
        if in_literal:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            title_text = m.group(2).strip()
            title_text = re.sub(r"`([^`]+)`", r"\1", title_text)
            title_text = re.sub(r"\*([^*]+)\*", r"\1", title_text)
            title_text = re.sub(r"_([^_]+)_", r"\1", title_text)
            page_title = title_text
            if len(m.group(1)) == 1:
                break  # found H1, use it
    if not page_title and page_url:
        # Fallback: derive from URL path (e.g. config_txt.html -> Config txt)
        path = urlparse(page_url).path
        name = (
            (path.rstrip("/").split("/")[-1] or "")
            .replace(".html", "")
            .replace(".htm", "")
        )
        if name:
            page_title = name.replace("_", " ").replace("-", " ").title()

    sections: list[Section] = []
    pending_anchor: str | None = None  # [#id] seen on the previous line
    in_literal = False  # inside ---- or .... blocks

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track listing/literal blocks so we don't parse code as headings
        if line.strip() in ("----", "...."):
            in_literal = not in_literal
            i += 1
            continue
        if in_literal:
            i += 1
            continue

        # Explicit anchor [#id]
        m_anchor = _ANCHOR_RE.match(line.strip())
        if m_anchor:
            pending_anchor = m_anchor.group(1)
            i += 1
            continue

        # Heading line
        m_head = _HEADING_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            heading = m_head.group(2).strip()

            # Strip inline markup from heading text
            heading = re.sub(r"`([^`]+)`", r"\1", heading)
            heading = re.sub(r"\*([^*]+)\*", r"\1", heading)
            heading = re.sub(r"_([^_]+)_", r"\1", heading)

            anchor = pending_anchor or _asciidoc_slug(heading)
            pending_anchor = None
            full_url = f"{page_url}#{anchor}"

            # Collect body text up to the next heading or anchor marker
            body_lines: list[str] = []
            j = i + 1
            body_in_literal = False
            while j < len(lines):
                bl = lines[j]
                if bl.strip() in ("----", "...."):
                    body_in_literal = not body_in_literal
                    j += 1
                    continue
                if not body_in_literal:
                    if _HEADING_RE.match(bl):
                        break
                    if _ANCHOR_RE.match(bl.strip()):
                        break
                    # Strip common AsciiDoc markup for plain-text body
                    cleaned = re.sub(
                        r"https?://\S+\[([^\]]*)\]", r"\1", bl
                    )  # link macros
                    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
                    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
                    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
                    cleaned = re.sub(
                        r"^\s*[|+*-]+\s*", "", cleaned
                    )  # list markers / table cols
                    cleaned = cleaned.strip()
                    if (
                        cleaned
                        and not cleaned.startswith("[")
                        and not cleaned.startswith("//")
                    ):
                        body_lines.append(cleaned)
                j += 1

            body = re.sub(r"\s{3,}", "  ", " ".join(body_lines)).strip()

            sections.append(
                Section(
                    url=full_url,
                    page_url=page_url,
                    anchor=anchor,
                    level=level,
                    heading=heading,
                    body=body,
                    page_title=page_title,
                )
            )
            i += 1
            continue

        pending_anchor = None
        i += 1

    return sections


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Playwright live crawler (CF-capable)
# ══════════════════════════════════════════════════════════════════════════════

HEADING_TAGS = ["h1", "h2", "h3", "h4"]
CRAWL_DELAY = 1.0


def _slug_fallback(text: str) -> str:
    return re.sub(r"[^\w-]", "-", text.lower().strip()).strip("-")


def _extract_sections_from_html(html: str, page_url: str) -> list[Section]:
    """Parse a rendered HTML string into flat Sections."""
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("nav, footer, aside, script, style, [role=navigation]"):
        tag.decompose()

    # Page title from <title> or first h1.
    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    if not page_title:
        first_h1 = soup.find("h1")
        if first_h1:
            page_title = first_h1.get_text(" ", strip=True)

    selector = ", ".join(HEADING_TAGS)
    sections: list[Section] = []

    for h_tag in soup.select(selector):
        level = int(h_tag.name[1])
        heading_text = h_tag.get_text(" ", strip=True)

        anchor = h_tag.get("id") or ""
        if not anchor:
            a = h_tag.find("a", attrs={"name": True})
            anchor = a["name"] if a else _slug_fallback(heading_text)

        full_url = f"{page_url}#{anchor}" if anchor else page_url

        body_parts: list[str] = []
        for sibling in h_tag.next_siblings:
            if isinstance(sibling, Tag):
                if sibling.name in HEADING_TAGS:
                    break
                body_parts.append(sibling.get_text(" ", strip=True))
        body = re.sub(r"\s{3,}", "  ", " ".join(body_parts)).strip()

        sections.append(
            Section(
                url=full_url,
                page_url=page_url,
                anchor=anchor,
                level=level,
                heading=heading_text,
                body=body,
                page_title=page_title,
            )
        )

    return sections


def _discover_links_from_html(html: str, base_url: str, allow_prefix: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urljoin(base_url, href).split("#")[0]
        if abs_url.startswith(allow_prefix):
            links.append(abs_url)
    return links


def crawl_with_playwright(
    start_url: str,
    max_pages: int = 50,
    allow_prefix: str | None = None,
    headless: bool = True,
) -> list[Section]:
    """
    Crawl a live documentation site using a real Chromium browser via
    Playwright.  Handles Cloudflare challenges and JS-rendered content.

    Requires:
        pip install playwright beautifulsoup4
        playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "Install playwright: pip install playwright && playwright install chromium"
        )

    allow_prefix = allow_prefix or start_url
    visited: set[str] = set()
    queue: list[str] = [start_url]
    all_flat: list[Section] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            print(f"[playwright] ({len(visited)}/{max_pages}) {url}")
            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
                # Give CF challenge time to resolve if present
                page.wait_for_timeout(2_000)
                html = page.content()
            except Exception as e:
                print(f"[playwright] SKIP {url} — {e}")
                continue

            sections = _extract_sections_from_html(html, url)
            build_hierarchy(sections)
            all_flat.extend(sections)

            for link in _discover_links_from_html(html, url, allow_prefix):
                if link not in visited and link not in queue:
                    queue.append(link)

            time.sleep(CRAWL_DELAY)

        browser.close()

    build_hierarchy(all_flat)
    return all_flat


# ══════════════════════════════════════════════════════════════════════════════
# Embedding + ChromaDB
# ══════════════════════════════════════════════════════════════════════════════


def index_in_chroma(flat: list[Section], collection_name: str = "docs") -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
    except ImportError:
        print("[index] pip install sentence-transformers chromadb")
        return

    print("[index] Loading embedding model…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    texts = [s.embed_text() for s in flat]
    print(f"[index] Embedding {len(texts)} sections…")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    batch = 500
    for i in range(0, len(flat), batch):
        sl = flat[i : i + batch]
        collection.add(
            ids=[f"sec_{j}" for j in range(i, i + len(sl))],
            embeddings=embeddings[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=[
                dict(
                    url=s.url,
                    heading=s.heading,
                    level=s.level,
                    page_title=s.page_title,
                    breadcrumb=" > ".join(s.breadcrumb),
                )
                for s in sl
            ],
        )
    print(f"[index] Indexed {len(flat)} sections into '{collection_name}'.")


def query_chroma(
    question: str, collection_name: str = "docs", top_k: int = 5
) -> list[dict]:
    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    coll = client.get_collection(collection_name)

    results = coll.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=top_k,
    )
    hits = []
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        hits.append(
            dict(
                url=m["url"],
                heading=m["heading"],
                breadcrumb=m["breadcrumb"],
                score=results["distances"][0][i],
                snippet=results["documents"][0][i][:200],
            )
        )
    return hits


# ── FAQ generation via Ollama ─────────────────────────────────────────────────

FAQ_SYSTEM_PROMPT_ASCIIDOC = """
You are generating training questions for a voice assistant that answers Raspberry Pi questions.

You will be given a section of Raspberry Pi documentation. Generate questions that real users would actually SAY OUT LOUD to a voice assistant — the kind of casual, sometimes frustrated questions a beginner would speak, not type.

STYLE RULES:
- Write like someone talking, not typing. Use contractions (don't, I've, it's, can't).
- Many good questions start with a situation or problem, then ask the question:
  "My audio is coming out of HDMI but I want it through the headphone jack. How do I change that?"
  "I installed the Lite version and there's no media player. What do I need to get one?"
- Avoid formal question openers: never start with "Is it possible to", "What option", "Can I tell the player to", "What command should I use to", "How does one".
- Use plain words, not doc words. Say "flags" not "CLI options", say "headphone jack" not "audio output device", say "close when it's done" not "terminate upon completion".
- It's fine for a question to be incomplete or slightly rambly, like someone thinking out loud:
  "I just want the video to go fullscreen automatically. Is there a flag for that or something?"
  "Wait, can I play audio and video at the same time to different outputs?"
- Mix question lengths: some short ("Does VLC work on Pi Lite?"), some longer with context.
- Include beginner-level confusion questions, not just how-to questions:
  "I'm not sure if I have VLC installed. How do I check?"
  "What even is ALSA? Do I need to know about it to change my audio output?"

BAD examples (too formal, don't write like this):
  "What command should I run to see a list of all available audio devices?"
  "Can I navigate to a file directly from the Media menu option?"
  "Is there a way to run this without opening any graphical windows at all?"

GOOD examples (casual spoken style):
  "How do I see what audio devices I've got?"
  "I just want to open a video without using the terminal. Can I do that?"
  "Can I run it without a desktop, like just from the command line?"

OUTPUT FORMAT:
- Group items under the relevant section heading as a markdown heading line starting with #, plus a "# Whole Document" section for questions that apply to the whole passage.
- For each hypothetical question, output exactly two lines:
  Q: <the question — same casual spoken style as above>
  A: <one line; exactly 10 to 15 words (count carefully). Summarize only information that appears in the source text for that section (or in the full passage for Whole Document items). Do not add facts, guesses, or general knowledge not stated there. If the text does not support a concise answer, write a short refusal like "Not specified in this section.">
- Leave one blank line between Q/A pairs if you like, but keep Q: then A: adjacent for each item.
- Do not refer to "the document" or "the text" in the questions themselves."""

FAQ_SYSTEM_PROMPT_MARKDOWN = """
You are generating training questions for a voice assistant that answers questions using the document you are given.

You will be given Markdown content. Match the topic of your questions to this material (policies, terms, product information, technical help, etc.). Generate questions that real users would actually SAY OUT LOUD to a voice assistant — the kind of casual, sometimes frustrated questions a beginner would speak, not type.

STYLE RULES:
- Write like someone talking, not typing. Use contractions (don't, I've, it's, can't).
- Many good questions start with a situation or problem, then ask the question:
  "They said I have 30 days to cancel but I'm past that—what happens now?"
  "I'm not sure I'm reading this right. Does this apply to me if I'm outside the US?"
- Avoid formal question openers: never start with "Is it possible to", "What option", "Per the documentation", "What command should I use to", "How does one".
- Use plain words, not doc or legal jargon, unless the user would naturally say it.
- It's fine for a question to be incomplete or slightly rambly, like someone thinking out loud:
  "I just want to know if I can share this with my team or if it's only for me."
  "Wait, does this mean I lose access immediately or at the end of the period?"
- Mix question lengths: some short ("Is there a fee?"), some longer with context.
- Include beginner-level confusion questions, not just how-to questions:
  "I'm not sure what this section is saying. Can someone explain it in plain English?"
  "What's the difference between this and the other option?"

BAD examples (too formal, don't write like this):
  "What are the contractual obligations delineated in section 4.2?"
  "Is compliance with the aforementioned policy mandatory for all users?"

GOOD examples (casual spoken style):
  "What happens if I break this rule by accident?"
  "Can I get a refund if I change my mind?"
  "I'm on my phone—where do I find that?"

OUTPUT FORMAT:
- Group items under the relevant section heading as a markdown heading line starting with #, plus a "# Whole Document" section for questions that apply to the whole passage.
- For each hypothetical question, output exactly two lines:
  Q: <the question — same casual spoken style as above>
  A: <one line; exactly 10 to 15 words (count carefully). Summarize only information that appears in the source Markdown for that section (or in the full passage for Whole Document items). Do not add facts, guesses, or general knowledge not stated there. If the text does not support a concise answer, write a short refusal like "Not specified in this section.">
- Leave one blank line between Q/A pairs if you like, but keep Q: then A: adjacent for each item.
- Do not refer to "the document" or "the text" in the questions themselves."""

# Appended when --question-prefix is set; {prefix} is the user-supplied string (trimmed).
FAQ_QUESTION_PREFIX_BLOCK = """
QUESTION PREFIX MODE (required):
- For every section heading in your output (each line that starts with # followed by a title), EXCEPT "# Whole Document", you MUST include exactly one additional Q/A pair in that section's block, in addition to your other questions for that section.

PREFIX QUESTION SHAPE (strict — read carefully):
- The Q: line must be VERY SHORT. Pattern: {prefix} [at most one grammar helper word if required] <exact section title>?
- After the section title, STOP. End with ? only. Do not add anything else to the question — no extra clauses, no "if…", "when…", "actually", "how come", no commentary from the section body, and no follow-up phrases.
- The full section title from the # line (no #) must appear as an exact substring — same spelling and capitalization (e.g. "Save Muni and BART", "Universal Pre-K and Daycare").
- You MAY insert at most one short verb or preposition between "{prefix}" and the title ONLY when grammar requires it (e.g. "provide" before "Paid Parental Leave", "address" before a noun phrase). Do not insert extra words for color or emphasis.

BAD (too long — never do this):
  Q: How will he Save Muni and BART if there's no money for their operations?
  Q: How will he get Universal Pre-K and Daycare actually passed when it's been talked about forever?

GOOD (minimal — do this):
  Q: How will he Save Muni and BART?
  Q: How will he provide Universal Pre-K and Daycare?
  Q: How will he End the Wars?

- The A: line for that pair must follow the same rules as all other answers: exactly 10 to 15 words, summarizing only information from that section in the source. Put nuance and detail in A:, not in the Q:.
- For AsciiDoc source, use the same section titles as you use on your output # heading lines (derived from the AsciiDoc headings)."""


def _faq_system_prompt(
    prompt_kind: str, question_prefix: str | None = None
) -> str:
    if prompt_kind == "markdown":
        base = FAQ_SYSTEM_PROMPT_MARKDOWN.strip()
    else:
        base = FAQ_SYSTEM_PROMPT_ASCIIDOC.strip()
    if question_prefix and question_prefix.strip():
        base = base + "\n\n" + FAQ_QUESTION_PREFIX_BLOCK.format(
            prefix=question_prefix.strip()
        )
    return base


def generate_faq_questions(
    adoc_content: str,
    model: str = "qwen3.5:9b",
    prompt_kind: str = "asciidoc",
    question_prefix: str | None = None,
) -> str:
    """
    Ask the Ollama model for Q:/A: pairs: spoken-style questions plus 10–15 word
    answers grounded only in the given source (AsciiDoc or Markdown per prompt_kind).
    """
    if ollama_chat is None:
        raise RuntimeError("ollama package is not installed. pip install ollama")

    system_prompt = _faq_system_prompt(prompt_kind, question_prefix=question_prefix)

    response = ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": adoc_content},
        ],
    )
    # Support both object and dict response (ollama package version variance)
    if hasattr(response, "message"):
        return response.message.content or ""
    return (response.get("message") or {}).get("content", "")


def generate_faq_questions_anthropic(
    adoc_content: str,
    adoc_path: Path,
    model: str = "claude-sonnet-4-6",
    prompt_kind: str = "asciidoc",
    question_prefix: str | None = None,
    force: bool = False,
):
    """
    Same as generate_faq_questions() but uses Anthropic's API with Claude.
    Default is claude-sonnet-4-6. Requires ANTHROPIC_API_KEY and: pip install anthropic
    """
    if anthropic is None:
        raise RuntimeError(
            "anthropic package is not installed. pip install anthropic"
        )

    system_prompt = _faq_system_prompt(
        prompt_kind, question_prefix=question_prefix
    )

    output_file = adoc_path.with_suffix(".faq.txt")
    if output_file.exists() and not force:
        print(f"[faq] FAQ already exists for {adoc_path}")
        return

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_prompt,
                cache_control={"type": "ephemeral"},
                messages=[{"role": "user", "content": adoc_content}],
            )
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            print(f"[faq] Error generating FAQ for {adoc_path} (attempt {attempt + 1}/{max_attempts}): {e}")
            time_to_wait = 20 * (attempt + 1)
            for i in range(time_to_wait):
                print(f"[faq] Waiting {i + 1} of {time_to_wait} seconds...", end="\r")
                time.sleep(1)

    with open(output_file, "w") as f:
        f.write(response.content[0].text)


def run_faq_for_adoc(
    adoc_path: Path,
    model: str = "qwen3.5:9b",
    use_anthropic: bool = False,
    question_prefix: str | None = None,
    force: bool = False,
) -> str:
    """Load an .adoc file (with includes resolved), then generate FAQ questions via Ollama or Anthropic."""
    lines = _resolve_includes(Path(adoc_path))
    if not lines:
        raise ValueError(f"[faq] No lines found in {adoc_path}")
    adoc_content = "\n".join(lines)
    if use_anthropic:
        return generate_faq_questions_anthropic(
            adoc_content,
            adoc_path,
            model=model,
            question_prefix=question_prefix,
            force=force,
        )
    return generate_faq_questions(
        adoc_content, model=model, question_prefix=question_prefix
    )


def run_faq_for_markdown(
    md_path: Path,
    model: str = "qwen3.5:9b",
    use_anthropic: bool = False,
    question_prefix: str | None = None,
    force: bool = False,
) -> None:
    """Load a Markdown file and write hypothetical questions beside it as <name>.faq.txt (same as AsciiDoc flow)."""
    md_path = Path(md_path).resolve()
    if not md_path.is_file():
        raise ValueError(f"[faq] Not a file: {md_path}")
    text = md_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"[faq] Empty file: {md_path}")

    if use_anthropic:
        generate_faq_questions_anthropic(
            text,
            md_path,
            model=model,
            prompt_kind="markdown",
            question_prefix=question_prefix,
            force=force,
        )
        return

    output_file = md_path.with_suffix(".faq.txt")
    if output_file.exists() and not force:
        print(f"[faq] FAQ already exists for {md_path}")
        return

    result = generate_faq_questions(
        text,
        model=model,
        prompt_kind="markdown",
        question_prefix=question_prefix,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"[faq] Wrote {output_file}")


def run_faq_for_all_adocs(
    adoc_dir: Path,
    model: str = "qwen3.5:9b",
    use_anthropic: bool = False,
    question_prefix: str | None = None,
    force: bool = False,
) -> str:
    """Load all .adoc files in a directory, then generate FAQ questions via Ollama or Anthropic."""
    for adoc_path in adoc_dir.rglob("*.adoc"):
        print(f"[faq] Generating FAQ for {adoc_path}")
        try:
            run_faq_for_adoc(
                adoc_path,
                model=model,
                use_anthropic=use_anthropic,
                question_prefix=question_prefix,
                force=force,
            )
        except Exception as e:
            print(f"[faq] Error generating FAQ for {adoc_path}: {e}")
            continue

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a section index from docs.")
    parser.add_argument(
        "--mode", choices=["asciidoc", "playwright"], default="asciidoc"
    )
    parser.add_argument(
        "--repo",
        default="./documentation",
        help="Path to cloned raspberrypi/documentation repo (asciidoc mode)",
    )
    parser.add_argument(
        "--url",
        default="https://www.raspberrypi.com/documentation/",
        help="Start URL (playwright mode)",
    )
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Skip building the ChromaDB embedding index",
    )
    parser.add_argument(
        "--output", type=str, default="sections.json", help="Output file name"
    )
    parser.add_argument(
        "--faq",
        action="store_true",
        help="Generate FAQ questions via Ollama or Anthropic (see --adoc or --markdown)",
    )
    parser.add_argument(
        "--adoc",
        type=str,
        help="Directory of .adoc files to process recursively (*.adoc); used with --faq",
    )
    parser.add_argument(
        "--markdown",
        nargs="+",
        metavar="PATH",
        help="One or more Markdown files (e.g. data/saikat-policies.md); used with --faq",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:9b",
        help="Model for --faq: Ollama model (default: qwen3.5:9b) or Anthropic model when --anthropic (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--anthropic",
        action="store_true",
        help="Use Anthropic API (Claude 3.5 Sonnet) for --faq instead of Ollama; requires ANTHROPIC_API_KEY",
    )
    parser.add_argument(
        "--question-prefix",
        type=str,
        default=None,
        metavar="TEXT",
        help='With --faq: one extra Q/A per section; Q starts with TEXT, includes heading verbatim, may add words for grammar (e.g. "How will he provide Paid Parental Leave?")',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --faq: overwrite existing .faq.txt files instead of skipping",
    )
    args = parser.parse_args()

    if args.faq:
        if args.markdown and args.adoc:
            parser.error("Use either --markdown or --adoc with --faq, not both")
        if not args.adoc and not args.markdown:
            parser.error("--faq requires --adoc (directory of .adoc files) or --markdown (one or more .md files)")
        model = args.model
        if args.anthropic and model == "qwen3.5:9b":
            model = "claude-sonnet-4-6"
        if args.markdown:
            for raw in args.markdown:
                md_path = Path(raw)
                if not md_path.exists():
                    parser.error(f"Markdown path not found: {md_path}")
                if not md_path.is_file():
                    parser.error(f"Not a file: {md_path}")
                print(f"[faq] Generating FAQ for {md_path}")
                try:
                    run_faq_for_markdown(
                        md_path,
                        model=model,
                        use_anthropic=args.anthropic,
                        question_prefix=args.question_prefix,
                        force=args.force,
                    )
                except Exception as e:
                    print(f"[faq] Error generating FAQ for {md_path}: {e}")
            return
        run_faq_for_all_adocs(
            Path(args.adoc),
            model=model,
            use_anthropic=args.anthropic,
            question_prefix=args.question_prefix,
            force=args.force,
        )
        return

    if args.mode == "asciidoc":
        print(f"[main] Parsing AsciiDoc repo at {args.repo} …")
        flat = parse_asciidoc_repo(args.repo)
    else:
        print(f"[main] Crawling {args.url} with Playwright …")
        flat = crawl_with_playwright(args.url, max_pages=args.max_pages)

    print(f"\n[main] Found {len(flat)} sections total.")

    with open("sections.json", "w") as f:
        json.dump([s.to_dict(include_children=False) for s in flat], f, indent=2)
    print("[main] Wrote sections.json")


if __name__ == "__main__":
    main()
