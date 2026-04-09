#!/usr/bin/env python3
"""
Embed lines of text with Moonshine Voice Embedding Gemma 300M via ONNX Runtime.

1. Downloads the model with moonshine_voice.get_embedding_model().
2. Loads the ONNX model with onnxruntime.
3. Reads .faq.txt files:
   - Directory mode: all *.faq.txt under a directory (legacy: one line per sentence, or Q:/A: pairs).
   - File mode: one or more paths like data/saikat-policies.faq.txt with a sibling
     saikat-policies.md (same stem). Parses Q:/A: pairs, matches # sections to Markdown
     ### headings, and stores stub (header text from the .md section) plus answer text.
4. Writes one JSON object per line:
   {"sentence": "...", "question": "...", "stub": "...", "answer": "...", "embedding": [...], "source": "..."}

Default input: ../documentation/documentation/asciidoc (all *.faq.txt files there).

Dependencies (in addition to moonshine-voice):
  pip install onnxruntime transformers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _normalize_l2(vec: list[float]) -> list[float]:
    s = sum(v * v for v in vec) ** 0.5
    if s <= 0.0:
        return vec
    return [v / s for v in vec]


def _model_onnx_path(model_dir: str, variant: str) -> str:
    if variant == "fp32":
        name = "model.onnx"
    else:
        name = f"model_{variant}.onnx"
    return os.path.join(model_dir, name)


def _slugify(text: str) -> str:
    """Convert heading text to a URL-style slug (lowercase, hyphens)."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "section"


def _norm_heading_key(text: str) -> str:
    """Normalize FAQ or Markdown heading text for lookup."""
    t = text.strip()
    t = re.sub(r"^#+\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _md_heading_stubs(md_text: str) -> dict[str, str]:
    """
    Map normalized heading text -> stub string (visible heading from the .md file).
    Later headings with the same normalized key overwrite earlier ones.
    """
    out: dict[str, str] = {}
    for line in md_text.splitlines():
        m = _MD_HEADING_RE.match(line.strip())
        if not m:
            continue
        title = m.group(2).strip()
        out[_norm_heading_key(title)] = title
    return out


def _sibling_markdown_path(faq_path: Path) -> Path:
    """saikat-policies.faq.txt -> saikat-policies.md"""
    name = faq_path.name
    if not name.endswith(".faq.txt"):
        raise ValueError(f"Expected *.faq.txt, got {faq_path}")
    return faq_path.parent / f"{name[:-len('.faq.txt')]}.md"


def _iter_faq_qa_records(
    faq_path: Path,
    *,
    path_part: str,
    md_stubs: dict[str, str] | None,
):
    """
    Yield dicts with keys: question, answer, stub, source (and section slug in source).

    md_stubs: from _md_heading_stubs; if None, stub is always "".
    """
    current_heading = "Whole Document"
    current_slug = _slugify(current_heading)

    lines = faq_path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n\r")
        line = raw.lstrip("- ").strip()
        if not line:
            i += 1
            continue
        if line.startswith("#"):
            current_heading = line.lstrip("#").strip() or "Section"
            current_slug = _slugify(current_heading)
            i += 1
            continue

        upper = line[:2].upper()
        if upper == "Q:":
            q = line[2:].strip()
            ans = ""
            i += 1
            if i < len(lines):
                nxt = lines[i].lstrip("- ").strip()
                if nxt[:2].upper() == "A:":
                    ans = nxt[2:].strip()
                    i += 1
            stub = ""
            if md_stubs is not None:
                stub = md_stubs.get(_norm_heading_key(current_heading), "")
            source = f"{path_part}#{current_slug}"
            yield {
                "question": q,
                "answer": ans,
                "stub": stub,
                "source": source,
            }
            continue

        if upper == "A:":
            i += 1
            continue

        # Legacy: one line = one sentence to embed
        stub = ""
        if md_stubs is not None:
            stub = md_stubs.get(_norm_heading_key(current_heading), "")
        yield {
            "question": line,
            "answer": "",
            "stub": stub,
            "source": f"{path_part}#{current_slug}",
        }
        i += 1


def _faq_path_part_for_source(faq_path: Path) -> str:
    """path/to/foo.faq.txt -> path/to/foo.faq (for source ids, prefer cwd-relative)."""
    faq_path = faq_path.resolve()
    try:
        rel = faq_path.relative_to(Path.cwd())
    except ValueError:
        rel = faq_path
    return str(rel.with_suffix("")).replace("\\", "/")


def _records_from_faq_and_md(faq_path: Path, md_path: Path) -> list[dict]:
    """Pair a .faq.txt with its sibling .md."""
    md_text = md_path.read_text(encoding="utf-8", errors="replace")
    stubs = _md_heading_stubs(md_text)
    path_part = _faq_path_part_for_source(faq_path)
    return list(
        _iter_faq_qa_records(
            faq_path.resolve(),
            path_part=path_part,
            md_stubs=stubs,
        )
    )


def _records_from_faq_dir(dir_path: Path) -> list[dict]:
    """All *.faq.txt under dir; no .md pairing (stub left empty)."""
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")
    out: list[dict] = []
    for faq_path in sorted(dir_path.rglob("*.faq.txt")):
        try:
            rel = faq_path.relative_to(dir_path)
        except ValueError:
            rel = faq_path
        path_part = str(rel.with_suffix("")).replace("\\", "/")
        out.extend(
            _iter_faq_qa_records(faq_path, path_part=path_part, md_stubs=None)
        )
    return out


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_faq_dir = script_dir / ".." / "documentation" / "documentation" / "asciidoc"

    parser = argparse.ArgumentParser(
        description="Embed lines with Embedding Gemma 300M (ONNX Runtime)"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=None,
        help="Directory of .faq.txt files (recursive), or one or more *.faq.txt paths "
        "with sibling .md files (same stem). Omit to use the default asciidoc directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file (JSONL: one JSON object per line)",
    )
    parser.add_argument(
        "--variant",
        default="fp32",
        choices=["fp32", "fp16", "q8", "q4", "q4f16"],
        help="Model variant to download/load (default: fp32)",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/embeddinggemma-300m",
        help="Hugging Face model id for AutoTokenizer (default: google/embeddinggemma-300m)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length for tokenization (default: 2048)",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy is required.", file=sys.stderr)
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime", file=sys.stderr)
        return 1

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "Install transformers: pip install transformers",
            file=sys.stderr,
        )
        return 1

    from moonshine_voice import get_embedding_model

    print("Downloading/ensuring embedding model...", file=sys.stderr)
    model_dir, _arch = get_embedding_model("embeddinggemma-300m", variant=args.variant)
    onnx_path = _model_onnx_path(model_dir, args.variant)
    if not os.path.isfile(onnx_path):
        print(f"ONNX model not found at {onnx_path}", file=sys.stderr)
        return 1

    print(f"Loading ONNX session from {onnx_path}...", file=sys.stderr)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    input_names = {i.name for i in session.get_inputs()}
    if "input_ids" not in input_names or "attention_mask" not in input_names:
        print(
            f"Model inputs expected input_ids + attention_mask; got {input_names}",
            file=sys.stderr,
        )
        return 1

    output_names = [o.name for o in session.get_outputs()]
    if "sentence_embedding" not in output_names:
        print(
            f"Expected output sentence_embedding; outputs are {output_names}",
            file=sys.stderr,
        )
        return 1

    print(f"Loading tokenizer {args.tokenizer}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.inputs:
        paths = [Path(p).resolve() for p in args.inputs]
    else:
        paths = [default_faq_dir.resolve()]

    if len(paths) == 1 and paths[0].is_dir():
        records = _records_from_faq_dir(paths[0])
        src_desc = str(paths[0])
    else:
        records = []
        for p in paths:
            if not p.is_file():
                print(f"Not a file: {p}", file=sys.stderr)
                return 1
            if not p.name.endswith(".faq.txt"):
                print(f"Expected a .faq.txt file: {p}", file=sys.stderr)
                return 1
            md = _sibling_markdown_path(p)
            if not md.is_file():
                print(f"Missing sibling Markdown for {p}: expected {md}", file=sys.stderr)
                return 1
            records.extend(_records_from_faq_and_md(p, md))
        src_desc = ", ".join(str(p) for p in paths)

    if not records:
        print(f"No FAQ records found (inputs: {src_desc})", file=sys.stderr)
        return 1
    print(f"Embedding {len(records)} questions from {src_desc}...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            sentence = rec["question"]
            encoded = tokenizer(
                sentence,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

            outputs = session.run(
                ["sentence_embedding"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            embedding = outputs[0]
            # Shape (1, dim) -> list[float]
            if embedding.ndim == 2:
                embedding = embedding[0]
            vec = embedding.astype(np.float32).tolist()
            vec = _normalize_l2(vec)

            obj = {
                "sentence": sentence,
                "question": rec["question"],
                "stub": rec["stub"],
                "answer": rec["answer"],
                "embedding": vec,
                "source": rec["source"],
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
