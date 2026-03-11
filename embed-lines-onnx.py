#!/usr/bin/env python3
"""
Embed lines of text with Moonshine Voice Embedding Gemma 300M via ONNX Runtime.

1. Downloads the model with moonshine_voice.get_embedding_model().
2. Loads the ONNX model with onnxruntime.
3. Reads sentences from .faq.txt files in a directory (skips lines starting with #).
4. Writes one JSON object per line: {"sentence": "...", "embedding": [...], "source": "path/to/file#section-slug"}

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


def _sentences_from_faq_dir(dir_path: Path):
    """Yield (sentence, source) from all .faq.txt files. source = path/to/file#section-slug."""
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")
    for faq_path in sorted(dir_path.rglob("*.faq.txt")):
        try:
            rel = faq_path.relative_to(dir_path)
        except ValueError:
            rel = faq_path
        path_part = str(rel.with_suffix("")).replace("\\", "/")
        current_section = "whole-document"
        with open(faq_path, encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n\r")
                line = raw.lstrip("- ")
                if not line:
                    continue
                if line.lstrip().startswith("#"):
                    heading = line.lstrip("#").strip()
                    current_section = _slugify(heading) if heading else "section"
                    continue
                source = f"{path_part}#{current_section}"
                yield line, source


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_faq_dir = script_dir / ".." / "documentation" / "documentation" / "asciidoc"

    parser = argparse.ArgumentParser(
        description="Embed lines with Embedding Gemma 300M (ONNX Runtime)"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(default_faq_dir.resolve()),
        help="Directory containing .faq.txt files (default: %(default)s)",
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

    faq_dir = Path(args.input_dir).resolve()
    items = list(_sentences_from_faq_dir(faq_dir))
    if not items:
        print(f"No sentences found in .faq.txt files under {faq_dir}", file=sys.stderr)
        return 1
    print(f"Embedding {len(items)} sentences from .faq.txt under {faq_dir}...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as fout:
        for sentence, source in items:
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

            obj = {"sentence": sentence, "embedding": vec, "source": source}
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
