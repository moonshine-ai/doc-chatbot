#!/usr/bin/env python3
"""
Find the three closest FAQ questions to input text using cosine similarity.

Loads question embeddings from question-embeddings.json (JSONL with "sentence", "embedding", and optional "source" as path#section-slug).
loads the same embedding model (Gemma 300M via ONNX) to embed the input text,
then for each input line prints the three closest questions and their distance
(1 - cosine_similarity; 0 = identical, 2 = opposite).

Usage:
  python question_search.py                    # interactive: type a sentence, press Enter, see results
  python question_search.py "How do I ...?"   # single query from argument
  echo "How do I use the AI camera?" | python question_search.py  # batch from stdin

Dependencies: moonshine-voice, onnxruntime, transformers, numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Assumes both vectors are L2-normalized."""
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_embeddings = script_dir / "question-embeddings.json"

    parser = argparse.ArgumentParser(
        description="Find closest FAQ questions by embedding similarity"
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help="Query strings (if none, read one per line from stdin)",
    )
    parser.add_argument(
        "--embeddings",
        default=str(default_embeddings),
        help=f"JSONL file with sentence/embedding (default: {default_embeddings})",
    )
    parser.add_argument(
        "--variant",
        default="fp32",
        choices=["fp32", "fp16", "q8", "q4", "q4f16"],
        help="Model variant (default: fp32)",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/embeddinggemma-300m",
        help="Hugging Face tokenizer (default: google/embeddinggemma-300m)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length (default: 2048)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of nearest questions to show (default: 3)",
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
        print("Install transformers: pip install transformers", file=sys.stderr)
        return 1

    from moonshine_voice import get_embedding_model

    # Load question embeddings (JSONL)
    embeddings_path = Path(args.embeddings).resolve()
    if not embeddings_path.is_file():
        print(f"Embeddings file not found: {embeddings_path}", file=sys.stderr)
        return 1

    print("Loading question embeddings...", file=sys.stderr)
    questions: list[tuple[str, list[float], str]] = []
    with open(embeddings_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sentence = obj.get("sentence", "")
            emb = obj.get("embedding", [])
            source = obj.get("source", "")
            if sentence and emb:
                questions.append((sentence, emb, source))

    if not questions:
        print("No questions found in embeddings file.", file=sys.stderr)
        return 1
    print(f"Loaded {len(questions)} questions.", file=sys.stderr)

    # Load ONNX model and tokenizer
    print("Downloading/loading embedding model...", file=sys.stderr)
    model_dir, _arch = get_embedding_model("embeddinggemma-300m", variant=args.variant)
    onnx_path = _model_onnx_path(model_dir, args.variant)
    if not os.path.isfile(onnx_path):
        print(f"ONNX model not found at {onnx_path}", file=sys.stderr)
        return 1

    session = ort.InferenceSession(
        onnx_path,
        ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def embed_text(text: str) -> list[float]:
        encoded = tokenizer(
            text,
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
        emb = outputs[0]
        if emb.ndim == 2:
            emb = emb[0]
        vec = emb.astype(np.float32).tolist()
        return _normalize_l2(vec)

    # Gather query lines: from args, or interactive REPL, or batch stdin
    if args.queries:
        query_lines = [q.strip() for q in args.queries if q.strip()]
        if not query_lines:
            print("No non-empty queries provided.", file=sys.stderr)
            return 1
        for query in query_lines:
            _run_query(query, questions, embed_text, args.k)
        return 0

    if sys.stdin.isatty():
        # Interactive: prompt, read line, process on Enter, repeat until EOF (Ctrl-D)
        try:
            while True:
                try:
                    line = input("Query> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                _run_query(line, questions, embed_text, args.k)
        except KeyboardInterrupt:
            print(file=sys.stderr)
        return 0

    # Batch: read all lines from stdin (piped input)
    query_lines = [line.strip() for line in sys.stdin if line.strip()]
    if not query_lines:
        print("No queries provided. Pass strings as arguments or pipe lines to stdin.", file=sys.stderr)
        return 1
    for query in query_lines:
        _run_query(query, questions, embed_text, args.k)
    return 0


def _run_query(
    query: str,
    questions: list[tuple[str, list[float], str]],
    embed_text: Callable[[str], list[float]],
    k: int,
) -> None:
    """Embed query, find top-k closest questions, print results."""
    q_emb = embed_text(query)
    scored = [
        (_cosine_similarity(q_emb, emb), sentence, source)
        for sentence, emb, source in questions
    ]
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    print(f"Query: {query}")
    for i, (sim, sentence, source) in enumerate(top, 1):
        distance = 1.0 - sim
        if source:
            print(f"  {i}. [{distance:.4f}] {sentence}")
            print(f"      {source}")
        else:
            print(f"  {i}. [{distance:.4f}] {sentence}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
