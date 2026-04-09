#!/usr/bin/env python3
"""
Find the three closest FAQ questions to input text using cosine similarity.

Loads question embeddings from a JSONL file with "sentence", "embedding", optional "source"
(path#section-slug), and optional short "answer" when present (e.g. from embed-lines-onnx).
loads the same embedding model (Gemma 300M via ONNX) to embed the input text,
then for each input line prints the three closest questions and their distance
(1 - cosine_similarity; 0 = identical, 2 = opposite).

Usage:
  python question_search.py   # interactive: type a sentence, press Enter, see results

Dependencies: moonshine-voice, onnxruntime, transformers, numpy
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer
from moonshine_voice import get_embedding_model


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


class QuestionSearch:
    """
    Loads FAQ question embeddings and an embedding model, and provides
    query(text, n) to return the top n closest questions by cosine similarity.
    """

    def __init__(
        self,
        embeddings_path: Path | str,
        *,
        variant: str = "fp32",
        tokenizer_name: str = "google/embeddinggemma-300m",
        max_length: int = 2048,
    ) -> None:
        embeddings_path = Path(embeddings_path).resolve()
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        self._questions: list[
            tuple[str, list[float], str, str | None, str | None]
        ] = []
        with open(embeddings_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sentence = obj.get("sentence", "")
                emb = obj.get("embedding", [])
                source = obj.get("source", "")
                raw_ans = obj.get("answer")
                raw_stub = obj.get("stub")
                answer: str | None = None
                if isinstance(raw_ans, str) and raw_ans.strip():
                    answer = raw_ans.strip()
                stub: str | None = None
                if isinstance(raw_stub, str) and raw_stub.strip():
                    stub = raw_stub.strip()
                if sentence and emb:
                    self._questions.append((sentence, emb, source, answer, stub))

        if not self._questions:
            raise ValueError("No questions found in embeddings file.")

        model_dir, _arch = get_embedding_model("embeddinggemma-300m", variant=variant)
        onnx_path = _model_onnx_path(model_dir, variant)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        self._session = ort.InferenceSession(
            onnx_path,
            ort.SessionOptions(),
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="models-cache")
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._questions)

    def _embed_text(self, text: str) -> list[float]:
        encoded = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        outputs = self._session.run(
            ["sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        emb = outputs[0]
        if emb.ndim == 2:
            emb = emb[0]
        vec = emb.astype(np.float32).tolist()
        return _normalize_l2(vec)

    def query(
        self, text: str, n: int = 3
    ) -> list[tuple[float, str, str, str | None, str | None]]:
        """
        Return the top n closest questions to the given text.

        Returns a list of (similarity, sentence, source, answer, stub) tuples, ordered by
        similarity descending. ``stub`` is the section heading from embeddings JSON when
        present. Distance for display is 1.0 - similarity.
        """
        q_emb = self._embed_text(text)
        scored = [
            (_cosine_similarity(q_emb, emb), sentence, source, answer, stub)
            for sentence, emb, source, answer, stub in self._questions
        ]
        scored.sort(key=lambda x: -x[0])
        return scored[:n]


def _print_results(
    query: str, results: list[tuple[float, str, str, str | None, str | None]]
) -> None:
    """Print query and top results (including short answer when available)."""
    print(f"Query: {query}")
    for i, (sim, sentence, source, answer, stub) in enumerate(results, 1):
        distance = 1.0 - sim
        print(f"  {i}. [{distance:.4f}] {sentence}")
        if source:
            print(f"      {source}")
        if answer:
            print(f"      A: {answer}")
    print()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_embeddings = script_dir / "question-embeddings.json"

    parser = argparse.ArgumentParser(
        description="Find closest FAQ questions by embedding similarity"
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

    print("Loading question embeddings...", file=sys.stderr)
    try:
        search = QuestionSearch(
            args.embeddings,
            variant=args.variant,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Loaded {len(search)} questions.", file=sys.stderr)

    def run_query(q: str) -> None:
        results = search.query(q, n=args.k)
        _print_results(q, results)

    try:
        while True:
            try:
                line = input("Query> ").strip()
            except EOFError:
                break
            if not line:
                continue
            run_query(line)
    except KeyboardInterrupt:
        print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
