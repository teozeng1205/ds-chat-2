#!/usr/bin/env python3
"""Synthetic QA evaluation harness for the DS Chat semantic KB.

Generates N question/answer pairs from KB chunks using gpt-5.4-mini,
then measures recall@k — whether the source chunk appears in the top-k
retrieval results for each question.

Usage:
    # Baseline (before improvements):
    .venv/bin/python scripts/eval_kb_recall.py --n-pairs 50 --top-k 8

    # After rebuilding with contextual retrieval:
    .venv/bin/python scripts/eval_kb_recall.py --n-pairs 50 --top-k 8

    # Write QA pairs to disk so the same set is reused for comparison:
    .venv/bin/python scripts/eval_kb_recall.py --save-pairs /tmp/qa_pairs.json
    .venv/bin/python scripts/eval_kb_recall.py --load-pairs /tmp/qa_pairs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from openai import OpenAI

log = logging.getLogger(__name__)


def _generate_qa_pairs(
    chunks: list[dict[str, Any]],
    client: OpenAI,
    *,
    n: int = 50,
    model: str = "gpt-5.4-mini",
) -> list[dict[str, Any]]:
    """Generate one question per sampled chunk. Returns [{question, chunk_id, kind}]."""
    sample = random.sample(chunks, min(n, len(chunks)))
    pairs: list[dict[str, Any]] = []
    for chunk in sample:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are writing test questions for a data-pipeline knowledge base.\n"
                        "Write ONE specific question whose PRECISE answer can be found in the "
                        "text below. The question must use different words than the text "
                        "(no copy-paste) so it tests semantic recall.\n"
                        "Return ONLY the question, no preamble.\n\n"
                        f"Text:\n{chunk['text'][:600]}"
                    ),
                }],
                max_completion_tokens=80,
                temperature=0.5,
            )
            question = resp.choices[0].message.content.strip()
            if question:
                pairs.append({
                    "question": question,
                    "chunk_id": chunk["id"],
                    "kind": chunk.get("kind", "?"),
                    "source": (chunk.get("metadata") or {}).get("source", "?"),
                })
        except Exception as exc:  # noqa: BLE001
            log.debug("QA gen failed for %s: %s", chunk["id"], exc)
    return pairs


def _measure_recall(
    pairs: list[dict[str, Any]],
    client: OpenAI,
    *,
    top_k: int = 8,
    model: str = "text-embedding-3-large",
) -> dict[str, Any]:
    """Embed each question and measure whether its source chunk is in top-k."""
    from app.investigation.semantic_index import SemanticIndex, tokenize

    index_path = BACKEND_ROOT / "app" / ".data" / "ds-chat-semantic.sqlite"
    if not index_path.exists():
        raise RuntimeError(f"Semantic index not found at {index_path}. Run build_embeddings first.")

    idx = SemanticIndex(index_path)
    hits = misses = 0
    by_kind: dict[str, dict[str, int]] = {}
    miss_examples: list[dict] = []
    try:
        for pair in pairs:
            question = pair["question"]
            target_id = pair["chunk_id"]
            kind = pair.get("kind", "?")

            try:
                resp = client.embeddings.create(model=model, input=[question])
                q_vec = list(resp.data[0].embedding)
                results = idx.hybrid_search(q_vec, lexical_terms=tokenize(question), top_k=top_k)
                found_ids = {r.id for r in results}
            except Exception as exc:  # noqa: BLE001
                log.debug("search failed: %s", exc)
                continue

            by_kind.setdefault(kind, {"hits": 0, "total": 0})
            by_kind[kind]["total"] += 1

            if target_id in found_ids:
                hits += 1
                by_kind[kind]["hits"] += 1
            else:
                misses += 1
                if len(miss_examples) < 5:
                    miss_examples.append({
                        "question": question,
                        "expected": target_id,
                        "got": [r.id for r in results[:3]],
                    })
    finally:
        idx.close()

    total = hits + misses
    recall = hits / total if total > 0 else 0.0
    by_kind_recall = {
        k: round(v["hits"] / v["total"], 3) if v["total"] else 0
        for k, v in by_kind.items()
    }
    return {
        "top_k": top_k,
        "total_pairs": total,
        "hits": hits,
        "misses": misses,
        "recall_at_k": round(recall, 4),
        "by_kind": by_kind_recall,
        "miss_examples": miss_examples,
    }


def _load_index_chunks() -> list[dict[str, Any]]:
    """Read every chunk from the semantic index as dicts."""
    import sqlite3

    index_path = BACKEND_ROOT / "app" / ".data" / "ds-chat-semantic.sqlite"
    conn = sqlite3.connect(index_path)
    try:
        rows = conn.execute(
            "SELECT id, text, kind, COALESCE(metadata, '{}') FROM chunks"
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r[0],
            "text": r[1],
            "kind": r[2],
            "metadata": json.loads(r[3] or "{}"),
        }
        for r in rows
    ]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-pairs", type=Path, default=None,
                   help="Save generated QA pairs for reuse across runs")
    p.add_argument("--load-pairs", type=Path, default=None,
                   help="Load previously saved QA pairs instead of generating new ones")
    p.add_argument("--model", default="gpt-5.4-mini", help="QA generation model")
    args = p.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return 1

    random.seed(args.seed)
    client = OpenAI()

    if args.load_pairs and args.load_pairs.exists():
        pairs = json.loads(args.load_pairs.read_text(encoding="utf-8"))
        print(f"Loaded {len(pairs)} QA pairs from {args.load_pairs}")
    else:
        print("Loading index chunks…")
        chunks = _load_index_chunks()
        # Filter to meaningful chunk kinds only
        chunks = [c for c in chunks if c["kind"] in ("doc", "tables", "sql_best_practices", "pipeline")]
        print(f"  {len(chunks)} chunks available; generating {args.n_pairs} QA pairs")
        pairs = _generate_qa_pairs(chunks, client, n=args.n_pairs, model=args.model)
        print(f"  Generated {len(pairs)} QA pairs")
        if args.save_pairs:
            args.save_pairs.write_text(json.dumps(pairs, indent=2, ensure_ascii=True), encoding="utf-8")
            print(f"  Saved pairs → {args.save_pairs}")

    if not pairs:
        print("No QA pairs available")
        return 1

    print(f"\nMeasuring recall@{args.top_k}…")
    result = _measure_recall(pairs, client, top_k=args.top_k)
    print(json.dumps(result, indent=2))

    r = result["recall_at_k"]
    grade = "✅ GOOD" if r >= 0.75 else "⚠️  OK" if r >= 0.60 else "❌ LOW"
    print(f"\nRecall@{args.top_k}: {r:.1%}  {grade}")
    for kind, kr in sorted(result["by_kind"].items()):
        print(f"  {kind:<20} {kr:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
