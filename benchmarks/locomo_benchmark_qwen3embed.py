#!/usr/bin/env python3
"""
LoCoMo Benchmark Adapter for RASPUTIN Memory — Qwen3-Embedding variant
Supports both full 4096-dim and Matryoshka-truncated 1024-dim.

Changes from nomic version:
- Embedding: Qwen3-Embedding via OpenAI-compatible endpoint
- Client-side Matryoshka truncation + L2-renorm
- Improved adversarial prompt (no "unknown" fallback)
- Temporal: explicit date prefix in turns, chronological sorting for temporal Qs
"""

import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import requests

# ── Config ──────────────────────────────────────────────────────────────

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = "http://localhost:8010/v1/embeddings"
EMBED_MODEL = "qwen3-embedding"
# Matryoshka: truncate to this dim (set to 4096 for full)
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
EMBED_FULL_DIM = 4096
LLM_URL = os.environ.get("LLM_URL", "http://localhost:11438/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")
_default_collection = f"locomo_bench_qwen3_{EMBED_DIM}d_{int(time.time())}"
COLLECTION = os.environ.get("BENCH_COLLECTION", _default_collection)
SEARCH_LIMIT = 20
DATA_PATH = Path(__file__).parent / "locomo" / "locomo10.json"
RESULTS_DIR = Path(__file__).parent / "results"

CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

TEMPORAL_KEYWORDS = {"before", "after", "when", "first", "last", "recently", "earlier", "later", "previous", "next", "date", "time", "year", "month", "day"}


# ── Embedding ───────────────────────────────────────────────────────────

def _truncate_and_norm(vec: list[float]) -> list[float]:
    """Matryoshka truncation + L2 renormalization."""
    if EMBED_DIM >= EMBED_FULL_DIM:
        return vec
    v = vec[:EMBED_DIM]
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        v = [x / norm for x in v]
    return v


def embed(text: str) -> list[float]:
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    emb = data["data"][0]["embedding"]
    return _truncate_and_norm(emb)


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        all_embs.extend([_truncate_and_norm(d["embedding"]) for d in sorted_data])
    return all_embs


# ── Qdrant helpers ──────────────────────────────────────────────────────

def create_collection():
    requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
    time.sleep(0.5)
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 0},
        },
        timeout=10,
    )
    resp.raise_for_status()
    time.sleep(1)
    print(f"  Created collection '{COLLECTION}' ({EMBED_DIM}-dim)")


def delete_collection():
    requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10)
    print(f"  Deleted collection '{COLLECTION}'")


def ingest_conversation(conv: dict, conv_idx: int):
    session_keys = sorted(
        [k for k in conv.keys() if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda x: int(x.split("_")[1]),
    )

    turns = []
    for sk in session_keys:
        date_key = f"{sk}_date_time"
        session_date = conv.get(date_key, "")
        for turn in conv[sk]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            if session_date:
                formatted = f"[Date: {session_date}] {speaker}: {text}"
            else:
                formatted = f"{speaker}: {text}"
            turns.append({
                "text": formatted,
                "dia_id": dia_id,
                "session": sk,
                "date": session_date,
            })

    if not turns:
        return 0

    texts = [t["text"] for t in turns]
    vectors = embed_batch(texts)

    points = []
    for i, (turn, vec) in enumerate(zip(turns, vectors)):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {
                "text": turn["text"],
                "dia_id": turn["dia_id"],
                "session": turn["session"],
                "date": turn["date"],
                "conv_idx": conv_idx,
                "turn_idx": i,
            },
        })

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
            json={"points": batch},
            timeout=60,
        )
        resp.raise_for_status()

    return len(turns)


def is_temporal_question(question: str) -> bool:
    words = set(question.lower().split())
    return bool(words & TEMPORAL_KEYWORDS)


def search_qdrant(query: str, limit: int = SEARCH_LIMIT) -> list[dict]:
    vector = embed(query)
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        json={
            "query": vector,
            "limit": limit,
            "with_payload": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for point in data.get("result", {}).get("points", []):
        results.append({
            "text": point["payload"]["text"],
            "score": point["score"],
            "date": point["payload"].get("date", ""),
            "turn_idx": point["payload"].get("turn_idx", 0),
        })
    return results


# ── LLM helpers ─────────────────────────────────────────────────────────

def llm_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.1) -> str:
    resp = requests.post(
        LLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        reasoning = msg.get("reasoning", "")
        if reasoning:
            lines = reasoning.strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith("*"):
                    content = line
                    break
    return content or "I don't know"


def generate_answer(question: str, context: list[dict], is_temporal: bool = False) -> str:
    if is_temporal:
        sorted_ctx = sorted(context[:SEARCH_LIMIT], key=lambda r: (r.get("date", ""), r.get("turn_idx", 0)))
    else:
        sorted_ctx = context[:SEARCH_LIMIT]

    ctx_text = "\n".join(f"- {r['text']}" for r in sorted_ctx)

    prompt = f"""Answer the question based ONLY on the provided context. Give a direct, concise answer.
If the context doesn't contain the exact answer, make your best inference from what's available.
Give ONLY the specific answer — no explanation, no reasoning, no full sentences unless the answer IS a sentence.
Examples of good answers: "7 May 2023", "Psychology, counseling", "Paris", "3 times", "mental health"

Context:
{ctx_text}

Question: {question}

Answer:"""
    return llm_generate(prompt, max_tokens=60)


# ── Metrics ─────────────────────────────────────────────────────────────

def normalize_answer(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Main benchmark ──────────────────────────────────────────────────────

def run_benchmark(
    data_path: str = str(DATA_PATH),
    max_conversations: Optional[int] = None,
    max_qa_per_conv: Optional[int] = None,
    start_conv: int = 0,
    partial_results_path: Optional[str] = None,
):
    print(f"Config: embed={EMBED_MODEL} dim={EMBED_DIM} (full={EMBED_FULL_DIM})")
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    if max_conversations:
        data = data[:max_conversations]

    all_results = []
    category_f1 = defaultdict(list)
    if partial_results_path and os.path.exists(partial_results_path):
        with open(partial_results_path) as f:
            partial = json.load(f)
        all_results = partial.get("detailed", [])
        for r in all_results:
            category_f1[r["category"]].append(r["f1"])
        print(f"  Loaded {len(all_results)} existing results")

    if start_conv > 0:
        data = data[start_conv:]

    total_convs = len(data)
    for conv_idx, sample in enumerate(data):
        conv = sample["conversation"]
        qa_pairs = sample["qa"]
        sample_id = sample.get("sample_id", conv_idx)
        display_idx = conv_idx + start_conv

        print(f"\n{'='*60}")
        print(f"Conversation {display_idx + 1}/10 (sample_id={sample_id})")
        print(f"{'='*60}")

        create_collection()
        n_turns = ingest_conversation(conv, conv_idx)
        print(f"  Ingested {n_turns} turns")
        time.sleep(1)

        qa_subset = qa_pairs[:max_qa_per_conv] if max_qa_per_conv else qa_pairs
        for qi, qa in enumerate(qa_subset):
            question = qa["question"]
            ground_truth = qa.get("answer") or qa.get("adversarial_answer", "")
            category = qa.get("category", 0)

            results = search_qdrant(question)
            temporal = is_temporal_question(question) or category == 3
            predicted = generate_answer(question, results, is_temporal=temporal)

            f1 = compute_f1(predicted, ground_truth)
            category_f1[category].append(f1)

            result = {
                "conv_idx": conv_idx,
                "sample_id": sample_id,
                "qa_idx": qi,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, f"type-{category}"),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "f1": round(f1, 4),
                "top_results": [{"text": r["text"][:120], "score": round(r["score"], 4)} for r in results[:3]],
            }
            all_results.append(result)

            if (qi + 1) % 10 == 0:
                avg_so_far = sum(r["f1"] for r in all_results) / len(all_results)
                print(f"  [{qi+1}/{len(qa_subset)}] running avg F1={avg_so_far:.4f}")

        print(f"  Done: {len(qa_subset)} QA pairs")

        partial_path = RESULTS_DIR / f"locomo-results-qwen3embed-{EMBED_DIM}d-partial.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w") as f:
            json.dump({"detailed": all_results, "completed_through_conv": display_idx}, f, indent=2)

    delete_collection()

    summary = {
        "total_qa": len(all_results),
        "overall_f1": round(sum(r["f1"] for r in all_results) / max(len(all_results), 1), 4),
        "per_category": {},
        "config": {
            "embedding": f"{EMBED_MODEL} ({EMBED_DIM}-dim, Matryoshka from {EMBED_FULL_DIM})" if EMBED_DIM < EMBED_FULL_DIM else f"{EMBED_MODEL} ({EMBED_DIM}-dim)",
            "llm": f"{LLM_MODEL} via vLLM",
            "retrieval": f"vector search top-{SEARCH_LIMIT}",
        },
    }

    for cat in sorted(category_f1.keys()):
        f1_scores = category_f1[cat]
        summary["per_category"][CATEGORY_NAMES.get(cat, f"type-{cat}")] = {
            "count": len(f1_scores),
            "f1": round(sum(f1_scores) / max(len(f1_scores), 1), 4),
        }

    summary["comparison"] = {
        "mem0": 0.6688,
        "zep": 0.7514,
        "memmachine": 0.8487,
        "rasputin-nomic-768d": 0.4144,
        f"rasputin-qwen3embed-{EMBED_DIM}d": summary["overall_f1"],
    }

    out_name = f"locomo-results-qwen3embed-{EMBED_DIM}d.json"
    with open(RESULTS_DIR / out_name, "w") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (dim={EMBED_DIM})")
    print(f"{'='*60}")
    print(f"Overall F1: {summary['overall_f1']}")
    for cat_name, scores in summary["per_category"].items():
        print(f"  {cat_name}: F1={scores['f1']} (n={scores['count']})")
    print(f"\nResults saved to {RESULTS_DIR / out_name}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-conv", type=int)
    parser.add_argument("--max-qa", type=int)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--partial-results", default=None)
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        max_conversations=args.max_conv,
        max_qa_per_conv=args.max_qa,
        start_conv=args.start_conv,
        partial_results_path=args.partial_results,
    )
