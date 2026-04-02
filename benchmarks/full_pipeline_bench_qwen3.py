#!/usr/bin/env python3
"""
RASPUTIN Memory — LoCoMo Benchmark with Qwen3-Embedding (Phase 2)
Uses qwen3-embedding via port 8010 (OpenAI format), 1024-dim Matryoshka truncation
matching the production second_brain_v2 configuration.

Pipeline: vector (qwen3 1024d) + neural reranker + answer generation via vLLM
"""

import json
import math
import time
import hashlib
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
CHECKPOINT_FILE = RESULTS_DIR / "locomo_qwen3_checkpoint.json"
LOCOMO_FILE = BENCH_DIR / "locomo" / "locomo10.json"

QDRANT_URL = "http://localhost:6333"
EMBED_URL = "http://localhost:8010/v1/embeddings"
EMBED_MODEL = "qwen3-embedding"
EMBED_DIM = 1024  # Matryoshka truncation to match production second_brain_v2
RERANKER_URL = "http://localhost:8006/rerank"
VLLM_URL = "http://localhost:11435"
SEARCH_LIMIT = 20

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def http_json(url, data=None, method=None, timeout=60):
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def normalize_l2(vec):
    mag = math.sqrt(sum(x * x for x in vec))
    if mag == 0:
        return vec
    return [x / mag for x in vec]


def get_embedding(text, prefix="search_document: "):
    """Embed via vLLM qwen3-embedding, truncate to EMBED_DIM (Matryoshka)."""
    resp = http_json(
        EMBED_URL,
        data={
            "model": EMBED_MODEL,
            "input": prefix + text[:2000],
        },
        timeout=60,
    )
    vec = resp["data"][0]["embedding"]
    # Matryoshka: truncate to 1024 and L2 normalize
    vec = vec[:EMBED_DIM]
    return normalize_l2(vec)


def embed_batch(texts, prefix="search_document: ", batch_size=16):
    """Embed a batch of texts."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        padded = [prefix + t[:2000] for t in batch]
        resp = http_json(
            EMBED_URL,
            data={
                "model": EMBED_MODEL,
                "input": padded,
            },
            timeout=120,
        )
        for item in resp["data"]:
            vec = item["embedding"][:EMBED_DIM]
            all_vecs.append(normalize_l2(vec))
    return all_vecs


def create_collection(name):
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    time.sleep(0.3)
    http_json(
        f"{QDRANT_URL}/collections/{name}",
        data={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}, "optimizers_config": {"indexing_threshold": 0}},
        method="PUT",
    )
    print(f"  Created collection: {name} ({EMBED_DIM}d)")


def delete_collection(name):
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def commit_conversation(conv, collection):
    points = []
    committed = 0
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        date_key = f"session_{session_idx}_date_time"
        if session_key not in conv:
            break
        session_date = conv.get(date_key, f"Session {session_idx}")
        for turn in conv[session_key]:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            if not text:
                continue
            full_text = f"[{session_date}] {speaker}: {text}"
            point_id = int(hashlib.md5(full_text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": point_id,
                    "text": full_text,
                }
            )
            committed += 1
        session_idx += 1

    if not points:
        return 0

    # Batch embed
    texts = [p["text"] for p in points]
    print(f"  Embedding {len(texts)} turns with qwen3-embedding...")
    vecs = embed_batch(texts)

    # Upsert to Qdrant in batches
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch_pts = points[i : i + batch_size]
        batch_vecs = vecs[i : i + batch_size]
        qdrant_points = [
            {
                "id": p["id"],
                "vector": v,
                "payload": {
                    "text": p["text"][:4000],
                    "source": "locomo_bench",
                    "source_weight": 1.0,
                    "date": datetime.now().isoformat(),
                    "importance": 70,
                    "retrieval_count": 0,
                },
            }
            for p, v in zip(batch_pts, batch_vecs)
        ]
        http_json(
            f"{QDRANT_URL}/collections/{collection}/points",
            data={"points": qdrant_points},
            method="PUT",
            timeout=30,
        )

    time.sleep(2)  # indexing
    print(f"  Committed {committed} turns")
    return committed


def search(question, collection, limit=SEARCH_LIMIT):
    """Vector search + optional reranker."""
    q_vec = get_embedding(question, prefix="search_query: ")
    resp = http_json(
        f"{QDRANT_URL}/collections/{collection}/points/search",
        data={"vector": q_vec, "limit": limit, "with_payload": True},
    )
    results = [{"text": p["payload"]["text"], "score": p["score"]} for p in resp.get("result", [])]

    # Neural reranker
    try:
        texts = [r["text"] for r in results]
        rr = http_json(RERANKER_URL, data={"query": question, "passages": texts}, timeout=15)
        if "results" in rr:
            ranked = sorted(rr["results"], key=lambda x: x.get("score", 0), reverse=True)
            reranked = []
            for item in ranked:
                idx = item.get("index", 0)
                if 0 <= idx < len(results):
                    reranked.append({**results[idx], "rerank_score": item.get("score", 0)})
            results = reranked
    except Exception:
        pass  # reranker optional

    return results[:10]


def generate_answer(question, chunks, model):
    context = "\n\n".join(f"[Memory {i + 1}]: {c.get('text', '')}" for i, c in enumerate(chunks[:10]))
    prompt = f"""Answer the question using ONLY the memories below. Give the shortest possible answer (a few words or a short phrase). Do NOT explain or add context.

Memories:
{context}

Question: {question}
Answer:"""
    try:
        resp = http_json(
            f"{VLLM_URL}/v1/chat/completions",
            data={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning") or ""
        return content.strip()
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


def normalize_answer(s):
    s = s.lower()
    s = s.replace(",", " ").replace(".", " ").replace("'", " ").replace('"', " ")
    s = " ".join(s.split())
    return s


def compute_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len([t for t in pred_tokens if t in common]) / len(pred_tokens)
    recall = len([t for t in gold_tokens if t in common]) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def detect_vllm_model():
    try:
        data = http_json(f"{VLLM_URL}/v1/models")
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"  vLLM model: {models[0]}")
            return models[0]
    except Exception as e:
        print(f"  WARNING: {e}")
    return "qwen3.5-122b-a10b"


def run_benchmark():
    model = detect_vllm_model()

    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
    all_results = checkpoint.get("results", {})

    for idx, conv_data in enumerate(conversations):
        conv_id = conv_data.get("sample_id", f"conv-{idx}")
        if conv_id in all_results:
            print(f"\n[{idx + 1}/10] Skipping {conv_id} (done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/10] {conv_id}")
        print(f"{'=' * 60}")

        collection = f"locomo_qwen3_{conv_id.replace('-', '_')}"
        try:
            create_collection(collection)
            n_committed = commit_conversation(conv_data["conversation"], collection)

            qa_list = conv_data.get("qa", [])
            print(f"  Running {len(qa_list)} questions...")
            qa_results = []

            for qi, qa in enumerate(qa_list):
                question = qa["question"]
                gold = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)
                if not gold:
                    continue

                chunks = search(question, collection)
                pred = generate_answer(question, chunks, model)
                f1 = compute_f1(pred, gold)

                qa_results.append(
                    {
                        "question": question,
                        "ground_truth": gold,
                        "prediction": pred,
                        "f1": f1,
                        "category": category,
                        "n_chunks": len(chunks),
                    }
                )

                time.sleep(0.5)

                if (qi + 1) % 20 == 0:
                    avg = sum(r["f1"] for r in qa_results) / len(qa_results)
                    print(f"    {qi + 1}/{len(qa_list)} questions, F1={avg:.4f}")

            conv_f1 = sum(r["f1"] for r in qa_results) / len(qa_results) if qa_results else 0
            cat_scores = defaultdict(list)
            for r in qa_results:
                cat_scores[r["category"]].append(r["f1"])
            cat_f1 = {cat: sum(scores) / len(scores) for cat, scores in cat_scores.items()}

            all_results[conv_id] = {
                "conv_id": conv_id,
                "n_questions": len(qa_results),
                "n_committed": n_committed,
                "mean_f1": conv_f1,
                "category_f1": cat_f1,
                "details": qa_results,
            }

            print(f"  ✅ {conv_id}: F1={conv_f1:.4f} ({len(qa_results)} Qs)")
            for cat, score in sorted(cat_f1.items()):
                print(f"     {CATEGORY_NAMES.get(cat, f'cat-{cat}')}: {score:.4f} ({len(cat_scores[cat])} Qs)")

            checkpoint["results"] = all_results
            checkpoint["last_updated"] = datetime.now().isoformat()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            delete_collection(collection)

    return all_results


if __name__ == "__main__":
    results = run_benchmark()

    # Save final results
    from collections import defaultdict

    all_scores = []
    cat_scores = defaultdict(list)
    for v in results.values():
        for qa in v.get("details", []):
            all_scores.append(qa["f1"])
            cat_scores[qa["category"]].append(qa["f1"])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n{'=' * 60}")
    print(f"OVERALL F1 (Qwen3-Embedding 1024d): {overall:.4f} ({len(all_scores)} Qs)")
    for cat in sorted(cat_scores.keys()):
        scores = cat_scores[cat]
        print(f"  {CATEGORY_NAMES.get(cat, '?')}: {sum(scores) / len(scores):.4f} ({len(scores)} Qs)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "locomo-fullpipeline-qwen3embed.json", "w") as f:
        json.dump(
            {
                "metadata": {
                    "embed": "qwen3-embedding",
                    "dim": 1024,
                    "pipeline": "vector(qwen3-1024d-matryoshka)+reranker",
                    "n_convs": len(results),
                    "overall_f1": overall,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print("\nSaved to benchmarks/results/locomo-fullpipeline-qwen3embed.json")
