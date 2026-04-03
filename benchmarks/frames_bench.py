#!/usr/bin/env python3
"""
RASPUTIN Memory — FRAMES Benchmark (Google Research 2024)
824 multi-hop factual reasoning questions over Wikipedia evidence.
Dataset: google/frames-benchmark

Usage:
    python3 benchmarks/frames_bench.py [--reset] [--limit N]
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
DATASET_FILE = BENCH_DIR / "frames" / "test.json"
CHECKPOINT_FILE = RESULTS_DIR / "frames-checkpoint.json"
OUTPUT_FILE = RESULTS_DIR / "frames-results.json"
COMPARISON_FILE = RESULTS_DIR / "frames-comparison.md"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("BENCH_EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
BENCH_PORT = 7782

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPUS_MODEL = "claude-opus-4-6"
JUDGE_MODEL = "gpt-4o-mini"

SEARCH_LIMIT = 60
CONTEXT_CHUNKS = 50


# ─── HTTP helpers ────────────────────────────────────────────


def http_json(url, data=None, method=None, timeout=60, headers=None):
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_wikipedia_text(url):
    title = url.rsplit("/", 1)[-1]
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "RASPUTIN-Memory-Bench/0.7")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "")
        if not extract:
            return None
        return f"[{data.get('title', title)}] {extract}"
    except Exception:
        return None


# ─── Embedding ───────────────────────────────────────────────


def get_embedding(text, prefix="search_document: "):
    prefixed = f"{prefix}{text}" if prefix else text
    result = http_json(EMBED_URL, data={"model": EMBED_MODEL, "input": prefixed}, timeout=30)
    if "embeddings" in result:
        return result["embeddings"][0]
    if "data" in result:
        return result["data"][0]["embedding"]
    raise ValueError(f"Unexpected embed response: {list(result.keys())}")


# ─── Deduplication ───────────────────────────────────────────


def deduplicate_results(results, overlap_threshold=0.75):
    if not results:
        return results
    tokenize = re.compile(r"\w+", re.UNICODE)
    selected = []
    selected_token_sets = []
    for r in results:
        tokens = set(tokenize.findall((r.get("text") or "").lower()))
        if not tokens:
            continue
        is_dup = False
        for existing_tokens in selected_token_sets:
            if not existing_tokens:
                continue
            overlap = len(tokens & existing_tokens) / min(len(tokens), len(existing_tokens))
            if overlap > overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            selected.append(r)
            selected_token_sets.append(tokens)
    return selected


# ─── LLM: Opus for answers ──────────────────────────────────


def generate_opus_answer(question, context_chunks, max_chunks=CONTEXT_CHUNKS):
    context = "\n".join(f"- {c.get('text', c) if isinstance(c, dict) else c}" for c in context_chunks[:max_chunks])
    prompt = f"""You are answering factual questions using retrieved reference passages.
Answer concisely and precisely. Give only the specific answer requested — no explanations.

Reference passages:
{context}

Question: {question}
Answer:"""

    for attempt in range(5):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(
                    {
                        "model": OPUS_MODEL,
                        "max_tokens": 100,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                ).encode(),
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("x-api-key", ANTHROPIC_API_KEY)
            req.add_header("anthropic-version", "2023-06-01")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            return data["content"][0]["text"].strip()
        except Exception:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise


# ─── LLM: GPT-4o-mini for judging ───────────────────────────


def judge_gpt4o_mini(question, prediction, ground_truth):
    prompt = f"""You are evaluating a factual QA system's answer.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

Is the system's answer correct? Be generous with formatting differences (e.g., "Jane Ballou" vs "jane ballou"). Score CORRECT if the essential factual content matches. Score WRONG only if the answer is factually incorrect or missing.

Reply with exactly one word: CORRECT or WRONG"""

    data = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    for attempt in range(5):
        try:
            result = http_json(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=30,
            )
            text = result["choices"][0]["message"]["content"].strip()
            return 1.0 if "CORRECT" in text.upper() else 0.0
        except Exception:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise


# ─── Qdrant operations ──────────────────────────────────────


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


def delete_collection(name):
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def commit_wikipedia_pages(wiki_urls, collection):
    points = []
    committed = 0
    for url in wiki_urls:
        text = fetch_wikipedia_text(url)
        if not text or len(text) < 20:
            continue
        try:
            vec = get_embedding(text[:2000], prefix="search_document: ")
            point_id = int(hashlib.md5(text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": text[:4000],
                        "source": "frames_wiki",
                        "source_weight": 1.0,
                        "date": datetime.now().isoformat(),
                        "importance": 80,
                        "retrieval_count": 0,
                    },
                }
            )
            committed += 1
        except Exception:
            pass
        time.sleep(0.1)

    if points:
        http_json(f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30)
    time.sleep(0.5)
    return committed


# ─── Server management ───────────────────────────────────────


def start_bench_server(collection, port=BENCH_PORT):
    env = os.environ.copy()
    env["QDRANT_COLLECTION"] = collection
    env["PORT"] = str(port)
    env["DISABLE_FALKORDB"] = "true"
    env["DISABLE_RERANKER"] = "true"
    env["RERANKER_ENABLED"] = "false"
    env["LLM_RERANKER"] = "false"
    env["RATE_LIMIT_SEARCH"] = "0"
    env["EMBED_URL"] = EMBED_URL
    env["EMBED_MODEL"] = EMBED_MODEL
    env["EMBED_PREFIX_QUERY"] = "search_query: "
    env["EMBED_PREFIX_DOC"] = "search_document: "
    env["PYTHONPATH"] = str(REPO / "tools")

    server_log = RESULTS_DIR / "frames-server.log"
    server_log_fh = open(server_log, "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "tools" / "hybrid_brain.py"), "--port", str(port)],
        cwd=str(REPO / "tools"),
        env=env,
        stdout=server_log_fh,
        stderr=server_log_fh,
    )
    url = f"http://localhost:{port}/health"
    for _ in range(30):
        time.sleep(1)
        try:
            http_json(url)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("Server died")
    proc.kill()
    raise RuntimeError("Server failed to start in 30s")


def kill_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Search ──────────────────────────────────────────────────


def search_query(query, port=BENCH_PORT, limit=SEARCH_LIMIT):
    url = f"http://localhost:{port}/search"
    params = urllib.parse.urlencode({"q": query, "limit": limit, "expand": "false"})
    for attempt in range(4):
        try:
            result = http_json(f"{url}?{params}", timeout=60)
            return result.get("results", [])
        except Exception:
            if attempt < 3:
                time.sleep(2**attempt)
            else:
                return []


# ─── Checkpoint ──────────────────────────────────────────────


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(state):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Report generation ───────────────────────────────────────


def generate_report(state):
    results = state.get("results", {})
    if not results:
        return

    by_type = {}
    all_scores = []
    for qid, info in results.items():
        score = info.get("judge_score", 0)
        for rtype in info.get("reasoning_types", "unknown").split(" | "):
            rtype = rtype.strip()
            by_type.setdefault(rtype, []).append(score)
        all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0

    lines = [
        "# RASPUTIN Memory — FRAMES Benchmark (Google Research 2024)",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Pipeline:** Wikipedia fetch → Embed → Search (top-{SEARCH_LIMIT}) → Claude Opus 4 → GPT-4o-mini judge",
        f"**Total questions:** {len(all_scores)}",
        f"\n## Overall Accuracy: {overall:.2f}%",
        "\n## Per-Reasoning-Type Breakdown",
    ]

    for rtype in sorted(by_type.keys()):
        scores = by_type[rtype]
        acc = sum(scores) / len(scores) * 100 if scores else 0
        lines.append(f"- **{rtype}**: {acc:.1f}% ({sum(1 for s in scores if s > 0)}/{len(scores)})")

    lines.append("\n## Comparison")
    lines.append("| System | Accuracy |")
    lines.append("|--------|----------|")
    lines.append(f"| **RASPUTIN Memory v0.7** | **{overall:.2f}%** |")

    report = "\n".join(lines)
    with open(COMPARISON_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nReport: {COMPARISON_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "overall_accuracy": overall,
                "per_type": {t: sum(s) / len(s) * 100 for t, s in by_type.items()},
                "total_questions": len(all_scores),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results: {OUTPUT_FILE}")


# ─── Main pipeline ───────────────────────────────────────────


def run_benchmark(dataset, limit=None):
    state = load_checkpoint()
    results = state.get("results", {})
    questions = dataset[:limit] if limit else dataset

    done_ids = set(results.keys())
    pending = [(i, q) for i, q in enumerate(questions) if str(q.get("Unnamed: 0", i)) not in done_ids]

    print(f"FRAMES: {len(questions)} questions, {len(done_ids)} already done, {len(pending)} pending\n")

    proc = None
    current_collection = None

    try:
        for idx, (qi, q) in enumerate(pending):
            qid = str(q.get("Unnamed: 0", qi))
            question = q["Prompt"]
            answer = q["Answer"]
            reasoning = q.get("reasoning_types", "unknown")

            wiki_links_raw = q.get("wiki_links", "[]")
            if isinstance(wiki_links_raw, str):
                try:
                    wiki_links = json.loads(wiki_links_raw.replace("'", '"'))
                except json.JSONDecodeError:
                    wiki_links = []
            else:
                wiki_links = wiki_links_raw or []

            if not wiki_links:
                wiki_links = [q.get(f"wikipedia_link_{i}") for i in range(1, 12) if q.get(f"wikipedia_link_{i}")]

            collection = f"frames_{hashlib.md5(qid.encode()).hexdigest()[:12]}"

            print(f"[{idx + 1}/{len(pending)}] Q{qid} ({reasoning[:40]})")

            kill_server(proc)
            if current_collection:
                delete_collection(current_collection)

            create_collection(collection)
            committed = commit_wikipedia_pages(wiki_links, collection)
            print(f"  Committed {committed} wiki pages from {len(wiki_links)} links")

            if committed == 0:
                results[qid] = {
                    "question_id": qid,
                    "reasoning_types": reasoning,
                    "question": question,
                    "ground_truth": answer,
                    "prediction": "No evidence available",
                    "judge_score": 0.0,
                    "num_chunks": 0,
                }
                state["results"] = results
                save_checkpoint(state)
                print("  ✗ (no evidence)")
                current_collection = collection
                continue

            proc = start_bench_server(collection, BENCH_PORT)
            current_collection = collection

            chunks = search_query(question, port=BENCH_PORT, limit=SEARCH_LIMIT)
            chunks = deduplicate_results(chunks)

            if not chunks:
                prediction = "I don't have enough information."
            else:
                prediction = generate_opus_answer(question, chunks)

            score = judge_gpt4o_mini(question, prediction, answer)

            results[qid] = {
                "question_id": qid,
                "reasoning_types": reasoning,
                "question": question,
                "ground_truth": answer,
                "prediction": prediction,
                "judge_score": score,
                "num_chunks": len(chunks),
            }

            state["results"] = results
            save_checkpoint(state)

            status = "✓" if score > 0 else "✗"
            total_done = len(results)
            total_correct = sum(1 for r in results.values() if r.get("judge_score", 0) > 0)
            print(f"  {status} ({total_correct}/{total_done} = {total_correct / total_done * 100:.1f}%)")

    finally:
        kill_server(proc)
        if current_collection:
            delete_collection(current_collection)

    generate_report(state)


# ─── Entry point ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN Memory — FRAMES Benchmark")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared")

    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}")
        print(
            "Download: python3 -c \"from datasets import load_dataset; load_dataset('google/frames-benchmark')['test'].to_json('benchmarks/frames/test.json')\""
        )
        sys.exit(1)

    with open(DATASET_FILE) as f:
        dataset = [json.loads(line) for line in f]

    print(f"Loaded {len(dataset)} questions")
    types = Counter(d.get("reasoning_types", "unknown") for d in dataset)
    for t, c in types.most_common(5):
        print(f"  {t}: {c}")
    print()

    run_benchmark(dataset, limit=args.limit)


if __name__ == "__main__":
    main()
