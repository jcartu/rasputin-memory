#!/usr/bin/env python3
"""
RASPUTIN Memory — LongMemEval Benchmark (ICLR 2025)
500 conversational memory questions across 6 categories.
Dataset: xiaowu0162/longmemeval-cleaned (oracle split)

Usage:
    python3 benchmarks/longmemeval_bench.py [--reset] [--limit N]
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
DATASET_FILE = BENCH_DIR / "longmemeval" / "longmemeval_oracle.json"
CHECKPOINT_FILE = RESULTS_DIR / "longmemeval-checkpoint.json"
OUTPUT_FILE = RESULTS_DIR / "longmemeval-results.json"
COMPARISON_FILE = RESULTS_DIR / "longmemeval-comparison.md"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("BENCH_EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
BENCH_PORT = 7781

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANSWER_MODEL = os.environ.get("BENCH_ANSWER_MODEL", "claude-haiku-4-5-20251001")
BENCH_MODE = os.environ.get("BENCH_MODE", "production")
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gpt-4o-mini-2024-07-18")

SEARCH_LIMIT = int(os.environ.get("BENCH_SEARCH_LIMIT", "60"))
CONTEXT_CHUNKS = int(os.environ.get("BENCH_CONTEXT_CHUNKS", "60"))

_DEFAULT_JUDGE_PROMPT = (
    "Is the system's answer correct? Score CORRECT only if the answer contains the specific "
    "information asked for. Score WRONG if the answer is vague, missing key facts, or incorrect. "
    "Do not give credit for answers that are technically true but don't answer the question."
)
JUDGE_INSTRUCTION = os.environ.get("BENCH_JUDGE_PROMPT", _DEFAULT_JUDGE_PROMPT)


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


# ─── Embedding ───────────────────────────────────────────────


BENCH_EMBED_PROVIDER = os.environ.get("BENCH_EMBED_PROVIDER", os.environ.get("EMBED_PROVIDER", "gemini"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def get_embedding(text, prefix="search_document: "):
    if BENCH_EMBED_PROVIDER == "gemini" and GEMINI_API_KEY:
        import math

        if "query" in prefix.lower():
            task_type = "RETRIEVAL_QUERY"
        else:
            task_type = "RETRIEVAL_DOCUMENT"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
        )
        body = json.dumps(
            {
                "content": {"parts": [{"text": text[:8000]}]},
                "taskType": task_type,
                "outputDimensionality": EMBED_DIM,
            }
        ).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                values = data["embedding"]["values"]
                mag = math.sqrt(sum(v * v for v in values))
                if mag > 0 and EMBED_DIM != 3072:
                    values = [v / mag for v in values]
                return values
            except Exception:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise
    else:
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
    prompt = f"""You are answering questions about past conversations based on retrieved memory snippets.
Answer concisely in 1-3 sentences. Be direct and specific.

If NO relevant facts exist in the memories, say "I don't have enough information to answer this question."

Memories:
{context}

Question: {question}
Answer:"""

    for attempt in range(5):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(
                    {
                        "model": ANSWER_MODEL,
                        "max_tokens": 150,
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
    prompt = f"""You are evaluating an AI memory system's answer to a question about past conversations.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

{JUDGE_INSTRUCTION}

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


def commit_sessions(sessions, session_dates, collection):
    points = []
    committed = 0
    window_committed = 0

    all_turns = []
    for si, session in enumerate(sessions):
        date_str = session_dates[si] if si < len(session_dates) else f"Session {si + 1}"
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if not content:
                continue
            commit_text = f"[{date_str}] {role}: {content}"
            all_turns.append(commit_text)

    for text in all_turns:
        try:
            vec = get_embedding(text[:2000], prefix="search_document: ")
            point_id = int(hashlib.md5(text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": text[:4000],
                        "source": "longmemeval",
                        "source_weight": 1.0,
                        "date": datetime.now().isoformat(),
                        "importance": 70,
                        "retrieval_count": 0,
                        "chunk_type": "turn",
                    },
                }
            )
            committed += 1
            if len(points) >= 50:
                http_json(
                    f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30
                )
                points = []
        except Exception as e:
            if committed < 3:
                print(f"    Embed error: {e}")

    window_size, stride = 5, 2
    for i in range(0, max(len(all_turns) - window_size + 1, 1), stride):
        window = all_turns[i : i + window_size]
        window_text = "\n".join(window)
        try:
            vec = get_embedding(window_text[:2000], prefix="search_document: ")
            point_id = int(hashlib.md5(window_text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": window_text[:4000],
                        "source": "longmemeval",
                        "source_weight": 1.0,
                        "date": datetime.now().isoformat(),
                        "importance": 75,
                        "retrieval_count": 0,
                        "chunk_type": "window",
                    },
                }
            )
            window_committed += 1
            if len(points) >= 50:
                http_json(
                    f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30
                )
                points = []
        except Exception:
            pass

    if points:
        http_json(f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30)
    time.sleep(1)
    return committed, window_committed


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
    env["EMBED_PROVIDER"] = os.environ.get("EMBED_PROVIDER", "gemini")
    env["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    env["CONSTRAINTS_ENABLED"] = os.environ.get("CONSTRAINTS_ENABLED", "false")
    env["CONSTRAINTS_PROVIDER"] = os.environ.get("CONSTRAINTS_PROVIDER", "anthropic")
    env["RERANK_PROVIDER"] = os.environ.get("RERANK_PROVIDER", "cohere")
    env["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY", "")
    env["LLM_RERANKER"] = os.environ.get("LLM_RERANKER", "false")
    env["PYTHONPATH"] = str(REPO / "tools")

    server_log = RESULTS_DIR / "longmemeval-server.log"
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
        except Exception as e:
            if attempt < 3:
                time.sleep(2**attempt)
            else:
                print(f"    Search error: {e}")
                return []


def multi_query_search(question, port=BENCH_PORT):
    queries = [question]
    seen_texts = set()
    merged = []
    for q in queries:
        results = search_query(q, port=port, limit=SEARCH_LIMIT)
        for r in results:
            text_key = (r.get("text") or "").strip().lower()[:200]
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(r)
        time.sleep(0.1)
    merged.sort(key=lambda r: r.get("score", 0), reverse=True)
    return deduplicate_results(merged)


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
        qtype = info.get("question_type", "unknown")
        by_type.setdefault(qtype, []).append(score)
        all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0

    lines = [
        "# RASPUTIN Memory — LongMemEval Benchmark (ICLR 2025)",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Pipeline:** Multi-query search (top-{SEARCH_LIMIT}) → {ANSWER_MODEL} → {JUDGE_MODEL} judge",
        f"**Total questions:** {len(all_scores)}",
        f"\n## Overall Accuracy: {overall:.2f}%",
        "\n## Per-Category Breakdown",
    ]

    for qtype in sorted(by_type.keys()):
        scores = by_type[qtype]
        acc = sum(scores) / len(scores) * 100 if scores else 0
        lines.append(f"- **{qtype}**: {acc:.1f}% ({sum(1 for s in scores if s > 0)}/{len(scores)})")

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
    pending = [(i, q) for i, q in enumerate(questions) if q["question_id"] not in done_ids]

    print(f"LongMemEval: {len(questions)} questions, {len(done_ids)} already done, {len(pending)} pending\n")

    proc = None
    current_collection = None

    try:
        for idx, (qi, q) in enumerate(pending):
            qid = q["question_id"]
            qtype = q["question_type"]
            question = q["question"]
            answer = q["answer"]
            sessions = q["haystack_sessions"]
            session_dates = q.get("haystack_dates", [])
            session_ids = q.get("haystack_session_ids", [])

            collection = f"lme_{hashlib.md5('_'.join(sorted(session_ids)).encode()).hexdigest()[:12]}"

            print(f"[{idx + 1}/{len(pending)}] {qid} ({qtype})")

            if collection != current_collection:
                kill_server(proc)
                if current_collection:
                    delete_collection(current_collection)

                create_collection(collection)
                turns, windows = commit_sessions(sessions, session_dates, collection)
                print(f"  Committed {turns} turns + {windows} windows")

                proc = start_bench_server(collection, BENCH_PORT)
                print("  Server ready")
                current_collection = collection

            chunks = multi_query_search(question, port=BENCH_PORT)
            if not chunks:
                prediction = "I don't have enough information to answer this question."
            else:
                prediction = generate_opus_answer(question, chunks)

            score = judge_gpt4o_mini(question, prediction, answer)

            results[qid] = {
                "question_id": qid,
                "question_type": qtype,
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
    parser = argparse.ArgumentParser(description="RASPUTIN Memory — LongMemEval Benchmark")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared")

    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}")
        print(
            "Download: python3 -c \"from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('xiaowu0162/longmemeval-cleaned', 'longmemeval_oracle.json', repo_type='dataset'), 'benchmarks/longmemeval/longmemeval_oracle.json')\""
        )
        sys.exit(1)

    with open(DATASET_FILE) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} questions")
    types = Counter(d["question_type"] for d in dataset)
    for t, c in types.most_common():
        print(f"  {t}: {c}")
    print()

    run_benchmark(dataset, limit=args.limit)


if __name__ == "__main__":
    main()
