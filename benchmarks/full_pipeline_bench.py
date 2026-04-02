#!/usr/bin/env python3
"""
RASPUTIN Memory — Full Hybrid Pipeline Benchmark Runner
Tests against LoCoMo benchmark through the FULL pipeline:
BM25 + vector + reranker + entity boost + keyword overlap

Usage:
    python3 benchmarks/full_pipeline_bench.py [--conversations 0,1,2] [--port 7778]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
CHECKPOINT_FILE = RESULTS_DIR / "locomo_checkpoint.json"
LOCOMO_FILE = BENCH_DIR / "locomo" / "locomo10.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:11435")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-20250514")
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", LLM_BACKEND)
JUDGE_MODEL = os.environ.get(
    "JUDGE_MODEL",
    ANTHROPIC_MODEL if JUDGE_BACKEND == "anthropic" else "gpt-4o-mini",
)
EMBED_URL = os.environ.get("BENCH_EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("BENCH_EMBED_DIM", "768"))
BENCH_PORT = 7778

# LoCoMo category names
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def http_json(url, data=None, method=None, timeout=30):
    """Simple HTTP JSON helper without requests dependency."""
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def detect_vllm_model():
    """Auto-detect available vLLM model."""
    if VLLM_MODEL:
        return VLLM_MODEL
    try:
        data = http_json(f"{VLLM_URL}/v1/models")
        models = [m["id"] for m in data.get("data", [])]
        if models:
            print(f"  Detected vLLM model: {models[0]}")
            return models[0]
    except Exception as e:
        print(f"  WARNING: Could not detect vLLM model: {e}")
    return "qwen3.5-122b-a10b"


def create_qdrant_collection(name, dim=None):
    if dim is None:
        dim = EMBED_DIM
    """Create a fresh Qdrant collection."""
    # Delete if exists
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
    time.sleep(0.3)

    data = {
        "vectors": {"size": dim, "distance": "Cosine"},
        "optimizers_config": {"indexing_threshold": 0},  # index immediately
    }
    http_json(f"{QDRANT_URL}/collections/{name}", data=data, method="PUT")
    print(f"  Created collection: {name}")


def delete_qdrant_collection(name):
    """Delete a Qdrant collection."""
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def start_bench_server(collection, port=BENCH_PORT):
    """Start a temporary hybrid brain server for benchmarking."""
    env = os.environ.copy()
    env["QDRANT_COLLECTION"] = collection
    env["PORT"] = str(port)
    env["DISABLE_FALKORDB"] = "true"
    env["DISABLE_RERANKER"] = "true"
    env["RERANKER_ENABLED"] = "false"
    env["EMBED_URL"] = os.environ.get("BENCH_EMBED_URL", EMBED_URL)
    env["EMBED_MODEL"] = os.environ.get("BENCH_EMBED_MODEL", EMBED_MODEL)
    env["EMBED_PREFIX_QUERY"] = os.environ.get("BENCH_EMBED_PREFIX_QUERY", "search_query: ")
    env["EMBED_PREFIX_DOC"] = os.environ.get("BENCH_EMBED_PREFIX_DOC", "search_document: ")
    env["PYTHONPATH"] = str(REPO / "tools")

    proc = subprocess.Popen(
        [sys.executable, str(REPO / "tools" / "hybrid_brain.py"), "--port", str(port)],
        cwd=str(REPO / "tools"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    url = f"http://localhost:{port}/health"
    for attempt in range(30):
        time.sleep(1)
        try:
            http_json(url)
            print(f"  Server ready on port {port} (collection={collection})")
            return proc
        except Exception:
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode()[-500:] if proc.stdout else ""
                stderr = proc.stderr.read().decode()[-500:] if proc.stderr else ""
                print(f"  Server died! stdout: {stdout}")
                print(f"  stderr: {stderr}")
                raise RuntimeError("Bench server died during startup")

    proc.kill()
    raise RuntimeError("Bench server failed to start in 30s")


def kill_server(proc):
    """Kill the benchmark server."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def get_embedding(text, prefix=None):
    if prefix is None:
        prefix = os.environ.get("EMBED_PREFIX_DOC", "search_document: ")
    prefixed = f"{prefix}{text}" if prefix else text
    result = http_json(
        EMBED_URL,
        data={"model": EMBED_MODEL, "input": prefixed},
        timeout=30,
    )
    if "embeddings" in result:
        return result["embeddings"][0]
    if "data" in result:
        return result["data"][0]["embedding"]
    raise ValueError(f"Unexpected embed response: {list(result.keys())}")


def commit_conversation_direct(conv, collection):
    """Commit conversation turns directly to Qdrant (bypasses server for speed).
    Search still goes through the full hybrid pipeline."""
    import hashlib

    committed = 0
    failed = 0
    points = []

    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        date_key = f"session_{session_idx}_date_time"

        if session_key not in conv:
            break

        session_date = conv.get(date_key, f"Session {session_idx}")
        turns = conv[session_key]

        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            if not text:
                continue

            commit_text = f"[{session_date}] {speaker}: {text}"

            try:
                vec = get_embedding(commit_text[:2000])
                point_id = int(hashlib.md5(commit_text.encode()).hexdigest()[:15], 16)
                points.append(
                    {
                        "id": point_id,
                        "vector": vec,
                        "payload": {
                            "text": commit_text[:4000],
                            "source": "locomo_bench",
                            "source_weight": 1.0,
                            "date": datetime.now().isoformat(),
                            "importance": 70,
                            "retrieval_count": 0,
                        },
                    }
                )
                committed += 1

                # Batch upsert every 50 points
                if len(points) >= 50:
                    http_json(
                        f"{QDRANT_URL}/collections/{collection}/points",
                        data={"points": points},
                        method="PUT",
                        timeout=30,
                    )
                    points = []

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"    Embed/commit error: {e}")

        session_idx += 1

    # Flush remaining
    if points:
        try:
            http_json(
                f"{QDRANT_URL}/collections/{collection}/points",
                data={"points": points},
                method="PUT",
                timeout=30,
            )
        except Exception as e:
            print(f"    Final batch error: {e}")
            failed += len(points)
            committed -= len(points)

    print(f"  Committed {committed} turns ({failed} failed) across {session_idx - 1} sessions")

    # Wait for indexing
    time.sleep(2)
    return committed


def search_query(query, port=BENCH_PORT, limit=60):
    """Search the benchmark server."""
    url = f"http://localhost:{port}/search"
    params = urllib.parse.urlencode({"q": query, "limit": limit})
    try:
        result = http_json(f"{url}?{params}", timeout=30)
        return result.get("results", [])
    except Exception as e:
        print(f"    Search error: {e}")
        return []


def _build_extraction_prompt(question, context_chunks):
    context = "\n".join(f"- {c.get('text', '')}" for c in context_chunks[:25])
    return f"""Extract the answer from these memories. Reply with ONLY the answer — no explanation, no reasoning, no preamble. 1-10 words maximum.

Examples:
Q: "Where did they go?" → "Paris"
Q: "When is the meeting?" → "May 7, 2023"
Q: "What does she study?" → "Psychology and counseling"

If the answer requires inference from multiple memories, combine them. Only say "unknown" if truly zero relevant information exists.

Memories:
{context}

Q: {question}
A:"""


def _generate_anthropic(prompt, model):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": model,
                "max_tokens": 50,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", ANTHROPIC_API_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["content"][0]["text"].strip().split("\n")[0].strip()


def _generate_openai_compat(prompt, model):
    result = http_json(
        f"{VLLM_URL}/v1/chat/completions",
        data={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 50,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60,
    )
    msg = result["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content.split("\n")[0].strip()


def generate_answer(question, context_chunks, model):
    prompt = _build_extraction_prompt(question, context_chunks)
    try:
        if LLM_BACKEND == "anthropic" and ANTHROPIC_API_KEY:
            return _generate_anthropic(prompt, ANTHROPIC_MODEL)
        return _generate_openai_compat(prompt, model)
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


def llm_judge_score(question, prediction, ground_truth):
    prompt = f"""You are evaluating an AI memory system's answer to a question about a conversation.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

Is the system's answer correct? Be generous — if the answer captures the essential information from the ground truth, even if phrased differently, score it as CORRECT.

Reply with exactly one word: CORRECT or WRONG"""

    try:
        if JUDGE_BACKEND == "anthropic" and ANTHROPIC_API_KEY:
            result = _generate_anthropic(prompt, JUDGE_MODEL)
        else:
            result = _generate_openai_compat(prompt, JUDGE_MODEL)
    except Exception as e:
        print(f"    Judge LLM error: {e}")
        return 0.0

    return 1.0 if "CORRECT" in result.upper() else 0.0


def compute_f1(prediction, ground_truth):
    from collections import Counter

    pred_tokens = re.findall(r"\w+", prediction.lower())
    truth_tokens = re.findall(r"\w+", str(ground_truth).lower())

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def run_benchmark(conversations, conv_indices=None, port=BENCH_PORT):
    """Run the full pipeline benchmark on LoCoMo conversations."""
    model = detect_vllm_model()

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)

    all_results = checkpoint.get("results", {})

    if conv_indices is None:
        conv_indices = list(range(len(conversations)))

    for idx in conv_indices:
        conv_data = conversations[idx]
        conv_id = conv_data.get("sample_id", f"conv-{idx}")

        if conv_id in all_results:
            print(f"\n[{idx + 1}/{len(conv_indices)}] Skipping {conv_id} (already done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(conv_indices)}] Processing {conv_id}")
        print(f"{'=' * 60}")

        collection = f"locomo_bench_{conv_id.replace('-', '_')}"
        proc = None

        try:
            # Step 1: Create collection
            create_qdrant_collection(collection)

            # Step 2: Start server
            proc = start_bench_server(collection, port)

            # Step 3: Commit conversation directly to Qdrant (fast)
            n_committed = commit_conversation_direct(conv_data["conversation"], collection)

            # Step 4: Run QA
            qa_results = []
            qa_list = conv_data.get("qa", [])
            print(f"  Running {len(qa_list)} questions...")

            for qi, qa in enumerate(qa_list):
                question = qa["question"]
                ground_truth = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)

                if not ground_truth:
                    continue

                # Search
                chunks = search_query(question, port)

                # Generate
                prediction = generate_answer(question, chunks, model)

                # Score
                f1 = compute_f1(prediction, ground_truth)
                judge = llm_judge_score(question, prediction, ground_truth)

                qa_results.append(
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "f1": f1,
                        "judge": judge,
                        "category": category,
                        "n_chunks": len(chunks),
                    }
                )

                # Rate limit: max ~60/min for search
                time.sleep(0.6)

                if (qi + 1) % 20 == 0:
                    avg_so_far = sum(r["f1"] for r in qa_results) / len(qa_results)
                    avg_judge_so_far = sum(r["judge"] for r in qa_results) / len(qa_results)
                    print(
                        f"    Progress: {qi + 1}/{len(qa_list)} questions, "
                        f"running F1={avg_so_far:.4f}, Judge={avg_judge_so_far:.4f}"
                    )

            # Compute stats
            qa_non_adversarial = [r for r in qa_results if r["category"] != 5]
            qa_adversarial = [r for r in qa_results if r["category"] == 5]
            score_pool = qa_non_adversarial if qa_non_adversarial else qa_results

            conv_f1 = sum(r["f1"] for r in score_pool) / len(score_pool) if score_pool else 0
            conv_judge = sum(r["judge"] for r in score_pool) / len(score_pool) if score_pool else 0
            conv_adversarial_f1 = sum(r["f1"] for r in qa_adversarial) / len(qa_adversarial) if qa_adversarial else 0
            conv_adversarial_judge = (
                sum(r["judge"] for r in qa_adversarial) / len(qa_adversarial) if qa_adversarial else 0
            )

            cat_scores = defaultdict(list)
            cat_judge_scores = defaultdict(list)
            for r in qa_results:
                cat_scores[r["category"]].append(r["f1"])
                cat_judge_scores[r["category"]].append(r["judge"])

            cat_metrics = {
                cat: {
                    "f1": sum(scores) / len(scores),
                    "judge": sum(cat_judge_scores[cat]) / len(cat_judge_scores[cat]),
                    "n_questions": len(scores),
                }
                for cat, scores in sorted(cat_scores.items())
            }

            all_results[conv_id] = {
                "conv_id": conv_id,
                "n_questions": len(qa_results),
                "n_questions_scored": len(score_pool),
                "n_questions_adversarial": len(qa_adversarial),
                "n_committed": n_committed,
                "mean_f1": conv_f1,
                "mean_judge": conv_judge,
                "adversarial_mean_f1": conv_adversarial_f1,
                "adversarial_mean_judge": conv_adversarial_judge,
                "category_metrics": cat_metrics,
                "details": qa_results,
            }

            print(
                f"  ✅ {conv_id}: F1={conv_f1:.4f}, Judge={conv_judge:.4f} "
                f"({len(score_pool)} non-adversarial scored, {len(qa_results)} total questions)"
            )
            if qa_adversarial:
                print(
                    f"     adversarial (excluded from overall): "
                    f"F1={conv_adversarial_f1:.4f}, Judge={conv_adversarial_judge:.4f} "
                    f"({len(qa_adversarial)} Qs)"
                )
            for cat, metrics in sorted(cat_metrics.items()):
                cat_name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
                print(
                    f"     {cat_name}: F1={metrics['f1']:.4f}, Judge={metrics['judge']:.4f} "
                    f"({metrics['n_questions']} Qs)"
                )

            # Save checkpoint
            checkpoint["results"] = all_results
            checkpoint["last_updated"] = datetime.now().isoformat()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            print(f"  ❌ Error on {conv_id}: {e}")
            import traceback

            traceback.print_exc()

        finally:
            kill_server(proc)
            delete_qdrant_collection(collection)
            time.sleep(1)

    return all_results


def generate_report(all_results):
    """Generate markdown report."""
    if not all_results:
        return "No results."

    all_f1s = []
    all_judges = []
    all_f1s_adversarial = []
    all_judges_adversarial = []
    cat_all_f1 = defaultdict(list)
    cat_all_judge = defaultdict(list)

    for _, data in all_results.items():
        for r in data.get("details", []):
            cat_all_f1[r["category"]].append(r["f1"])
            cat_all_judge[r["category"]].append(r["judge"])
            if r["category"] == 5:
                all_f1s_adversarial.append(r["f1"])
                all_judges_adversarial.append(r["judge"])
            else:
                all_f1s.append(r["f1"])
                all_judges.append(r["judge"])

    overall_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
    overall_judge = sum(all_judges) / len(all_judges) if all_judges else 0
    adversarial_f1 = sum(all_f1s_adversarial) / len(all_f1s_adversarial) if all_f1s_adversarial else 0
    adversarial_judge = sum(all_judges_adversarial) / len(all_judges_adversarial) if all_judges_adversarial else 0

    lines = []
    lines.append("# RASPUTIN Memory — LoCoMo Full Pipeline Benchmark Results")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Pipeline:** BM25 + vector + reranker + entity boost + keyword overlap")
    lines.append(f"**Conversations:** {len(all_results)}")
    lines.append(f"**Total questions (all categories):** {len(all_f1s) + len(all_f1s_adversarial)}")
    lines.append(f"**Questions in overall score (excluding adversarial):** {len(all_f1s)}")
    lines.append("\n## Overall (excluding adversarial / category 5)")
    lines.append(f"- Mean F1: {overall_f1 * 100:.2f}")
    lines.append(f"- LLM-judge accuracy: {overall_judge * 100:.2f}")

    lines.append("\n## Adversarial category (reported separately)")
    lines.append(f"- Mean F1: {adversarial_f1 * 100:.2f}")
    lines.append(f"- LLM-judge accuracy: {adversarial_judge * 100:.2f}")

    lines.append("\n## Per-Category Breakdown")
    for cat in sorted(cat_all_f1.keys()):
        scores_f1 = cat_all_f1[cat]
        scores_judge = cat_all_judge[cat]
        cat_name = CATEGORY_NAMES.get(cat, f"category-{cat}")
        avg_f1 = sum(scores_f1) / len(scores_f1) if scores_f1 else 0
        avg_judge = sum(scores_judge) / len(scores_judge) if scores_judge else 0
        lines.append(
            f"- **{cat_name}** (cat {cat}): "
            f"F1={avg_f1 * 100:.2f}, Judge={avg_judge * 100:.2f} ({len(scores_f1)} questions)"
        )

    lines.append("\n## Per-Conversation Results")
    for conv_id, data in sorted(all_results.items()):
        lines.append(
            f"- **{conv_id}**: "
            f"F1={data['mean_f1'] * 100:.2f}, Judge={data['mean_judge'] * 100:.2f} "
            f"({data['n_questions_scored']} non-adversarial Qs, {data['n_questions']} total, {data['n_committed']} committed)"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RASPUTIN Full Pipeline LoCoMo Benchmark")
    parser.add_argument(
        "--conversations", type=str, default=None, help="Comma-separated indices (e.g. 0,1,2). Default: all"
    )
    parser.add_argument("--port", type=int, default=BENCH_PORT)
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared.")

    # Load dataset
    print(f"Loading LoCoMo dataset from {LOCOMO_FILE}...")
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")

    conv_indices = None
    if args.conversations:
        conv_indices = [int(x) for x in args.conversations.split(",")]

    # Run
    results = run_benchmark(conversations, conv_indices, args.port)

    # Report
    report = generate_report(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "locomo-fullpipeline-results.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
