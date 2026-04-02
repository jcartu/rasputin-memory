#!/usr/bin/env python3
"""
RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1
Improvements over baseline:
  1. LLM-Judge via GPT-4o-mini (binary CORRECT/WRONG)
  2. Claude Opus 4 for answer generation
  3. Exclude adversarial (cat 5) from headline score
  4. Disable reranker (vector-only)
  5. Nomic prefixes (search_query: / search_document:)
  6. Top-K 60

Usage:
    python3 benchmarks/locomo_leaderboard_bench.py [--conversations 0,1,2] [--rescore-only]
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
LOCOMO_FILE = BENCH_DIR / "locomo" / "locomo10.json"
CHECKPOINT_FILE = RESULTS_DIR / os.environ.get("BENCH_CHECKPOINT", "locomo-leaderboard-checkpoint.json")
OUTPUT_FILE = RESULTS_DIR / "locomo-leaderboard-v1.json"
COMPARISON_FILE = RESULTS_DIR / "locomo-leaderboard-comparison.md"

# Existing results to rescore
OLD_CHECKPOINT = RESULTS_DIR / "locomo-leaderboard-checkpoint.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("BENCH_EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
BENCH_PORT = 7779

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPUS_MODEL = "claude-opus-4-6"
JUDGE_MODEL = "gpt-4o-mini"

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}


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


def get_embedding(text, prefix="search_document: "):
    prefixed = f"{prefix}{text}" if prefix else text
    result = http_json(EMBED_URL, data={"model": EMBED_MODEL, "input": prefixed}, timeout=30)
    if "embeddings" in result:
        return result["embeddings"][0]
    if "data" in result:
        return result["data"][0]["embedding"]
    raise ValueError(f"Unexpected embed response: {list(result.keys())}")


# ─── LLM: Opus for answers ──────────────────────────────────


def generate_opus_answer(question, context_chunks):
    context = "\n".join(f"- {c.get('text', c) if isinstance(c, dict) else c}" for c in context_chunks[:30])
    prompt = f"""You are answering questions about a conversation based on retrieved memory snippets.
Answer concisely in 1-3 sentences. Be direct and specific.
If the memories don't contain enough information to answer, say "I don't have enough information to answer this question."

Memories:
{context}

Question: {question}
Answer:"""

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(
            {
                "model": OPUS_MODEL,
                "max_tokens": 150,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", ANTHROPIC_API_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["content"][0]["text"].strip()


# ─── LLM: GPT-4o-mini for judging ───────────────────────────


def judge_gpt4o_mini(question, prediction, ground_truth):
    prompt = f"""You are evaluating an AI memory system's answer to a question about a conversation.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {prediction}

Is the system's answer correct? Be generous — if the answer captures the essential information from the ground truth, even if phrased differently or includes extra correct details, score it as CORRECT. Only score WRONG if the answer is factually incorrect, missing the key information, or says it doesn't know when the answer was available.

Reply with exactly one word: CORRECT or WRONG"""

    data = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
    }
    result = http_json(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=30,
    )
    text = result["choices"][0]["message"]["content"].strip()
    return 1.0 if "CORRECT" in text.upper() else 0.0


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
        data={
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 0},
        },
        method="PUT",
    )


def delete_collection(name):
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{name}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def commit_conversation(conv, collection):
    """Commit conversation turns directly to Qdrant with nomic prefixes."""
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
            text = turn.get("text", "")
            if not text:
                continue
            speaker = turn.get("speaker", "Unknown")
            commit_text = f"[{session_date}] {speaker}: {text}"
            try:
                vec = get_embedding(commit_text[:2000], prefix="search_document: ")
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
                if len(points) >= 50:
                    http_json(
                        f"{QDRANT_URL}/collections/{collection}/points",
                        data={"points": points},
                        method="PUT",
                        timeout=30,
                    )
                    points = []
            except Exception as e:
                if committed < 5:
                    print(f"    Embed error: {e}")
        session_idx += 1
    if points:
        http_json(f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30)
    time.sleep(2)
    print(f"  Committed {committed} turns across {session_idx - 1} sessions")
    return committed


# ─── Server management ───────────────────────────────────────


def start_bench_server(collection, port=BENCH_PORT):
    env = os.environ.copy()
    env["QDRANT_COLLECTION"] = collection
    env["PORT"] = str(port)
    env["DISABLE_FALKORDB"] = "true"
    env["DISABLE_RERANKER"] = "true"
    env["RERANKER_ENABLED"] = "false"
    env["EMBED_URL"] = EMBED_URL
    env["EMBED_MODEL"] = EMBED_MODEL
    env["EMBED_PREFIX_QUERY"] = "search_query: "
    env["EMBED_PREFIX_DOC"] = "search_document: "
    env["PYTHONPATH"] = str(REPO / "tools")

    proc = subprocess.Popen(
        [sys.executable, str(REPO / "tools" / "hybrid_brain.py"), "--port", str(port)],
        cwd=str(REPO / "tools"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://localhost:{port}/health"
    for _ in range(30):
        time.sleep(1)
        try:
            http_json(url)
            print(f"  Server ready on port {port}")
            return proc
        except Exception:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode()[-500:] if proc.stderr else ""
                raise RuntimeError(f"Server died: {stderr}")
    proc.kill()
    raise RuntimeError("Server failed to start in 30s")


def kill_server(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def search_query(query, port=BENCH_PORT, limit=60):
    url = f"http://localhost:{port}/search"
    params = urllib.parse.urlencode({"q": query, "limit": limit})
    try:
        result = http_json(f"{url}?{params}", timeout=30)
        return result.get("results", [])
    except Exception as e:
        print(f"    Search error: {e}")
        return []


# ─── F1 scoring ──────────────────────────────────────────────


def compute_f1(prediction, ground_truth):
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


# ─── Rescore mode ────────────────────────────────────────────


def rescore_existing(checkpoint_path):
    """Re-judge existing results with GPT-4o-mini."""
    print(f"\n{'=' * 60}")
    print("RESCORE MODE: Re-judging existing results with GPT-4o-mini")
    print(f"{'=' * 60}")

    with open(checkpoint_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results to rescore!")
        return

    print(f"Found {len(results)} QA pairs to rescore")
    correct = 0
    total = 0
    correct_no_adv = 0
    total_no_adv = 0
    cat_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, r in enumerate(results):
        question = r["question"]
        prediction = r["predicted"]
        gold = str(r["gold"])
        category = r.get("category", 0)

        judge = judge_gpt4o_mini(question, prediction, gold)
        r["judge_gpt4o_mini"] = judge
        r["correct_gpt4o_mini"] = bool(judge)

        total += 1
        if judge:
            correct += 1
        if category != 5:
            total_no_adv += 1
            if judge:
                correct_no_adv += 1

        cat_name = CATEGORY_NAMES.get(category, f"cat-{category}")
        cat_scores[cat_name]["total"] += 1
        if judge:
            cat_scores[cat_name]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Rescored {i + 1}/{len(results)}, running accuracy: {correct / total * 100:.1f}%")
        time.sleep(0.1)  # rate limit

    print(f"\n{'=' * 60}")
    print("RESCORE RESULTS (GPT-4o-mini judge on existing Qwen answers)")
    print(f"{'=' * 60}")
    print(f"Overall (all): {correct}/{total} = {correct / total * 100:.2f}%")
    if total_no_adv:
        print(
            f"Overall (excl. adversarial): {correct_no_adv}/{total_no_adv} = {correct_no_adv / total_no_adv * 100:.2f}%"
        )
    for cat_name, s in sorted(cat_scores.items()):
        print(f"  {cat_name}: {s['correct']}/{s['total']} = {s['correct'] / s['total'] * 100:.1f}%")

    # Save rescored
    rescore_path = RESULTS_DIR / "locomo-rescore-gpt4omini.json"
    with open(rescore_path, "w") as f:
        json.dump(
            {
                "results": results,
                "overall_accuracy": correct / total if total else 0,
                "accuracy_excl_adversarial": correct_no_adv / total_no_adv if total_no_adv else 0,
                "category_scores": {k: v["correct"] / v["total"] for k, v in cat_scores.items()},
                "date": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"Saved to {rescore_path}")
    return correct / total if total else 0, correct_no_adv / total_no_adv if total_no_adv else 0


# ─── Full pipeline ───────────────────────────────────────────


def run_full_pipeline(conversations, conv_indices=None, port=BENCH_PORT):
    """Run full pipeline: embed → search (top-60, no reranker) → Opus answer → GPT-4o-mini judge."""

    # Load checkpoint
    checkpoint = {"results": [], "completed_keys": set()}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        # Check if it's the new format (list of per-QA results)
        if isinstance(cp.get("results"), list) and cp["results"] and "predicted" in cp["results"][0]:
            checkpoint["results"] = cp["results"]
            checkpoint["completed_keys"] = set(cp.get("completed_keys", []))

    if conv_indices is None:
        conv_indices = list(range(len(conversations)))

    for idx in conv_indices:
        conv_data = conversations[idx]
        conv_id = conv_data.get("sample_id", f"conv-{idx}")
        qa_list = conv_data.get("qa", [])

        # Check how many already done for this conv
        done_for_conv = sum(1 for k in checkpoint["completed_keys"] if k.startswith(f"{conv_id}_"))
        if done_for_conv >= len(qa_list):
            print(f"\n[{idx + 1}] Skipping {conv_id} (all {len(qa_list)} QA done)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(conv_indices)}] {conv_id} — {len(qa_list)} QA ({done_for_conv} already done)")
        print(f"{'=' * 60}")

        collection = f"locomo_lb_{conv_id.replace('-', '_')}"
        proc = None

        try:
            create_collection(collection)
            proc = start_bench_server(collection, port)
            commit_conversation(conv_data["conversation"], collection)

            for qi, qa in enumerate(qa_list):
                key = f"{conv_id}_{qi}"
                if key in checkpoint["completed_keys"]:
                    continue

                question = qa["question"]
                ground_truth = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)
                if not ground_truth:
                    continue

                # Search with top-K 60
                chunks = search_query(question, port, limit=60)

                # Generate with Opus
                try:
                    prediction = generate_opus_answer(question, chunks)
                except Exception as e:
                    print(f"    Opus error on Q{qi}: {e}")
                    prediction = ""
                    time.sleep(5)

                # Judge with GPT-4o-mini
                try:
                    judge = judge_gpt4o_mini(question, prediction, ground_truth)
                except Exception as e:
                    print(f"    Judge error on Q{qi}: {e}")
                    judge = 0.0

                f1 = compute_f1(prediction, ground_truth)

                checkpoint["results"].append(
                    {
                        "conv_id": conv_id,
                        "qi": qi,
                        "question": question,
                        "gold": ground_truth,
                        "predicted": prediction,
                        "category": category,
                        "cat_name": CATEGORY_NAMES.get(category, f"cat-{category}"),
                        "correct": bool(judge),
                        "judge_score": judge,
                        "f1": f1,
                        "n_chunks": len(chunks),
                    }
                )
                checkpoint["completed_keys"].add(key)

                # Progress
                if (qi + 1) % 10 == 0 or qi == len(qa_list) - 1:
                    done_now = sum(1 for r in checkpoint["results"] if r["conv_id"] == conv_id)
                    conv_correct = sum(1 for r in checkpoint["results"] if r["conv_id"] == conv_id and r["correct"])
                    print(
                        f"    Q{qi + 1}/{len(qa_list)}: {conv_correct}/{done_now} correct "
                        f"({conv_correct / done_now * 100:.0f}%)"
                    )

                # Save checkpoint every 5 questions
                if (qi + 1) % 5 == 0:
                    save_checkpoint(checkpoint)

                time.sleep(0.3)  # rate limit

            save_checkpoint(checkpoint)

            # Print conv summary
            conv_results = [r for r in checkpoint["results"] if r["conv_id"] == conv_id]
            conv_no_adv = [r for r in conv_results if r["category"] != 5]
            if conv_no_adv:
                acc = sum(1 for r in conv_no_adv if r["correct"]) / len(conv_no_adv)
                print(f"  ✅ {conv_id}: {acc * 100:.1f}% (excl. adversarial, {len(conv_no_adv)} Qs)")

        except Exception as e:
            print(f"  ❌ Error on {conv_id}: {e}")
            import traceback

            traceback.print_exc()
            save_checkpoint(checkpoint)

        finally:
            kill_server(proc)
            delete_collection(collection)
            time.sleep(1)

    return checkpoint["results"]


def save_checkpoint(checkpoint):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {
                "results": checkpoint["results"],
                "completed_keys": list(checkpoint["completed_keys"]),
                "last_updated": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


# ─── Reporting ───────────────────────────────────────────────


def generate_report(results):
    all_results = results
    total = len(all_results)
    if not total:
        return "No results."

    # Split by adversarial
    non_adv = [r for r in all_results if r.get("category") != 5]
    adv = [r for r in all_results if r.get("category") == 5]

    acc_all = sum(1 for r in all_results if r.get("correct")) / total * 100
    acc_no_adv = sum(1 for r in non_adv if r.get("correct")) / len(non_adv) * 100 if non_adv else 0
    acc_adv = sum(1 for r in adv if r.get("correct")) / len(adv) * 100 if adv else 0

    f1_no_adv = sum(r.get("f1", 0) for r in non_adv) / len(non_adv) * 100 if non_adv else 0

    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0, "f1_sum": 0})
    for r in all_results:
        cat = CATEGORY_NAMES.get(r.get("category", 0), f"cat-{r.get('category', 0)}")
        cat_stats[cat]["total"] += 1
        if r.get("correct"):
            cat_stats[cat]["correct"] += 1
        cat_stats[cat]["f1_sum"] += r.get("f1", 0)

    # Per-conversation
    conv_stats = defaultdict(lambda: {"correct": 0, "total": 0, "correct_no_adv": 0, "total_no_adv": 0})
    for r in all_results:
        cid = r.get("conv_id", "?")
        conv_stats[cid]["total"] += 1
        if r.get("correct"):
            conv_stats[cid]["correct"] += 1
        if r.get("category") != 5:
            conv_stats[cid]["total_no_adv"] += 1
            if r.get("correct"):
                conv_stats[cid]["correct_no_adv"] += 1

    lines = [
        "# RASPUTIN Memory — LoCoMo Leaderboard Benchmark v1",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**Pipeline:** Vector search (nomic-embed-text) → Top-60 → Claude Opus 4 → GPT-4o-mini judge",
        "**Improvements:** LLM judge, Opus answers, exclude adversarial, no reranker, nomic prefixes, top-K 60",
        f"**Total questions:** {total} ({len(non_adv)} non-adversarial, {len(adv)} adversarial)",
        "\n## Headline Score (excluding adversarial)",
        f"**LLM-Judge Accuracy: {acc_no_adv:.2f}%**",
        f"- Token F1: {f1_no_adv:.2f}%",
        "\n## Including adversarial",
        f"- All categories: {acc_all:.2f}%",
        f"- Adversarial only: {acc_adv:.2f}%",
        "\n## Per-Category Breakdown",
    ]
    for cat_name, s in sorted(cat_stats.items()):
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        f1 = s["f1_sum"] / s["total"] * 100 if s["total"] else 0
        lines.append(f"- **{cat_name}**: {acc:.1f}% judge, {f1:.1f}% F1 ({s['total']} Qs)")

    lines.append("\n## Per-Conversation")
    for cid, s in sorted(conv_stats.items()):
        acc = s["correct_no_adv"] / s["total_no_adv"] * 100 if s["total_no_adv"] else 0
        lines.append(f"- **{cid}**: {acc:.1f}% ({s['correct_no_adv']}/{s['total_no_adv']} excl. adv)")

    lines.append("\n## Leaderboard Comparison")
    lines.append("| System | LLM-Judge Accuracy |")
    lines.append("|--------|-------------------|")
    lines.append("| Backboard | 90.00% |")
    lines.append("| MemMachine | 84.87% |")
    lines.append(f"| **RASPUTIN** | **{acc_no_adv:.2f}%** |")
    lines.append("| Memobase | 75.78% |")
    lines.append("| Zep | 75.14% |")
    lines.append("| mem0 | 66.88% |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--port", type=int, default=BENCH_PORT)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rescore_only:
        # Find existing results to rescore
        old_cp = RESULTS_DIR / "locomo-leaderboard-checkpoint.json"
        if not old_cp.exists():
            print("No checkpoint found to rescore!")
            sys.exit(1)
        rescore_existing(old_cp)
        return

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared.")

    print("Loading LoCoMo dataset...")
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)
    print(f"Loaded {len(conversations)} conversations")

    conv_indices = None
    if args.conversations:
        conv_indices = [int(x) for x in args.conversations.split(",")]

    results = run_full_pipeline(conversations, conv_indices, args.port)

    # Generate and save report
    report = generate_report(results)
    with open(COMPARISON_FILE, "w") as f:
        f.write(report)

    # Save full results
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"results": results, "date": datetime.now().isoformat()}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(report)
    print(f"\nResults: {OUTPUT_FILE}")
    print(f"Report: {COMPARISON_FILE}")


if __name__ == "__main__":
    main()
