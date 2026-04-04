#!/usr/bin/env python3
"""
RASPUTIN Memory — LoCoMo Leaderboard Benchmark v2
v2 improvements (v0.7.0):
  1. Adversarial-resistant answer prompt (entity-swap tolerant)
  2. Conversation-window chunking (5-turn, stride 2) for cross-turn recall
  3. Multi-query retrieval (name + topic decomposition)
  4. Top-K 60 per sub-query with token-overlap deduplication
  5. Context window 30→50 chunks for answer generation
Inherited from v1:
  6. LLM-Judge via GPT-4o-mini (binary CORRECT/WRONG)
  7. Claude Opus 4 for answer generation
  8. Nomic prefixes (search_query: / search_document:)

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
import urllib.error
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
ANSWER_MODEL = os.environ.get("BENCH_ANSWER_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gpt-4o-mini-2024-07-18")
BENCH_MODE = os.environ.get("BENCH_MODE", "production")

CATEGORY_NAMES = {1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}

SEARCH_LIMIT = int(os.environ.get("BENCH_SEARCH_LIMIT", "60"))
CONTEXT_CHUNKS = int(os.environ.get("BENCH_CONTEXT_CHUNKS", "60"))

_DEFAULT_JUDGE_PROMPT = (
    "Is the system's answer correct? Score CORRECT only if the answer contains the specific "
    "information asked for. Score WRONG if the answer is vague, missing key facts, or incorrect. "
    "Do not give credit for answers that are technically true but don't answer the question."
)
JUDGE_INSTRUCTION = os.environ.get("BENCH_JUDGE_PROMPT", _DEFAULT_JUDGE_PROMPT)


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


# ─── Multi-query expansion ───────────────────────────────────


_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def expand_search_queries(question, speakers=None):
    """Generate query variants for better recall on entity and temporal questions."""
    queries = [question]
    speakers = speakers or []
    speakers_lower = {s.lower() for s in speakers}

    names_in_q = [m.group(1) for m in _NAME_RE.finditer(question) if m.group(1).lower() in speakers_lower]

    for name in names_in_q[:2]:
        queries.append(name)
        topic = question.replace(name, "").strip(" ?.,")
        if len(topic) > 10:
            queries.append(topic)

    return queries[:5]


def extract_speakers(conv):
    speakers = set()
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        if session_key not in conv:
            break
        for turn in conv[session_key]:
            speaker = turn.get("speaker", "")
            if speaker:
                speakers.add(speaker)
        session_idx += 1
    return sorted(speakers)


def deduplicate_results(results, overlap_threshold=0.75):
    """Remove near-duplicate passages using token overlap."""
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


def generate_opus_answer(question, context_chunks, max_chunks=50):
    context = "\n".join(f"- {c.get('text', c) if isinstance(c, dict) else c}" for c in context_chunks[:max_chunks])
    prompt = f"""You are answering questions about a conversation based on retrieved memory snippets.
Answer concisely in 1-3 sentences. Be direct and specific.

IMPORTANT: If the question attributes an action or fact to Person A, but the memories show
it was actually Person B who did it, STILL PROVIDE THE FACTUAL ANSWER. For example, if
asked "What did Alice cook?" but the memories show Bob cooked pasta, answer "pasta."
The question is asking about the ACTION/FACT, not verifying who did it.

If the question names a person or entity that doesn't appear in the memories, say so
rather than substituting a similar entity. For example, if asked about "David" but
only "Daniel" appears in the memories, say you don't have information about David.

If NO relevant facts exist in the memories for ANY person, say "I don't have enough
information to answer this question." Only refuse when the memories genuinely contain
nothing relevant — not when the person attribution is wrong.

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
        except Exception as e:
            if attempt < 4:
                wait = 2**attempt
                print(f"    Opus retry {attempt + 1}/5 ({e}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


# ─── LLM: GPT-4o-mini for judging ───────────────────────────


def judge_gpt4o_mini(question, prediction, ground_truth):
    prompt = f"""You are evaluating an AI memory system's answer to a question about a conversation.

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
        except Exception as e:
            if attempt < 4:
                wait = 2**attempt
                print(f"    Judge retry {attempt + 1}/5 ({e}), waiting {wait}s...")
                time.sleep(wait)
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


def _build_conversation_windows(all_turns, window_size=5, stride=2):
    """Create overlapping multi-turn windows so vector search captures cross-turn context."""
    windows = []
    for i in range(0, max(len(all_turns) - window_size + 1, 1), stride):
        window = all_turns[i : i + window_size]
        combined = "\n".join(t["commit_text"] for t in window)
        windows.append(
            {
                "text": combined,
                "session_date": window[0]["session_date"],
                "speakers": list({t["speaker"] for t in window}),
            }
        )
    return windows


def commit_conversation(conv, collection):
    """Commit individual turns + overlapping conversation windows to Qdrant."""
    points = []
    committed = 0
    session_idx = 1

    # Phase 1: collect all turns with metadata
    all_turns = []
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
            all_turns.append({"commit_text": commit_text, "speaker": speaker, "session_date": session_date})
        session_idx += 1

    num_sessions = session_idx - 1

    # Phase 2: embed and commit individual turns
    for turn_info in all_turns:
        commit_text = turn_info["commit_text"]
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
                        "chunk_type": "turn",
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

    # Phase 3: build and commit overlapping conversation windows
    windows = _build_conversation_windows(all_turns, window_size=5, stride=2)
    window_committed = 0
    for window in windows:
        window_text = window["text"]
        try:
            vec = get_embedding(window_text[:2000], prefix="search_document: ")
            point_id = int(hashlib.md5(window_text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": window_text[:4000],
                        "source": "locomo_bench",
                        "source_weight": 1.0,
                        "date": datetime.now().isoformat(),
                        "importance": 75,
                        "retrieval_count": 0,
                        "chunk_type": "window",
                        "speakers": window["speakers"],
                    },
                }
            )
            window_committed += 1
            if len(points) >= 50:
                http_json(
                    f"{QDRANT_URL}/collections/{collection}/points",
                    data={"points": points},
                    method="PUT",
                    timeout=30,
                )
                points = []
        except Exception as e:
            if window_committed < 3:
                print(f"    Window embed error: {e}")

    if points:
        http_json(f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30)
    time.sleep(2)
    print(f"  Committed {committed} turns + {window_committed} windows across {num_sessions} sessions")
    return committed + window_committed


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

    server_log = RESULTS_DIR / "bench-server.log"
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


def search_query(query, port=BENCH_PORT, limit=SEARCH_LIMIT):
    url = f"http://localhost:{port}/search"
    params = urllib.parse.urlencode({"q": query, "limit": limit, "expand": "false"})
    for attempt in range(4):
        try:
            result = http_json(f"{url}?{params}", timeout=60)
            return result.get("results", [])
        except urllib.error.HTTPError as e:
            if attempt < 3:
                time.sleep(2**attempt)
            else:
                body = ""
                try:
                    body = e.read().decode()[:300]
                except Exception:
                    pass
                print(f"    Search error after 4 tries: {e} | {body}")
                return []
        except Exception as e:
            if attempt < 3:
                time.sleep(2**attempt)
            else:
                print(f"    Search error after 4 tries: {e}")
                return []


def multi_query_search(question, speakers=None, port=BENCH_PORT):
    """Search with multiple query variants and merge deduplicated results."""
    queries = expand_search_queries(question, speakers=speakers)
    seen_texts = set()
    merged = []
    for q in queries:
        results = search_query(q, port=port, limit=SEARCH_LIMIT)
        for r in results:
            text_key = (r.get("text") or "").strip().lower()[:200]
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                merged.append(r)
        time.sleep(0.3)
    merged.sort(key=lambda r: r.get("score", 0), reverse=True)
    return deduplicate_results(merged)


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
    """Run full pipeline: embed + windows → multi-query search (top-60) → dedup → Opus answer → judge."""

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
            speakers = extract_speakers(conv_data["conversation"])
            create_collection(collection)
            proc = start_bench_server(collection, port)
            commit_conversation(conv_data["conversation"], collection)

            # Process QA pairs concurrently (5 at a time)
            from concurrent.futures import ThreadPoolExecutor, as_completed

            CONCURRENCY = 2

            # Step 1: Serial search (embedding server can't handle concurrent calls)
            pending_qas = [
                (qi, qa) for qi, qa in enumerate(qa_list) if f"{conv_id}_{qi}" not in checkpoint["completed_keys"]
            ]

            print(f"  Searching {len(pending_qas)} questions (multi-query, top-{SEARCH_LIMIT})...")
            searched = []
            for qi, qa in pending_qas:
                question = qa["question"]
                ground_truth = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)
                if not ground_truth:
                    continue
                chunks = multi_query_search(question, speakers=speakers, port=port)
                searched.append((qi, question, ground_truth, category, chunks))
                time.sleep(0.1)

            print(f"  Generating answers for {len(searched)} questions (concurrent)...")

            def process_single_qa(item):
                qi, question, ground_truth, category, chunks = item
                try:
                    prediction = generate_opus_answer(question, chunks)
                except Exception as e:
                    print(f"    Opus error on Q{qi}: {e}")
                    prediction = ""
                try:
                    judge = judge_gpt4o_mini(question, prediction, ground_truth)
                except Exception as e:
                    print(f"    Judge error on Q{qi}: {e}")
                    judge = 0.0
                f1 = compute_f1(prediction, ground_truth)
                return (qi, question, ground_truth, category, prediction, judge, f1)

            with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                futures = {executor.submit(process_single_qa, item): item[0] for item in searched}
                done_count = done_for_conv
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        continue
                    qi, question, ground_truth, category, prediction, judge, f1 = result

                    key = f"{conv_id}_{qi}"
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
                        }
                    )
                    checkpoint["completed_keys"].add(key)
                    done_count += 1

                    # Progress every 10
                    if done_count % 10 == 0 or done_count == len(pending_qas):
                        conv_correct = sum(1 for r in checkpoint["results"] if r["conv_id"] == conv_id and r["correct"])
                        conv_done = sum(1 for r in checkpoint["results"] if r["conv_id"] == conv_id)
                        print(
                            f"    Q{conv_done}/{len(qa_list)}: {conv_correct}/{conv_done} correct "
                            f"({conv_correct / conv_done * 100:.0f}%)"
                        )

                    # Checkpoint every 10
                    if done_count % 10 == 0:
                        save_checkpoint(checkpoint)

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
        f"**Pipeline:** Window chunking → Multi-query search (top-{SEARCH_LIMIT}) → Dedup → {ANSWER_MODEL} → {JUDGE_MODEL} judge",
        f"**Mode:** {BENCH_MODE} | top-K {SEARCH_LIMIT}, {CONTEXT_CHUNKS}-chunk context",
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


def run_search_only(conversations, conv_indices, port):
    search_items = []
    conv_ids = list(conversations.keys())
    indices = conv_indices or list(range(len(conv_ids)))

    for ci in indices:
        if ci >= len(conv_ids):
            continue
        conv_id = conv_ids[ci]
        conv_data = conversations[conv_id]
        qa_list = conv_data.get("qa_pairs", [])
        collection = f"locomo_lb_{conv_id}"
        speakers = extract_speakers(conv_data.get("conversation", []))

        create_collection(collection)
        proc = start_bench_server(collection, port)

        try:
            commit_conversation(conv_data["conversation"], collection)

            for qi, qa in enumerate(qa_list):
                question = qa["question"]
                gold = str(qa.get("answer", qa.get("adversarial_answer", "")))
                category = qa.get("category", 0)
                if not gold:
                    continue

                chunks = multi_query_search(question, speakers=speakers, port=port)
                chunk_texts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in chunks]

                search_items.append(
                    {
                        "id": f"{conv_id}_{qi}",
                        "question": question,
                        "gold": gold,
                        "category": category,
                        "conv_id": conv_id,
                        "chunks": chunk_texts,
                        "num_chunks": len(chunk_texts),
                    }
                )
                time.sleep(0.1)

            print(f"  {conv_id}: {len([i for i in search_items if i['conv_id'] == conv_id])} questions searched")
        finally:
            kill_server(proc)
            delete_collection(collection)

    output_path = os.environ.get("BENCH_SEARCH_OUTPUT", str(RESULTS_DIR / "search-output.json"))
    with open(output_path, "w") as f:
        json.dump({"items": search_items, "total": len(search_items)}, f, indent=2)
    print(f"\n  Search-only: {len(search_items)} items saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, default=None)
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--search-only", action="store_true")
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

    if args.search_only:
        run_search_only(conversations, conv_indices, args.port)
        return

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
