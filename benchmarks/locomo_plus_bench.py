#!/usr/bin/env python3
"""
RASPUTIN Memory — LoCoMo-Plus Benchmark (ARR 2026)
Unified 6-category evaluation: LoCoMo (5 factual) + Cognitive (implicit memory).
Dataset: github.com/xjtuleeyf/Locomo-Plus
Scoring: correct=1, partial=0.5, wrong=0 (LLM-as-judge per category).

Usage:
    python3 benchmarks/locomo_plus_bench.py [--reset] [--limit N] [--category Cognitive]
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
UNIFIED_INPUT = BENCH_DIR / "locomo_plus_data" / "data" / "unified_input_samples_v2.json"
CHECKPOINT_FILE = RESULTS_DIR / "locomo-plus-checkpoint.json"
OUTPUT_FILE = RESULTS_DIR / "locomo-plus-results.json"
COMPARISON_FILE = RESULTS_DIR / "locomo-plus-comparison.md"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("BENCH_EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768
BENCH_PORT = 7783

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANSWER_MODEL = os.environ.get("BENCH_ANSWER_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = "gpt-4o-mini"

SEARCH_LIMIT = int(os.environ.get("BENCH_SEARCH_LIMIT", "10"))
CONTEXT_CHUNKS = int(os.environ.get("BENCH_CONTEXT_CHUNKS", "10"))

LABEL_TO_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

HAIKU_MODEL = "claude-3-haiku-20240307"

_CONSTRAINT_PROMPT = """Extract IMPLICIT constraints from the conversation below. Only extract what is actually present — never invent or assume.

Constraint types:
- GOAL: An objective the speaker is working toward
- STATE: A current situation or condition of the speaker
- VALUE: Something the speaker cares about or prioritizes
- CAUSAL: A cause-effect relationship mentioned or implied

Text:
{text}

Return a JSON array of constraints found. If none exist, return [].
Format: [{{"type": "goal|state|value|causal", "constraint": "concise description"}}]
Return ONLY the JSON array, nothing else."""


def extract_constraints_anthropic(text):
    if not text or len(text) < 30 or not ANTHROPIC_API_KEY:
        return []

    prompt = _CONSTRAINT_PROMPT.format(text=text[:2000])
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(
                    {
                        "model": HAIKU_MODEL,
                        "max_tokens": 500,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ).encode(),
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("x-api-key", ANTHROPIC_API_KEY)
            req.add_header("anthropic-version", "2023-06-01")
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode())
            raw = result["content"][0]["text"].strip()

            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1] if len(parts) >= 3 else parts[-1]
                if raw.startswith("json"):
                    raw = raw[4:]

            si = raw.find("[")
            ei = raw.rfind("]") + 1
            if si >= 0 and ei > si:
                constraints = json.loads(raw[si:ei])
                if isinstance(constraints, list):
                    return [c for c in constraints if isinstance(c, dict) and "constraint" in c][:10]
            return []
        except Exception:
            if attempt < 2:
                time.sleep(2**attempt)
    return []


# ─── Judge prompts (exact from LoCoMo-Plus paper) ───────────

JUDGE_PROMPTS = {
    "multi-hop": """You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (multi-hop fact QA).
Labels:
- "correct": The answer matches the reference entities (names, places, times) exactly.
- "partial": The answer misses some details or contains minor inaccuracies but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.
Reference Answer: {gold}
Model Prediction: {pred}
Relevant Evidence: {evidence}
Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}""",
    "single-hop": """You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (single-hop fact QA).
Labels:
- "correct": The answer matches the reference entities exactly.
- "partial": The answer misses some details but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.
Reference Answer: {gold}
Model Prediction: {pred}
Relevant Evidence: {evidence}
Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}""",
    "temporal": """You are a Temporal Logic Judge.
Your task: Check the calculation, duration, or sequence of events.
Labels:
- "correct": The calculated time, duration, or date matches the reference exactly (semantic equivalents are allowed).
- "wrong": The calculation is incorrect, the sequence is reversed, or the specific time is wrong.
Reference Answer: {gold}
Model Prediction: {pred}
Relevant Evidence: {evidence}
Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}""",
    "common-sense": """You are a Knowledge Logic Judge.
Your task: Assess if the prediction applies correct commonsense/world knowledge consistent with the reference.
Labels:
- "correct": The logic and inference are sound and match the reference conclusion.
- "partial": The reasoning is mostly correct but the final conclusion is vague or slightly off.
- "wrong": The reasoning contradicts commonsense or the reference.
Reference Answer: {gold}
Model Prediction: {pred}
Relevant Evidence: {evidence}
Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}""",
    "adversarial": """You are a Skeptical Judge evaluating robustness.
The question is inherently misleading (e.g., asks about something not in the conversation).
Your task: Judge whether the model's answer conveys that "this was not mentioned in the conversation" (or equivalent refusal).
Labels:
- "correct": The prediction clearly conveys that the information was not mentioned / cannot be answered from the conversation.
- "wrong": The prediction does NOT convey that meaning.
Model Prediction: {pred}
Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}""",
    "Cognitive": """You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).
Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.
Memory/Evidence: {evidence}
Model Prediction: {pred}
Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}""",
}


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
GEMINI_API_KEY_BENCH = os.environ.get("GEMINI_API_KEY", "")


def get_embedding(text, prefix="search_document: "):
    if BENCH_EMBED_PROVIDER == "gemini" and GEMINI_API_KEY_BENCH:
        import math

        if "query" in prefix.lower():
            task_type = "RETRIEVAL_QUERY"
        else:
            task_type = "RETRIEVAL_DOCUMENT"

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-embedding-001:embedContent?key={GEMINI_API_KEY_BENCH}"
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


# ─── Conversation parsing ────────────────────────────────────

_TURN_RE = re.compile(r'^(.+?)\s+said,\s+"(.+)"$', re.MULTILINE)
_DATE_RE = re.compile(r"^DATE:\s*(.+)$", re.MULTILINE)


def parse_turns_from_prompt(input_prompt):
    turns = []
    current_date = ""
    for line in input_prompt.split("\n"):
        line = line.strip()
        if not line:
            continue
        date_m = _DATE_RE.match(line)
        if date_m:
            current_date = date_m.group(1).strip()
            continue
        if line == "CONVERSATION:":
            continue
        turn_m = _TURN_RE.match(line)
        if turn_m:
            speaker = turn_m.group(1)
            text = turn_m.group(2)
            turns.append({"speaker": speaker, "text": text, "date": current_date})
    return turns


# ─── Qdrant + commit ────────────────────────────────────────


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


def commit_turns(turns, collection, constraints_fn=None):
    points = []
    committed = 0
    all_texts = []

    for t in turns:
        text = f"[{t['date']}] {t['speaker']}: {t['text']}"
        all_texts.append(text)

    for text in all_texts:
        try:
            vec = get_embedding(text[:2000], prefix="search_document: ")
            pid = int(hashlib.md5(text.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": pid,
                    "vector": vec,
                    "payload": {
                        "text": text[:4000],
                        "source": "locomo_plus",
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
        except Exception:
            pass

    window_committed = 0
    constraint_committed = 0
    for i in range(0, max(len(all_texts) - 4, 1), 2):
        window = all_texts[i : i + 5]
        wtext = "\n".join(window)
        try:
            vec = get_embedding(wtext[:2000], prefix="search_document: ")
            pid = int(hashlib.md5(wtext.encode()).hexdigest()[:15], 16)
            points.append(
                {
                    "id": pid,
                    "vector": vec,
                    "payload": {
                        "text": wtext[:4000],
                        "source": "locomo_plus",
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

        if constraints_fn:
            try:
                extracted = constraints_fn(wtext)
                if extracted:
                    summary = " | ".join(c.get("constraint", "") for c in extracted)
                    ctext = f"[CONSTRAINTS] {summary}"
                    cvec = get_embedding(ctext[:2000], prefix="search_document: ")
                    cpid = int(hashlib.md5(ctext.encode()).hexdigest()[:15], 16)
                    points.append(
                        {
                            "id": cpid,
                            "vector": cvec,
                            "payload": {
                                "text": ctext[:4000],
                                "source": "locomo_plus",
                                "source_weight": 1.0,
                                "date": datetime.now().isoformat(),
                                "importance": 80,
                                "retrieval_count": 0,
                                "chunk_type": "constraint",
                                "constraints": extracted,
                            },
                        }
                    )
                    constraint_committed += 1
                    if len(points) >= 50:
                        http_json(
                            f"{QDRANT_URL}/collections/{collection}/points",
                            data={"points": points},
                            method="PUT",
                            timeout=30,
                        )
                        points = []
            except Exception:
                pass

    if points:
        http_json(f"{QDRANT_URL}/collections/{collection}/points", data={"points": points}, method="PUT", timeout=30)
    time.sleep(1)
    return committed, window_committed, constraint_committed


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
    env["CONSTRAINTS_ENABLED"] = os.environ.get("CONSTRAINTS_ENABLED", "true")
    env["CONSTRAINTS_PROVIDER"] = os.environ.get("CONSTRAINTS_PROVIDER", "anthropic")
    env["RERANK_PROVIDER"] = os.environ.get("RERANK_PROVIDER", "cohere")
    env["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY", "")
    env["LLM_RERANKER"] = os.environ.get("LLM_RERANKER", "false")
    env["PYTHONPATH"] = str(REPO / "tools")
    log_fh = open(RESULTS_DIR / "locomo-plus-server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "tools" / "hybrid_brain.py"), "--port", str(port)],
        cwd=str(REPO / "tools"),
        env=env,
        stdout=log_fh,
        stderr=log_fh,
    )
    for _ in range(30):
        time.sleep(1)
        try:
            http_json(f"http://localhost:{port}/health")
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


def deduplicate_results(results, overlap_threshold=0.75):
    if not results:
        return results
    tokenize = re.compile(r"\w+", re.UNICODE)
    selected, token_sets = [], []
    for r in results:
        tokens = set(tokenize.findall((r.get("text") or "").lower()))
        if not tokens:
            continue
        is_dup = any(len(tokens & ex) / min(len(tokens), len(ex)) > overlap_threshold for ex in token_sets if ex)
        if not is_dup:
            selected.append(r)
            token_sets.append(tokens)
    return selected


# ─── LLM: answer generation ─────────────────────────────────


def generate_answer(question, context_chunks, max_chunks=CONTEXT_CHUNKS):
    context = "\n".join(f"- {c.get('text', c) if isinstance(c, dict) else c}" for c in context_chunks[:max_chunks])
    prompt = f"""You are answering questions about a conversation based on retrieved memory snippets.
Answer concisely in 1-3 sentences. Be direct and specific.

IMPORTANT: If the question attributes an action or fact to Person A, but the memories show
it was actually Person B who did it, STILL PROVIDE THE FACTUAL ANSWER.

If NO relevant facts exist in the memories for ANY person, say "I don't have enough
information to answer this question."

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
                        "max_tokens": 200,
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
        except Exception:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise


# ─── LLM: judge ──────────────────────────────────────────────


def judge_sample(category, prediction, evidence, gold=""):
    template = JUDGE_PROMPTS.get(category, JUDGE_PROMPTS.get("single-hop"))
    prompt = template.format(gold=gold or "", pred=prediction or "", evidence=evidence or "")

    data = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    for attempt in range(5):
        try:
            result = http_json(
                "https://api.openai.com/v1/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=30,
            )
            raw = result["choices"][0]["message"]["content"].strip()
            label = ""
            try:
                m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
                if m:
                    label = m.group(1).strip().lower()
                else:
                    obj = json.loads(raw)
                    label = (obj.get("label") or "").strip().lower()
            except Exception:
                if "correct" in raw.lower():
                    label = "correct"
                elif "wrong" in raw.lower():
                    label = "wrong"
                elif "partial" in raw.lower():
                    label = "partial"
            return LABEL_TO_SCORE.get(label, 0.0), label
        except Exception:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise


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


# ─── Report ──────────────────────────────────────────────────


def generate_report(state):
    results = state.get("results", {})
    if not results:
        return

    by_cat = defaultdict(list)
    all_scores = []
    for info in results.values():
        score = info.get("judge_score", 0)
        by_cat[info.get("category", "unknown")].append(score)
        all_scores.append(score)

    overall = sum(all_scores) / len(all_scores) * 100 if all_scores else 0
    factual_cats = ["single-hop", "multi-hop", "temporal", "common-sense"]
    factual_scores = [s for cat in factual_cats for s in by_cat.get(cat, [])]
    factual_avg = sum(factual_scores) / len(factual_scores) * 100 if factual_scores else 0
    cognitive_scores = by_cat.get("Cognitive", [])
    cognitive_avg = sum(cognitive_scores) / len(cognitive_scores) * 100 if cognitive_scores else 0

    lines = [
        "# RASPUTIN Memory — LoCoMo-Plus Benchmark (ARR 2026)",
        f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total samples:** {len(all_scores)}",
        f"\n## Overall: {overall:.2f}%",
        f"## Factual (Level-1): {factual_avg:.2f}%",
        f"## Cognitive (Level-2): {cognitive_avg:.2f}%",
        "\n## Per-Category",
    ]
    for cat in ["single-hop", "multi-hop", "temporal", "common-sense", "adversarial", "Cognitive"]:
        scores = by_cat.get(cat, [])
        if not scores:
            continue
        avg = sum(scores) / len(scores) * 100
        lines.append(f"- **{cat}**: {avg:.1f}% ({sum(1 for s in scores if s >= 1)}/{len(scores)})")

    report = "\n".join(lines)
    with open(COMPARISON_FILE, "w") as f:
        f.write(report + "\n")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "overall": overall,
                "factual": factual_avg,
                "cognitive": cognitive_avg,
                "per_category": {c: sum(s) / len(s) * 100 for c, s in by_cat.items()},
                "total": len(all_scores),
            },
            f,
            indent=2,
        )
    print(f"\nReport: {COMPARISON_FILE}")


# ─── Main pipeline ───────────────────────────────────────────


def run_benchmark(samples, limit=None, category_filter=None, use_constraints=False):
    constraints_fn = extract_constraints_anthropic if use_constraints else None

    if category_filter:
        samples = [s for s in samples if s["category"] == category_filter]
    if limit:
        samples = samples[:limit]

    state = load_checkpoint()
    results = state.get("results", {})
    done_ids = set(results.keys())

    pending = []
    for i, s in enumerate(samples):
        sid = f"{s['category']}_{i}"
        if sid not in done_ids:
            pending.append((sid, s))

    mode = " [+constraints]" if use_constraints else ""
    print(f"LoCoMo-Plus{mode}: {len(samples)} samples, {len(done_ids)} done, {len(pending)} pending\n")

    conv_groups = defaultdict(list)
    for sid, s in pending:
        conv_key = hashlib.md5(s["input_prompt"][:500].encode()).hexdigest()[:12]
        conv_groups[conv_key].append((sid, s))

    proc = None
    current_collection = None

    try:
        for gi, (conv_key, group) in enumerate(conv_groups.items()):
            cpfx = "lcc" if use_constraints else "lcp"
            collection = f"{cpfx}_{conv_key}"

            kill_server(proc)
            if current_collection:
                delete_collection(current_collection)

            first_sample = group[0][1]
            turns = parse_turns_from_prompt(first_sample["input_prompt"])
            create_collection(collection)
            nt, nw, nc = commit_turns(turns, collection, constraints_fn=constraints_fn)
            constraint_info = f" + {nc} constraints" if nc else ""
            print(
                f"\n[Group {gi + 1}/{len(conv_groups)}] {len(group)} samples, {nt} turns + {nw} windows{constraint_info}"
            )

            proc = start_bench_server(collection, BENCH_PORT)
            current_collection = collection

            for si, (sid, sample) in enumerate(group):
                trigger = sample.get("trigger", "")
                category = sample.get("category", "")
                evidence = sample.get("evidence", "")
                gold = sample.get("answer", "")

                chunks = search_query(trigger, port=BENCH_PORT, limit=SEARCH_LIMIT)
                chunks = deduplicate_results(chunks)

                if not chunks:
                    prediction = "I don't have enough information to answer this question."
                else:
                    prediction = generate_answer(trigger, chunks)

                score, label = judge_sample(category, prediction, evidence, gold)

                results[sid] = {
                    "category": category,
                    "trigger": trigger,
                    "evidence": evidence[:300],
                    "gold": gold,
                    "prediction": prediction,
                    "judge_label": label,
                    "judge_score": score,
                    "num_chunks": len(chunks),
                }

                total = len(results)
                correct = sum(1 for r in results.values() if r.get("judge_score", 0) >= 1)
                status = "✓" if score >= 1 else ("½" if score == 0.5 else "✗")
                print(
                    f"  [{si + 1}/{len(group)}] {status} {category[:12]:12s} ({correct}/{total} = {correct / total * 100:.1f}%)"
                )

                if total % 50 == 0:
                    state["results"] = results
                    save_checkpoint(state)

            state["results"] = results
            save_checkpoint(state)

    finally:
        kill_server(proc)
        if current_collection:
            delete_collection(current_collection)

    generate_report(state)


# ─── Entry point ─────────────────────────────────────────────


def main():
    global CHECKPOINT_FILE, OUTPUT_FILE, COMPARISON_FILE, BENCH_PORT

    parser = argparse.ArgumentParser(description="RASPUTIN Memory — LoCoMo-Plus Benchmark")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--constraints", action="store_true", help="Enable constraint extraction via Claude Haiku")
    parser.add_argument("--tag", type=str, default=None, help="Run tag for file naming (auto-set by --constraints)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default 7783, auto 7784 for constraints)")
    args = parser.parse_args()

    tag = args.tag or ("constraints" if args.constraints else "baseline")
    if args.port:
        BENCH_PORT = args.port
    elif args.constraints:
        BENCH_PORT = 7784

    if tag != "baseline":
        CHECKPOINT_FILE = RESULTS_DIR / f"locomo-plus-{tag}-checkpoint.json"
        OUTPUT_FILE = RESULTS_DIR / f"locomo-plus-{tag}-results.json"
        COMPARISON_FILE = RESULTS_DIR / f"locomo-plus-{tag}-comparison.md"

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared")

    if not UNIFIED_INPUT.exists():
        print(f"Unified input not found: {UNIFIED_INPUT}")
        print("Run: cd benchmarks/locomo_plus_data/data && python3 unified_input.py")
        sys.exit(1)

    with open(UNIFIED_INPUT) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples (tag={tag}, constraints={'ON' if args.constraints else 'OFF'})")
    cats = Counter(s["category"] for s in samples)
    for c, n in cats.most_common():
        print(f"  {c}: {n}")

    run_benchmark(samples, limit=args.limit, category_filter=args.category, use_constraints=args.constraints)


if __name__ == "__main__":
    main()
