from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
BRAIN_URL = os.environ.get("BRAIN_URL", "http://localhost:7777")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-sonnet-4-20250514")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-20250414")
SEARCH_LIMIT = int(os.environ.get("SEARCH_LIMIT", "60"))
LOCOMO_PATH = Path(__file__).parent / "locomo" / "locomo10.json"

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "locomo-checkpoint.json"

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
EMBED_PREFIX_DOC = os.environ.get("EMBED_PREFIX_DOC", "search_document: ")

CATEGORY_NAMES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Multi-hop",
    4: "Open-domain",
    5: "Adversarial",
}

LEADERBOARD_BASELINES = {
    "Single-hop": {"Backboard": 89.4, "MemMachine": 93.3, "Memvid": 80.1, "Zep": 74.1, "mem0": None},
    "Multi-hop": {"Backboard": 75.0, "MemMachine": 80.5, "Memvid": 80.4, "Zep": 66.0, "mem0": None},
    "Temporal": {"Backboard": 91.9, "MemMachine": 72.6, "Memvid": 71.9, "Zep": 79.8, "mem0": None},
    "Open-domain": {"Backboard": 91.2, "MemMachine": 64.6, "Memvid": 91.1, "Zep": 67.7, "mem0": None},
    "Overall": {"Backboard": 90.0, "MemMachine": 84.9, "Memvid": 85.7, "Zep": 75.1, "mem0": 66.9},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_json(
    url: str,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int = 30,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method or "POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method or "GET")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = str(error)
        raise RuntimeError(f"HTTP {error.code} for {url}: {body}") from error
    except Exception as error:
        raise RuntimeError(f"Request failed for {url}: {error}") from error

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON from {url}: {raw[:200]}") from error


def anthropic_completion(prompt: str, model: str, max_tokens: int) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    backoff = 0.5
    for attempt in range(3):
        time.sleep(0.1)
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ).encode("utf-8"),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", ANTHROPIC_API_KEY)
        req.add_header("anthropic-version", "2023-06-01")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["content"][0]["text"].strip()
            if text:
                return text
            return ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(backoff)
            backoff *= 2
    return ""


def get_brain_collection() -> str:
    try:
        stats = http_json(f"{BRAIN_URL}/stats", timeout=10)
        collection = str(stats.get("qdrant", {}).get("collection") or "").strip()
        if collection:
            return collection
    except Exception:
        pass
    return os.environ.get("QDRANT_COLLECTION", "second_brain")


def get_vector_size(collection: str) -> int:
    try:
        payload = http_json(f"{QDRANT_URL}/collections/{collection}", timeout=10)
        vectors = payload.get("result", {}).get("config", {}).get("params", {}).get("vectors")
        if isinstance(vectors, dict):
            if "size" in vectors:
                return int(vectors["size"])
            if vectors:
                first_value = next(iter(vectors.values()))
                if isinstance(first_value, dict) and "size" in first_value:
                    return int(first_value["size"])
    except Exception:
        pass
    return EMBED_DIM


def delete_collection(collection: str) -> None:
    try:
        req = urllib.request.Request(f"{QDRANT_URL}/collections/{collection}", method="DELETE")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def create_collection(collection: str, dim: int) -> None:
    delete_collection(collection)
    time.sleep(0.2)
    http_json(
        f"{QDRANT_URL}/collections/{collection}",
        data={
            "vectors": {"size": dim, "distance": "Cosine"},
            "optimizers_config": {"indexing_threshold": 0},
        },
        method="PUT",
        timeout=20,
    )


def parse_embedding_response(payload: dict[str, Any]) -> list[float]:
    if "embeddings" in payload and payload["embeddings"]:
        return payload["embeddings"][0]
    if "data" in payload and payload["data"]:
        return payload["data"][0]["embedding"]
    if "embedding" in payload:
        return payload["embedding"]
    raise RuntimeError(f"Unexpected embedding response keys: {list(payload.keys())}")


def get_embedding(text: str) -> list[float]:
    prefixed = f"{EMBED_PREFIX_DOC}{text}" if EMBED_PREFIX_DOC else text
    payload = http_json(
        EMBED_URL,
        data={"model": EMBED_MODEL, "input": prefixed},
        method="POST",
        timeout=45,
    )
    return parse_embedding_response(payload)


def iter_turns(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    session_idx = 1
    while True:
        session_key = f"session_{session_idx}"
        if session_key not in conversation:
            break
        session_date = str(conversation.get(f"session_{session_idx}_date_time", f"session-{session_idx}"))
        session_turns = conversation.get(session_key, [])
        for turn_index, turn in enumerate(session_turns):
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            speaker = str(turn.get("speaker", "Unknown")).strip() or "Unknown"
            turns.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "session_index": session_idx,
                    "session_date": session_date,
                    "turn_index": turn_index,
                }
            )
        session_idx += 1
    return turns


def upsert_points(collection: str, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    http_json(
        f"{QDRANT_URL}/collections/{collection}/points?wait=true",
        data={"points": points},
        method="PUT",
        timeout=60,
    )


def ingest_conversation(conversation: dict[str, Any], sample_id: str, collection: str) -> dict[str, Any]:
    turns = iter_turns(conversation)
    speakers = sorted({turn["speaker"] for turn in turns})
    points: list[dict[str, Any]] = []
    committed = 0
    for turn in turns:
        memory_text = f"[{turn['session_date']}] {turn['speaker']}: {turn['text']}"
        vector = get_embedding(memory_text[:2000])
        hash_base = f"{sample_id}:{turn['session_index']}:{turn['turn_index']}:{memory_text}"
        point_id = int(hashlib.md5(hash_base.encode("utf-8")).hexdigest()[:15], 16)
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "text": memory_text[:4000],
                    "source": "locomo_bench",
                    "source_weight": 1.0,
                    "date": now_iso(),
                    "importance": 70,
                    "retrieval_count": 0,
                    "speaker": turn["speaker"],
                    "session_date": turn["session_date"],
                    "session_index": turn["session_index"],
                    "turn_index": turn["turn_index"],
                    "sample_id": sample_id,
                },
            }
        )
        committed += 1
        if len(points) >= 64:
            upsert_points(collection, points)
            points = []
    upsert_points(collection, points)
    time.sleep(1)
    return {"turns": committed, "speakers": speakers}


def search_memories(question: str, collection: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "q": question,
            "limit": limit,
            "collection": collection,
        }
    )
    payload = http_json(f"{BRAIN_URL}/search?{params}", timeout=45)
    return payload.get("results", []), payload.get("stats", {})


def generate_answer(question: str, context_chunks: list[dict[str, Any]], model: str) -> str:
    context = "\n".join(f"- {row.get('text', '')}" for row in context_chunks[:25])
    prompt = f"""Extract the answer from these conversation memories.
Reply with ONLY the answer — 1-10 words maximum. No explanation.

Memories:
{context}

Q: {question}
A:"""
    text = anthropic_completion(prompt, model=model, max_tokens=40)
    return text.strip().splitlines()[0].strip()


def judge_answer(question: str, prediction: str, ground_truth: str, model: str) -> float:
    prompt = f"""Is this answer correct?

Question: {question}
Ground Truth: {ground_truth}
System Answer: {prediction}

Score CORRECT only if the answer contains the specific information asked for. Score WRONG if the answer is vague, missing key facts, or incorrect.
Reply with exactly: CORRECT or WRONG"""
    text = anthropic_completion(prompt, model=model, max_tokens=10)
    return 1.0 if "CORRECT" in text.upper() else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = re.findall(r"\w+", prediction.lower())
    truth_tokens = re.findall(r"\w+", str(ground_truth).lower())
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    overlap = Counter(pred_tokens) & Counter(truth_tokens)
    matched = sum(overlap.values())
    if matched == 0:
        return 0.0
    precision = matched / len(pred_tokens)
    recall = matched / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_map[int(row.get("category", 0))].append(row)

    category_metrics: dict[int, dict[str, Any]] = {}
    for category, values in sorted(category_map.items()):
        category_metrics[category] = {
            "name": CATEGORY_NAMES.get(category, f"Category {category}"),
            "n_questions": len(values),
            "token_f1": sum(v["f1"] for v in values) / len(values),
            "judge_accuracy": sum(v["judge"] for v in values) / len(values),
        }

    non_adv = [row for row in rows if int(row.get("category", 0)) != 5]
    target_rows = non_adv if non_adv else rows
    overall = {
        "n_questions": len(rows),
        "n_questions_scored": len(target_rows),
        "token_f1": (sum(row["f1"] for row in target_rows) / len(target_rows)) if target_rows else 0.0,
        "judge_accuracy": (sum(row["judge"] for row in target_rows) / len(target_rows)) if target_rows else 0.0,
    }
    return {"overall": overall, "categories": category_metrics}


def format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}%"


def leaderboard_markdown(summary: dict[str, Any], date_tag: str, answer_model: str, judge_model: str) -> str:
    categories = summary.get("categories", {})

    def judge_pct(category_id: int) -> float:
        metric = categories.get(category_id)
        if not metric:
            return 0.0
        return float(metric.get("judge_accuracy", 0.0)) * 100

    overall_pct = float(summary.get("overall", {}).get("judge_accuracy", 0.0)) * 100
    lines = [
        "# LoCoMo Benchmark Results — RASPUTIN Memory",
        "",
        f"**Date:** {date_tag}",
        f"**Config:** {answer_model} (answer) + {judge_model} (judge + reranker)",
        f"**Embedding:** {EMBED_MODEL} ({EMBED_DIM}d) via Ollama",
        "**Pipeline:** Vector + BM25 + LLM Reranker + Entity Boost",
        "",
        "## Results (LLM-Judge Accuracy)",
        "",
        "| Category | RASPUTIN | Backboard | MemMachine | Memvid | Zep | mem0 |",
        "|----------|----------|-----------|------------|--------|-----|------|",
    ]

    rows = [
        ("Single-hop", judge_pct(1)),
        ("Multi-hop", judge_pct(3)),
        ("Temporal", judge_pct(2)),
        ("Open-domain", judge_pct(4)),
        ("Overall", overall_pct),
    ]

    for name, rasputin in rows:
        baselines = LEADERBOARD_BASELINES[name]
        prefix = "**" if name == "Overall" else ""
        suffix = "**" if name == "Overall" else ""
        lines.append(
            f"| {prefix}{name}{suffix} | {prefix}{format_pct(rasputin)}{suffix} | "
            f"{prefix}{format_pct(baselines['Backboard'])}{suffix} | "
            f"{prefix}{format_pct(baselines['MemMachine'])}{suffix} | "
            f"{prefix}{format_pct(baselines['Memvid'])}{suffix} | "
            f"{prefix}{format_pct(baselines['Zep'])}{suffix} | "
            f"{prefix}{format_pct(baselines['mem0'])}{suffix} |"
        )

    lines.append("")
    lines.append("## Internal Token-F1 (excluding adversarial)")
    lines.append("")
    lines.append(f"- Overall F1: {float(summary.get('overall', {}).get('token_f1', 0.0)) * 100:.2f}%")
    return "\n".join(lines)


def save_checkpoint(state: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {}
    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_conversation(
    conv_index: int,
    conv_data: dict[str, Any],
    vector_size: int,
    answer_model: str,
    judge_model: str,
    search_limit: int,
) -> dict[str, Any]:
    sample_id = str(conv_data.get("sample_id", f"conv-{conv_index}"))
    safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", sample_id).strip("_") or f"conv_{conv_index}"
    collection = f"locomo_bench_{safe_id}_{int(time.time())}"
    create_collection(collection, vector_size)
    try:
        ingest_info = ingest_conversation(conv_data["conversation"], sample_id, collection)
        qa_rows: list[dict[str, Any]] = []
        for qa_index, qa in enumerate(conv_data.get("qa", [])):
            question = str(qa.get("question", "")).strip()
            ground_truth = str(qa.get("answer", qa.get("adversarial_answer", ""))).strip()
            category = int(qa.get("category", 0) or 0)
            if not question or not ground_truth:
                continue
            chunks, search_stats = search_memories(question, collection, search_limit)
            prediction = generate_answer(question, chunks, answer_model)
            judge = judge_answer(question, prediction, ground_truth, judge_model)
            f1 = token_f1(prediction, ground_truth)
            qa_rows.append(
                {
                    "qa_index": qa_index,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, f"Category {category}"),
                    "f1": f1,
                    "judge": judge,
                    "retrieved": len(chunks),
                    "search_stats": search_stats,
                }
            )
        summary = summarize_rows(qa_rows)
        return {
            "conversation_index": conv_index,
            "sample_id": sample_id,
            "collection": collection,
            "ingested_turns": ingest_info["turns"],
            "speakers": ingest_info["speakers"],
            "summary": summary,
            "qa": qa_rows,
            "completed_at": now_iso(),
        }
    finally:
        delete_collection(collection)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark harness for RASPUTIN Memory")
    parser.add_argument("--conversations", type=int, nargs="+", help="Conversation indices to run")
    parser.add_argument("--resume", action="store_true", help="Resume from benchmarks/results/locomo-checkpoint.json")
    parser.add_argument("--answer-model", type=str, default=ANSWER_MODEL)
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--search-limit", type=int, default=SEARCH_LIMIT)
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is required")

    with LOCOMO_PATH.open(encoding="utf-8") as f:
        conversations = json.load(f)

    selected_indices = args.conversations or list(range(len(conversations)))

    state = load_checkpoint() if args.resume else {}
    completed = state.get("completed", {})
    errors = state.get("errors", {})

    brain_collection = get_brain_collection()
    vector_size = get_vector_size(brain_collection)

    for conv_index in selected_indices:
        conv_data = conversations[conv_index]
        sample_id = str(conv_data.get("sample_id", f"conv-{conv_index}"))
        if args.resume and sample_id in completed:
            print(f"[{conv_index}] skipping {sample_id} (checkpoint)")
            continue

        print(f"[{conv_index}] running {sample_id}")
        try:
            result = run_conversation(
                conv_index,
                conv_data,
                vector_size,
                args.answer_model,
                args.judge_model,
                args.search_limit,
            )
            completed[sample_id] = result
            if sample_id in errors:
                del errors[sample_id]
            judge_score = float(result["summary"]["overall"]["judge_accuracy"]) * 100
            print(f"[{conv_index}] done {sample_id} judge={judge_score:.2f}%")
        except Exception as error:
            errors[sample_id] = str(error)
            print(f"[{conv_index}] failed {sample_id}: {error}")
        finally:
            state = {
                "updated_at": now_iso(),
                "answer_model": args.answer_model,
                "judge_model": args.judge_model,
                "search_limit": args.search_limit,
                "brain_collection": brain_collection,
                "vector_size": vector_size,
                "completed": completed,
                "errors": errors,
            }
            save_checkpoint(state)

    completed_results = sorted(completed.values(), key=lambda row: row.get("conversation_index", 0))
    all_rows = [qa for conv in completed_results for qa in conv.get("qa", [])]
    summary = summarize_rows(all_rows)

    date_tag = datetime.now().strftime("%Y-%m-%d")
    output = {
        "date": date_tag,
        "config": {
            "qdrant_url": QDRANT_URL,
            "brain_url": BRAIN_URL,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "search_limit": args.search_limit,
            "embedding_url": EMBED_URL,
            "embedding_model": EMBED_MODEL,
            "embedding_dimensions": EMBED_DIM,
            "brain_collection": brain_collection,
            "vector_size": vector_size,
        },
        "summary": summary,
        "conversations": completed_results,
        "errors": errors,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"locomo-{date_tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    markdown = leaderboard_markdown(summary, date_tag, args.answer_model, args.judge_model)
    markdown_path = RESULTS_DIR / "LATEST-LOCOMO.md"
    with markdown_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"saved: {json_path}")
    print(f"saved: {markdown_path}")


if __name__ == "__main__":
    main()
