#!/usr/bin/env python3
"""Pre-compute consolidation observations for all LoCoMo conversations.

Reads facts from existing Qdrant collections (created by the benchmark with
BENCH_KEEP_COLLECTIONS=1), consolidates them via LLM into higher-level
observations, and stores them in {collection}_obs collections.

Phase 2 search adds these observations as a third retrieval lane.

Usage:
    python3 benchmarks/precompute_consolidation.py [--conversations 0,1,2]
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCOMO_FILE = REPO / "benchmarks" / "locomo" / "locomo10.json"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

CONSOLIDATION_URL = os.environ.get("CONSOLIDATION_URL", "https://api.cerebras.ai/v1/chat/completions")
CONSOLIDATION_MODEL = os.environ.get("CONSOLIDATION_MODEL", "qwen-3-235b-a22b-instruct-2507")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

BATCH_SIZE = int(os.environ.get("CONSOLIDATION_BATCH_SIZE", "30"))


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


def get_embedding(text, prefix="search_document: "):
    prefixed = f"{prefix}{text}" if prefix else text
    result = http_json(EMBED_URL, data={"model": EMBED_MODEL, "input": prefixed}, timeout=30)
    if "embeddings" in result:
        return result["embeddings"][0]
    elif "data" in result:
        return result["data"][0]["embedding"]
    raise ValueError(f"Unexpected embed response: {list(result.keys())}")


CONSOLIDATION_PROMPT = """You are a memory consolidation system. Synthesize facts into observations.

Track every meaningful detail: names, numbers, dates, places, relationships, preferences,
personality traits, hobbies, career goals, and life circumstances. Prefer specifics over
abstractions. Aggregate when multiple facts describe the same topic.

RULES:
1. ONE OBSERVATION PER DISTINCT FACET. "Alice's hobbies" and "Alice's job" are separate.
2. AGGREGATE scattered mentions. If pottery in fact 3, camping in fact 7 → one observation.
3. STATE CHANGES → UPDATE.
4. RESOLVE REFERENCES. "home country" → "Sweden" when context provides it.
5. CAPTURE PERSONALITY AND PREFERENCES — these are HIGH VALUE.
6. CAPTURE RELATIONSHIPS. "Emily is user's college roommate."
7. Include entity names, dates, specifics. Each observation must be independently useful.

NEW FACTS:
{facts_text}

EXISTING OBSERVATIONS:
{observations_text}

Return JSON: {{"creates": [{{"text": "...", "source_fact_ids": ["id1"]}}],
               "updates": [{{"text": "...", "observation_id": "...", "source_fact_ids": ["id2"]}}],
               "deletes": [{{"observation_id": "..."}}]}}

Return {{"creates": [], "updates": [], "deletes": []}} if nothing durable."""


def llm_call(prompt):
    body = json.dumps(
        {
            "model": CONSOLIDATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4000,
        }
    ).encode()
    req = urllib.request.Request(CONSOLIDATION_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {CEREBRAS_API_KEY}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            wait = 10 * (attempt + 1) if e.code in (403, 429) else 2**attempt
            if attempt < 4:
                print(f"      LLM retry {attempt + 1}/5 (HTTP {e.code}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < 4:
                time.sleep(2**attempt)
            else:
                raise


def parse_response(content):
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) >= 3 else parts[-1]
        if content.startswith("json"):
            content = content[4:]
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
    return {"creates": [], "updates": [], "deletes": []}


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


def scroll_facts(collection):
    all_facts = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        data = http_json(f"{QDRANT_URL}/collections/{collection}/points/scroll", data=body, method="POST", timeout=30)
        points = data.get("result", {}).get("points", [])
        for p in points:
            payload = p.get("payload", {})
            if payload.get("chunk_type") == "fact":
                all_facts.append({"id": str(p["id"]), "text": payload.get("text", "")})
        next_offset = data.get("result", {}).get("next_page_offset")
        if not next_offset:
            break
        offset = next_offset
    return all_facts


def recall_observations(query_text, obs_collection, limit=10):
    vec = get_embedding(query_text[:2000], prefix="search_query: ")
    try:
        data = http_json(
            f"{QDRANT_URL}/collections/{obs_collection}/points/query",
            data={"query": vec, "limit": limit, "with_payload": True, "score_threshold": 0.4},
            method="POST",
            timeout=10,
        )
        return [
            {
                "id": str(p["id"]),
                "text": p.get("payload", {}).get("text", ""),
                "proof_count": p.get("payload", {}).get("proof_count", 1),
            }
            for p in data.get("result", {}).get("points", [])
        ]
    except Exception:
        return []


def store_observation(text, source_ids, obs_collection):
    vec = get_embedding(text[:2000], prefix="search_document: ")
    point_id = int(hashlib.md5(text.encode()).hexdigest()[:15], 16)
    http_json(
        f"{QDRANT_URL}/collections/{obs_collection}/points",
        data={
            "points": [
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "text": text,
                        "chunk_type": "observation",
                        "fact_type": "observation",
                        "source_fact_ids": source_ids,
                        "proof_count": len(source_ids),
                        "source": "consolidation",
                        "importance": 80,
                        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                }
            ]
        },
        method="PUT",
        timeout=10,
    )


def consolidate_collection(collection):
    obs_col = f"{collection}_obs"
    create_collection(obs_col)

    facts = scroll_facts(collection)
    if not facts:
        print(f"  {collection}: no facts found, skipping")
        return 0

    print(f"  {collection}: consolidating {len(facts)} facts...")
    total_created = 0

    for i in range(0, len(facts), BATCH_SIZE):
        batch = facts[i : i + BATCH_SIZE]
        facts_text = "\n".join(f"[{f['id']}] {f['text']}" for f in batch)

        batch_summary = " ".join(f["text"][:100] for f in batch[:3])
        existing = recall_observations(batch_summary, obs_col)
        obs_text = json.dumps(existing, indent=2) if existing else "[]"

        prompt = CONSOLIDATION_PROMPT.format(facts_text=facts_text, observations_text=obs_text)

        try:
            content = llm_call(prompt)
            actions = parse_response(content)
            for create in actions.get("creates", []):
                text = create.get("text", "")
                if text and len(text) >= 15:
                    store_observation(text, create.get("source_fact_ids", []), obs_col)
                    total_created += 1
            for update in actions.get("updates", []):
                text = update.get("text", "")
                if text and len(text) >= 15:
                    store_observation(text, update.get("source_fact_ids", []), obs_col)
                    total_created += 1
        except Exception as e:
            print(f"    Batch {i // BATCH_SIZE + 1} failed: {e}")
        time.sleep(3)

    print(f"  {collection}: created {total_created} observations")
    return total_created


def main():
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)

    conv_indices = list(range(len(conversations)))
    if len(sys.argv) > 1 and "--conversations" in sys.argv:
        idx = sys.argv.index("--conversations")
        conv_indices = [int(x) for x in sys.argv[idx + 1].split(",")]

    total_obs = 0
    for idx in conv_indices:
        conv = conversations[idx]
        conv_id = conv.get("sample_id", f"conv-{idx}")
        collection = f"locomo_lb_{conv_id.replace('-', '_')}"
        total_obs += consolidate_collection(collection)

    print(f"\nTotal observations created: {total_obs}")


if __name__ == "__main__":
    main()
