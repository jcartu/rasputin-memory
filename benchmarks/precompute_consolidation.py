#!/usr/bin/env python3
"""Pre-compute consolidation observations for LoCoMo conversations.

Hindsight-style consolidation: per-fact recall, batch-8, update-not-create,
post-processing dedup. Target: 30-80 observations per conversation.

Usage:
    python3 benchmarks/precompute_consolidation.py [--conversations 0,1,2]
"""

import hashlib
import json
import math
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

CONSOLIDATION_URL = os.environ.get("CONSOLIDATION_URL", "https://api.openai.com/v1/chat/completions")
CONSOLIDATION_MODEL = os.environ.get("CONSOLIDATION_MODEL", "gpt-4o-mini")
LLM_API_KEY = (
    os.environ.get("OPENAI_API_KEY", "") or os.environ.get("GROQ_API_KEY", "") or os.environ.get("CEREBRAS_API_KEY", "")
)

BATCH_SIZE = 8
RECALL_PER_FACT = 5
DEDUP_THRESHOLD = 0.85


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
    req.add_header("User-Agent", "rasputin-memory/1.0")
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


CONSOLIDATION_PROMPT = """You are a memory consolidation system. Given new facts and existing observations, produce a MINIMAL set of durable observations.

CRITICAL RULES:
1. SAME ENTITY + SAME FACET = UPDATE the existing observation. Do NOT create a duplicate.
   - "Alice's hobbies" already exists and new facts mention a hobby → UPDATE it.
2. ONE observation per entity-facet pair. "Alice's career" and "Alice's hobbies" are separate.
3. AGGREGATE all scattered mentions into a single observation per facet.
4. Each observation must be independently useful — include names, dates, specifics.
5. PREFER UPDATES over creates. Only create when NO existing observation covers the facet.
6. State changes → update the existing observation with the new state, noting the change.

NEW FACTS:
{facts_text}

EXISTING OBSERVATIONS (with their source facts):
{observations_text}

Return JSON:
{{"creates": [{{"text": "...", "source_fact_ids": ["id1"]}}],
  "updates": [{{"text": "FULL updated text replacing old observation", "observation_id": "obs_id", "source_fact_ids": ["id2"]}}],
  "deletes": [{{"observation_id": "obs_id_to_remove"}}]}}

Return {{"creates": [], "updates": [], "deletes": []}} if nothing durable.
MINIMIZE creates. MAXIMIZE updates."""


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
    req.add_header("Authorization", f"Bearer {LLM_API_KEY}")
    req.add_header("User-Agent", "rasputin-memory/1.0")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            wait = 10 * (attempt + 1) if e.code in (403, 429) else 2**attempt
            if attempt < 4:
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


def recall_for_fact(fact_text, obs_collection, limit=5):
    vec = get_embedding(fact_text[:2000], prefix="search_query: ")
    try:
        data = http_json(
            f"{QDRANT_URL}/collections/{obs_collection}/points/query",
            data={"query": vec, "limit": limit, "with_payload": True, "score_threshold": 0.3},
            method="POST",
            timeout=10,
        )
        return [
            {
                "id": str(p["id"]),
                "text": p.get("payload", {}).get("text", ""),
                "source_fact_ids": p.get("payload", {}).get("source_fact_ids", []),
                "score": p.get("score", 0),
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
    return point_id


def delete_observation(obs_id, obs_collection):
    try:
        http_json(
            f"{QDRANT_URL}/collections/{obs_collection}/points/delete",
            data={"points": [int(obs_id)]},
            method="POST",
            timeout=10,
        )
    except Exception:
        pass


def dedup_observations(obs_collection):
    all_obs = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        data = http_json(
            f"{QDRANT_URL}/collections/{obs_collection}/points/scroll", data=body, method="POST", timeout=30
        )
        points = data.get("result", {}).get("points", [])
        for p in points:
            if p.get("vector"):
                all_obs.append(
                    {
                        "id": p["id"],
                        "vector": p["vector"],
                        "text": p.get("payload", {}).get("text", ""),
                        "source_fact_ids": p.get("payload", {}).get("source_fact_ids", []),
                    }
                )
        next_offset = data.get("result", {}).get("next_page_offset")
        if not next_offset:
            break
        offset = next_offset

    if len(all_obs) < 2:
        return 0

    merged = 0
    to_delete = set()
    for i, obs_a in enumerate(all_obs):
        if obs_a["id"] in to_delete:
            continue
        for j in range(i + 1, len(all_obs)):
            obs_b = all_obs[j]
            if obs_b["id"] in to_delete:
                continue
            dot = sum(a * b for a, b in zip(obs_a["vector"], obs_b["vector"]))
            norm_a = math.sqrt(sum(x * x for x in obs_a["vector"]))
            norm_b = math.sqrt(sum(x * x for x in obs_b["vector"]))
            sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            if sim >= DEDUP_THRESHOLD:
                keep = obs_a if len(obs_a["text"]) >= len(obs_b["text"]) else obs_b
                remove = obs_b if keep is obs_a else obs_a
                to_delete.add(remove["id"])
                merged += 1

    for obs_id in to_delete:
        delete_observation(obs_id, obs_collection)

    return merged


def consolidate_collection(collection):
    obs_col = f"{collection}_obs"
    create_collection(obs_col)

    facts = scroll_facts(collection)
    if not facts:
        print(f"  {collection}: no facts found, skipping")
        return 0

    print(f"  {collection}: consolidating {len(facts)} facts (batch={BATCH_SIZE})...")
    total_created = 0
    total_updated = 0

    for i in range(0, len(facts), BATCH_SIZE):
        batch = facts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        facts_text = "\n".join(f"[{f['id']}] {f['text']}" for f in batch)

        seen_obs = {}
        for fact in batch:
            results = recall_for_fact(fact["text"], obs_col, limit=RECALL_PER_FACT)
            for obs in results:
                if obs["id"] not in seen_obs or obs["score"] > seen_obs[obs["id"]].get("score", 0):
                    seen_obs[obs["id"]] = obs

        existing = list(seen_obs.values())
        if existing:
            obs_lines = []
            for obs in existing:
                src = ", ".join(obs.get("source_fact_ids", [])[:5])
                obs_lines.append(f"[obs_id={obs['id']}] {obs['text']} (source facts: {src})")
            obs_text = "\n".join(obs_lines)
        else:
            obs_text = "(none yet)"

        prompt = CONSOLIDATION_PROMPT.format(facts_text=facts_text, observations_text=obs_text)

        try:
            content = llm_call(prompt)
            actions = parse_response(content)

            creates = 0
            for create in actions.get("creates", []):
                text = create.get("text", "")
                if text and len(text) >= 15:
                    store_observation(text, create.get("source_fact_ids", []), obs_col)
                    creates += 1
                    total_created += 1

            updates = 0
            for update in actions.get("updates", []):
                text = update.get("text", "")
                obs_id = update.get("observation_id", "")
                if text and len(text) >= 15:
                    if obs_id:
                        delete_observation(obs_id, obs_col)
                    store_observation(text, update.get("source_fact_ids", []), obs_col)
                    updates += 1
                    total_updated += 1

            for delete in actions.get("deletes", []):
                obs_id = delete.get("observation_id", "")
                if obs_id:
                    delete_observation(obs_id, obs_col)

            if batch_num % 5 == 0 or batch_num == 1:
                print(
                    f"    Batch {batch_num}: +{creates} created, ~{updates} updated, {len(existing)} existing recalled"
                )

        except Exception as e:
            print(f"    Batch {batch_num} failed: {e}")

        time.sleep(2)

    merged = dedup_observations(obs_col)

    try:
        data = http_json(f"{QDRANT_URL}/collections/{obs_col}", timeout=10)
        final_count = data.get("result", {}).get("points_count", 0)
    except Exception:
        final_count = "?"

    print(
        f"  {collection}: {total_created} created, {total_updated} updated, {merged} deduped → {final_count} final observations"
    )
    return total_created + total_updated


def main():
    with open(LOCOMO_FILE) as f:
        conversations = json.load(f)

    conv_indices = list(range(len(conversations)))
    if len(sys.argv) > 1 and "--conversations" in sys.argv:
        idx = sys.argv.index("--conversations")
        conv_indices = [int(x) for x in sys.argv[idx + 1].split(",")]

    total = 0
    for idx in conv_indices:
        conv = conversations[idx]
        conv_id = conv.get("sample_id", f"conv-{idx}")
        collection = f"locomo_lb_{conv_id.replace('-', '_')}"
        total += consolidate_collection(collection)

    print(f"\nTotal operations: {total}")


if __name__ == "__main__":
    main()
